#!/usr/bin/env python3
"""Extract reviewable domain-knowledge candidates from classified errors."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.io_utils import read_json, write_json
from utils.openai_compat import call_chat_completion, read_openai_key


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "knowledge"
OUTPUT_NAME = "domain_knowledge_for_review.json"
KEY_FILE = ROOT.parent / "openai_key.txt"

TEMPERATURE = 0.0
MAX_TOKENS = 500
TIMEOUT_SECONDS = 300.0
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 2.0
EVIDENCE_PER_BATCH = 6
MAX_KNOWLEDGE_PER_BATCH = 6
PLACEHOLDER = "{variables}"

PROMPT_ONLY_CATEGORIES = {
    "Exclusive Token",
    "Exclusive Phrase",
    "Event Predicate",
    "Username",
    "Equation",
    "Key-Value Pair",
    "Others",
}

CATEGORY_GUIDANCE = {
    "Multi Symbol Text": (
        "Find recurring paths, file names, package/class names, and structured "
        "identifiers. Prefer a regex when their complete separator structure is stable."
    ),
    "Alphanumeric Mixture": (
        "Distinguish numbers, hexadecimal values, UUIDs, versions, block IDs, and "
        "context-specific identifiers. Prefer a regex only when it is narrower than a "
        "generic number pattern; otherwise state the shared semantic context."
    ),
    "Exclusive Token": (
        "Identify recurring domain tokens with the same role. Abstract varying tokens, "
        "but preserve repeated fixed tokens that the ground-truth template keeps literal."
    ),
    "Key-Value Pair": (
        "Determine whether the variable is the value or the complete key-value field. "
        "Keep fixed keys and delimiters unless the ground truth abstracts them."
    ),
    "Equation": (
        "Describe the variable's role in an equality, comparison, ratio, or mapping. "
        "Do not treat an operand as an unrelated standalone number."
    ),
    "Composite Field": (
        "Find recurring values with units, wrappers, ranges, lists, or nested structure. "
        "Prefer a regex when it can cover the complete composite value safely."
    ),
    "Exclusive Phrase": (
        "Identify repeated fixed phrases or varying phrases with the same explicit role. "
        "Preserve the complete literal phrase when required by the ground truth."
    ),
    "Timestamp": (
        "Separate dates, clock times, durations, timezones, and full timestamps. Prefer "
        "a regex when the complete temporal format is stable."
    ),
    "Network Address": (
        "Keep IP address, hostname, port, host-port endpoint, URL, and URI rules "
        "separate. Prefer a regex that matches the complete address subtype."
    ),
    "Username": (
        "Extract a username rule only when user, login, account, authentication, or "
        "principal context supports that role."
    ),
    "Event Predicate": (
        "Find recurring actions, states, lifecycle events, results, or outcomes. Use "
        "natural-language knowledge because their meaning depends on context."
    ),
    "Others": (
        "Inspect placeholder, punctuation, wrapper, whitespace, or literal-format "
        "mismatches and extract only a reusable formatting or normalization rule."
    ),
}

NAMED_PLACEHOLDER_RE = re.compile(
    r"\{(?!variables\})[A-Za-z_][A-Za-z0-9_.-]*\}"
)
DOUBLE_PLACEHOLDER_RE = re.compile(r"\{\{variables\}\}")
FIELD_PREFIX_RE = re.compile(
    r"(?i)^(ACTION|RULE|PATTERN|EVIDENCE|EXPLANATION)\s*:"
)
BROAD_REGEXES = {
    ".*",
    ".+",
    r"\S+",
    r".*?",
    r".+?",
    r"\d+",
    r"\d*",
    r"\b\d+\b",
    r"[0-9]+",
    r"\b[0-9]+\b",
}


def iter_categories(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for coarse in data.get("hierarchical_categories", []):
        for intermediate in coarse.get("intermediate_grained_categories", []):
            for fine in intermediate.get("fine_grained_categories", []):
                yield {
                    "coarse": coarse.get("coarse_grained_level", ""),
                    "intermediate": intermediate.get(
                        "intermediate_grained_level", ""
                    ),
                    "fine": fine.get("fine_grained_level", ""),
                    "definition": fine.get("definition", ""),
                    "examples": fine.get("examples", []),
                }


def category_name(category: Dict[str, Any]) -> str:
    return " / ".join(
        str(category.get(level, ""))
        for level in ("coarse", "intermediate", "fine")
    )


def label_matches(label: Dict[str, Any], category: Dict[str, Any]) -> bool:
    return (
        label.get("coarse_grained_level") == category["coarse"]
        and label.get("intermediate_grained_level")
        == category["intermediate"]
        and label.get("fine_grained_level") == category["fine"]
    )


def classified_differences(
    example: Dict[str, Any], category: Dict[str, Any]
) -> List[Dict[str, Any]]:
    difference_ids = {
        str(difference_id)
        for label in example.get("labels", [])
        if isinstance(label, dict) and label_matches(label, category)
        for difference_id in label.get("difference_ids", [])
    }
    selected = []
    for index, difference in enumerate(example.get("knowledge_spans", [])):
        if not isinstance(difference, dict):
            continue
        difference_id = difference.get("difference_id", index)
        action = difference.get("action")
        if str(difference_id) not in difference_ids:
            continue
        if action not in {"abstract", "preserve"}:
            continue
        target = {
            "action": action,
            "log_span": str(difference.get("log_span") or ""),
            "ground_truth_span": str(
                difference.get("ground_truth_span") or ""
            ),
        }
        if action == "preserve":
            target["llm_span"] = str(difference.get("llm_span") or "")
        selected.append(
            {
                "evidence_id": f"{example.get('example_id')}@{difference_id}",
                "example_id": str(example.get("example_id") or ""),
                "action": action,
                "target": target,
                "log_message": str(example.get("log_message") or ""),
                "llm_template": str(
                    example.get("parsed_template") or ""
                ),
                "ground_truth_template": str(
                    example.get("ground_truth_template") or ""
                ),
            }
        )
    return selected


def collect_evidence(
    category: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    evidence = []
    skipped = []
    for example in category.get("examples", []):
        selected = classified_differences(example, category)
        if selected:
            evidence.extend(selected)
        else:
            skipped.append(
                {
                    "example_id": example.get("example_id"),
                    "reason": "missing classified knowledge span",
                }
            )
    return evidence, skipped


def span_shape(text: str) -> str:
    value = re.sub(r"[A-Za-z]+", "A", text)
    value = re.sub(r"\d+", "0", value)
    return value.casefold()


def sort_key(evidence: Dict[str, Any]) -> Tuple[str, str, str]:
    target = str(evidence["target"].get("log_span") or "")
    return evidence["action"], span_shape(target), target.casefold()


def balanced_batches(
    evidence: Sequence[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    if not evidence:
        return []
    batch_count = math.ceil(len(evidence) / EVIDENCE_PER_BATCH)
    base, extra = divmod(len(evidence), batch_count)
    sizes = [
        base + (1 if index < extra else 0)
        for index in range(batch_count)
    ]
    batches = []
    start = 0
    for size in sizes:
        batches.append(list(evidence[start : start + size]))
        start += size
    return batches


def build_batches(
    category: Dict[str, Any],
) -> Tuple[
    List[List[Dict[str, Any]]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Submit every classified difference exactly once."""
    evidence, skipped = collect_evidence(category)
    batches = []
    for action in ("abstract", "preserve"):
        action_evidence = sorted(
            (item for item in evidence if item["action"] == action),
            key=sort_key,
        )
        batches.extend(balanced_batches(action_evidence))
    return batches, skipped, evidence


def model_evidence(alias: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "evidence_id": alias,
        "target_difference": evidence["target"],
        "log_message": evidence["log_message"],
        "llm_template": evidence["llm_template"],
        "ground_truth_template": evidence["ground_truth_template"],
    }


def regex_allowed(category: Dict[str, Any], action: str) -> bool:
    return (
        action == "abstract"
        and category["fine"] not in PROMPT_ONLY_CATEGORIES
    )


def build_prompt(
    category: Dict[str, Any],
    batch: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    action = batch[0]["action"]
    evidence = [
        model_evidence(f"E{index}", item)
        for index, item in enumerate(batch, 1)
    ]
    regex_instruction = (
        "For each recurring group, prefer one raw Python regex when a stable surface "
        "form can safely full-match the complete targets. Use a natural-language rule "
        "when semantic context is required. "
        if regex_allowed(category, action)
        else "Return natural-language rules for this category and action. "
    )
    system = (
        "Extract reusable candidate domain knowledge from log-template errors. Compare "
        "each target difference with its complete log, LLM template, and ground-truth "
        "template. Group similar errors and produce a separate knowledge item for each "
        "shared pattern or semantic role. Do not rewrite every example as its own rule. "
        "A knowledge item may describe any similar subset of the input; it does not need "
        "to fit unrelated differences. State only one shared form or role per item. Do "
        "not enumerate observed values or add unsupported alternatives. "
        + regex_instruction
        + "A natural-language item must be one concise English imperative sentence "
        "beginning with 'When ...,' and contain either 'abstract ... as {variables}' "
        "or 'preserve ... as literal text', according to the required action. Use only "
        "{variables} as the placeholder."
    )
    user = (
        f"Category: {category['fine']}\n"
        f"Definition: {category['definition']}\n"
        f"Required action: {action}\n"
        f"Extraction guidance: {CATEGORY_GUIDANCE.get(category['fine'], category['fine'])}\n\n"
        f"Return zero to {MAX_KNOWLEDGE_PER_BATCH} knowledge items, one item per line. "
        "A line must be either one natural-language rule or one raw Python regex. For a "
        "regex, return only the pattern without quotes, delimiters, or code fences. Do "
        "not return labels, bullets, numbering, evidence IDs, JSON, Markdown, examples, "
        "or explanations. Do not output both a regex and a natural-language restatement "
        "for the same knowledge. Return NONE if no reusable knowledge can be extracted.\n\n"
        + json.dumps(evidence, ensure_ascii=False, indent=2)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_lines(text: str) -> List[str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    if text.upper() == "NONE":
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty model response")
    if len(lines) > MAX_KNOWLEDGE_PER_BATCH:
        raise ValueError("too many knowledge items")
    return lines


def validate_language(rule: str, action: str) -> None:
    lower = rule.casefold()
    if not rule.startswith("When "):
        raise ValueError("natural-language knowledge must begin with 'When '")
    if NAMED_PLACEHOLDER_RE.search(rule) or DOUBLE_PLACEHOLDER_RE.search(rule):
        raise ValueError("invalid placeholder")
    if action == "abstract":
        if "abstract " not in lower or f"as {PLACEHOLDER}" not in lower:
            raise ValueError(
                "abstract knowledge must say 'abstract ... as {variables}'"
            )
        if "preserve " in lower:
            raise ValueError("abstract knowledge must not preserve text")
    else:
        if "preserve " not in lower or "as literal text" not in lower:
            raise ValueError(
                "preserve knowledge must say 'preserve ... as literal text'"
            )
        if "abstract " in lower or PLACEHOLDER in rule:
            raise ValueError("preserve knowledge must not abstract text")


def build_candidate(
    content: str,
    category: Dict[str, Any],
    batch: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    action = batch[0]["action"]
    if FIELD_PREFIX_RE.match(content):
        raise ValueError("return knowledge content without a field prefix")
    if content.startswith(("- ", "* ")) or re.match(r"^\d+[.)]\s", content):
        raise ValueError("return knowledge without bullets or numbering")

    if content.startswith("When "):
        validate_language(content, action)
        return {
            "knowledge_type": "llm_prompt_rule",
            "knowledge": content,
            "direct_artifact": None,
            "action": action,
        }

    if not regex_allowed(category, action):
        raise ValueError("this batch requires natural-language knowledge")
    if content in BROAD_REGEXES:
        raise ValueError("regex is too broad")
    try:
        compiled = re.compile(content)
    except re.error as exc:
        raise ValueError(f"invalid Python regex: {exc}") from exc

    target_values = {
        str(evidence["target"].get("log_span") or "")
        for evidence in batch
    }
    matching_values = {
        value for value in target_values
        if value and compiled.fullmatch(value) is not None
    }
    if len(matching_values) < 2:
        raise ValueError("regex must full-match at least two different targets")
    return {
        "knowledge_type": "preprocessing_regex",
        "knowledge": (
            f"Use preprocessing regex `{content}` and replace each complete "
            f"match with {PLACEHOLDER}."
        ),
        "direct_artifact": {
            "pattern": content,
            "replacement": PLACEHOLDER,
        },
        "action": action,
    }


def normalize_response(
    response: str,
    category: Dict[str, Any],
    batch: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, List[str]]:
    candidates = []
    errors = []
    for index, line in enumerate(parse_lines(response)):
        try:
            candidates.append(build_candidate(line, category, batch))
        except ValueError as exc:
            errors.append(f"line {index + 1}: {exc}")
    if errors and not candidates:
        raise ValueError("; ".join(errors))
    return candidates, len(errors), errors


def request_batch(
    model: str,
    category: Dict[str, Any],
    batch: Sequence[Dict[str, Any]],
    api_base: Optional[str],
    api_key: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    base_messages = build_prompt(category, batch)
    messages = list(base_messages)
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        response = ""
        try:
            response = call_chat_completion(
                messages,
                model=model,
                api_base=api_base,
                api_key=api_key,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=TIMEOUT_SECONDS,
            )
            candidates, invalid_count, errors = normalize_response(
                response, category, batch
            )
            return {
                "knowledge_items": candidates,
                "invalid_candidate_lines": invalid_count,
                "invalid_candidate_errors": errors,
            }, None
        except Exception as exc:
            last_error = str(exc)
            print(
                f"  attempt {attempt}/{MAX_RETRIES} failed: {last_error}"
            )
            if attempt < MAX_RETRIES:
                messages = [
                    *base_messages,
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": (
                            f"Correct this error: {last_error}. Return only "
                            "knowledge items, one per line, or NONE."
                        ),
                    },
                ]
                time.sleep(RETRY_SLEEP_SECONDS)
    return None, last_error


def item_key(item: Dict[str, Any], category: str) -> str:
    payload = {
        "category": category,
        "action": item["action"],
        "knowledge_type": item["knowledge_type"],
    }
    if item["knowledge_type"] == "preprocessing_regex":
        payload["artifact"] = item["direct_artifact"]
    else:
        payload["knowledge"] = " ".join(
            item["knowledge"].split()
        ).rstrip(".").casefold()
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def consolidate(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove exact duplicate candidates without claiming evidence mappings."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for result in results:
        for candidate in result.get("knowledge_items", []):
            key = item_key(candidate, str(result.get("category") or ""))
            source = {
                "category": result.get("category"),
                "batch_index": result.get("batch_index"),
            }
            if key not in grouped:
                grouped[key] = {**candidate, "sources": [source]}
            elif source not in grouped[key]["sources"]:
                grouped[key]["sources"].append(source)

    output = []
    for key, item in grouped.items():
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        item["knowledge_id"] = f"knowledge_{digest}"
        item["review_status"] = "pending"
        output.append(item)
    return sorted(output, key=lambda item: item["knowledge_id"])


def build_output(
    args: argparse.Namespace,
    source: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    knowledge_items = consolidate(results)
    batches = [
        result for result in results if "batch_index" in result
    ]
    rejected = [
        result for result in batches
        if result.get("status") == "rejected"
    ]
    skipped = [
        result for result in batches
        if result.get("status") == "skipped"
    ]
    skipped_examples = [
        {"category": result.get("category"), **example}
        for result in results
        for example in result.get("skipped_examples", [])
    ]
    eligible = {
        evidence_id
        for result in results
        for evidence_id in result.get("eligible_evidence_ids", [])
    }
    submitted = {
        evidence_id
        for result in batches
        for evidence_id in result.get("evidence_ids", [])
    }
    source_metadata = source.get("metadata", {})
    complete = (
        source_metadata.get("complete") is not False
        and not rejected
        and not skipped_examples
        and eligible == submitted
    )
    return {
        "metadata": {
            "source_file": args.input,
            "source_taxonomy_name": source_metadata.get("taxonomy_name"),
            "source_classification_complete": (
                source_metadata.get("complete") is not False
            ),
            "source_total_errors": source_metadata.get("total_errors"),
            "source_classified_errors": source_metadata.get(
                "classified_errors"
            ),
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_dir": args.run_dir,
            "model": args.model,
            "temperature": TEMPERATURE,
            "evidence_per_batch": EVIDENCE_PER_BATCH,
            "extraction_batches": len(batches),
            "accepted_batches": sum(
                result.get("status") == "accepted"
                for result in batches
            ),
            "rejected_batches": len(rejected),
            "skipped_batches": len(skipped),
            "skipped_examples": len(skipped_examples),
            "eligible_evidence_units": len(eligible),
            "submitted_evidence_units": len(submitted),
            "llm_input_coverage": (
                len(eligible & submitted) / len(eligible)
                if eligible else 1.0
            ),
            "raw_knowledge_items": sum(
                len(result.get("knowledge_items", []))
                for result in batches
            ),
            "total_knowledge_items": len(knowledge_items),
            "exact_duplicates_removed": (
                sum(
                    len(result.get("knowledge_items", []))
                    for result in batches
                )
                - len(knowledge_items)
            ),
            "knowledge_items_by_type": dict(
                collections.Counter(
                    item["knowledge_type"] for item in knowledge_items
                )
            ),
            "knowledge_items_by_action": dict(
                collections.Counter(
                    item["action"] for item in knowledge_items
                )
            ),
            "invalid_candidate_lines": sum(
                result.get("invalid_candidate_lines", 0)
                for result in batches
            ),
            "complete": complete,
        },
        "review_instructions": {
            "status": "All items are candidates requiring human review.",
            "review_status": "pending | accepted | revised | rejected",
        },
        "knowledge_items": knowledge_items,
        "rejected_batches": rejected,
        "skipped_batches": skipped,
        "skipped_examples": skipped_examples,
    }


def prepare_output_path(args: argparse.Namespace) -> None:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_timestamp = timestamp
    run_dir = RESULTS_DIR / f"{timestamp}_pending"
    args.run_dir = str(run_dir)
    args.output = str(run_dir / OUTPUT_NAME)


def finalize_output_path(
    args: argparse.Namespace, knowledge_count: int
) -> None:
    current_dir = Path(args.run_dir)
    final_dir = RESULTS_DIR / (
        f"{args.run_timestamp}_{knowledge_count}"
    )
    if final_dir.exists() and final_dir != current_dir:
        raise FileExistsError(
            f"final output directory already exists: {final_dir}"
        )
    if current_dir.exists() and current_dir != final_dir:
        current_dir.rename(final_dir)
    args.run_dir = str(final_dir)
    args.output = str(final_dir / OUTPUT_NAME)


def write_output(
    args: argparse.Namespace,
    source: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    output = build_output(args, source, results)
    write_json(Path(args.output), output)
    return output


def run(args: argparse.Namespace) -> Dict[str, Any]:
    prepare_output_path(args)
    source = read_json(Path(args.input))
    if not isinstance(source, dict) or not isinstance(
        source.get("hierarchical_categories"), list
    ):
        raise ValueError(
            "input must be a hierarchical classification JSON object"
        )

    api_base: Optional[str] = None
    api_key = ""
    if not args.dry_run:
        api_base, api_key = read_openai_key(KEY_FILE, args.model)

    categories = list(iter_categories(source))
    results: List[Dict[str, Any]] = []
    for category_index, category in enumerate(categories, 1):
        if not category["examples"]:
            continue
        name = category_name(category)
        batches, skipped_examples, evidence = build_batches(category)
        results.append(
            {
                "category": name,
                "eligible_evidence_ids": [
                    item["evidence_id"] for item in evidence
                ],
            }
        )
        if skipped_examples:
            results.append(
                {
                    "category": name,
                    "skipped_examples": skipped_examples,
                }
            )

        for batch_index, batch in enumerate(batches, 1):
            print(
                f"[{category_index}/{len(categories)} "
                f"batch {batch_index}/{len(batches)}] extracting: "
                f"{name} ({len(batch)} evidence units)"
            )
            result: Dict[str, Any] = {
                "category": name,
                "batch_index": batch_index,
                "evidence_ids": [
                    item["evidence_id"] for item in batch
                ],
            }
            if args.dry_run:
                result.update(
                    {
                        "status": "skipped",
                        "reason": "dry_run",
                        "prompt_messages": build_prompt(category, batch),
                    }
                )
            else:
                response, error = request_batch(
                    args.model,
                    category,
                    batch,
                    api_base,
                    api_key,
                )
                if response is None:
                    result.update(
                        {"status": "rejected", "error": error}
                    )
                else:
                    items = response["knowledge_items"]
                    result.update(
                        {
                            "status": (
                                "accepted" if items else "skipped"
                            ),
                            "reason": None if items else "no_candidate",
                            "knowledge_items": items,
                            "invalid_candidate_lines": response[
                                "invalid_candidate_lines"
                            ],
                            "invalid_candidate_errors": response[
                                "invalid_candidate_errors"
                            ],
                        }
                    )
            results.append(result)

        write_output(args, source, results)

    output = write_output(args, source, results)
    finalize_output_path(
        args, output["metadata"]["total_knowledge_items"]
    )
    return write_output(args, source, results)


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract candidate domain knowledge from classified parsing errors."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Hierarchical classification JSON",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Chat model",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        output = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    metadata = output["metadata"]
    print(
        "Done. "
        f"batches={metadata['extraction_batches']}, "
        f"accepted={metadata['accepted_batches']}, "
        f"rejected={metadata['rejected_batches']}, "
        f"knowledge_items={metadata['total_knowledge_items']}"
    )
    print(f"Saved to: {args.output}")
    print(f"Run directory: {args.run_dir}")
    return 0 if metadata["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
