#!/usr/bin/env python3
"""Classify IR parsing errors using the current fixed taxonomy."""

from __future__ import annotations

import argparse
import collections
import copy
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.io_utils import read_json, write_json
from utils.openai_compat import call_chat_completion, read_openai_key
from utils.template_alignment import extract_knowledge_spans


PROJECT_ROOT = Path(__file__).resolve().parent
LILAC_COPY_ROOT = PROJECT_ROOT.parent
DEFAULT_INPUT = PROJECT_ROOT / "IR_results.json"
DEFAULT_TAXONOMY_FILE = PROJECT_ROOT / "IR_error_taxonomy.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "classification"
DEFAULT_OUTPUT_NAME = "classification_results.json"
DEFAULT_KEY_FILE = LILAC_COPY_ROOT / "openai_key.txt"
DIFFERENCE_SOURCE = "deterministic_log_template_alignment"
FIRST_PASS_TEMPERATURE = 0.0
SECOND_PASS_TEMPERATURE = 0.1
CLASSIFICATION_MAX_TOKENS = 400
REQUEST_TIMEOUT_SECONDS = 300.0
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 2.0
MAX_LABELS_PER_EXAMPLE = 4
CLASSIFICATION_BLOCK_RE = re.compile(
    r"\[CLASSIFICATION\](.*?)\[/CLASSIFICATION\]",
    re.DOTALL,
)
SYMBOL_RICH_RE = re.compile(r"[/\\._@-]|::")
MEANINGFUL_COLON_RE = re.compile(r"[A-Za-z0-9_]:[A-Za-z0-9_]")
LEXICAL_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*")
QUALIFIED_IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Za-z_$][A-Za-z0-9_$]*\.){2,}[A-Z_$][A-Za-z0-9_$]*\b"
)
UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:bytes?|kib|mib|gib|tib|kb|mb|gb|tb|ms|msec|"
    r"milliseconds?|sec|seconds?|minutes?|hours?|hz|khz|mhz|ghz)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
TIMEZONE_RE = re.compile(
    r"\b(?:UTC|GMT|PST|PDT|MST|MDT|CST|CDT|EST|EDT)\b",
    re.IGNORECASE,
)
URL_SCHEME_RE = re.compile(r"\b(?:https?|hdfs|ftp|ssh|tcp|udp|akka\.tcp)://", re.IGNORECASE)
SLASH_IP_RE = re.compile(r"/(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b")
PATH_RE = re.compile(r"(?:^|[\s:(])(?:[A-Za-z]:)?[/\\](?:[^\s/\\]+[/\\])+[^\s]*")
KEY_VALUE_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_.-]*\s*=")
EVENT_RE = re.compile(
    r"\b(?:start(?:ed|ing)?|stop(?:ped|ping)?|failed|succeed(?:ed)?|completed?|"
    r"cancel(?:led|ed)|exceeded|shut(?:ting)?\s+down|shutdown|created?|deleted?|"
    r"killed|connect(?:ed|ing)?|disconnect(?:ed|ing)?|plugged|unplugged)\b",
    re.IGNORECASE,
)
STATE_TOKEN_RE = re.compile(
    r"^(?:true|false|yes|no|on|off|enabled|disabled|active|inactive|unknown)$",
    re.IGNORECASE,
)


def prepare_output_path(args: argparse.Namespace, error_count: int) -> None:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = DEFAULT_RESULTS_DIR / f"{timestamp}_{error_count}"
    args.run_dir = str(run_dir)
    args.output = str(run_dir / DEFAULT_OUTPUT_NAME)
    args.input_error_count = error_count


def count_ir_errors(data: Sequence[Dict[str, Any]]) -> int:
    return sum(len(dataset_block.get("errors", [])) for dataset_block in data)


def iter_taxonomy_fine_labels(taxonomy: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for coarse in taxonomy:
        for intermediate in coarse.get("intermediate_grained_categories", []):
            for fine in intermediate.get("fine_grained_categories", []):
                yield {
                    "coarse_grained_level": coarse.get("coarse_grained_level", ""),
                    "coarse_definition": coarse.get("definition", ""),
                    "intermediate_grained_level": intermediate.get("intermediate_grained_level", ""),
                    "intermediate_definition": intermediate.get("definition", ""),
                    "fine_grained_level": fine.get("fine_grained_level", ""),
                    "fine_definition": fine.get("definition", ""),
                    "knowledge_extraction_focus": fine.get("knowledge_extraction_focus", []),
                }


def _render_list(title: str, values: Any, indent: str) -> List[str]:
    if not values:
        return []
    if not isinstance(values, list):
        values = [values]
    lines = [f"{indent}{title}:"]
    for value in values:
        lines.append(f"{indent}  - {value}")
    return lines


def taxonomy_prompt(taxonomy: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for coarse in taxonomy:
        for intermediate in coarse.get("intermediate_grained_categories", []):
            for fine in intermediate.get("fine_grained_categories", []):
                lines.append(f"- Fine label: {fine.get('fine_grained_level')}")
                lines.append(f"  Definition: {fine.get('definition', '')}")
                lines.extend(_render_list("Exclude", fine.get("exclude_conditions"), "  "))
                lines.extend(_render_list("Priority", fine.get("priority_notes"), "  "))
    return "\n".join(lines)


def flatten_ir_errors(data: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    parsing_errors: List[Dict[str, Any]] = []
    non_parsing_failures: List[Dict[str, Any]] = []
    for dataset_block in data:
        dataset = dataset_block.get("dataset", "unknown")
        for error_index, error in enumerate(dataset_block.get("errors", []), 1):
            item = {
                "example_id": f"{dataset}#{error_index:04d}",
                "dataset": dataset,
                "error_index": error_index,
                "log_message": error.get("log_message"),
                "parsed_template": error.get("parsed_template"),
                "ground_truth_template": error.get("ground_truth_template"),
                "llm_call_success": error.get("llm_call_success"),
            }
            if error.get("llm_call_success") is not True:
                failure = copy.deepcopy(item)
                if "llm_call_success" not in error:
                    failure["reason"] = (
                        "Missing llm_call_success field; only explicitly successful LLM calls are classified."
                    )
                else:
                    failure["reason"] = (
                        "llm_call_success is false; this record is excluded from parsing-error classification."
                    )
                non_parsing_failures.append(failure)
            else:
                parsing_errors.append(item)
    return parsing_errors, non_parsing_failures


def example_knowledge_spans(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    return copy.deepcopy(example.get("knowledge_spans", []))


def compact_example(example: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "log_message": str(example.get("log_message") or ""),
        "ground_truth_template": str(example.get("ground_truth_template") or ""),
        "knowledge_spans": example_knowledge_spans(example),
    }


def build_prompt(
    taxonomy: Sequence[Dict[str, Any]],
    example: Dict[str, Any],
    *,
    validation_errors: Optional[Sequence[str]] = None,
) -> Sequence[Dict[str, str]]:
    example_payload = compact_example(example)
    system_prompt = "Classify extracted log spans with the fixed taxonomy. Return only [CLASSIFICATION] blocks."
    user_prompt = f"""Assign every difference_id one semantic fine label.

Taxonomy:
{taxonomy_prompt(taxonomy)}

Input:
{json.dumps(example_payload, ensure_ascii=False, indent=2)}

Output format:
[CLASSIFICATION]
difference_ids=0,1
fine_label=<exact fine label>
[/CLASSIFICATION]

Rules:
1. Cover each difference_id exactly once; group IDs only when they share one label.
2. action and log_span are code-derived and authoritative. Classify what log_span represents; do not change its action.
3. Use the complete log and ground-truth template only to resolve the span's meaning.
4. Choose the most specific taxonomy label and use at most {MAX_LABELS_PER_EXAMPLE} different labels.
5. Apply these boundaries:
   - Network Address: IPv4, IPv6, domain, hostname, port, host:port, URL, URI, socket address, or network endpoint.
   - Timestamp: date, time, datetime, epoch timestamp, timezone name, or timezone offset.
   - Username: username, account name, login name, user ID, or principal when context indicates user identity.
   - Multi Symbol Text: file path, directory path, file name, package/class name, source location, or one symbol-rich identifier.
   - Alphanumeric Mixture: hexadecimal value, version ID, block ID, device ID, or generated letter-digit identifier.
   - Exclusive Token: one fixed protocol, method, service, component, module, command, flag, filesystem type, or system marker.
   - Composite Field: multiple connected parts represented by one ground-truth variable, including value-plus-unit, list, tuple, bracketed value, or nested field.
   - Key-Value Pair: named field, attribute, or option written as key=value or key:value.
   - Equation: equality, inequality, comparison, ratio, mapping, or relation between peer operands.
   - Exclusive Phrase: fixed multi-token domain phrase, diagnostic reason, error message, or message fragment.
   - Event Predicate: action, state, transition, lifecycle stage, outcome, success/failure, start/stop, completion, or cancellation.
   - Others: placeholder, whitespace, punctuation, quote, bracket, wrapper, spelling, or format-only error with no applicable specific label.
6. Hard constraints:
   - A plain single technical/domain word without internal symbols must not be Multi Symbol Text; use Exclusive Token when applicable.
   - Exclusive Token contains one lexical token; Exclusive Phrase contains at least two lexical tokens.
   - A dotted package/class identifier is Multi Symbol Text unless it is a network address.
   - A standalone action, outcome, or state is Event Predicate unless it is used as a method name.
   - A measurement or time unit absorbed into the ground-truth variable belongs to Composite Field, not Others.
   - A date, time, or timezone marker belongs to Timestamp, not Others.
7. Output no explanation or extra text."""
    if validation_errors:
        user_prompt += "\n\nCorrection requirements:\n" + "\n".join(validation_errors[:5])
        if any("[CORRECTION]" in error for error in validation_errors):
            user_prompt += "\nDo not return a forbidden label; use only the listed allowed labels."
        else:
            user_prompt += "\nReturn corrected classification blocks."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def canonical_labels_by_fine(taxonomy: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_fine: Dict[str, Dict[str, Any]] = {}
    for label in iter_taxonomy_fine_labels(taxonomy):
        fine = label["fine_grained_level"]
        if fine in by_fine:
            raise ValueError(f"Fine-grained taxonomy label must be unique: {fine!r}")
        by_fine[fine] = label
    return by_fine


def canonical_fine_name(
    value: str,
    labels_by_fine: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    if value in labels_by_fine:
        return value
    compact = re.sub(r"[\s_-]+", "", value).casefold()
    matches = [
        name
        for name in labels_by_fine
        if re.sub(r"[\s_-]+", "", name).casefold() == compact
    ]
    return matches[0] if len(matches) == 1 else None


def normalize_classifications(
    raw_output: str,
    example: Dict[str, Any],
    taxonomy: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    by_fine = canonical_labels_by_fine(taxonomy)
    example_id = example["example_id"]
    expected_difference_ids = {
        int(difference["difference_id"])
        for difference in example_knowledge_spans(example)
    }
    assigned_difference_ids = set()
    label_to_difference_ids: Dict[str, List[int]] = {}
    issues: List[str] = []
    blocks = CLASSIFICATION_BLOCK_RE.findall(raw_output)
    if not blocks:
        return {}, ["no [CLASSIFICATION] block found"]
    if CLASSIFICATION_BLOCK_RE.sub("", raw_output).strip():
        issues.append("output contains text outside [CLASSIFICATION] blocks")

    for block_index, block in enumerate(blocks):
        fields: Dict[str, str] = {}
        for line in block.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in {"difference_ids", "fine_label"}:
                fields[key] = value.strip()
        missing = [field for field in ("difference_ids", "fine_label") if not fields.get(field)]
        if missing:
            issues.append(f"classification block {block_index} is missing {missing}")
            continue
        try:
            difference_ids = [int(value.strip()) for value in fields["difference_ids"].split(",")]
        except ValueError:
            issues.append(f"classification block {block_index} has invalid difference_ids")
            continue
        if len(difference_ids) != len(set(difference_ids)):
            issues.append(f"classification block {block_index} repeats a difference_id")
            continue
        unknown_ids = sorted(set(difference_ids) - expected_difference_ids)
        if unknown_ids:
            issues.append(f"classification block {block_index} has unknown difference_ids {unknown_ids}")
            continue
        repeated_ids = sorted(set(difference_ids) & assigned_difference_ids)
        if repeated_ids:
            issues.append(f"difference_ids {repeated_ids} are classified more than once")
            continue
        raw_fine = fields["fine_label"]
        fine = canonical_fine_name(raw_fine, by_fine)
        if fine is None:
            issues.append(f"unknown fine label {raw_fine!r}")
            continue
        assigned_difference_ids.update(difference_ids)
        label_to_difference_ids.setdefault(fine, []).extend(difference_ids)

    missing_ids = sorted(expected_difference_ids - assigned_difference_ids)
    if missing_ids:
        issues.append(f"missing classifications for difference_ids {missing_ids}")
    if len(label_to_difference_ids) > MAX_LABELS_PER_EXAMPLE:
        issues.append(
            f"{len(label_to_difference_ids)} labels returned; at most {MAX_LABELS_PER_EXAMPLE} are allowed"
        )
    if issues:
        return {}, issues

    ordered_labels = sorted(
        label_to_difference_ids.items(),
        key=lambda item: min(item[1]),
    )
    normalized_labels: List[Dict[str, Any]] = []
    for label_index, (fine, difference_ids) in enumerate(ordered_labels):
        canonical = by_fine[fine]
        normalized_labels.append(
            {
                "role": "primary" if label_index == 0 else "secondary",
                "coarse_grained_level": canonical["coarse_grained_level"],
                "intermediate_grained_level": canonical["intermediate_grained_level"],
                "fine_grained_level": fine,
                "difference_ids": sorted(difference_ids),
            }
        )
    return {example_id: {"labels": normalized_labels}}, []


def _difference_literal_text(difference: Dict[str, Any]) -> str:
    return str(difference.get("log_span", "")).strip()


def _minimal_literal_text(difference: Dict[str, Any]) -> str:
    return _difference_literal_text(difference)


def _is_method_name(literal: str, log_message: str) -> bool:
    words = LEXICAL_TOKEN_RE.findall(literal)
    if len(words) != 1:
        return False
    token = re.escape(words[0])
    return bool(re.search(rf"(?:::|\.){token}\b|\b{token}\s*\(", log_message, re.IGNORECASE))


def _is_path_continuation(literal: str, log_message: str) -> bool:
    value = literal.strip()
    if not value:
        return False
    return bool(
        re.search(
            rf"(?:^|\s)(?:/[^\s]+)+\s+{re.escape(value)}(?=\s|$)",
            log_message,
        )
    )


def _supports_multi_symbol(literal: str, log_message: str) -> bool:
    return bool(
        SYMBOL_RICH_RE.search(literal)
        or MEANINGFUL_COLON_RE.search(literal)
        or _is_path_continuation(literal, log_message)
    )


def correction_requirement(
    difference_ids: Sequence[int],
    forbidden_label: str,
    allowed_labels: Sequence[str],
    reason: str,
) -> str:
    return (
        "[CORRECTION]\n"
        f"difference_ids={','.join(str(value) for value in difference_ids)}\n"
        f"forbidden_label={forbidden_label}\n"
        f"allowed_labels={','.join(allowed_labels)}\n"
        f"reason={reason}\n"
        "[/CORRECTION]"
    )


def validate_semantic_classifications(
    normalized: Dict[str, Dict[str, Any]],
    example: Dict[str, Any],
) -> List[str]:
    """Reject only high-confidence taxonomy conflicts so the LLM can retry."""
    example_id = example["example_id"]
    classification = normalized.get(example_id, {})
    differences = {
        int(difference["difference_id"]): difference
        for difference in example_knowledge_spans(example)
    }
    issues: List[str] = []

    for label in classification.get("labels", []):
        fine = label["fine_grained_level"]
        label_difference_ids = [int(value) for value in label.get("difference_ids", [])]
        group_literal = " ".join(
            _difference_literal_text(differences[difference_id])
            for difference_id in label_difference_ids
            if difference_id in differences
        )
        group_words: List[str] = []
        for difference_id in label_difference_ids:
            if difference_id in differences:
                group_words.extend(
                    LEXICAL_TOKEN_RE.findall(_minimal_literal_text(differences[difference_id]))
                )
        issue_count_before_label = len(issues)

        def reject(
            difference_ids: Sequence[int],
            allowed_labels: Sequence[str],
            reason: str,
        ) -> None:
            issues.append(
                correction_requirement(
                    difference_ids,
                    fine,
                    allowed_labels,
                    reason,
                )
            )

        invalid_multi_symbol_ids = []
        if fine == "Multi Symbol Text":
            for difference_id in label_difference_ids:
                difference = differences.get(difference_id)
                if difference is None:
                    continue
                literal = _difference_literal_text(difference)
                log_message = str(example.get("log_message", ""))
                if not literal or _supports_multi_symbol(literal, log_message):
                    continue
                invalid_multi_symbol_ids.append(difference_id)
                words = LEXICAL_TOKEN_RE.findall(literal)
                if len(words) >= 2:
                    allowed = ["Exclusive Phrase", "Composite Field", "Event Predicate"]
                elif re.search(r"[A-Za-z]", literal) and re.search(r"\d", literal):
                    allowed = ["Alphanumeric Mixture", "Exclusive Token"]
                else:
                    allowed = [
                        "Exclusive Token",
                        "Alphanumeric Mixture",
                        "Event Predicate",
                    ]
                reject(
                    [difference_id],
                    allowed,
                    "This span is not a path or an identifier with an internal separator.",
                )
        for difference_id in label_difference_ids:
            difference = differences.get(difference_id)
            if difference is None:
                continue
            literal = _difference_literal_text(difference)
            minimal_literal = _minimal_literal_text(difference)
            log_message = str(example.get("log_message", ""))

            if difference_id in invalid_multi_symbol_ids:
                continue

            if UNIT_RE.search(literal) and fine != "Composite Field":
                reject([difference_id], ["Composite Field"], "A value or time unit belongs to one composite field.")
                continue

            if TIMEZONE_RE.search(literal) and fine != "Timestamp":
                reject([difference_id], ["Timestamp"], "The changed text is a timezone marker.")
                continue

            is_network_boundary = bool(URL_SCHEME_RE.search(literal)) or (
                literal.strip() == "/" and bool(SLASH_IP_RE.search(log_message))
            )
            if is_network_boundary and fine != "Network Address":
                reject([difference_id], ["Network Address"], "The changed text is part of a network address.")
                continue

            path_belongs_to_grouped_url = (
                fine == "Network Address" and bool(URL_SCHEME_RE.search(group_literal))
            )
            if (
                PATH_RE.search(literal)
                and not URL_SCHEME_RE.search(literal)
                and not path_belongs_to_grouped_url
                and fine != "Multi Symbol Text"
            ):
                reject([difference_id], ["Multi Symbol Text"], "The changed text is a file or directory path.")
                continue

            if KEY_VALUE_RE.search(literal) and fine != "Key-Value Pair":
                reject([difference_id], ["Key-Value Pair"], "The changed text is a named key=value boundary.")
                continue

            if (
                QUALIFIED_IDENTIFIER_RE.search(minimal_literal)
                and fine != "Multi Symbol Text"
            ):
                reject(
                    [difference_id],
                    ["Multi Symbol Text"],
                    "The changed text is a dotted package or class identifier.",
                )
                continue

            literal_words = LEXICAL_TOKEN_RE.findall(minimal_literal)
            is_state = (
                len(literal_words) == 1
                and STATE_TOKEN_RE.fullmatch(literal_words[0]) is not None
            )
            if (
                (EVENT_RE.search(minimal_literal) or is_state)
                and not _is_method_name(minimal_literal, log_message)
                and fine != "Event Predicate"
            ):
                reject([difference_id], ["Event Predicate"], "The changed text expresses an event or outcome.")
                continue

            if (
                fine == "Exclusive Token"
                and len(literal_words) != 1
            ):
                reject(
                    [difference_id],
                    ["Exclusive Phrase", "Multi Symbol Text", "Event Predicate", "Composite Field", "Others"],
                    "Exclusive Token must contain exactly one lexical token.",
                )
                continue

            if (
                fine == "Composite Field"
                and re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", minimal_literal)
            ):
                reject(
                    [difference_id],
                    ["Exclusive Token", "Event Predicate", "Alphanumeric Mixture"],
                    "One plain literal token is not a composite field.",
                )

        if (
            len(issues) == issue_count_before_label
            and fine == "Exclusive Phrase"
            and len(group_words) < 2
        ):
            reject(
                label_difference_ids,
                [
                    "Exclusive Token",
                    "Multi Symbol Text",
                    "Alphanumeric Mixture",
                    "Composite Field",
                    "Event Predicate",
                    "Others",
                ],
                "Exclusive Phrase must contain at least two lexical tokens.",
            )

    return issues


def deterministic_classification(
    example: Dict[str, Any],
    taxonomy: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Classify only when every difference has one high-confidence rule label."""
    grouped: Dict[str, List[int]] = collections.defaultdict(list)
    log_message = str(example.get("log_message", ""))
    for difference in example_knowledge_spans(example):
        difference_id = int(difference["difference_id"])
        literal = _difference_literal_text(difference)
        minimal_literal = _minimal_literal_text(difference)
        words = LEXICAL_TOKEN_RE.findall(minimal_literal)

        if UNIT_RE.search(literal):
            fine = "Composite Field"
        elif TIMEZONE_RE.search(literal):
            fine = "Timestamp"
        elif URL_SCHEME_RE.search(literal) or (
            minimal_literal == "/" and SLASH_IP_RE.search(log_message)
        ):
            fine = "Network Address"
        elif PATH_RE.search(literal) or QUALIFIED_IDENTIFIER_RE.search(minimal_literal):
            fine = "Multi Symbol Text"
        elif (
            EVENT_RE.search(minimal_literal)
            or (len(words) == 1 and STATE_TOKEN_RE.fullmatch(words[0]))
        ) and not _is_method_name(minimal_literal, log_message):
            fine = "Event Predicate"
        else:
            return None
        grouped[fine].append(difference_id)

    if not grouped or len(grouped) > MAX_LABELS_PER_EXAMPLE:
        return None

    by_fine = canonical_labels_by_fine(taxonomy)
    labels = []
    for label_index, (fine, difference_ids) in enumerate(
        sorted(grouped.items(), key=lambda item: min(item[1]))
    ):
        canonical = by_fine[fine]
        labels.append(
            {
                "role": "primary" if label_index == 0 else "secondary",
                "coarse_grained_level": canonical["coarse_grained_level"],
                "intermediate_grained_level": canonical["intermediate_grained_level"],
                "fine_grained_level": fine,
                "difference_ids": sorted(difference_ids),
            }
        )
    return {"labels": labels}


def classify_example(
    model: str,
    taxonomy: Sequence[Dict[str, Any]],
    example: Dict[str, Any],
    api_base: Optional[str],
    api_key: str,
    *,
    temperature: float,
) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    last_errors: List[str] = []
    invalid_responses = set()
    for attempt in range(1, MAX_RETRIES + 1):
        messages = build_prompt(
            taxonomy,
            example,
            validation_errors=last_errors,
        )
        try:
            raw_response = call_chat_completion(
                messages,
                model=model,
                api_base=api_base,
                api_key=api_key,
                temperature=temperature,
                max_tokens=CLASSIFICATION_MAX_TOKENS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            normalized, issues = normalize_classifications(
                raw_response,
                example,
                taxonomy,
            )
            if not issues:
                issues = validate_semantic_classifications(normalized, example)
            if not issues:
                return normalized, None
            response_key = raw_response.strip()
            if response_key in invalid_responses:
                return {}, "model repeated the same invalid classification: " + "; ".join(issues)
            invalid_responses.add(response_key)
            last_errors = issues
            print(f"  classification validation failed: {'; '.join(issues[:5])}")
        except Exception as exc:
            last_errors = [str(exc)]
            print(f"  API attempt {attempt} failed: {exc}")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_SLEEP_SECONDS)
    return {}, "; ".join(last_errors) or f"failed after {MAX_RETRIES} attempts"


def dataset_counts(examples: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return dict(collections.Counter(str(example.get("dataset", "unknown")) for example in examples))


def build_hierarchical_categories(
    template_taxonomy: Sequence[Dict[str, Any]],
    examples_by_label: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    output = []
    for coarse in template_taxonomy:
        coarse_copy = {
            "coarse_grained_level": coarse.get("coarse_grained_level"),
            "definition": coarse.get("definition"),
            "intermediate_grained_categories": [],
        }
        for intermediate in coarse.get("intermediate_grained_categories", []):
            intermediate_copy = {
                "intermediate_grained_level": intermediate.get("intermediate_grained_level"),
                "definition": intermediate.get("definition"),
                "fine_grained_categories": [],
            }
            for fine in intermediate.get("fine_grained_categories", []):
                path = (
                    coarse.get("coarse_grained_level"),
                    intermediate.get("intermediate_grained_level"),
                    fine.get("fine_grained_level"),
                )
                examples = examples_by_label.get(path, [])
                if not examples:
                    continue
                fine_copy = copy.deepcopy(fine)
                fine_copy["assignment_count"] = len(examples)
                fine_copy["dataset_counts"] = dataset_counts(examples)
                fine_copy["examples"] = examples
                intermediate_copy["fine_grained_categories"].append(fine_copy)
            if intermediate_copy["fine_grained_categories"]:
                coarse_copy["intermediate_grained_categories"].append(intermediate_copy)
        if coarse_copy["intermediate_grained_categories"]:
            output.append(coarse_copy)
    return output


def build_output(
    args: argparse.Namespace,
    non_parsing_failures: Sequence[Dict[str, Any]],
    difference_extraction_failures: Sequence[Dict[str, Any]],
    classification_failures: Sequence[Dict[str, Any]],
    classification_disagreements: Sequence[Dict[str, Any]],
    classified_examples: Sequence[Dict[str, Any]],
    template_taxonomy: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    examples_by_label: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    classification_sources = collections.Counter(
        example.get("classification_source", "two_pass_agreement")
        for example in classified_examples
    )

    for example in classified_examples:
        labels = example.get("labels", [])
        for label in labels:
            path = (
                label["coarse_grained_level"],
                label["intermediate_grained_level"],
                label["fine_grained_level"],
            )
            category_example = {
                "example_id": example.get("example_id"),
                "dataset": example.get("dataset"),
                "log_message": example.get("log_message"),
                "parsed_template": example.get("parsed_template"),
                "ground_truth_template": example.get("ground_truth_template"),
                "difference_source": example.get("difference_source", DIFFERENCE_SOURCE),
                "classification_source": example.get("classification_source"),
                "labels": copy.deepcopy(example.get("labels", [])),
                "knowledge_spans": copy.deepcopy(example.get("knowledge_spans", [])),
            }
            examples_by_label[path].append(category_example)

    return {
        "metadata": {
            "source_file": args.input,
            "taxonomy_name": "LLM Log Parsing Error Taxonomy",
            "model": args.model,
            "total_errors": getattr(
                args,
                "input_error_count",
                len(classified_examples)
                + len(classification_disagreements)
                + len(classification_failures)
                + len(difference_extraction_failures)
                + len(non_parsing_failures),
            ),
            "classified_errors": len(classified_examples),
            "excluded_llm_failures": len(non_parsing_failures),
            "difference_source": DIFFERENCE_SOURCE,
            "difference_extraction_failures": len(difference_extraction_failures),
            "classification_failures": len(classification_failures),
            "classification_disagreements": len(classification_disagreements),
            "classification_sources": dict(classification_sources),
            "double_classification": True,
            "first_pass_temperature": FIRST_PASS_TEMPERATURE,
            "second_pass_temperature": SECOND_PASS_TEMPERATURE,
            "complete": (
                not difference_extraction_failures
                and not classification_failures
                and not classification_disagreements
            ),
        },
        "hierarchical_categories": build_hierarchical_categories(template_taxonomy, examples_by_label),
    }


def classification_signature(
    classification: Dict[str, Any],
) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
    return tuple(
        (
            label["fine_grained_level"],
            tuple(label.get("difference_ids", [])),
        )
        for label in classification.get("labels", [])
    )


def write_auxiliary_results(
    args: argparse.Namespace,
    difference_extraction_failures: Sequence[Dict[str, Any]],
    classification_failures: Sequence[Dict[str, Any]],
    classification_disagreements: Sequence[Dict[str, Any]],
) -> None:
    if difference_extraction_failures:
        write_json(
            Path(args.run_dir) / "difference_extraction_failures.json",
            difference_extraction_failures,
        )
    if classification_failures:
        write_json(Path(args.run_dir) / "classification_failures.json", classification_failures)
    if classification_disagreements:
        write_json(
            Path(args.run_dir) / "classification_disagreements.json",
            classification_disagreements,
        )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    ir_data = read_json(Path(args.input))
    prepare_output_path(args, count_ir_errors(ir_data))
    taxonomy_source = read_json(DEFAULT_TAXONOMY_FILE)
    template_taxonomy = taxonomy_source.get("taxonomy") or taxonomy_source.get("hierarchical_categories")
    if not template_taxonomy:
        raise ValueError(f"No taxonomy found in {DEFAULT_TAXONOMY_FILE}")

    parsing_errors, non_parsing_failures = flatten_ir_errors(ir_data)
    api_base, api_key = read_openai_key(DEFAULT_KEY_FILE, args.model)
    classified_examples: List[Dict[str, Any]] = []
    difference_extraction_failures: List[Dict[str, Any]] = []
    classification_failures: List[Dict[str, Any]] = []
    classification_disagreements: List[Dict[str, Any]] = []

    def save_checkpoint() -> Dict[str, Any]:
        current_output = build_output(
            args,
            non_parsing_failures,
            difference_extraction_failures,
            classification_failures,
            classification_disagreements,
            classified_examples,
            template_taxonomy,
        )
        write_json(Path(args.output), current_output)
        write_auxiliary_results(
            args,
            difference_extraction_failures,
            classification_failures,
            classification_disagreements,
        )
        return current_output

    output: Optional[Dict[str, Any]] = None
    for example_no, source_example in enumerate(parsing_errors, 1):
        example = copy.deepcopy(source_example)
        print(f"[{example_no}/{len(parsing_errors)}] classifying 1 example")
        print("  extracting abstract/preserve spans")
        differences, extraction_warnings = extract_knowledge_spans(
            example.get("log_message"),
            example.get("parsed_template"),
            example.get("ground_truth_template"),
        )
        for warning in extraction_warnings:
            print(f"  extraction warning: {warning}")
        if not differences:
            failure = copy.deepcopy(example)
            failure.update(
                {
                    "status": "difference_extraction_failed",
                    "failure_stage": "deterministic_log_template_alignment",
                    "reason": "; ".join(extraction_warnings)
                    or "no abstract or preserve span was found",
                }
            )
            difference_extraction_failures.append(failure)
            output = save_checkpoint()
            continue
        example["difference_source"] = DIFFERENCE_SOURCE
        example["knowledge_spans"] = differences

        pass_results = []
        for pass_name, temperature in (
            ("first", FIRST_PASS_TEMPERATURE),
            ("second", SECOND_PASS_TEMPERATURE),
        ):
            print(f"  {pass_name} classification")
            pass_results.append(
                classify_example(
                    args.model,
                    template_taxonomy,
                    example,
                    api_base,
                    api_key,
                    temperature=temperature,
                )
            )
        (first_result, first_error), (second_result, second_error) = pass_results
        example_id = example["example_id"]
        valid_classifications = []
        if first_error is None:
            valid_classifications.append(first_result[example_id])
        if second_error is None:
            valid_classifications.append(second_result[example_id])

        if not valid_classifications:
            fallback = deterministic_classification(example, template_taxonomy)
            if fallback is not None:
                print("  using deterministic classification fallback")
                example["labels"] = fallback["labels"]
                example["label_count"] = len(fallback["labels"])
                example["classification_source"] = "rule_fallback"
                classified_examples.append(example)
                output = save_checkpoint()
                continue

            print(f"  classification failed: first={first_error!r}, second={second_error!r}")
            failure = copy.deepcopy(example)
            failure.update(
                {
                    "status": "classification_failed",
                    "failure_stage": "llm_classification",
                    "failed_after_retries": MAX_RETRIES,
                    "first_classification_error": first_error,
                    "second_classification_error": second_error,
                    "first_labels": copy.deepcopy(
                        first_result.get(example["example_id"], {}).get("labels", [])
                    ),
                    "second_labels": copy.deepcopy(
                        second_result.get(example["example_id"], {}).get("labels", [])
                    ),
                    "labels": [],
                    "label_count": 0,
                }
            )
            classification_failures.append(failure)
            output = save_checkpoint()
            continue

        if len(valid_classifications) == 1:
            print("  accepting the single valid classification pass")
            accepted = valid_classifications[0]
            example["classification_source"] = "single_valid_pass"
        elif classification_signature(valid_classifications[0]) != classification_signature(
            valid_classifications[1]
        ):
            disagreement = copy.deepcopy(example)
            disagreement["status"] = "classification_disagreement"
            disagreement["first_labels"] = copy.deepcopy(valid_classifications[0]["labels"])
            disagreement["second_labels"] = copy.deepcopy(valid_classifications[1]["labels"])
            classification_disagreements.append(disagreement)
            output = save_checkpoint()
            continue
        else:
            accepted = valid_classifications[0]
            example["classification_source"] = "two_pass_agreement"

        example["labels"] = accepted["labels"]
        example["label_count"] = len(accepted["labels"])
        classified_examples.append(example)

        output = save_checkpoint()

    return output if output is not None else save_checkpoint()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify IR parsing errors with a fixed taxonomy.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="IR_results.json path.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Chat model used for classification.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        output = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    meta = output["metadata"]
    print(
        "Done. "
        f"classified={meta['classified_errors']}, "
        f"excluded={meta['excluded_llm_failures']}, "
        f"difference_failures={meta['difference_extraction_failures']}, "
        f"failures={meta['classification_failures']}, "
        f"disagreements={meta['classification_disagreements']}"
    )
    print(f"Saved to: {args.output}")
    print(f"Run directory: {args.run_dir}")
    return 0 if meta["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
