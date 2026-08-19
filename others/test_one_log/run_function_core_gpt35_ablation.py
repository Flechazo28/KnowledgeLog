#!/usr/bin/env python3
"""Targeted GPT-3.5 ablation for Function= and core.<number> errors."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(HERE))

import run_zookeeper_bgl_gpt35_ablation as base  # noqa: E402
from logparser.KnowledgeLog import gpt_query as gq  # noqa: E402


REPETITIONS = 5
MAX_WORKERS = 4
CASES = {
    case["id"]: case
    for case in base.CASES
    if case["id"] in {"bgl_core_suffix", "bgl_function_field"}
}


def without_prompt(rules, prefix):
    return [rule for rule in rules if not rule.startswith(prefix)]


def configurations():
    prompt = list(gq.CURATED_PROMPT_RULES)
    preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    without_kv_regex = [
        rule for rule in preprocessing if rule.name != "curated_0031_key_value"
    ]
    common = {
        "current": (prompt, preprocessing),
        "prompt_only": (prompt, []),
        "preprocessing_only": ([], preprocessing),
        "no_domain_knowledge": ([], []),
        "without_name_process_prompt": (
            without_prompt(prompt, "Preserve a package, class, module"),
            preprocessing,
        ),
        "without_symbol_identifier_prompt": (
            without_prompt(prompt, "Abstract a symbol-rich runtime identifier"),
            preprocessing,
        ),
        "without_version_prompt": (
            without_prompt(prompt, "Abstract a complete version number"),
            preprocessing,
        ),
    }
    core = dict(common)
    core["candidate_core_regex"] = (
        prompt,
        [base.CANDIDATE_RULES["core"]] + preprocessing,
    )
    function = dict(common)
    function.update(
        {
            "without_key_value_prompt": (
                without_prompt(prompt, "In a key-value field"),
                preprocessing,
            ),
            "without_key_value_regex": (prompt, without_kv_regex),
            "candidate_function_regex": (
                prompt,
                [base.CANDIDATE_RULES["function"]] + preprocessing,
            ),
        }
    )
    return {
        "bgl_core_suffix": core,
        "bgl_function_field": function,
    }


def run_job(case_id, variant, config, repetition, regs):
    prompt, preprocessing = config
    result = base.parse_one(
        variant,
        CASES[case_id],
        prompt,
        preprocessing,
        False,
        regs,
    )
    result["repetition"] = repetition
    return result


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["case"], record["variant"])].append(record)
    result = {}
    for (case_id, variant), rows in sorted(grouped.items()):
        result.setdefault(case_id, {})[variant] = {
            "correct": sum(row["correct"] for row in rows),
            "runs": len(rows),
            "api_success": sum(row["api_success"] for row in rows),
            "outputs": dict(Counter(str(row["final"]) for row in rows)),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Verify only the current production configuration.",
    )
    args = parser.parse_args()
    output_dir = (
        HERE
        / "results/domain_rule_ablation"
        / f"function_core_gpt35_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "partial_results.jsonl"
    regs = base.common_regexes()
    jobs = []
    for case_id, variants in configurations().items():
        for variant, config in variants.items():
            if args.current_only and variant != "current":
                continue
            for repetition in range(1, REPETITIONS + 1):
                jobs.append((case_id, variant, config, repetition))

    records = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_job, *job, regs): (job[0], job[1], job[3])
            for job in jobs
        }
        with partial_path.open("a", encoding="utf-8") as partial:
            for index, future in enumerate(as_completed(futures), 1):
                case_id, variant, repetition = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # preserve every failed call as evidence
                    record = {
                        "case": case_id,
                        "variant": variant,
                        "repetition": repetition,
                        "api_success": False,
                        "correct": False,
                        "error": repr(exc),
                    }
                records.append(record)
                partial.write(json.dumps(record, ensure_ascii=False) + "\n")
                partial.flush()
                print(
                    f"[{index}/{len(jobs)}] {case_id} / {variant} / "
                    f"run {repetition}: correct={record.get('correct', False)}"
                )

    summary = summarize(records)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
