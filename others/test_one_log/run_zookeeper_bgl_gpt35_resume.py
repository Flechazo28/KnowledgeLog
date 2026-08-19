#!/usr/bin/env python3
"""Resume the interrupted Zookeeper/BGL GPT-3.5 ablation experiment."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(TEST_DIR))

import run_zookeeper_bgl_gpt35_ablation as base  # noqa: E402
from logparser.KnowledgeLog import gpt_query as gq  # noqa: E402


FIXED_NAME_RULE = (
    "Abstract only values that can change between event occurrences. Preserve "
    "CamelCase class or worker names and descriptive event-role phrases unless "
    "an explicit field presents them as runtime values."
)
PROTECTED_CONSTANT_RULE = (
    "The message may already contain {variables}. Keep every existing "
    "{variables} placeholder and preserve every other word and punctuation mark "
    "exactly; abstract only any remaining dynamic value."
)
STATE_RULE = gq.RegexRule(
    name="candidate_labeled_state",
    pattern=r"\b(?:LOOKING|FOLLOWING|LEADING)(?=\s+\((?:n\.state|my state)\))",
    replacement=gq.VARIABLE_PLACEHOLDER,
)

CASE_BY_ID = {case["id"]: case for case in base.CASES + base.REGRESSION_CASES}
FIXED_NAME_IDS = {
    "zk_interrupt_worker",
    "zk_worker_leaving",
    "bgl_handler_phrase",
}
DOTTED_IDS = {
    "bgl_dotted_program",
    "bgl_dotted_store",
    "bgl_dotted_address",
}
BUNDLE_IDS = {
    "zk_empty_error_value",
    "zk_notification_state",
    "bgl_core_suffix",
    "bgl_function_field",
    "bgl_torus_direction",
}
REGRESSION_IDS = {case["id"] for case in base.REGRESSION_CASES}


def configuration():
    prompt = list(gq.CURATED_PROMPT_RULES)
    pre = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    validated_pre = [
        base.CANDIDATE_RULES["core"],
        base.CANDIDATE_RULES["function"],
        base.CANDIDATE_RULES["direction"],
        STATE_RULE,
    ] + pre
    dotted_pre = [base.CANDIDATE_RULES["dotted"]] + pre
    return {
        "current_regression_baseline": (prompt, pre, False),
        "short_fixed_name_prompt": ([FIXED_NAME_RULE], pre, False),
        "short_fixed_name_no_pre": ([FIXED_NAME_RULE], [], False),
        "protected_dotted_prompt": ([PROTECTED_CONSTANT_RULE], dotted_pre, False),
        "validated_bundle": (prompt, validated_pre, True),
    }


def jobs():
    configs = configuration()
    plan = {
        "current_regression_baseline": (REGRESSION_IDS, 3),
        "short_fixed_name_prompt": (FIXED_NAME_IDS, 5),
        "short_fixed_name_no_pre": (FIXED_NAME_IDS, 5),
        "protected_dotted_prompt": (DOTTED_IDS, 5),
        "validated_bundle": (BUNDLE_IDS | REGRESSION_IDS, 3),
    }
    result = []
    for variant, (case_ids, repetitions) in plan.items():
        for case_id in sorted(case_ids):
            for repetition in range(1, repetitions + 1):
                result.append(
                    (variant, configs[variant], CASE_BY_ID[case_id], repetition)
                )
    return result


def summarize(records):
    valid_records = [record for record in records if record["api_success"]]
    grouped = defaultdict(list)
    for record in valid_records:
        grouped[record["variant"]].append(record)
    summary = {}
    for variant, rows in grouped.items():
        cases = {}
        for case_id in sorted({row["case"] for row in rows}):
            selected = [row for row in rows if row["case"] == case_id]
            cases[case_id] = {
                "correct": sum(row["correct"] for row in selected),
                "runs": len(selected),
                "outputs": dict(Counter(str(row["final"]) for row in selected)),
            }
        summary[variant] = {
            "correct": sum(row["correct"] for row in rows),
            "runs": len(rows),
            "cases": cases,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or (
        ROOT
        / "others/test_one_log/results/domain_rule_ablation"
        / f"zk_bgl_gpt35_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "partial_results.jsonl"
    records = []
    regs = base.common_regexes()
    planned_jobs = jobs()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for variant, config, case, repetition in planned_jobs:
            prompt_rules, preprocessing_rules, post_lock = config
            future = pool.submit(
                base.parse_one,
                variant,
                case,
                prompt_rules,
                preprocessing_rules,
                post_lock,
                regs,
            )
            futures[future] = repetition

        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            record["repetition"] = futures[future]
            records.append(record)
            with partial_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            status = "correct" if record["correct"] else repr(record["final"])
            if not record["api_success"]:
                status = "API_FAILED"
            print(
                f"[{index}/{len(planned_jobs)}] {record['variant']} "
                f"{record['case']}: {status}",
                flush=True,
            )

    records.sort(key=lambda row: (row["variant"], row["case"], row["repetition"]))
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summarize(records), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
