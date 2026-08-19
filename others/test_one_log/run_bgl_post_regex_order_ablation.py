#!/usr/bin/env python3
"""Compare BGL complete-field regexes before and after LLM inference.

This experiment changes rule placement only in memory. It does not edit
gpt_query.py.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from run_bgl_knowledge_ablation import (
    CASES,
    MODEL,
    candidate_preprocessing_rules,
    common_regexes,
    gq,
    knowledge_text,
    preprocess,
    summarize,
)


ROOT = Path(__file__).resolve().parents[2]
CURRENT_TWO_RULES = {"colon_hardware_address", "ctime_timestamp"}


def parse_one(variant, case, prompt_rules, preprocessing_rules, post_llm_rules, regs):
    instruction = (
        "I want you to act like an expert of log parsing. I will give you rules "
        "that you must follow and a log message delimited by backticks. You must "
        "identify and abstract all dynamic variables in logs with {variables}, then "
        "output a static log template. Please note that there might be no variables "
        "in logs. You must print the input log's template following this exact "
        "format: 'Log template:`...`'"
    )
    preprocessed = preprocess(case["log"], preprocessing_rules)
    response = gq.infer_llm(
        instruction,
        knowledge_text(prompt_rules),
        f"Log message: `{preprocessed}`",
        case["log"],
        model=MODEL,
    )
    extracted = gq._extract_template_from_response(response)
    postprocessed = None
    final = None
    valid = False
    if extracted is not None:
        postprocessed = preprocess(extracted, post_llm_rules)
        final, valid = gq.post_process_template(
            postprocessed,
            regs_common=regs,
            log_message=case["log"],
        )
    return {
        "variant": variant,
        "case": case["id"],
        "family": case["family"],
        "log": case["log"],
        "expected": case["expected"],
        "preprocessed": preprocessed,
        "response": response,
        "extracted": extracted,
        "postprocessed": postprocessed,
        "final": final,
        "valid": valid,
        "correct": bool(valid and final == case["expected"]),
    }


def variants():
    prompt = list(gq.CURATED_PROMPT_RULES)
    current_pre = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    original_pre = [rule for rule in current_pre if rule.name not in CURRENT_TWO_RULES]
    candidates = candidate_preprocessing_rules()
    candidate_two = [
        rule
        for rule in candidates
        if rule.name
        in {"candidate_colon_hardware_address", "candidate_ctime_timestamp"}
    ]
    return {
        "baseline_without_six": (prompt, original_pre, []),
        "current_two_pre_llm": (prompt, current_pre, []),
        "same_two_post_llm": (prompt, original_pre, candidate_two),
        "all_six_pre_llm": (prompt, candidates + original_pre, []),
        "all_six_post_llm": (prompt, original_pre, candidates),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    regs = common_regexes()
    records = []
    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for variant, (prompt_rules, pre_rules, post_rules) in variants().items():
            for case in CASES:
                for repetition in range(1, args.repetitions + 1):
                    future = pool.submit(
                        parse_one,
                        variant,
                        case,
                        prompt_rules,
                        pre_rules,
                        post_rules,
                        regs,
                    )
                    future.repetition = repetition
                    futures.append(future)

        for completed, future in enumerate(as_completed(futures), 1):
            record = future.result()
            record["repetition"] = future.repetition
            records.append(record)
            print(
                f"[{completed}/{len(futures)}] {record['variant']} "
                f"{record['case']}: "
                f"{'correct' if record['correct'] else record['final']!r}",
                flush=True,
            )

    records.sort(
        key=lambda item: (
            item["variant"],
            item["family"],
            item["case"],
            item["repetition"],
        )
    )
    output_dir = args.output_dir or (
        ROOT
        / "others/test_one_log/results/domain_rule_ablation"
        / f"bgl_post_order_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = summarize(records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
