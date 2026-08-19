#!/usr/bin/env python3
"""Ablate the three dominant Zookeeper PA errors without changing production code."""

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


TARGET_IDS = {
    "zk_interrupt_worker",
    "zk_worker_leaving",
    "zk_empty_error_value",
}
REGRESSION_IDS = {
    "reg_linux_process",
    "reg_openssh_constant",
    "reg_openstack_http",
    "reg_proxifier_endpoint",
    "bgl_handler_phrase",
    "bgl_core_suffix",
    "bgl_function_field",
}
CASE_BY_ID = {case["id"]: case for case in base.CASES + base.REGRESSION_CASES}

NAME_PREFIX = "Preserve a package, class, module"
PID_PREFIX = "Abstract a process ID when"
STATE_PREFIX = "Abstract a state, lifecycle stage"

REWRITTEN_NAME_RULE = (
    "Preserve package, class, module, method, file, protocol, service, component, "
    "worker, thread, task, event-handler, and event-role names when they are fixed "
    "event wording. Abstract such a name only when an explicit field or changing "
    "identifier marks it as runtime data."
)
REWRITTEN_PID_RULE = (
    "Abstract a process identifier only when an explicit PID, numeric identifier, "
    "or process-ID field marks it as runtime data; worker or thread context alone "
    "does not make a neighboring name variable."
)
MINIMAL_PID_RULE = (
    "Abstract a process ID only when an explicit PID or process-ID field identifies "
    "it; preserve the surrounding key and punctuation."
)
FIXED_EVENT_BOUNDARY_RULE = (
    "Preserve a worker or thread name in fixed event wording; an action verb or "
    "worker/thread context alone does not make the name variable."
)
EMPTY_RHS_RULE = (
    "When a key has no value after '=', place {variables} immediately after '=' "
    "without adding a space."
)


def remove_rule(rules, prefix):
    return [rule for rule in rules if not rule.startswith(prefix)]


def replace_rule(rules, prefix, replacement):
    return [replacement if rule.startswith(prefix) else rule for rule in rules]


def configurations():
    prompt = list(gq.CURATED_PROMPT_RULES)
    preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    without_kv = [
        rule for rule in preprocessing if rule.name != "curated_0031_key_value"
    ]
    without_name = remove_rule(prompt, NAME_PREFIX)
    without_pid = remove_rule(prompt, PID_PREFIX)
    without_state = remove_rule(prompt, STATE_PREFIX)
    without_joint = remove_rule(remove_rule(remove_rule(prompt, NAME_PREFIX), PID_PREFIX), STATE_PREFIX)
    rewritten_name = replace_rule(prompt, NAME_PREFIX, REWRITTEN_NAME_RULE)
    rewritten_pid = replace_rule(prompt, PID_PREFIX, REWRITTEN_PID_RULE)
    rewritten_both = replace_rule(rewritten_name, PID_PREFIX, REWRITTEN_PID_RULE)

    shared = {
        "current": (prompt, preprocessing, False),
        "prompt_only": (prompt, [], False),
        "preprocessing_only": ([], preprocessing, False),
        "no_domain_knowledge": ([], [], False),
    }
    fixed_name = {
        **shared,
        "without_name_rule": (without_name, preprocessing, False),
        "without_pid_rule": (without_pid, preprocessing, False),
        "without_state_rule": (without_state, preprocessing, False),
        "without_name_pid_state": (without_joint, preprocessing, False),
        "rewrite_name_rule": (rewritten_name, preprocessing, False),
        "rewrite_pid_rule": (rewritten_pid, preprocessing, False),
        "rewrite_name_and_pid": (rewritten_both, preprocessing, False),
        "append_fixed_event_boundary": (
            prompt + [FIXED_EVENT_BOUNDARY_RULE],
            preprocessing,
            False,
        ),
    }
    empty_rhs = {
        **shared,
        "without_key_value_regex": (prompt, without_kv, False),
        "append_empty_rhs_rule": (prompt + [EMPTY_RHS_RULE], preprocessing, False),
        "deterministic_empty_rhs_lock": (prompt, preprocessing, True),
    }
    return {
        "zk_interrupt_worker": fixed_name,
        "zk_worker_leaving": fixed_name,
        "zk_empty_error_value": empty_rhs,
    }


def regression_configurations():
    prompt = list(gq.CURATED_PROMPT_RULES)
    preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    rewritten_name = replace_rule(prompt, NAME_PREFIX, REWRITTEN_NAME_RULE)
    rewritten_pid = replace_rule(prompt, PID_PREFIX, REWRITTEN_PID_RULE)
    return {
        "current": (prompt, preprocessing, False),
        "rewrite_name_rule": (rewritten_name, preprocessing, False),
        "rewrite_pid_rule": (rewritten_pid, preprocessing, False),
        "rewrite_name_and_pid": (
            replace_rule(rewritten_name, PID_PREFIX, REWRITTEN_PID_RULE),
            preprocessing,
            False,
        ),
        "append_fixed_event_boundary": (
            prompt + [FIXED_EVENT_BOUNDARY_RULE],
            preprocessing,
            False,
        ),
        "append_empty_rhs_rule": (prompt + [EMPTY_RHS_RULE], preprocessing, False),
    }


def pid_followup_configurations():
    prompt = list(gq.CURATED_PROMPT_RULES)
    preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    return {
        "current": (prompt, preprocessing, False),
        "without_pid_rule": (remove_rule(prompt, PID_PREFIX), preprocessing, False),
        "minimal_pid_rewrite": (
            replace_rule(prompt, PID_PREFIX, MINIMAL_PID_RULE),
            preprocessing,
            False,
        ),
    }


def run_one(case_id, variant, config, repetition, regs):
    prompt, preprocessing, post_lock = config
    record = base.parse_one(
        variant,
        CASE_BY_ID[case_id],
        prompt,
        preprocessing,
        post_lock,
        regs,
    )
    record["repetition"] = repetition
    return record


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["case"], record["variant"])].append(record)
    summary = {}
    for (case_id, variant), rows in sorted(grouped.items()):
        summary.setdefault(case_id, {})[variant] = {
            "correct": sum(row.get("correct", False) for row in rows),
            "runs": len(rows),
            "api_success": sum(row.get("api_success", False) for row in rows),
            "outputs": dict(Counter(str(row.get("final")) for row in rows)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("targets", "regression", "pid_followup"),
        default="targets",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        HERE
        / "results/domain_rule_ablation"
        / f"zookeeper_top3_gpt35_{args.stage}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    regs = base.common_regexes()
    repetitions = 5 if args.stage == "targets" else 3
    jobs = []
    if args.stage == "targets":
        for case_id, variants in configurations().items():
            for variant, config in variants.items():
                for repetition in range(1, repetitions + 1):
                    jobs.append((case_id, variant, config, repetition))
    elif args.stage == "regression":
        variants = regression_configurations()
        for case_id in sorted(REGRESSION_IDS):
            for variant, config in variants.items():
                for repetition in range(1, repetitions + 1):
                    jobs.append((case_id, variant, config, repetition))
    else:
        variants = pid_followup_configurations()
        for case_id in sorted({"zk_interrupt_worker", "zk_worker_leaving"} | REGRESSION_IDS):
            case_repetitions = 5 if case_id.startswith("zk_") else 3
            for variant, config in variants.items():
                for repetition in range(1, case_repetitions + 1):
                    jobs.append((case_id, variant, config, repetition))

    records = []
    partial_path = output_dir / "partial_results.jsonl"
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, *job, regs): (job[0], job[1], job[3])
            for job in jobs
        }
        with partial_path.open("a", encoding="utf-8") as partial:
            for index, future in enumerate(as_completed(futures), 1):
                case_id, variant, repetition = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
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
                    f"run {repetition}: correct={record.get('correct', False)}",
                    flush=True,
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
