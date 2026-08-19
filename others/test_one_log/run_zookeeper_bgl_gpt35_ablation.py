#!/usr/bin/env python3
"""Ablate GPT-3.5-Turbo failures on high-impact Zookeeper and BGL logs.

All candidate rules are applied in memory. The production parser is not edited.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import regex


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

from logparser.KnowledgeLog import gpt_query as gq  # noqa: E402


MODEL = "gpt-3.5-turbo"
INSTRUCTION = (
    "I want you to act like an expert of log parsing. I will give you rules "
    "that you must follow and a log message delimited by backticks. You must "
    "identify and abstract all dynamic variables in logs with {variables}, then "
    "output a static log template. Please note that there might be no variables "
    "in logs. You must print the input log's template following this exact "
    "format: 'Log template:`...`'"
)

CASES = [
    {
        "id": "zk_interrupt_worker",
        "dataset": "Zookeeper",
        "family": "fixed_event_name",
        "log": "Interrupting SendWorker",
        "expected": "Interrupting SendWorker",
    },
    {
        "id": "zk_worker_leaving",
        "dataset": "Zookeeper",
        "family": "fixed_event_name",
        "log": "Send worker leaving thread",
        "expected": "Send worker leaving thread",
    },
    {
        "id": "zk_empty_error_value",
        "dataset": "Zookeeper",
        "family": "protected_format",
        "log": "Connection broken for id 188978561024, my id = 1, error =",
        "expected": "Connection broken for id <*>, my id = <*>, error =<*>",
    },
    {
        "id": "zk_notification_state",
        "dataset": "Zookeeper",
        "family": "labeled_state",
        "log": (
            "Notification: 1 (n.leader), 0x0 (n.zxid), 0x1 (n.round), "
            "LOOKING (n.state), 1 (n.sid), 0x0 (n.peerEPoch), "
            "FOLLOWING (my state)"
        ),
        "expected": (
            "Notification: <*> (n.leader), <*> (n.zxid), <*> (n.round), "
            "<*> (n.state), <*> (n.sid), <*> (n.peerEPoch), <*> (my state)"
        ),
    },
    {
        "id": "zk_oracle_comma",
        "dataset": "Zookeeper",
        "family": "oracle_punctuation",
        "log": "Expiring session 0x34ede65503f0003, timeout of 10000ms exceeded",
        "expected": "Expiring session <*> timeout of <*> exceeded",
    },
    {
        "id": "bgl_core_suffix",
        "dataset": "BGL",
        "family": "fixed_prefix_numeric_suffix",
        "log": "generating core.304",
        "expected": "generating core.<*>",
    },
    {
        "id": "bgl_handler_phrase",
        "dataset": "BGL",
        "family": "fixed_event_name",
        "log": (
            "0 microseconds spent in the rbs signal handler during 0 calls. "
            "0 microseconds was the maximum time for a single instance of a "
            "correctable ddr."
        ),
        "expected": (
            "<*> microseconds spent in the rbs signal handler during <*> calls. "
            "<*> microseconds was the maximum time for a single instance of a "
            "correctable ddr."
        ),
    },
    {
        "id": "bgl_function_field",
        "dataset": "BGL",
        "family": "complete_key_value",
        "log": (
            "idoproxydb hit ASSERT condition: ASSERT expression=0 "
            "Source file=idotransportmgr.cpp Source line=1043 "
            "Function=int IdoTransportMgr::SendPacket(IdoUdpMgr*, BglCtlPavTrace*)"
        ),
        "expected": (
            "idoproxydb hit ASSERT condition: ASSERT expression=<*> "
            "Source file=<*> Source line=<*> Function=<*>"
        ),
    },
    {
        "id": "bgl_dotted_program",
        "dataset": "BGL",
        "family": "dotted_diagnostic",
        "log": "program interrupt: fp compare...............0",
        "expected": "program interrupt: fp compare...............<*>",
    },
    {
        "id": "bgl_dotted_store",
        "dataset": "BGL",
        "family": "dotted_diagnostic",
        "log": "data store interrupt caused by dcbf.........0",
        "expected": "data store interrupt caused by dcbf.........<*>",
    },
    {
        "id": "bgl_dotted_address",
        "dataset": "BGL",
        "family": "dotted_diagnostic",
        "log": "instruction address space.........0",
        "expected": "instruction address space.........<*>",
    },
    {
        "id": "bgl_torus_direction",
        "dataset": "BGL",
        "family": "direction_field",
        "log": (
            "17 torus receiver x+ input pipe error(s) (dcr 0x02ec) "
            "detected and corrected"
        ),
        "expected": (
            "<*> torus receiver <*> input pipe error(s) (dcr <*>) "
            "detected and corrected"
        ),
    },
    {
        "id": "bgl_oracle_comma",
        "dataset": "BGL",
        "family": "oracle_punctuation",
        "log": "1 ddr errors(s) detected and corrected on rank 0, symbol 0, bit 3",
        "expected": "<*> ddr errors(s) detected and corrected on rank <*>, symbol <*> bit <*>",
    },
    {
        "id": "bgl_lustre_point",
        "dataset": "BGL",
        "family": "fixed_field_label",
        "log": "Lustre mount FAILED : bglio118 : point /p/gb1",
        "expected": "Lustre mount FAILED : <*> : point <*>",
    },
]

REGRESSION_CASES = [
    {
        "id": "reg_openstack_http",
        "dataset": "OpenStack",
        "family": "regression",
        "log": (
            '10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/'
            'servers/detail HTTP/1.1" status: 200 len: 1583 time: 0.1919448'
        ),
        "expected": '<*> "GET <*> <*>" status: <*> len: <*> time: <*>',
    },
    {
        "id": "reg_proxifier_endpoint",
        "dataset": "Proxifier",
        "family": "regression",
        "log": (
            "proxy.cse.cuhk.edu.hk:5070 close, 0 bytes sent, 0 bytes received, "
            "lifetime 00:01"
        ),
        "expected": "<*> close, <*> bytes sent, <*> bytes received, lifetime <*>",
    },
    {
        "id": "reg_openssh_constant",
        "dataset": "OpenSSH",
        "family": "regression",
        "log": "pam_unix(sshd:auth): check pass; user unknown",
        "expected": "pam_unix(sshd:auth): check pass; user unknown",
    },
    {
        "id": "reg_linux_process",
        "dataset": "Linux",
        "family": "regression",
        "log": "Out of Memory: Killed process 28473 (httpd).",
        "expected": "Out of Memory: Killed process <*> (<*>).",
    },
]

PRESERVE_EVENT_RULE = (
    "Preserve class, worker, thread, component, handler, diagnostic, hardware-"
    "subsystem, and event names when they are fixed event wording. Do not "
    "abstract a noun or name only because it follows an action verb."
)
LABELED_STATE_RULE = (
    "When a value is immediately followed by a parenthesized semantic field "
    "label, such as LOOKING (n.state), abstract the value and preserve the label."
)


def regex_rule(name, pattern, replacement, flags=0):
    return gq.RegexRule(name=name, pattern=pattern, replacement=replacement, flags=flags)


CANDIDATE_RULES = {
    "core": regex_rule(
        "candidate_core_dump_numeric_suffix",
        r"\bcore\.\d+\b",
        f"core.{gq.VARIABLE_PLACEHOLDER}",
        re.IGNORECASE,
    ),
    "dotted": regex_rule(
        "candidate_dotted_diagnostic_counter",
        r"(\.{2,})[-+]?(?:0x[0-9A-Fa-f]+|\d+(?:\.\d+)?)\s*$",
        rf"\1{gq.VARIABLE_PLACEHOLDER}",
    ),
    "function": regex_rule(
        "candidate_complete_function_field",
        r"(\bFunction=).+$",
        r"\1{variables}",
    ),
    "direction": regex_rule(
        "candidate_torus_direction",
        r"(\btorus\s+(?:receiver|sender)\s+)[xyz][+-]",
        r"\1{variables}",
        re.IGNORECASE,
    ),
}


def common_regexes():
    path = ROOT / "benchmark/logparser/KnowledgeLog/common.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [regex.compile(pattern) for pattern in payload["COMMON"]["regex"]]


def knowledge_text(rules):
    if not rules:
        return ""
    body = "\n".join(f"{index + 1}. {rule}" for index, rule in enumerate(rules))
    return "\nYou must follow these final domain knowledge rules:\n" + body + "\n"


def preprocess(text, rules):
    for rule in rules:
        text = rule.apply(text)
    return text


def empty_rhs_postprocess(template):
    return re.sub(r"(\b[A-Za-z_][A-Za-z0-9_.-]*\s*=)\s*<\*>\s*$", r"\1<*>", template)


def parse_one(variant, case, prompt_rules, preprocessing_rules, post_lock, regs):
    preprocessed = preprocess(case["log"], preprocessing_rules)
    response = gq.infer_llm(
        INSTRUCTION,
        knowledge_text(prompt_rules),
        f"Log message: `{preprocessed}`",
        case["log"],
        model=MODEL,
        temperature=0.0,
    )
    extracted = gq._extract_template_from_response(response)
    final = None
    valid = False
    if extracted is not None:
        final, valid = gq.post_process_template(
            extracted,
            regs_common=regs,
            log_message=case["log"],
        )
        if post_lock:
            final = empty_rhs_postprocess(final)
    return {
        "variant": variant,
        "case": case["id"],
        "dataset": case["dataset"],
        "family": case["family"],
        "log": case["log"],
        "expected": case["expected"],
        "preprocessed": preprocessed,
        "response": response,
        "api_success": not response.startswith("Log message: `"),
        "extracted": extracted,
        "final": final,
        "valid": valid,
        "correct": bool(valid and final == case["expected"]),
    }


def variants():
    prompt = list(gq.CURATED_PROMPT_RULES)
    pre = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    without_name = [rule for index, rule in enumerate(prompt) if index != 5]
    without_symbol = [rule for index, rule in enumerate(prompt) if index != 6]
    without_kv = [rule for rule in pre if rule.name != "curated_0031_key_value"]
    candidate_pre = list(CANDIDATE_RULES.values()) + pre
    candidate_prompt = [PRESERVE_EVENT_RULE, LABELED_STATE_RULE] + prompt
    return {
        "current": (prompt, pre, False),
        "prompt_only": (prompt, [], False),
        "preprocessing_only": ([], pre, False),
        "no_domain_knowledge": ([], [], False),
        "without_name_rule": (without_name, pre, False),
        "without_symbol_rule": (without_symbol, pre, False),
        "without_key_value_regex": (prompt, without_kv, False),
        "candidate_preserve_prompt": ([PRESERVE_EVENT_RULE] + prompt, pre, False),
        "candidate_state_prompt": ([LABELED_STATE_RULE] + prompt, pre, False),
        "candidate_core_regex": (prompt, [CANDIDATE_RULES["core"]] + pre, False),
        "candidate_dotted_regex": (prompt, [CANDIDATE_RULES["dotted"]] + pre, False),
        "candidate_function_regex": (prompt, [CANDIDATE_RULES["function"]] + pre, False),
        "candidate_direction_regex": (prompt, [CANDIDATE_RULES["direction"]] + pre, False),
        "candidate_empty_rhs_lock": (prompt, pre, True),
        "candidate_combined": (candidate_prompt, candidate_pre, True),
    }


TARGET_CASES = {
    "candidate_preserve_prompt": {
        "zk_interrupt_worker", "zk_worker_leaving", "bgl_handler_phrase",
    },
    "candidate_state_prompt": {"zk_notification_state"},
    "candidate_core_regex": {"bgl_core_suffix"},
    "candidate_dotted_regex": {
        "bgl_dotted_program", "bgl_dotted_store", "bgl_dotted_address",
    },
    "candidate_function_regex": {"bgl_function_field"},
    "candidate_direction_regex": {"bgl_torus_direction"},
    "candidate_empty_rhs_lock": {"zk_empty_error_value"},
}


def schedule():
    all_cases = CASES
    regression_ids = {case["id"] for case in REGRESSION_CASES}
    case_by_id = {case["id"]: case for case in CASES + REGRESSION_CASES}
    jobs = []
    for variant, config in variants().items():
        if variant.startswith("candidate_") and variant != "candidate_combined":
            ids = TARGET_CASES[variant] | regression_ids
            repetitions = 2
            selected = [case_by_id[case_id] for case_id in sorted(ids)]
        elif variant == "candidate_combined":
            selected = CASES + REGRESSION_CASES
            repetitions = 3
        else:
            selected = all_cases + (REGRESSION_CASES if variant == "current" else [])
            repetitions = 1
        for case in selected:
            for repetition in range(1, repetitions + 1):
                jobs.append((variant, config, case, repetition))
    return jobs


def summarize(records):
    records = [record for record in records if record.get("api_success", True)]
    summary = {}
    grouped = defaultdict(list)
    for record in records:
        grouped[record["variant"]].append(record)
    for variant, selected in grouped.items():
        case_summary = {}
        for case_id in sorted({record["case"] for record in selected}):
            rows = [record for record in selected if record["case"] == case_id]
            outputs = Counter(str(record["final"]) for record in rows)
            case_summary[case_id] = {
                "correct": sum(record["correct"] for record in rows),
                "runs": len(rows),
                "outputs": dict(outputs),
            }
        summary[variant] = {
            "correct": sum(record["correct"] for record in selected),
            "runs": len(selected),
            "case_accuracy": sum(any(r["correct"] for r in selected if r["case"] == cid) for cid in case_summary) / len(case_summary),
            "cases": case_summary,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or (
        ROOT / "others/test_one_log/results/domain_rule_ablation" /
        f"zk_bgl_gpt35_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    regs = common_regexes()
    jobs = schedule()
    records = []
    partial_path = output_dir / "partial_results.jsonl"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for variant, config, case, repetition in jobs:
            prompt_rules, preprocessing_rules, post_lock = config
            future = pool.submit(
                parse_one,
                variant,
                case,
                prompt_rules,
                preprocessing_rules,
                post_lock,
                regs,
            )
            futures[future] = repetition
        for completed, future in enumerate(as_completed(futures), 1):
            record = future.result()
            record["repetition"] = futures[future]
            records.append(record)
            with partial_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{completed}/{len(futures)}] {record['variant']} "
                f"{record['case']}: "
                f"{'correct' if record['correct'] else record['final']!r}",
                flush=True,
            )

    records.sort(key=lambda item: (item["variant"], item["case"], item["repetition"]))
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
