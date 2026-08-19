#!/usr/bin/env python3
"""Ablate KnowledgeLog domain knowledge on BGL template-explosion cases.

This experiment mutates only in-memory rule groups. It does not edit gpt_query.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import regex


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

from logparser.KnowledgeLog import gpt_query as gq  # noqa: E402


MODEL = "Llama-3.3-70B-Instruct"

CASES = [
    {
        "id": "prepare_n3",
        "family": "prepare",
        "log": "PrepareForService shutting down NodeCard(mLctn(R63-M0-N3), mCardSernum(203231503833343000000000594c31304b34333431303857), mLp(FF:F2:9F:16:C2:20:00:0D:60:E9:3D:DF), mIp(10.6.0.145), mType(4)) as part of Service Action 409",
        "expected": "PrepareForService shutting down NodeCard(<*>, <*>, <*>, <*>, <*>) as part of Service Action <*>",
    },
    {
        "id": "prepare_n9",
        "family": "prepare",
        "log": "PrepareForService shutting down NodeCard(mLctn(R63-M0-N9), mCardSernum(203231503833343000000000594c31304b34333435303242), mLp(FF:F2:9F:16:C7:C5:00:0D:60:E9:38:3A), mIp(10.6.0.155), mType(4)) as part of Service Action 409",
        "expected": "PrepareForService shutting down NodeCard(<*>, <*>, <*>, <*>, <*>) as part of Service Action <*>",
    },
    {
        "id": "prepare_na",
        "family": "prepare",
        "log": "PrepareForService shutting down NodeCard(mLctn(R63-M0-NA), mCardSernum(203231503833343000000000594c31304b34333435303043), mLp(FF:F2:9F:16:C7:D4:00:0D:60:E9:38:2B), mIp(10.6.0.148), mType(4)) as part of Service Action 409",
        "expected": "PrepareForService shutting down NodeCard(<*>, <*>, <*>, <*>, <*>) as part of Service Action <*>",
    },
    {
        "id": "ido_status_1",
        "family": "ido_status",
        "log": "Ido chip status changed: FF:F2:9F:15:7D:76:00:0D:60:EA:82:89 ip=10.2.3.53 v=13 t=1 status=M Fri Jul 01 08:40:30 PDT 2005",
        "expected": "Ido chip status changed: <*> ip=<*> v=<*> t=<*> status=<*> <*>",
    },
    {
        "id": "ido_status_2",
        "family": "ido_status",
        "log": "Ido chip status changed: FF:F2:9F:16:E8:97:00:0D:60:E9:17:68 ip=10.0.0.108 v=9 t=4 status=M Fri Jul 01 08:01:53 PDT 2005",
        "expected": "Ido chip status changed: <*> ip=<*> v=<*> t=<*> status=<*> <*>",
    },
    {
        "id": "ido_status_3",
        "family": "ido_status",
        "log": "Ido chip status changed: FF:F2:9F:16:BF:E4:00:0D:60:E9:40:1B ip=10.3.0.18 v=13 t=4 status=M Fri Jul 01 08:54:18 PDT 2005",
        "expected": "Ido chip status changed: <*> ip=<*> v=<*> t=<*> status=<*> <*>",
    },
    {
        "id": "link_training_1",
        "family": "link_training",
        "log": "rts tree/torus link training failed: wanted: A B C X+ X- Y+ Y- Z+ Z- got: A B C X+ X- Y- Z+ Z-",
        "expected": "rts tree/torus link training failed: wanted: <*> got: <*>",
    },
    {
        "id": "link_training_2",
        "family": "link_training",
        "log": "rts tree/torus link training failed: wanted: A C X+ X- Y+ Y- Z+ Z- got: A C X+ Y+ Y- Z+ Z-",
        "expected": "rts tree/torus link training failed: wanted: <*> got: <*>",
    },
    {
        "id": "link_training_3",
        "family": "link_training",
        "log": "rts tree/torus link training failed: wanted: B C X+ X- Y+ Y- Z+ Z- got: B C X+ X- Y- Z+ Z-",
        "expected": "rts tree/torus link training failed: wanted: <*> got: <*>",
    },
    {
        "id": "assert_send",
        "family": "assert_function",
        "log": "idoproxydb hit ASSERT condition: ASSERT expression=0 Source file=idotransportmgr.cpp Source line=1043 Function=int IdoTransportMgr::SendPacket(IdoUdpMgr*, BglCtlPavTrace*)",
        "expected": "idoproxydb hit ASSERT condition: ASSERT expression=<*> Source file=<*> Source line=<*> Function=<*>",
    },
    {
        "id": "assert_close",
        "family": "assert_function",
        "log": "idoproxydb hit ASSERT condition: ASSERT expression=pTargetMgr Source file=idoclientmgr.cpp Source line=353 Function=int IdoClientMgr::TargetClose(const char*)",
        "expected": "idoproxydb hit ASSERT condition: ASSERT expression=<*> Source file=<*> Source line=<*> Function=<*>",
    },
    {
        "id": "torus_x",
        "family": "torus_direction",
        "log": "17 torus receiver x+ input pipe error(s) (dcr 0x02ec) detected and corrected",
        "expected": "<*> torus receiver <*> input pipe error(s) (dcr <*>) detected and corrected",
    },
    {
        "id": "torus_y",
        "family": "torus_direction",
        "log": "6 torus receiver y+ input pipe error(s) (dcr 0x02ee) detected and corrected",
        "expected": "<*> torus receiver <*> input pipe error(s) (dcr <*>) detected and corrected",
    },
    {
        "id": "torus_z",
        "family": "torus_direction",
        "log": "9 torus receiver z- input pipe error(s) (dcr 0x02f1) detected and corrected",
        "expected": "<*> torus receiver <*> input pipe error(s) (dcr <*>) detected and corrected",
    },
]

REGRESSION_CASES = [
    {
        "id": "reg_openstack_http",
        "family": "regression",
        "log": '10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1583 time: 0.1919448',
        "expected": '<*> "GET <*> <*>" status: <*> len: <*> time: <*>',
    },
    {
        "id": "reg_proxifier_endpoint",
        "family": "regression",
        "log": "proxy.cse.cuhk.edu.hk:5070 close, 0 bytes sent, 0 bytes received, lifetime 00:01",
        "expected": "<*> close, <*> bytes sent, <*> bytes received, lifetime <*>",
    },
    {
        "id": "reg_openssh_pam",
        "family": "regression",
        "log": "pam_unix(sshd:auth): check pass; user unknown",
        "expected": "pam_unix(sshd:auth): check pass; user unknown",
    },
    {
        "id": "reg_openssh_user",
        "family": "regression",
        "log": "pam_unix(sshd:session): session closed for user fztu",
        "expected": "pam_unix(sshd:session): session closed for user <*>",
    },
    {
        "id": "reg_linux_oom",
        "family": "regression",
        "log": "Out of Memory: Killed process 28473 (httpd).",
        "expected": "Out of Memory: Killed process <*> (<*>).",
    },
    {
        "id": "reg_bgl_ddr",
        "family": "regression",
        "log": "1 ddr errors(s) detected and corrected on rank 0, symbol 0, bit 3",
        "expected": "<*> ddr errors(s) detected and corrected on rank <*>, symbol <*> bit <*>",
    },
]

CANDIDATE_PROMPT_RULES = [
    "In a constructor-like field whose parenthesized arguments are named runtime records, replace each complete comma-separated argument, including its nested name and parentheses, with one {variables}; preserve the outer constructor name and its parentheses.",
    "Replace a complete colon-delimited hardware address with one {variables}; never preserve or abstract individual octets separately.",
    "Replace a complete ctime-style timestamp such as 'Fri Jul 01 08:40:30 PDT 2005' with one {variables}, including the weekday, month, day, time, timezone, and year.",
    "For a field named Function=, replace the complete function value with one {variables}, including return-type qualifiers, the qualified function or method name, and its complete parameter list.",
    "For repeated marker lists introduced by wanted: and got:, replace the complete list after each label with one {variables}; preserve the wanted: and got: labels.",
    "In torus sender or receiver messages, treat a direction token such as x+, x-, y+, y-, z+, or z- as one runtime field and replace it with {variables}.",
]

PROMPT_GROUPS = {
    "network_path": range(0, 5),
    "name_symbol_version": range(5, 8),
    "runtime_fields": range(8, 14),
    "composite_fields": range(14, 17),
    "account": range(17, 18),
    "state_format": range(18, 22),
    "specific_constants": range(22, 39),
}

PREPROCESSING_GROUPS = {
    "network": range(0, 6),
    "identifiers": range(6, 10),
    "time": range(10, 14),
    "numeric": range(14, 17),
    "account_sequence_kv": range(17, 20),
}


def common_regexes():
    path = ROOT / "benchmark/logparser/KnowledgeLog/common.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [regex.compile(pattern) for pattern in data["COMMON"]["regex"]]


def knowledge_text(rules):
    numbered = "\n".join(f"{index + 1}. {rule}" for index, rule in enumerate(rules))
    return "\nYou must follow these final domain knowledge rules:\n" + numbered + "\n"


def preprocess(log, rules):
    value = log
    for rule in rules:
        value = rule.apply(value)
    return value


def parse_one(variant, case, prompt_rules, preprocessing_rules, regs):
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
    final = None
    valid = False
    if extracted is not None:
        final, valid = gq.post_process_template(
            extracted,
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
        "final": final,
        "valid": valid,
        "correct": bool(valid and final == case["expected"]),
    }


def group_variants():
    prompt = list(gq.CURATED_PROMPT_RULES)
    pre = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    variants = {
        "current": (prompt, pre),
        "prompt_only": (prompt, []),
        "preprocessing_only": ([], pre),
        "no_domain_knowledge": ([], []),
    }
    for name, indices in PROMPT_GROUPS.items():
        removed = set(indices)
        variants[f"without_prompt_{name}"] = (
            [rule for index, rule in enumerate(prompt) if index not in removed],
            pre,
        )
    for name, indices in PREPROCESSING_GROUPS.items():
        removed = set(indices)
        variants[f"without_pre_{name}"] = (
            prompt,
            [rule for index, rule in enumerate(pre) if index not in removed],
        )
    return variants


def individual_variants(indices):
    prompt = list(gq.CURATED_PROMPT_RULES)
    pre = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    variants = {"current": (prompt, pre)}
    for one_based in indices:
        index = one_based - 1
        variants[f"without_rule_{one_based:02d}"] = (
            [rule for i, rule in enumerate(prompt) if i != index],
            pre,
        )
    return variants


def preprocessing_individual_variants(indices):
    prompt = list(gq.CURATED_PROMPT_RULES)
    pre = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    variants = {"current": (prompt, pre)}
    for one_based in indices:
        index = one_based - 1
        variants[f"without_pre_rule_{one_based:02d}_{pre[index].name}"] = (
            prompt,
            [rule for i, rule in enumerate(pre) if i != index],
        )
    return variants


def candidate_preprocessing_rules():
    return [
        gq.RegexRule(
            name="candidate_nested_runtime_argument",
            pattern=r"\bm(?:Lctn|CardSernum|Lp|Ip|Type)\([^)]*\)",
            replacement=gq.VARIABLE_PLACEHOLDER,
            note="Collapse each complete named runtime argument in a nested record.",
        ),
        gq.RegexRule(
            name="candidate_colon_hardware_address",
            pattern=r"(?<![A-Za-z0-9])(?:[0-9A-Fa-f]{2}:){5,}[0-9A-Fa-f]{2}(?![A-Za-z0-9])",
            replacement=gq.VARIABLE_PLACEHOLDER,
            note="Collapse a complete colon-delimited hardware address.",
        ),
        gq.RegexRule(
            name="candidate_ctime_timestamp",
            pattern=(
                r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
                r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
                r"\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+\d{4}\b"
            ),
            replacement=gq.VARIABLE_PLACEHOLDER,
            note="Collapse a complete ctime-style timestamp.",
        ),
        gq.RegexRule(
            name="candidate_function_field",
            pattern=r"(\bFunction=).+$",
            replacement=r"\1{variables}",
            note="Collapse the complete value of a trailing Function field.",
        ),
        gq.RegexRule(
            name="candidate_wanted_got_lists",
            pattern=r"(\bwanted:\s+).+?(\s+got:\s+).+$",
            replacement=r"\1{variables}\2{variables}",
            note="Collapse complete wanted and got marker lists.",
        ),
        gq.RegexRule(
            name="candidate_torus_direction",
            pattern=r"(\btorus\s+(?:receiver|sender)\s+)[xyz][+-]",
            replacement=r"\1{variables}",
            flags=gq.re.IGNORECASE,
            note="Collapse one torus direction token.",
        ),
    ]


def candidate_variants():
    prompt = list(gq.CURATED_PROMPT_RULES)
    pre = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    candidate_pre = candidate_preprocessing_rules()
    return {
        "current": (prompt, pre),
        "candidate_prompt_first": (CANDIDATE_PROMPT_RULES + prompt, pre),
        "candidate_prompt_last": (prompt + CANDIDATE_PROMPT_RULES, pre),
        "candidate_preprocessing_first": (prompt, candidate_pre + pre),
        "candidate_combined": (
            CANDIDATE_PROMPT_RULES + prompt,
            candidate_pre + pre,
        ),
        "candidate_remove_two_culprits": (
            CANDIDATE_PROMPT_RULES + prompt,
            [rule for index, rule in enumerate(pre) if index not in {5, 18}],
        ),
    }


def summarize(records):
    summary = {}
    variants = dict.fromkeys(record["variant"] for record in records)
    for variant in variants:
        selected = [record for record in records if record["variant"] == variant]
        family_summary = {}
        for family in dict.fromkeys(record["family"] for record in selected):
            family_records = [record for record in selected if record["family"] == family]
            outputs = Counter(record["final"] for record in family_records)
            family_summary[family] = {
                "correct": sum(record["correct"] for record in family_records),
                "runs": len(family_records),
                "unique_templates": len(outputs),
                "templates": dict(outputs),
            }
        summary[variant] = {
            "correct": sum(record["correct"] for record in selected),
            "runs": len(selected),
            "total_unique_within_families": sum(
                item["unique_templates"] for item in family_summary.values()
            ),
            "families": family_summary,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("group", "individual", "pre_individual", "candidate"),
        default="group",
    )
    parser.add_argument("--rules", default="6,7,13,15,16,17,22")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.stage == "group":
        variants = group_variants()
    elif args.stage == "individual":
        indices = [int(value) for value in args.rules.split(",") if value.strip()]
        variants = individual_variants(indices)
    elif args.stage == "pre_individual":
        indices = [int(value) for value in args.rules.split(",") if value.strip()]
        variants = preprocessing_individual_variants(indices)
    else:
        variants = candidate_variants()

    regs = common_regexes()
    futures = []
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        cases = CASES + REGRESSION_CASES if args.stage == "candidate" else CASES
        for variant, (prompt_rules, preprocessing_rules) in variants.items():
            for case in cases:
                for repetition in range(1, args.repetitions + 1):
                    future = pool.submit(
                            parse_one,
                            variant,
                            case,
                            prompt_rules,
                            preprocessing_rules,
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
                f"{record['case']}: {'correct' if record['correct'] else record['final']!r}",
                flush=True,
            )

    records.sort(key=lambda item: (item["variant"], item["family"], item["case"]))
    output_dir = args.output_dir or (
        ROOT
        / "others/test_one_log/results/domain_rule_ablation"
        / f"bgl_{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
