#!/usr/bin/env python3
"""Ablate prompt knowledge for one OpenSSH username parsing error."""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import regex


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

from logparser.KnowledgeLog import gpt_query as gq  # noqa: E402


MODEL = "Llama-3.3-70B-Instruct"
REPETITIONS = 3
TARGET = (
    "Failed password for invalid user webmaster from 173.234.31.186 port 38926 ssh2",
    "Failed password for invalid user <*> from <*> port <*> ssh2",
)
REGRESSION_CASES = {
    "failed_root": (
        "Failed password for root from 183.62.140.253 port 60221 ssh2",
        "Failed password for <*> from <*> port <*> ssh2",
    ),
    "accepted_root": (
        "Accepted publickey for root from ::ffff:10.100.4.251 port 35558 ssh2",
        "Accepted publickey for <*> from <*> port <*> ssh2",
    ),
    "pam_unknown": (
        "pam_unix(sshd:auth): check pass; user unknown",
        "pam_unix(sshd:auth): check pass; user unknown",
    ),
}

GROUPS = {
    "network_path": range(0, 5),
    "symbol_version": range(5, 8),
    "runtime_fields": range(8, 14),
    "composite_fields": range(14, 17),
    "account": range(17, 18),
    "state_format": range(18, 22),
    "specific_constants": range(22, 39),
}

CANDIDATE_ACCOUNT_RULE = (
    "In authentication messages, abstract a concrete account name as one "
    "{variables} when it follows 'invalid user', 'for', or 'user'. Preserve the "
    "words 'invalid user', status words such as 'unknown', and fixed protocol "
    "markers such as 'ssh2'."
)


def common_regexes():
    path = ROOT / "benchmark/logparser/KnowledgeLog/common.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [regex.compile(pattern) for pattern in data["COMMON"]["regex"]]


def parse(log, expected, rules, preprocessing, regs):
    prompt_group = "_webmaster_prompt"
    pre_group = "_webmaster_pre"
    gq.PROMPT_GROUPS[prompt_group] = list(rules)
    gq.PREPROCESSING_RULE_GROUPS[pre_group] = list(preprocessing)
    response = gq.get_response_from_openai_key(
        log,
        model=MODEL,
        enabled_prompt_groups=(prompt_group,),
        enabled_preprocessing_groups=(pre_group,),
    )
    extracted = gq._extract_template_from_response(response)
    if extracted is None:
        final, valid = None, False
    else:
        final, valid = gq.post_process_template(
            extracted, regs_common=regs, log_message=log
        )
    return {
        "log": log,
        "expected": expected,
        "preprocessed": gq.apply_final_domain_preprocessing(
            log, enabled_groups=(pre_group,)
        ),
        "response": response,
        "final": final,
        "valid": valid,
        "correct": bool(valid and final == expected),
    }


def main():
    rules = list(gq.CURATED_PROMPT_RULES)
    preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    regs = common_regexes()
    variants = {
        "current": (rules, preprocessing),
        "prompt_only": (rules, []),
        "preprocessing_only": ([], preprocessing),
        "no_knowledge": ([], []),
    }
    for name, indices in GROUPS.items():
        removed = set(indices)
        variants[f"without_{name}"] = (
            [rule for i, rule in enumerate(rules) if i not in removed],
            preprocessing,
        )
        variants[f"only_{name}"] = (
            [rule for i, rule in enumerate(rules) if i in removed],
            preprocessing,
        )

    records = []
    for variant, (variant_rules, variant_pre) in variants.items():
        for repetition in range(1, REPETITIONS + 1):
            record = parse(*TARGET, variant_rules, variant_pre, regs)
            record.update(variant=variant, repetition=repetition, case="target")
            records.append(record)
            print(
                f"{variant:28s} {repetition}/{REPETITIONS}: "
                f"{'CORRECT' if record['correct'] else 'wrong'} {record['final']!r}",
                flush=True,
            )

    candidate_rules = list(rules)
    candidate_rules[17] = CANDIDATE_ACCOUNT_RULE
    candidate_variants = {
        "candidate_replace_account": (candidate_rules, preprocessing),
        "candidate_account_first": (
            [CANDIDATE_ACCOUNT_RULE]
            + [rule for i, rule in enumerate(rules) if i != 17],
            preprocessing,
        ),
        "candidate_account_only": ([CANDIDATE_ACCOUNT_RULE], preprocessing),
    }
    cases = {"target": TARGET, **REGRESSION_CASES}
    for variant, (variant_rules, variant_pre) in candidate_variants.items():
        for case, (log, expected) in cases.items():
            for repetition in range(1, REPETITIONS + 1):
                record = parse(log, expected, variant_rules, variant_pre, regs)
                record.update(
                    variant=variant, repetition=repetition, case=case
                )
                records.append(record)
                print(
                    f"{variant:28s} {case:14s} {repetition}/{REPETITIONS}: "
                    f"{'CORRECT' if record['correct'] else 'wrong'} "
                    f"{record['final']!r}",
                    flush=True,
                )

    output_dir = (
        ROOT
        / "others/test_one_log/results/domain_rule_ablation"
        / f"webmaster_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {}
    for variant in dict.fromkeys(record["variant"] for record in records):
        summary[variant] = {}
        for case in dict.fromkeys(record["case"] for record in records):
            selected = [
                record
                for record in records
                if record["variant"] == variant and record["case"] == case
            ]
            if not selected:
                continue
            summary[variant][case] = {
                "correct": sum(record["correct"] for record in selected),
                "runs": len(selected),
                "templates": dict(
                    Counter(record["final"] for record in selected)
                ),
            }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
