#!/usr/bin/env python3
"""Validate a general regex for a missing OpenSSH authentication principal."""

import json
import re
from collections import Counter
from datetime import datetime

from run_webmaster_rule_ablation import ROOT, common_regexes, gq, parse


MODEL_REPETITIONS = 5

MISSING_AUTHENTICATION_PRINCIPAL = gq.RegexRule(
    name="missing_authentication_principal",
    pattern=(
        r"(\bFailed\s+password\s+for\s+)"
        r"(?:(?:invalid|unknown|illegal|nonexistent|unauthorized)\s+user)"
        r"(?=\s+from\b)"
    ),
    replacement=r"\1{variables}",
    flags=re.IGNORECASE,
    note=(
        "Abstract a status-only authentication principal when no concrete "
        "account name appears before the source field."
    ),
)

CASES = {
    "missing_principal": (
        "Failed password for invalid user from 103.204.177.144 port 2223 ssh2",
        "Failed password for <*> from <*> port <*> ssh2",
    ),
    "named_invalid_user": (
        "Failed password for invalid user webmaster from 173.234.31.186 port 38926 ssh2",
        "Failed password for invalid user <*> from <*> port <*> ssh2",
    ),
    "named_user": (
        "Failed password for root from 183.62.140.253 port 60221 ssh2",
        "Failed password for <*> from <*> port <*> ssh2",
    ),
    "different_event": (
        "pam_unix(sshd:auth): check pass; user unknown",
        "pam_unix(sshd:auth): check pass; user unknown",
    ),
    "generalized_status": (
        "Failed password for unknown user from 192.0.2.1 port 22 ssh2",
        "Failed password for <*> from <*> port <*> ssh2",
    ),
}


def main():
    prompt_rules = list(gq.CURATED_PROMPT_RULES)
    current_preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    variants = {
        "current": current_preprocessing,
        "candidate_regex": [
            MISSING_AUTHENTICATION_PRINCIPAL,
            *current_preprocessing,
        ],
    }
    regs = common_regexes()
    records = []

    for variant, preprocessing in variants.items():
        for case, (log_message, expected) in CASES.items():
            for repetition in range(1, MODEL_REPETITIONS + 1):
                record = parse(
                    log_message,
                    expected,
                    prompt_rules,
                    preprocessing,
                    regs,
                )
                record.update(
                    variant=variant,
                    case=case,
                    repetition=repetition,
                )
                records.append(record)
                print(
                    f"{variant:16s} {case:22s} {repetition}/{MODEL_REPETITIONS}: "
                    f"{'CORRECT' if record['correct'] else 'wrong'} "
                    f"{record['final']!r}",
                    flush=True,
                )

    output_dir = (
        ROOT
        / "others/test_one_log/results/regex_validation"
        / f"openssh_missing_principal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {}
    for variant in variants:
        summary[variant] = {}
        for case in CASES:
            selected = [
                record
                for record in records
                if record["variant"] == variant and record["case"] == case
            ]
            summary[variant][case] = {
                "correct": sum(record["correct"] for record in selected),
                "runs": len(selected),
                "templates": dict(Counter(record["final"] for record in selected)),
            }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
