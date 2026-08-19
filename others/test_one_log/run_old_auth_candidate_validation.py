#!/usr/bin/env python3
"""Validate concise old-derived authentication prompt improvements."""

import json
from collections import Counter
from datetime import datetime

from run_webmaster_rule_ablation import ROOT, common_regexes, gq, parse


REPETITIONS = 3
MARKER_RULE = (
    "Preserve fixed authentication module, service, and context names inside "
    "wrappers, such as pam_unix(sshd:auth); punctuation inside the wrapper does "
    "not make the text variable."
)
STATUS_RULE = (
    "Preserve fixed authentication status phrases literally, such as 'user "
    "unknown'; a status word in such a phrase is not an account identifier."
)
ACCOUNT_RULE = (
    "In an authentication message, abstract a concrete account name as one "
    "{variables} when it follows 'for', 'user', or 'invalid user'. Preserve "
    "those cue words and preserve the fixed status phrase 'user unknown'; in "
    "that phrase, 'unknown' is a status rather than an account name."
)

CASES = {
    "pam_fixed": (
        "pam_unix(sshd:auth): check pass; user unknown",
        "pam_unix(sshd:auth): check pass; user unknown",
    ),
    "pam_dynamic_fields": (
        "pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 "
        "tty=ssh ruser= rhost=183.192.189.131 user=root",
        "pam_unix(sshd:auth): authentication failure; logname=<*> uid=<*> "
        "euid=<*> tty=<*> ruser=<*> rhost=<*> user=<*>",
    ),
    "failed_password": (
        "Failed password for root from 183.62.140.253 port 60221 ssh2",
        "Failed password for <*> from <*> port <*> ssh2",
    ),
    "invalid_user": (
        "Failed password for invalid user webmaster from 173.234.31.186 "
        "port 38926 ssh2",
        "Failed password for invalid user <*> from <*> port <*> ssh2",
    ),
}


def main():
    current = list(gq.CURATED_PROMPT_RULES)
    replaced = list(current)
    replaced[17] = ACCOUNT_RULE
    variants = {
        "current": current,
        "add_marker_and_status": [MARKER_RULE, STATUS_RULE] + current,
        "replace_account_plus_marker": [MARKER_RULE] + replaced,
    }
    preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    regs = common_regexes()
    records = []
    for case, (log, expected) in CASES.items():
        for variant, prompt in variants.items():
            for repetition in range(1, REPETITIONS + 1):
                result = parse(log, expected, prompt, preprocessing, regs)
                result.update(
                    case=case,
                    variant=variant,
                    repetition=repetition,
                )
                records.append(result)
                print(
                    f"{case:20s} {variant:28s} {repetition}/{REPETITIONS}: "
                    f"{'CORRECT' if result['correct'] else 'wrong'} "
                    f"{result['final']!r}",
                    flush=True,
                )

    output_dir = (
        ROOT
        / "others/test_one_log/results/domain_rule_ablation"
        / f"old_auth_candidate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {}
    for case in CASES:
        summary[case] = {}
        for variant in variants:
            selected = [
                item
                for item in records
                if item["case"] == case and item["variant"] == variant
            ]
            summary[case][variant] = {
                "correct": sum(item["correct"] for item in selected),
                "runs": len(selected),
                "templates": dict(Counter(item["final"] for item in selected)),
            }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
