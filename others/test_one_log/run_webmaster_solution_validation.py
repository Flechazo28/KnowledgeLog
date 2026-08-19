#!/usr/bin/env python3
"""Validate minimal prompt and preprocessing solutions for OpenSSH accounts."""

import json
from collections import Counter
from datetime import datetime

from run_webmaster_individual_ablation import MISSING_ACCOUNT_CASE
from run_webmaster_rule_ablation import (
    REGRESSION_CASES,
    ROOT,
    TARGET,
    common_regexes,
    gq,
    parse,
)


PRESERVE_UNKNOWN_RULE = (
    "Preserve the fixed authentication status phrase 'user unknown' exactly; "
    "'unknown' is a status word, not an account name."
)
ACCOUNT_RULE = (
    "In password or authentication messages, abstract a concrete account name "
    "after 'invalid user', 'for', or 'user' as one {variables}. If 'invalid user' "
    "is followed directly by 'from', no account name is present. Preserve fixed "
    "protocol markers such as 'ssh2'."
)
INVALID_USER_ACCOUNT_REGEX = gq.RegexRule(
    name="candidate_invalid_user_account",
    pattern=(
        r"(\binvalid\s+user\s+)"
        r"(?!(?:unknown|from)\b)"
        r"[A-Za-z_][A-Za-z0-9._-]*\b"
    ),
    replacement=r"\1{variables}",
    flags=gq.re.IGNORECASE,
    note="Abstract a concrete account name after the authentication phrase invalid user.",
)

SYNTHETIC_CASES = {
    "invalid_postgres": (
        "Failed password for invalid user postgres from 10.0.0.1 port 2201 ssh2",
        "Failed password for invalid user <*> from <*> port <*> ssh2",
    ),
    "invalid_git_deploy": (
        "Failed password for invalid user git_deploy from 10.0.0.2 port 2202 ssh2",
        "Failed password for invalid user <*> from <*> port <*> ssh2",
    ),
}


def main():
    rules = list(gq.CURATED_PROMPT_RULES)
    preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    prompt_rules = [
        PRESERVE_UNKNOWN_RULE,
        ACCOUNT_RULE,
        *[rule for i, rule in enumerate(rules) if i != 17],
    ]
    variants = {
        "current": (rules, preprocessing),
        "prompt_solution": (prompt_rules, preprocessing),
        "regex_solution": (
            rules,
            [INVALID_USER_ACCOUNT_REGEX, *preprocessing],
        ),
        "prompt_and_regex_solution": (
            prompt_rules,
            [INVALID_USER_ACCOUNT_REGEX, *preprocessing],
        ),
    }
    cases = {
        "target": TARGET,
        **REGRESSION_CASES,
        "missing_account": MISSING_ACCOUNT_CASE,
        **SYNTHETIC_CASES,
    }
    regs = common_regexes()
    records = []
    for variant, (variant_rules, variant_preprocessing) in variants.items():
        for case, (log, expected) in cases.items():
            for repetition in range(1, 6):
                record = parse(
                    log,
                    expected,
                    variant_rules,
                    variant_preprocessing,
                    regs,
                )
                record.update(
                    variant=variant,
                    case=case,
                    repetition=repetition,
                )
                records.append(record)
                print(
                    f"{variant:26s} {case:18s} {repetition}/5: "
                    f"{'CORRECT' if record['correct'] else 'wrong'} "
                    f"{record['final']!r}",
                    flush=True,
                )

    output_dir = (
        ROOT
        / "others/test_one_log/results/domain_rule_ablation"
        / f"webmaster_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {}
    for variant in variants:
        summary[variant] = {}
        for case in cases:
            selected = [
                record
                for record in records
                if record["variant"] == variant and record["case"] == case
            ]
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
