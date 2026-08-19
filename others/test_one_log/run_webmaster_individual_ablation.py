#!/usr/bin/env python3
"""Run individual prompt-rule ablation and candidate regression tests."""

import json
from collections import Counter
from datetime import datetime

from run_webmaster_rule_ablation import (
    MODEL,
    REGRESSION_CASES,
    ROOT,
    TARGET,
    common_regexes,
    gq,
    parse,
)


CANDIDATE_ACCOUNT_RULE = (
    "In password or authentication messages, abstract a concrete account name "
    "after 'invalid user', 'for', or 'user' as one {variables}. Do not abstract "
    "the status word 'unknown' or the source marker 'from'. Preserve fixed "
    "protocol markers such as 'ssh2'."
)

MISSING_ACCOUNT_CASE = (
    "Failed password for invalid user from 103.204.177.144 port 2223 ssh2",
    "Failed password for <*> from <*> port <*> ssh2",
)


def main():
    rules = list(gq.CURATED_PROMPT_RULES)
    preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    regs = common_regexes()
    records = []

    # Add one rule to the otherwise successful preprocessing-only baseline.
    for index, rule in enumerate(rules):
        record = parse(*TARGET, [rule], preprocessing, regs)
        record.update(
            variant=f"only_rule_{index + 1:02d}",
            case="target",
            rule_index=index + 1,
            rule=rule,
        )
        records.append(record)
        print(
            f"only rule {index + 1:02d}: "
            f"{'CORRECT' if record['correct'] else 'wrong'} {record['final']!r}",
            flush=True,
        )

    # Remove one rule from the current full prompt.
    for index, rule in enumerate(rules):
        remaining = [item for i, item in enumerate(rules) if i != index]
        record = parse(*TARGET, remaining, preprocessing, regs)
        record.update(
            variant=f"without_rule_{index + 1:02d}",
            case="target",
            rule_index=index + 1,
            rule=rule,
        )
        records.append(record)
        print(
            f"without rule {index + 1:02d}: "
            f"{'CORRECT' if record['correct'] else 'wrong'} {record['final']!r}",
            flush=True,
        )

    candidate_variants = {
        "existing_account_first": (
            [rules[17]] + [rule for i, rule in enumerate(rules) if i != 17]
        ),
        "candidate_account_first": (
            [CANDIDATE_ACCOUNT_RULE]
            + [rule for i, rule in enumerate(rules) if i != 17]
        ),
        "candidate_account_first_without_rule_06": (
            [CANDIDATE_ACCOUNT_RULE]
            + [rule for i, rule in enumerate(rules) if i not in {5, 17}]
        ),
    }
    cases = {
        "target": TARGET,
        **REGRESSION_CASES,
        "missing_account": MISSING_ACCOUNT_CASE,
    }
    for variant, variant_rules in candidate_variants.items():
        for case, (log, expected) in cases.items():
            for repetition in range(1, 6):
                record = parse(log, expected, variant_rules, preprocessing, regs)
                record.update(
                    variant=variant,
                    case=case,
                    repetition=repetition,
                )
                records.append(record)
                print(
                    f"{variant:42s} {case:15s} {repetition}/5: "
                    f"{'CORRECT' if record['correct'] else 'wrong'} "
                    f"{record['final']!r}",
                    flush=True,
                )

    output_dir = (
        ROOT
        / "others/test_one_log/results/domain_rule_ablation"
        / f"webmaster_individual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            if selected:
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
