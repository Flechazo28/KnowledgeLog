#!/usr/bin/env python3
"""Ablate individual prompt and preprocessing rules on Thunderbird cases."""

import json
from datetime import datetime

from run_thunderbird_rule_ablation import CASES
from run_webmaster_rule_ablation import ROOT, common_regexes, gq, parse


def main():
    rules = list(gq.CURATED_PROMPT_RULES)
    preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    regs = common_regexes()
    records = []

    # Preprocessing interaction: mail queue IDs and complete source locations.
    for index, removed_rule in enumerate(preprocessing):
        remaining = [
            rule for i, rule in enumerate(preprocessing) if i != index
        ]
        for case in ("mail_queue_id", "source_location"):
            record = parse(*CASES[case], rules, remaining, regs)
            record.update(
                stage="preprocessing",
                case=case,
                removed_index=index + 1,
                removed_rule=removed_rule.name,
            )
            records.append(record)
            print(
                f"without pre {index + 1:02d} {removed_rule.name:38s} "
                f"{case:18s}: "
                f"{'CORRECT' if record['correct'] else 'wrong'} "
                f"{record['final']!r}",
                flush=True,
            )

    # Prompt interference: process names and lifecycle subjects.
    for index, removed_rule in enumerate(rules):
        remaining = [rule for i, rule in enumerate(rules) if i != index]
        for case in (
            "segfault_process",
            "shutdown_service",
            "startup_service",
        ):
            record = parse(*CASES[case], remaining, preprocessing, regs)
            record.update(
                stage="prompt",
                case=case,
                removed_index=index + 1,
                removed_rule=removed_rule,
            )
            records.append(record)
            print(
                f"without prompt {index + 1:02d} {case:18s}: "
                f"{'CORRECT' if record['correct'] else 'wrong'} "
                f"{record['final']!r}",
                flush=True,
            )

    output_dir = (
        ROOT
        / "others/test_one_log/results/domain_rule_ablation"
        / f"thunderbird_individual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
