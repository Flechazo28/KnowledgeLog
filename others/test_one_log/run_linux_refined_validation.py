#!/usr/bin/env python3
"""Validate the refined Linux candidate against its former regressions."""

import json
from collections import Counter
from datetime import datetime

from run_linux_solution_validation import CANDIDATE_REGEXES
from run_webmaster_rule_ablation import ROOT, common_regexes, gq, parse


REPETITIONS = 5
CASES = {
    "plain_uid_actor": (
        "session opened for user htt by (uid=0)",
        "session opened for user <*> by <*>",
    ),
    "login_uid_actor": (
        "session opened for user root by LOGIN(uid=0)",
        "session opened for user <*> by <*>",
    ),
    "named_uid_actor": (
        "session opened for user test by root(uid=0)",
        "session opened for user <*> by <*>",
    ),
    "preserve_errno_wrapper": (
        "bind failed (Address already in use (errno = 98)). service = telnet",
        "bind failed (Address already in use (errno = <*>)). service = <*>",
    ),
    "preserve_current_wrapper": (
        "microcode: CPU0 already at revision 0x8 (current=0x8)",
        "microcode: CPU0 already at revision <*> (current=<*>)",
    ),
}


def main():
    prompt_rules = list(gq.CURATED_PROMPT_RULES)
    preprocessing = (
        list(CANDIDATE_REGEXES)
        + list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    )
    regs = common_regexes()
    records = []
    for case, (log, expected) in CASES.items():
        for repetition in range(1, REPETITIONS + 1):
            record = parse(
                log,
                expected,
                prompt_rules,
                preprocessing,
                regs,
            )
            record.update(case=case, repetition=repetition)
            records.append(record)
            print(
                f"{case:28s} {repetition}/{REPETITIONS}: "
                f"{'CORRECT' if record['correct'] else 'wrong'} "
                f"{record['final']!r}",
                flush=True,
            )

    output_dir = (
        ROOT
        / "others/test_one_log/results/domain_rule_ablation"
        / f"linux_refined_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {}
    for case in CASES:
        selected = [record for record in records if record["case"] == case]
        summary[case] = {
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
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
