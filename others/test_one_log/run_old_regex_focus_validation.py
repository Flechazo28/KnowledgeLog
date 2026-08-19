#!/usr/bin/env python3
"""Validate old regexes on errors from the latest 2k experiment."""

import json
import re
from collections import Counter
from datetime import datetime

from run_oldgptquery_comparison import candidate_rules
from run_webmaster_rule_ablation import ROOT, common_regexes, gq, parse


REPETITIONS = 2
CASES = {
    "dotted_dcbf": (
        "dotted_counter",
        "data store interrupt caused by dcbf.........0",
        "data store interrupt caused by dcbf.........<*>",
    ),
    "compact_cpu_cache": (
        "compact_quantity",
        "CPU: L1 I cache: 16K, L1 D cache: 16K",
        "CPU: L1 I cache: <*>, L1 D cache: <*>",
    ),
    "compact_agp": (
        "compact_quantity",
        "agpgart: AGP aperture is 64M @ 0xf4000000",
        "agpgart: AGP aperture is <*> @ <*>",
    ),
    "compact_hda": (
        "compact_quantity",
        "hda: 29336832 sectors (15020 MB) w/1916KiB Cache, "
        "CHS=29104/16/63, UDMA(66)",
        "hda: <*> sectors (<*> MB) w/<*> Cache, CHS=<*>, UDMA(<*>)",
    ),
    "empty_role_thunderbird": (
        "empty_role_uid",
        "session opened for user root by (uid=0)",
        "session opened for user <*> by <*>(uid=<*>)",
    ),
    "empty_role_linux": (
        "empty_role_uid",
        "session opened for user cyrus by (uid=0)",
        "session opened for user <*> by <*>",
    ),
    "parenthesized_root_cron": (
        "parenthesized_root",
        "(root) CMD (run-parts /etc/cron.hourly)",
        "(<*>) CMD (<*>)",
    ),
    "parenthesized_root_syserr": (
        "parenthesized_root",
        "jBHB4Mg7002713: SYSERR(root): returntosender: cannot select queue "
        "for postmaster",
        "<*>: SYSERR(<*>): returntosender: cannot select queue for postmaster",
    ),
}


def main():
    candidates = candidate_rules()
    candidates["parenthesized_root"] = gq.RegexRule(
        "old_parenthesized_root",
        r"\(\s*root\s*\)",
        "({variables})",
        re.IGNORECASE,
    )
    prompt = list(gq.CURATED_PROMPT_RULES)
    current_pre = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    regs = common_regexes()
    records = []
    for case, (candidate, log, expected) in CASES.items():
        variants = {
            "current": current_pre,
            f"add_old_{candidate}": [candidates[candidate]] + current_pre,
        }
        for variant, preprocessing in variants.items():
            for repetition in range(1, REPETITIONS + 1):
                result = parse(log, expected, prompt, preprocessing, regs)
                result.update(
                    case=case,
                    variant=variant,
                    repetition=repetition,
                )
                records.append(result)
                print(
                    f"{case:28s} {variant:30s} "
                    f"{repetition}/{REPETITIONS}: "
                    f"{'CORRECT' if result['correct'] else 'wrong'} "
                    f"{result['final']!r}",
                    flush=True,
                )

    output_dir = (
        ROOT
        / "others/test_one_log/results/domain_rule_ablation"
        / f"old_regex_focus_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {}
    for case in CASES:
        summary[case] = {}
        for variant in dict.fromkeys(
            item["variant"] for item in records if item["case"] == case
        ):
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
