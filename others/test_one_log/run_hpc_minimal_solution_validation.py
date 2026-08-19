#!/usr/bin/env python3
"""Validate small, general prompt and preprocessing changes for HPC errors."""

import json
import re
from collections import Counter
from datetime import datetime

from run_webmaster_rule_ablation import ROOT, common_regexes, gq, parse


REPETITIONS = 5

CURRENT_STATE_RULE = (
    "Abstract a state, lifecycle stage, or boolean only when it occupies a "
    "variable state or status field; preserve fixed action and event wording."
)
CANDIDATE_STATE_RULE = (
    "Preserve state words inside grammatical event descriptions of the form "
    "'is in the ... state'; abstract a state only when an explicit state or "
    "status field presents it as a value."
)

STANDALONE_KEY_VALUE_RECORD = gq.RegexRule(
    name="standalone_key_value_record",
    pattern=(
        r"^([A-Za-z_][A-Za-z0-9_.-]*\s*=\s*)"
        r"(?!.*[,;])"
        r"(?!.*\b[A-Za-z_][A-Za-z0-9_.-]*\s*=)"
        r"\S+(?:\s+\S+){2,}$"
    ),
    replacement=r"\1{variables}",
    note=(
        "Abstract a multi-token value that occupies the complete right-hand "
        "side of a standalone single key-value record."
    ),
)

PARENTHESIZED_COMPACT_QUANTITY = gq.RegexRule(
    name="parenthesized_compact_quantity",
    pattern=(
        r"\(\s*[-+]?\d+(?:\.\d+)?"
        r"(?:KiB|Kbytes|GHz|KB|kB|MB|GB|TB|ms|B|K|M|G|C)\s*\)"
    ),
    replacement=gq.VARIABLE_PLACEHOLDER,
    note="Abstract a complete parenthesized compact numeric quantity.",
)

OCTAL_ESCAPED_QUOTED_VALUE = gq.RegexRule(
    name="octal_escaped_quoted_value",
    pattern=r"\\042.*?\\042",
    replacement=gq.VARIABLE_PLACEHOLDER,
    note="Abstract a complete value enclosed by octal-escaped double quotes.",
)

CASES = {
    "component_state": (
        r"Component State Change: Component \042alt0\042 is in the "
        "available state (HWID=1328)",
        "Component State Change: Component <*> is in the available state (HWID=<*>)",
    ),
    "component_state_symbolic_id": (
        r"Component State Change: Component "
        r"\042SCSI-WWID:01000010:6005-08b4-0001-00c6-0006-3000-002b-0000\042 "
        "is in the available state (HWID=1971)",
        "Component State Change: Component <*> is in the available state (HWID=<*>)",
    ),
    "temperature": (
        "Temperature (41C) exceeds warning threshold",
        "Temperature <*> exceeds warning threshold",
    ),
    "standalone_key_value_with_phrase": (
        "ambient=40 threshold exceeded",
        "ambient=<*>",
    ),
    "standalone_key_value_simple": (
        "ambient=29",
        "ambient=<*>",
    ),
    "explicit_state_field": (
        "state=available",
        "state=<*>",
    ),
    "multiple_key_values": (
        "uid=0 euid=0",
        "uid=<*> euid=<*>",
    ),
}


def candidate_prompt_rules():
    rules = list(gq.CURATED_PROMPT_RULES)
    rules[rules.index(CURRENT_STATE_RULE)] = CANDIDATE_STATE_RULE
    return rules


def candidate_preprocessing():
    rules = [STANDALONE_KEY_VALUE_RECORD, OCTAL_ESCAPED_QUOTED_VALUE]
    for rule in gq.PREPROCESSING_RULE_GROUPS["curated"]:
        if rule.name == "compact_quantity_value":
            rules.append(PARENTHESIZED_COMPACT_QUANTITY)
        rules.append(rule)
    return rules


def main():
    variants = {
        "current": (
            list(gq.CURATED_PROMPT_RULES),
            list(gq.PREPROCESSING_RULE_GROUPS["curated"]),
        ),
        "candidate": (
            candidate_prompt_rules(),
            candidate_preprocessing(),
        ),
    }
    regs = common_regexes()
    records = []

    for variant, (prompt_rules, preprocessing) in variants.items():
        for case, (log_message, expected) in CASES.items():
            for repetition in range(1, REPETITIONS + 1):
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
                    f"{variant:9s} {case:32s} {repetition}/{REPETITIONS}: "
                    f"{'CORRECT' if record['correct'] else 'wrong'} "
                    f"{record['final']!r}",
                    flush=True,
                )

    output_dir = (
        ROOT
        / "others/test_one_log/results/regex_validation"
        / f"hpc_minimal_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
