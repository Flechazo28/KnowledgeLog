#!/usr/bin/env python3
"""Validate concise Thunderbird template-explosion solutions."""

import json
import re
from collections import Counter
from datetime import datetime

from run_thunderbird_rule_ablation import CASES
from run_webmaster_rule_ablation import ROOT, common_regexes, gq, parse


PROMPT_RULES = [
    (
        "Abstract a leading mail message or queue identifier as one {variables} "
        "when it appears before a colon followed by key-value fields."
    ),
    (
        "Abstract a complete compound runtime object or transaction identifier "
        "as one {variables}; do not preserve or split its internal letter-digit "
        "segments joined by '-', '_', '@', '::', or '.'."
    ),
    (
        "In a process-crash prefix of the form name[pid]: segfault, abstract the "
        "process name and PID separately as {variables}; preserve the brackets."
    ),
    (
        "In a lifecycle result of the form name startup/shutdown "
        "succeeded/failed, abstract the complete leading process or service name "
        "as one {variables} and preserve the lifecycle words."
    ),
    (
        "For a source location written as path/file[line]:, abstract the complete "
        "path and file as one {variables} and the line number as another "
        "{variables}; preserve the brackets and colon."
    ),
]

REGEX_RULES = [
    gq.RegexRule(
        name="leading_message_or_queue_id",
        pattern=(
            r"^[A-Za-z0-9]{10,}"
            r"(?=:\s+[A-Za-z_][A-Za-z0-9_.-]*\s*=)"
        ),
        replacement=gq.VARIABLE_PLACEHOLDER,
        note="Abstract a leading message or queue ID before key-value fields.",
    ),
    gq.RegexRule(
        name="compound_pan_runtime_id",
        pattern=(
            r"(?<![A-Za-z0-9_])"
            r"I-xD[0-9A-Fa-f]+-xG[0-9A-Fa-f]+-xU[0-9A-Fa-f]+"
            r"(?![A-Za-z0-9_])"
        ),
        replacement=gq.VARIABLE_PLACEHOLDER,
        note="Abstract a complete compound storage runtime identifier.",
    ),
    gq.RegexRule(
        name="source_path_with_bracketed_line",
        pattern=(
            r"(?<!\S)"
            r"(?:/?[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"
            r"\[\d+\](?=:)"
        ),
        replacement=f"{gq.VARIABLE_PLACEHOLDER}[{gq.VARIABLE_PLACEHOLDER}]",
        note="Abstract a source path and its bracketed line number separately.",
    ),
    gq.RegexRule(
        name="hex_before_adjacent_key",
        pattern=(
            r"(?<![A-Za-z0-9/])0[xX][0-9A-Fa-f]+"
            r"(?=[A-Za-z_][A-Za-z0-9_]*=)"
        ),
        replacement=gq.VARIABLE_PLACEHOLDER,
        note="Abstract a hexadecimal value immediately followed by another key.",
    ),
    gq.RegexRule(
        name="segfault_process_name",
        pattern=(
            r"^[A-Za-z_][A-Za-z0-9_.-]*"
            r"(?=\[\d+\]:\s+segfault\b)"
        ),
        replacement=gq.VARIABLE_PLACEHOLDER,
        flags=re.IGNORECASE,
        note="Abstract the process name in a name[pid]: segfault prefix.",
    ),
    gq.RegexRule(
        name="lifecycle_subject",
        pattern=(
            r"^[A-Za-z_][A-Za-z0-9_.-]*"
            r"(?=\s+(?:startup|shutdown)\s+(?:succeeded|failed)\b)"
        ),
        replacement=gq.VARIABLE_PLACEHOLDER,
        flags=re.IGNORECASE,
        note="Abstract the process or service name in a lifecycle result.",
    ),
]

KEY_VALUE_EXCEPT_OSD_OP = gq.RegexRule(
    name="key_value_except_osd_op",
    pattern=(
        r"(?<![\w.-])"
        r"((?!osd_op\s*=)[A-Za-z_][A-Za-z0-9_.-]*\s*=\s*?)"
        r"(?:"
        r"[A-Za-z0-9._]+(?=\s|[,;]|$)"
        r"|"
        r"(?=\s*(?:[A-Za-z_][A-Za-z0-9_.-]*\s*=|[,;]|$))"
        r")"
    ),
    replacement=r"\1{variables}",
    note="Apply generic key-value abstraction except to osd_op.",
)

GUARD_CASES = {
    "fixed_colon_prefix": (
        "tftp: client does not accept options",
        "tftp: client does not accept options",
    ),
    "fixed_shutdown_phrase": (
        "Shutting down GANGLIA gmetad:",
        "Shutting down GANGLIA gmetad:",
    ),
}


def candidate_preprocessing():
    existing = []
    for rule in gq.PREPROCESSING_RULE_GROUPS["curated"]:
        if rule.name == "curated_0031_key_value":
            existing.append(KEY_VALUE_EXCEPT_OSD_OP)
        else:
            existing.append(rule)
    return [*REGEX_RULES, *existing]


def main():
    current_prompt = list(gq.CURATED_PROMPT_RULES)
    current_preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    prompt_candidate = [*PROMPT_RULES, *current_prompt]
    preprocessing_candidate = candidate_preprocessing()
    variants = {
        "prompt_solution": (prompt_candidate, current_preprocessing),
        "regex_solution": (current_prompt, preprocessing_candidate),
        "combined_solution": (prompt_candidate, preprocessing_candidate),
    }
    cases = {**CASES, **GUARD_CASES}
    regs = common_regexes()
    records = []
    for variant, (prompt_rules, preprocessing_rules) in variants.items():
        for case, (log, expected) in cases.items():
            for repetition in range(1, 6):
                record = parse(
                    log, expected, prompt_rules, preprocessing_rules, regs
                )
                record.update(
                    variant=variant,
                    case=case,
                    repetition=repetition,
                )
                records.append(record)
                print(
                    f"{variant:18s} {case:22s} {repetition}/5: "
                    f"{'CORRECT' if record['correct'] else 'wrong'} "
                    f"{record['final']!r}",
                    flush=True,
                )

    output_dir = (
        ROOT
        / "others/test_one_log/results/domain_rule_ablation"
        / f"thunderbird_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
