#!/usr/bin/env python3
"""Validate compact candidate fixes for high-impact Linux template errors."""

import json
import re
from collections import Counter
from datetime import datetime

from run_linux_rule_ablation import CASES
from run_webmaster_rule_ablation import ROOT, common_regexes, gq, parse


REPETITIONS = 5

PARENTHESIZED_SERVICE_RULE = (
    "Preserve a parenthesized service or subsystem name when it identifies the "
    "fixed source of an event; do not abstract the name merely because it is "
    "enclosed in parentheses."
)
VERSIONED_COMPONENT_RULE = (
    "In a versioned component field written as 'NAME ver VERSION', abstract "
    "both NAME and VERSION as {variables} when NAME is the varying component "
    "of the event; preserve the word 'ver'."
)
DRIVER_ATTRIBUTION_RULE = (
    "In a driver attribution message, abstract the driver identifier before "
    "the first colon as {variables}; preserve the attribution text that follows."
)

CANDIDATE_REGEXES = [
    gq.RegexRule(
        name="linux_listening_interface",
        pattern=(
            r"(\blistening on IPv[46] interface\s+)"
            r"[A-Za-z][A-Za-z0-9_.:-]*(?=,\s)"
        ),
        replacement=r"\1{variables}",
        flags=re.IGNORECASE,
        note="Abstract an interface name in an explicit listening-interface field.",
    ),
    gq.RegexRule(
        name="bluetooth_versioned_component",
        pattern=(
            r"(\bBluetooth:\s+)"
            r"[A-Za-z][A-Za-z0-9_.-]*(?=\s+ver\s+)"
        ),
        replacement=r"\1{variables}",
        flags=re.IGNORECASE,
        note="Abstract the varying component name in a Bluetooth version field.",
    ),
    gq.RegexRule(
        name="quoted_filesystem_path",
        pattern=r"(?P<quote>['\"])/[^'\"\s]+(?P=quote)",
        replacement=gq.VARIABLE_PLACEHOLDER,
        note="Abstract a complete quoted filesystem path, including its quotes.",
    ),
    gq.RegexRule(
        name="session_actor_uid",
        pattern=(
            r"(?<=\bby\s)"
            r"(?:[A-Za-z_][A-Za-z0-9_.-]*)?"
            r"\(\s*uid\s*=\s*[^()\s]+\s*\)"
        ),
        replacement=gq.VARIABLE_PLACEHOLDER,
        flags=re.IGNORECASE,
        note="Abstract a complete session actor expressed with a uid field.",
    ),
    gq.RegexRule(
        name="bracketed_numeric_field",
        pattern=r"\[\s*[+-]?\d+\s*\]",
        replacement=gq.VARIABLE_PLACEHOLDER,
        note="Abstract a complete bracketed numeric field.",
    ),
    gq.RegexRule(
        name="linux_kernel_stack_frame",
        pattern=(
            r"^\[<[0-9A-Fa-f]+>\]\s+"
            r"([A-Za-z_][A-Za-z0-9_.]*)"
            r"\+0[xX][0-9A-Fa-f]+/0[xX][0-9A-Fa-f]+"
            r"(?:\s+\[[A-Za-z0-9_.-]+\])?$"
        ),
        replacement=r"[{variables}] \1+{variables}",
        note="Normalize the address and offset fields of a Linux kernel stack frame.",
    ),
    gq.RegexRule(
        name="selinux_denied_permissions",
        pattern=r"(?<=\bavc:\sdenied\s)\{[^}]+\}",
        replacement=gq.VARIABLE_PLACEHOLDER,
        flags=re.IGNORECASE,
        note="Abstract the permission set in a SELinux AVC denial.",
    ),
    gq.RegexRule(
        name="selinux_path_or_context",
        pattern=r"(\b(?:exe|scontext|tcontext)=)\S+",
        replacement=r"\1{variables}",
        flags=re.IGNORECASE,
        note="Abstract SELinux executable paths and security contexts.",
    ),
]

GUARDS = {
    "fixed_bracketed_module": (
        "EXT3-fs: mounted filesystem with ordered data mode [ext3]",
        "EXT3-fs: mounted filesystem with ordered data mode [ext3]",
    ),
    "fixed_parenthesized_text": (
        "getpeername (ftpd): Transport endpoint is not connected",
        "getpeername (ftpd): Transport endpoint is not connected",
    ),
    "fixed_driver_message": (
        "imps2: Auto-detected intellimouse PS/2",
        "imps2: Auto-detected intellimouse PS/2",
    ),
}


def main():
    prompt_rules = list(gq.CURATED_PROMPT_RULES)
    preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    natural_candidates = [
        PARENTHESIZED_SERVICE_RULE,
        VERSIONED_COMPONENT_RULE,
        DRIVER_ATTRIBUTION_RULE,
    ]
    variants = {
        "regex_solution": (
            prompt_rules,
            CANDIDATE_REGEXES + preprocessing,
        ),
        "natural_solution": (
            natural_candidates + prompt_rules,
            preprocessing,
        ),
        "combined_solution": (
            natural_candidates + prompt_rules,
            CANDIDATE_REGEXES + preprocessing,
        ),
    }
    selected_cases = {
        **CASES,
        **GUARDS,
    }
    regs = common_regexes()
    records = []
    for variant, (rules, pre_rules) in variants.items():
        for case, (log, expected) in selected_cases.items():
            for repetition in range(1, REPETITIONS + 1):
                record = parse(log, expected, rules, pre_rules, regs)
                record.update(
                    variant=variant,
                    case=case,
                    repetition=repetition,
                )
                records.append(record)
                print(
                    f"{variant:20s} {case:28s} "
                    f"{repetition}/{REPETITIONS}: "
                    f"{'CORRECT' if record['correct'] else 'wrong'} "
                    f"{record['final']!r}",
                    flush=True,
                )

    output_dir = (
        ROOT
        / "others/test_one_log/results/domain_rule_ablation"
        / f"linux_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {}
    for variant in variants:
        summary[variant] = {}
        for case in selected_cases:
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
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
