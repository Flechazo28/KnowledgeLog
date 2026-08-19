#!/usr/bin/env python3
"""Run focused one-rule ablations on representative Linux errors."""

import json
from datetime import datetime

from run_linux_rule_ablation import CASES
from run_webmaster_rule_ablation import ROOT, common_regexes, gq, parse


PREPROCESSING_TESTS = {
    "curated_0002_ipv4": ("network_interface",),
    "curated_0008_hexadecimal": (
        "kernel_stack_address",
        "kernel_stack_module",
    ),
    "curated_0031_key_value": (
        "session_uid_wrapper",
        "selinux_avc_fields",
    ),
    "domain_or_ip_with_required_port": ("selinux_avc_fields",),
    "lifecycle_subject": ("lifecycle_subject",),
}

PROMPT_TESTS = {
    6: (
        "fixed_parenthesized_service",
        "variable_component",
        "driver_identifier",
    ),
    13: ("network_interface", "driver_identifier"),
    14: ("numeric_range",),
    16: ("session_uid_wrapper", "selinux_avc_fields"),
    22: (
        "session_uid_wrapper",
        "quoted_device_path",
        "bracketed_exit_code",
        "fixed_parenthesized_service",
    ),
}


def main():
    prompt_rules = list(gq.CURATED_PROMPT_RULES)
    preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    regs = common_regexes()
    records = []

    for removed_name, cases in PREPROCESSING_TESTS.items():
        remaining = [
            rule for rule in preprocessing if rule.name != removed_name
        ]
        for case in cases:
            record = parse(*CASES[case], prompt_rules, remaining, regs)
            record.update(
                stage="preprocessing",
                removed_rule=removed_name,
                case=case,
            )
            records.append(record)
            print(
                f"without pre {removed_name:34s} {case:28s}: "
                f"{'CORRECT' if record['correct'] else 'wrong'} "
                f"{record['final']!r}",
                flush=True,
            )

    for one_based_index, cases in PROMPT_TESTS.items():
        removed_rule = prompt_rules[one_based_index - 1]
        remaining = [
            rule
            for index, rule in enumerate(prompt_rules, start=1)
            if index != one_based_index
        ]
        for case in cases:
            record = parse(*CASES[case], remaining, preprocessing, regs)
            record.update(
                stage="prompt",
                removed_index=one_based_index,
                removed_rule=removed_rule,
                case=case,
            )
            records.append(record)
            print(
                f"without prompt {one_based_index:02d} {case:28s}: "
                f"{'CORRECT' if record['correct'] else 'wrong'} "
                f"{record['final']!r}",
                flush=True,
            )

    output_dir = (
        ROOT
        / "others/test_one_log/results/domain_rule_ablation"
        / f"linux_individual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
