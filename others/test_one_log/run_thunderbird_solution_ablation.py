#!/usr/bin/env python3
"""Remove each candidate component to verify that it is necessary."""

from run_thunderbird_rule_ablation import CASES
from run_thunderbird_solution_validation import (
    KEY_VALUE_EXCEPT_OSD_OP,
    REGEX_RULES,
)
from run_thunderbird_queue_rule_validation import BROAD_QUEUE_RULE
from run_webmaster_rule_ablation import common_regexes, gq, parse


RULES = [BROAD_QUEUE_RULE, *REGEX_RULES[1:]]
TARGETS = {
    "sendmail_queue_id": ("mail_queue_id",),
    "compound_pan_runtime_id": ("pan_capability_id", "pan_failure_id"),
    "source_path_with_bracketed_line": ("source_location",),
    "hex_before_adjacent_key": ("source_location",),
    "segfault_process_name": ("segfault_process",),
    "lifecycle_subject": ("shutdown_service", "startup_service"),
    "key_value_except_osd_op": ("pan_failure_id",),
}


def preprocessing_without(removed_name):
    result = [rule for rule in RULES if rule.name != removed_name]
    for rule in gq.PREPROCESSING_RULE_GROUPS["curated"]:
        if rule.name == "curated_0031_key_value":
            result.append(
                rule
                if removed_name == "key_value_except_osd_op"
                else KEY_VALUE_EXCEPT_OSD_OP
            )
        else:
            result.append(rule)
    return result


def main():
    regs = common_regexes()
    for removed_name, cases in TARGETS.items():
        preprocessing = preprocessing_without(removed_name)
        for case in cases:
            for repetition in range(1, 6):
                record = parse(
                    *CASES[case],
                    gq.CURATED_PROMPT_RULES,
                    preprocessing,
                    regs,
                )
                print(
                    f"without {removed_name:34s} {case:20s} "
                    f"{repetition}/5: "
                    f"{'CORRECT' if record['correct'] else 'wrong'} "
                    f"{record['final']!r}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
