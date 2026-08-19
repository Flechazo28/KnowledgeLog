#!/usr/bin/env python3
"""Validate the refined, minimal Thunderbird candidate set."""

import json
import re
from collections import Counter
from datetime import datetime

from run_thunderbird_major_error_validation import CASES, REGEX_RULES
from run_webmaster_rule_ablation import ROOT, common_regexes, gq, parse


REPETITIONS = 3
CURRENT_STATE_RULE = (
    "Abstract a state, lifecycle stage, or boolean only when it occupies a "
    "variable state or status field; preserve fixed action and event wording."
)
REFINED_STATE_RULE = (
    "Abstract a state value when it is an explicit field value or an endpoint "
    "of a transition; preserve a named state when it is fixed event wording in "
    "a predicate or explanatory phrase."
)
FIXED_STATUS_LABEL_RULE = (
    "Preserve a parenthesized explanatory status label when it is fixed event "
    "wording after a variable state value; do not delete the label or its "
    "parentheses."
)

REFINED_RULES = [
    rule for rule in REGEX_RULES if rule.name != "parenthesized_execution_identity"
]
REFINED_RULES.extend(
    [
        gq.RegexRule(
            name="parenthesized_actor_and_action_payload",
            pattern=(
                r"^\([^()\s]+\)"
                r"(\s+[A-Z][A-Z0-9_-]*\s+)"
                r"\(.*\)$"
            ),
            replacement=(
                rf"({gq.VARIABLE_PLACEHOLDER})"
                rf"\1({gq.VARIABLE_PLACEHOLDER})"
            ),
            note=(
                "Abstract a parenthesized execution actor and the complete "
                "parenthesized payload of its action."
            ),
        ),
        gq.RegexRule(
            name="runtime_name_before_port",
            pattern=(
                r"(\bfor\s+)"
                r"[A-Za-z][A-Za-z0-9_.-]*"
                r"(?=\s+port\s+\d+\b)"
            ),
            replacement=r"\1{variables}",
            note="Abstract a runtime endpoint or device name before its port field.",
        ),
        gq.RegexRule(
            name="hex_before_adjacent_key",
            pattern=(
                r"(?<![A-Za-z0-9/])"
                r"0[xX][0-9A-Fa-f]+"
                r"(?=[A-Za-z_][A-Za-z0-9_]*=)"
            ),
            replacement=gq.VARIABLE_PLACEHOLDER,
            note="Abstract a hexadecimal value immediately followed by another key.",
        ),
        gq.RegexRule(
            name="pci_address",
            pattern=(
                r"(?<![0-9A-Fa-f:])"
                r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:"
                r"[0-9A-Fa-f]{2}\.[0-7]"
                r"(?:\[[A-Za-z]\])?"
                r"(?![0-9A-Fa-f])"
            ),
            replacement=gq.VARIABLE_PLACEHOLDER,
            note="Abstract a complete PCI bus address, including its optional pin.",
        ),
    ]
)

REFINED_CASES = {
    **CASES,
    "transition_state_guard": (
        "attempt_1445087491445_0005_m_000000_0 "
        "TaskAttempt Transitioned from NEW to UNASSIGNED",
        "<*> TaskAttempt Transitioned from <*> to <*>",
    ),
}


def main():
    prompt_rules = list(gq.CURATED_PROMPT_RULES)
    prompt_rules[prompt_rules.index(CURRENT_STATE_RULE)] = REFINED_STATE_RULE
    prompt_rules.insert(0, FIXED_STATUS_LABEL_RULE)
    preprocessing_rules = [
        *REFINED_RULES,
        *gq.PREPROCESSING_RULE_GROUPS["curated"],
    ]
    regs = common_regexes()
    records = []

    for case, (log_message, expected) in REFINED_CASES.items():
        for repetition in range(1, REPETITIONS + 1):
            record = parse(
                log_message,
                expected,
                prompt_rules,
                preprocessing_rules,
                regs,
            )
            record.update(case=case, repetition=repetition)
            records.append(record)
            print(
                f"{case:30s} {repetition}/{REPETITIONS}: "
                f"{'CORRECT' if record['correct'] else 'wrong'} "
                f"{record['final']!r}",
                flush=True,
            )

    output_dir = (
        ROOT
        / "others/test_one_log/results/regex_validation"
        / f"thunderbird_refined_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {}
    for case in REFINED_CASES:
        selected = [record for record in records if record["case"] == case]
        summary[case] = {
            "correct": sum(record["correct"] for record in selected),
            "runs": len(selected),
            "templates": dict(Counter(record["final"] for record in selected)),
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
