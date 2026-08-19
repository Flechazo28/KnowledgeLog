#!/usr/bin/env python3
"""Check whether the broad fixed-name rule conflicts with account parsing."""

from run_webmaster_rule_ablation import (
    ROOT,
    TARGET,
    common_regexes,
    gq,
    parse,
)
from run_webmaster_solution_validation import (
    ACCOUNT_RULE,
    PRESERVE_UNKNOWN_RULE,
    SYNTHETIC_CASES,
)


REVISED_FIXED_NAME_RULE = (
    "Preserve a package, class, module, method, file, operating-system, driver, "
    "protocol, service, component, task, process, or event-handler name when it "
    "is fixed event wording. This rule does not apply when an authentication "
    "message explicitly identifies the token as an account name."
)


def main():
    rules = list(gq.CURATED_PROMPT_RULES)
    preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    base = [
        PRESERVE_UNKNOWN_RULE,
        ACCOUNT_RULE,
        *[rule for i, rule in enumerate(rules) if i != 17],
    ]
    without_fixed_name = [
        PRESERVE_UNKNOWN_RULE,
        ACCOUNT_RULE,
        *[rule for i, rule in enumerate(rules) if i not in {5, 17}],
    ]
    revised_rules = list(rules)
    revised_rules[5] = REVISED_FIXED_NAME_RULE
    revised_fixed_name = [
        PRESERVE_UNKNOWN_RULE,
        ACCOUNT_RULE,
        *[rule for i, rule in enumerate(revised_rules) if i != 17],
    ]
    variants = {
        "prompt_solution": base,
        "without_fixed_name_rule": without_fixed_name,
        "revised_fixed_name_rule": revised_fixed_name,
    }
    cases = {"target": TARGET, **SYNTHETIC_CASES}
    regs = common_regexes()
    for variant, variant_rules in variants.items():
        for case, (log, expected) in cases.items():
            for repetition in range(1, 6):
                record = parse(
                    log, expected, variant_rules, preprocessing, regs
                )
                print(
                    f"{variant:26s} {case:18s} {repetition}/5: "
                    f"{'CORRECT' if record['correct'] else 'wrong'} "
                    f"{record['final']!r}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
