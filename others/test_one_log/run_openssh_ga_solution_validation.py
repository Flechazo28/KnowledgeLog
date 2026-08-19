#!/usr/bin/env python3
"""Validate the two complementary OpenSSH account preprocessing rules."""

import re

from run_webmaster_rule_ablation import (
    REGRESSION_CASES,
    TARGET,
    common_regexes,
    gq,
    parse,
)


INVALID_USER_ACCOUNT = gq.RegexRule(
    name="invalid_user_account",
    pattern=(
        r"(\binvalid\s+user\s+)"
        r"(?!(?:unknown|from)\b)"
        r"[A-Za-z_][A-Za-z0-9._-]*\b"
    ),
    replacement=r"\1{variables}",
    flags=re.IGNORECASE,
    note="Abstract a concrete account name after invalid user.",
)
MISSING_ACCOUNT_FIELD = gq.RegexRule(
    name="failed_password_missing_account",
    pattern=r"(\bFailed\s+password\s+for\s+)invalid\s+user(?=\s+from\b)",
    replacement=r"\1{variables}",
    flags=re.IGNORECASE,
    note="Abstract the missing-account field before the source marker from.",
)

CASES = {
    "webmaster": TARGET,
    "missing_1": (
        "Failed password for invalid user from 103.204.177.144 port 2223 ssh2",
        "Failed password for <*> from <*> port <*> ssh2",
    ),
    "missing_2": (
        "Failed password for invalid user from 159.203.87.8 port 60796 ssh2",
        "Failed password for <*> from <*> port <*> ssh2",
    ),
    "missing_3": (
        "Failed password for invalid user from 159.203.87.8 port 37920 ssh2",
        "Failed password for <*> from <*> port <*> ssh2",
    ),
    "missing_4": (
        "Failed password for invalid user from 188.163.99.50 port 36266 ssh2",
        "Failed password for <*> from <*> port <*> ssh2",
    ),
    "invalid_user_event": (
        "Invalid user webmaster from 173.234.31.186",
        "Invalid user <*> from <*>",
    ),
    **REGRESSION_CASES,
}


def main():
    prompt_rules = list(gq.CURATED_PROMPT_RULES)
    preprocessing = [
        MISSING_ACCOUNT_FIELD,
        INVALID_USER_ACCOUNT,
        *gq.PREPROCESSING_RULE_GROUPS["curated"],
    ]
    regs = common_regexes()
    correct = 0
    total = 0
    for case, (log, expected) in CASES.items():
        for repetition in range(1, 6):
            record = parse(log, expected, prompt_rules, preprocessing, regs)
            correct += int(record["correct"])
            total += 1
            print(
                f"{case:20s} {repetition}/5: "
                f"{'CORRECT' if record['correct'] else 'wrong'} "
                f"{record['final']!r}",
                flush=True,
            )
    print(f"Total: {correct}/{total}")


if __name__ == "__main__":
    main()
