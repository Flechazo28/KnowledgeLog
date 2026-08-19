#!/usr/bin/env python3
"""Validate a broader but format-specific Thunderbird queue-ID rule."""

from run_thunderbird_solution_validation import (
    KEY_VALUE_EXCEPT_OSD_OP,
    REGEX_RULES,
)
from run_webmaster_rule_ablation import common_regexes, gq, parse


BROAD_QUEUE_RULE = gq.RegexRule(
    name="sendmail_queue_id",
    pattern=r"^[A-Za-z0-9]{14}(?=:)",
    replacement=gq.VARIABLE_PLACEHOLDER,
    note="Abstract a 14-character sendmail queue identifier before a colon.",
)

CASES = {
    "mail": (
        "jA9IBUZj019524: from=root, size=629060, class=0, nrcpts=1, "
        "msgid=<200511091811.jA9IBUZj019524@eadmin1>, relay=#7#@localhost",
        "<*>: from=<*>, size=<*>, class=<*>, nrcpts=<*>, msgid=<*>, relay=<*>",
    ),
    "fifo": (
        "jAAB2403021621: 1: fl=0x1, mode=10600: FIFO: dev=0/7, "
        "ino=5719211, nlink=1, u/gid=0/0, size=0",
        "<*>: <*>: fl=<*>, mode=<*>: FIFO: dev=<*>, ino=<*>, "
        "nlink=<*>, u/gid=<*>, size=<*>",
    ),
    "character_device": (
        "jAOB31sV009860: 1: fl=0x8002, mode=20666: CHR: dev=0/13, "
        "ino=1259, nlink=1, u/gid=0/0, size=0",
        "<*>: <*>: fl=<*>, mode=<*>: CHR: dev=<*>, ino=<*>, "
        "nlink=<*>, u/gid=<*>, size=<*>",
    ),
    "queue_error": (
        "jAOB31sV009860: SYSERR(root): queueup: cannot create queue file "
        "./qfjAOB31sV009860, euid=51, fd=-1, fp=0x0: Read-only file system",
        "<*>: SYSERR(<*>): queueup: cannot create queue file <*>, "
        "euid=<*>, fd=<*>, fp=<*>: Read-only file system",
    ),
    "low_space": (
        "jAAB2403021621: low on space (SMTP-DAEMON needs 3447 bytes + "
        "100 blocks in /var/spool/clientmqueue), max avail: 0",
        "<*>: low on space (SMTP-DAEMON needs <*> bytes + <*> blocks in <*>), "
        "max avail: <*>",
    ),
    "guard": (
        "tftp: client does not accept options",
        "tftp: client does not accept options",
    ),
}


def main():
    preprocessing = [BROAD_QUEUE_RULE, *REGEX_RULES[1:]]
    for rule in gq.PREPROCESSING_RULE_GROUPS["curated"]:
        preprocessing.append(
            KEY_VALUE_EXCEPT_OSD_OP
            if rule.name == "curated_0031_key_value"
            else rule
        )
    regs = common_regexes()
    for case, (log, expected) in CASES.items():
        for repetition in range(1, 6):
            record = parse(
                log,
                expected,
                gq.CURATED_PROMPT_RULES,
                preprocessing,
                regs,
            )
            print(
                f"{case:20s} {repetition}/5: "
                f"{'CORRECT' if record['correct'] else 'wrong'} "
                f"{record['final']!r}",
                flush=True,
            )


if __name__ == "__main__":
    main()
