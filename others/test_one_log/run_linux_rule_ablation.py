#!/usr/bin/env python3
"""Run grouped domain-knowledge ablations on high-impact Linux errors."""

import json
from collections import Counter
from datetime import datetime

from run_webmaster_rule_ablation import ROOT, common_regexes, gq, parse


CASES = {
    "session_uid_wrapper": (
        "session opened for user htt by (uid=0)",
        "session opened for user <*> by <*>",
    ),
    "quoted_device_path": (
        "creating device node '/udev/lp0'",
        "creating device node <*>",
    ),
    "bracketed_exit_code": (
        "ALERT exited abnormally with [1]",
        "ALERT exited abnormally with <*>",
    ),
    "network_interface": (
        "listening on IPv4 interface eth0, 63.126.79.67#53",
        "listening on IPv4 interface <*>, <*>",
    ),
    "kernel_stack_address": (
        "[<0212ebf3>] __alloc_pages+0x274/0x281",
        "[<*>] __alloc_pages+<*>",
    ),
    "kernel_stack_module": (
        "[<0a8ddb99>] ext3_file_write+0x19/0x88 [ext3]",
        "[<*>] ext3_file_write+<*>",
    ),
    "lifecycle_subject": (
        "cupsd startup succeeded",
        "<*> startup succeeded",
    ),
    "fixed_parenthesized_service": (
        "getpeername (ftpd): Transport endpoint is not connected",
        "getpeername (ftpd): Transport endpoint is not connected",
    ),
    "numeric_range": (
        "ip_tables: (C) 2000-2002 Netfilter core team",
        "ip_tables: (C) <*> Netfilter core team",
    ),
    "variable_component": (
        "Bluetooth: Core ver 2.4",
        "Bluetooth: <*> ver <*>",
    ),
    "driver_identifier": (
        "3c59x: Donald Becker and others. www.scyld.com/network/vortex.html",
        "<*>: Donald Becker and others. <*>",
    ),
    "selinux_avc_fields": (
        "audit(1138278089.855:0): avc: denied { read write } for pid=1 "
        "exe=/sbin/init name=initctl dev=hda2 ino=1031635 "
        "scontext=system_u:system_r:kernel_t "
        "tcontext=system_u:object_r:file_t tclass=fifo_file",
        "audit(<*>): avc: denied <*> for pid=<*> exe=<*> name=<*> dev=<*> "
        "ino=<*> scontext=<*> tcontext=<*> tclass=<*>",
    ),
}


GROUPS = {
    "network_path": {0, 1, 2, 3, 4},
    "names_and_devices": {5, 6, 12},
    "runtime_and_range": {7, 8, 9, 10, 11, 13},
    "field_structure": {14, 15, 16},
    "account_and_state": {17, 18, 19, 20},
    "punctuation": {21},
    "specific_constants": set(range(22, 39)),
}


def main():
    prompt_rules = list(gq.CURATED_PROMPT_RULES)
    preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    variants = {
        "current": (prompt_rules, preprocessing),
        "prompt_only": (prompt_rules, []),
        "preprocessing_only": ([], preprocessing),
        "no_knowledge": ([], []),
    }
    for name, removed in GROUPS.items():
        variants[f"without_{name}"] = (
            [
                rule
                for index, rule in enumerate(prompt_rules)
                if index not in removed
            ],
            preprocessing,
        )

    regs = common_regexes()
    records = []
    for variant, (rules, pre_rules) in variants.items():
        for case, (log, expected) in CASES.items():
            record = parse(log, expected, rules, pre_rules, regs)
            record.update(variant=variant, case=case)
            records.append(record)
            print(
                f"{variant:28s} {case:28s}: "
                f"{'CORRECT' if record['correct'] else 'wrong'} "
                f"{record['final']!r}",
                flush=True,
            )

    output_dir = (
        ROOT
        / "others/test_one_log/results/domain_rule_ablation"
        / f"linux_groups_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
