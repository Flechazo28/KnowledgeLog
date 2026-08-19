#!/usr/bin/env python3
"""Screen prompt-knowledge groups on Thunderbird template explosion cases."""

import json
from collections import Counter
from datetime import datetime

from run_webmaster_rule_ablation import ROOT, common_regexes, gq, parse


CASES = {
    "mail_queue_id": (
        "jA9IBUZj019524: from=root, size=629060, class=0, nrcpts=1, "
        "msgid=<200511091811.jA9IBUZj019524@eadmin1>, relay=#7#@localhost",
        "<*>: from=<*>, size=<*>, class=<*>, nrcpts=<*>, msgid=<*>, relay=<*>",
    ),
    "pan_capability_id": (
        "pan_sam: I-xD020038da11df0002-xGf5edfff4-xU026e35b9121707b1: "
        "error report: cap_handle=0x2f47f6374c3625cc op=OP_WRITE "
        "op_offset=4096 op_length=82 action_mask=0xbf400000 cap_offset=0 "
        "cap_length=65536",
        "pan_sam: <*>: error report: cap_handle=<*> op=<*> op_offset=<*> "
        "op_length=<*> action_mask=<*> cap_offset=<*> cap_length=<*>",
    ),
    "pan_failure_id": (
        "pan_sam: I-xD020038da11df0002-xGf5edfff4-xUf029f5d7e546a4a5: "
        "error report: failure 1/2: 0x06003b0c122f0001(OBSD) "
        "osd_op=OP_WRITE offset=7536640 length=65536 error=0x7e9 "
        "(RPC: No response within the timeout period)",
        "pan_sam: <*>: error report: failure <*>: <*>(OBSD) "
        "osd_op=OP_WRITE offset=<*> length=<*> error=<*> "
        "(RPC: No response within the timeout period)",
    ),
    "source_location": (
        "VIPKL(1): mnt_projects/sysapps/src/ib/topspin/topspin-src-3.2.0-16/"
        "third_party/thca4_linux/kernel/vip/obj_host_amd64_custom1_rhel4/"
        "mod_vip/mmu.c[243]: make_iobuf: MOSAL_iobuf_register failed: "
        "va=0x1D0750C0size=0x1A2D",
        "VIPKL(<*>): <*>[<*>]: make_iobuf: MOSAL_iobuf_register failed: "
        "va=<*>=<*>",
    ),
    "segfault_process": (
        "omsad32[2824]: segfault at 0000000000000000 rip 0000000000000000 "
        "rsp 00000000f40a601c error 14",
        "<*>[<*>]: segfault at <*> rip <*> rsp <*> error <*>",
    ),
    "shutdown_service": (
        "dcevt32d shutdown succeeded",
        "<*> shutdown succeeded",
    ),
    "startup_service": (
        "ntpd startup succeeded",
        "<*> startup succeeded",
    ),
}

GROUPS = {
    "network_path": range(0, 5),
    "fixed_and_runtime_names": range(5, 8),
    "runtime_fields": range(8, 14),
    "composite_fields": range(14, 17),
    "account": range(17, 18),
    "state_format": range(18, 22),
    "specific_constants": range(22, 39),
}


def main():
    rules = list(gq.CURATED_PROMPT_RULES)
    preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    variants = {
        "current": (rules, preprocessing),
        "prompt_only": (rules, []),
        "preprocessing_only": ([], preprocessing),
        "no_knowledge": ([], []),
    }
    for name, indices in GROUPS.items():
        removed = set(indices)
        variants[f"without_{name}"] = (
            [rule for i, rule in enumerate(rules) if i not in removed],
            preprocessing,
        )
        variants[f"only_{name}"] = (
            [rule for i, rule in enumerate(rules) if i in removed],
            preprocessing,
        )

    regs = common_regexes()
    records = []
    for variant, (variant_rules, variant_preprocessing) in variants.items():
        for case, (log, expected) in CASES.items():
            record = parse(
                log,
                expected,
                variant_rules,
                variant_preprocessing,
                regs,
            )
            record.update(variant=variant, case=case)
            records.append(record)
            print(
                f"{variant:34s} {case:20s}: "
                f"{'CORRECT' if record['correct'] else 'wrong'} "
                f"{record['final']!r}",
                flush=True,
            )

    output_dir = (
        ROOT
        / "others/test_one_log/results/domain_rule_ablation"
        / f"thunderbird_groups_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {}
    for variant in variants:
        selected = [record for record in records if record["variant"] == variant]
        summary[variant] = {
            "correct": sum(record["correct"] for record in selected),
            "runs": len(selected),
            "templates": {
                record["case"]: record["final"] for record in selected
            },
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
