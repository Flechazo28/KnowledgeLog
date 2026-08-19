#!/usr/bin/env python3
"""Validate generalized fixes for major Thunderbird errors without editing gpt_query.py."""

import json
import re
from collections import Counter
from datetime import datetime

from run_webmaster_rule_ablation import ROOT, common_regexes, gq, parse


REPETITIONS = 3

CASES = {
    "missing_session_actor": (
        "session opened for user root by (uid=0)",
        "session opened for user <*> by <*>(uid=<*>)",
    ),
    "cron_execution_identity": (
        "(root) CMD (run-parts /etc/cron.hourly)",
        "(<*>) CMD (<*>)",
    ),
    "bracketed_source_location": (
        "[KERNEL_IB][ib_mad_dispatch]"
        "[/mnt_projects/sysapps/src/ib/topspin/topspin-src-3.2.0-16/"
        "ib/ts_api_ng/mad/obj_host_amd64_custom1_rhel4/"
        "ts_ib_mad/mad_filter.c:292]mad_process failed (0) for "
        "InfiniHost0 port 1 QPN 0 (class 0x81, aid 0x0011)",
        "[KERNEL_IB][ib_mad_dispatch][<*>]mad_process failed (<*>) "
        "for <*> port <*> QPN <*> (class <*>, aid <*>)",
    ),
    "simple_bracketed_source": (
        "[ib_sm_discovery.c:1103]: Failed discover node test, "
        "node 5ad0000028dce, port_num= 18, error code 1",
        "[<*>]: Failed discover node test, node <*>, port_num= <*>, "
        "error code <*>",
    ),
    "path_with_bracketed_line": (
        "VIPKL(1): mnt_projects/sysapps/src/ib/topspin/"
        "topspin-src-3.2.0-16/third_party/thca4_linux/kernel/vip/"
        "obj_host_amd64_custom1_rhel4/mod_vip/mmu.c[243]: "
        "make_iobuf: MOSAL_iobuf_register failed: "
        "va=0xCE9A000size=0x29000",
        "VIPKL(<*>): <*>[<*>]: make_iobuf: MOSAL_iobuf_register "
        "failed: va=<*>=<*>",
    ),
    "path_with_colon_line": (
        "MOSAL(1): xhpl[9285]: mnt_projects/sysapps/src/ib/topspin/"
        "topspin-src-3.2.0-16/third_party/thca4_linux/kernel/mlxsys/"
        "obj_host_amd64_custom1_rhel4/mlxsys/mosal_iobuf.c:1136: "
        "MOSAL_mlock_iobuf failed - MT_ENORSC",
        "MOSAL(<*>): xhpl[<*>]: <*>: MOSAL_mlock_iobuf failed - MT_ENORSC",
    ),
    "fixed_parenthesized_status": (
        "Instrumentation Service EventID: 1053 Temperature sensor detected "
        "a warning value Sensor location: BMC Ambient Temp Chassis location: "
        "Main System Chassis Previous state was: OK (Normal) Temperature "
        "sensor value (in Degrees Celsius): 10.0",
        "Instrumentation Service EventID: <*> Temperature sensor detected "
        "a warning value Sensor location: <*> Chassis location: <*> Previous "
        "state was: <*> (Normal) Temperature sensor value "
        "(in Degrees Celsius): <*>",
    ),
    "interface_field": (
        "Listening on interface wildcard, 0.0.0.0#123",
        "Listening on interface <*>, <*>",
    ),
    "dhcp_interface": (
        "DHCPDISCOVER from 00:11:43:e3:c1:e7 via eth1",
        "DHCPDISCOVER from <*> via <*>",
    ),
    "device_driver_subject": (
        "dcdipm device driver loaded",
        "<*> device driver loaded",
    ),
    "termination_subject": (
        "messagebus -TERM succeeded",
        "<*> -TERM succeeded",
    ),
    "segfault_process": (
        "omsad32[7635]: segfault at 0000000000000000 rip "
        "0000000000000000 rsp 00000000f40a601c error 14",
        "<*>[<*>]: segfault at <*> rip <*> rsp <*> error <*>",
    ),
    "explicit_type_field": (
        "Type: Direct-Access ANSI SCSI revision: 03",
        "Type: <*> ANSI SCSI revision: <*>",
    ),
    "explicit_level_field": (
        "ACPI: PCI interrupt 0000:00:1d.7[D] -> GSI 23 "
        "(level, low) -> IRQ 193",
        "ACPI: PCI interrupt <*> -> GSI <*> (level, <*>) -> IRQ <*>",
    ),
    "acpi_table_record": (
        "ACPI: FADT (v001 DELL PE BKC 0x00000001 MSFT 0x0100000a) "
        "@ 0x00000000000fd6b0",
        "ACPI: <*> (<*>) @ <*>",
    ),
    "assigned_state": (
        "[ib_sm_sweep.c:411]: IB node 0x0005AD0000027556 "
        "port 24 is INIT state",
        "[<*>]: IB node <*> port <*> is INIT state",
    ),
    "fixed_state_phrase": (
        "[INFO]: Configuration caused by some ports in INIT state",
        "[INFO]: Configuration caused by some ports in INIT state",
    ),
    "punctuation_annotation_guard": (
        "synchronized to 10.100.26.250, stratum 3",
        "synchronized to <*> stratum <*>",
    ),
}

PROMPT_RULES = [
    (
        "Abstract a parenthesized account or execution identity as {variables} "
        "when it is the actor immediately before an action label; preserve the "
        "parentheses and action label."
    ),
    (
        "Preserve a parenthesized explanatory status label when it is fixed "
        "event wording after a variable state value; do not delete the label "
        "or its parentheses."
    ),
    (
        "Abstract a state value only when a specific entity is assigned that "
        "state by an explicit field or predicate; preserve a state word when "
        "it is part of a general explanatory event phrase."
    ),
    (
        "Abstract a network-interface identifier as {variables} when it is the "
        "value after an explicit network role such as interface or via."
    ),
]

REGEX_RULES = [
    gq.RegexRule(
        name="missing_actor_before_uid",
        pattern=r"(\bby\s+)(?=\(uid\s*=)",
        replacement=r"\1{variables}",
        note="Represent an omitted actor field before a parenthesized UID field.",
    ),
    gq.RegexRule(
        name="parenthesized_execution_identity",
        pattern=r"^\([^()\s]+\)(?=\s+[A-Z][A-Z0-9_-]*\s+\()",
        replacement=f"({gq.VARIABLE_PLACEHOLDER})",
        note="Abstract a parenthesized execution identity before an action label.",
    ),
    gq.RegexRule(
        name="bracketed_source_location",
        pattern=(
            r"(?<=\[)"
            r"(?:/?[A-Za-z0-9_.-]+/)*"
            r"[A-Za-z0-9_.-]+\.[A-Za-z][A-Za-z0-9]*:\d+"
            r"(?=\])"
        ),
        replacement=gq.VARIABLE_PLACEHOLDER,
        note="Abstract a bracketed source file and line number as one location.",
    ),
    gq.RegexRule(
        name="source_path_with_bracketed_line",
        pattern=(
            r"(?<![A-Za-z0-9_.-])"
            r"(?:/?[A-Za-z0-9_.-]+/)+"
            r"[A-Za-z0-9_.-]+\.[A-Za-z][A-Za-z0-9]*"
            r"\[\d+\](?=:)"
        ),
        replacement=f"{gq.VARIABLE_PLACEHOLDER}[{gq.VARIABLE_PLACEHOLDER}]",
        note="Abstract a source path and its bracketed line number separately.",
    ),
    gq.RegexRule(
        name="source_path_with_colon_line",
        pattern=(
            r"(?<![A-Za-z0-9_.-])"
            r"(?:/?[A-Za-z0-9_.-]+/)+"
            r"[A-Za-z0-9_.-]+\.[A-Za-z][A-Za-z0-9]*:\d+"
            r"(?=:)"
        ),
        replacement=gq.VARIABLE_PLACEHOLDER,
        note="Abstract a source path ending in a colon-delimited line number.",
    ),
    gq.RegexRule(
        name="network_interface_value",
        pattern=(
            r"(\b(?:via|interface)\s+)"
            r"(?:wildcard|lo|(?:eth|ib|enp?|bond)\d+)"
            r"(?=[:,.\s]|$)"
        ),
        replacement=r"\1{variables}",
        flags=re.IGNORECASE,
        note="Abstract a network-interface value after an explicit network role.",
    ),
    gq.RegexRule(
        name="extended_lifecycle_subject",
        pattern=(
            r"^[A-Za-z_][A-Za-z0-9_.-]*"
            r"(?=\s+(?:device driver (?:loaded|unloaded)"
            r"|kernel module initialized successfully"
            r"|-[A-Z][A-Z0-9_-]* succeeded)\b)"
        ),
        replacement=gq.VARIABLE_PLACEHOLDER,
        flags=re.IGNORECASE,
        note="Abstract the subject of an explicit lifecycle result.",
    ),
    gq.RegexRule(
        name="segfault_process_name",
        pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*(?=\[\d+\]:\s+segfault\b)",
        replacement=gq.VARIABLE_PLACEHOLDER,
        flags=re.IGNORECASE,
        note="Abstract the process name in a process[pid] crash prefix.",
    ),
    gq.RegexRule(
        name="explicit_type_value",
        pattern=r"(\bType\s*:\s*)[A-Za-z][A-Za-z0-9_.-]*",
        replacement=r"\1{variables}",
        note="Abstract the value of an explicitly labeled Type field.",
    ),
    gq.RegexRule(
        name="explicit_level_value",
        pattern=r"(\blevel,\s*)[A-Za-z][A-Za-z0-9_.-]*(?=\))",
        replacement=r"\1{variables}",
        note="Abstract the value of an explicitly labeled level field.",
    ),
    gq.RegexRule(
        name="acpi_table_record",
        pattern=r"^(ACPI:\s+)\S+\s+\([^)]*\)(\s+@\s+)\S+$",
        replacement=(
            rf"\1{gq.VARIABLE_PLACEHOLDER} "
            rf"({gq.VARIABLE_PLACEHOLDER})"
            rf"\2{gq.VARIABLE_PLACEHOLDER}"
        ),
        note="Abstract the table name, metadata block, and address of an ACPI record.",
    ),
]


def main():
    current_prompt = list(gq.CURATED_PROMPT_RULES)
    current_preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    variants = {
        "current": (current_prompt, current_preprocessing),
        "prompt_candidate": ([*PROMPT_RULES, *current_prompt], current_preprocessing),
        "regex_candidate": (
            current_prompt,
            [*REGEX_RULES, *current_preprocessing],
        ),
        "combined_candidate": (
            [*PROMPT_RULES, *current_prompt],
            [*REGEX_RULES, *current_preprocessing],
        ),
    }
    regs = common_regexes()
    records = []

    for variant, (prompt_rules, preprocessing_rules) in variants.items():
        for case, (log_message, expected) in CASES.items():
            for repetition in range(1, REPETITIONS + 1):
                record = parse(
                    log_message,
                    expected,
                    prompt_rules,
                    preprocessing_rules,
                    regs,
                )
                record.update(
                    variant=variant,
                    case=case,
                    repetition=repetition,
                )
                records.append(record)
                print(
                    f"{variant:20s} {case:30s} {repetition}/{REPETITIONS}: "
                    f"{'CORRECT' if record['correct'] else 'wrong'} "
                    f"{record['final']!r}",
                    flush=True,
                )

    output_dir = (
        ROOT
        / "others/test_one_log/results/regex_validation"
        / f"thunderbird_major_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
                "templates": dict(Counter(record["final"] for record in selected)),
            }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
