#!/usr/bin/env python3
"""Compare current and candidate domain rules on representative parsing errors."""

import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import regex


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

from logparser.KnowledgeLog import gpt_query as gq  # noqa: E402


MODEL = "Llama-3.3-70B-Instruct"
REPETITIONS = 3

CASES = {
    "spark_added": (
        "Added broadcast_2_piece0 in memory on 10.10.34.31:54234 "
        "(size: 1024.0 MB, free: 25.4 GB)",
        "Added <*> in memory on <*> (size: <*>, free: <*>)",
    ),
    "spark_block": (
        "Block broadcast_2_piece0 stored as bytes in memory "
        "(estimated size 92.0 B, free 1402.7 MB)",
        "Block <*> stored as bytes in memory (estimated size <*>, free <*>)",
    ),
    "hpc_temperature": (
        "Temperature (41C) exceeds warning threshold",
        "Temperature <*> exceeds warning threshold",
    ),
    "zk_timeout": (
        "Expiring session 0x34ede65503f0003, timeout of 10000ms exceeded",
        "Expiring session <*> timeout of <*> exceeded",
    ),
    "health_report": (
        "REPORT : 7007 5002 150089 240",
        "REPORT : <*>",
    ),
    "health_upload": (
        "upLoadOneMinuteDataToEngine time=25233975,0,93,0,20002",
        "upLoadOneMinuteDataToEngine time=<*>",
    ),
    "health_new_date": (
        "new date =20171223, type=40002,7163.0,old=6983.0",
        "new date =<*>, type=<*>,old=<*>",
    ),
    "openssh_pam": (
        "pam_unix(sshd:auth): check pass; user unknown",
        "pam_unix(sshd:auth): check pass; user unknown",
    ),
    "tb_source": (
        "[KERNEL_IB][ib_mad_dispatch]"
        "[/mnt_projects/sysapps/src/ib/topspin/topspin-src-3.2.0-16/ib/"
        "ts_api_ng/mad/obj_host_amd64_custom1_rhel4/ts_ib_mad/"
        "mad_filter.c:292]mad_process failed (0) for InfiniHost0 port 1 "
        "QPN 0 (class 0x81, aid 0x0015)",
        "[KERNEL_IB][ib_mad_dispatch][<*>]mad_process failed (<*>) "
        "for <*> port <*> QPN <*> (class <*>, aid <*>)",
    ),
    "tb_source_simple": (
        "[ib_sm_discovery.c:470]: Failed to GetNodeInfo() because of NO RESPONSE",
        "[<*>]: Failed to GetNodeInfo() because of NO RESPONSE",
    ),
    "zk_class_value": (
        "Follower sid: 2 : info : "
        "org.apache.zookeeper.server.quorum.QuorumPeer$QuorumServer@9317ccb",
        "Follower sid: <*> : info : <*>",
    ),
    "bgl_function": (
        "idoproxydb hit ASSERT condition: ASSERT expression=0 "
        "Source file=idotransportmgr.cpp Source line=1043 Function=int "
        "IdoTransportMgr::SendPacket(IdoUdpMgr*, BglCtlPavTrace*)",
        "idoproxydb hit ASSERT condition: ASSERT expression=<*> "
        "Source file=<*> Source line=<*> Function=<*>",
    ),
    "openssh_comma": (
        "Address 82.209.247.10 maps to mail.trionis.ru, but this does not "
        "map back to the address - POSSIBLE BREAK-IN ATTEMPT!",
        "Address <*> maps to <*> but this does not map back to the address "
        "- POSSIBLE BREAK-IN ATTEMPT!",
    ),
    "linux_quotes": (
        "creating device node '/udev/lp0'",
        "creating device node <*>",
    ),
    "hadoop_brackets": (
        "Failed to renew lease for [DFSClient_NONMAPREDUCE_483047941_1] "
        "for 46 seconds. Will retry shortly ...",
        "Failed to renew lease for <*> for <*> seconds. Will retry shortly ...",
    ),
    "tb_mapped_ipv4": (
        "Accepted publickey for root from ::ffff:10.100.4.251 "
        "port 35558 ssh2",
        "Accepted publickey for <*> from <*> port <*> ssh2",
    ),
    "ssh_plain_ipv4": (
        "Failed password for root from 183.62.140.253 port 60221 ssh2",
        "Failed password for <*> from <*> port <*> ssh2",
    ),
    "version_guard": (
        "component version 1.2.3.4.5 loaded",
        "component version <*> loaded",
    ),
    "proxifier_bytes": (
        "183.62.156.108:22 close, 89652 bytes sent, 599249 bytes received, "
        "lifetime 37:51",
        "<*> close, <*> bytes sent, <*> bytes received, lifetime <*>",
    ),
    "spark_named_users": (
        "SecurityManager: authentication disabled; ui acls disabled; users with "
        "view permissions: Set(yarn, curi); users with modify permissions: "
        "Set(yarn, curi)",
        "SecurityManager: authentication disabled; ui acls disabled; users with "
        "view permissions: Set(<*>, <*>); users with modify permissions: "
        "Set(<*>, <*>)",
    ),
    "ipv4_list_guard": (
        "peers 10.1.2.3,10.2.3.4 connected",
        "peers <*> connected",
    ),
}

GROUPS = {
    "measurement": ["spark_added", "spark_block", "hpc_temperature", "zk_timeout"],
    "sequence": ["health_report", "health_upload", "health_new_date"],
    "source_symbol": ["openssh_pam", "tb_source", "zk_class_value"],
    "punctuation": ["openssh_comma", "linux_quotes", "hadoop_brackets"],
    "key_value_regex": ["bgl_function", "health_upload", "health_new_date"],
    "mapped_ipv4_regex": ["tb_mapped_ipv4", "ssh_plain_ipv4", "version_guard"],
}

MEASUREMENT_RULE = (
    "Abstract a measurement value as one {variables}, including both its number "
    "and unit, when the unit is part of the value or may change its scale, such "
    "as B, KB, MB, GB, ms, or C. Preserve a unit only when it is fixed "
    "grammatical text outside the value, such as 'for {variables} seconds'."
)
SEQUENCE_RULE = (
    "Abstract an unlabeled packed or repeated sequence as one {variables}, "
    "regardless of the number of whitespace- or comma-separated items. Split it "
    "into separate variables only when fixed field labels explicitly identify "
    "separate values."
)
SOURCE_RULES = [
    "Treat a complete source location, including its path or file name and line "
    "number, as one {variables} when it identifies where an event occurred.",
    "Preserve package, class, module, method, file, protocol, service, component, "
    "and event-context names when they are fixed event wording. Abstract the "
    "complete name as one {variables} only when the surrounding syntax identifies "
    "it as a changing field value.",
]
PUNCTUATION_RULE = (
    "Do not invent or rewrite punctuation. Preserve punctuation that belongs to "
    "fixed event syntax; when punctuation only wraps one changing field, treat "
    "the complete wrapped text as the variable field."
)
KV_PATTERN = (
    r"(?<![\w.-])"
    r"([A-Za-z_][A-Za-z0-9_.-]*\s*=\s*?)"
    r"(?:[A-Za-z0-9._]+)?"
    r"(?=$|(?:\s+|[,;]\s*)(?=[A-Za-z_][A-Za-z0-9_.-]*\s*=))"
)
IPV4 = (
    r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
)
MAPPED_IPV4_PATTERN = (
    r"(?<![A-Za-z0-9_./:])/?(?:::ffff:)?"
    + IPV4
    + r"(?:,/?(?:::ffff:)?"
    + IPV4
    + r")*(?![\d.:])"
)
MEASUREMENT_RULE_V2 = (
    "When a numeric measurement uses a scale token such as B, KB, MB, GB, or "
    "TB, or an attached unit such as ms or C, replace the number and unit "
    "together with one {variables}; never leave the scale token after the "
    "placeholder. Preserve a spelled-out unit only when it is fixed grammar "
    "outside the value, as in '{variables} bytes sent' or 'for {variables} "
    "seconds'."
)
SEQUENCE_RULE_V2 = (
    "After one label, abstract a consecutive unlabeled sequence as exactly one "
    "{variables}, including all whitespace- or comma-separated items up to the "
    "next explicit label. Never emit one placeholder for each item."
)
KEY_VALUE_RULE_V2 = (
    "In a key-value field, preserve the key and delimiter and abstract the "
    "complete value as one {variables}. The value continues through spaces or "
    "commas until the next explicit key-value field; do not stop after its first "
    "token or list item."
)
SOURCE_LOCATION_RULE_V2 = (
    "Always abstract a complete source location as one {variables}, including "
    "every path segment, file name, version-like segment, and line number. A "
    "source location reports where an event occurred, often as path/file:line "
    "or [file:line]; never preserve or split any part of it."
)
EVENT_CONTEXT_RULE_V2 = (
    "Preserve a function-style event prefix of the form component(context): as "
    "fixed event wording. Do not abstract the component or context unless an "
    "explicit key identifies it as a changing field value."
)
RUNTIME_OBJECT_RULE_V2 = (
    "When an explicit labeled field contains a package or class name followed "
    "by @ and an instance hash, treat the entire Class@hash value as one runtime "
    "object identifier and replace all of it with one {variables}."
)
MEASUREMENT_RULE_V3 = (
    "For every field named size, estimated size, or free, replace the complete "
    "number plus storage unit B, KB, MB, GB, or TB with one {variables}; output "
    "'size: {variables}' and never 'size: {variables} MB'. Keep a fixed "
    "grammatical unit in phrases such as '{variables} bytes sent'."
)
SEQUENCE_RULE_V3 = (
    "When one label is followed by two or more unlabeled numbers, treat the "
    "entire consecutive sequence as one composite value and replace it with "
    "exactly one {variables}; for example, 'metrics: 12 34 56' becomes "
    "'metrics: {variables}'."
)
KEY_VALUE_RULE_V3 = (
    "A comma-separated tuple after one key belongs to that value until the next "
    "explicit key. Replace the complete tuple with one {variables}; for example, "
    "'field=12,34.5,next=7' becomes 'field={variables},next={variables}'."
)
FUNCTION_FIELD_RULE_V3 = (
    "For a Function= field, the return type and complete function signature are "
    "one value. Replace all text after Function= through the closing parameter "
    "list with one {variables}."
)
SOURCE_LOCATION_RULE_V3 = (
    "A bracketed file:line marker immediately before an event, such as "
    "[module.c:123]:, is one source-location variable. Replace the entire text "
    "inside the brackets with one {variables}."
)
EVENT_CONTEXT_RULE_V3 = (
    "In a PAM-style prefix component(service:stage):, preserve service:stage as "
    "fixed event context; for example, pam_unix(sshd:auth): remains unchanged."
)
RUNTIME_OBJECT_RULE_V3 = (
    "When an info or object field contains ClassName@instanceHash, abstract the "
    "complete ClassName@instanceHash as one {variables}; never preserve only the "
    "class-name prefix."
)
ACCOUNT_RULE_V4 = (
    "Abstract an account identifier as {variables} when the context explicitly "
    "denotes a user, login, account, authentication identity, or principal. "
    "After 'for' or 'user', abstract the next token only when it names an "
    "account; preserve the fixed status phrase 'user unknown' because 'unknown' "
    "is not an account identifier."
)
EVENT_CONTEXT_RULE_V4 = (
    "In a PAM-style prefix component(service:stage):, preserve service:stage as "
    "fixed event context; for example, pam_unix(sshd:auth): remains unchanged."
)
USER_UNKNOWN_RULE_V5 = (
    "Preserve the fixed authentication status phrase 'user unknown' exactly; "
    "'unknown' is a status word, not a username."
)
ACCOUNT_RULE_V6 = (
    "Abstract a concrete account name such as root, admin, or ubuntu when the "
    "context identifies a user, login, account, authentication identity, or "
    "principal. Do not abstract status words such as unknown."
)
CLASS_INSTANCE_PATTERN = (
    r"(?<![\w$])"
    r"(?:[A-Za-z_$][A-Za-z0-9_$]*\.)+"
    r"[A-Za-z_$][A-Za-z0-9_$]*@[0-9A-Fa-f]+"
    r"(?![\w])"
)
LABELED_NUMERIC_SEQUENCE_PATTERN = (
    r"([:=]\s*)"
    r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
    r"(?:[ \t]+[+-]?(?:\d+(?:\.\d+)?|\.\d+)){1,}"
    r"(?=$|[)\]])"
)


def common_regexes():
    path = ROOT / "benchmark/logparser/KnowledgeLog/common.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [regex.compile(pattern) for pattern in payload["COMMON"]["regex"]]


def candidate_prompt(group):
    rules = list(gq.CURATED_PROMPT_RULES)
    if group == "measurement":
        rules[14] = MEASUREMENT_RULE
    elif group == "sequence":
        rules[16] = SEQUENCE_RULE
    elif group == "source_symbol":
        rules[5:8] = SOURCE_RULES
    elif group == "punctuation":
        rules[23] = PUNCTUATION_RULE
    elif group == "measurement_v2":
        rules[14] = MEASUREMENT_RULE_V2
    elif group == "sequence_key_value_v2":
        rules[16] = SEQUENCE_RULE_V2
        rules[17] = KEY_VALUE_RULE_V2
    elif group == "source_location_v2":
        rules[5] = SOURCE_LOCATION_RULE_V2
    elif group == "event_context_v2":
        rules[7] = EVENT_CONTEXT_RULE_V2
    elif group == "runtime_object_v2":
        rules[7] = RUNTIME_OBJECT_RULE_V2
    elif group == "measurement_v3":
        rules.pop(14)
        rules.append(MEASUREMENT_RULE_V3)
    elif group == "sequence_key_value_v3":
        rules.pop(17)
        rules.pop(16)
        rules.extend([SEQUENCE_RULE_V3, KEY_VALUE_RULE_V3])
    elif group == "function_field_v3":
        rules.append(FUNCTION_FIELD_RULE_V3)
    elif group == "source_location_v3":
        rules.pop(5)
        rules.append(SOURCE_LOCATION_RULE_V3)
    elif group == "event_context_v3":
        rules.pop(7)
        rules.append(EVENT_CONTEXT_RULE_V3)
    elif group == "runtime_object_v3":
        rules.pop(7)
        rules.append(RUNTIME_OBJECT_RULE_V3)
    elif group == "event_context_account_v4":
        rules[19] = ACCOUNT_RULE_V4
        rules.pop(7)
        rules.append(EVENT_CONTEXT_RULE_V4)
    elif group == "event_context_account_v5":
        rules[19] = ACCOUNT_RULE_V4
        rules.pop(7)
        rules.extend([EVENT_CONTEXT_RULE_V4, USER_UNKNOWN_RULE_V5])
    elif group == "event_context_account_v6":
        rules[19] = ACCOUNT_RULE_V6
        rules.pop(7)
        rules.extend([EVENT_CONTEXT_RULE_V4, USER_UNKNOWN_RULE_V5])
    return rules


def candidate_preprocessing(group):
    rules = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    replacements = {}
    if group in {
        "key_value_regex",
        "sequence_key_value_v2",
        "sequence_key_value_v3",
        "function_field_v3",
    }:
        replacements["curated_0031_key_value"] = KV_PATTERN
    elif group == "mapped_ipv4_regex":
        replacements["curated_0002_ipv4"] = MAPPED_IPV4_PATTERN
    candidate_rules = [
        gq.RegexRule(
            name=rule.name,
            pattern=replacements.get(rule.name, rule.pattern),
            replacement=rule.replacement,
            flags=rule.flags,
            enabled=rule.enabled,
            note=rule.note,
        )
        for rule in rules
    ]
    if group == "class_instance_regex_v4":
        candidate_rules.append(
            gq.RegexRule(
                name="candidate_class_instance",
                pattern=CLASS_INSTANCE_PATTERN,
                replacement=gq.VARIABLE_PLACEHOLDER,
            )
        )
    elif group == "numeric_sequence_regex_v4":
        candidate_rules.append(
            gq.RegexRule(
                name="candidate_labeled_numeric_sequence",
                pattern=LABELED_NUMERIC_SEQUENCE_PATTERN,
                replacement=r"\1{variables}",
            )
        )
    return candidate_rules


def run_case(case_id, variant, prompt_group, preprocessing_group, regs):
    log, expected = CASES[case_id]
    response = gq.get_response_from_openai_key(
        log,
        model=MODEL,
        enabled_prompt_groups=(prompt_group,),
        enabled_preprocessing_groups=(preprocessing_group,),
    )
    extracted = gq._extract_template_from_response(response)
    if extracted is None:
        final, valid = None, False
    else:
        final, valid = gq.post_process_template(
            extracted,
            regs_common=regs,
            log_message=log,
        )
    return {
        "case_id": case_id,
        "variant": variant,
        "log": log,
        "expected": expected,
        "preprocessed": gq.apply_final_domain_preprocessing(
            log, enabled_groups=(preprocessing_group,)
        ),
        "raw_response": response,
        "extracted": extracted,
        "final": final,
        "valid": valid,
        "correct": bool(valid and final == expected),
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        ROOT / "others/test_one_log/results/domain_rule_ablation" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    regs = common_regexes()
    records = []

    if "--round7" in sys.argv:
        variants = [
            (
                "mapped_ipv4_regex",
                [
                    "tb_mapped_ipv4",
                    "ssh_plain_ipv4",
                    "version_guard",
                    "ipv4_list_guard",
                ],
            ),
        ]
    elif "--round6" in sys.argv:
        variants = [
            (
                "event_context_account_v6",
                ["openssh_pam", "ssh_plain_ipv4"],
            ),
        ]
    elif "--round5" in sys.argv:
        variants = [
            (
                "event_context_account_v5",
                ["openssh_pam", "ssh_plain_ipv4"],
            ),
        ]
    elif "--round4" in sys.argv:
        variants = [
            (
                "event_context_account_v4",
                ["openssh_pam", "ssh_plain_ipv4"],
            ),
            ("class_instance_regex_v4", ["zk_class_value"]),
            (
                "numeric_sequence_regex_v4",
                ["health_report", "spark_named_users"],
            ),
        ]
    elif "--round3" in sys.argv:
        variants = [
            ("measurement_v3", ["spark_added", "spark_block", "proxifier_bytes"]),
            (
                "sequence_key_value_v3",
                [
                    "health_report",
                    "health_upload",
                    "health_new_date",
                    "spark_named_users",
                ],
            ),
            ("function_field_v3", ["bgl_function"]),
            ("source_location_v3", ["tb_source_simple"]),
            ("event_context_v3", ["openssh_pam"]),
            ("runtime_object_v3", ["zk_class_value"]),
        ]
    elif "--round2" in sys.argv:
        variants = [
            ("measurement_v2", ["spark_added", "spark_block"]),
            (
                "sequence_key_value_v2",
                ["health_report", "health_upload", "health_new_date", "bgl_function"],
            ),
            ("source_location_v2", ["tb_source_simple"]),
            ("event_context_v2", ["openssh_pam"]),
            ("runtime_object_v2", ["zk_class_value"]),
        ]
    else:
        all_cases = sorted({case for cases in GROUPS.values() for case in cases})
        variants = [("baseline", all_cases)]
        variants.extend((group, cases) for group, cases in GROUPS.items())

    for variant, case_ids in variants:
        group_name = f"ablation_{variant}"
        if variant == "baseline":
            gq.PROMPT_GROUPS[group_name] = list(gq.CURATED_PROMPT_RULES)
            gq.PREPROCESSING_RULE_GROUPS[group_name] = list(
                gq.PREPROCESSING_RULE_GROUPS["curated"]
            )
        else:
            gq.PROMPT_GROUPS[group_name] = candidate_prompt(variant)
            gq.PREPROCESSING_RULE_GROUPS[group_name] = candidate_preprocessing(
                variant
            )

        for case_id in case_ids:
            for repetition in range(1, REPETITIONS + 1):
                started = time.monotonic()
                record = run_case(
                    case_id,
                    variant,
                    group_name,
                    group_name,
                    regs,
                )
                record["repetition"] = repetition
                record["seconds"] = round(time.monotonic() - started, 3)
                records.append(record)
                print(
                    f"{variant:18s} {case_id:20s} {repetition}/{REPETITIONS} "
                    f"{'CORRECT' if record['correct'] else 'wrong'}: "
                    f"{record['final']!r}",
                    flush=True,
                )

    result_path = output_dir / "results.json"
    result_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {}
    for variant, case_ids in variants:
        summary[variant] = {}
        for case_id in case_ids:
            selected = [
                record
                for record in records
                if record["variant"] == variant and record["case_id"] == case_id
            ]
            summary[variant][case_id] = {
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
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
