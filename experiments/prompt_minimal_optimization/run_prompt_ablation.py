#!/usr/bin/env python3
"""Compare the current prompt with compact alternatives on high-impact errors."""

import argparse
import json
import re
import sys
from pathlib import Path

import regex


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

from logparser.KnowledgeLog import gpt_query as gq  # noqa: E402


MODEL = "Llama-3.3-70B-Instruct"
RESULT_DIR = (
    ROOT
    / "result"
    / "KnowledgeLog_full_14datasets_20260730_200403"
)
OUTPUT_PATH = Path(__file__).with_name("prompt_ablation_results.json")

INSTRUCTION = (
    "I want you to act like an expert of log parsing. I will give you rules "
    "that you must follow and a log message delimited by backticks. You must "
    "identify and abstract all dynamic variables in logs with {variables}, then "
    "output a static log template. Please note that there might be no variables "
    "in logs. You must print the input log's template following this exact "
    "format: 'Log template:`...`'"
)

COMPACT_GENERAL_RULES = [
    "Replace each complete dynamic field with {variables}; preserve all static words, field order, and punctuation from the log.",
    "Treat a complete path, URL, request target, hostname:port, or IP:port as one variable field.",
    "Abstract runtime identifiers, account names, timestamps, measured values, and explicitly labeled numbers or codes.",
    "For key-value fields, preserve the key and delimiter and abstract only the variable value.",
    "Preserve a technical name when it is fixed event wording; abstract it when it fills a labeled field or a variable grammatical role.",
    "Preserve fixed status and event wording. In an explicit transition from X to Y, abstract X and Y separately.",
    "Treat a wrapped field or a variable-length repeated list as one variable when the wrapper or list belongs to that field.",
    "Never invent, delete, move, or rewrite constant text or punctuation.",
]

COMPACT_ADJUSTED_RULES = COMPACT_GENERAL_RULES + [
    "In authentication messages, preserve fixed module and status text such as pam_unix(sshd:auth) and user unknown; abstract only concrete account identities.",
    "Preserve fixed disconnect reasons and fixed state names such as available and INIT; abstract them only when they occupy a varying field.",
    "When several labeled fields occur in one log, retain every key and abstract every corresponding value.",
]

PRUNED_CURRENT_RULES = (
    gq.CURATED_PROMPT_RULES[:5]
    + [
        "Preserve a technical name when it is fixed event wording; abstract the complete name only when it fills a labeled field or a variable grammatical role.",
        "Abstract a symbol-rich token only when context identifies it as a runtime field; preserve fixed module, service, protocol, and wrapper names.",
    ]
    + gq.CURATED_PROMPT_RULES[7:18]
    + [
        "In an explicit transition from X to Y, abstract X and Y separately. Otherwise preserve a named state when it is fixed event wording; abstract it only as a labeled state or status value.",
    ]
    + gq.CURATED_PROMPT_RULES[19:22]
    + gq.CURATED_PROMPT_RULES[37:39]
)

REVISED_CURRENT_RULES = list(gq.CURATED_PROMPT_RULES)
REVISED_CURRENT_RULES[5] = (
    "Preserve fixed package, class, module, method, protocol, service, component, "
    "and wrapper names. Abstract a technical name only when it fills an explicit "
    "field or a variable grammatical role; internal punctuation alone is not evidence."
)
REVISED_CURRENT_RULES[17] = (
    "Abstract a concrete account identity in a user, login, account, authentication, "
    "or principal field. Preserve fixed authentication result text such as user unknown."
)
REVISED_CURRENT_RULES[18] = (
    "Preserve state words in fixed phrases such as 'in X state' and 'is in the X state'. "
    "Abstract states only after an explicit state/status key or as the separate endpoints "
    "of an explicit transition from X to Y."
)


CASES = [
    {
        "id": "openssh_pam_wrapper",
        "dataset": "OpenSSH",
        "log": "pam_unix(sshd:auth): check pass; user unknown",
        "expected": "pam_unix(sshd:auth): check pass; user unknown",
    },
    {
        "id": "openssh_invalid_user_name",
        "dataset": "OpenSSH",
        "log": "Failed password for invalid user webmaster from 173.234.31.186 port 38926 ssh2",
        "expected": "Failed password for invalid user <*> from <*> port <*> ssh2",
    },
    {
        "id": "openssh_invalid_user_without_name",
        "dataset": "OpenSSH",
        "log": "Failed password for invalid user from 103.204.177.144 port 2223 ssh2",
        "expected": "Failed password for <*> from <*> port <*> ssh2",
    },
    {
        "id": "openssh_disconnect_reason",
        "dataset": "OpenSSH",
        "log": "Received disconnect from 139.59.209.18: 11: Normal Shutdown, Thank you for playing [preauth]",
        "expected": "Received disconnect from <*>: <*>: Normal Shutdown, Thank you for playing [preauth]",
    },
    {
        "id": "hdfs_allocate_path_and_block",
        "dataset": "HDFS",
        "log": "BLOCK* NameSystem.allocateBlock: /mnt/hadoop/mapred/system/job_200811092030_0001/job.jar. blk_-1608999687919862906",
        "expected": "BLOCK* NameSystem.allocateBlock: <*> <*>",
    },
    {
        "id": "hdfs_delete_path",
        "dataset": "HDFS",
        "log": "Deleting block blk_-919439116365725304 file /mnt/hadoop/dfs/data/current/subdir30/blk_-919439116365725304",
        "expected": "Deleting block <*> file <*>",
    },
    {
        "id": "hpc_fixed_state",
        "dataset": "HPC",
        "log": 'Component State Change: Component \\042alt0\\042 is in the available state (HWID=1328)',
        "expected": "Component State Change: Component <*> is in the available state (HWID=<*>)",
    },
    {
        "id": "hpc_annotation_suffix",
        "dataset": "HPC",
        "log": "ambient=40 threshold exceeded",
        "expected": "ambient=<*>",
    },
    {
        "id": "hadoop_three_fields",
        "dataset": "Hadoop",
        "log": "bufstart = 0; bufend = 34171787; bufvoid = 104857600",
        "expected": "bufstart = <*>; bufend = <*>; bufvoid = <*>",
    },
    {
        "id": "hadoop_compound_fields",
        "dataset": "Hadoop",
        "log": "kvstart = 26214396(104857584); kvend = 13785828(55143312); length = 12428569/6553600",
        "expected": "kvstart = <*>(<*>); kvend = <*>(<*>); length = <*>",
    },
    {
        "id": "hadoop_lease_wrapper",
        "dataset": "Hadoop",
        "log": "Failed to renew lease for [DFSClient_NONMAPREDUCE_483047941_1] for 46 seconds. Will retry shortly ...",
        "expected": "Failed to renew lease for <*> for <*> seconds. Will retry shortly ...",
    },
    {
        "id": "hadoop_topology_path",
        "dataset": "Hadoop",
        "log": "Resolved MININT-FNANLI5.fareast.corp.microsoft.com to /default-rack",
        "expected": "Resolved <*> to <*>",
    },
    {
        "id": "hadoop_transition",
        "dataset": "Hadoop",
        "log": "attempt_1445087491445_0005_m_000000_0 TaskAttempt Transitioned from NEW to UNASSIGNED",
        "expected": "<*> TaskAttempt Transitioned from <*> to <*>",
    },
    {
        "id": "hadoop_labeled_fields",
        "dataset": "Hadoop",
        "log": "Kind: mapreduce.job, Service: job_1445087491445_0005, Ident: (org.apache.hadoop.mapreduce.security.token.JobTokenIdentifier@7f0eb4b4)",
        "expected": "Kind: <*>, Service: <*>, Ident: (<*>)",
    },
    {
        "id": "thunderbird_empty_actor",
        "dataset": "Thunderbird",
        "log": "session opened for user root by (uid=0)",
        "expected": "session opened for user <*> by <*>(uid=<*>)",
    },
    {
        "id": "thunderbird_source_location",
        "dataset": "Thunderbird",
        "log": "[ib_sm_discovery.c:1103]: Failed discover node test, node 66a0001000156, port_num= 12, error code 1",
        "expected": "[<*>]: Failed discover node test, node <*>, port_num= <*>, error code <*>",
    },
    {
        "id": "thunderbird_cron_account",
        "dataset": "Thunderbird",
        "log": "(root) CMD (/projects/tbird/temps/get_temps a)",
        "expected": "(<*>) CMD (<*>)",
    },
    {
        "id": "thunderbird_guid_prefix_fields",
        "dataset": "Thunderbird",
        "log": "[ib_sm_assign.c:229]: Set port GUID=5ad000004cd50, port=1 DOWN (port prefix fe80000000000000, subnet_prefix fe80000000",
        "expected": "[<*>]: Set port GUID=<*>, port=<*> DOWN (port prefix <*>, subnet_prefix <*>",
    },
    {
        "id": "thunderbird_fixed_init",
        "dataset": "Thunderbird",
        "log": "[INFO]: Configuration caused by some ports in INIT state",
        "expected": "[INFO]: Configuration caused by some ports in INIT state",
    },
    {
        "id": "thunderbird_fixed_normal",
        "dataset": "Thunderbird",
        "log": "Instrumentation Service EventID: 1053 Temperature sensor detected a warning value Sensor location: BMC Ambient Temp Chassis location: Main System Chassis Previous state was: OK (Normal) Temperature sensor value (in Degrees Celsius): 10.0",
        "expected": "Instrumentation Service EventID: <*> Temperature sensor detected a warning value Sensor location: <*> Chassis location: <*> Previous state was: <*> (Normal) Temperature sensor value (in Degrees Celsius): <*>",
    },
]


def knowledge(rules):
    return "\nYou must follow these domain knowledge rules:\n" + "\n".join(
        f"{index}. {rule}" for index, rule in enumerate(rules, 1)
    )


def candidate_preprocess(log_message):
    text = gq.apply_final_domain_preprocessing(log_message)
    text = re.sub(
        r"(?<![\w/])/[^\s\]\[()\"',;]+",
        gq.VARIABLE_PLACEHOLDER,
        text,
    )
    text = re.sub(
        r"\[[-A-Za-z0-9_./]+:\d+\]",
        f"[{gq.VARIABLE_PLACEHOLDER}]",
        text,
    )
    text = re.sub(
        r"^\(([^()\s]+)\)(?=\s+CMD\s)",
        f"({gq.VARIABLE_PLACEHOLDER})",
        text,
    )
    text = re.sub(
        r"(\bCMD\s+)\([^)]*\)",
        rf"\1({gq.VARIABLE_PLACEHOLDER})",
        text,
    )
    text = re.sub(
        r"((?:port prefix|subnet_prefix)\s+)[0-9A-Fa-f]+",
        rf"\1{gq.VARIABLE_PLACEHOLDER}",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(\bby\s+)(?=\(uid\s*=)",
        rf"\1{gq.VARIABLE_PLACEHOLDER}",
        text,
    )
    text = re.sub(
        r"(\bFailed password for invalid user\s+)\S+(?=\s+from\s)",
        rf"\1{gq.VARIABLE_PLACEHOLDER}",
        text,
    )
    text = re.sub(
        r"(\bFailed password for\s+)invalid user(?=\s+from\s)",
        rf"\1{gq.VARIABLE_PLACEHOLDER}",
        text,
    )
    return text


def load_common_regexes():
    path = ROOT / "benchmark" / "logparser" / "KnowledgeLog" / "common.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [regex.compile(pattern) for pattern in payload["COMMON"]["regex"]]


def run_case(case, variant, common_regexes):
    preprocessed = (
        candidate_preprocess(case["log"])
        if variant["candidate_preprocessing"]
        else gq.apply_final_domain_preprocessing(case["log"])
    )
    response = gq.infer_llm(
        INSTRUCTION,
        variant["knowledge"],
        f"Log message: `{preprocessed}`",
        case["log"],
        model=MODEL,
        temperature=0.0,
    )
    extracted = gq._extract_template_from_response(response)
    final_template = None
    valid = False
    if extracted is not None:
        final_template, valid = gq.post_process_template(
            extracted,
            regs_common=common_regexes,
            log_message=case["log"],
        )
    return {
        "case_id": case["id"],
        "dataset": case["dataset"],
        "variant": variant["name"],
        "log": case["log"],
        "preprocessed_log": preprocessed,
        "expected": case["expected"],
        "response": response,
        "extracted": extracted,
        "final_template": final_template,
        "valid": valid,
        "correct": bool(valid and final_template == case["expected"]),
    }


def summarize(records):
    summary = {}
    for variant in sorted({record["variant"] for record in records}):
        selected = [record for record in records if record["variant"] == variant]
        per_dataset = {}
        for dataset in sorted({record["dataset"] for record in selected}):
            subset = [record for record in selected if record["dataset"] == dataset]
            per_dataset[dataset] = {
                "correct": sum(record["correct"] for record in subset),
                "total": len(subset),
            }
        summary[variant] = {
            "correct": sum(record["correct"] for record in selected),
            "total": len(selected),
            "per_dataset": per_dataset,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    variants = [
        {
            "name": "current_39_rules",
            "knowledge": gq.build_final_domain_knowledge(),
            "candidate_preprocessing": False,
        },
        {
            "name": "compact_8_rules",
            "knowledge": knowledge(COMPACT_GENERAL_RULES),
            "candidate_preprocessing": False,
        },
        {
            "name": "compact_adjusted_11_rules",
            "knowledge": knowledge(COMPACT_ADJUSTED_RULES),
            "candidate_preprocessing": False,
        },
        {
            "name": "compact_adjusted_plus_minimal_preprocessing",
            "knowledge": knowledge(COMPACT_ADJUSTED_RULES),
            "candidate_preprocessing": True,
        },
        {
            "name": "current_39_rules_plus_minimal_preprocessing",
            "knowledge": gq.build_final_domain_knowledge(),
            "candidate_preprocessing": True,
        },
        {
            "name": "pruned_24_rules",
            "knowledge": knowledge(PRUNED_CURRENT_RULES),
            "candidate_preprocessing": False,
        },
        {
            "name": "pruned_24_rules_plus_minimal_preprocessing",
            "knowledge": knowledge(PRUNED_CURRENT_RULES),
            "candidate_preprocessing": True,
        },
        {
            "name": "revised_39_rules_plus_minimal_preprocessing",
            "knowledge": knowledge(REVISED_CURRENT_RULES),
            "candidate_preprocessing": True,
        },
    ]
    if args.variant:
        requested = set(args.variant)
        variants = [variant for variant in variants if variant["name"] in requested]
        missing = requested - {variant["name"] for variant in variants}
        if missing:
            raise ValueError(f"Unknown variants: {sorted(missing)}")
    common_regexes = load_common_regexes()
    records = []
    total = len(CASES) * len(variants)
    for index, (case, variant) in enumerate(
        ((case, variant) for variant in variants for case in CASES),
        1,
    ):
        print(f"[{index}/{total}] {variant['name']} / {case['id']}", flush=True)
        record = run_case(case, variant, common_regexes)
        records.append(record)
        print(
            f"  {'correct' if record['correct'] else 'wrong'}: "
            f"{record['final_template']!r}",
            flush=True,
        )
        args.output.write_text(
            json.dumps(
                {"summary": summarize(records), "records": records},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summarize(records), ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
