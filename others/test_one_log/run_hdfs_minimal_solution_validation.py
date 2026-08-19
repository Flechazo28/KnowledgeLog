#!/usr/bin/env python3
"""Validate two small, general preprocessing changes for HDFS errors."""

import json
import re
from collections import Counter
from datetime import datetime

from run_webmaster_rule_ablation import ROOT, common_regexes, gq, parse


REPETITIONS = 5

IPV4 = r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)"
PORT = r"(?:[1-9]\d{0,3}|[1-5]\d{4}|6[0-4]\d{3}|65[0-4]\d{2}|655[0-2]\d|6553[0-5])"
ENDPOINT = rf"/?{IPV4}:{PORT}"

IPV4_ENDPOINT_OR_LIST = gq.RegexRule(
    name="curated_0003_ipv4_port",
    pattern=rf"(?<![A-Za-z0-9_/]){ENDPOINT}(?:\s+{ENDPOINT})*\b",
    replacement=gq.VARIABLE_PLACEHOLDER,
    note=(
        "Abstract one IPv4 endpoint or a whitespace-separated list of IPv4 "
        "endpoints as one field."
    ),
)

MULTI_COMPONENT_ABSOLUTE_PATH = gq.RegexRule(
    name="multi_component_absolute_path",
    pattern=r"(?<![A-Za-z0-9_.-])/(?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+/?",
    replacement=gq.VARIABLE_PLACEHOLDER,
    note="Abstract a complete multi-component absolute filesystem path.",
)

CASES = {
    "allocate_path_and_block": (
        "BLOCK* NameSystem.allocateBlock: /mnt/hadoop/mapred/system/"
        "job_200811092030_0001/job.jar. blk_-1608999687919862906",
        "BLOCK* NameSystem.allocateBlock: <*> <*>",
    ),
    "delete_block_file": (
        "Deleting block blk_-919439116365725304 file "
        "/mnt/hadoop/dfs/data/current/subdir30/blk_-919439116365725304",
        "Deleting block <*> file <*>",
    ),
    "endpoint_list": (
        "BLOCK* ask 10.250.14.224:50010 to replicate "
        "blk_-1608999687919862906 to datanode(s) "
        "10.251.215.16:50010 10.251.71.193:50010",
        "BLOCK* ask <*> to replicate <*> to datanode(s) <*>",
    ),
    "separate_labeled_endpoints": (
        "Receiving block blk_-1608999687919862906 "
        "src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
        "Receiving block <*> src: <*> dest: <*>",
    ),
    "single_source_address": (
        "Received block blk_4471756998729740136 of size 28503973 "
        "from /10.251.42.16",
        "Received block <*> of size <*> from <*>",
    ),
    "apache_path_regression": (
        "[client 61.152.90.96] File does not exist: /var/www/html/wordpress",
        "[client <*>] File does not exist: <*>",
    ),
}


def candidate_preprocessing():
    rules = []
    for rule in gq.PREPROCESSING_RULE_GROUPS["curated"]:
        rules.append(
            IPV4_ENDPOINT_OR_LIST
            if rule.name == "curated_0003_ipv4_port"
            else rule
        )
    rules.append(MULTI_COMPONENT_ABSOLUTE_PATH)
    return rules


def main():
    prompt_rules = list(gq.CURATED_PROMPT_RULES)
    variants = {
        "current": list(gq.PREPROCESSING_RULE_GROUPS["curated"]),
        "candidate": candidate_preprocessing(),
    }
    regs = common_regexes()
    records = []

    for variant, preprocessing in variants.items():
        for case, (log_message, expected) in CASES.items():
            for repetition in range(1, REPETITIONS + 1):
                record = parse(
                    log_message,
                    expected,
                    prompt_rules,
                    preprocessing,
                    regs,
                )
                record.update(
                    variant=variant,
                    case=case,
                    repetition=repetition,
                )
                records.append(record)
                print(
                    f"{variant:9s} {case:28s} {repetition}/{REPETITIONS}: "
                    f"{'CORRECT' if record['correct'] else 'wrong'} "
                    f"{record['final']!r}",
                    flush=True,
                )

    output_dir = (
        ROOT
        / "others/test_one_log/results/regex_validation"
        / f"hdfs_minimal_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
