#!/usr/bin/env python3
"""Test one generalized key-value prompt change without editing gpt_query.py."""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import regex


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

from logparser.KnowledgeLog import gpt_query as gq  # noqa: E402


MODEL = "Llama-3.3-70B-Instruct"
LOCAL_API_BASE = "http://127.0.0.1:8001/v1"
LOCAL_API_KEY = "EMPTY"
SAMPLES_PER_TEMPLATE = 10
CURRENT_RULE = (
    "In a key-value field, preserve a fixed key and the '=' or ':' delimiter "
    "and replace only its variable value with {variables}. If the key also "
    "varies, replace the key and value separately while preserving the delimiter."
)
CANDIDATE_RULE = (
    "In a sequence of key-value fields, preserve every key, delimiter, separator, "
    "field, and original order; never omit or merge fields. Replace only each "
    "variable value with {variables}; if a key also varies, abstract the key and "
    "value separately."
)
TARGET_TEMPLATES = (
    "bufstart = <*>; bufend = <*>; bufvoid = <*>",
    "kvstart = <*>(<*>); kvend = <*>(<*>); length = <*>",
)
REGRESSION_TEMPLATES = (
    "bufstart = <*>; bufvoid = <*>",
    "kvstart = <*>; length = <*>",
)


def common_regexes():
    path = ROOT / "benchmark/logparser/KnowledgeLog/common.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [regex.compile(pattern) for pattern in data["COMMON"]["regex"]]


def select_samples(frame, template):
    rows = frame.loc[frame["EventTemplate"] == template, "Content"].drop_duplicates()
    positions = np.linspace(0, len(rows) - 1, min(SAMPLES_PER_TEMPLATE, len(rows)), dtype=int)
    return [rows.iloc[position] for position in positions]


def parse(log, expected, prompt_rules, preprocessing, regs):
    prompt_group = "_hadoop_key_value_prompt_test"
    preprocessing_group = "_hadoop_key_value_preprocessing_test"
    gq.PROMPT_GROUPS[prompt_group] = prompt_rules
    gq.PREPROCESSING_RULE_GROUPS[preprocessing_group] = preprocessing
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
        "log": log,
        "expected": expected,
        "preprocessed": gq.apply_final_domain_preprocessing(
            log,
            enabled_groups=(preprocessing_group,),
        ),
        "response": response,
        "final": final,
        "valid": valid,
        "correct": bool(valid and final == expected),
    }


def main():
    gq.openai.api_base = LOCAL_API_BASE
    gq.openai.api_key = LOCAL_API_KEY
    gq.configure_local_api_no_proxy(LOCAL_API_BASE)
    dataset = pd.read_csv(
        ROOT / "full_dataset/Hadoop/Hadoop_full.log_structured.csv"
    )
    preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    current_rules = list(gq.CURATED_PROMPT_RULES)
    rule_index = current_rules.index(CURRENT_RULE)
    candidate_rules = list(current_rules)
    candidate_rules[rule_index] = CANDIDATE_RULE
    variants = {
        "current_prompt": current_rules,
        "candidate_prompt": candidate_rules,
    }
    regs = common_regexes()
    records = []

    for template in (*TARGET_TEMPLATES, *REGRESSION_TEMPLATES):
        for sample_index, log in enumerate(select_samples(dataset, template), 1):
            for variant, prompt_rules in variants.items():
                result = parse(log, template, prompt_rules, preprocessing, regs)
                result.update(
                    variant=variant,
                    template=template,
                    sample_index=sample_index,
                    target=template in TARGET_TEMPLATES,
                )
                records.append(result)
                print(
                    f"{variant:16s} sample {sample_index:02d} "
                    f"{'CORRECT' if result['correct'] else 'wrong':7s} "
                    f"{result['final']!r}",
                    flush=True,
                )

    output_dir = (
        ROOT
        / "others/test_one_log/results/prompt_validation"
        / f"hadoop_key_value_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {}
    for variant in variants:
        summary[variant] = {}
        for template in (*TARGET_TEMPLATES, *REGRESSION_TEMPLATES):
            selected = [
                record
                for record in records
                if record["variant"] == variant and record["template"] == template
            ]
            summary[variant][template] = {
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
