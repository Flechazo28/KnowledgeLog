#!/usr/bin/env python3
"""Evaluate Linux full data with candidate rules without editing gpt_query.py."""

import json
import os
import runpy
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmark"
EVALUATION = BENCHMARK / "evaluation"
TEST_HELPERS = ROOT / "others" / "test_one_log"
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(TEST_HELPERS))

from logparser.KnowledgeLog import gpt_query as gq  # noqa: E402
from run_linux_solution_validation import CANDIDATE_REGEXES  # noqa: E402


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        ROOT / "result" / f"KnowledgeLog_full_Linux_candidate_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    candidate_names = {rule.name for rule in CANDIDATE_REGEXES}
    gq.PREPROCESSING_RULE_GROUPS["curated"] = list(CANDIDATE_REGEXES) + [
        rule for rule in existing if rule.name not in candidate_names
    ]

    (output_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "dataset": "Linux_full",
                "model": "Llama-3.3-70B-Instruct",
                "candidate_preprocessing_rules": [
                    rule.name for rule in CANDIDATE_REGEXES
                ],
                "production_gpt_query_modified": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    os.chdir(EVALUATION)
    sys.argv = [
        "KnowledgeLog_eval.py",
        "--full_data",
        "--datasets",
        "Linux",
        "--model",
        "Llama-3.3-70B-Instruct",
        "--output_dir",
        str(output_dir),
    ]
    print(f"Candidate result directory: {output_dir}", flush=True)
    runpy.run_path("KnowledgeLog_eval.py", run_name="__main__")


if __name__ == "__main__":
    main()
