#!/usr/bin/env python3
"""Ablate old prompt rules that preserve authentication constants."""

import json
from collections import Counter
from datetime import datetime

from run_oldgptquery_comparison import old_prompt_rules
from run_webmaster_rule_ablation import ROOT, common_regexes, gq, parse


REPETITIONS = 3
OLD_MARKER_RULE = (
    "Preserve fixed protocol, authentication, service, and bracketed state "
    "markers literally, such as sshd, ssh2, [preauth], and [state]."
)
OLD_COMPONENT_RULE = (
    "Preserve fixed component and service names literally when they are complete "
    "message constants, including lowerCamelCase names such as workerEnv, "
    "CamelCase names, method-like names, and fixed names inside grammar wrappers "
    "such as service(...)."
)
CANDIDATE_RULE = (
    "Preserve fixed authentication module, service, and context names inside "
    "wrappers, such as pam_unix(sshd:auth); punctuation inside the wrapper is not "
    "evidence that the text is variable."
)

CASES = {
    "pam_fixed": (
        "pam_unix(sshd:auth): check pass; user unknown",
        "pam_unix(sshd:auth): check pass; user unknown",
    ),
    "pam_dynamic_fields": (
        "pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 "
        "tty=ssh ruser= rhost=183.192.189.131 user=root",
        "pam_unix(sshd:auth): authentication failure; logname=<*> uid=<*> "
        "euid=<*> tty=<*> ruser=<*> rhost=<*> user=<*>",
    ),
    "failed_password": (
        "Failed password for root from 183.62.140.253 port 60221 ssh2",
        "Failed password for <*> from <*> port <*> ssh2",
    ),
}


def main():
    current = list(gq.CURATED_PROMPT_RULES)
    preprocessing = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    variants = {
        "current": current,
        "old_full_prompt": old_prompt_rules(),
        "add_old_marker": [OLD_MARKER_RULE] + current,
        "add_old_component": [OLD_COMPONENT_RULE] + current,
        "add_both_old_rules": [OLD_MARKER_RULE, OLD_COMPONENT_RULE] + current,
        "add_concise_candidate": [CANDIDATE_RULE] + current,
    }
    regs = common_regexes()
    records = []
    for case, (log, expected) in CASES.items():
        for variant, prompt in variants.items():
            for repetition in range(1, REPETITIONS + 1):
                result = parse(log, expected, prompt, preprocessing, regs)
                result.update(
                    case=case,
                    variant=variant,
                    repetition=repetition,
                )
                records.append(result)
                print(
                    f"{case:20s} {variant:22s} {repetition}/{REPETITIONS}: "
                    f"{'CORRECT' if result['correct'] else 'wrong'} "
                    f"{result['final']!r}",
                    flush=True,
                )

    output_dir = (
        ROOT
        / "others/test_one_log/results/domain_rule_ablation"
        / f"old_prompt_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {}
    for case in CASES:
        summary[case] = {}
        for variant in variants:
            selected = [
                item
                for item in records
                if item["case"] == case and item["variant"] == variant
            ]
            summary[case][variant] = {
                "correct": sum(item["correct"] for item in selected),
                "runs": len(selected),
                "templates": dict(Counter(item["final"] for item in selected)),
            }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
