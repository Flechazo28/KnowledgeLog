#!/usr/bin/env python3
"""Estimate metric changes if selected, experimentally tested errors are corrected."""

import json
from collections import Counter
from pathlib import Path

from analyze_metric_losses import analyze, pair_counts


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "result" / "KnowledgeLog_full_14datasets_20260730_200403"
OUTPUT = Path(__file__).with_name("targeted_correction_estimates.json")


TARGETS = {
    "OpenSSH": {
        "failed_password_boundaries": {
            "Failed password for <*> from <*> port <*> ssh2",
            "Failed password for invalid user <*> from <*> port <*> ssh2",
        },
        "pam_wrapper": {
            "pam_unix(sshd:auth): check pass; user unknown",
        },
        "disconnect_reasons": {
            "Received disconnect from <*>: <*>: Normal Shutdown, Thank you for playing [preauth]",
            "Received disconnect from <*>: <*>: Client disconnecting normally [preauth]",
            "Received disconnect from <*>: <*>: disconnect [preauth]",
        },
    },
    "HDFS": {
        "complete_paths": {
            "BLOCK* NameSystem.allocateBlock: <*> <*>",
            "Deleting block <*> file <*>",
        },
    },
    "HPC": {
        "fixed_available_state": {
            "Component State Change: Component <*> is in the available state (HWID=<*>)",
        },
        "annotation_only_suffix": {
            "ambient=<*>",
        },
    },
    "Hadoop": {
        "top_recoverable_boundaries": {
            "bufstart = <*>; bufend = <*>; bufvoid = <*>",
            "kvstart = <*>(<*>); kvend = <*>(<*>); length = <*>",
            "Failed to renew lease for <*> for <*> seconds. Will retry shortly ...",
            "Resolved <*> to <*>",
            "<*> TaskAttempt Transitioned from <*> to <*>",
            "Kind: <*>, Service: <*>, Ident: (<*>)",
        },
    },
    "Thunderbird": {
        "guid_and_prefix_fields": {
            "[<*>]: Set port GUID=<*>, port=<*> DOWN (port prefix <*>, subnet_prefix <*>",
        },
        "high_impact_boundaries": {
            "session opened for user <*> by <*>(uid=<*>)",
            "(<*>) CMD (<*>)",
            "[<*>]: Failed discover node test, node <*>, port_num= <*>, error code <*>",
            "[<*>]: IB node <*> port <*> is INIT state",
            "[INFO]: Configuration caused by some ports in INIT state",
            "Instrumentation Service EventID: <*> Temperature sensor detected a warning value Sensor location: <*> Chassis location: <*> Previous state was: <*> (Normal) Temperature sensor value (in Degrees Celsius): <*>",
        },
    },
}


METRICS = ("pa", "ga", "fga", "pta", "rta")


def correct_truths(counts, truths):
    corrected = Counter()
    for (truth, prediction), count in counts.items():
        corrected[(truth, truth if truth in truths else prediction)] += count
    return corrected


def metric_view(result):
    return {
        **{name: result[name] for name in METRICS},
        "predicted_templates": result["predicted_templates"],
        "accurate_group_templates": result["accurate_group_templates"],
        "correct_template_names": result["correct_template_names"],
    }


def main():
    output = {}
    for dataset, scenarios in TARGETS.items():
        print(f"Simulating {dataset}", flush=True)
        counts = pair_counts(RESULT_DIR, dataset, 500_000)
        baseline = analyze(counts)
        dataset_output = {"baseline": metric_view(baseline), "scenarios": {}}
        cumulative = set()
        for name, truths in scenarios.items():
            cumulative.update(truths)
            scenario = analyze(correct_truths(counts, truths))
            combined = analyze(correct_truths(counts, cumulative))
            dataset_output["scenarios"][name] = {
                "isolated": metric_view(scenario),
                "cumulative": metric_view(combined),
            }
        output[dataset] = dataset_output
    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
