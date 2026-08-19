import argparse
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "benchmark"))

from evaluation.settings import benchmark_settings
from logparser.KnowledgeLog import LogParser


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "result"
    / "KnowledgeLog_2k_curated50_llama70b_20260728_161000"
    / "misparsed_logs.csv"
)


def event_id(template: str) -> str:
    return hashlib.md5(template.encode("utf-8")).hexdigest()[:8]


def prepare_input(group: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "LineId": range(1, len(group) + 1),
            "Content": group["Content"].tolist(),
            "EventId": group["ground_truth_template"].map(event_id),
            "EventTemplate": group["ground_truth_template"].tolist(),
        }
    )
    frame.to_csv(path, index=False)
    path.with_name(path.name[: -len("_structured.csv")]).write_text(
        "\n".join(frame["Content"]) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="Llama-3.3-70B-Instruct")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        PROJECT_ROOT / "result" / "failure_retest" / f"{run_id}_previous_failures"
    )
    output_dir = output_dir.resolve()
    input_dir = output_dir / "input"
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    required = {
        "dataset",
        "LineId",
        "Content",
        "ground_truth_template",
        "parsed_template",
    }
    missing = required - set(failures.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")

    records = []
    original_cwd = Path.cwd()
    os.chdir(PROJECT_ROOT / "benchmark" / "evaluation")
    try:
        for dataset, group in failures.groupby("dataset", sort=True):
            group = group.reset_index(drop=True)
            log_name = f"{dataset}_failed.log"
            staged_path = input_dir / dataset / f"{log_name}_structured.csv"
            prepare_input(group, staged_path)

            parsed_path = output_dir / f"{log_name}_structured.csv"
            if not parsed_path.exists():
                print(f"\n=== Retesting {dataset}: {len(group)} logs ===")
                LogParser(
                    log_format=benchmark_settings[dataset]["log_format"],
                    indir=str(staged_path.parent),
                    outdir=str(output_dir),
                    rex=[],
                    data_type="failure_retest",
                    model=args.model,
                    run_id=run_id,
                ).parse(log_name)

            parsed = pd.read_csv(parsed_path, dtype=str, keep_default_na=False)
            if len(parsed) != len(group):
                raise RuntimeError(
                    f"{dataset}: expected {len(group)} results, got {len(parsed)}"
                )

            current = group.rename(
                columns={
                    "LineId": "original_line_id",
                    "parsed_template": "previous_parsed_template",
                }
            ).copy()
            current["current_parsed_template"] = parsed["EventTemplate"].tolist()
            current["corrected"] = (
                current["current_parsed_template"]
                == current["ground_truth_template"]
            )
            records.append(current)
    finally:
        os.chdir(original_cwd)

    results = pd.concat(records, ignore_index=True)
    summary = (
        results.groupby("dataset")["corrected"]
        .agg(total_logs="size", correct_logs="sum")
        .reset_index()
    )
    summary["remaining_errors"] = summary["total_logs"] - summary["correct_logs"]
    summary["PA"] = summary["correct_logs"] / summary["total_logs"]
    summary.loc[len(summary)] = {
        "dataset": "ALL",
        "total_logs": len(results),
        "correct_logs": int(results["corrected"].sum()),
        "remaining_errors": int((~results["corrected"]).sum()),
        "PA": float(results["corrected"].mean()),
    }

    summary.to_csv(output_dir / "retest_summary.csv", index=False)
    results[results["corrected"]].to_csv(
        output_dir / "corrected_logs.csv", index=False
    )
    results[~results["corrected"]].to_csv(
        output_dir / "remaining_misparsed_logs.csv", index=False
    )
    print("\n" + summary.to_string(index=False))
    print(f"\nResults: {output_dir}")


if __name__ == "__main__":
    main()
