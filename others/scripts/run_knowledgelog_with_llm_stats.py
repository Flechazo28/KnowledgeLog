#!/usr/bin/env python3
"""Run KnowledgeLog and summarize per-dataset runtime plus LLM calls.

This wrapper keeps the original KnowledgeLog evaluation outputs intact. It runs
benchmark/evaluation/KnowledgeLog_eval.py one dataset at a time, captures stdout/stderr
to run.log, reads the normal summary CSV, and adds two statistics that are not
written by KnowledgeLog_eval.py itself:

- logical LLM calls: count of "model:" lines printed by gpt_query.infer_llm
- cache-miss queries: final "Total query:" or "Query times:" value printed by
  KnowledgeLog.py

The two counts are usually the same, except for edge cases where the parser
handles a cache miss without entering infer_llm.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "benchmark" / "evaluation"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "result" / "result_datasets"

FULL_DATASETS = [
    "Proxifier",
    "Apache",
    "OpenSSH",
    "HDFS",
    "OpenStack",
    "HPC",
    "Zookeeper",
    "HealthApp",
    "Hadoop",
    "Spark",
    "BGL",
    "Linux",
    "Mac",
    "Thunderbird",
]

LUNAR_DATASETS = [
    "CTS",
    "HiBench",
    "LoFI",
]

# LILAC values from the published comparison table.  Values are fractions
# here, matching the metrics written by the evaluation script.
LILAC_BASELINE = {
    "Proxifier": {"GA": 1.000, "PA": 1.000, "FGA": 1.000, "FTA": 1.000},
    "HDFS": {"GA": 1.000, "PA": 0.999, "FGA": 0.968, "FTA": 0.946},
    "OpenStack": {"GA": 1.000, "PA": 1.000, "FGA": 1.000, "FTA": 0.979},
    "Spark": {"GA": 1.000, "PA": 0.973, "FGA": 0.901, "FTA": 0.759},
    "Zookeeper": {"GA": 1.000, "PA": 0.687, "FGA": 0.967, "FTA": 0.868},
    "BGL": {"GA": 0.894, "PA": 0.958, "FGA": 0.859, "FTA": 0.746},
    "HPC": {"GA": 0.869, "PA": 0.705, "FGA": 0.907, "FTA": 0.800},
    "Thunderbird": {"GA": 0.806, "PA": 0.559, "FGA": 0.793, "FTA": 0.572},
    "Linux": {"GA": 0.971, "PA": 0.765, "FGA": 0.931, "FTA": 0.740},
    "Mac": {"GA": 0.876, "PA": 0.638, "FGA": 0.825, "FTA": 0.553},
    "Apache": {"GA": 1.000, "PA": 0.996, "FGA": 1.000, "FTA": 0.862},
    "OpenSSH": {"GA": 0.690, "PA": 0.941, "FGA": 0.838, "FTA": 0.865},
    "HealthApp": {"GA": 1.000, "PA": 0.729, "FGA": 0.981, "FTA": 0.873},
    "Hadoop": {"GA": 0.872, "PA": 0.832, "FGA": 0.962, "FTA": 0.779},
}

METRIC_FIELDS = [
    "identified_templates",
    "ground_templates",
    "GA",
    "PA",
    "FGA",
    "PTA",
    "RTA",
    "FTA",
]

PER_RUN_FIELDS = [
    "repetition",
    "dataset",
    "status",
    "returncode",
    "parse_time_seconds",
    "llm_logical_calls",
    "cache_miss_queries",
    "api_attempts_estimated",
    "api_failed_attempts",
    "api_failed_queries",
    *METRIC_FIELDS,
    "result_dir",
    "log_path",
]

AGGREGATE_FIELDS = [
    "dataset",
    "runs",
    "successful_runs",
    "mean_parse_time_seconds",
    "mean_llm_logical_calls",
    "mean_cache_miss_queries",
    "mean_api_attempts_estimated",
    "mean_api_failed_attempts",
    "mean_api_failed_queries",
    "mean_GA",
    "mean_PA",
    "mean_FGA",
    "mean_PTA",
    "mean_RTA",
    "mean_FTA",
]

OVERALL_FIELDS = [
    "scope",
    "datasets",
    "successful_datasets",
    "mean_parse_time_seconds",
    "mean_llm_logical_calls",
    "mean_cache_miss_queries",
    "mean_api_attempts_estimated",
    "mean_api_failed_attempts",
    "mean_api_failed_queries",
    "mean_GA",
    "mean_PA",
    "mean_FGA",
    "mean_PTA",
    "mean_RTA",
    "mean_FTA",
]


def split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    items: List[str] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            items.append(part)
    return items


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def mean(values: Iterable[object]) -> Optional[float]:
    numbers = [parse_float(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    if not numbers:
        return None
    return statistics.mean(numbers)


def find_summary_file(result_dir: Path) -> Optional[Path]:
    candidates = sorted(result_dir.glob("summary_[[]*.csv"))
    if not candidates:
        candidates = sorted(result_dir.glob("summary_*.csv"))
    return candidates[0] if candidates else None


def read_dataset_summary(result_dir: Path, dataset: str) -> Dict[str, str]:
    summary_file = find_summary_file(result_dir)
    if not summary_file:
        return {}
    with summary_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for row in rows:
        if row.get("Dataset") == dataset:
            return row
    return rows[0] if rows else {}


def parse_log_counts(log_text: str) -> Dict[str, int]:
    logical_calls = len(re.findall(r"^model:\s+", log_text, flags=re.MULTILINE))
    query_times = [
        int(value)
        for value in re.findall(r"Query times:\s+(\d+)", log_text)
    ]
    total_queries = [
        int(value)
        for value in re.findall(r"Total query:\s+(\d+)", log_text)
    ]
    cache_miss_queries = 0
    if total_queries:
        cache_miss_queries = total_queries[-1]
    elif query_times:
        cache_miss_queries = query_times[-1]

    failed_attempts = len(re.findall(r"Exception on API attempt\s+\d+/\d+:", log_text))
    failed_queries = len(re.findall(r"Failed to get response from OpenAI after", log_text))
    successful_logical_calls = max(logical_calls - failed_queries, 0)
    api_attempts_estimated = failed_attempts + successful_logical_calls
    return {
        "llm_logical_calls": logical_calls,
        "cache_miss_queries": cache_miss_queries,
        "api_attempts_estimated": api_attempts_estimated,
        "api_failed_attempts": failed_attempts,
        "api_failed_queries": failed_queries,
    }


def run_command(command: List[str], cwd: Path, log_path: Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return process.wait()


def dataset_list(args: argparse.Namespace) -> List[str]:
    requested = split_csv(args.datasets)
    if requested:
        return requested
    if args.data_type == "lunar":
        return list(LUNAR_DATASETS)
    return list(FULL_DATASETS)


def build_command(args: argparse.Namespace, dataset: str, result_dir: Path) -> List[str]:
    command = [
        args.python,
        "-u",
        "KnowledgeLog_eval.py",
    ]
    if args.data_type == "lunar":
        command.append("--lunar_data")
    else:
        command.append("--full_data")
    command.extend([
        "--model",
        args.model,
        "--datasets",
        dataset,
        "--output_dir",
        str(result_dir),
    ])
    command.extend(args.extra_eval_args)
    return command


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def aggregate(rows: List[Dict[str, object]], datasets: List[str]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for dataset in datasets:
        current = [row for row in rows if row.get("dataset") == dataset]
        if not current:
            continue
        successful = [row for row in current if row.get("status") == "ok"]
        row = {
            "dataset": dataset,
            "runs": len(current),
            "successful_runs": len(successful),
            "mean_parse_time_seconds": fmt(mean(row.get("parse_time_seconds") for row in current)),
            "mean_llm_logical_calls": fmt(mean(row.get("llm_logical_calls") for row in current), 2),
            "mean_cache_miss_queries": fmt(mean(row.get("cache_miss_queries") for row in current), 2),
            "mean_api_attempts_estimated": fmt(mean(row.get("api_attempts_estimated") for row in current), 2),
            "mean_api_failed_attempts": fmt(mean(row.get("api_failed_attempts") for row in current), 2),
            "mean_api_failed_queries": fmt(mean(row.get("api_failed_queries") for row in current), 2),
        }
        for metric in ["GA", "PA", "FGA", "PTA", "RTA", "FTA"]:
            row[f"mean_{metric}"] = fmt(mean(item.get(metric) for item in current), 3)
        output.append(row)
    return output


def aggregate_overall(dataset_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not dataset_rows:
        return []
    row: Dict[str, object] = {
        "scope": "average_across_datasets",
        "datasets": len(dataset_rows),
        "successful_datasets": sum(
            1 for item in dataset_rows if parse_float(item.get("successful_runs")) and parse_float(item.get("successful_runs")) > 0
        ),
        "mean_parse_time_seconds": fmt(mean(item.get("mean_parse_time_seconds") for item in dataset_rows)),
        "mean_llm_logical_calls": fmt(mean(item.get("mean_llm_logical_calls") for item in dataset_rows), 2),
        "mean_cache_miss_queries": fmt(mean(item.get("mean_cache_miss_queries") for item in dataset_rows), 2),
        "mean_api_attempts_estimated": fmt(mean(item.get("mean_api_attempts_estimated") for item in dataset_rows), 2),
        "mean_api_failed_attempts": fmt(mean(item.get("mean_api_failed_attempts") for item in dataset_rows), 2),
        "mean_api_failed_queries": fmt(mean(item.get("mean_api_failed_queries") for item in dataset_rows), 2),
    }
    for metric in ["GA", "PA", "FGA", "PTA", "RTA", "FTA"]:
        row[f"mean_{metric}"] = fmt(mean(item.get(f"mean_{metric}") for item in dataset_rows), 3)
    return [row]


def markdown_cell(value: object) -> str:
    text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\n", " ").replace("`", "\\`")


def lower_metrics(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    output = []
    for row in rows:
        dataset = str(row.get("dataset", ""))
        baseline = LILAC_BASELINE.get(dataset)
        if not baseline:
            continue
        for metric in ["GA", "PA", "FGA", "FTA"]:
            current = parse_float(row.get(f"mean_{metric}"))
            reference = baseline[metric]
            if current is not None and current < reference - 1e-9:
                output.append({
                    "dataset": dataset,
                    "metric": metric,
                    "current": current,
                    "baseline": reference,
                    "difference": (reference - current) * 100,
                })
    return output


def top_errors_for_dataset(output_root: Path, dataset: str, data_type: str) -> List[Dict[str, object]]:
    if data_type != "full":
        return []
    predicted_path = output_root / "rep_01" / dataset / "result" / f"{dataset}_full.log_structured.csv"
    truth_path = PROJECT_ROOT / "full_dataset" / dataset / f"{dataset}_full.log_structured.csv"
    if not predicted_path.exists() or not truth_path.exists():
        return []
    # Keep report generation responsive for multi-million-line datasets.
    # Their full error ranking should be generated by a dedicated offline job.
    if predicted_path.stat().st_size > 200 * 1024 * 1024:
        return []
    # Count by parsed/true template pair and retain one representative log.
    # Including every raw log in the key is prohibitively large for BGL.
    # Misra-Gries keeps memory bounded even when every raw log is unique.
    candidate_limit = 2048
    candidates: Dict[tuple, int] = {}
    total_errors = 0
    with truth_path.open("r", encoding="utf-8", newline="") as truth_handle, \
            predicted_path.open("r", encoding="utf-8", newline="") as predicted_handle:
        for truth, predicted in zip(csv.DictReader(truth_handle), csv.DictReader(predicted_handle)):
            true_template = truth.get("EventTemplate", "")
            parsed_template = predicted.get("EventTemplate", "")
            if parsed_template == true_template:
                continue
            total_errors += 1
            key = (parsed_template, true_template)
            if key in candidates:
                candidates[key] += 1
            elif len(candidates) < candidate_limit:
                candidates[key] = 1
            else:
                for candidate in list(candidates):
                    candidates[candidate] -= 1
                    if candidates[candidate] <= 0:
                        del candidates[candidate]

    exact_counts: Dict[tuple, int] = {key: 0 for key in candidates}
    representatives: Dict[tuple, str] = {}
    if exact_counts:
        with truth_path.open("r", encoding="utf-8", newline="") as truth_handle, \
                predicted_path.open("r", encoding="utf-8", newline="") as predicted_handle:
            for truth, predicted in zip(csv.DictReader(truth_handle), csv.DictReader(predicted_handle)):
                key = (predicted.get("EventTemplate", ""), truth.get("EventTemplate", ""))
                if key in exact_counts:
                    exact_counts[key] += 1
                    representatives.setdefault(key, truth.get("Content", ""))
    ranked = sorted(exact_counts.items(), key=lambda item: (-item[1], item[0]))[:15]
    return [
        {
            "count": count,
            "percentage": (count / total_errors * 100) if total_errors else 0.0,
            "original": representatives.get(key, ""),
            "parsed": key[0],
            "truth": key[1],
        }
        for key, count in ranked
    ]


def write_markdown(
    path: Path,
    rows: List[Dict[str, object]],
    overall_rows: List[Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# KnowledgeLog Runtime and LLM Call Statistics",
        "",
        f"- Created at: `{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        f"- Model: `{args.model}`",
        f"- Data type: `{args.data_type}`",
        f"- Repetitions: `{args.repetitions}`",
        "",
        "## Columns",
        "",
        "- `mean_parse_time_seconds`: `parse_time` reported by KnowledgeLog's normal summary CSV. It does not include the later GA/PA/FGA/PTA/RTA/FTA metric computation time.",
        "- `mean_llm_logical_calls`: count of logical LLM calls, measured from `model:` lines printed by `gpt_query.py`.",
        "- `mean_cache_miss_queries`: final `Total query` or `Query times` value printed by `KnowledgeLog.py`.",
        "- `mean_api_attempts_estimated`: failed API attempts plus one successful attempt for each successful logical LLM call.",
        "- Overall averages are macro averages across datasets: each dataset contributes one value, regardless of its log-line count.",
        "",
        "## Average By Dataset",
        "",
        "| Dataset | GA | PA | FGA | FTA | PTA | RTA | Parse time | LLM calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {mean_GA} | {mean_PA} | {mean_FGA} | {mean_FTA} | "
            "{mean_PTA} | {mean_RTA} | {mean_parse_time_seconds} | {mean_llm_logical_calls} |".format(**row)
        )
    if overall_rows:
        overall = overall_rows[0]
        lines.extend(
            [
                "",
                "## Overall Summary",
                "",
                "| Scope | Datasets | GA | PA | FGA | PTA | RTA |",
                "|---|---:|---:|---:|---:|---:|---:|",
                (
                    "| {scope} | {datasets} | {mean_GA} | {mean_PA} | "
                    "{mean_FGA} | {mean_PTA} | {mean_RTA} |"
                ).format(**overall),
            ]
        )
        lines.extend([
            "",
            "## Overall Average Across Datasets",
            "",
            "| GA | PA | FGA | FTA |",
            "|---:|---:|---:|---:|",
            "| {mean_GA} | {mean_PA} | {mean_FGA} | {mean_FTA} |".format(**overall),
        ])
    comparisons = lower_metrics(rows)
    lines.extend(["", "## Metrics Lower Than LILAC", ""])
    if comparisons:
        lines.extend([
            "| Dataset | Metric | KnowledgeLog | LILAC | Lower by (percentage points) |",
            "|---|---|---:|---:|---:|",
        ])
        for item in comparisons:
            lines.append(
                f"| {item['dataset']} | {item['metric']} | {item['current'] * 100:.1f} | "
                f"{item['baseline'] * 100:.1f} | {item['difference']:.1f} |"
            )
        lines.extend(["", "## Top 15 Errors in Datasets Below LILAC", ""])
        for dataset in sorted({item["dataset"] for item in comparisons}):
            errors = top_errors_for_dataset(path.parent, dataset, args.data_type)
            lines.extend([f"### {dataset}", ""])
            if not errors:
                lines.append("No comparable errors found in the representative run.")
                lines.append("")
                continue
            lines.extend([
                "| Rank | Error count | Error percentage | Original log | LLM parsed template | True template |",
                "|---:|---:|---:|---|---|---|",
            ])
            for rank, error in enumerate(errors, 1):
                lines.append(
                    f"| {rank} | {error['count']} | {error['percentage']:.2f}% | "
                    f"{markdown_cell(error['original'])} | {markdown_cell(error['parsed'])} | "
                    f"{markdown_cell(error['truth'])} |"
                )
            lines.append("")
    else:
        lines.append("No GA, PA, FGA, or FTA value is lower than the LILAC baseline.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KnowledgeLog and summarize average runtime plus LLM call counts per dataset."
    )
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--data-type", choices=["full", "lunar"], default="full")
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset list. Defaults to all datasets for --data-type.",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Run output root. Defaults to result/result_datasets/TIMESTAMP_Ndatasets.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining datasets if one run exits with a non-zero status.",
    )
    parser.add_argument(
        "extra_eval_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to KnowledgeLog_eval.py after a literal --.",
    )
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")
    if args.extra_eval_args and args.extra_eval_args[0] == "--":
        args.extra_eval_args = args.extra_eval_args[1:]
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    datasets = dataset_list(args)
    if args.output_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_root = DEFAULT_OUTPUT_ROOT / f"{timestamp}_{len(datasets)}datasets"
    args.output_root.mkdir(parents=True, exist_ok=True)

    config = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "evaluation_dir": str(EVAL_DIR),
        "python": args.python,
        "model": args.model,
        "data_type": args.data_type,
        "datasets": datasets,
        "repetitions": args.repetitions,
        "extra_eval_args": args.extra_eval_args,
    }
    (args.output_root / "configuration.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows: List[Dict[str, object]] = []
    env = os.environ.copy()
    for repetition in range(1, args.repetitions + 1):
        for dataset in datasets:
            run_dir = args.output_root / f"rep_{repetition:02d}" / dataset
            result_dir = run_dir / "result"
            log_path = run_dir / "run.log"
            command = build_command(args, dataset, result_dir)
            print("=" * 72)
            print(f"[rep {repetition}/{args.repetitions}] dataset={dataset}")
            print(f"output={result_dir}")
            print("command=" + " ".join(command))
            print("=" * 72)
            returncode = run_command(command, EVAL_DIR, log_path, env)

            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            counts = parse_log_counts(log_text)
            summary = read_dataset_summary(result_dir, dataset)
            status = "ok" if returncode == 0 and summary else "failed"
            row: Dict[str, object] = {
                "repetition": repetition,
                "dataset": dataset,
                "status": status,
                "returncode": returncode,
                "parse_time_seconds": summary.get("parse_time", ""),
                "result_dir": str(result_dir),
                "log_path": str(log_path),
                **counts,
            }
            for field in METRIC_FIELDS:
                row[field] = summary.get(field, "")
            rows.append(row)
            write_csv(args.output_root / "per_run_stats.csv", rows, PER_RUN_FIELDS)
            aggregate_rows = aggregate(rows, datasets)
            overall_rows = aggregate_overall(aggregate_rows)
            write_csv(args.output_root / "average_stats_by_dataset.csv", aggregate_rows, AGGREGATE_FIELDS)
            write_csv(args.output_root / "overall_average_across_datasets.csv", overall_rows, OVERALL_FIELDS)
            write_markdown(args.output_root / "average_stats_by_dataset.md", aggregate_rows, overall_rows, args)

            if returncode != 0 and not args.continue_on_error:
                print(f"Run failed with exit code {returncode}. Stopping.")
                return returncode

    print("All runs finished.")
    print(f"Run root: {args.output_root}")
    print(f"Per-run stats: {args.output_root / 'per_run_stats.csv'}")
    print(f"Average stats: {args.output_root / 'average_stats_by_dataset.csv'}")
    print(f"Overall average: {args.output_root / 'overall_average_across_datasets.csv'}")
    print(f"Markdown summary: {args.output_root / 'average_stats_by_dataset.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
