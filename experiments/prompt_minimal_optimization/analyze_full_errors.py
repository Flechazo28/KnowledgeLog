#!/usr/bin/env python3
"""Summarize template splits, merges, and row-level errors from a full run."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


DATASETS = ("OpenSSH", "HDFS", "HPC", "Hadoop", "Thunderbird")


def analyze_dataset(result_dir: Path, dataset: str, chunk_size: int) -> dict:
    predicted_path = result_dir / f"{dataset}_full.log_structured.csv"
    truth_path = (
        result_dir.parents[1]
        / "full_dataset"
        / dataset
        / f"{dataset}_full.log_structured.csv"
    )
    columns = ["LineId", "Content", "EventTemplate"]
    predicted_chunks = pd.read_csv(
        predicted_path, usecols=columns, chunksize=chunk_size
    )
    truth_chunks = pd.read_csv(
        truth_path, usecols=columns, chunksize=chunk_size
    )

    pair_counts = Counter()
    truth_counts = Counter()
    predicted_counts = Counter()
    pair_samples = {}
    total = correct = 0

    for predicted, truth in zip(predicted_chunks, truth_chunks):
        if len(predicted) != len(truth):
            raise ValueError(f"{dataset}: chunk sizes differ")
        predicted = predicted.fillna("")
        truth = truth.fillna("")
        if not predicted["LineId"].reset_index(drop=True).equals(
            truth["LineId"].reset_index(drop=True)
        ):
            raise ValueError(f"{dataset}: LineId order differs")

        for content, expected, actual in zip(
            truth["Content"],
            truth["EventTemplate"],
            predicted["EventTemplate"],
        ):
            key = (expected, actual)
            pair_counts[key] += 1
            truth_counts[expected] += 1
            predicted_counts[actual] += 1
            total += 1
            correct += expected == actual
            if expected != actual and key not in pair_samples:
                pair_samples[key] = content

    truth_to_predictions = defaultdict(Counter)
    prediction_to_truths = defaultdict(Counter)
    for (expected, actual), count in pair_counts.items():
        truth_to_predictions[expected][actual] += count
        prediction_to_truths[actual][expected] += count

    wrong_pairs = [
        {
            "count": count,
            "share_of_all_rows": count / total,
            "ground_truth": expected,
            "predicted": actual,
            "log": pair_samples[(expected, actual)],
        }
        for (expected, actual), count in pair_counts.most_common()
        if expected != actual
    ]
    split_templates = []
    for expected, predictions in truth_to_predictions.items():
        if len(predictions) <= 1:
            continue
        dominant, dominant_count = predictions.most_common(1)[0]
        split_templates.append(
            {
                "ground_truth": expected,
                "rows": truth_counts[expected],
                "predicted_template_count": len(predictions),
                "dominant_prediction": dominant,
                "dominant_rows": dominant_count,
                "non_dominant_rows": truth_counts[expected] - dominant_count,
                "predictions": [
                    {"template": template, "count": count}
                    for template, count in predictions.most_common(10)
                ],
            }
        )
    split_templates.sort(
        key=lambda item: (
            item["non_dominant_rows"],
            item["predicted_template_count"],
        ),
        reverse=True,
    )

    merged_templates = []
    for actual, truths in prediction_to_truths.items():
        if len(truths) <= 1:
            continue
        dominant, dominant_count = truths.most_common(1)[0]
        merged_templates.append(
            {
                "predicted": actual,
                "rows": predicted_counts[actual],
                "ground_truth_template_count": len(truths),
                "dominant_ground_truth": dominant,
                "dominant_rows": dominant_count,
                "merged_non_dominant_rows": predicted_counts[actual]
                - dominant_count,
                "ground_truths": [
                    {"template": template, "count": count}
                    for template, count in truths.most_common(10)
                ],
            }
        )
    merged_templates.sort(
        key=lambda item: (
            item["merged_non_dominant_rows"],
            item["ground_truth_template_count"],
        ),
        reverse=True,
    )

    return {
        "dataset": dataset,
        "rows": total,
        "correct_rows": correct,
        "wrong_rows": total - correct,
        "row_template_accuracy": correct / total,
        "ground_truth_templates": len(truth_counts),
        "predicted_templates": len(predicted_counts),
        "wrong_pairs": wrong_pairs[:50],
        "split_templates": split_templates[:50],
        "merged_templates": merged_templates[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()
    output = {
        dataset: analyze_dataset(args.result_dir.resolve(), dataset, args.chunk_size)
        for dataset in DATASETS
    }
    output_path = args.result_dir / "five_dataset_error_analysis.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for dataset, summary in output.items():
        print(
            dataset,
            f"rows={summary['rows']}",
            f"wrong={summary['wrong_rows']}",
            f"pred={summary['predicted_templates']}",
            f"truth={summary['ground_truth_templates']}",
        )
    print(output_path)


if __name__ == "__main__":
    main()
