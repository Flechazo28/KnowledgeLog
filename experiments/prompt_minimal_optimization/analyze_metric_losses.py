#!/usr/bin/env python3
"""Explain GA/FGA/FTA losses using ground-truth/predicted template pairs."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


DATASETS = ("OpenSSH", "HDFS", "HPC", "Hadoop", "Thunderbird")


def pair_counts(result_dir: Path, dataset: str, chunk_size: int) -> Counter:
    result_path = result_dir / f"{dataset}_full.log_structured.csv"
    truth_path = (
        result_dir.parents[1]
        / "full_dataset"
        / dataset
        / f"{dataset}_full.log_structured.csv"
    )
    counts = Counter()
    result_chunks = pd.read_csv(
        result_path, usecols=["EventTemplate"], chunksize=chunk_size
    )
    truth_chunks = pd.read_csv(
        truth_path, usecols=["EventTemplate"], chunksize=chunk_size
    )
    for result, truth in zip(result_chunks, truth_chunks):
        pairs = pd.DataFrame(
            {
                "truth": truth["EventTemplate"].fillna(""),
                "prediction": result["EventTemplate"].fillna(""),
            }
        ).value_counts(sort=False)
        counts.update(
            {
                (truth_template, predicted_template): int(count)
                for (truth_template, predicted_template), count in pairs.items()
            }
        )
    return counts


def analyze(counts: Counter) -> dict:
    truth_to_predictions = defaultdict(Counter)
    prediction_to_truths = defaultdict(Counter)
    for (truth, prediction), count in counts.items():
        truth_to_predictions[truth][prediction] += count
        prediction_to_truths[prediction][truth] += count

    grouping_failures = []
    accurate_group_rows = 0
    accurate_group_templates = 0
    for truth, predictions in truth_to_predictions.items():
        truth_rows = sum(predictions.values())
        is_exact_group = (
            len(predictions) == 1
            and len(prediction_to_truths[next(iter(predictions))]) == 1
        )
        if is_exact_group:
            accurate_group_rows += truth_rows
            accurate_group_templates += 1
            continue
        grouping_failures.append(
            {
                "ground_truth": truth,
                "rows_lost_from_ga": truth_rows,
                "predicted_template_count": len(predictions),
                "merged_prediction_count": sum(
                    len(prediction_to_truths[prediction]) > 1
                    for prediction in predictions
                ),
                "predictions": [
                    {"template": prediction, "count": count}
                    for prediction, count in predictions.most_common(8)
                ],
            }
        )
    grouping_failures.sort(
        key=lambda item: item["rows_lost_from_ga"], reverse=True
    )

    correct_template_names = {
        prediction
        for prediction, truths in prediction_to_truths.items()
        if set(truths) == {prediction}
    }
    truth_templates = set(truth_to_predictions)
    predicted_templates = set(prediction_to_truths)
    template_failures = []
    for truth, predictions in truth_to_predictions.items():
        if truth in correct_template_names:
            continue
        template_failures.append(
            {
                "ground_truth": truth,
                "rows": sum(predictions.values()),
                "exact_text_generated": truth in predicted_templates,
                "predictions": [
                    {"template": prediction, "count": count}
                    for prediction, count in predictions.most_common(5)
                ],
            }
        )
    template_failures.sort(key=lambda item: item["rows"], reverse=True)

    total_rows = sum(counts.values())
    exact_rows = sum(
        count for (truth, prediction), count in counts.items() if truth == prediction
    )
    grouping_precision = accurate_group_templates / len(predicted_templates)
    grouping_recall = accurate_group_templates / len(truth_templates)
    fga = (
        2 * grouping_precision * grouping_recall
        / (grouping_precision + grouping_recall)
        if grouping_precision or grouping_recall
        else 0.0
    )
    return {
        "rows": total_rows,
        "ground_truth_templates": len(truth_templates),
        "predicted_templates": len(predicted_templates),
        "exact_rows": exact_rows,
        "pa": exact_rows / total_rows,
        "accurate_group_rows": accurate_group_rows,
        "ga": accurate_group_rows / total_rows,
        "accurate_group_templates": accurate_group_templates,
        "grouping_precision": grouping_precision,
        "grouping_recall": grouping_recall,
        "fga": fga,
        "correct_template_names": len(correct_template_names),
        "pta": len(correct_template_names) / len(predicted_templates),
        "rta": len(correct_template_names) / len(truth_templates),
        "grouping_failures": grouping_failures,
        "template_failures": template_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    output = {}
    for dataset in DATASETS:
        print(f"Analyzing {dataset}", flush=True)
        output[dataset] = analyze(
            pair_counts(result_dir, dataset, args.chunk_size)
        )
    output_path = result_dir / "metric_loss_analysis.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
