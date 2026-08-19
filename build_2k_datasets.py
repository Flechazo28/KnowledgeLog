#!/usr/bin/env python3
"""Build reproducible 2k datasets from the full datasets."""

import argparse
import csv
import hashlib
import math
import random
import shutil
from collections import Counter
from pathlib import Path


SAMPLE_SIZE = 2000
SEED = 20260728


def dataset_rng(dataset: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{dataset}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def read_templates(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"EventId", "EventTemplate", "Occurrences"} <= rows[0].keys():
        raise ValueError(f"Invalid template file: {path}")
    for row in rows:
        row["Occurrences"] = int(row["Occurrences"])
        if row["Occurrences"] <= 0:
            raise ValueError(f"Non-positive occurrence count in {path}: {row}")
    return rows


def select_templates(templates, rng: random.Random):
    """Select ceil(half) of the templates with occurrence-weighted sampling."""
    selected_count = (len(templates) + 1) // 2
    weighted_keys = [
        (-math.log(1.0 - rng.random()) / row["Occurrences"], row)
        for row in templates
    ]
    weighted_keys.sort(key=lambda item: item[0])
    return [row for _, row in weighted_keys[:selected_count]]


def requested_template_counts(selected, sample_size: int, rng: random.Random):
    """Give every selected template one slot, then sample remaining slots by frequency."""
    if len(selected) > sample_size:
        raise ValueError(
            f"Cannot cover {len(selected)} templates with only {sample_size} rows"
        )
    counts = Counter(row["EventId"] for row in selected)
    remaining = sample_size - len(selected)
    if remaining:
        event_ids = [row["EventId"] for row in selected]
        weights = [row["Occurrences"] for row in selected]
        counts.update(rng.choices(event_ids, weights=weights, k=remaining))
    return counts


def requested_occurrence_ranks(selected, requested, rng: random.Random):
    """Choose occurrence ranks with replacement for each selected template."""
    ranks = {}
    for row in selected:
        event_id = row["EventId"]
        ranks[event_id] = Counter(
            rng.randint(1, row["Occurrences"])
            for _ in range(requested[event_id])
        )
    return ranks


def collect_rows(log_path: Path, structured_path: Path, selected, ranks):
    expected = {row["EventId"]: row["Occurrences"] for row in selected}
    seen = Counter()
    samples = []

    with log_path.open("rb") as raw_stream, structured_path.open(
        newline="", encoding="utf-8-sig", errors="replace"
    ) as csv_stream:
        reader = csv.DictReader(csv_stream)
        fieldnames = reader.fieldnames
        if not fieldnames or "LineId" not in fieldnames or "EventId" not in fieldnames:
            raise ValueError(f"Invalid structured file: {structured_path}")

        for row_number, row in enumerate(reader, 1):
            raw_line = raw_stream.readline()
            if raw_line == b"":
                raise ValueError(f"Raw log ended before row {row_number}: {log_path}")
            if int(row["LineId"]) != row_number:
                raise ValueError(
                    f"Non-sequential LineId {row['LineId']} at row {row_number}"
                )

            event_id = row["EventId"]
            if event_id in ranks:
                seen[event_id] += 1
                copies = ranks[event_id].get(seen[event_id], 0)
                for _ in range(copies):
                    samples.append((raw_line.rstrip(b"\r\n"), row.copy()))

            if row_number % 2_000_000 == 0:
                print(f"    scanned {row_number:,} rows", flush=True)

        if raw_stream.readline() != b"":
            raise ValueError(f"Raw log has more rows than structured CSV: {log_path}")

    mismatches = {
        event_id: (seen[event_id], count)
        for event_id, count in expected.items()
        if seen[event_id] != count
    }
    if mismatches:
        raise ValueError(f"Template occurrence mismatch: {mismatches}")
    return samples, fieldnames


def write_dataset(output_root: Path, dataset: str, samples, fieldnames, selected, rng):
    rng.shuffle(samples)
    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=False)

    log_path = output_dir / f"{dataset}_2k.log"
    with log_path.open("wb") as stream:
        for raw_line, _ in samples:
            stream.write(raw_line + b"\n")

    structured_path = output_dir / f"{dataset}_2k.log_structured.csv"
    with structured_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for line_id, (_, row) in enumerate(samples, 1):
            row["LineId"] = str(line_id)
            writer.writerow(row)

    sample_counts = Counter(row["EventId"] for _, row in samples)
    template_path = output_dir / f"{dataset}_2k.log_templates.csv"
    with template_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["EventId", "EventTemplate", "Occurrences"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "EventId": row["EventId"],
                    "EventTemplate": row["EventTemplate"],
                    "Occurrences": sample_counts[row["EventId"]],
                }
            )


def build_dataset(source_root: Path, output_root: Path, dataset: str, sample_size: int, seed: int):
    source_dir = source_root / dataset
    templates = read_templates(source_dir / f"{dataset}_full.log_templates.csv")
    rng = dataset_rng(dataset, seed)
    selected = select_templates(templates, rng)
    requested = requested_template_counts(selected, sample_size, rng)
    ranks = requested_occurrence_ranks(selected, requested, rng)

    print(
        f"[{dataset}] templates {len(selected)}/{len(templates)}, "
        f"sampling {sample_size} rows",
        flush=True,
    )
    samples, fieldnames = collect_rows(
        source_dir / f"{dataset}_full.log",
        source_dir / f"{dataset}_full.log_structured.csv",
        selected,
        ranks,
    )
    if len(samples) != sample_size:
        raise ValueError(f"{dataset}: expected {sample_size} samples, got {len(samples)}")
    write_dataset(output_root, dataset, samples, fieldnames, selected, rng)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("full_dataset"))
    parser.add_argument("--output", type=Path, default=Path("2k_datasets.generated"))
    parser.add_argument("--samples", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    datasets = sorted(
        path.name
        for path in args.source.iterdir()
        if path.is_dir()
        and (path / f"{path.name}_full.log").exists()
        and (path / f"{path.name}_full.log_structured.csv").exists()
        and (path / f"{path.name}_full.log_templates.csv").exists()
    )
    if len(datasets) != 14:
        raise ValueError(f"Expected 14 datasets, found {len(datasets)}: {datasets}")
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    for dataset in datasets:
        build_dataset(args.source, args.output, dataset, args.samples, args.seed)
    print(f"Generated {len(datasets)} datasets in {args.output}", flush=True)


if __name__ == "__main__":
    main()
