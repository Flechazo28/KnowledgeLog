import argparse
from pathlib import Path

import pandas as pd


def error_type(ground_template: str, parsed_template: str) -> str:
    ground_variables = ground_template.count("<*>")
    parsed_variables = parsed_template.count("<*>")
    if parsed_variables < ground_variables:
        return "under_abstraction"
    if parsed_variables > ground_variables:
        return "over_abstraction"
    return "boundary_or_literal_mismatch"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    all_errors = []
    summaries = []
    for parsed_path in sorted(args.result_dir.glob("*_2k.log_structured.csv")):
        dataset = parsed_path.name.split("_2k", 1)[0]
        ground_path = args.ground_dir / dataset / parsed_path.name
        ground = pd.read_csv(ground_path, dtype=str, keep_default_na=False)
        parsed = pd.read_csv(parsed_path, dtype=str, keep_default_na=False)

        compared = ground[["LineId", "Content", "EventTemplate"]].rename(
            columns={"EventTemplate": "ground_truth_template"}
        ).merge(
            parsed[["LineId", "EventTemplate"]].rename(
                columns={"EventTemplate": "parsed_template"}
            ),
            on="LineId",
            how="inner",
            validate="one_to_one",
        )
        errors = compared[
            compared["ground_truth_template"] != compared["parsed_template"]
        ].copy()
        errors.insert(0, "dataset", dataset)
        errors["error_type"] = [
            error_type(ground_template, parsed_template)
            for ground_template, parsed_template in zip(
                errors["ground_truth_template"], errors["parsed_template"]
            )
        ]
        all_errors.append(errors)

        counts = errors["error_type"].value_counts()
        summaries.append(
            {
                "dataset": dataset,
                "total_logs": len(compared),
                "error_logs": len(errors),
                "error_rate": len(errors) / len(compared),
                "under_abstraction": int(counts.get("under_abstraction", 0)),
                "over_abstraction": int(counts.get("over_abstraction", 0)),
                "boundary_or_literal_mismatch": int(
                    counts.get("boundary_or_literal_mismatch", 0)
                ),
            }
        )

    errors = pd.concat(all_errors, ignore_index=True)
    summary = pd.DataFrame(summaries)
    totals = summary.select_dtypes("number").sum()
    summary.loc[len(summary)] = {
        "dataset": "ALL",
        "total_logs": int(totals["total_logs"]),
        "error_logs": int(totals["error_logs"]),
        "error_rate": totals["error_logs"] / totals["total_logs"],
        "under_abstraction": int(totals["under_abstraction"]),
        "over_abstraction": int(totals["over_abstraction"]),
        "boundary_or_literal_mismatch": int(
            totals["boundary_or_literal_mismatch"]
        ),
    }
    pairs = (
        errors.groupby(
            [
                "dataset",
                "ground_truth_template",
                "parsed_template",
                "error_type",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="error_logs")
        .sort_values(["error_logs", "dataset"], ascending=[False, True])
    )

    errors.to_csv(args.result_dir / "misparsed_logs.csv", index=False)
    summary.to_csv(args.result_dir / "misparse_summary.csv", index=False)
    pairs.to_csv(args.result_dir / "misparsed_template_pairs.csv", index=False)


if __name__ == "__main__":
    main()
