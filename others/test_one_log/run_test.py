#!/usr/bin/env python3
"""Run independent, end-to-end KnowledgeLog tests for arbitrary log messages."""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import regex


TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
COMMON_REGEX_PATH = BENCHMARK_DIR / "logparser" / "KnowledgeLog" / "common.json"

sys.path.insert(0, str(BENCHMARK_DIR))

from logparser.KnowledgeLog.gpt_query import (  # noqa: E402
    DEFAULT_ENABLED_POSTPROCESSING_GROUPS,
    DEFAULT_ENABLED_PREPROCESSING_GROUPS,
    DEFAULT_ENABLED_PROMPT_GROUPS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    FINAL_PLACEHOLDER,
    VARIABLE_PLACEHOLDER,
    _extract_template_from_response,
    apply_final_domain_preprocessing,
    build_final_domain_knowledge,
    get_response_from_openai_key,
    post_process_template,
)


LOG_FIELD_NAMES = ("log", "log_message", "query", "Content")
TEMPLATE_FIELD_NAMES = (
    "expected_template",
    "ground_truth_template",
    "true_template",
    "answer",
    "EventTemplate",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Independently parse arbitrary logs through KnowledgeLog's complete "
            "preprocessing, prompt, response extraction, postprocessing, common "
            "regex, and ground-truth comparison flow without ParsingCache."
        )
    )
    parser.add_argument(
        "--log",
        help="One original log message. Use together with --expected-template.",
    )
    parser.add_argument(
        "--expected-template",
        help=(
            "Ground-truth template for --log. Both <*> and {variables} "
            "placeholders are accepted."
        ),
    )
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        default=[],
        metavar=("ID", "LOG", "EXPECTED_TEMPLATE"),
        help=(
            "Add an inline case. May be repeated to test several arbitrary logs, "
            "for example: --case case1 'raw log' 'true template'."
        ),
    )
    parser.add_argument(
        "--cases-file",
        type=Path,
        help=(
            "JSON or JSONL case file. Supports log/log_message/query/Content and "
            "expected_template/ground_truth_template/answer/EventTemplate fields."
        ),
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only the selected case ID from the assembled cases. Repeatable.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Independent full parsing runs per selected log. Default: 1.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to wait between independent calls. Default: 0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory. Default: a timestamped directory under "
            "test_one_log/results/generic/."
        ),
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print selected case IDs and inputs without calling the model.",
    )
    return parser.parse_args()


def first_present(record, field_names):
    for name in field_names:
        value = record.get(name)
        if value is not None:
            return str(value)
    return None


def normalize_expected_template(template):
    return template.replace(VARIABLE_PLACEHOLDER, FINAL_PLACEHOLDER)


def normalize_case(record, default_id):
    if not isinstance(record, dict):
        raise ValueError(f"Case {default_id!r} must be a JSON object")
    log_message = first_present(record, LOG_FIELD_NAMES)
    expected_template = first_present(record, TEMPLATE_FIELD_NAMES)
    if log_message is None or expected_template is None:
        raise ValueError(
            f"Case {default_id!r} must contain a log field and a ground-truth "
            "template field"
        )
    case_id = str(record.get("id") or record.get("case_id") or default_id)
    return {
        "id": case_id,
        "log": log_message,
        "expected_template_input": expected_template,
        "expected_template": normalize_expected_template(expected_template),
    }


def load_cases_file(path):
    if not path.is_file():
        raise ValueError(f"Cases file does not exist: {path}")

    if path.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on {path}:{line_number}: {error}"
                    ) from error
    else:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Existing sampled-example files use JSON objects, one per line,
            # despite carrying a .json suffix. Accept that format as JSONL too.
            records = []
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSON on {path}:{line_number}: {error}"
                        ) from error
        else:
            if isinstance(payload, list):
                records = payload
            elif isinstance(payload, dict) and isinstance(payload.get("cases"), list):
                records = payload["cases"]
            elif isinstance(payload, dict):
                records = [payload]
            else:
                raise ValueError(f"Unsupported JSON case structure in {path}")

    return [
        normalize_case(record, f"{path.stem}_{index:03d}")
        for index, record in enumerate(records, 1)
    ]


def collect_cases(args):
    cases = []
    if (args.log is None) != (args.expected_template is None):
        raise ValueError("--log and --expected-template must be provided together")
    if args.log is not None:
        cases.append(
            normalize_case(
                {
                    "id": "single",
                    "log": args.log,
                    "expected_template": args.expected_template,
                },
                "single",
            )
        )
    for case_id, log_message, expected_template in args.case:
        cases.append(
            normalize_case(
                {
                    "id": case_id,
                    "log": log_message,
                    "expected_template": expected_template,
                },
                case_id,
            )
        )
    if args.cases_file is not None:
        cases.extend(load_cases_file(args.cases_file))
    if not cases:
        raise ValueError(
            "No test cases supplied; use --log with --expected-template, repeat "
            "--case, or provide --cases-file"
        )

    duplicate_ids = [
        case_id
        for case_id, count in Counter(case["id"] for case in cases).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"Duplicate case IDs: {duplicate_ids}")

    if args.case_id:
        requested = set(args.case_id)
        available = {case["id"] for case in cases}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"Unknown --case-id values: {unknown}")
        cases = [case for case in cases if case["id"] in requested]
    return cases


def load_common_regexes():
    payload = json.loads(COMMON_REGEX_PATH.read_text(encoding="utf-8"))
    return [regex.compile(pattern) for pattern in payload["COMMON"]["regex"]]


def markdown_escape(value):
    if value is None:
        return ""
    return str(value).replace("|", r"\|").replace("\n", "<br>")


def build_summary(args, cases, records, started_at, finished_at):
    correct_count = sum(record["is_correct"] for record in records)
    lines = [
        "# Generic End-to-end Single-log Test",
        "",
        f"- Started: `{started_at}`",
        f"- Finished: `{finished_at}`",
        f"- Model: `{args.model}`",
        f"- Temperature: `{DEFAULT_TEMPERATURE}`",
        f"- Cases: `{len(cases)}`",
        f"- Independent repetitions per case: `{args.repetitions}`",
        f"- Total runs: `{len(records)}`",
        f"- Exact matches: `{correct_count}/{len(records)}`",
        "",
        "## Per-case Summary",
        "",
        "| Case | Correct | Runs | Accuracy | Final-template variants |",
        "|---|---:|---:|---:|---:|",
    ]

    for case in cases:
        case_records = [record for record in records if record["case_id"] == case["id"]]
        case_correct = sum(record["is_correct"] for record in case_records)
        variants = len({record["final_template"] for record in case_records})
        accuracy = case_correct / len(case_records) if case_records else 0.0
        lines.append(
            f"| `{markdown_escape(case['id'])}` | {case_correct} | "
            f"{len(case_records)} | {accuracy:.1%} | {variants} |"
        )

    lines.extend(["", "## Case Details", ""])
    for case in cases:
        case_records = [record for record in records if record["case_id"] == case["id"]]
        template_counts = Counter(record["final_template"] for record in case_records)
        lines.extend(
            [
                f"### `{markdown_escape(case['id'])}`",
                "",
                f"- Original log: `{markdown_escape(case['log'])}`",
                f"- Preprocessed log: `{markdown_escape(apply_final_domain_preprocessing(case['log']))}`",
                f"- Ground-truth template: `{markdown_escape(case['expected_template'])}`",
                "",
                "| Final template | Count |",
                "|---|---:|",
            ]
        )
        for template, count in template_counts.most_common():
            lines.append(f"| `{markdown_escape(template)}` | {count} |")
        lines.extend(
            [
                "",
                "| Run | Valid | Seconds | Final template | Exact match |",
                "|---:|---|---:|---|---|",
            ]
        )
        for record in case_records:
            lines.append(
                f"| {record['repetition']} | "
                f"{'yes' if record['valid'] else 'no'} | "
                f"{record['elapsed_seconds']:.3f} | "
                f"`{markdown_escape(record['final_template'])}` | "
                f"{'yes' if record['is_correct'] else 'no'} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def default_output_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return TEST_DIR / "results" / "generic" / f"single_log_result_{timestamp}"


def main():
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")
    if args.sleep < 0:
        raise ValueError("--sleep must be non-negative")

    cases = collect_cases(args)
    if args.list_cases:
        for case in cases:
            print(f"{case['id']}\t{case['log']}\t{case['expected_template']}")
        return

    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    common_regexes = load_common_regexes()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    configuration = {
        "started_at": started_at,
        "model": args.model,
        "temperature": DEFAULT_TEMPERATURE,
        "repetitions_per_log": args.repetitions,
        "uses_parsing_cache": False,
        "common_regex_path": str(COMMON_REGEX_PATH),
        "enabled_prompt_groups": list(DEFAULT_ENABLED_PROMPT_GROUPS),
        "enabled_preprocessing_groups": list(DEFAULT_ENABLED_PREPROCESSING_GROUPS),
        "enabled_postprocessing_groups": list(DEFAULT_ENABLED_POSTPROCESSING_GROUPS),
        "domain_knowledge": build_final_domain_knowledge(),
        "cases": [
            {
                **case,
                "preprocessed_log": apply_final_domain_preprocessing(case["log"]),
            }
            for case in cases
        ],
    }
    (output_dir / "configuration.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    records = []
    results_path = output_dir / "results.jsonl"
    total_runs = len(cases) * args.repetitions
    run_number = 0
    with results_path.open("w", encoding="utf-8") as output:
        for case in cases:
            for repetition in range(1, args.repetitions + 1):
                run_number += 1
                call_started = time.monotonic()
                response = get_response_from_openai_key(
                    case["log"],
                    model=args.model,
                )
                fallback_response = f"Log message: `{case['log']}`"
                extracted_template = None
                final_template = None
                valid = False
                if response == fallback_response:
                    status = "api_error"
                else:
                    extracted_template = _extract_template_from_response(response)
                    if extracted_template is None:
                        status = "format_error"
                    else:
                        final_template, valid = post_process_template(
                            extracted_template,
                            regs_common=common_regexes,
                            log_message=case["log"],
                        )
                        status = "ok" if valid else "postprocess_error"
                elapsed = time.monotonic() - call_started
                is_correct = status == "ok" and final_template == case["expected_template"]
                record = {
                    "run": run_number,
                    "case_id": case["id"],
                    "repetition": repetition,
                    "elapsed_seconds": round(elapsed, 3),
                    "status": status,
                    "valid": bool(valid),
                    "log": case["log"],
                    "preprocessed_log": apply_final_domain_preprocessing(case["log"]),
                    "expected_template": case["expected_template"],
                    "extracted_template": extracted_template,
                    "final_template": final_template,
                    "is_correct": is_correct,
                    "raw_response": response,
                }
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    f"[{run_number}/{total_runs}] {case['id']} run={repetition}: "
                    f"{final_template or f'<{status.upper()}>'} "
                    f"({'correct' if is_correct else 'wrong'})"
                )
                if args.sleep > 0 and run_number < total_runs:
                    time.sleep(args.sleep)

    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    (output_dir / "summary.md").write_text(
        build_summary(args, cases, records, started_at, finished_at),
        encoding="utf-8",
    )
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
