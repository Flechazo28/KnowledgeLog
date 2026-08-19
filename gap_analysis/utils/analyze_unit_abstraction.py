import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "sampled_examples"
OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "unit_abstraction_in_14datasets_128shot_20260610.md"
)

MULTI_CHARACTER_UNITS = (
    r"milliseconds?|second\(s\)|seconds?|millis|msec|secs?|ms|us|"
    r"Kbytes|bytes?|KiB|kB|KB|MB|GB|TB|GHz|MHz|kHz|Hz|"
    r"BogoMIPS|CPUs?|VCPU|vCores"
)

QUERY_PATTERNS = (
    re.compile(
        rf"(?<![A-Za-z0-9_.-])[-+]?\d+(?:\.\d+)?(?:\s*|-)"
        rf"(?P<unit>{MULTI_CHARACTER_UNITS})(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![A-Za-z0-9_.-])[-+]?\d+(?:\.\d+)?\s+(?P<unit>s)\b"
    ),
    re.compile(
        r"(?<![A-Za-z0-9_.-])[-+]?\d+(?:\.\d+)?(?:\s*|-)(?P<unit>[KMGC])\b"
    ),
)

ANSWER_UNIT_PATTERN = re.compile(
    rf"(?:\{{variables\}}|[-+]?\d+(?:\.\d+)?)(?:\s*|-)"
    rf"(?P<unit>{MULTI_CHARACTER_UNITS}|s|[KMGC])(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

CANONICAL_UNITS = {
    "bogomips": "BogoMIPS",
    "cpu": "CPU",
    "cpus": "CPUs",
    "gb": "GB",
    "ghz": "GHz",
    "kb": "KB",
    "kbytes": "Kbytes",
    "kib": "KiB",
    "mb": "MB",
    "mhz": "MHz",
    "vcpu": "VCPU",
}


def normalize_unit(unit):
    if len(unit) == 1:
        return unit
    return CANONICAL_UNITS.get(unit.lower(), unit.lower())


def units_equal(left, right):
    if len(left) == 1 or len(right) == 1:
        return left == right
    return left.lower() == right.lower()


def collect_records():
    records = []
    for path in sorted(ROOT.glob("*/128shot.json")):
        dataset = path.parent.name
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            sample = json.loads(raw_line)
            query = sample["query"]
            answer = sample["answer"]

            matches = []
            for pattern in QUERY_PATTERNS:
                matches.extend(pattern.finditer(query))
            matches.sort(key=lambda match: match.start())

            unique_matches = []
            seen_spans = set()
            for match in matches:
                if match.span() not in seen_spans:
                    seen_spans.add(match.span())
                    unique_matches.append(match)

            answer_units = [
                match.group("unit") for match in ANSWER_UNIT_PATTERN.finditer(answer)
            ]
            for match in unique_matches:
                unit = match.group("unit")
                answer_index = next(
                    (
                        index
                        for index, answer_unit in enumerate(answer_units)
                        if units_equal(unit, answer_unit)
                    ),
                    None,
                )
                status = "abstracted" if answer_index is None else "kept"
                if answer_index is not None:
                    answer_units.pop(answer_index)

                records.append(
                    {
                        "dataset": dataset,
                        "line": line_number,
                        "unit": normalize_unit(unit),
                        "expression": match.group(0),
                        "status": status,
                        "query": query,
                        "answer": answer,
                    }
                )
    return records


def escape_cell(text):
    return text.replace("|", r"\|").replace("\n", " ")


def build_report(records):
    unit_counts = defaultdict(Counter)
    unit_datasets = defaultdict(lambda: defaultdict(set))
    dataset_counts = defaultdict(Counter)

    for record in records:
        unit = record["unit"]
        status = record["status"]
        unit_counts[unit][status] += 1
        unit_datasets[unit][status].add(record["dataset"])
        dataset_counts[record["dataset"]][status] += 1

    abstracted_total = sum(
        counts["abstracted"] for counts in unit_counts.values()
    )
    kept_total = sum(counts["kept"] for counts in unit_counts.values())

    lines = [
        "# 14个数据集128-shot日志变量单位抽象情况",
        "",
        "## 统计口径",
        "",
        "- 数据来源：`gap_analysis/sampled_examples/*/128shot.json`。",
        "- `数值+单位`整体从GT模板中消失，记为“单位需要抽象”。",
        "- GT保留单位、仅将数值替换为`{variables}`，记为“单位不需要抽象”。",
        "- 按单位出现次数统计；同一日志包含多个单位时分别计数。",
        "- 单字符容量/温度单位仅识别大写`K`、`M`、`G`、`C`，避免把普通单词误判为单位。",
        "",
        "## 总览",
        "",
        f"- 单位需要抽象：**{abstracted_total}** 次。",
        f"- 单位不需要抽象：**{kept_total}** 次。",
        f"- 总计：**{abstracted_total + kept_total}** 次。",
        "",
        "| 单位 | 需要抽象 | 不需要抽象 | 抽象来源数据集 | 保留来源数据集 |",
        "|---|---:|---:|---|---|",
    ]

    for unit in sorted(unit_counts, key=lambda value: value.lower()):
        counts = unit_counts[unit]
        abstracted_datasets = ", ".join(
            sorted(unit_datasets[unit]["abstracted"])
        ) or "-"
        kept_datasets = ", ".join(sorted(unit_datasets[unit]["kept"])) or "-"
        lines.append(
            f"| `{unit}` | {counts['abstracted']} | {counts['kept']} | "
            f"{abstracted_datasets} | {kept_datasets} |"
        )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            "### 仅在GT中整体抽象的单位",
            "",
        ]
    )
    abstract_only = [
        unit
        for unit, counts in unit_counts.items()
        if counts["abstracted"] and not counts["kept"]
    ]
    lines.append(", ".join(f"`{unit}`" for unit in sorted(abstract_only)))

    lines.extend(
        [
            "",
            "### 仅在GT中保留的单位",
            "",
        ]
    )
    kept_only = [
        unit
        for unit, counts in unit_counts.items()
        if counts["kept"] and not counts["abstracted"]
    ]
    lines.append(", ".join(f"`{unit}`" for unit in sorted(kept_only)))

    lines.extend(
        [
            "",
            "### 两种标注均存在的单位",
            "",
        ]
    )
    mixed = [
        unit
        for unit, counts in unit_counts.items()
        if counts["abstracted"] and counts["kept"]
    ]
    lines.append(", ".join(f"`{unit}`" for unit in sorted(mixed)))
    lines.extend(
        [
            "",
            "因此，是否抽象不能只由单位词决定，还取决于单位所在的字段边界和日志语法。",
            "",
            "## 全部日志证据",
            "",
        ]
    )

    for unit in sorted(unit_counts, key=lambda value: value.lower()):
        lines.extend([f"### 单位 `{unit}`", ""])
        for status, label in (
            ("abstracted", "需要抽象"),
            ("kept", "不需要抽象"),
        ):
            matching_records = [
                record
                for record in records
                if record["unit"] == unit and record["status"] == status
            ]
            if not matching_records:
                continue
            lines.extend(
                [
                    f"#### {label}（{len(matching_records)}次）",
                    "",
                ]
            )
            for record in matching_records:
                lines.extend(
                    [
                        (
                            f"- **{record['dataset']} 第{record['line']}行**，"
                            f"单位表达式：`{escape_cell(record['expression'])}`"
                        ),
                        f"  - 日志：`{escape_cell(record['query'])}`",
                        f"  - GT：`{escape_cell(record['answer'])}`",
                    ]
                )
            lines.append("")

    return "\n".join(lines) + "\n"


def main():
    records = collect_records()
    OUTPUT.write_text(build_report(records), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
