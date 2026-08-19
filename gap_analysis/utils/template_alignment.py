from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


VARIABLE_PLACEHOLDER = "{variables}"


def _template_parts(template: Any) -> List[str]:
    text = "" if template is None else str(template)
    parts = re.split(f"({re.escape(VARIABLE_PLACEHOLDER)})", text)
    merged: List[str] = []
    for part in parts:
        if not part:
            continue
        if (
            part == VARIABLE_PLACEHOLDER
            and merged
            and merged[-1] == VARIABLE_PLACEHOLDER
        ):
            continue
        merged.append(part)
    return merged


def align_template_to_log(
    log_message: Any,
    template: Any,
) -> Optional[Dict[str, Any]]:
    """Map a {variables} template to exact character spans in one log message."""
    log = "" if log_message is None else str(log_message)
    parts = _template_parts(template)
    group_names: List[str] = []
    pattern_parts: List[str] = []
    variable_index = 0

    for part in parts:
        if part == VARIABLE_PLACEHOLDER:
            group_name = f"variable_{variable_index}"
            group_names.append(group_name)
            pattern_parts.append(f"(?P<{group_name}>.*?)")
            variable_index += 1
        else:
            pattern_parts.append(re.escape(part))

    match = re.fullmatch("".join(pattern_parts), log, flags=re.DOTALL)
    if match is None:
        return None

    variable_spans = [
        {
            "start": match.start(group_name),
            "end": match.end(group_name),
            "text": match.group(group_name),
        }
        for group_name in group_names
    ]
    variable_mask = [False] * len(log)
    for span in variable_spans:
        for position in range(span["start"], span["end"]):
            variable_mask[position] = True

    return {
        "log_message": log,
        "variable_spans": variable_spans,
        "variable_mask": variable_mask,
    }


def _trim_span(log: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and log[start].isspace():
        start += 1
    while end > start and log[end - 1].isspace():
        end -= 1
    return start, end


def _context(log: str, start: int, end: int, width: int = 80) -> str:
    return log[max(0, start - width) : min(len(log), end + width)]


def _contiguous_true_spans(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, value in enumerate([*mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((start, index))
            start = None
    return spans


def extract_knowledge_spans(
    log_message: Any,
    parsed_template: Any,
    ground_truth_template: Any,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Extract all ground-truth variables and LLM-overabstracted constants."""
    log = "" if log_message is None else str(log_message)
    truth_alignment = align_template_to_log(log, ground_truth_template)
    if truth_alignment is None:
        return [], ["ground_truth_template does not align with log_message"]

    parsed_alignment = align_template_to_log(log, parsed_template)
    warnings: List[str] = []
    if parsed_alignment is None:
        warnings.append(
            "parsed_template does not align with log_message; preserve spans were skipped"
        )

    spans: List[Dict[str, Any]] = []
    for span in truth_alignment["variable_spans"]:
        start, end = _trim_span(log, span["start"], span["end"])
        if start == end:
            continue
        spans.append(
            {
                "action": "abstract",
                "log_span": log[start:end],
                "ground_truth_span": VARIABLE_PLACEHOLDER,
                "log_context": _context(log, start, end),
            }
        )

    if parsed_alignment is not None:
        preserve_mask = [
            parsed_variable and not truth_variable
            for parsed_variable, truth_variable in zip(
                parsed_alignment["variable_mask"],
                truth_alignment["variable_mask"],
            )
        ]
        for raw_start, raw_end in _contiguous_true_spans(preserve_mask):
            start, end = _trim_span(log, raw_start, raw_end)
            literal = log[start:end]
            if start == end or not any(character.isalnum() for character in literal):
                continue
            spans.append(
                {
                    "action": "preserve",
                    "log_span": literal,
                    "llm_span": VARIABLE_PLACEHOLDER,
                    "ground_truth_span": literal,
                    "log_context": _context(log, start, end),
                }
            )

    unique: List[Dict[str, Any]] = []
    seen = set()
    for span in spans:
        key = (
            span["action"],
            span["log_span"],
            span["ground_truth_span"],
            span["log_context"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            {
                "difference_id": len(unique),
                **span,
            }
        )
    return unique, warnings
