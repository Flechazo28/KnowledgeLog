from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List


def template_tokens(template: Any) -> List[str]:
    text = "" if template is None else str(template)
    return re.findall(r"\{variables\}|\s+|[^\s{}]+|[{}]", text)


def basic_template_differences(
    parsed_template: Any,
    ground_truth_template: Any,
) -> List[Dict[str, str]]:
    parsed_tokens = template_tokens(parsed_template)
    truth_tokens = template_tokens(ground_truth_template)
    matcher = difflib.SequenceMatcher(a=parsed_tokens, b=truth_tokens, autojunk=False)
    return [
        {
            "operation": tag,
            "llm_span": "".join(parsed_tokens[i1:i2]),
            "ground_truth_span": "".join(truth_tokens[j1:j2]),
        }
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]
