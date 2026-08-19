#!/usr/bin/env python3
"""Run the 14-dataset KnowledgeLog baseline without injected domain knowledge.

使用方法：
1. 启动本地模型服务。
2. 激活 knowledgelog 环境。
3. 在项目根目录运行：

   python -u benchmark/evaluation/KnowledgeLog_no_knowledge_eval.py \
     --full_data --model Llama-3.3-70B-Instruct

本入口不修改原有代码，并在运行时关闭：
- prompt 领域知识；
- 领域知识预处理规则；
- common.json 正则；
- post_process_template() 及 correct_single_template()。

仅保留 {variables} 到解析器内部 <*> 的必要格式转换和模板自匹配检查。
"""

from __future__ import annotations

import importlib
import os
import runpy
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple


EVAL_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = EVAL_DIR.parent
ORIGINAL_EVAL = EVAL_DIR / "KnowledgeLog_eval.py"

# These defaults are read while gpt_query.py is imported, so set them first.
os.environ["LILAC_PROMPT_GROUPS"] = "none"
os.environ["LILAC_PREPROCESSING_GROUPS"] = "none"
os.environ["LILAC_POSTPROCESSING_GROUPS"] = "none"

if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

gpt_query = importlib.import_module("logparser.KnowledgeLog.gpt_query")
parser_module = importlib.import_module("logparser.KnowledgeLog.KnowledgeLog")
ParsingCache = importlib.import_module(
    "logparser.KnowledgeLog.parsing_cache"
).ParsingCache


def query_without_postprocessing(
    log_message: str,
    regs_common=None,
    examples: Optional[Sequence[str]] = None,
    model: str = "gpt-3.5-turbo",
) -> Tuple[str, bool]:
    """Query the base prompt and bypass every template post-processing rule."""
    del regs_common
    template, success = gpt_query.query_template_from_gpt(
        log_message,
        examples,
        model,
    )
    if not success or not template:
        return log_message, False

    # ParsingCache recognizes <*> rather than the model-facing {variables}.
    # This exact format conversion does not infer or alter template content.
    internal_template = template.replace("{variables}", "<*>")
    if internal_template == "<*>":
        return log_message, False

    tree = ParsingCache()
    if tree.add_templates(internal_template) < 0:
        return log_message, False
    if tree.match_event(log_message)[0] == "NoMatch":
        return log_message, False
    return internal_template, True


def configure_baseline() -> None:
    # Prevent common.json loading and replace the imported query function used
    # by KnowledgeLog.py. The original modules and files remain unchanged.
    parser_module.load_regs = lambda: []
    parser_module.query_template_from_gpt_with_check = query_without_postprocessing
    gpt_query.build_final_domain_knowledge = lambda enabled_prompt_groups=(): ""
    gpt_query.apply_final_domain_preprocessing = (
        lambda log_message, enabled_groups=(): log_message
    )


def main() -> None:
    configure_baseline()
    os.chdir(EVAL_DIR)
    runpy.run_path(str(ORIGINAL_EVAL), run_name="__main__")


if __name__ == "__main__":
    main()
