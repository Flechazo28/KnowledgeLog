# -*- coding: utf-8 -*-
"""Identify LLM knowledge gaps (IR list) for KnowledgeLog datasets."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import error, request
from urllib.parse import urlparse

from utils.io_utils import read_json, write_json


PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGELOG_ROOT = PROJECT_ROOT.parent
KNOWLEDGELOG_POST_PROCESS_FILE = KNOWLEDGELOG_ROOT / "benchmark" / "logparser" / "KnowledgeLog" / "post_process.py"
KNOWLEDGELOG_COMMON_REGEX_FILE = KNOWLEDGELOG_ROOT / "benchmark" / "logparser" / "KnowledgeLog" / "common.json"
DEFAULT_KEY_FILE = KNOWLEDGELOG_ROOT / "openai_key.txt"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_KNOWLEDGE_GAPS_RESULTS_DIR = DEFAULT_RESULTS_DIR / "knowledge_gaps"
DEFAULT_SAMPLED_EXAMPLES_DIR = PROJECT_ROOT / "sampled_examples"
DEFAULT_SHOT = 32
DEFAULT_MAX_TOKENS = 512
DEFAULT_REQUEST_TIMEOUT = 300
DEFAULT_RETRY_TIMES = 3
DEFAULT_RETRY_SLEEP_SECONDS = 2
VARIABLE_PLACEHOLDER = "{variables}"


def _load_knowledgelog_template_corrector():
    """Load KnowledgeLog/post_process.py without importing the parser package."""
    spec = importlib.util.spec_from_file_location(
        "knowledgelog_lilac_post_process",
        KNOWLEDGELOG_POST_PROCESS_FILE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load KnowledgeLog post-processing module: {KNOWLEDGELOG_POST_PROCESS_FILE}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.correct_single_template


def _load_common_regexes(common_json_file: Path = KNOWLEDGELOG_COMMON_REGEX_FILE) -> Tuple[re.Pattern, ...]:
    """Load and compile the shared COMMON regex rules used by KnowledgeLog."""
    data = read_json(common_json_file)
    try:
        patterns = data["COMMON"]["regex"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid KnowledgeLog common regex file: {common_json_file}") from exc
    if not isinstance(patterns, list) or not all(isinstance(pattern, str) for pattern in patterns):
        raise ValueError(f"Invalid COMMON.regex list in: {common_json_file}")
    return tuple(re.compile(pattern) for pattern in patterns)


correct_single_template = _load_knowledgelog_template_corrector()
COMMON_REGEXES = _load_common_regexes()


def _load_sampled_examples_file(fp: Path) -> List[Tuple[str, str]]:
    """Load sampled_examples JSONL records as (log_message, ground_truth_template)."""
    records: List[Tuple[str, str]] = []
    for line_no, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or "query" not in item or "answer" not in item:
            raise ValueError(f"Invalid sampled example at {fp}:{line_no}; expected query/answer fields.")
        log_message = str(item["query"]).strip()
        ground_truth_template = _normalize_variable_placeholders(str(item["answer"]).strip())
        if (
            log_message
            and ground_truth_template
            and log_message.lower() != "nan"
            and ground_truth_template.lower() != "nan"
        ):
            records.append((log_message, ground_truth_template))
    return records


def _load_api_config(key_file: Path, model: str) -> Tuple[str, str]:
    """
    Read API config from openai_key.txt as model/api_base/api_key triples.
    """
    lines = [line.strip() for line in key_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or len(lines) % 3 != 0:
        raise ValueError(f"Invalid key file format: {key_file}. Expected model/api_base/api_key triples.")

    configs: List[Tuple[str, str, str]] = []
    for idx in range(0, len(lines), 3):
        name, api_base, api_key = lines[idx], lines[idx + 1], lines[idx + 2]
        if not name or not api_base.startswith(("http://", "https://")) or not api_key:
            raise ValueError(f"Invalid API config entry starting at line {idx + 1} in {key_file}")
        configs.append((name, api_base.rstrip("/"), api_key))

    if not configs:
        raise ValueError(
            f"Invalid key file format: {key_file}. Expected model/api_base/api_key triples."
        )

    for name, api_base, api_key in configs:
        if name == model:
            return api_base, api_key

    available = ", ".join(name for name, _, _ in configs)
    raise ValueError(f"No API config for model {model!r} in {key_file}. Available models: {available}")


def _is_local_api_base(api_base: str) -> bool:
    host = (urlparse(api_base).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _configure_local_api_no_proxy(api_base: str) -> None:
    """Keep requests to a locally deployed model out of HTTP proxy routing."""
    if not _is_local_api_base(api_base):
        return
    for env_name in ("no_proxy", "NO_PROXY"):
        entries = [item.strip() for item in os.environ.get(env_name, "").split(",") if item.strip()]
        for host in ("localhost", "127.0.0.1", "::1"):
            if host not in entries:
                entries.append(host)
        os.environ[env_name] = ",".join(entries)


def _resolve_api_config(
    key_file: Path,
    model: str,
    api_base_override: Optional[str] = None,
    api_key_override: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolve an OpenAI-compatible endpoint.

    Explicit CLI values take precedence, followed by KnowledgeLog/OpenAI
    environment variables. Without an endpoint override, use openai_key.txt.
    Local loopback endpoints may use a dummy key; online endpoints require one.
    """
    api_base = (
        (api_base_override or "").strip()
        or os.environ.get("KNOWLEDGELOG_API_BASE", "").strip()
        or os.environ.get("OPENAI_API_BASE", "").strip()
        or os.environ.get("OPENAI_BASE_URL", "").strip()
    )
    explicit_key = (
        (api_key_override or "").strip()
        or os.environ.get("KNOWLEDGELOG_API_KEY", "").strip()
    )

    if not api_base:
        return _load_api_config(key_file, model)
    if not api_base.startswith(("http://", "https://")):
        raise ValueError(f"Invalid API base URL: {api_base}")

    if _is_local_api_base(api_base):
        return api_base.rstrip("/"), explicit_key or "EMPTY"

    api_key = explicit_key or os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "An online API endpoint requires --api-key, KNOWLEDGELOG_API_KEY, "
            "or OPENAI_API_KEY."
        )
    return api_base.rstrip("/"), api_key


def _chat_completions_endpoint(api_base: str) -> str:
    api_base = api_base.rstrip("/")
    if api_base.endswith("/chat/completions"):
        return api_base
    return api_base + "/chat/completions"


def _normalize_variable_placeholders(template: str) -> str:
    """Normalize all variable placeholders to sampled-example {variables} form."""
    template = re.sub(r"\{[A-Za-z_][A-Za-z0-9_]*\}", VARIABLE_PLACEHOLDER, template)
    template = template.replace("<*>", VARIABLE_PLACEHOLDER)
    return template


def _post_process_template(template: str) -> str:
    template = _normalize_variable_placeholders(template)
    template = template.replace(VARIABLE_PLACEHOLDER, "<*>")
    for reg in COMMON_REGEXES:
        template = reg.sub("<*>", template)
    template = correct_single_template(template)
    return template.replace("<*>", VARIABLE_PLACEHOLDER).strip()


def _extract_template_from_response(response: str) -> Optional[str]:
    """Extract the `...` template using the same response contract as KnowledgeLog."""
    lines = response.split("\n")
    log_template = None
    for line in lines:
        if "Log template:" in line:
            log_template = line
            break
    if log_template is None:
        for line in lines:
            if "`" in line:
                log_template = line
                break
    if log_template is None:
        return None

    start_index = log_template.find("`") + 1
    end_index = log_template.rfind("`")
    if start_index == 0 or end_index == -1:
        start_index = log_template.find('"') + 1
        end_index = log_template.rfind('"')
    if start_index != 0 and end_index != -1 and start_index < end_index:
        return log_template[start_index:end_index]
    return None


def _request_llm_content(
    messages: Sequence[Dict[str, str]],
    api_key: str,
    api_base: str,
    model: str = "gpt-3.5-turbo",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    endpoint = _chat_completions_endpoint(api_base)
    payload = {
        "model": model,
        "messages": list(messages),
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    raw_response = ""
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw_response = resp.read().decode("utf-8")
        body = json.loads(raw_response)
        choices = [
            choice["message"]["content"]
            for choice in body["choices"]
            if choice.get("finish_reason") != "length"
        ]
        if choices:
            return choices[0]
        raise RuntimeError("No complete response choices returned by the chat API.")
    except (error.HTTPError, error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
        if raw_response:
            exc.raw_response = raw_response
        raise


def _run_llm_call_with_retries(
    messages: Sequence[Dict[str, str]],
    api_key: str,
    api_base: str,
    model: str = "gpt-3.5-turbo",
    retry_times: int = DEFAULT_RETRY_TIMES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> Optional[str]:
    for attempt in range(1, retry_times + 1):
        try:
            return _request_llm_content(
                messages=messages,
                api_key=api_key,
                api_base=api_base,
                model=model,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as exc:
            print(f"Exception on API attempt {attempt}/{retry_times}: {exc}")
            if attempt < retry_times:
                time.sleep(DEFAULT_RETRY_SLEEP_SECONDS)

    print(f"Failed to get response from OpenAI after {retry_times} retries.")
    return None


def parse_template_with_llm(
    log_message: str,
    api_key: str,
    api_base: str,
    model: str = "gpt-3.5-turbo",
    retry_times: int = DEFAULT_RETRY_TIMES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> Tuple[str, bool]:
    """
    Parse one log message and return (template, llm_call_success).

    A request failure or an unparseable response uses the original log as the
    fallback template and reports llm_call_success=False.
    """
    instruction = (
    "I want you to act like an expert of log parsing. I will give you rules "
    "that you must follow and a log message delimited by backticks. You must "
    "identify and abstract all dynamic variables in logs with {variables}, then "
    "output a static log template. Please note that there might be no variables "
    "in logs. Every placeholder must be exactly {variables}; never use any other "
    "placeholder name or form. Except for replacing dynamic variables, copy every "
    "constant and punctuation mark exactly; never add, remove, correct, or rewrite "
    "them. You must print the input log's template following "
    "this exact format: 'Log template:`...`'"
)
    messages = [
        {"role": "system", "content": "You are an expert of log parsing."},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": "Sure, I can help you with log parsing."},
        {"role": "user", "content": f"Log message: `{log_message}`"},
    ]
    response = _run_llm_call_with_retries(
        messages=messages,
        api_key=api_key,
        api_base=api_base,
        model=model,
        retry_times=retry_times,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    if response is None:
        return log_message, False

    template = _extract_template_from_response(response)
    if template is not None:
        return template, True

    print("======================================")
    print("ChatGPT response format error: ")
    print(response)
    print("======================================")
    return log_message, False


def templates_match(parsed_template: str, ground_truth_template: str) -> bool:
    return parsed_template.strip() == ground_truth_template.strip()


def _default_output_json(shot: int, timestamp: Optional[str] = None) -> Path:
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_KNOWLEDGE_GAPS_RESULTS_DIR / f"{timestamp}_{shot}shot" / "IR_results.json"


def _infer_shot_from_dataset_files(dataset_files: Sequence[str], fallback_shot: int) -> int:
    detected_shots = set()
    for dataset_file in dataset_files:
        match = re.fullmatch(r"(\d+)shot", Path(dataset_file).stem)
        if match:
            detected_shots.add(int(match.group(1)))
    if len(detected_shots) > 1:
        raise ValueError(f"Mixed shot sizes are not supported in one run: {sorted(detected_shots)}")
    return next(iter(detected_shots), fallback_shot)


def identify_gaps(
    dataset_files: Sequence[str],
    api_key: str,
    api_base: str,
    model: str = "gpt-3.5-turbo",
    output_json: Optional[str] = None,
    shot: int = DEFAULT_SHOT,
) -> List[Tuple[str, str, str, str, str, bool]]:
    """
    Build IR list of mismatches:
    (dataset_name, original_log, raw_parsed_template, parsed_template,
    ground_truth_template, llm_call_success)
    """
    # Keep ground-truth template with each selected sample to avoid cross-dataset ID collision.
    output_path = Path(output_json) if output_json else _default_output_json(shot)
    sampled_items: List[Tuple[str, str, str]] = []  # (log_message, ground_truth_template, dataset_name)
    dataset_order: List[str] = []
    dataset_total = len(dataset_files)

    print(f"Output JSON: {output_path}")
    print(f"[1/3] loading sampled example files from {dataset_total} dataset files...")

    for idx, fp in enumerate(dataset_files, start=1):
        records = _load_sampled_examples_file(Path(fp))
        dataset_name = f"{Path(fp).parent.name}_{Path(fp).stem}"
        dataset_order.append(dataset_name)
        for log_message, ground_truth_template in records:
            sampled_items.append((log_message, ground_truth_template, dataset_name))
        print(
            f"[load {idx}/{dataset_total}] {dataset_name}: "
            f"logs={len(records)} unique_templates={len({template for _, template in records})} selected={len(records)}"
        )

    ir_by_content: Dict[
        Tuple[str, str, str, str],
        Tuple[str, str, str, str, str, bool],
    ] = {}
    total_pairs = len(sampled_items)
    print(f"[2/3] parsing {total_pairs} sampled logs one by one...")
    start_time = time.time()
    for processed, (log_message, gt_template, dataset_name) in enumerate(sampled_items, start=1):
        parsed_raw, llm_call_success = parse_template_with_llm(
            log_message=log_message,
            api_key=api_key,
            api_base=api_base,
            model=model,
        )

        parsed = _post_process_template(parsed_raw)
        if not templates_match(parsed, gt_template):
            deduplication_key = (log_message, parsed_raw, parsed, gt_template)
            existing = ir_by_content.get(deduplication_key)
            if existing is None:
                ir_by_content[deduplication_key] = (
                    dataset_name,
                    log_message,
                    parsed_raw,
                    parsed,
                    gt_template,
                    llm_call_success,
                )
            elif llm_call_success and not existing[-1]:
                ir_by_content[deduplication_key] = existing[:-1] + (True,)

        if processed == 1 or processed % 10 == 0 or processed == total_pairs:
            elapsed = time.time() - start_time
            print(f"[parse {processed}/{total_pairs}] IR={len(ir_by_content)} elapsed={elapsed:.1f}s")

    ir = list(ir_by_content.values())
    ir_by_dataset: Dict[str, List[Dict[str, Any]]] = {dataset_name: [] for dataset_name in dataset_order}
    for (
        dataset_name,
        log_message,
        raw_parsed_template,
        parsed_template,
        ground_truth_template,
        llm_call_success,
    ) in ir:
        ir_by_dataset.setdefault(dataset_name, []).append(
            {
                "log_message": log_message,
                "raw_parsed_template": raw_parsed_template,
                "parsed_template": parsed_template,
                "ground_truth_template": ground_truth_template,
                "llm_call_success": llm_call_success,
            }
        )
    payload = [
        {
            "dataset": dataset_name,
            "error_count": len(errors),
            "errors": errors,
        }
        for dataset_name, errors in ir_by_dataset.items()
        if errors
    ]
    print(f"[3/3] saving IR json...")
    write_json(output_path, payload)

    print(f"Dataset files: {len(dataset_files)}")
    print(f"Selected sampled logs: {len(sampled_items)}")
    print(f"IR size: {len(ir)}")
    print("Template post-processing: enabled")
    print(f"Saved IR to: {output_path}")
    return ir


def _discover_default_dataset_files(
    sampled_examples_dir: str = str(DEFAULT_SAMPLED_EXAMPLES_DIR),
    shot: int = DEFAULT_SHOT,
) -> List[str]:
    files = sorted(Path(sampled_examples_dir).glob(f"*/{shot}shot.json"))
    return [str(p) for p in files]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify parsing gaps for KnowledgeLog datasets.")
    parser.add_argument(
        "--dataset-files",
        nargs="*",
        default=None,
        help="Optional custom sampled-example files. If omitted, auto-load --sampled-examples-dir/*/{shot}shot.json.",
    )
    parser.add_argument(
        "--sampled-examples-dir",
        default=str(DEFAULT_SAMPLED_EXAMPLES_DIR),
        help="Directory containing sampled_examples organized as DATASET/*shot.json.",
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=DEFAULT_SHOT,
        choices=[8, 16, 32, 64, 128],
        help="Which sampled example file to load from each dataset, e.g. 8 loads DATASET/8shot.json.",
    )
    parser.add_argument("--key-file", default=str(DEFAULT_KEY_FILE), help="openai_key.txt path.")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM model name.")
    parser.add_argument(
        "--api-base",
        default=None,
        help=(
            "Optional OpenAI-compatible API base, e.g. http://127.0.0.1:8001/v1 "
            "or https://api.openai.com/v1. Overrides openai_key.txt."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "Optional API key override. Prefer KNOWLEDGELOG_API_KEY or OPENAI_API_KEY "
            "for online APIs; local loopback services default to EMPTY."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help=(
            "Optional IR output JSON path. By default, write to "
            "gap_analysis/results/knowledge_gaps/TIMESTAMP_SHOTshot/IR_results.json."
        ),
    )
    args = parser.parse_args()

    dataset_files = (
        args.dataset_files
        if args.dataset_files
        else _discover_default_dataset_files(args.sampled_examples_dir, args.shot)
    )
    if not dataset_files:
        raise FileNotFoundError(
            f"No dataset files found. Expected pattern: {Path(args.sampled_examples_dir) / f'*/{args.shot}shot.json'}"
        )
    run_shot = _infer_shot_from_dataset_files(dataset_files, args.shot)

    api_base, api_key = _resolve_api_config(
        key_file=Path(args.key_file),
        model=args.model,
        api_base_override=args.api_base,
        api_key_override=args.api_key,
    )
    _configure_local_api_no_proxy(api_base)
    api_mode = "local" if _is_local_api_base(api_base) else "online"
    print(f"API mode: {api_mode}")
    print(f"API base: {api_base}")
    print(f"Model: {args.model}")
    identify_gaps(
        dataset_files=dataset_files,
        api_key=api_key,
        api_base=api_base,
        model=args.model,
        output_json=args.output_json,
        shot=run_shot,
    )
