import openai
import json
import os
import re
import signal
import time
import string
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .parsing_cache import ParsingCache
from .post_process import correct_single_template


# =============================================================================
# Basic configuration
# =============================================================================

VARIABLE_PLACEHOLDER = "{variables}"
FINAL_PLACEHOLDER = "<*>"
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_REQUEST_TIMEOUT = 300
DEFAULT_RETRY_TIMES = 3
DEFAULT_RETRY_SLEEP_SECONDS = 2
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class APICallWallClockTimeout(TimeoutError):
    """Raised when an API attempt exceeds its wall-clock deadline."""


@contextmanager
def api_call_wall_clock_timeout(seconds: int):
    """Enforce a real deadline even if requests/urllib3 stops honoring timeout.

    KnowledgeLog performs model calls in the main thread of its parser child on
    Linux, where SIGALRM can safely interrupt a stuck Python/network code path.
    Keep a no-op fallback for platforms or threads without setitimer support.
    """
    if not hasattr(signal, "setitimer"):
        yield
        return

    def handle_timeout(_signum, _frame):
        raise APICallWallClockTimeout(
            f"API call exceeded the {seconds}-second wall-clock deadline"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def reset_openai_http_session():
    """Discard OpenAI 0.x's thread-local requests session before each attempt."""
    try:
        from openai.api_requestor import _thread_context
    except ImportError:
        return
    session = getattr(_thread_context, "session", None)
    if session is not None:
        session.close()
        delattr(_thread_context, "session")
    if hasattr(_thread_context, "session_create_time"):
        delattr(_thread_context, "session_create_time")


def env_list(name: str, default: Sequence[str]) -> Tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return tuple(default)
    raw = raw.strip()
    if not raw or raw.lower() in {"0", "false", "no", "off", "none", "empty"}:
        return ()
    return tuple(item.strip() for item in re.split(r"[,;:\s]+", raw) if item.strip())

# =============================================================================
# OpenAI key loading
# =============================================================================


def get_openai_key(file_path: str, model: str = DEFAULT_MODEL):
    candidates = [
        file_path,
        PROJECT_ROOT / "openai_key.txt",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, "r") as file:
                lines = [line.strip() for line in file if line.strip()]
            if len(lines) < 2:
                raise ValueError(f"Invalid OpenAI key file format: {candidate}")

            # Support multiple blocks: model, API base, API key.
            for index in range(0, len(lines) - 2, 3):
                configured_model, api_base, key_str = lines[index:index + 3]
                if configured_model == model and api_base.startswith("http"):
                    return api_base, key_str

            # Backward-compatible two-line format: API base, API key.
            if lines[0].startswith("http"):
                return lines[0], lines[1]

            raise ValueError(
                f"No API configuration found for model {model!r} in {candidate}"
            )
    raise FileNotFoundError(f"Cannot find OpenAI key file from candidates: {candidates}")


def configure_local_api_no_proxy(api_base: str):
    """Ensure localhost model requests are not routed through an HTTP proxy."""
    if not api_base.startswith(("http://127.0.0.1", "http://localhost")):
        return
    for env_name in ("no_proxy", "NO_PROXY"):
        entries = [item.strip() for item in os.environ.get(env_name, "").split(",") if item.strip()]
        for host in ("localhost", "127.0.0.1"):
            if host not in entries:
                entries.append(host)
        os.environ[env_name] = ",".join(entries)


# =============================================================================
# Curated prompt-level domain knowledge
# =============================================================================

DOMAIN_KNOWLEDGE_PATH = Path(__file__).with_name("domain_knowledge.json")

def load_domain_knowledge(path: Path = DOMAIN_KNOWLEDGE_PATH):
    """Load and validate data-only prompt and preprocessing knowledge."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load domain knowledge from {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Domain knowledge must be a JSON object.")
    expected_keys = {"prompt_rules", "preprocessing_rules"}
    if set(payload) != expected_keys:
        raise ValueError(
            "Domain knowledge keys must be exactly "
            f"{sorted(expected_keys)}; got {sorted(payload)}"
        )
    prompt_rules = payload["prompt_rules"]
    preprocessing_rules = payload["preprocessing_rules"]
    if not isinstance(prompt_rules, list) or not prompt_rules:
        raise ValueError("prompt_rules must be a non-empty list.")
    if not all(isinstance(rule, str) and rule.strip() for rule in prompt_rules):
        raise ValueError("Every prompt rule must be a non-empty string.")
    if not isinstance(preprocessing_rules, list) or not preprocessing_rules:
        raise ValueError("preprocessing_rules must be a non-empty list.")
    return payload


_DOMAIN_KNOWLEDGE = load_domain_knowledge()
CURATED_PROMPT_RULES = list(_DOMAIN_KNOWLEDGE["prompt_rules"])

PROMPT_GROUPS = {"curated": CURATED_PROMPT_RULES}
DEFAULT_ENABLED_PROMPT_GROUPS = env_list("KNOWLEDGELOG_PROMPT_GROUPS", ("curated",))


def build_final_domain_knowledge(
    enabled_prompt_groups: Sequence[str] = DEFAULT_ENABLED_PROMPT_GROUPS,
) -> str:
    """Build the source-traceable curated prompt knowledge."""
    rules: List[str] = []
    for group_name in enabled_prompt_groups:
        rules.extend(PROMPT_GROUPS.get(group_name, []))

    numbered_rules = "\n".join(f"{i + 1}. {rule}" for i, rule in enumerate(rules))
    return "\nYou must follow these final domain knowledge rules:\n" + numbered_rules + "\n"


# =============================================================================
# Modular preprocessing
# =============================================================================
# Avoid broad regexes that match dotted non-network tokens such as java.io.IOException,
# zookeeper.version, os.name, user.home. Endpoints must have a scheme or an explicit port.


@dataclass(frozen=True)
class RegexRule:
    name: str
    pattern: str
    replacement: str
    flags: int = 0

    def apply(self, text: str) -> str:
        return re.sub(self.pattern, self.replacement, text, flags=self.flags)


SUPPORTED_REGEX_FLAGS = {
    "IGNORECASE": re.IGNORECASE,
}


def build_preprocessing_rules(entries):
    """Construct RegexRule objects from the minimal JSON representation."""
    rules = []
    names = set()
    required_keys = {"name", "pattern", "replacement"}
    allowed_keys = required_keys | {"flags"}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"preprocessing_rules[{index}] must be an object.")
        keys = set(entry)
        missing = required_keys - keys
        extra = keys - allowed_keys
        if missing or extra:
            raise ValueError(
                f"Invalid preprocessing rule at index {index}: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        name = entry["name"]
        pattern = entry["pattern"]
        replacement = entry["replacement"]
        if not all(isinstance(value, str) for value in (name, pattern, replacement)):
            raise ValueError(
                f"name, pattern, and replacement must be strings at index {index}."
            )
        if not name or not pattern:
            raise ValueError(f"Rule name and pattern cannot be empty at index {index}.")
        if name in names:
            raise ValueError(f"Duplicate preprocessing rule name: {name}")
        names.add(name)

        flag_name = entry.get("flags")
        if flag_name is None:
            flags = 0
        elif flag_name in SUPPORTED_REGEX_FLAGS:
            flags = SUPPORTED_REGEX_FLAGS[flag_name]
        else:
            raise ValueError(
                f"Unsupported regex flag {flag_name!r} in preprocessing rule {name!r}."
            )
        try:
            re.compile(pattern, flags=flags)
        except re.error as exc:
            raise ValueError(
                f"Invalid regex pattern in preprocessing rule {name!r}: {exc}"
            ) from exc
        rules.append(
            RegexRule(
                name=name,
                pattern=pattern,
                replacement=replacement,
                flags=flags,
            )
        )
    return rules


PREPROCESSING_RULE_GROUPS = {
    "curated": build_preprocessing_rules(
        _DOMAIN_KNOWLEDGE["preprocessing_rules"]
    ),
}

DEFAULT_ENABLED_PREPROCESSING_GROUPS = env_list(
    "KNOWLEDGELOG_PREPROCESSING_GROUPS",
    ("curated",),
)


def apply_preprocessing_group(text: str, group_name: str) -> str:
    for rule in PREPROCESSING_RULE_GROUPS.get(group_name, []):
        text = rule.apply(text)
    return text


def apply_final_domain_preprocessing(
    log_message: str,
    enabled_groups: Sequence[str] = DEFAULT_ENABLED_PREPROCESSING_GROUPS,
) -> str:
    query = log_message
    for group_name in enabled_groups:
        query = apply_preprocessing_group(query, group_name)
    return query


# =============================================================================
# Modular postprocessing
# =============================================================================
# Default postprocessing is intentionally limited to output-format cleanup.
# Domain-specific content changes should happen through prompt rules or generic
# preprocessing.


def normalize_template_placeholders(
    template: str,
    _log_message: Optional[str] = None,
) -> str:
    template = re.sub(r"\{(?!variables\})[A-Za-z_][A-Za-z0-9_.-]*\}", VARIABLE_PLACEHOLDER, template)
    template = re.sub(r"\{\{\s*variables\s*\}\}", VARIABLE_PLACEHOLDER, template)
    return template


POSTPROCESSING_RULE_GROUPS = {
    "safe_format": [normalize_template_placeholders],
}

DEFAULT_ENABLED_POSTPROCESSING_GROUPS = env_list(
    "KNOWLEDGELOG_POSTPROCESSING_GROUPS",
    ("safe_format",),
)


def apply_postprocessing_group(template: str, log_message: Optional[str], group_name: str) -> str:
    for rule in POSTPROCESSING_RULE_GROUPS.get(group_name, []):
        template = rule(template, log_message)
    return template


def apply_final_domain_postprocessing(
    template: str,
    log_message: Optional[str] = None,
    enabled_groups: Sequence[str] = DEFAULT_ENABLED_POSTPROCESSING_GROUPS,
) -> str:
    for group_name in enabled_groups:
        template = apply_postprocessing_group(template, log_message, group_name)
    return template


# =============================================================================
# OpenAI inference and parser interface
# =============================================================================


def is_chat_model(model: str) -> bool:
    normalized_model = model.lower()
    return any(name in normalized_model for name in ("turbo", "gpt", "qwen", "glm", "llama"))


def infer_llm(
    instruction: str,
    knowledge: str,
    query: str,
    log_message: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    retry_times: int = DEFAULT_RETRY_TIMES,
):
    # Select the API credentials that correspond to the requested model.
    api_base, api_key = get_openai_key(PROJECT_ROOT / "openai_key.txt", model)
    openai.api_base = api_base
    openai.api_key = api_key
    configure_local_api_no_proxy(api_base)
    chat_model = is_chat_model(model)
    if chat_model:
        messages = [
            {"role": "system", "content": "You are an expert of log parsing."},
            {"role": "user", "content": instruction + knowledge},
            {"role": "assistant", "content": "Sure, I can help you with log parsing."},
            {"role": "user", "content": query},
        ]
    else:
        messages = f"{instruction}\nPlease parse the following log message:\n{query}\n"

    print("model: ", model)
    for attempt in range(1, retry_times + 1):
        try:
            # A fresh connection prevents a stale keep-alive socket from
            # spinning in CLOSE_WAIT after many sequential template queries.
            reset_openai_http_session()
            if chat_model:
                with api_call_wall_clock_timeout(request_timeout):
                    answers = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False,
                        request_timeout=request_timeout,
                    )
                choices = [
                    response["message"]["content"]
                    for response in answers["choices"]
                    if response.get("finish_reason") != "length"
                ]
                if choices:
                    return choices[0]
                raise RuntimeError("No complete response choices returned by the chat API.")
            with api_call_wall_clock_timeout(request_timeout):
                response = openai.Completion.create(
                    model=model,
                    prompt=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout=request_timeout,
                )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            print(f"Exception on API attempt {attempt}/{retry_times}: {e}")
            if attempt < retry_times:
                time.sleep(DEFAULT_RETRY_SLEEP_SECONDS)

    print(f"Failed to get response from OpenAI after {retry_times} retries.")
    return "Log message: `{}`".format(log_message)


def get_response_from_openai_key(
    query: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    enabled_prompt_groups: Sequence[str] = DEFAULT_ENABLED_PROMPT_GROUPS,
    enabled_preprocessing_groups: Sequence[str] = DEFAULT_ENABLED_PREPROCESSING_GROUPS,
):
    original_log = query
    instruction = (
        "I want you to act like an expert of log parsing. I will give you rules "
        "that you must follow and a log message delimited by backticks. You must "
        "identify and abstract all dynamic variables in logs with {variables}, then "
        "output a static log template. Please note that there might be no variables "
        "in logs. You must print the input log's template following this exact "
        "format: 'Log template:`...`'"
    )
    knowledge = build_final_domain_knowledge(
        enabled_prompt_groups=enabled_prompt_groups,
    )
    preprocessed_query = apply_final_domain_preprocessing(
        query,
        enabled_groups=enabled_preprocessing_groups,
    )
    question = "Log message: `{}`".format(preprocessed_query)
    return infer_llm(
        instruction,
        knowledge,
        question,
        original_log,
        model=model,
        temperature=temperature,
        max_tokens=DEFAULT_MAX_TOKENS,
        request_timeout=DEFAULT_REQUEST_TIMEOUT,
    )


def _extract_template_from_response(response: str) -> Optional[str]:
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


def query_template_from_gpt(log_message, model=DEFAULT_MODEL):
    if len(log_message.split()) == 1:
        return log_message, False
    response = get_response_from_openai_key(log_message, model=model)
    template = _extract_template_from_response(response)
    if template is not None:
        return template, True

    print("======================================")
    print("ChatGPT response format error: ")
    print(response)
    print("======================================")
    return log_message, False


ROLE_SPAN_RE = re.compile(r"^[A-Za-z]+(?:\s+[A-Za-z]+)*$")
WORKER_THREAD_RE = re.compile(r"(?:worker|thread)", re.IGNORECASE)
IDENTITY_CONTEXT_RE = re.compile(
    r"\b(?:user|login|account|principal|for)\s+$",
    re.IGNORECASE,
)
EMPTY_RHS_LOG_RE = re.compile(
    r"\b(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*$"
)


def restore_fixed_role_spans(
    template: str,
    log_message: Optional[str],
) -> str:
    """Restore fixed worker/thread role names that the LLM over-abstracted."""
    if not log_message or FINAL_PLACEHOLDER not in template:
        return template

    parts = template.split(FINAL_PLACEHOLDER)
    pattern = "(.*?)".join(re.escape(part) for part in parts)
    match = re.fullmatch(pattern, log_message)
    if match is None:
        return template

    output = [parts[0]]
    for index, (span, suffix) in enumerate(
        zip(match.groups(), parts[1:]),
        start=1,
    ):
        prefix = log_message[:match.start(index)]
        preserve = bool(
            ROLE_SPAN_RE.fullmatch(span)
            and WORKER_THREAD_RE.search(span)
            and not IDENTITY_CONTEXT_RE.search(prefix)
        )
        output.extend([span if preserve else FINAL_PLACEHOLDER, suffix])
    return "".join(output)


def normalize_empty_rhs_placeholder(
    template: str,
    log_message: Optional[str],
) -> str:
    """Normalize the placeholder boundary only for a truly empty key=value RHS."""
    if not log_message:
        return template

    match = EMPTY_RHS_LOG_RE.search(log_message)
    if match is None:
        return template

    key = re.escape(match.group("key"))
    return re.sub(
        rf"(\b{key}\s*=)\s*{re.escape(FINAL_PLACEHOLDER)}\s*$",
        lambda result: f"{result.group(1)}{FINAL_PLACEHOLDER}",
        template,
    )


def post_process_template(
    template,
    regs_common=None,
    log_message=None,
    enabled_postprocessing_groups: Sequence[str] = DEFAULT_ENABLED_POSTPROCESSING_GROUPS,
):
    if regs_common is None:
        regs_common = []
    template = apply_final_domain_postprocessing(
        template,
        log_message,
        enabled_groups=enabled_postprocessing_groups,
    )
    template = re.sub(r"\{(\w+)\}", FINAL_PLACEHOLDER, template)
    for reg in regs_common:
        template = reg.sub(FINAL_PLACEHOLDER, template)
    template = correct_single_template(template)
    template = restore_fixed_role_spans(template, log_message)
    template = normalize_empty_rhs_placeholder(template, log_message)
    static_part = template.replace(FINAL_PLACEHOLDER, "")
    if any(character != " " and character not in string.punctuation for character in static_part):
        return template, True
    print("Get a too general template. Error.")
    return "", False


def query_template_from_gpt_with_check(log_message, regs_common=None, model=DEFAULT_MODEL):
    if regs_common is None:
        regs_common = []
    template, flag = query_template_from_gpt(log_message, model)
    if len(template) == 0 or not flag:
        print("ChatGPT error")
    else:
        tree = ParsingCache()
        template, flag = post_process_template(template, regs_common, log_message)
        if flag:
            tree.add_templates(template)
            if tree.match_event(log_message)[0] == "NoMatch":
                print("==========================================================")
                print(log_message)
                print("ChatGPT template wrong: cannot match itself! And the wrong template is : ")
                print(template)
                print("==========================================================")
            else:
                return template, True
    return post_process_template(log_message, regs_common, log_message)
