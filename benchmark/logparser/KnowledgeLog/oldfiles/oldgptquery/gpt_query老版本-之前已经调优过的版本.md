import openai
import os
import re
import time
import string
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

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
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../openai_key.txt")),
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


openai.api_base, openai.api_key = get_openai_key("../../openai_key.txt")
configure_local_api_no_proxy(openai.api_base)
print(openai.api_base)
print("OpenAI key loaded.")


# =============================================================================
# Modular prompt-level domain knowledge
# =============================================================================
# Design principle:
# 1. Keep the prompt short enough for GPT-3.5-turbo.
# 2. Use broad/general knowledge in prompt rules with source-traceable examples.
# 3. Avoid separate word-list injection; high-frequency terms appear only as examples.
# 4. Keep postprocessing limited to source-traceable placeholder-format cleanup.
# 5. Keep benchmark/template-directed preprocessing out of the prompt.
#
# Source trace:
# - Placeholder format: others_placeholder_exact, others_placeholder_braces,
#   others_named_placeholder_to_generic.
# - Dynamic span boundaries: mst_abstract_full_paths, mst_abstract_symbolic_ids,
#   km1, composite_* rules, timestamp_whole_field_abstraction,
#   network_address_* and na_* rules.
# - Semantic roles: username_under_abstraction_1, username_boundary_1, and
#   OpenSSH placeholder-boundary evidence in others_whitespace_punctuation_cleanup.
# - Literal preservation: exclusive_token_* rules, fixed-message-fragment rules,
#   and event_predicate_* rules.

CORE_PROMPT_RULES = [
    "Use exactly one placeholder form for dynamic values: {variables}.",
    "Do not output named placeholders, double braces, or semantic placeholders, such as {field.name}, {{variables}}, {ip}, or {user}.",
    "Keep the original token order, punctuation, brackets, commas, colons, equal signs, and spacing whenever possible.",
    "Do not rewrite the sentence style. Only replace dynamic spans with {variables}.",
    "Replace the whole dynamic span with one {variables} if a fixed-looking word appears inside a path, URL, host, port, username, ID, version, timestamp, request target, numeric value, or payload.",
]

STRUCTURE_PROMPT_RULES = [
    "Replace the whole path, URL, URI, host:port, IP:port, or endpoint with one {variables}. Do not leave directory, scheme, host, or port prefixes outside the placeholder.",
    "Replace one complete instance-specific identifier token with one {variables}, including job/container IDs, mixed alphanumeric IDs, versions, interface names, and hex-like IDs.",
    "Replace one encoded value after a single key or label with {variables}, including values joined by repeated delimiters and whitespace-separated payload sequences. Do not merge separately punctuated grammar arguments into one placeholder.",
    "Preserve fixed exception names, class names, method names, and component names literally when they are complete message constants.",
]

SEMANTIC_ROLE_PROMPT_RULES = [
    "Replace authentication key-value field values with {variables}; preserve the key names and equal signs.",
    "If a key-value field has an empty value, output the value as {variables} while preserving the key name and equal sign.",
    "Replace the user/account value after username cues such as for, for user, account, or login name.",
    "Preserve fixed authentication status phrases literally when they describe the event outcome, such as user unknown.",
    "Replace a complete timestamp or date-time expression with one {variables}, such as Jan 01 12:00:00, 2020-01-01 12:00:00, or Mon Feb 06 12:24:00 UTC 2006.",
    "Replace the request target and HTTP version in HTTP request lines with {variables}; preserve HTTP methods such as GET, POST, PUT, and DELETE.",
]

LITERAL_PROMPT_RULES = [
    "Preserve fixed protocol, authentication, service, and bracketed state markers literally, such as sshd, ssh2, [preauth], and [state].",
    "Preserve fixed technical marker tokens literally when they are grammar fields, such as GET, HTTPS, RAID, DELETE, DISK, SMART, SCSI, VPD, and PDT.",
    "Preserve fixed exception classes and complete fixed error phrases literally, such as java.io.IOException, No such file or directory, Connection reset by peer, and Out of memory.",
    "Preserve fixed component and service names literally when they are complete message constants, including lowerCamelCase names such as workerEnv, CamelCase names, method-like names, and fixed names inside grammar wrappers such as service(...).",
    "Preserve fixed event phrases literally when the whole phrase describes the log event, such as SendWorker, worker leaving thread, stored as bytes in memory, message on queue.",
    "Preserve fixed lifecycle, state, and result phrases literally when they describe the log event outcome, such as startup, (spawning), shutdown, Shutting down, restart, succeeded, failed, interrupted, disabled, and loaded.",
    "Replace variable values around fixed literals with {variables}, including IDs, endpoints, paths, counts, usernames, runtime process names paired with process IDs such as httpd, payload values, and key-value values, such as httpd.",
    "Replace event-like words with {variables} only when they appear as a key-value value, inside a payload, inside a username/account value, inside a path or URL, or inside a request target.",
    "Apply phrase-level decisions before token-level decisions. Preserve a complete fixed phrase even when it contains broad words such as user, worker, message, or unknown.",
]

PROMPT_GROUPS = {
    "core": CORE_PROMPT_RULES,
    "structure": STRUCTURE_PROMPT_RULES,
    "semantic_role": SEMANTIC_ROLE_PROMPT_RULES,
    "literal": LITERAL_PROMPT_RULES,
}

DEFAULT_ENABLED_PROMPT_GROUPS = env_list("LILAC_PROMPT_GROUPS", (
    "core",
    "structure",
    "semantic_role",
    "literal",
))


def build_final_domain_knowledge(
    enabled_prompt_groups: Sequence[str] = DEFAULT_ENABLED_PROMPT_GROUPS,
) -> str:
    """Build compact, modular domain knowledge for GPT-3.5-turbo."""
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
    enabled: bool = True
    note: str = ""

    def apply(self, text: str) -> str:
        if not self.enabled:
            return text
        return re.sub(self.pattern, self.replacement, text, flags=self.flags)


PREPROCESSING_RULE_GROUPS = {
    "safe_url_endpoint": [
        RegexRule(
            name="absolute_uri_with_scheme",
            pattern=(
                r"(?<![A-Za-z0-9+.-])"
                r"(?:https?|hdfs|file)://"
                r"[^\s\]\)\}\>,\"']+"
                r"(?<![.,:;])"
            ),
            replacement=VARIABLE_PLACEHOLDER,
            flags=re.IGNORECASE,
            note="Collapse complete HTTP, HDFS, or file URIs.",
        ),
        RegexRule(
            name="leading_slash_ip_port",
            pattern=r"/(?:\d{1,3}\.){3}\d{1,3}:\d{1,5}\b",
            replacement=VARIABLE_PLACEHOLDER,
            note="Collapse /IP:port endpoints.",
        ),
        RegexRule(
            name="domain_or_ip_with_required_port",
            pattern=r"\b(?:[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+|(?:\d{1,3}\.){3}\d{1,3}):\d{1,5}\b",
            replacement=VARIABLE_PLACEHOLDER,
            flags=re.IGNORECASE,
            note="Collapse domain:port or IP:port only; do not match plain dotted names.",
        ),
        # RegexRule(
        #     name="socket_local_remote",
        #     pattern=r"local=([^\s\]]+)\s+remote=([^\s\]]+)",
        #     replacement="local={variables} remote={variables}",
        #     note="Normalize local/remote endpoint values inside socket wrappers.",
        # ),
    ],
    "safe_request_fields": [
        RegexRule(
            name="quoted_http_request_line",
            pattern=r'"(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS|CONNECT|TRACE)\s+[^"\s]+\s+HTTP/[0-9.]+"',
            replacement=r'"\1 {variables} {variables}"',
            flags=re.IGNORECASE,
            note="Preserve HTTP method and collapse request target plus protocol version.",
        ),
        RegexRule(
            name="pam_auth_key_value_fields",
            pattern=r"\blogname=\S*\s+uid=\S+\s+euid=\S+\s+tty=\S+\s+ruser=\S*\s+rhost=\S+\s+user=\S+",
            replacement=(
                f"logname={VARIABLE_PLACEHOLDER} uid={VARIABLE_PLACEHOLDER} "
                f"euid={VARIABLE_PLACEHOLDER} tty={VARIABLE_PLACEHOLDER} "
                f"ruser={VARIABLE_PLACEHOLDER} rhost={VARIABLE_PLACEHOLDER} "
                f"user={VARIABLE_PLACEHOLDER}"
            ),
            note="Normalize source-traced authentication key-value field sequences before prompting.",
        ),
        RegexRule(
            name="empty_role_before_parenthesized_kv",
            pattern=(
                r"(\b(?:by|from|for|as)\s+)"
                r"\(\s*([A-Za-z][A-Za-z0-9_.-]*)\s*=\s*[^()]*\)"
            ),
            replacement=(
                rf"\1{VARIABLE_PLACEHOLDER}"
                rf"(\2={VARIABLE_PLACEHOLDER})"
            ),
            note="Normalize an omitted role before parenthesized metadata.",
        ),
    ],
    "safe_identifier": [
        RegexRule(
            name="parenthesized_root_identity",
            pattern=r"\(\s*root\s*\)",
            replacement=rf"({VARIABLE_PLACEHOLDER})",
            flags=re.IGNORECASE,
            note="Abstract a parenthesized root identity.",
        ),
        RegexRule(
            name="dotted_diagnostic_counter",
            pattern=(
                r"(\.{2,})"
                r"[-+]?(?:0x[0-9A-Fa-f]+|\d+(?:\.\d+)?)"
                r"\s*$"
            ),
            replacement=rf"\1{VARIABLE_PLACEHOLDER}",
            note="Abstract a trailing counter after a dotted diagnostic leader.",
        ),
        RegexRule(
            name="block_identifier",
            pattern=r"\bblk_-?\d+\b",
            replacement=VARIABLE_PLACEHOLDER,
            note="Collapse source-traced HDFS-style block IDs as one complete identifier.",
        ),
        RegexRule(
            name="core_dump_numeric_suffix",
            pattern=r"\bcore\.\d+\b",
            replacement=f"core.{VARIABLE_PLACEHOLDER}",
            flags=re.IGNORECASE,
            note="Preserve the core-dump artifact prefix and abstract only its numeric suffix.",
        ),
        RegexRule(
            name="attempt_container_identifier",
            pattern=r"\b(?:attempt|container)_[A-Za-z0-9_]+(?:_[A-Za-z0-9_]+)*\b",
            replacement=VARIABLE_PLACEHOLDER,
            flags=re.IGNORECASE,
            note="Collapse distributed-job attempt/container IDs.",
        ),
        RegexRule(
            name="version_token",
            pattern=r"\bv\d+(?:\.\d+)+\b",
            replacement=VARIABLE_PLACEHOLDER,
            flags=re.IGNORECASE,
            note="Collapse explicit version tokens like v1.12.",
        ),
        RegexRule(
            name="compact_quantity_value",
            pattern=(
                r"(?<![A-Za-z0-9_.-])"
                r"[-+]?\d+(?:\.\d+)?"
                r"(?:KiB|Kbytes|GHz|KB|kB|MB|GB|TB|ms|B|K|M|G|C)"
                r"(?![A-Za-z0-9_])"
            ),
            replacement=VARIABLE_PLACEHOLDER,
            note="Collapse compact numeric quantities whose units are attached.",
        ),
        RegexRule(
            name="parenthesized_secondary_size_value",
            pattern=(
                r"(\b\d+(?:\.\d+)?\s+bytes\s+)"
                r"\(\s*\d+(?:\.\d+)?\s+"
                r"(?:B|KB|MB|GB|TB|kB|bytes)\s*\)"
            ),
            replacement=rf"\1({VARIABLE_PLACEHOLDER})",
            flags=re.IGNORECASE,
            note="Collapse a parenthesized converted size following a primary byte count.",
        ),
        RegexRule(
            name="spaced_numeric_unit_value",
            pattern=(
                r"(?<![A-Za-z0-9_.-])"
                r"([-+]?\d+(?:\.\d+)?)"
                r"(\s+)"
                r"(KB|MB|GB|TB|VCPUS|VCPU|CPU)\b"
            ),
            replacement=rf"{VARIABLE_PLACEHOLDER}\2\3",
            flags=re.IGNORECASE,
            note="Abstract numeric values while preserving separated units.",
        ),
    ],
    "safe_lifetime_duration": [
        RegexRule(
            name="lifetime_duration_value",
            pattern=(
                r"(\blifetime\s+)"
                r"(?:(?:[<>]=?)?\d+(?:\.\d+)?\s*"
                r"(?:ms|msec|s|sec|secs|second|seconds)\b|\d{1,2}:\d{2}(?::\d{2})?)"
            ),
            replacement=f"\\1{VARIABLE_PLACEHOLDER}",
            flags=re.IGNORECASE,
            note="Collapse lifetime duration values only; sampled examples map lifetime values to one placeholder.",
        ),
    ],
    "safe_empty_kv_fields": [
        RegexRule(
            name="trailing_empty_key_value",
            pattern=r"(\b[A-Za-z][A-Za-z0-9_.-]*\s*=\s*)$",
            replacement=f"\\1{VARIABLE_PLACEHOLDER}",
            note="Fill an empty trailing key-value value with {variables}.",
        ),
    ],
    # "windows_path": [
    #     RegexRule(
    #         name="windows_path",
    #         pattern=r"(?<!\S)(?:[A-Za-z]:)?(?:[A-Za-z0-9_.-]+\\){2,}[A-Za-z0-9_.-]+(?!\S)",
    #         replacement=VARIABLE_PLACEHOLDER,
    #         note="Collapse Windows paths only.",
    #     ),
    # ],
}
PREPROCESSING_RULE_GROUPS["auth_empty_kv_fields"] = PREPROCESSING_RULE_GROUPS["safe_empty_kv_fields"]

_DEFAULT_ENABLED_PREPROCESSING_GROUPS = (
    "safe_url_endpoint",
    "safe_request_fields",
    "safe_identifier",
    "safe_lifetime_duration",
    "safe_empty_kv_fields",
)
DEFAULT_ENABLED_PREPROCESSING_GROUPS = env_list(
    "LILAC_PREPROCESSING_GROUPS",
    _DEFAULT_ENABLED_PREPROCESSING_GROUPS,
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


@dataclass(frozen=True)
class PostprocessRule:
    name: str
    func: Callable[[str, Optional[str]], str]
    enabled: bool = True
    note: str = ""

    def apply(self, template: str, log_message: Optional[str] = None) -> str:
        if not self.enabled:
            return template
        return self.func(template, log_message)


def normalize_template_placeholders(template: str, log_message: Optional[str] = None) -> str:
    del log_message
    template = re.sub(r"\{(?!variables\})[A-Za-z_][A-Za-z0-9_.-]*\}", VARIABLE_PLACEHOLDER, template)
    template = re.sub(r"\{\{\s*variables\s*\}\}", VARIABLE_PLACEHOLDER, template)
    return template


POSTPROCESSING_RULE_GROUPS = {
    "safe_format": [
        PostprocessRule("normalize_template_placeholders", normalize_template_placeholders),
    ],
}

_DEFAULT_ENABLED_POSTPROCESSING_GROUPS = (
    "safe_format",
)
DEFAULT_ENABLED_POSTPROCESSING_GROUPS = env_list(
    "LILAC_POSTPROCESSING_GROUPS",
    _DEFAULT_ENABLED_POSTPROCESSING_GROUPS,
)


def apply_postprocessing_group(template: str, log_message: Optional[str], group_name: str) -> str:
    for rule in POSTPROCESSING_RULE_GROUPS.get(group_name, []):
        template = rule.apply(template, log_message)
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
    global openai_api_base, openai_api_key
    openai_api_base, openai_api_key = get_openai_key("../../openai_key.txt", model)
    openai.api_base = openai_api_base
    openai.api_key = openai_api_key
    configure_local_api_no_proxy(openai_api_base)
    if is_chat_model(model):
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
            if is_chat_model(model):
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


def query_template_from_gpt(log_message, examples=None, model=DEFAULT_MODEL):
    del examples
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
    # Dataset-oriented path punctuation repair is retained as commented code
    # in post_process.py, so the default pipeline uses generic processing only.
    template = correct_single_template(template)
    static_part = template.replace(FINAL_PLACEHOLDER, "")
    punc = string.punctuation
    for s in static_part:
        if s != " " and s not in punc:
            return template, True
    print("Get a too general template. Error.")
    return "", False


def query_template_from_gpt_with_check(log_message, regs_common=None, examples=None, model=DEFAULT_MODEL):
    if regs_common is None:
        regs_common = []
    template, flag = query_template_from_gpt(log_message, examples, model)
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
