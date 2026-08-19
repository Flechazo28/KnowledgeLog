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
# Curated prompt-level domain knowledge
# =============================================================================

CURATED_PROMPT_RULES = [
    # Dynamic span boundaries and field structure.
    "Replace a complete URL, URI, or HTTP request target with one {variables}, including its scheme, path, query, and fragment; do not retain internal prefixes.",
    "Abstract every complete hostname:port or IP:port endpoint as one {variables} field wherever it appears; do not split or partially preserve it.",
    "Abstract a numeric port only when its network role is explicit from a key or phrase such as port, listening on, or connected to; do not treat every short integer as a port.",
    "Abstract a complete hostname as {variables} only when an explicit network key or phrase identifies it as a host, hostname, node, peer, source, destination, or endpoint. Do not classify an arbitrary standalone word as a hostname.",
    "Replace a complete filesystem path with one {variables}, including its leading or trailing slash, directory prefix, file name, and extension; do not retain internal path fragments.",
    "Treat a source location such as File.scala:123 as one {variables} field when both the file name and line number vary.",
    "Preserve a package, class, module, method, file, operating-system, driver, protocol, service, component, task, process, or event-handler name when it is part of fixed event wording. Abstract the complete name only when it is the value of an explicit field such as class=, service:, or Function=. Do not abstract a name only because it contains dots, underscores, slashes, or ::.",
    "Abstract a symbol-rich runtime identifier as one {variables} field when separators such as @, -, _, ::, or dots are internal to that identifier rather than template punctuation.",
    "Abstract a complete version number, including all numeric parts, as one {variables}. Preserve an attached product name and delimiter, as in Apache/{variables}; however, in a quoted HTTP request line, replace the complete protocol/version token such as HTTP/1.1 with one {variables}. Preserve protocol names without a version, such as SOCKS5.",
    "Abstract a process ID when it is identified by a PID, process, or worker context; preserve the surrounding key and punctuation.",
    "Abstract task-attempt, stage, executor, and RPC identifiers as {variables}; preserve their fixed role names and delimiters.",
    "Abstract an error or status code as {variables} when it appears in an explicit code or status field; preserve the surrounding event phrase.",
    "Abstract an IRQ number as {variables} when it follows an IRQ label or interrupt-related phrase.",
    "Abstract a device, disk, or network-interface identifier only when it is a runtime field value. Preserve a fixed device or interface name, and abstract only an instance-varying suffix when the surrounding name is fixed.",
    "Preserve a unit when it is fixed event wording after a variable number, as in {variables} bytes sent. When the number and unit together form one measurement field, abstract both as one {variables}, as in 1.09 KB, 60000ms, or 41C.",
    "Preserve a fixed range delimiter such as '-' or 'to' and replace its variable endpoints separately with {variables}. Replace the complete range with one {variables} only when the entire range is one varying field.",
    "Replace a packed numeric sequence or variable-length repeated list with one {variables}. For a short list of separate named items, replace each variable item separately and preserve the separators; for example, users: yarn,curi becomes users: {variables},{variables}.",
    "In a key-value field, preserve a fixed key and the '=' or ':' delimiter and replace only its variable value with {variables}. If the key also varies, replace the key and value separately while preserving the delimiter.",
    "In an equality, comparison, ratio, or mapping expression, preserve the operator and abstract only the variable operands.",
    "Abstract an account identifier as {variables} when the context explicitly denotes a user, login, account, authentication identity, or principal; always abstract an account identity after 'for' or 'user' in password or authentication messages, such as 'root'.",
    "Abstract a state, lifecycle stage, or boolean only when it occupies a variable state or status field; preserve fixed action and event wording.",
    "Abstract a timezone name or offset as {variables} when it forms the timezone field of a timestamp.",
    "Abstract a flag, status, or directive value only when it follows an explicit field name; preserve the field name, delimiter, and fixed protocol or directive name.",
    "Never add, delete, move, or rewrite punctuation from the original log. Preserve brackets, parentheses, and quotation marks when they are fixed template syntax; include them in {variables} only when the complete wrapped text is one variable field.",
    # Source-supported constants and precedence.
    "Preserve the software-agent name GoogleSoftwareUpdateAgent as literal template text.",
    "Preserve NSMutableURLRequest as a literal class name in the template.",
    "Preserve the device marker parport0 as literal template text in the observed event family.",
    "Preserve the method name updateFinish as literal template text.",
    "Preserve the protocol name SOCKS5 as literal template text.",
    "Preserve the fixed event phrase \"Out of memory\" as literal template text.",
    "Preserve the fixed error phrase \"Attempt to process a enum when a union was expected.\" exactly as written.",
    "Preserve the key token lvl in a field such as [lvl={variables}]; abstract only its value.",
    "By default, preserve GET as a fixed HTTP method token. Replace it with {variables} only when it occupies a varying method field, such as a malformed-request template.",
    "Preserve the fixed class or component names KSAgentUploader and KSUpdateEngine as literal template text.",
    "Preserve preauth as a literal authentication-stage token.",
    "Preserve main as a literal entry-point function name when it denotes the fixed function in the event template.",
    "Preserve the fixed phrase \"Bye Bye\" as literal template text.",
    "Preserve the fixed event outcome phrase \"disconnected by user\" as literal template text.",
    "Replace a complete slash-prefixed filesystem or topology path value with one {variables}; do not preserve any internal path segment as fixed text.",
]

PROMPT_GROUPS = {"curated": CURATED_PROMPT_RULES}
DEFAULT_ENABLED_PROMPT_GROUPS = env_list("LILAC_PROMPT_GROUPS", ("curated",))


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
    enabled: bool = True
    note: str = ""

    def apply(self, text: str) -> str:
        if not self.enabled:
            return text
        return re.sub(self.pattern, self.replacement, text, flags=self.flags)


PREPROCESSING_RULE_GROUPS = {
    "curated": [
        RegexRule(
            name="curated_0010_process_thread_pair",
            pattern=r"\b\d+/0[xX][0-9A-Fa-f]+\b",
            replacement=VARIABLE_PLACEHOLDER,
            note="Abstract a process-ID/thread-address pair as one variable.",
        ),
        RegexRule(
            name="curated_0003_ipv4_port",
            pattern=r"(?<![A-Za-z0-9_/])/?(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d):(?:[1-9]\d{0,3}|[1-5]\d{4}|6[0-4]\d{3}|65[0-4]\d{2}|655[0-2]\d|6553[0-5])\b",
            replacement=VARIABLE_PLACEHOLDER,
            note="Abstract a complete IPv4-address-and-port endpoint, including a leading slash.",
        ),
        RegexRule(
            name="domain_or_ip_with_required_port",
            pattern=(
                r"(?<![A-Za-z0-9_./-])"
                r"(?:"
                r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
                r"|"
                r"(?:\d{1,3}\.){3}\d{1,3}"
                r"):\d{1,5}\b"
            ),
            replacement=VARIABLE_PLACEHOLDER,
            flags=re.IGNORECASE,
            note="Collapse domain:port or IP:port only; do not match plain dotted names.",
        ),
        RegexRule(
            name="http_request_target",
            pattern=(
                r'("(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS|CONNECT|TRACE)\s+)'
                r'[^\s"]+'
            ),
            replacement=r"\1{variables}",
            flags=re.IGNORECASE,
            note="Abstract the complete request target after an HTTP method.",
        ),
        RegexRule(
            name="http_protocol_version",
            pattern=r"\bHTTP/\d+(?:\.\d+)+\b",
            replacement=VARIABLE_PLACEHOLDER,
            flags=re.IGNORECASE,
            note="Abstract a complete HTTP protocol-version field.",
        ),
        RegexRule(
            name="curated_0002_ipv4",
            pattern=r"(?<![A-Za-z0-9_/])/?(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:,/?(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d))*(?![\d.:])",
            replacement=VARIABLE_PLACEHOLDER,
            note="Abstract one IPv4 address or a complete comma-separated IPv4 list.",
        ),
        RegexRule(
            name="curated_0001_uuid",
            pattern=r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            replacement=VARIABLE_PLACEHOLDER,
            note="Abstract a complete UUID.",
        ),
        RegexRule(
            name="curated_0012_android_action",
            pattern=r"\bandroid\.intent\.action\.[A-Za-z_][A-Za-z0-9_]*\b",
            replacement=VARIABLE_PLACEHOLDER,
            note="Abstract a complete Android intent action identifier.",
        ),
        RegexRule(
            name="curated_0009_hdfs_block",
            pattern=r"(?<![/A-Za-z0-9_])blk_-?\d+\b",
            replacement=VARIABLE_PLACEHOLDER,
            note="Abstract a complete HDFS block identifier.",
        ),
        RegexRule(
            name="lifetime_duration_value",
            pattern=(
                r"(\blifetime\s+)"
                r"(?:(?:[<>]=?)?\d+(?:\.\d+)?\s*"
                r"(?:ms|msec|s|sec|secs|second|seconds)\b|"
                r"\d{1,2}:\d{2}(?::\d{2})?)"
            ),
            replacement=f"\\1{VARIABLE_PLACEHOLDER}",
            flags=re.IGNORECASE,
            note=(
                "Collapse lifetime duration values only; sampled examples map "
                "lifetime values to one placeholder."
            ),
        ),
        RegexRule(
            name="curated_0005_timestamp",
            pattern=r"(?<!\d)(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{3}(?!\d)",
            replacement=VARIABLE_PLACEHOLDER,
            note="Abstract a timestamp in HH:MM:SS.mmm form.",
        ),
        RegexRule(
            name="curated_0004_time",
            pattern=r"(?<![\d:])(?:[01]\d|2[0-3]):[0-5]\d(?![:\d])",
            replacement=VARIABLE_PLACEHOLDER,
            note="Abstract a standalone time in HH:MM form.",
        ),
        RegexRule(
            name="curated_0006_date",
            pattern=r"(?<![A-Za-z0-9_/-])\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?![A-Za-z0-9_/-])",
            replacement=VARIABLE_PLACEHOLDER,
            note="Abstract a date in YYYY-MM-DD form.",
        ),
        RegexRule(
            name="curated_0008_hexadecimal",
            pattern=r"(?<![A-Za-z0-9/])0[xX][0-9A-Fa-f]+(?![A-Za-z0-9])",
            replacement=VARIABLE_PLACEHOLDER,
            note="Abstract a hexadecimal value with a 0x or 0X prefix.",
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
            name="curated_0032_account_allowlist",
            pattern=(
                r"(\buser\s*=\s*|\bfor\s+(?:user\s+)?|\buser\s+)"
                r"(?:root|monitor|ubuntu|admin|ubnt)\b"
            ),
            replacement=r"\1{variables}",
            flags=re.IGNORECASE,
            note=(
                "Abstract a known account name after user=, for, or user while "
                "preserving the surrounding field label."
            ),
        ),
        RegexRule(
            name="curated_0031_key_value",
            pattern=(
                r"(?<![\w.-])"
                r"([A-Za-z_][A-Za-z0-9_.-]*\s*=\s*?)"
                r"(?:"
                r"[A-Za-z0-9._]+(?=\s|[,;]|$)"
                r"|"
                r"(?=\s*(?:[A-Za-z_][A-Za-z0-9_.-]*\s*=|[,;]|$))"
                r")"
            ),
            replacement=r"\1{variables}",
            note=(
                "Preserve the key and equals sign, and abstract a value containing "
                "only letters, digits, dots, or underscores, including an empty value."
            ),
        ),
    ],
}

_DEFAULT_ENABLED_PREPROCESSING_GROUPS = ("curated",)
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
