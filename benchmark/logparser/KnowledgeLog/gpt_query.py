import openai
import json
import re
import time
import string
from pathlib import Path
from typing import Optional

from .parsing_cache import ParsingCache
from .post_process import correct_single_template



VARIABLE_PLACEHOLDER = "{variables}"
FINAL_PLACEHOLDER = "<*>"
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_REQUEST_TIMEOUT = 300
DEFAULT_RETRY_TIMES = 3
DEFAULT_RETRY_SLEEP_SECONDS = 2
OPENAI_KEY_PATH = Path(__file__).resolve().parents[3] / "openai_key.txt"



def get_openai_key(model: str = DEFAULT_MODEL):
    """Read the model configuration from the project key file.

    ``openai_key.txt`` contains exactly three lines in this order: model name,
    API base, and API key.
    """
    if not OPENAI_KEY_PATH.is_file():
        raise FileNotFoundError(f"API configuration file not found: {OPENAI_KEY_PATH}")

    with OPENAI_KEY_PATH.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if len(lines) != 3:
        raise ValueError(
            f"Invalid API configuration in {OPENAI_KEY_PATH}: "
            "expected exactly three non-empty lines: model, API base, and API key"
        )

    configured_model, api_base, api_key = lines
    if configured_model != model:
        raise ValueError(
            f"Configured model {configured_model!r} does not match requested "
            f"model {model!r}"
        )
    if not api_base.startswith(("http://", "https://")):
        raise ValueError(f"Invalid API base in {OPENAI_KEY_PATH}")

    return api_base, api_key



DOMAIN_KNOWLEDGE_PATH = Path(__file__).with_name("domain_knowledge.json")

with DOMAIN_KNOWLEDGE_PATH.open("r", encoding="utf-8") as handle:
    _DOMAIN_KNOWLEDGE = json.load(handle)
CURATED_PROMPT_RULES = list(_DOMAIN_KNOWLEDGE["prompt_rules"])

def build_final_domain_knowledge() -> str:
    numbered_rules = "\n".join(
        f"{i + 1}. {rule}" for i, rule in enumerate(CURATED_PROMPT_RULES)
    )
    return "\nYou must follow these final domain knowledge rules:\n" + numbered_rules + "\n"



def build_preprocessing_rules(entries):
    rules = []
    for entry in entries:
        flags = re.IGNORECASE if entry.get("flags") == "IGNORECASE" else 0
        rules.append((re.compile(entry["pattern"], flags), entry["replacement"]))
    return rules


CURATED_PREPROCESSING_RULES = build_preprocessing_rules(
    _DOMAIN_KNOWLEDGE["preprocessing_rules"]
)

def apply_final_domain_preprocessing(log_message: str) -> str:
    query = log_message
    for pattern, replacement in CURATED_PREPROCESSING_RULES:
        query = pattern.sub(replacement, query)
    return query



def normalize_template_placeholders(template: str) -> str:
    template = re.sub(r"\{(?!variables\})[A-Za-z_][A-Za-z0-9_.-]*\}", VARIABLE_PLACEHOLDER, template)
    template = re.sub(r"\{\{\s*variables\s*\}\}", VARIABLE_PLACEHOLDER, template)
    return template


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
    api_base, api_key = get_openai_key(model)
    openai.api_base = api_base
    openai.api_key = api_key
    messages = [
        {"role": "system", "content": "You are an expert of log parsing."},
        {"role": "user", "content": instruction + knowledge},
        {"role": "assistant", "content": "Sure, I can help you with log parsing."},
        {"role": "user", "content": query},
    ]

    print("model: ", model)
    for attempt in range(1, retry_times + 1):
        try:
            answers = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                request_timeout=request_timeout,
            )
            for response in answers["choices"]:
                if response.get("finish_reason") != "length":
                    return response["message"]["content"]
            raise RuntimeError("No complete response choices returned by the chat API.")
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
    knowledge = build_final_domain_knowledge()
    preprocessed_query = apply_final_domain_preprocessing(query)
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


EMPTY_RHS_LOG_RE = re.compile(
    r"\b(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*$"
)


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
):
    if regs_common is None:
        regs_common = []
    template = normalize_template_placeholders(template)
    template = re.sub(r"\{(\w+)\}", FINAL_PLACEHOLDER, template)
    for reg in regs_common:
        template = reg.sub(FINAL_PLACEHOLDER, template)
    template = correct_single_template(template)
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
