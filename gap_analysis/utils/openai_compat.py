from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


def read_openai_key(key_file: Path, model: str) -> Tuple[Optional[str], str]:
    """Read one-line, two-line, or multi-model OpenAI-compatible key files."""
    lines = [line.strip() for line in key_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty key file: {key_file}")
    if len(lines) == 1:
        return None, lines[0]
    if lines[0].startswith(("http://", "https://")):
        return lines[0].rstrip("/"), lines[1]

    configs = []
    for index in range(0, len(lines) - 2, 3):
        name, api_base, api_key = lines[index : index + 3]
        if api_base.startswith(("http://", "https://")):
            configs.append((name, api_base.rstrip("/"), api_key))

    for name, api_base, api_key in configs:
        if name == model:
            return api_base, api_key
    for name, api_base, api_key in configs:
        if model.startswith(name) or name.startswith(model):
            return api_base, api_key

    raise ValueError(
        f"No API config for model {model!r} in {key_file}. "
        f"Available model entries: {[name for name, _, _ in configs]}"
    )


def call_chat_completion(
    messages: Sequence[Dict[str, str]],
    *,
    model: str,
    api_base: Optional[str],
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: float = 300.0,
) -> str:
    """Call Chat Completions with either pre-1.0 or current OpenAI SDKs."""
    import openai

    try:
        major_version = int(str(getattr(openai, "__version__", "0")).split(".", 1)[0])
    except ValueError:
        major_version = 0

    if major_version < 1:
        openai.api_key = api_key
        if api_base:
            openai.api_base = api_base
        response = openai.ChatCompletion.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            request_timeout=timeout,
        )
        return response["choices"][0]["message"]["content"]

    from openai import OpenAI

    client_options = {"api_key": api_key, "timeout": timeout}
    if api_base:
        client_options["base_url"] = api_base
    client = OpenAI(**client_options)
    response = client.chat.completions.create(
        model=model,
        messages=list(messages),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
