"""Cross provider token counter."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any, Literal

import requests
import tiktoken
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately

from ..const import (  # noqa: TID252
    CONF_GEMINI_API_KEY,
    CONF_OLLAMA_CHAT_URL,
    CONF_OLLAMA_URL,
    OLLAMA_EXACT_TOKEN_COUNT,
)

OPENAI_PREFIXES: tuple[str, ...] = ("gpt",)

LOGGER = logging.getLogger(__name__)

# ---- Providers ----
Provider = Literal["openai", "gemini", "ollama"]

# ---- Message shapes ----
MessageLike = BaseMessage | str | tuple[str, str] | list[str] | Mapping[str, Any]
MessageLikeRepresentation = str | tuple[str, str] | list[str] | BaseMessage


def _flatten_text_content(content: Any) -> str:
    """
    Flatten message content to plain text.

    Supports:
      - str
      - list[str]
      - list[Mapping] with OpenAI-style parts:
          {"type":"text","text": "..."} or {"text": "..."}
    Non-textual parts are ignored for counting.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if (
                    part.get("type") == "text" and isinstance(part.get("text"), str)
                ) or isinstance(part.get("text"), str):
                    out.append(part["text"])
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out)
    return str(content)


def _to_role_content_pairs(messages: Sequence[MessageLike]) -> list[tuple[str, str]]:
    """
    Normalize arbitrary message-like inputs into (role, content) pairs.

    Role defaults to "user" when unspecified.
    """
    out: list[tuple[str, str]] = []
    msg_len = 2  # enforces the tuple shape (role, content)
    for m in messages:
        if isinstance(m, BaseMessage):
            role_map = {
                "human": "user",
                "ai": "assistant",
                "system": "system",
                "tool": "tool",
                "function": "tool",
            }
            role = role_map.get(m.type, m.type)
            out.append((str(role), _flatten_text_content(m.content)))
            continue

        if isinstance(m, Mapping):
            role = str(m.get("role") or m.get("type") or "user")
            out.append((role, _flatten_text_content(m.get("content"))))
            continue

        if (
            isinstance(m, tuple)
            and len(m) == msg_len
            and all(isinstance(x, str) for x in m)
        ):
            out.append((m[0], m[1]))
            continue

        if isinstance(m, list) and all(isinstance(x, str) for x in m):
            out.append(("user", "\n".join(m)))
            continue

        if isinstance(m, str):
            out.append(("user", m))
            continue

        msg = f"Unsupported message type: {type(m)}"
        raise TypeError(msg)
    return out


def _concat_messages_for_count(messages: Sequence[MessageLike]) -> str:
    pairs = _to_role_content_pairs(messages)
    return "\n\n".join(f"{role}:\n{content}" for role, content in pairs)


@lru_cache(maxsize=16)
def _pick_encoding_for_model(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def _count_tokens_tiktoken(
    messages: Iterable[MessageLike],
    model: str,
) -> int:
    """Count tokens using tiktoken with a simple role+content layout."""
    enc = _pick_encoding_for_model(model)
    total = 0
    for role, content in _to_role_content_pairs(list(messages)):
        total += len(enc.encode(f"{role}:\n"))
        if content:
            total += len(enc.encode(content))
    return total


def _count_gemini_tokens(
    messages: Sequence[MessageLike],
    model: str,
    gemini_api_key: str,
    *,
    timeout: float = 10.0,
    endpoint_base: str = "https://generativelanguage.googleapis.com",
) -> int:
    """Count tokens for Gemini via REST countTokens."""
    if not gemini_api_key:
        msg = "Gemini API key required for token counting."
        raise RuntimeError(msg)

    url = f"{endpoint_base.rstrip('/')}/v1beta/models/{model}:countTokens"
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": _concat_messages_for_count(messages)}]}
        ]
    }

    try:
        response = requests.post(
            url, params={"key": gemini_api_key}, json=payload, timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        msg = f"Gemini countTokens failed: {exc!s}"
        raise RuntimeError(msg) from exc

    val = data.get("totalTokens", data.get("total_tokens"))
    if isinstance(val, int):
        return val
    msg = f"Unexpected Gemini countTokens response: {data}"
    raise RuntimeError(msg)


def _count_ollama_tokens_via_tokenize(
    prompt: str, *, model: str, base_url: str, timeout: float
) -> int | None:
    """Fast path using /api/tokenize. Return None if not available."""
    url = f"{base_url.rstrip('/')}/api/tokenize"
    try:
        r = requests.post(url, json={"model": model, "prompt": prompt}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        tokens = data.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
    except requests.RequestException:
        return None
    return None


def _count_ollama_tokens(
    messages: Sequence[MessageLike],
    model: str,
    base_url: str,
    *,
    timeout: float = 60.0,
    options: Mapping[str, Any] | None = None,
) -> int:
    """
    Count prompt tokens for Ollama.

    Tries /api/tokenize first (fast, no runner).
    Falls back to /api/generate with num_predict=0 (exact, slower).
    """
    combined = _concat_messages_for_count(messages)

    fast = _count_ollama_tokens_via_tokenize(
        combined, model=model, base_url=base_url, timeout=timeout
    )
    if isinstance(fast, int):
        return fast

    url = f"{base_url.rstrip('/')}/api/generate"
    opts: dict[str, Any] = dict(options or {})
    opts["num_predict"] = 0

    try:
        r = requests.post(
            url,
            json={"model": model, "prompt": combined, "stream": False, "options": opts},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as exc:
        msg = f"Ollama /api/generate failed: {exc!s}"
        raise RuntimeError(msg) from exc

    n = data.get("prompt_eval_count")
    if isinstance(n, int):
        return n
    msg = f"Unexpected /api/generate response: {data}"
    raise RuntimeError(msg)


def _as_message_reprs(
    messages: Sequence[MessageLike],
) -> list[MessageLikeRepresentation]:
    """Produce a lossy but fast representation for approximate counting."""
    out: list[MessageLikeRepresentation] = []
    msg_len = 2  # enforces the tuple shape (role, content)
    for m in messages:
        if (
            isinstance(m, (BaseMessage, str))
            or (
                isinstance(m, tuple)
                and len(m) == msg_len
                and all(isinstance(x, str) for x in m)
            )
            or (isinstance(m, list) and all(isinstance(x, str) for x in m))
        ):
            out.append(m)
        elif isinstance(m, Mapping):
            role = str(m.get("role") or m.get("type") or "user")
            content_obj = m.get("content") or m.get("text") or ""
            content = content_obj if isinstance(content_obj, str) else str(content_obj)
            out.append((role, content))
        else:
            out.append(str(m))
    return out


def _looks_like_openai_model(name: str) -> bool:
    """Treat Ollama model as OpenAI GPT if it starts with an OpenAI prefix."""
    n = name.strip().lower()
    return any(n.startswith(p) for p in OPENAI_PREFIXES)


def count_tokens_cross_provider(
    messages: Sequence[MessageLike],
    model: str,
    provider: Provider,
    *,
    options: Mapping[str, Any],
    chat_model_options: Mapping[str, Any],
) -> int:
    """
    Count tokens for different providers.

    openai  -> tiktoken
    gemini  -> REST models/{model}:countTokens
    ollama  -> If model looks like OpenAI (e.g., gpt-oss), use tiktoken;
               otherwise /api/tokenize (fast) or /api/generate with (exact)
    """
    if provider == "openai":
        return _count_tokens_tiktoken(messages, model=model)

    if provider == "gemini":
        gemini_api_key_any = options.get(CONF_GEMINI_API_KEY)
        if not isinstance(gemini_api_key_any, str) or not gemini_api_key_any.strip():
            msg = "Gemini API key must be a non-empty string in options."
            raise ValueError(msg)
        gemini_api_key: str = gemini_api_key_any
        return _count_gemini_tokens(
            messages, model=model, gemini_api_key=gemini_api_key
        )

    # Provider "ollama"
    # If Ollama is hosting an OpenAI-style GPT (e.g., gpt-oss), prefer tiktoken.
    if _looks_like_openai_model(model):
        return _count_tokens_tiktoken(messages, model=model)

    ollama_base_url_any = options.get(CONF_OLLAMA_CHAT_URL) or options.get(
        CONF_OLLAMA_URL
    )
    if not isinstance(ollama_base_url_any, str) or not ollama_base_url_any.strip():
        msg = "Ollama base URL must be a non-empty string in options."
        raise ValueError(msg)
    ollama_base_url: str = ollama_base_url_any

    if OLLAMA_EXACT_TOKEN_COUNT:
        n = _count_ollama_tokens(
            messages, model=model, base_url=ollama_base_url, options=chat_model_options
        )
        LOGGER.debug("Ollama token count (exact): %d", n)
        return n

    approx = count_tokens_approximately(_as_message_reprs(messages))
    LOGGER.debug("Ollama token count (approx): %d", approx)
    return approx
