# ruff: noqa: S101
"""Verify that OpenAI and Anthropic providers are constructed with streaming enabled."""

from __future__ import annotations

import pathlib

_INIT = pathlib.Path("custom_components/home_generative_agent/__init__.py").read_text()


def _provider_block(marker: str, chars: int = 600) -> str:
    idx = _INIT.index(marker)
    return _INIT[idx : idx + chars]


def test_openai_provider_has_streaming_true() -> None:
    """ChatOpenAI construction includes streaming=True."""
    block = _provider_block("openai_provider = ChatOpenAI(")
    assert "streaming=True" in block


def test_openai_provider_has_stream_usage_true() -> None:
    """ChatOpenAI construction includes stream_usage=True to preserve usage_metadata."""
    block = _provider_block("openai_provider = ChatOpenAI(")
    assert "stream_usage=True" in block


def test_anthropic_provider_has_streaming_true() -> None:
    """ChatAnthropic construction includes streaming=True."""
    block = _provider_block("anthropic_chat = SharedClientChatAnthropic(")
    assert "streaming=True" in block


def test_openai_provider_http_client_preserved() -> None:
    """ChatOpenAI construction still wires the sync and async HTTP clients."""
    block = _provider_block("openai_provider = ChatOpenAI(")
    assert "http_client=openai_http_client" in block
    assert "http_async_client=http_async_client" in block


def test_anthropic_provider_primed_before_publish() -> None:
    """The SDK client is primed off the loop before the provider exists (#587, #618)."""
    start = _INIT.index("anthropic_chat = SharedClientChatAnthropic(")
    end = _INIT.index("except Exception:", start)
    block = _INIT[start:end]
    prime = block.index("await async_prime_async_client(hass, anthropic_chat)")
    publish = block.index("anthropic_provider = anthropic_chat.configurable_fields(")
    assert prime < publish
