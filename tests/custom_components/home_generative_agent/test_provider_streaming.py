# ruff: noqa: S101
"""Verify that OpenAI and Anthropic providers are constructed with streaming enabled."""

from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage

from custom_components.home_generative_agent.agent.graph import _invoke_model

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
    block = _provider_block("anthropic_provider = ChatAnthropic(")
    assert "streaming=True" in block


def test_openai_provider_http_client_preserved() -> None:
    """ChatOpenAI construction still wires the sync and async HTTP clients."""
    block = _provider_block("openai_provider = ChatOpenAI(")
    assert "http_client=openai_http_client" in block
    assert "http_async_client=http_async_client" in block


def test_anthropic_provider_prewarm_preserved() -> None:
    """Anthropic async-client pre-warm call is still present after streaming change."""
    assert "_get_default_async_httpx_client" in _INIT


async def test_stream_usage_metadata_flows_through_invoke_model() -> None:
    """_invoke_model preserves usage_metadata; empty means stream_usage=True was removed."""
    expected_usage: dict[str, int] = {
        "input_tokens": 42,
        "output_tokens": 17,
        "total_tokens": 59,
    }
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = AIMessage(
        content="Hello", usage_metadata=expected_usage
    )

    result = await _invoke_model(mock_model, [], {})
    assert result.usage_metadata == expected_usage
