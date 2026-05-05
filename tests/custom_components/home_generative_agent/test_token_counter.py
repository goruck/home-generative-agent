# ruff: noqa: S101
"""Unit tests for agent/token_counter.py — Anthropic provider branch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from custom_components.home_generative_agent.agent.token_counter import (
    count_tokens_cross_provider,
)


def _make_fake_encoding(token_count: int = 5) -> MagicMock:
    enc = MagicMock()
    enc.encode.return_value = list(range(token_count))
    return enc


def test_count_tokens_cross_provider_anthropic_returns_int() -> None:
    """Anthropic provider uses tiktoken fallback (gpt-4o) and returns a positive int."""
    messages = [HumanMessage(content="Hello, how are you?")]
    with patch(
        "custom_components.home_generative_agent.agent.token_counter._pick_encoding_for_model",
        return_value=_make_fake_encoding(5),
    ):
        result = count_tokens_cross_provider(
            messages,
            model="claude-sonnet-4-5",
            provider="anthropic",
            options={},
            chat_model_options={},
        )
    assert isinstance(result, int)
    assert result > 0


def test_count_tokens_cross_provider_anthropic_empty_messages() -> None:
    """Anthropic provider with no messages returns 0."""
    with patch(
        "custom_components.home_generative_agent.agent.token_counter._pick_encoding_for_model",
        return_value=_make_fake_encoding(5),
    ):
        result = count_tokens_cross_provider(
            [],
            model="claude-opus-4-5",
            provider="anthropic",
            options={},
            chat_model_options={},
        )
    assert isinstance(result, int)
    assert result == 0
