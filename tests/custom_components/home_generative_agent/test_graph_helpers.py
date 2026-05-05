# ruff: noqa: S101
"""Unit tests for graph.py helper functions — Anthropic provider branches."""

from __future__ import annotations

from custom_components.home_generative_agent.agent.graph import _determine_model_name
from custom_components.home_generative_agent.const import CONF_ANTHROPIC_CHAT_MODEL


def test_determine_model_name_anthropic_returns_configured_model() -> None:
    """Anthropic provider reads CONF_ANTHROPIC_CHAT_MODEL from opts."""
    opts = {CONF_ANTHROPIC_CHAT_MODEL: "claude-sonnet-4-5"}
    assert _determine_model_name("anthropic", opts) == "claude-sonnet-4-5"


def test_determine_model_name_anthropic_missing_key_returns_empty() -> None:
    """Anthropic provider with no key in opts returns empty string."""
    assert _determine_model_name("anthropic", {}) == ""
