# ruff: noqa: S101
"""An OpenAI-compatible server keeps the model's own top_p unless one is set."""

from __future__ import annotations

import pytest

import custom_components.home_generative_agent as hga
from custom_components.home_generative_agent.const import (
    CONF_FEATURE_MODEL_TOP_P,
    MODEL_CATEGORY_SPECS,
)
from custom_components.home_generative_agent.core.subentry_resolver import (
    _apply_feature_model_to_options,
)


def test_top_p_is_sent_only_when_set() -> None:
    assert hga._top_p_if_set(None) == {}
    assert hga._top_p_if_set(0.9) == {"top_p": 0.9}


@pytest.mark.parametrize(
    ("category", "key"),
    [
        ("chat", "chat_model_top_p"),
        ("vlm", "vlm_top_p"),
        ("summarization", "summarization_model_top_p"),
        ("sentinel", "sentinel_model_top_p"),
    ],
)
def test_every_chat_like_category_has_a_top_p_option(category: str, key: str) -> None:
    assert MODEL_CATEGORY_SPECS[category]["top_p_key"] == key


def test_a_feature_top_p_reaches_the_options() -> None:
    options: dict[str, object] = {}
    _apply_feature_model_to_options(
        options, "chat", "openai_compatible", {CONF_FEATURE_MODEL_TOP_P: 0.95}
    )
    assert options["chat_model_top_p"] == 0.95


def test_an_empty_feature_top_p_sets_nothing() -> None:
    options: dict[str, object] = {}
    _apply_feature_model_to_options(options, "vlm", "ollama", {})
    assert "vlm_top_p" not in options
