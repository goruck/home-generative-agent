"""Tests for RAG embedding strip and instruction fusion."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from custom_components.home_generative_agent.agent.rag_embedding_text import (
    fused_similarity,
    instruction_keys_fused_from_search_results,
    strip_for_embedding,
)
from custom_components.home_generative_agent.const import (
    RECOMMENDED_INSTRUCTION_RAG_NOISE_FLOOR,
)


def test_strip_for_embedding_preserves_word_boundaries() -> None:
    """Jinja removal must not concatenate adjacent words."""
    text = "house{% if true %}door"
    assert strip_for_embedding(text) == "house door"


def test_strip_for_embedding_order_comments_blocks_vars() -> None:
    """Strip comments, blocks, then replace variables."""
    text = "{# c #}Hello{% set x = 1 %}{{ name }}\nWorld"
    out = strip_for_embedding(text)
    assert "[value]" in out
    assert "Hello" in out
    assert "World" in out


def test_fused_similarity_noise_floor_drops_weak_body() -> None:
    """Low body similarity is ignored so strong intent is not averaged down."""
    alpha = 0.65
    nf = RECOMMENDED_INSTRUCTION_RAG_NOISE_FLOOR
    fused = fused_similarity(0.9, 0.1, alpha, noise_floor=nf)
    assert fused == pytest.approx(0.9)


def test_fused_similarity_intent_only() -> None:
    """Single-channel intent passes through."""
    assert fused_similarity(0.8, None, 0.65, noise_floor=0.25) == 0.8


def test_fused_similarity_body_only() -> None:
    """Single-channel body passes through."""
    assert fused_similarity(None, 0.7, 0.65, noise_floor=0.25) == 0.7


@dataclass
class _FakeHit:
    key: str
    score: float | None


def test_instruction_keys_fused_from_search_results() -> None:
    """Merged keys sorted by fused score; threshold applied."""
    intent = [_FakeHit("a", 0.9), _FakeHit("b", 0.3)]
    body = [_FakeHit("a", 0.05), _FakeHit("c", 0.85)]
    keys = instruction_keys_fused_from_search_results(
        intent,
        body,
        instruction_limit=10,
        instruction_threshold=0.5,
        alpha=0.65,
        noise_floor=RECOMMENDED_INSTRUCTION_RAG_NOISE_FLOOR,
    )
    # a: intent 0.9, body 0.05 -> body nulled -> 0.9
    # b: 0.3 only intent -> 0.3 dropped
    # c: 0.85 body only -> 0.85
    assert "a" in keys
    assert "c" in keys
    assert "b" not in keys
    assert keys[0] == "a"  # 0.9 > 0.85
    assert keys[1] == "c"
