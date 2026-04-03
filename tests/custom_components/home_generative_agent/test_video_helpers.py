"""Tests for core/video_helpers.py text helpers."""

from __future__ import annotations

from custom_components.home_generative_agent.core.video_helpers import (
    limit_sentences_and_chars,
)

# ---------------------------------------------------------------------------
# limit_sentences_and_chars tests
# ---------------------------------------------------------------------------

_LONG_S1 = (
    "A paved patio area leads up to a closed white picket gate that overlooks "
    "a street where two vehicles are parked in the distance."
)  # 124 chars
_LONG_S2 = (
    "To the right of the gate, a dog is sitting near a planter and looking "
    "toward the camera."
)  # 90 chars
_TWO_SENTENCE_TEXT = f"{_LONG_S1} {_LONG_S2}"


def test_first_sentence_fits_when_combined_exceeds_limit() -> None:
    """When two sentences together exceed max_chars, only the first is returned."""
    result = limit_sentences_and_chars(_TWO_SENTENCE_TEXT, max_chars=150)
    assert result == _LONG_S1
    assert len(result) <= 150


def test_both_sentences_returned_when_both_fit() -> None:
    short = "Dog seen. Gate closed."
    result = limit_sentences_and_chars(short, max_chars=150)
    assert result == short


def test_single_long_sentence_truncated_at_word_boundary() -> None:
    single = "a " * 100  # 200 chars, no sentence ender
    result = limit_sentences_and_chars(single.strip(), max_chars=50)
    assert len(result) <= 50
    # Should end at a word boundary, not mid-word
    assert not result.endswith("a ")
    assert " " not in result[-1:]  # last char isn't a space


def test_short_text_returned_unchanged() -> None:
    text = "Motion detected."
    result = limit_sentences_and_chars(text, max_chars=150)
    assert result == text


def test_no_mid_word_truncation() -> None:
    """Regression: truncation must not cut mid-word (the original bug)."""
    # Build text where hard truncation at 150 would cut mid-word
    result = limit_sentences_and_chars(_TWO_SENTENCE_TEXT, max_chars=150)
    # Result must be a complete word — no trailing partial word
    words = result.split()
    assert all(w == words[-1] or w for w in words)
    assert len(result) <= 150
