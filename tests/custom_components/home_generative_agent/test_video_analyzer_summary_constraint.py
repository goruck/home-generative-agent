# ruff: noqa: S101
"""
Tests for the single-person summary constraint (issue #543, caption-side).

Field failures showed the summarizer narrating one person as two actors
("a person stands ... then Lindo appears") when some frames carried
"Indeterminate" identities but person-mentioning captions. When face
recognition (post identity-merge) proves exactly one known person, the
summary prompt now states that fact outright. The constraint must NEVER be
emitted when the evidence still allows a second person.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.home_generative_agent.core.video_analyzer import (
    VideoAnalyzer,
    _single_person_constraint,
)

# ---------------------------------------------------------------------------
# Override autouse fixtures from pytest-homeassistant-custom-component
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def enable_event_loop_debug() -> None:
    """No-op override: pure-asyncio tests don't need HA's debug-mode hook."""


@pytest.fixture(autouse=True)
def verify_cleanup() -> None:
    """No-op override: all tasks explicitly awaited; no HA resources to clean up."""


_KNOWN = "Lindo"


# ---------------------------------------------------------------------------
# Decision table for the pure helper
# ---------------------------------------------------------------------------


def test_constraint_emitted_for_single_known_person() -> None:
    """One distinct known name across frames: constraint names them."""
    constraint = _single_person_constraint(
        [
            {"A person stands in a doorway.": ["Indeterminate"]},
            {"Lindo checks his phone.": [_KNOWN]},
            {"He walks off the porch.": [_KNOWN]},
        ]
    )

    assert constraint is not None
    assert "exactly one person, Lindo," in constraint
    assert "<single person constraint>" in constraint


def test_constraint_absent_with_unknown_person() -> None:
    """A surviving unknown face means a second person may exist."""
    assert (
        _single_person_constraint(
            [
                {"Lindo checks his phone.": [_KNOWN]},
                {"A man stands nearby.": ["Unknown Person"]},
            ]
        )
        is None
    )


def test_constraint_absent_with_two_known_names() -> None:
    """Two enrolled people in the batch: no single-person claim."""
    assert (
        _single_person_constraint(
            [
                {"Lindo checks his phone.": [_KNOWN]},
                {"Anna waters the plants.": ["Anna"]},
            ]
        )
        is None
    )


def test_constraint_absent_when_no_faces_detected() -> None:
    """All-Indeterminate batches prove nothing; stay silent."""
    assert (
        _single_person_constraint(
            [
                {"A person stands in a doorway.": ["Indeterminate"]},
                {"The yard is empty.": []},
            ]
        )
        is None
    )


def test_constraint_absent_when_frame_has_two_detected_faces() -> None:
    """Two detected faces in one frame may be two people, even same-named."""
    assert (
        _single_person_constraint(
            [
                {"Two figures near the door.": [_KNOWN, _KNOWN]},
                {"Lindo checks his phone.": [_KNOWN]},
            ]
        )
        is None
    )


def test_constraint_absent_for_legacy_reserved_label() -> None:
    """A gallery row named like a reserved label is never a verified person."""
    assert _single_person_constraint([{"A man walks by.": ["unknown person"]}]) is None


# ---------------------------------------------------------------------------
# End-to-end: the constraint lands in (or stays out of) the LLM prompt
# ---------------------------------------------------------------------------


def _va_with_summary_capture() -> tuple[VideoAnalyzer, AsyncMock]:
    entry = MagicMock()
    entry.runtime_data.options = {}
    entry.runtime_data.model_deployments.get.return_value = "cloud"
    configured = MagicMock()
    configured.ainvoke = AsyncMock(return_value=MagicMock(content="Lindo walks by."))
    entry.runtime_data.summarization_model.with_config.return_value = configured
    entry.runtime_data.summarization_model.config = {}
    return VideoAnalyzer(MagicMock(), entry), configured.ainvoke


@pytest.mark.asyncio
async def test_prompt_carries_constraint_for_single_known() -> None:
    """The verified-single-person fact reaches the summarizer prompt."""
    va, ainvoke = _va_with_summary_capture()

    await va._generate_summary(
        [
            {"A person stands in a doorway.": ["Indeterminate"]},
            {"Lindo checks his phone.": [_KNOWN]},
        ]
    )

    messages = ainvoke.call_args.args[0]
    prompt = messages[1].content
    assert "exactly one person, Lindo," in prompt
    # Identity tags themselves are unchanged: the system rules key on the
    # literal "Indeterminate" string.
    assert "<person identity>\nIndeterminate\n</person identity>" in prompt


@pytest.mark.asyncio
async def test_prompt_has_no_constraint_when_unknown_survives() -> None:
    """A refused-merge batch must reach the summarizer unconstrained."""
    va, ainvoke = _va_with_summary_capture()

    await va._generate_summary(
        [
            {"Lindo checks his phone.": [_KNOWN]},
            {"A man stands nearby.": ["Unknown Person"]},
        ]
    )

    messages = ainvoke.call_args.args[0]
    assert "<single person constraint>" not in messages[1].content
