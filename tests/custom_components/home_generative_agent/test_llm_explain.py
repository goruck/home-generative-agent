# ruff: noqa: S101
"""Tests for compact Sentinel LLM explanation output."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from custom_components.home_generative_agent.core.utils import SentinelLLMDeferredError
from custom_components.home_generative_agent.explain.llm_explain import (
    LLMExplainer,
    _display_type,
    _friendly_type,
    _iso_to_relative,
    _redact_person_names,
    _relativize_timestamps,
)
from custom_components.home_generative_agent.explain.prompts import (
    LANGUAGE_INSTRUCTION_TEMPLATE,
    SYSTEM_PROMPT,
)
from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
from custom_components.home_generative_agent.sentinel.notifier import (
    _redact_if_sensitive,
)


class DummyModel:
    """Model stub returning preconfigured content."""

    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, _messages: list[Any]) -> SimpleNamespace:
        return SimpleNamespace(content=self._content)


class CapturingModel:
    """Model stub that records the messages it was invoked with."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.messages: list[Any] | None = None

    async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
        self.messages = messages
        return SimpleNamespace(content=self._content)


def _finding() -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id="a1",
        type="open_entry_at_night_when_home_window",
        severity="high",
        confidence=0.6,
        triggering_entities=["binary_sensor.garage_and_play_room_doors"],
        evidence={"entity_id": "binary_sensor.garage_and_play_room_doors"},
        suggested_actions=["close_entry"],
        is_sensitive=True,
    )


def _low_finding() -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id="a2",
        type="camera_entry_unsecured",
        severity="low",
        confidence=0.4,
        triggering_entities=["camera.driveway"],
        evidence={"camera_entity_id": "camera.driveway"},
        suggested_actions=["check_entry"],
        is_sensitive=True,
    )


@pytest.mark.asyncio
async def test_async_explain_sanitizes_markdown() -> None:
    explainer = LLMExplainer(DummyModel("**Door open** at night.\n`Close it now.`"))
    result = await explainer.async_explain(_finding())
    assert result == "Door open at night. Close it now."


@pytest.mark.asyncio
async def test_async_explain_strips_think_blocks() -> None:
    """<think> reasoning blocks emitted by qwen3/qwen3.5 must be stripped."""
    content = "<think>reasoning here</think>Door open recently. Close it now."
    explainer = LLMExplainer(DummyModel(content))
    result = await explainer.async_explain(_finding())
    assert result is not None
    assert "<think>" not in result
    assert "reasoning here" not in result
    assert "Door open recently." in result


@pytest.mark.asyncio
async def test_async_explain_falls_back_when_too_long() -> None:
    long_text = "very long explanation " * 30
    explainer = LLMExplainer(DummyModel(long_text))
    result = await explainer.async_explain(_finding())
    assert result is not None
    assert len(result) <= 220
    assert "Urgent:" in result
    assert "open_entry_at_night_when_home_window" not in result


@pytest.mark.asyncio
async def test_async_explain_low_severity_uses_relaxed_hint() -> None:
    explainer = LLMExplainer(DummyModel("very long explanation " * 30))
    result = await explainer.async_explain(_low_finding())
    assert result is not None
    assert "Review when convenient." in result


# --- Tests for timestamp relativization ---

_FIXED_NOW = datetime(2025, 1, 15, 21, 0, 0, tzinfo=UTC)


@patch(
    "custom_components.home_generative_agent.explain.llm_explain.datetime",
    wraps=datetime,
)
def test_iso_to_relative_just_now(mock_dt: Any) -> None:
    mock_dt.now.return_value = _FIXED_NOW
    assert _iso_to_relative("2025-01-15T21:00:00+00:00") == "just now"


@patch(
    "custom_components.home_generative_agent.explain.llm_explain.datetime",
    wraps=datetime,
)
def test_iso_to_relative_minutes(mock_dt: Any) -> None:
    mock_dt.now.return_value = _FIXED_NOW
    assert _iso_to_relative("2025-01-15T20:50:00+00:00") == "about 10 minutes ago"


@patch(
    "custom_components.home_generative_agent.explain.llm_explain.datetime",
    wraps=datetime,
)
def test_iso_to_relative_one_minute(mock_dt: Any) -> None:
    mock_dt.now.return_value = _FIXED_NOW
    assert _iso_to_relative("2025-01-15T20:59:00+00:00") == "about 1 minute ago"


@patch(
    "custom_components.home_generative_agent.explain.llm_explain.datetime",
    wraps=datetime,
)
def test_iso_to_relative_hours(mock_dt: Any) -> None:
    mock_dt.now.return_value = _FIXED_NOW
    assert _iso_to_relative("2025-01-15T18:30:00+00:00") == "about 2 hours ago"


def test_iso_to_relative_non_timestamp() -> None:
    assert _iso_to_relative("not-a-timestamp") == "not-a-timestamp"


@patch(
    "custom_components.home_generative_agent.explain.llm_explain.datetime",
    wraps=datetime,
)
def test_relativize_timestamps_replaces_iso_values(mock_dt: Any) -> None:
    mock_dt.now.return_value = _FIXED_NOW
    evidence = {
        "entity_id": "binary_sensor.window",
        "last_changed": "2025-01-15T20:09:00+00:00",
        "state": "on",
        "anyone_home": True,
    }
    result = _relativize_timestamps(evidence)
    assert result["entity_id"] == "binary_sensor.window"
    assert result["state"] == "on"
    assert result["anyone_home"] is True
    assert "ago" in result["last_changed"]
    assert "2025" not in result["last_changed"]


@pytest.mark.asyncio
async def test_async_explain_does_not_pass_raw_timestamps() -> None:
    """Verify the evidence sent to the LLM has timestamps relativized."""
    captured_messages: list[Any] = []

    class CapturingModel:
        async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
            captured_messages.extend(messages)
            return SimpleNamespace(content="Windows opened recently. Close them now.")

    finding = AnomalyFinding(
        anomaly_id="a3",
        type="open_entry_at_night_when_home_window",
        severity="high",
        confidence=0.6,
        triggering_entities=["binary_sensor.window"],
        evidence={
            "entity_id": "binary_sensor.window",
            "last_changed": "2025-01-15T20:09:00+00:00",
        },
        suggested_actions=["close_entry"],
        is_sensitive=True,
    )
    explainer = LLMExplainer(CapturingModel())
    await explainer.async_explain(finding)
    prompt_text = captured_messages[1].content
    assert "2025-01-15T20:09:00" not in prompt_text
    assert "ago" in prompt_text


@pytest.mark.asyncio
async def test_async_explain_returns_none_when_deferred() -> None:
    """async_explain must return None when the model call is deferred."""
    explainer = LLMExplainer(DummyModel("some content"))
    with patch(
        "custom_components.home_generative_agent.explain.llm_explain.run_sentinel_model_call",
        side_effect=SentinelLLMDeferredError("explain", "chat is active"),
    ):
        result = await explainer.async_explain(_finding())
    assert result is None


@pytest.mark.asyncio
async def test_async_explain_returns_none_on_timeout() -> None:
    """async_explain must return None when the LLM call times out."""
    explainer = LLMExplainer(DummyModel("irrelevant"))
    with patch(
        "custom_components.home_generative_agent.explain.llm_explain.run_sentinel_model_call",
        side_effect=TimeoutError(),
    ):
        result = await explainer.async_explain(_finding())
    assert result is None


@pytest.mark.asyncio
async def test_async_explain_returns_none_on_value_error() -> None:
    """async_explain must return None on unexpected LLM errors."""
    explainer = LLMExplainer(DummyModel("irrelevant"))
    with patch(
        "custom_components.home_generative_agent.explain.llm_explain.run_sentinel_model_call",
        side_effect=ValueError("bad response"),
    ):
        result = await explainer.async_explain(_finding())
    assert result is None


def test_friendly_type_open_entry_at_night_variants() -> None:
    """Issue #504: presence-agnostic night rule IDs get the clean entry label."""
    for anomaly_type in (
        "open_entry_at_night",
        "open_entry_at_night_window",
        "open_entry_at_night_door",
        "open_entry_at_night_entry",
    ):
        assert _friendly_type(anomaly_type) == "Open entry at night"


def test_friendly_type_motion_at_night_while_away() -> None:
    """Issue #516: the motion-while-away rule ID gets a clean label."""
    assert (
        _friendly_type("motion_detected_at_night_while_away")
        == "Motion at night while away"
    )


def test_display_type_prefers_template_label_for_slug_rule_ids() -> None:
    """Slugified candidate rule IDs display the curated template label."""
    finding = AnomalyFinding(
        anomaly_id="slug-1",
        type="v1_subject_motion_sensor_candidate_slug",
        severity="medium",
        confidence=0.8,
        triggering_entities=["binary_sensor.hall_motion"],
        evidence={"template_id": "motion_detected_at_night_while_away"},
        suggested_actions=["check_camera"],
        is_sensitive=False,
    )
    assert _display_type(finding) == "Motion at night while away"


def test_friendly_type_motion_while_away() -> None:
    """Issue #518: the day-agnostic away-motion template gets a clean label."""
    assert _friendly_type("motion_detected_while_away") == "Motion while away"
    finding = AnomalyFinding(
        anomaly_id="slug-518",
        type="motion_kitchen_while_away",
        severity="low",
        confidence=0.6,
        triggering_entities=["binary_sensor.xiao_esp32_c5_espectre_motion"],
        evidence={"template_id": "motion_detected_while_away"},
        suggested_actions=["check_camera"],
        is_sensitive=False,
    )
    assert _display_type(finding) == "Motion while away"


# ---------------------------------------------------------------
# sentinel_response_language override (issue #523, reworked per review)
# ---------------------------------------------------------------


def _sensitive_finding_with_person(name: str) -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id="a3",
        type="camera_entry_unsecured",
        severity="medium",
        confidence=0.7,
        triggering_entities=["camera.front_door"],
        evidence={
            "camera_entity_id": "camera.front_door",
            "recognized_people": [name],
        },
        suggested_actions=["check_entry"],
        is_sensitive=True,
    )


@pytest.mark.asyncio
async def test_no_language_override_leaves_system_prompt_unchanged() -> None:
    """Without a response_language, the system prompt is exactly SYSTEM_PROMPT."""
    model = CapturingModel("Door open at night. Close it now.")
    explainer = LLMExplainer(model)
    await explainer.async_explain(_finding())
    assert model.messages is not None
    assert cast("str", model.messages[0].content) == SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_language_override_appends_to_system_prompt() -> None:
    """A response_language appends after SYSTEM_PROMPT, never replacing it."""
    model = CapturingModel("Dveře byly v noci otevřené. Zavřete je.")
    explainer = LLMExplainer(model, response_language="Czech")
    await explainer.async_explain(_finding())

    assert model.messages is not None
    system = cast("str", model.messages[0].content)
    assert system.startswith(SYSTEM_PROMPT)
    appended = system[len(SYSTEM_PROMPT) :]
    assert appended == LANGUAGE_INSTRUCTION_TEMPLATE.format(language="Czech")
    assert "Write your explanation in Czech" in appended


@pytest.mark.asyncio
async def test_language_override_instructs_nominative_person_names() -> None:
    """
    The language instruction must tell the model to keep names uninflected.

    notifier._redact_if_sensitive matches finding.evidence['recognized_people']
    names against the explanation with an exact (case-insensitive) string
    match. Inflected languages like Czech decline names by grammatical case
    ("Petra" -> "Petru"), so without this instruction a translated
    explanation could contain an inflected name that redaction would miss.
    """
    model = CapturingModel("...")
    explainer = LLMExplainer(model, response_language="Czech")
    await explainer.async_explain(_finding())

    assert model.messages is not None
    system = cast("str", model.messages[0].content)
    assert "base (nominative, dictionary) form" in system
    assert "do not decline, conjugate, or otherwise inflect" in system


@pytest.mark.asyncio
async def test_redaction_succeeds_when_model_keeps_nominative_name() -> None:
    """
    Locks in the redaction contract for language-instruction-compliant output.

    When the model honors the nominative-form instruction (as instructed by
    test_language_override_instructs_nominative_person_names above), the
    stored evidence name "Petra" matches verbatim in the explanation and
    notifier._redact_if_sensitive successfully redacts it -- even though the
    surrounding Czech sentence declines other words normally.
    """
    finding = _sensitive_finding_with_person("Petra")
    # Simulates a model that followed the nominative-form instruction: the
    # name "Petra" appears unchanged even though Czech grammar would
    # otherwise decline it to "Petru" (accusative) in this sentence position.
    explanation = "Kamera zaznamenala Petra u předních dveří v neobvyklou dobu."
    redacted = _redact_if_sensitive(explanation, finding)
    assert redacted is not None
    assert "Petra" not in redacted
    assert "a recognised person" in redacted


def test_redaction_misses_inflected_name_documents_the_gap() -> None:
    """
    Documents the exact-match gap that motivates prompt-input pre-redaction.

    If an explanation contained a name declined per normal Czech grammar
    ("Petra" -> accusative "Petru"), exact-match redaction would not catch
    it. _redact_if_sensitive does no linguistic normalization -- which is
    why, when a response language is set, the explainer removes recognized
    names from the model's input entirely (_redact_person_names) so no
    inflection can ever be produced. The nominative-form prompt instruction
    is defense in depth on top of that deterministic boundary.
    """
    finding = _sensitive_finding_with_person("Petra")
    explanation = "Kamera zaznamenala Petru u předních dveří v neobvyklou dobu."
    redacted = _redact_if_sensitive(explanation, finding)
    assert redacted is not None
    assert "Petru" in redacted  # not redacted -- exact-match only, by design
    assert "a recognised person" not in redacted


@pytest.mark.asyncio
async def test_translated_sensitive_finding_names_never_reach_model() -> None:
    """
    Deterministic privacy boundary: names are stripped from the prompt input.

    With a response language set and a sensitive finding, the recognized
    name must not appear anywhere in the messages sent to the model -- the
    model cannot inflect a name it never saw, so the exact-match gap
    documented above becomes unreachable.
    """
    model = CapturingModel("...")
    explainer = LLMExplainer(model, response_language="Czech")
    await explainer.async_explain(_sensitive_finding_with_person("Petra"))

    assert model.messages is not None
    human = cast("str", model.messages[1].content)
    assert "Petra" not in human
    assert "a recognised person" in human


@pytest.mark.asyncio
async def test_translated_sensitive_caption_embedded_name_redacted() -> None:
    """Names embedded in free-text evidence (e.g. captions) are also stripped."""
    finding = _sensitive_finding_with_person("Petra")
    finding.evidence["caption"] = "Petra standing at the front door."
    model = CapturingModel("...")
    explainer = LLMExplainer(model, response_language="Czech")
    await explainer.async_explain(finding)

    assert model.messages is not None
    human = cast("str", model.messages[1].content)
    assert "Petra" not in human
    assert "a recognised person standing at the front door." in human


@pytest.mark.asyncio
async def test_translated_non_sensitive_finding_keeps_names() -> None:
    """Non-sensitive findings are never redacted, translated or not."""
    finding = AnomalyFinding(
        anomaly_id="a4",
        type="camera_entry_unsecured",
        severity="medium",
        confidence=0.7,
        triggering_entities=["camera.front_door"],
        evidence={
            "camera_entity_id": "camera.front_door",
            "recognized_people": ["Petra"],
        },
        suggested_actions=["check_entry"],
        is_sensitive=False,
    )
    model = CapturingModel("...")
    explainer = LLMExplainer(model, response_language="Czech")
    await explainer.async_explain(finding)

    assert model.messages is not None
    assert "Petra" in cast("str", model.messages[1].content)


@pytest.mark.asyncio
async def test_english_sensitive_finding_prompt_unchanged() -> None:
    """
    Without a response language the prompt input is not pre-redacted.

    English explanations keep today's behavior: the name reaches the model
    and notifier._redact_if_sensitive redacts the exact (uninflected) form
    before notification dispatch.
    """
    model = CapturingModel("...")
    explainer = LLMExplainer(model)
    await explainer.async_explain(_sensitive_finding_with_person("Petra"))

    assert model.messages is not None
    assert "Petra" in cast("str", model.messages[1].content)


def test_redact_person_names_covers_all_structural_positions() -> None:
    """Structural redaction catches names in values, keys, and containers."""
    evidence: dict[str, Any] = {
        "recognized_people": ["Petra"],
        "caption": "Petra at the door. Note: petra again.",
        "Petra": "seen",
        "nested": {"people": ("Petra", "unknown"), "tags": {"petra"}},
    }
    redacted = _redact_person_names(evidence)
    rendered = repr(redacted)
    assert "Petra" not in rendered
    assert "petra" not in rendered
    assert redacted["caption"] == (
        "a recognised person at the door. Note: a recognised person again."
    )
    assert redacted["a recognised person"] == "seen"
    assert redacted["nested"]["people"] == ("a recognised person", "unknown")
    assert redacted["nested"]["tags"] == {"a recognised person"}


def test_redact_person_names_does_not_mutate_original() -> None:
    """The original evidence dict (and nesting) must never be mutated."""
    evidence: dict[str, Any] = {
        "recognized_people": ["Petra"],
        "caption": "Petra at the door.",
        "nested": {"ids": ["Petra", 3]},
    }
    redacted = _redact_person_names(evidence)
    assert evidence["caption"] == "Petra at the door."
    assert evidence["recognized_people"] == ["Petra"]
    assert evidence["nested"]["ids"] == ["Petra", 3]
    assert redacted["recognized_people"] == ["a recognised person"]
    assert redacted["nested"]["ids"] == ["a recognised person", 3]


def test_redact_person_names_overlapping_names_longest_first() -> None:
    """
    Overlapping names must never leave a partial name behind.

    A shortest-first alternation would turn "Alexander" into
    "a recognised personander" -- the "ander" residue still identifies the
    person. Longest-first ordering makes the whole name match first.
    """
    evidence: dict[str, Any] = {
        "recognized_people": ["Alex", "Alexander"],
        "caption": "Alexander and Alex arrived.",
    }
    redacted = _redact_person_names(evidence)
    assert "Alexander" not in repr(redacted)
    assert "Alex" not in repr(redacted)
    assert redacted["caption"] == (
        "a recognised person and a recognised person arrived."
    )


@pytest.mark.asyncio
async def test_translated_sensitive_name_with_quotes_never_reaches_model() -> None:
    """
    Names containing quotes must be redacted despite repr escaping.

    Redaction happens on the evidence structure before the prompt is
    rendered. A post-render approach would miss names like this one: repr
    turns an embedded apostrophe into an escaped form the raw-name pattern
    no longer matches.
    """
    finding = _sensitive_finding_with_person('D\'Angelo "Junior"')
    finding.evidence["caption"] = 'D\'Angelo "Junior" stood at the front door.'
    model = CapturingModel("...")
    explainer = LLMExplainer(model, response_language="Czech")
    await explainer.async_explain(finding)

    assert model.messages is not None
    human = cast("str", model.messages[1].content)
    assert "D'Angelo" not in human
    assert "Angelo" not in human
    assert "Junior" not in human


@pytest.mark.asyncio
async def test_translated_sensitive_short_name_keeps_template_text() -> None:
    """
    Redaction must never touch the prompt template's own instruction text.

    A recognized person named "Max" must not corrupt the template's
    "Max 2 short sentences." output rule -- only evidence content is
    redacted, because redaction runs before the template is rendered.
    """
    finding = _sensitive_finding_with_person("Max")
    finding.evidence["caption"] = "Max stood at the front door."
    model = CapturingModel("...")
    explainer = LLMExplainer(model, response_language="Czech")
    await explainer.async_explain(finding)

    assert model.messages is not None
    human = cast("str", model.messages[1].content)
    assert "Max 2 short sentences." in human
    assert "a recognised person stood at the front door." in human


@pytest.mark.asyncio
async def test_over_length_returns_none_under_response_language() -> None:
    """
    The English compact fallback is not a translation — report failure instead.

    Returning it would let vague English ("Open entry at night: Front Door.
    Urgent: ...") outrank the notifier's precise deterministic copy, since it
    fits comfortably inside the 220-character mobile cap.
    """
    explainer = LLMExplainer(
        DummyModel("velmi dlouhé vysvětlení " * 30), response_language="Czech"
    )
    assert await explainer.async_explain(_finding()) is None


@pytest.mark.asyncio
async def test_empty_output_returns_none_under_response_language() -> None:
    """Same for empty model output when a language is configured."""
    explainer = LLMExplainer(DummyModel("   "), response_language="Czech")
    assert await explainer.async_explain(_finding()) is None


@pytest.mark.asyncio
async def test_english_still_gets_compact_fallback() -> None:
    """English behaviour is unchanged: the compact fallback still applies."""
    explainer = LLMExplainer(DummyModel("very long explanation " * 30))
    result = await explainer.async_explain(_finding())
    assert result is not None
    assert "Urgent:" in result
