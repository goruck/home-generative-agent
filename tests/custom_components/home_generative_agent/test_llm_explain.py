"""Tests for compact Sentinel LLM explanation output."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from custom_components.home_generative_agent.explain.llm_explain import (
    LLMExplainer,
    _iso_to_relative,
    _relativize_timestamps,
)
from custom_components.home_generative_agent.sentinel.models import AnomalyFinding


class DummyModel:
    """Model stub returning preconfigured content."""

    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, _messages: list[Any]) -> SimpleNamespace:
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
