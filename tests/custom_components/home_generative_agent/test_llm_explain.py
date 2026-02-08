# ruff: noqa: S101
"""Tests for compact Sentinel LLM explanation output."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.home_generative_agent.explain.llm_explain import LLMExplainer
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
