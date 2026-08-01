# ruff: noqa: S101
"""
Sentinel LLM layers must survive provider errors (issue #465).

``ollama.ResponseError`` (e.g. an un-pulled model returning 404) and httpx
transport errors subclass ``Exception`` directly, not the ValueError /
RuntimeError families the Sentinel layers used to catch.  An escaped error
permanently killed the discovery loop and broke triage's fail-open contract.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from ollama import ResponseError as OllamaResponseError

from custom_components.home_generative_agent.const import (
    CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
)
from custom_components.home_generative_agent.core.fallback import FallbackChatModel
from custom_components.home_generative_agent.explain.llm_explain import LLMExplainer
from custom_components.home_generative_agent.sentinel.discovery_engine import (
    SentinelDiscoveryEngine,
    _resolved_model_name,
)
from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
from custom_components.home_generative_agent.sentinel.triage import (
    TRIAGE_NOTIFY,
    TRIAGE_REASON_ERROR,
    SentinelTriageService,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.sentinel.discovery_store import (
        DiscoveryStore,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

_SNAPSHOT: dict[str, Any] = {
    "entities": [],
    "camera_activity": [],
    "derived": {"is_night": False, "now": "2026-01-01T00:00:00Z"},
    "generated_at": "2026-01-01T00:00:00Z",
}


class _DummyStore:
    async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
        return []

    async def async_append(self, _payload: Any) -> None:
        pass


def _finding() -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id="test-id-1",
        type="open_entry_while_away",
        severity="medium",
        confidence=0.8,
        triggering_entities=["binary_sensor.front_door"],
        evidence={},
        suggested_actions=["lock_door"],
        is_sensitive=False,
    )


def _triage_snapshot() -> FullStateSnapshot:
    return cast(
        "FullStateSnapshot",
        {
            "schema_version": 1,
            "generated_at": "2026-01-01T00:00:00+00:00",
            "entities": [],
            "camera_activity": [],
            "derived": {
                "now": "2026-01-01T10:00:00+00:00",
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": False,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
        },
    )


# ---------------------------------------------------------------------------
# Discovery: provider error in the LLM call must not escape _run_once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discovery_run_once_survives_provider_error(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An ollama.ResponseError from the model call is logged, not raised."""
    engine = SentinelDiscoveryEngine(
        hass=hass,
        options={},
        model=object(),
        store=cast("DiscoveryStore", _DummyStore()),
    )

    with (
        patch.object(
            engine,
            "_existing_semantic_context",
            new_callable=AsyncMock,
            return_value=(set(), set(), set()),
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.async_build_full_state_snapshot",
            new_callable=AsyncMock,
            return_value=dict(_SNAPSHOT),
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.run_sentinel_model_call",
            side_effect=OllamaResponseError("model 'gpt-oss' not found", 404),
        ),
    ):
        await engine._run_once()

    assert "Discovery LLM call failed" in caplog.text
    assert "gpt-oss" in caplog.text


# ---------------------------------------------------------------------------
# Discovery: the loop itself must survive any unexpected cycle error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discovery_loop_survives_unexpected_cycle_error(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cycle that raises must not end the loop; the next cycle still runs."""
    engine = SentinelDiscoveryEngine(
        hass=hass,
        options={CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS: 0},
        model=object(),
        store=cast("DiscoveryStore", _DummyStore()),
    )
    calls = 0

    async def _run_once() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            msg = "model 'gpt-oss' not found"
            raise OllamaResponseError(msg, 404)
        engine._stop_event.set()

    with patch.object(engine, "_run_once", side_effect=_run_once):
        await engine._run_loop()

    assert calls == 2, "Loop died after the first failing cycle"
    assert "Discovery cycle failed unexpectedly" in caplog.text


# ---------------------------------------------------------------------------
# Triage: fail-open must hold for provider errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triage_fails_open_on_provider_error() -> None:
    """A provider error must produce a notify decision, not an exception."""
    svc = SentinelTriageService(object())
    with patch(
        "custom_components.home_generative_agent.sentinel.triage.run_sentinel_model_call",
        side_effect=OllamaResponseError("model 'gpt-oss' not found", 404),
    ):
        result = await svc.triage(_finding(), _triage_snapshot())

    assert result.decision == TRIAGE_NOTIFY
    assert result.reason_code == TRIAGE_REASON_ERROR


# ---------------------------------------------------------------------------
# Explain: provider errors degrade to None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explain_returns_none_on_provider_error() -> None:
    """A provider error must return None instead of raising."""
    explainer = LLMExplainer(object())
    with patch(
        "custom_components.home_generative_agent.explain.llm_explain.run_sentinel_model_call",
        side_effect=OllamaResponseError("model 'gpt-oss' not found", 404),
    ):
        result = await explainer.async_explain(_finding())
    assert result is None


# ---------------------------------------------------------------------------
# Discovery payload records the actually-configured model name
# ---------------------------------------------------------------------------


def test_resolved_model_name_from_bound_runnable_config() -> None:
    """A with_config() binding exposes the effective model via configurable."""
    bound = SimpleNamespace(config={"configurable": {"model": "qwen3:8b"}})
    assert _resolved_model_name(bound) == "qwen3:8b"


def test_resolved_model_name_prefers_model_name_key_for_openai_style() -> None:
    bound = SimpleNamespace(config={"configurable": {"model_name": "gpt-4o-mini"}})
    assert _resolved_model_name(bound) == "gpt-4o-mini"


def test_resolved_model_name_from_fallback_chain_primary() -> None:
    """A FallbackChatModel's empty wrapper config must not hide the primary."""
    primary = SimpleNamespace(config={"configurable": {"model": "qwen3:8b"}})
    fallback = SimpleNamespace(config={"configurable": {"model": "gpt-4o-mini"}})
    wrapper = FallbackChatModel(
        [(primary, "edge", "provider-1"), (fallback, "cloud", "provider-2")]
    )
    assert _resolved_model_name(wrapper) == "qwen3:8b"


def test_resolved_model_name_from_empty_fallback_chain() -> None:
    wrapper = FallbackChatModel([])
    assert _resolved_model_name(wrapper) == "unknown"


def test_resolved_model_name_from_raw_chat_model_attribute() -> None:
    raw = SimpleNamespace(model="gpt-oss")
    assert _resolved_model_name(raw) == "gpt-oss"


def test_resolved_model_name_unknown_fallback() -> None:
    assert _resolved_model_name(object()) == "unknown"
