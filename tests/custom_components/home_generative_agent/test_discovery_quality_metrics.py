"""
Tests for discovery quality metrics (Item 2).

Covers:
- _discovery_cycle_stats reset at each _run_once() start
- candidates_generated / candidates_novel / candidates_deduplicated incremented
- unsupported_ttl_expired incremented when cleanup returns > 0
- discovery_cycle_stats property returns a copy
- SentinelHealthSensor exposes discovery_* attributes when engine is present
- SentinelHealthSensor exposes None discovery_* attributes when engine is absent
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.home_generative_agent.sentinel.discovery_engine import (
    SentinelDiscoveryEngine,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.sentinel.discovery_store import (
        DiscoveryStore,
    )

# Minimal snapshot returned by the patched async_build_full_state_snapshot.
_MINIMAL_SNAPSHOT: dict[str, Any] = {"entities": [], "derived": {}}

_SNAPSHOT_PATCH = patch(
    "custom_components.home_generative_agent.sentinel.discovery_engine"
    ".async_build_full_state_snapshot",
    new=AsyncMock(return_value=_MINIMAL_SNAPSHOT),
)
_REDUCER_PATCH = patch(
    "custom_components.home_generative_agent.sentinel.discovery_engine"
    ".reduce_snapshot_for_discovery",
    return_value={},
)


# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------


class _DummyStore:
    async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
        return []

    async def async_append(self, _payload: dict[str, Any]) -> None:
        pass


class _DummyProposalStore:
    def __init__(
        self, proposals: list[dict[str, Any]] | None = None, ttl_expired: int = 0
    ) -> None:
        self._proposals = proposals or []
        self._ttl_expired = ttl_expired

    async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
        return list(self._proposals)

    async def cleanup_unsupported_ttl(self) -> int:
        return self._ttl_expired


def _make_engine(
    model: Any = None,
    proposal_store: Any = None,
) -> SentinelDiscoveryEngine:
    return SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=model,
        store=cast("DiscoveryStore", _DummyStore()),
        proposal_store=proposal_store,
    )


# ---------------------------------------------------------------------------
# Stats are reset at start of each _run_once()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stats_reset_each_run() -> None:
    """_discovery_cycle_stats is fully reset at the start of every run."""
    engine = _make_engine()
    # Manually set stale values to confirm they get wiped.
    engine._discovery_cycle_stats = {
        "candidates_generated": 99,
        "candidates_novel": 99,
        "candidates_deduplicated": 99,
        "proposals_promoted": 99,
        "unsupported_ttl_expired": 99,
    }
    # No model → returns early after reset.
    await engine._run_once()
    stats = engine.discovery_cycle_stats
    assert stats["candidates_generated"] == 0
    assert stats["candidates_novel"] == 0
    assert stats["candidates_deduplicated"] == 0
    assert stats["proposals_promoted"] == 0
    assert stats["unsupported_ttl_expired"] == 0


# ---------------------------------------------------------------------------
# discovery_cycle_stats property returns a copy (mutation safety)
# ---------------------------------------------------------------------------


def test_discovery_cycle_stats_property_returns_copy() -> None:
    engine = _make_engine()
    engine._discovery_cycle_stats = {"candidates_generated": 3}
    stats = engine.discovery_cycle_stats
    stats["candidates_generated"] = 999
    # Original must be unaffected.
    assert engine._discovery_cycle_stats["candidates_generated"] == 3


# ---------------------------------------------------------------------------
# unsupported_ttl_expired increments when cleanup returns > 0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsupported_ttl_expired_incremented() -> None:
    """unsupported_ttl_expired accumulates the value returned by cleanup."""
    proposal_store = _DummyProposalStore(ttl_expired=5)
    model = AsyncMock()
    model.ainvoke = AsyncMock(return_value=MagicMock(content=""))
    engine = _make_engine(model=model, proposal_store=proposal_store)
    with _SNAPSHOT_PATCH, _REDUCER_PATCH:
        await engine._run_once()
    assert engine.discovery_cycle_stats["unsupported_ttl_expired"] == 5


@pytest.mark.asyncio
async def test_unsupported_ttl_expired_zero_when_no_cleanup() -> None:
    """unsupported_ttl_expired stays 0 when cleanup returns 0."""
    proposal_store = _DummyProposalStore(ttl_expired=0)
    model = AsyncMock()
    model.ainvoke = AsyncMock(return_value=MagicMock(content=""))
    engine = _make_engine(model=model, proposal_store=proposal_store)
    with _SNAPSHOT_PATCH, _REDUCER_PATCH:
        await engine._run_once()
    assert engine.discovery_cycle_stats["unsupported_ttl_expired"] == 0


# ---------------------------------------------------------------------------
# candidates_generated / candidates_novel / candidates_deduplicated
# ---------------------------------------------------------------------------


def _valid_llm_response(candidates: list[dict[str, Any]]) -> MagicMock:
    """Build a fake LLM result object whose content is valid JSON."""
    payload = {
        "schema_version": 1,
        "generated_at": "2026-01-01T00:00:00",
        "model": "test-model",
        "candidates": candidates,
    }
    mock = MagicMock()
    mock.content = json.dumps(payload)
    return mock


@pytest.mark.asyncio
async def test_candidates_generated_counts_all_returned() -> None:
    """candidates_generated reflects raw LLM output count."""
    candidates = [
        {
            "candidate_id": f"c{i}",
            "title": f"Novel candidate {i}",
            "summary": f"Summary {i}",
            "pattern": f"pattern {i}",
            "confidence_hint": 0.5,
            "evidence_paths": [f"entities[entity_ids contains sensor.foo_{i}].state"],
        }
        for i in range(3)
    ]
    model = AsyncMock()
    model.ainvoke = AsyncMock(return_value=_valid_llm_response(candidates))
    engine = _make_engine(model=model)
    with _SNAPSHOT_PATCH, _REDUCER_PATCH:
        await engine._run_once()
    stats = engine.discovery_cycle_stats
    assert stats["candidates_generated"] == 3


@pytest.mark.asyncio
async def test_candidates_novel_vs_deduplicated_counts() -> None:
    """Duplicate candidate is counted in candidates_deduplicated, novel in candidates_novel."""
    # Two candidates with the same title/summary → identical identity hash → second is batch duplicate.
    raw = [
        {
            "candidate_id": "c1",
            "title": "Unique novel idea",
            "summary": "Details about the unique novel idea.",
            "pattern": "some pattern",
            "confidence_hint": 0.7,
            "evidence_paths": ["entities[entity_ids contains lock.front_door].state"],
        },
        {
            "candidate_id": "c2",
            "title": "Unique novel idea",  # same title
            "summary": "Details about the unique novel idea.",  # same summary → hash match
            "pattern": "some pattern",
            "confidence_hint": 0.6,
            "evidence_paths": ["entities[entity_ids contains lock.front_door].state"],
        },
    ]
    model = AsyncMock()
    model.ainvoke = AsyncMock(return_value=_valid_llm_response(raw))
    engine = _make_engine(model=model)
    with _SNAPSHOT_PATCH, _REDUCER_PATCH:
        await engine._run_once()
    stats = engine.discovery_cycle_stats
    assert stats["candidates_generated"] == 2
    assert stats["candidates_novel"] == 1
    assert stats["candidates_deduplicated"] == 1


# ---------------------------------------------------------------------------
# derived_only_paths filter — candidates with only derived.* evidence dropped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derived_only_paths_candidate_is_dropped() -> None:
    """Candidates whose every evidence_path starts with 'derived.' are dropped."""
    raw = [
        {
            "candidate_id": "c_derived",
            "title": "Derived-only candidate",
            "summary": "Summary of derived-only candidate.",
            "pattern": "some pattern",
            "confidence_hint": 0.6,
            # All paths are derived.* → should be dropped before dedup
            "evidence_paths": ["derived.is_night", "derived.anyone_home"],
        },
        {
            "candidate_id": "c_concrete",
            "title": "Concrete entity candidate",
            "summary": "Summary of concrete candidate.",
            "pattern": "some pattern",
            "confidence_hint": 0.7,
            "evidence_paths": ["entities[entity_ids contains lock.front_door].state"],
        },
    ]
    model = AsyncMock()
    model.ainvoke = AsyncMock(return_value=_valid_llm_response(raw))
    engine = _make_engine(model=model)
    with _SNAPSHOT_PATCH, _REDUCER_PATCH:
        await engine._run_once()
    stats = engine.discovery_cycle_stats
    # Both candidates counted as generated
    assert stats["candidates_generated"] == 2
    # Only the concrete candidate survives
    assert stats["candidates_novel"] == 1
    # The derived-only candidate is counted as deduplicated/dropped
    assert stats["candidates_deduplicated"] == 1


# ---------------------------------------------------------------------------
# SentinelHealthSensor exposes discovery_* attributes
# ---------------------------------------------------------------------------


def _make_health_sensor(discovery_engine: Any = None) -> Any:
    """Instantiate SentinelHealthSensor with minimal deps."""
    from custom_components.home_generative_agent.core.sentinel_health_sensor import (
        SentinelHealthSensor,
    )

    hass = MagicMock()
    hass.states = MagicMock()
    audit_store = MagicMock()
    audit_store.async_get_latest = AsyncMock(return_value=[])
    return SentinelHealthSensor(
        hass=hass,
        options={},
        audit_store=audit_store,
        sentinel=None,
        entry_id="test-entry",
        baseline_updater=None,
        discovery_engine=discovery_engine,
    )


@pytest.mark.asyncio
async def test_health_sensor_exposes_discovery_stats_when_engine_present() -> None:
    """SentinelHealthSensor reads discovery_cycle_stats from the engine."""
    engine = _make_engine()
    engine._discovery_cycle_stats = {
        "candidates_generated": 4,
        "candidates_novel": 2,
        "candidates_deduplicated": 2,
        "proposals_promoted": 0,
        "unsupported_ttl_expired": 1,
    }
    sensor = _make_health_sensor(discovery_engine=engine)
    sensor.async_write_ha_state = MagicMock()
    await sensor._async_refresh()

    attrs = sensor._attrs
    assert attrs["discovery_candidates_generated"] == 4
    assert attrs["discovery_candidates_novel"] == 2
    assert attrs["discovery_candidates_deduplicated"] == 2
    assert attrs["discovery_proposals_promoted"] == 0
    assert attrs["discovery_unsupported_ttl_expired"] == 1


@pytest.mark.asyncio
async def test_health_sensor_exposes_none_discovery_attrs_when_no_engine() -> None:
    """When discovery_engine is None, all discovery_* attributes are None."""
    sensor = _make_health_sensor(discovery_engine=None)
    sensor.async_write_ha_state = MagicMock()
    await sensor._async_refresh()

    attrs = sensor._attrs
    assert attrs["discovery_candidates_generated"] is None
    assert attrs["discovery_candidates_novel"] is None
    assert attrs["discovery_candidates_deduplicated"] is None
    assert attrs["discovery_proposals_promoted"] is None
    assert attrs["discovery_unsupported_ttl_expired"] is None
