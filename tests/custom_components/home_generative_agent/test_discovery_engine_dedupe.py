# ruff: noqa: S101
"""Tests for discovery engine novelty filtering."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.home_generative_agent.sentinel.discovery_engine import (
    _MAX_SEMANTIC_KEYS_IN_PROMPT,
    _STATIC_RULE_IDS,
    SentinelDiscoveryEngine,
    _candidate_identity_hash,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.sentinel.discovery_store import (
        DiscoveryStore,
    )


class _DummyStore:
    async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
        return []


class _DummyProposalStore:
    def __init__(self, proposals: list[dict[str, Any]] | None = None) -> None:
        self._proposals = proposals or []

    async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
        return list(self._proposals)

    async def cleanup_unsupported_ttl(self) -> int:
        return 0


def test_filter_novel_candidates_drops_existing_and_batch_duplicates() -> None:
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    candidates = [
        {
            "candidate_id": "c1",
            "title": "Open windows at night while someone is home",
            "summary": "Detect windows open during nighttime when someone is present.",
            "pattern": "window open at night while home",
            "suggested_type": "security_risk",
            "confidence_hint": 0.6,
            "evidence_paths": [
                "entities[entity_id=binary_sensor.playroom_window].state",
                "derived.is_night",
                "derived.anyone_home",
            ],
        },
        {
            "candidate_id": "c2",
            "title": "Garage and playroom windows open while home",
            "summary": "Windows open while occupants are present at night.",
            "pattern": "night home windows open",
            "suggested_type": "security_state",
            "confidence_hint": 0.7,
            "evidence_paths": [
                "derived.anyone_home",
                "entities[entity_id=binary_sensor.playroom_window].state",
                "derived.is_night",
            ],
        },
    ]
    existing_keys = {
        "v1|subject=entry_window|predicate=open|night=1|home=1|scope=any|"
        "entities=binary_sensor.playroom_window"
    }
    filtered, dropped = engine._filter_novel_candidates(candidates, existing_keys)
    assert filtered == []
    assert len(dropped) == 2
    assert dropped[0]["dedupe_reason"] == "existing_semantic_key"
    assert dropped[1]["dedupe_reason"] == "existing_semantic_key"


def test_filter_novel_candidates_sets_novel_reason() -> None:
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    candidates = [
        {
            "candidate_id": "c3",
            "title": "Front door unlocked while home",
            "summary": "Lock left unlocked with occupant home.",
            "pattern": "unlocked lock while home",
            "suggested_type": "security",
            "confidence_hint": 0.8,
            "evidence_paths": [
                "entities[entity_id=lock.front_door].state",
                "derived.anyone_home",
            ],
        }
    ]
    filtered, dropped = engine._filter_novel_candidates(candidates, set())
    assert len(filtered) == 1
    assert filtered[0]["dedupe_reason"] == "novel"
    assert dropped == []


# ---------------------------------------------------------------------------
# Bug 2: null-key candidates (unknown subject/predicate) must be deduplicated
# ---------------------------------------------------------------------------

_NULL_KEY_CANDIDATE: dict[str, Any] = {
    "candidate_id": "nk1",
    "title": "Stale person tracking while away",
    "summary": "Person tracking data appears outdated.",
    "pattern": "stale tracking",
    "suggested_type": "data_quality",
    "confidence_hint": 0.4,
    "evidence_paths": [],
}


def test_candidate_identity_hash_is_stable() -> None:
    """Same title+summary always produces the same hash."""
    h1 = _candidate_identity_hash(_NULL_KEY_CANDIDATE)
    h2 = _candidate_identity_hash(dict(_NULL_KEY_CANDIDATE))
    assert h1 == h2
    assert h1.startswith("ident|sha256=")


def test_candidate_identity_hash_differs_on_content() -> None:
    """Different titles produce different hashes."""
    other = dict(_NULL_KEY_CANDIDATE)
    other["title"] = "Something completely different"
    assert _candidate_identity_hash(_NULL_KEY_CANDIDATE) != _candidate_identity_hash(
        other
    )


def test_filter_null_key_candidate_dropped_when_hash_in_existing() -> None:
    """A null-key candidate whose identity hash is in existing_keys is dropped."""
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    hash_key = _candidate_identity_hash(_NULL_KEY_CANDIDATE)
    filtered, dropped = engine._filter_novel_candidates(
        [_NULL_KEY_CANDIDATE], {hash_key}
    )
    assert filtered == []
    assert len(dropped) == 1
    assert dropped[0]["dedupe_reason"] == "existing_identity_hash"
    assert dropped[0]["identity_hash"] == hash_key


def test_filter_null_key_candidate_batch_dedup() -> None:
    """Two identical null-key candidates in the same batch: second is batch_duplicate."""
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    twin = dict(_NULL_KEY_CANDIDATE)
    twin["candidate_id"] = "nk2"
    filtered, dropped = engine._filter_novel_candidates(
        [_NULL_KEY_CANDIDATE, twin], set()
    )
    assert len(filtered) == 1
    assert len(dropped) == 1
    assert dropped[0]["dedupe_reason"] == "batch_duplicate"


def test_filter_null_key_candidate_novel_when_not_seen() -> None:
    """A null-key candidate that has never been seen passes through as novel."""
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    filtered, dropped = engine._filter_novel_candidates([_NULL_KEY_CANDIDATE], set())
    assert len(filtered) == 1
    assert filtered[0]["dedupe_reason"] == "novel"
    assert dropped == []


# ---------------------------------------------------------------------------
# hint_keys vs filter_keys split
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_keys_in_filter_not_hint() -> None:
    """
    Discovery history record keys must appear in filter_keys but NOT hint_keys.

    This prevents multi-entity bundle history records from misleading the LLM
    into thinking individual entities are already covered.
    """
    history_candidate = {
        "candidate_id": "hist_power_bundle",
        "title": "Kitchen power mismatch",
        "summary": "Multiple kitchen appliances power deviates from baseline.",
        "pattern": "deviation_from_baseline",
        "suggested_type": "statistical_anomaly",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "entities[entity_ids contains sensor.fridge_switch_0_power].state",
            "entities[entity_ids contains sensor.kettle_switch_0_power].state",
        ],
    }

    class _HistoryStore:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return [{"candidates": [history_candidate], "filtered_candidates": []}]

        async def async_append(self, _payload: Any) -> None:
            pass

    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _HistoryStore()),
    )
    _active, hint_keys, filter_keys = await engine._existing_semantic_context()

    # The history candidate's key should appear in filter_keys (post-hoc dedup)
    assert any("fridge" in k for k in filter_keys), (
        "filter_keys must contain history key"
    )
    # But NOT in hint_keys (LLM prompt) — it would suppress standalone fridge proposals
    assert not any("fridge" in k for k in hint_keys), (
        "hint_keys must NOT contain history keys (they mislead the LLM)"
    )


# ---------------------------------------------------------------------------
# Bug 1: rejected proposals must still block re-suggestion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_context_rejected_proposal_adds_key() -> None:
    """A rejected proposal's candidate key must appear in semantic_keys."""
    rejected_candidate = {
        "candidate_id": "r1",
        "title": "Front door unlocked while home",
        "summary": "Lock left unlocked with occupant home.",
        "pattern": "unlocked lock while home",
        "suggested_type": "security",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "entities[entity_id=lock.front_door].state",
            "derived.anyone_home",
        ],
    }
    proposal = {
        "candidate_id": "r1",
        "candidate": rejected_candidate,
        "status": "rejected",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    proposal_store = _DummyProposalStore([proposal])
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
        proposal_store=cast("Any", proposal_store),
    )
    _, hint_keys, _filter_keys = await engine._existing_semantic_context()
    # The lock+home semantic key must be present even though status=="rejected".
    assert any("lock" in k and "unlocked" in k for k in hint_keys)


@pytest.mark.asyncio
async def test_existing_context_null_key_rejected_proposal_adds_hash() -> None:
    """A rejected null-key proposal's identity hash must appear in semantic_keys."""
    proposal = {
        "candidate_id": "nk_r1",
        "candidate": _NULL_KEY_CANDIDATE,
        "status": "rejected",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    proposal_store = _DummyProposalStore([proposal])
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
        proposal_store=cast("Any", proposal_store),
    )
    _, hint_keys, _filter_keys = await engine._existing_semantic_context()
    expected_hash = _candidate_identity_hash(_NULL_KEY_CANDIDATE)
    assert expected_hash in hint_keys


# ---------------------------------------------------------------------------
# Bug 5: accepted proposals must NOT block re-suggestion when rule is disabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_context_approved_proposal_does_not_add_key_to_hints() -> None:
    """
    An approved proposal must NOT appear in hint_keys.

    When a proposal is approved, a rule is created to track coverage.  If the
    user later disables that rule, the topic should become re-proposable.
    Keeping the approved proposal in hint_keys would silently suppress it
    forever, regardless of whether the rule is still active.
    """
    accepted_candidate = {
        "candidate_id": "a1",
        "title": "Fridge power baseline deviation",
        "summary": "Fridge power deviates from rolling average.",
        "pattern": "deviation_from_normal",
        "suggested_type": "statistical_anomaly",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "entities[entity_ids contains sensor.fridge_switch_0_power].state",
        ],
    }
    proposal = {
        "candidate_id": "a1",
        "candidate": accepted_candidate,
        "status": "approved",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    proposal_store = _DummyProposalStore([proposal])
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
        proposal_store=cast("Any", proposal_store),
    )
    _, hint_keys, _filter_keys = await engine._existing_semantic_context()
    # Accepted proposal key must NOT block re-proposal of the fridge.
    assert not any("fridge" in k for k in hint_keys), (
        "Accepted proposal key must not appear in hint_keys"
    )


@pytest.mark.asyncio
async def test_existing_context_pending_proposal_still_blocks() -> None:
    """A pending proposal's candidate key must still appear in hint_keys."""
    pending_candidate = {
        "candidate_id": "p1",
        "title": "Front door unlocked while home",
        "summary": "Lock left unlocked with occupant home.",
        "pattern": "unlocked lock while home",
        "suggested_type": "security",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "entities[entity_id=lock.front_door].state",
            "derived.anyone_home",
        ],
    }
    proposal = {
        "candidate_id": "p1",
        "candidate": pending_candidate,
        "status": "pending",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    proposal_store = _DummyProposalStore([proposal])
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
        proposal_store=cast("Any", proposal_store),
    )
    _, hint_keys, _filter_keys = await engine._existing_semantic_context()
    assert any("lock" in k and "unlocked" in k for k in hint_keys)


# ---------------------------------------------------------------------------
# Bug 4: static built-in rule IDs must appear in active_rule_ids
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_context_includes_static_rule_ids() -> None:
    """Static built-in rule IDs must be in active_rule_ids even with no registry."""
    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
    )
    active_rule_ids, _hint, _filter = await engine._existing_semantic_context()
    assert _STATIC_RULE_IDS.issubset(active_rule_ids)


@pytest.mark.asyncio
async def test_existing_context_static_ids_present_alongside_dynamic() -> None:
    """Static IDs appear alongside any dynamic rule IDs from the registry."""

    class _DummyRegistry:
        def list_rules(self) -> list[dict[str, Any]]:
            return [{"rule_id": "my_dynamic_rule"}]

    engine = SentinelDiscoveryEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        model=None,
        store=cast("DiscoveryStore", _DummyStore()),
        rule_registry=cast("Any", _DummyRegistry()),
    )
    active_rule_ids, _hint, _filter = await engine._existing_semantic_context()
    assert "my_dynamic_rule" in active_rule_ids
    assert _STATIC_RULE_IDS.issubset(active_rule_ids)


# ---------------------------------------------------------------------------
# Semantic key prompt cap tests
# ---------------------------------------------------------------------------


def test_max_semantic_keys_in_prompt_constant() -> None:
    """_MAX_SEMANTIC_KEYS_IN_PROMPT must be defined and positive."""
    assert isinstance(_MAX_SEMANTIC_KEYS_IN_PROMPT, int)
    assert _MAX_SEMANTIC_KEYS_IN_PROMPT > 0


@pytest.mark.asyncio
async def test_discovery_prompt_caps_semantic_keys(hass: HomeAssistant) -> None:
    """When existing_semantic_keys exceeds the cap, only cap entries reach the prompt."""
    oversized_keys = {f"key_{i}" for i in range(_MAX_SEMANTIC_KEYS_IN_PROMPT + 20)}
    captured_prompts: list[str] = []

    class _CapturingModel:
        async def ainvoke(self, messages: list[Any]) -> Any:
            captured_prompts.extend(
                str(msg.content) for msg in messages if hasattr(msg, "content")
            )
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at": "2026-01-01T00:00:00Z",
                        "model": "test",
                        "candidates": [],
                    }
                )
            )

    class _FullDummyStore:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return []

        async def async_append(self, _payload: Any) -> None:
            pass

    engine = SentinelDiscoveryEngine(
        hass=hass,
        options={},
        model=_CapturingModel(),
        store=_FullDummyStore(),  # type: ignore[arg-type]
    )

    async def _fake_run(model: Any, messages: Any, **_kw: Any) -> Any:
        return await model.ainvoke(messages)

    with (
        patch.object(
            engine,
            "_existing_semantic_context",
            new_callable=AsyncMock,
            return_value=(set(), oversized_keys, oversized_keys),
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.async_build_full_state_snapshot",
            new_callable=AsyncMock,
            return_value={
                "entities": [],
                "camera_activity": [],
                "derived": {"is_night": False, "now": "2026-01-01T00:00:00Z"},
                "generated_at": "2026-01-01T00:00:00Z",
            },
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.run_sentinel_model_call",
            side_effect=_fake_run,
        ),
    ):
        await engine._run_once()

    assert captured_prompts, "Model was never invoked"
    # Parse the JSON array from the human message to get an exact key count.
    human_content = captured_prompts[-1]
    keys_start = human_content.find("Existing semantic keys (do not duplicate): [")
    assert keys_start != -1, "Prompt missing existing_semantic_keys section"
    array_start = human_content.index("[", keys_start)
    array_end = human_content.index("]", array_start) + 1
    keys_in_prompt: list[str] = json.loads(human_content[array_start:array_end])
    assert len(keys_in_prompt) <= _MAX_SEMANTIC_KEYS_IN_PROMPT


# ---------------------------------------------------------------------------
# Monitoring gap: unavailability keys must not suppress baseline-ready entities
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_monitoring_gap_ignores_non_baseline_hint_keys(
    hass: HomeAssistant,
) -> None:
    """
    A sensor in a predicate=unavailable key must still be in unmonitored.

    Broad rules like unavailable_sensors cover many entity_ids as side-effects;
    the gap analysis must only check baseline/power-anomaly keys.
    """
    # Hint key that mentions the fridge entity_id but is for unavailability, not
    # baseline monitoring.
    unavail_key = (
        "v1|subject=sensor|predicate=unavailable|night=any|home=1|scope=any|"
        "entities=sensor.backyard_vmd3_0,sensor.fridge_switch_0_power"
    )
    captured_prompts: list[str] = []

    class _CapturingModel:
        async def ainvoke(self, messages: list[Any]) -> Any:
            captured_prompts.extend(
                str(msg.content) for msg in messages if hasattr(msg, "content")
            )
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at": "2026-01-01T00:00:00Z",
                        "model": "test",
                        "candidates": [],
                    }
                )
            )

    class _FullDummyStore:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return []

        async def async_append(self, _payload: Any) -> None:
            pass

    engine = SentinelDiscoveryEngine(
        hass=hass,
        options={},
        model=_CapturingModel(),
        store=_FullDummyStore(),  # type: ignore[arg-type]
    )

    async def _fake_run(model: Any, messages: Any, **_kw: Any) -> Any:
        return await model.ainvoke(messages)

    with (
        patch.object(
            engine,
            "_existing_semantic_context",
            new_callable=AsyncMock,
            return_value=(set(), {unavail_key}, {unavail_key}),
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.async_build_full_state_snapshot",
            new_callable=AsyncMock,
            return_value={
                "entities": [],
                "camera_activity": [],
                "derived": {
                    "is_night": False,
                    "now": "2026-01-01T00:00:00Z",
                    "baseline_ready_entities": ["sensor.fridge_switch_0_power"],
                },
                "generated_at": "2026-01-01T00:00:00Z",
            },
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.run_sentinel_model_call",
            side_effect=_fake_run,
        ),
    ):
        await engine._run_once()

    assert captured_prompts, "Model was never invoked"
    human_content = captured_prompts[-1]
    # The fridge must appear in the MONITORING GAPS section even though it
    # appears in an unavailability hint key.
    assert "sensor.fridge_switch_0_power" in human_content, (
        "Fridge must be in MONITORING GAPS when only covered by unavailability key"
    )


@pytest.mark.asyncio
async def test_monitoring_gap_suppressed_by_power_anomaly_key(
    hass: HomeAssistant,
) -> None:
    """A sensor with an active power_anomaly key must NOT be in unmonitored."""
    power_key = (
        "v1|subject=sensor|predicate=power_anomaly"
        "|template=baseline_deviation|entities=sensor.fridge_switch_0_power"
    )
    captured_prompts: list[str] = []

    class _CapturingModel:
        async def ainvoke(self, messages: list[Any]) -> Any:
            captured_prompts.extend(
                str(msg.content) for msg in messages if hasattr(msg, "content")
            )
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at": "2026-01-01T00:00:00Z",
                        "model": "test",
                        "candidates": [],
                    }
                )
            )

    class _FullDummyStore:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return []

        async def async_append(self, _payload: Any) -> None:
            pass

    engine = SentinelDiscoveryEngine(
        hass=hass,
        options={},
        model=_CapturingModel(),
        store=_FullDummyStore(),  # type: ignore[arg-type]
    )

    async def _fake_run(model: Any, messages: Any, **_kw: Any) -> Any:
        return await model.ainvoke(messages)

    with (
        patch.object(
            engine,
            "_existing_semantic_context",
            new_callable=AsyncMock,
            return_value=(set(), {power_key}, {power_key}),
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.async_build_full_state_snapshot",
            new_callable=AsyncMock,
            return_value={
                "entities": [],
                "camera_activity": [],
                "derived": {
                    "is_night": False,
                    "now": "2026-01-01T00:00:00Z",
                    "baseline_ready_entities": ["sensor.fridge_switch_0_power"],
                },
                "generated_at": "2026-01-01T00:00:00Z",
            },
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.run_sentinel_model_call",
            side_effect=_fake_run,
        ),
    ):
        await engine._run_once()

    assert captured_prompts, "Model was never invoked"
    human_content = captured_prompts[-1]
    gaps_start = human_content.find("MONITORING GAPS:")
    assert gaps_start != -1
    gaps_section = human_content[gaps_start : gaps_start + 200]
    # Fridge is already baseline-monitored; it must NOT appear in MONITORING GAPS.
    assert "sensor.fridge_switch_0_power" not in gaps_section, (
        "Fridge must not be in MONITORING GAPS when covered by power_anomaly key"
    )


@pytest.mark.asyncio
async def test_monitoring_gap_bundle_candidate_key_does_not_suppress(
    hass: HomeAssistant,
) -> None:
    """
    A multi-entity bundle candidate key must NOT suppress individual entity gaps.

    A rejected proposal that bundles many appliances into one candidate key
    (no |template=| marker) must not prevent each individual appliance from
    appearing in unmonitored_baseline_entities.
    """
    # A candidate key covering 8 appliances as a bundle (no template= marker).
    bundle_key = (
        "v1|subject=sensor|predicate=power_anomaly|night=any|home=any|scope=any|"
        "entities=sensor.dishwasher_switch_0_energy,sensor.dishwasher_switch_0_power,"
        "sensor.fridge_switch_0_energy,sensor.fridge_switch_0_power,"
        "sensor.kettle_switch_0_energy,sensor.kettle_switch_0_power,"
        "sensor.microwave_switch_0_energy,sensor.microwave_switch_0_power"
    )
    captured_prompts: list[str] = []

    class _CapturingModel:
        async def ainvoke(self, messages: list[Any]) -> Any:
            captured_prompts.extend(
                str(msg.content) for msg in messages if hasattr(msg, "content")
            )
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "schema_version": 1,
                        "generated_at": "2026-01-01T00:00:00Z",
                        "model": "test",
                        "candidates": [],
                    }
                )
            )

    class _FullDummyStore:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return []

        async def async_append(self, _payload: Any) -> None:
            pass

    engine = SentinelDiscoveryEngine(
        hass=hass,
        options={},
        model=_CapturingModel(),
        store=_FullDummyStore(),  # type: ignore[arg-type]
    )

    async def _fake_run(model: Any, messages: Any, **_kw: Any) -> Any:
        return await model.ainvoke(messages)

    with (
        patch.object(
            engine,
            "_existing_semantic_context",
            new_callable=AsyncMock,
            return_value=(set(), {bundle_key}, {bundle_key}),
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.async_build_full_state_snapshot",
            new_callable=AsyncMock,
            return_value={
                "entities": [],
                "camera_activity": [],
                "derived": {
                    "is_night": False,
                    "now": "2026-01-01T00:00:00Z",
                    "baseline_ready_entities": ["sensor.fridge_switch_0_power"],
                },
                "generated_at": "2026-01-01T00:00:00Z",
            },
        ),
        patch(
            "custom_components.home_generative_agent.sentinel.discovery_engine.run_sentinel_model_call",
            side_effect=_fake_run,
        ),
    ):
        await engine._run_once()

    assert captured_prompts, "Model was never invoked"
    human_content = captured_prompts[-1]
    # Bundle key has no |template=| marker — fridge must still appear in MONITORING GAPS.
    assert "sensor.fridge_switch_0_power" in human_content, (
        "Fridge must appear in MONITORING GAPS even when covered only by a bundle key"
    )
