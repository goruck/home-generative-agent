"""Tests for Sentinel service handlers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.const import CONF_NOTIFY_SERVICE
from custom_components.home_generative_agent.sentinel.discovery_store import (
    DiscoveryStore,
)
from custom_components.home_generative_agent.sentinel.dynamic_rules import (
    evaluate_dynamic_rule,
)
from custom_components.home_generative_agent.sentinel.proposal_store import (
    ProposalStore,
)
from custom_components.home_generative_agent.snapshot.builder import (
    async_build_full_state_snapshot,
)

_hga_component = cast("Any", hga_component)


class DummyRuleRegistry:
    """Minimal dynamic rule registry stand-in for service tests."""

    def __init__(
        self,
        *,
        rules: list[dict[str, Any]] | None = None,
        add_result: bool = True,
    ) -> None:
        self._rules = list(rules or [])
        self._add_result = add_result
        self.added_rules: list[dict[str, Any]] = []

    def list_rules(self, include_disabled: bool = False) -> list[dict[str, Any]]:
        _ = include_disabled
        return list(self._rules)

    def find_rule(self, rule_id: str) -> dict[str, Any] | None:
        for rule in self._rules:
            if rule.get("rule_id") == rule_id:
                return rule
        return None

    async def async_add_rule(self, rule_spec: dict[str, Any]) -> bool:
        self.added_rules.append(rule_spec)
        if self._add_result:
            self._rules.append(rule_spec)
        return self._add_result

    async def async_set_rule_enabled(self, rule_id: str, *, enabled: bool) -> bool:
        _ = (rule_id, enabled)
        return True


def _make_entry(
    *,
    discovery_store: DiscoveryStore | None = None,
    discovery_engine: Any = None,
    proposal_store: ProposalStore | None = None,
    rule_registry: Any = None,
    sentinel: Any = None,
    options: dict[str, Any] | None = None,
) -> Any:
    runtime_data = SimpleNamespace(
        options=options or {},
        audit_store=None,
        discovery_store=discovery_store,
        discovery_engine=discovery_engine,
        proposal_store=proposal_store,
        rule_registry=rule_registry,
        sentinel=sentinel,
        person_gallery=None,
        face_api_url="http://face-api",
    )
    return SimpleNamespace(entry_id="entry1", runtime_data=runtime_data)


@pytest.mark.asyncio
async def test_trigger_sentinel_discovery_service_runs_engine() -> None:
    discovery_engine = SimpleNamespace(async_run_now=AsyncMock(return_value=True))
    entry = _make_entry(discovery_engine=discovery_engine)
    response = await _hga_component._trigger_sentinel_discovery(entry)

    assert response == {"status": "ok"}
    discovery_engine.async_run_now.assert_awaited_once()


@pytest.mark.asyncio
async def test_promote_discovery_candidate_notification_is_richer(hass) -> None:
    discovery_store = DiscoveryStore(hass, max_records=10)
    await discovery_store.async_append(
        {
            "schema_version": 1,
            "generated_at": "2026-01-01T00:00:00+00:00",
            "model": "test",
            "candidates": [
                {
                    "candidate_id": "c_lock",
                    "title": "Garage lock unlocked while home",
                    "summary": "Detect lock left unlocked with someone present.",
                    "pattern": "lock unlocked while home",
                    "suggested_type": "security",
                    "confidence_hint": 0.8,
                    "evidence_paths": [
                        "entities[entity_id=lock.garage_door_lock].state",
                        "derived.anyone_home",
                    ],
                }
            ],
        }
    )
    proposal_store = ProposalStore(hass)
    notifications: list[dict[str, Any]] = []

    async def _capture_notify(call) -> None:
        notifications.append(dict(call.data))

    hass.services.async_register("notify", "mobile_app_test", _capture_notify)
    entry = _make_entry(
        discovery_store=discovery_store,
        proposal_store=proposal_store,
        rule_registry=DummyRuleRegistry(),
        options={CONF_NOTIFY_SERVICE: "notify.mobile_app_test"},
    )
    response = await _hga_component._promote_discovery_candidate(
        hass,
        entry,
        candidate_id="c_lock",
    )

    assert response == {"status": "ok", "candidate_id": "c_lock"}
    assert notifications
    assert notifications[0]["data"]["template_id"] == "unlocked_lock_when_home"
    assert notifications[0]["data"]["severity"] == "medium"
    assert notifications[0]["data"]["confidence"] == 0.8
    assert notifications[0]["data"]["service_hint"] == "approve_rule_proposal"


@pytest.mark.asyncio
async def test_approve_rule_proposal_returns_normalization_reason(hass) -> None:
    proposal_store = ProposalStore(hass)
    await proposal_store.async_append(
        {
            "candidate_id": "bad_lock",
            "candidate": {
                "candidate_id": "bad_lock",
                "title": "Front lock unlocked while home",
                "summary": "Detect unlocked lock with someone present.",
                "pattern": "lock unlocked while home",
                "suggested_type": "security",
                "confidence_hint": 0.8,
                "evidence_paths": ["derived.anyone_home"],
            },
            "notes": "",
            "status": "draft",
        }
    )
    entry = _make_entry(
        proposal_store=proposal_store,
        rule_registry=DummyRuleRegistry(),
        sentinel=SimpleNamespace(async_run_now=AsyncMock(return_value=True)),
    )
    response = await _hga_component._approve_rule_proposal(
        entry,
        candidate_id="bad_lock",
    )

    assert response["status"] == "unsupported"
    assert response["reason_code"] == "missing_required_entities"


@pytest.mark.asyncio
async def test_approve_rule_proposal_returns_builtin_vehicle_rule_coverage(
    hass,
) -> None:
    proposal_store = ProposalStore(hass)
    await proposal_store.async_append(
        {
            "candidate_id": "vehicle_parked_frontgate_home",
            "candidate": {
                "candidate_id": "vehicle_parked_frontgate_home",
                "title": "Vehicle Parked Near Front Gate While Home",
                "summary": "A vehicle is parked near the front gate while residents are home.",
                "pattern": "camera.frontgate vehicle while home",
                "suggested_type": "security",
                "confidence_hint": 0.8,
                "evidence_paths": [
                    "camera_activity[camera_entity_id=camera.frontgate].snapshot_summary",
                    "derived.anyone_home",
                ],
            },
            "notes": "",
            "status": "draft",
        }
    )
    entry = _make_entry(
        proposal_store=proposal_store,
        rule_registry=DummyRuleRegistry(),
        sentinel=SimpleNamespace(async_run_now=AsyncMock(return_value=True)),
    )

    response = await _hga_component._approve_rule_proposal(
        entry,
        candidate_id="vehicle_parked_frontgate_home",
    )

    assert response["status"] == "covered_by_existing_rule"
    assert response["rule_id"] == "vehicle_detected_near_camera_home"
    assert response["overlapping_entity_ids"] == ["camera.frontgate"]


@pytest.mark.asyncio
async def test_approve_rule_proposal_returns_builtin_vehicle_rule_coverage_no_domain_prefix(
    hass,
) -> None:
    """LLM-generated evidence paths without 'camera.' prefix still match the static rule."""
    proposal_store = ProposalStore(hass)
    await proposal_store.async_append(
        {
            "candidate_id": "vehicle_parked_frontgate_home",
            "candidate": {
                "candidate_id": "vehicle_parked_frontgate_home",
                "title": "Vehicle Parked Near Front Gate While Home",
                "summary": "A vehicle is parked near the front gate while residents are home.",
                "pattern": "camera.frontgate vehicle while home",
                "suggested_type": "security",
                "confidence_hint": 0.8,
                "evidence_paths": [
                    "camera_activity[camera_entity_id=frontgate].snapshot_summary",
                    "derived.anyone_home",
                ],
            },
            "notes": "",
            "status": "draft",
        }
    )
    entry = _make_entry(
        proposal_store=proposal_store,
        rule_registry=DummyRuleRegistry(),
        sentinel=SimpleNamespace(async_run_now=AsyncMock(return_value=True)),
    )

    response = await _hga_component._approve_rule_proposal(
        entry,
        candidate_id="vehicle_parked_frontgate_home",
    )

    assert response["status"] == "covered_by_existing_rule"
    assert response["rule_id"] == "vehicle_detected_near_camera_home"


@pytest.mark.asyncio
async def test_approve_rule_proposal_returns_builtin_camera_snapshot_coverage(
    hass,
) -> None:
    proposal_store = ProposalStore(hass)
    await proposal_store.async_append(
        {
            "candidate_id": "camera_backgarage_missing_snapshot_night_home",
            "candidate": {
                "candidate_id": "camera_backgarage_missing_snapshot_night_home",
                "title": "Camera backgarage missing snapshot summary at night while home present",
                "summary": "The backgarage camera has no snapshot summary recorded during nighttime while someone is home.",
                "pattern": "camera.backgarage missing snapshot at night while home",
                "suggested_type": "availability",
                "confidence_hint": 0.8,
                "evidence_paths": [
                    "camera_activity[camera_entity_id=camera.backgarage].snapshot_summary",
                    "derived.is_night",
                    "derived.anyone_home",
                ],
            },
            "notes": "",
            "status": "draft",
        }
    )
    entry = _make_entry(
        proposal_store=proposal_store,
        rule_registry=DummyRuleRegistry(),
        sentinel=SimpleNamespace(async_run_now=AsyncMock(return_value=True)),
    )

    response = await _hga_component._approve_rule_proposal(
        entry,
        candidate_id="camera_backgarage_missing_snapshot_night_home",
    )

    assert response["status"] == "covered_by_existing_rule"
    assert response["rule_id"] == "camera_missing_snapshot_night_home"
    assert response["overlapping_entity_ids"] == ["camera.backgarage"]


@pytest.mark.asyncio
async def test_approve_rule_proposal_returns_overlap_when_already_active(hass) -> None:
    proposal_store = ProposalStore(hass)
    candidate = {
        "candidate_id": "covered_lock",
        "title": "Garage lock unlocked while home",
        "summary": "Detect lock left unlocked with someone present.",
        "pattern": "lock unlocked while home",
        "suggested_type": "security",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "entities[entity_id=lock.garage_door_lock].state",
            "derived.anyone_home",
        ],
    }
    await proposal_store.async_append(
        {
            "candidate_id": "covered_lock",
            "candidate": candidate,
            "notes": "",
            "status": "draft",
        }
    )
    registry = DummyRuleRegistry(
        rules=[
            {
                "rule_id": "unlocked_lock_when_home_lock_garage_door_lock",
                "template_id": "unlocked_lock_when_home",
                "params": {"lock_entity_id": "lock.garage_door_lock"},
                "severity": "medium",
                "confidence": 0.8,
                "is_sensitive": True,
                "suggested_actions": ["lock.lock", "lock_entity"],
            }
        ]
    )
    entry = _make_entry(
        proposal_store=proposal_store,
        rule_registry=registry,
        sentinel=SimpleNamespace(async_run_now=AsyncMock(return_value=True)),
    )
    response = await _hga_component._approve_rule_proposal(
        entry,
        candidate_id="covered_lock",
    )

    assert response["status"] == "covered_by_existing_rule"
    assert response["rule_id"] == "unlocked_lock_when_home_lock_garage_door_lock"
    assert response["overlapping_entity_ids"] == ["lock.garage_door_lock"]


@pytest.mark.asyncio
async def test_preview_rule_proposal_returns_current_trigger_state(hass) -> None:
    hass.states.async_set("person.alex", "home")
    hass.states.async_set("lock.garage_door_lock", "unlocked")

    proposal_store = ProposalStore(hass)
    candidate = {
        "candidate_id": "c_lock",
        "title": "Garage lock unlocked while home",
        "summary": "Detect lock left unlocked with someone present.",
        "pattern": "lock unlocked while home",
        "suggested_type": "security",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "entities[entity_id=lock.garage_door_lock].state",
            "derived.anyone_home",
        ],
    }
    await proposal_store.async_append(
        {
            "candidate_id": "c_lock",
            "candidate": candidate,
            "notes": "",
            "status": "draft",
        }
    )
    registry = DummyRuleRegistry()
    entry = _make_entry(proposal_store=proposal_store, rule_registry=registry)

    response = await _hga_component._preview_rule_proposal(
        hass,
        entry,
        candidate_id="c_lock",
    )

    snapshot = await async_build_full_state_snapshot(hass)
    expected_findings = evaluate_dynamic_rule(
        snapshot,
        {
            "rule_id": "unlocked_lock_when_home_lock_garage_door_lock",
            "template_id": "unlocked_lock_when_home",
            "params": {"lock_entity_id": "lock.garage_door_lock"},
            "severity": "medium",
            "confidence": 0.8,
            "is_sensitive": True,
            "suggested_actions": ["lock.lock", "lock_entity"],
        },
    )

    assert response["status"] == "ok"
    assert response["would_trigger"] is True
    assert response["matching_entity_ids"] == ["lock.garage_door_lock"]
    assert response["findings"] == [finding.as_dict() for finding in expected_findings]
    assert registry.added_rules == []


@pytest.mark.asyncio
async def test_preview_rule_proposal_returns_unsupported_reason(hass) -> None:
    proposal_store = ProposalStore(hass)
    await proposal_store.async_append(
        {
            "candidate_id": "bad_lock",
            "candidate": {
                "candidate_id": "bad_lock",
                "title": "Front lock unlocked while home",
                "summary": "Detect unlocked lock with someone present.",
                "pattern": "lock unlocked while home",
                "suggested_type": "security",
                "confidence_hint": 0.8,
                "evidence_paths": ["derived.anyone_home"],
            },
            "notes": "",
            "status": "draft",
        }
    )
    entry = _make_entry(
        proposal_store=proposal_store,
        rule_registry=DummyRuleRegistry(),
    )

    response = await _hga_component._preview_rule_proposal(
        hass,
        entry,
        candidate_id="bad_lock",
    )

    assert response["status"] == "unsupported"
    assert response["reason_code"] == "missing_required_entities"


@pytest.mark.asyncio
async def test_approve_rule_proposal_triggers_immediate_activation(hass) -> None:
    proposal_store = ProposalStore(hass)
    await proposal_store.async_append(
        {
            "candidate_id": "c_lock",
            "candidate": {
                "candidate_id": "c_lock",
                "title": "Garage lock unlocked while home",
                "summary": "Detect lock left unlocked with someone present.",
                "pattern": "lock unlocked while home",
                "suggested_type": "security",
                "confidence_hint": 0.8,
                "evidence_paths": [
                    "entities[entity_id=lock.garage_door_lock].state",
                    "derived.anyone_home",
                ],
            },
            "notes": "",
            "status": "draft",
        }
    )
    sentinel = SimpleNamespace(async_run_now=AsyncMock(return_value=True))
    registry = DummyRuleRegistry()
    entry = _make_entry(
        proposal_store=proposal_store,
        rule_registry=registry,
        sentinel=sentinel,
    )
    response = await _hga_component._approve_rule_proposal(
        entry,
        candidate_id="c_lock",
    )

    assert response["status"] == "ok"
    assert registry.added_rules
    sentinel.async_run_now.assert_awaited_once()


# ---------------------------------------------------------------------------
# covered builtin rule detection — generic camera entity extraction
# ---------------------------------------------------------------------------


def test_covered_builtin_vehicle_generic_camera() -> None:
    """Vehicle candidate with any camera entity in evidence_paths matches static rule."""
    candidate = {
        "candidate_id": "vehicle_near_driveway",
        "title": "Vehicle detected near driveway camera",
        "summary": "A vehicle was detected near the driveway while someone is home.",
        "pattern": "camera.driveway vehicle while home",
        "suggested_type": "security",
        "confidence_hint": 0.7,
        "evidence_paths": [
            "camera_activity[camera_entity_id=camera.driveway].snapshot_summary",
            "derived.anyone_home",
        ],
    }
    result = _hga_component._covered_builtin_rule_for_candidate(candidate)
    assert result is not None
    rule_id, entity_ids = result
    assert rule_id == "vehicle_detected_near_camera_home"
    assert entity_ids == ["camera.driveway"]


def test_covered_builtin_vehicle_no_camera_in_paths_returns_none() -> None:
    """Vehicle candidate with no camera entity in evidence_paths returns None."""
    candidate = {
        "candidate_id": "vehicle_no_camera",
        "title": "Vehicle detected while home",
        "summary": "A vehicle was detected while someone is home.",
        "pattern": "vehicle while home",
        "suggested_type": "security",
        "confidence_hint": 0.7,
        "evidence_paths": [
            "derived.anyone_home",
        ],
    }
    result = _hga_component._covered_builtin_rule_for_candidate(candidate)
    assert result is None


def test_covered_builtin_snapshot_generic_camera() -> None:
    """Camera-snapshot candidate with any camera entity matches static rule."""
    candidate = {
        "candidate_id": "camera_garage_missing_snapshot",
        "title": "Camera garage missing snapshot summary at night while home present",
        "summary": "The garage camera has no snapshot summary recorded during nighttime while someone is home.",
        "pattern": "camera.garage missing snapshot at night while home",
        "suggested_type": "reliability",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "camera_activity[camera_entity_id=camera.garage].snapshot_summary",
            "derived.anyone_home",
        ],
    }
    result = _hga_component._covered_builtin_rule_for_candidate(candidate)
    assert result is not None
    rule_id, entity_ids = result
    assert rule_id == "camera_missing_snapshot_night_home"
    assert entity_ids == ["camera.garage"]


def test_covered_builtin_snapshot_no_camera_in_paths_returns_none() -> None:
    """Camera-snapshot candidate with no camera entity in paths returns None."""
    candidate = {
        "candidate_id": "snapshot_no_camera",
        "title": "Missing snapshot summary at night while home present",
        "summary": "No snapshot summary recorded during nighttime while someone is home.",
        "pattern": "missing snapshot at night while home",
        "suggested_type": "reliability",
        "confidence_hint": 0.8,
        "evidence_paths": [
            "derived.anyone_home",
            "derived.is_night",
        ],
    }
    result = _hga_component._covered_builtin_rule_for_candidate(candidate)
    assert result is None
