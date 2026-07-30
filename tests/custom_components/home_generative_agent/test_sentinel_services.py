# ruff: noqa: S101, FBT001, FBT002, PLR0913
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

    async def async_patch_rule_params(
        self, rule_id: str, params_patch: dict[str, Any]
    ) -> bool:
        _ = (rule_id, params_patch)
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

    def _drop_timestamps(findings: list) -> list:
        return [{k: v for k, v in f.items() if k != "detected_at"} for f in findings]

    assert response["status"] == "ok"
    assert response["would_trigger"] is True
    assert response["matching_entity_ids"] == ["lock.garage_door_lock"]
    assert _drop_timestamps(response["findings"]) == _drop_timestamps(
        [finding.as_dict() for finding in expected_findings]
    )
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


def _prose_motion_proposal_record(entity_id: str) -> dict[str, Any]:
    return {
        "candidate_id": "motion_kitchen_while_away",
        "candidate": {
            "candidate_id": "motion_kitchen_while_away",
            "title": "Unexpected Kitchen Motion While Away",
            "summary": (
                f"Detects motion in the Kitchen area ({entity_id}) when no one is home."
            ),
            "pattern": "state_change",
            "confidence_hint": 0.6,
            "evidence_paths": ["entities[31].state", "derived.anyone_home"],
        },
        "notes": "",
        "status": "draft",
    }


@pytest.mark.asyncio
async def test_approve_rule_proposal_rejects_hallucinated_prose_entity(hass) -> None:
    """
    A prose-derived motion ID that resolves to no entity blocks approval.

    Registering it would create a rule whose evaluator fails closed forever
    while its semantic key suppresses re-proposals — a silent, permanent
    monitoring gap (issue #518 red-team + Codex reviews).
    """
    proposal_store = ProposalStore(hass)
    await proposal_store.async_append(
        _prose_motion_proposal_record("binary_sensor.hallucinated_motion")
    )
    rule_registry = DummyRuleRegistry()
    entry = _make_entry(
        proposal_store=proposal_store,
        rule_registry=rule_registry,
        sentinel=SimpleNamespace(async_run_now=AsyncMock(return_value=True)),
    )
    response = await _hga_component._approve_rule_proposal(
        entry,
        hass=hass,
        candidate_id="motion_kitchen_while_away",
    )

    assert response["status"] == "unsupported"
    assert response["reason_code"] == "entities_unresolved"
    assert response["details"]["unresolved_entity_ids"] == [
        "binary_sensor.hallucinated_motion"
    ]
    assert rule_registry.added_rules == []


@pytest.mark.asyncio
async def test_approve_rule_proposal_accepts_existing_prose_entity(hass) -> None:
    """The exact issue #518 candidate approves when its sensor exists."""
    hass.states.async_set("binary_sensor.xiao_esp32_c5_espectre_motion", "off")
    proposal_store = ProposalStore(hass)
    await proposal_store.async_append(
        _prose_motion_proposal_record("binary_sensor.xiao_esp32_c5_espectre_motion")
    )
    rule_registry = DummyRuleRegistry()
    entry = _make_entry(
        proposal_store=proposal_store,
        rule_registry=rule_registry,
        sentinel=SimpleNamespace(async_run_now=AsyncMock(return_value=True)),
    )
    response = await _hga_component._approve_rule_proposal(
        entry,
        hass=hass,
        candidate_id="motion_kitchen_while_away",
    )

    assert response["status"] == "ok"
    assert rule_registry.added_rules
    assert rule_registry.added_rules[0]["params"]["motion_entity_ids"] == [
        "binary_sensor.xiao_esp32_c5_espectre_motion"
    ]


@pytest.mark.asyncio
async def test_approve_rule_proposal_drops_unresolved_prose_entities(hass) -> None:
    """Mixed resolved/hallucinated prose IDs approve with only the real ones."""
    hass.states.async_set("binary_sensor.kitchen_motion", "off")
    record = _prose_motion_proposal_record("binary_sensor.kitchen_motion")
    record["candidate"]["summary"] = (
        "Detects motion via binary_sensor.kitchen_motion and "
        "binary_sensor.hallucinated_motion when no one is home."
    )
    proposal_store = ProposalStore(hass)
    await proposal_store.async_append(record)
    rule_registry = DummyRuleRegistry()
    entry = _make_entry(
        proposal_store=proposal_store,
        rule_registry=rule_registry,
        sentinel=SimpleNamespace(async_run_now=AsyncMock(return_value=True)),
    )
    response = await _hga_component._approve_rule_proposal(
        entry,
        hass=hass,
        candidate_id="motion_kitchen_while_away",
    )

    assert response["status"] == "ok"
    assert rule_registry.added_rules
    assert rule_registry.added_rules[0]["params"]["motion_entity_ids"] == [
        "binary_sensor.kitchen_motion"
    ]


@pytest.mark.asyncio
async def test_approve_rule_proposal_cover_requires_template_match(hass) -> None:
    """
    A key-based cover with different template semantics does not stand.

    A motion-with-camera-evidence candidate keys as motion but normalizes
    to motion_without_camera_activity; a plain motion rule must not swallow
    it as already-active (issue #518 verification review P1).
    """
    hass.states.async_set("binary_sensor.kitchen_motion", "off")
    record = {
        "candidate_id": "kitchen_motion_no_camera",
        "candidate": {
            "candidate_id": "kitchen_motion_no_camera",
            "title": "Motion without camera activity while away",
            "summary": (
                "Motion on binary_sensor.kitchen_motion with no camera "
                "activity when no one is home."
            ),
            "pattern": "state_change",
            "confidence_hint": 0.6,
            "evidence_paths": [
                "entities[entity_id=binary_sensor.kitchen_motion].state",
                "camera_activity[entity_id=camera.kitchen]",
                "derived.anyone_home",
            ],
        },
        "notes": "",
        "status": "draft",
    }
    proposal_store = ProposalStore(hass)
    await proposal_store.async_append(record)
    rule_registry = DummyRuleRegistry(
        rules=[
            {
                "rule_id": "motion_kitchen_while_away",
                "template_id": "motion_detected_while_away",
                "params": {"motion_entity_ids": ["binary_sensor.kitchen_motion"]},
                "enabled": True,
            }
        ]
    )
    entry = _make_entry(
        proposal_store=proposal_store,
        rule_registry=rule_registry,
        sentinel=SimpleNamespace(async_run_now=AsyncMock(return_value=True)),
    )
    response = await _hga_component._approve_rule_proposal(
        entry,
        hass=hass,
        candidate_id="kitchen_motion_no_camera",
    )

    assert response["status"] == "ok"
    assert rule_registry.added_rules
    assert (
        rule_registry.added_rules[0]["template_id"] == "motion_without_camera_activity"
    )


@pytest.mark.asyncio
async def test_approve_rule_proposal_rechecks_cover_after_dropping_ids(hass) -> None:
    """
    Dropping a hallucinated ID re-checks coverage with the reduced set.

    Without the re-check a second overlapping rule registers under a
    different ID — duplicate findings and notifications (issue #518
    verification review).
    """
    hass.states.async_set("binary_sensor.kitchen_motion", "off")
    record = _prose_motion_proposal_record("binary_sensor.kitchen_motion")
    record["candidate"]["summary"] = (
        "Detects motion via binary_sensor.kitchen_motion and "
        "binary_sensor.hallucinated_motion when no one is home."
    )
    proposal_store = ProposalStore(hass)
    await proposal_store.async_append(record)
    rule_registry = DummyRuleRegistry(
        rules=[
            {
                "rule_id": "existing_kitchen_motion_away",
                "template_id": "motion_detected_while_away",
                "params": {"motion_entity_ids": ["binary_sensor.kitchen_motion"]},
                "enabled": True,
            }
        ]
    )
    entry = _make_entry(
        proposal_store=proposal_store,
        rule_registry=rule_registry,
        sentinel=SimpleNamespace(async_run_now=AsyncMock(return_value=True)),
    )
    response = await _hga_component._approve_rule_proposal(
        entry,
        hass=hass,
        candidate_id="motion_kitchen_while_away",
    )

    assert response["status"] == "covered_by_existing_rule"
    assert response["rule_id"] == "existing_kitchen_motion_away"
    assert response["overlapping_entity_ids"] == ["binary_sensor.kitchen_motion"]
    assert rule_registry.added_rules == []


@pytest.mark.asyncio
async def test_approve_rule_proposal_unsupported_not_reported_covered(hass) -> None:
    """
    An unsupported candidate is never reported covered by a coarse key match.

    A stale-sensor candidate keys exactly like an active away-motion rule,
    but that rule cannot detect the dead sensor the user described — the
    honest answer is unsupported (issue #518 verification round 3).
    """
    record = {
        "candidate_id": "kitchen_motion_stale",
        "candidate": {
            "candidate_id": "kitchen_motion_stale",
            "title": "Kitchen motion sensor stale while away",
            "summary": (
                "binary_sensor.kitchen_motion has not updated for days "
                "when no one is home."
            ),
            "pattern": "state_change",
            "confidence_hint": 0.6,
            "evidence_paths": [
                "entities[entity_id=binary_sensor.kitchen_motion].state",
                "derived.anyone_home",
            ],
        },
        "notes": "",
        "status": "draft",
    }
    proposal_store = ProposalStore(hass)
    await proposal_store.async_append(record)
    rule_registry = DummyRuleRegistry(
        rules=[
            {
                "rule_id": "motion_kitchen_while_away",
                "template_id": "motion_detected_while_away",
                "params": {"motion_entity_ids": ["binary_sensor.kitchen_motion"]},
                "enabled": True,
            }
        ]
    )
    entry = _make_entry(
        proposal_store=proposal_store,
        rule_registry=rule_registry,
        sentinel=SimpleNamespace(async_run_now=AsyncMock(return_value=True)),
    )
    response = await _hga_component._approve_rule_proposal(
        entry,
        hass=hass,
        candidate_id="kitchen_motion_stale",
    )

    assert response["status"] == "unsupported"
    assert rule_registry.added_rules == []


@pytest.mark.asyncio
async def test_approve_rule_proposal_superset_rule_covers_reduced_set(hass) -> None:
    """
    A same-template any-of rule covering a superset of the reduced set counts.

    An existing [kitchen, hall] rule already alerts on kitchen motion, so a
    candidate reduced to [kitchen] must not register a duplicate
    (issue #518 verification round 3).
    """
    hass.states.async_set("binary_sensor.kitchen_motion", "off")
    record = _prose_motion_proposal_record("binary_sensor.kitchen_motion")
    record["candidate"]["summary"] = (
        "Detects motion via binary_sensor.kitchen_motion and "
        "binary_sensor.hallucinated_motion when no one is home."
    )
    proposal_store = ProposalStore(hass)
    await proposal_store.async_append(record)
    rule_registry = DummyRuleRegistry(
        rules=[
            {
                "rule_id": "all_interior_motion_away",
                "template_id": "motion_detected_while_away",
                "params": {
                    "motion_entity_ids": [
                        "binary_sensor.kitchen_motion",
                        "binary_sensor.hall_motion",
                    ]
                },
                "enabled": True,
            }
        ]
    )
    entry = _make_entry(
        proposal_store=proposal_store,
        rule_registry=rule_registry,
        sentinel=SimpleNamespace(async_run_now=AsyncMock(return_value=True)),
    )
    response = await _hga_component._approve_rule_proposal(
        entry,
        hass=hass,
        candidate_id="motion_kitchen_while_away",
    )

    assert response["status"] == "covered_by_existing_rule"
    assert response["rule_id"] == "all_interior_motion_away"
    assert response["overlapping_entity_ids"] == ["binary_sensor.kitchen_motion"]
    assert rule_registry.added_rules == []


@pytest.mark.asyncio
async def test_approve_rule_proposal_superset_covers_fully_resolved_set(hass) -> None:
    """
    The superset check also runs when every candidate ID resolves.

    A kitchen-only candidate against an existing [kitchen, hall] any-of
    rule must not register a duplicate (verification round 4).
    """
    hass.states.async_set("binary_sensor.kitchen_motion", "off")
    record = _prose_motion_proposal_record("binary_sensor.kitchen_motion")
    proposal_store = ProposalStore(hass)
    await proposal_store.async_append(record)
    rule_registry = DummyRuleRegistry(
        rules=[
            {
                "rule_id": "all_interior_motion_away",
                "template_id": "motion_detected_while_away",
                "params": {
                    "motion_entity_ids": [
                        "binary_sensor.kitchen_motion",
                        "binary_sensor.hall_motion",
                    ]
                },
                "enabled": True,
            }
        ]
    )
    entry = _make_entry(
        proposal_store=proposal_store,
        rule_registry=rule_registry,
        sentinel=SimpleNamespace(async_run_now=AsyncMock(return_value=True)),
    )
    response = await _hga_component._approve_rule_proposal(
        entry,
        hass=hass,
        candidate_id="motion_kitchen_while_away",
    )

    assert response["status"] == "covered_by_existing_rule"
    assert response["rule_id"] == "all_interior_motion_away"
    assert rule_registry.added_rules == []


@pytest.mark.asyncio
async def test_approve_rule_proposal_all_of_template_not_superset_covered(hass) -> None:
    """
    All-of templates never cover subsets.

    A [kitchen, hall] alarm-motion rule fires only when BOTH are active,
    so it does not monitor kitchen-only events (verification round 5).
    """
    hass.states.async_set("binary_sensor.kitchen_motion", "off")
    hass.states.async_set("alarm_control_panel.home_alarm", "disarmed")
    record = {
        "candidate_id": "kitchen_motion_alarm_disarmed",
        "candidate": {
            "candidate_id": "kitchen_motion_alarm_disarmed",
            "title": "Kitchen motion at night while alarm disarmed",
            "summary": (
                "Motion on binary_sensor.kitchen_motion at night while the "
                "alarm is disarmed."
            ),
            "pattern": "state_change",
            "confidence_hint": 0.7,
            "evidence_paths": [
                "entities[entity_id=alarm_control_panel.home_alarm].state",
                "entities[entity_id=binary_sensor.kitchen_motion].state",
                "derived.is_night",
            ],
        },
        "notes": "",
        "status": "draft",
    }
    proposal_store = ProposalStore(hass)
    await proposal_store.async_append(record)
    rule_registry = DummyRuleRegistry(
        rules=[
            {
                "rule_id": "all_motion_alarm_disarmed",
                "template_id": "motion_detected_at_night_while_alarm_disarmed",
                "params": {
                    "alarm_entity_id": "alarm_control_panel.home_alarm",
                    "motion_entity_ids": [
                        "binary_sensor.kitchen_motion",
                        "binary_sensor.hall_motion",
                    ],
                    "required_entity_ids": [],
                },
                "enabled": True,
            }
        ]
    )
    entry = _make_entry(
        proposal_store=proposal_store,
        rule_registry=rule_registry,
        sentinel=SimpleNamespace(async_run_now=AsyncMock(return_value=True)),
    )
    response = await _hga_component._approve_rule_proposal(
        entry,
        hass=hass,
        candidate_id="kitchen_motion_alarm_disarmed",
    )

    # The superset shortcut must not fire for the all-of template; whatever
    # the final status, it must not be a superset-based cover by the
    # two-sensor rule.
    assert (
        not (
            response["status"] == "covered_by_existing_rule"
            and response.get("rule_id") == "all_motion_alarm_disarmed"
            and response.get("overlapping_entity_ids")
            == ["binary_sensor.kitchen_motion"]
        )
        or rule_registry.added_rules == []
    )
    # Precise expectation: candidate registers its own rule.
    assert response["status"] == "ok"
    assert rule_registry.added_rules
