# ruff: noqa: S101
"""Tests for per-rule entity exclusions and appliance threshold options (#462)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest

from custom_components.home_generative_agent.const import (
    CONF_SENTINEL_APPLIANCE_DURATION_MIN,
    CONF_SENTINEL_APPLIANCE_POWER_THRESHOLD_W,
    CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS,
)
from custom_components.home_generative_agent.sentinel.engine import (
    SentinelEngine,
    _entity_excluded,
    _parse_rule_entity_exclusions,
)
from custom_components.home_generative_agent.sentinel.rules.appliance_power_duration import (
    AppliancePowerDurationRule,
)
from custom_components.home_generative_agent.sentinel.suppression import (
    SuppressionManager,
    SuppressionState,
)
from custom_components.home_generative_agent.snapshot.schema import (
    FullStateSnapshot,
    validate_snapshot,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.audit.store import AuditStore
    from custom_components.home_generative_agent.sentinel.notifier import (
        SentinelNotifier,
    )


class DummySuppression(SuppressionManager):
    """Suppression manager stub."""

    def __init__(self) -> None:  # type: ignore[override]
        self._state = SuppressionState()
        self._read_only = False

    @property
    def state(self) -> SuppressionState:  # type: ignore[override]
        return self._state

    @property
    def is_read_only(self) -> bool:  # type: ignore[override]
        return self._read_only

    async def async_save(self) -> None:  # type: ignore[override]
        return None


class DummyNotifier:
    """Notification dispatcher stub."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def async_notify(self, finding, snapshot, explanation) -> None:  # type: ignore[no-untyped-def]
        self.calls.append({"finding": finding, "snapshot": snapshot})


class DummyAudit:
    """Audit store stub."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def async_append_finding(  # type: ignore[no-untyped-def]
        self, snapshot, finding, explanation, **kwargs: Any
    ) -> None:
        self.calls.append({"finding": finding, "snapshot": snapshot})


def _open_door_snapshot() -> FullStateSnapshot:
    """Snapshot with an open front door while everyone is away."""
    return validate_snapshot(
        {
            "schema_version": 1,
            "generated_at": "2025-01-01T00:00:00+00:00",
            "entities": [
                {
                    "entity_id": "binary_sensor.front_door",
                    "domain": "binary_sensor",
                    "state": "on",
                    "friendly_name": "Front Door",
                    "area": "Front",
                    "attributes": {"device_class": "door"},
                    "last_changed": "2025-01-01T00:00:00+00:00",
                    "last_updated": "2025-01-01T00:00:00+00:00",
                }
            ],
            "camera_activity": [],
            "derived": {
                "now": "2025-01-01T00:00:00+00:00",
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": False,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
        }
    )


def _make_engine(options: dict[str, object]) -> SentinelEngine:
    return SentinelEngine(
        hass=cast("HomeAssistant", object()),
        options={
            "sentinel_cooldown_minutes": 0,
            "sentinel_entity_cooldown_minutes": 0,
            "sentinel_interval_seconds": 60,
            "explain_enabled": False,
            **options,
        },
        suppression=DummySuppression(),
        notifier=cast("SentinelNotifier", DummyNotifier()),
        audit_store=cast("AuditStore", DummyAudit()),
        explainer=None,
    )


def _patch_snapshot(
    monkeypatch: pytest.MonkeyPatch, snapshot: FullStateSnapshot
) -> None:
    async def _fake_build(_hass: HomeAssistant) -> FullStateSnapshot:
        return snapshot

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine.async_build_full_state_snapshot",
        _fake_build,
    )


# --------------------------------------------------------------------------- #
# _parse_rule_entity_exclusions
# --------------------------------------------------------------------------- #


def test_parse_exclusions_normalizes_valid_map() -> None:
    """Valid map is normalized to frozensets, whitespace stripped."""
    parsed = _parse_rule_entity_exclusions(
        {
            "appliance_power_duration": ["sensor.ac_power", " sensor.other "],
            "*": ["sensor.test_bench"],
        }
    )
    assert parsed == {
        "appliance_power_duration": frozenset({"sensor.ac_power", "sensor.other"}),
        "*": frozenset({"sensor.test_bench"}),
    }


@pytest.mark.parametrize(
    "value",
    [
        None,
        "not a dict",
        ["sensor.ac_power"],
        42,
    ],
)
def test_parse_exclusions_non_dict_returns_empty(value: object) -> None:
    """Non-dict option values yield no exclusions rather than raising."""
    assert _parse_rule_entity_exclusions(value) == {}


def test_parse_exclusions_drops_malformed_entries() -> None:
    """Malformed keys/values are dropped; valid entries survive."""
    parsed = _parse_rule_entity_exclusions(
        {
            "": ["sensor.ac_power"],  # empty rule type
            42: ["sensor.ac_power"],  # non-string rule type
            "rule_a": "sensor.not_a_list",  # non-list value
            "rule_b": [42, "", "  ", "sensor.kept"],  # mixed list
            "rule_c": ["", 1],  # nothing valid left
        }
    )
    assert parsed == {"rule_b": frozenset({"sensor.kept"})}


def test_parse_exclusions_keeps_glob_patterns() -> None:
    """Glob patterns are ordinary entries and survive normalization."""
    parsed = _parse_rule_entity_exclusions(
        {"camera_entry_unsecured": ["camera.map_*", "camera.?_screenshot"]}
    )
    assert parsed == {
        "camera_entry_unsecured": frozenset({"camera.map_*", "camera.?_screenshot"})
    }


# --------------------------------------------------------------------------- #
# _entity_excluded matching
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("entity_id", "excluded", "expected"),
    [
        ("camera.map_home", frozenset({"camera.map_home"}), True),
        ("camera.map_home", frozenset({"camera.map_*"}), True),
        ("camera.map_alice_google", frozenset({"camera.map_*"}), True),
        ("camera.driveway", frozenset({"camera.map_*"}), False),
        ("camera.driveway", frozenset[str](), False),
        # "*" in a pattern matches across dots — whole-domain exclusion works.
        ("person.alice", frozenset({"person.*"}), True),
        # Matching is case-sensitive like entity IDs themselves.
        ("camera.Map_home", frozenset({"camera.map_*"}), False),
    ],
)
def test_entity_excluded_matching(
    entity_id: str,
    excluded: frozenset[str],
    expected: bool,  # noqa: FBT001
) -> None:
    """Exact IDs and fnmatch-style globs both match; others do not."""
    assert _entity_excluded(entity_id, excluded) is expected


# --------------------------------------------------------------------------- #
# Engine exclusion filtering
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_engine_drops_finding_for_excluded_entity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finding whose triggering entity is excluded for its type is dropped."""
    _patch_snapshot(monkeypatch, _open_door_snapshot())
    engine = _make_engine(
        {
            CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS: {
                "open_entry_while_away": ["binary_sensor.front_door"]
            }
        }
    )

    await engine._run_once()

    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    audit_store = cast("DummyAudit", cast("Any", engine)._audit_store)
    assert notifier.calls == []
    assert audit_store.calls == []


@pytest.mark.asyncio
async def test_engine_wildcard_exclusion_applies_to_all_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The "*" key excludes the entity from every rule."""
    _patch_snapshot(monkeypatch, _open_door_snapshot())
    engine = _make_engine(
        {CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS: {"*": ["binary_sensor.front_door"]}}
    )

    await engine._run_once()

    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    assert notifier.calls == []


@pytest.mark.asyncio
async def test_engine_drops_finding_for_glob_excluded_entity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A glob exclusion entry drops findings for every matching entity."""
    _patch_snapshot(monkeypatch, _open_door_snapshot())
    engine = _make_engine(
        {
            CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS: {
                "open_entry_while_away": ["binary_sensor.front_*"]
            }
        }
    )

    await engine._run_once()

    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    assert notifier.calls == []


@pytest.mark.asyncio
async def test_engine_keeps_findings_for_other_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclusions for unrelated entities or types leave findings untouched."""
    _patch_snapshot(monkeypatch, _open_door_snapshot())
    engine = _make_engine(
        {
            CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS: {
                "open_entry_while_away": ["binary_sensor.back_door"],
                "appliance_power_duration": ["binary_sensor.front_door"],
            }
        }
    )

    await engine._run_once()

    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    assert notifier.calls


@pytest.mark.asyncio
async def test_engine_no_exclusions_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no exclusion option at all, behavior is unchanged."""
    _patch_snapshot(monkeypatch, _open_door_snapshot())
    engine = _make_engine({})

    await engine._run_once()

    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    assert notifier.calls


# --------------------------------------------------------------------------- #
# Appliance threshold wiring
# --------------------------------------------------------------------------- #


def test_engine_wires_appliance_thresholds_from_options() -> None:
    """Configured thresholds reach the AppliancePowerDurationRule instance."""
    engine = _make_engine(
        {
            CONF_SENTINEL_APPLIANCE_POWER_THRESHOLD_W: 500,
            CONF_SENTINEL_APPLIANCE_DURATION_MIN: 180,
        }
    )
    rule = next(
        r
        for r in cast("Any", engine)._rules
        if isinstance(r, AppliancePowerDurationRule)
    )
    assert rule._power_threshold_w == 500.0
    assert rule._duration_min == 180


def test_engine_appliance_thresholds_default_when_unset() -> None:
    """Without options the rule keeps its recommended defaults."""
    engine = _make_engine({})
    rule = next(
        r
        for r in cast("Any", engine)._rules
        if isinstance(r, AppliancePowerDurationRule)
    )
    assert rule._power_threshold_w == 100.0
    assert rule._duration_min == 60


# --------------------------------------------------------------------------- #
# Event-driven trigger exclusions (#481)
# --------------------------------------------------------------------------- #


def _state_change_event(entity_id: str) -> Any:
    """Build a minimal state-change event for _on_state_changed."""
    state = MagicMock()
    state.entity_id = entity_id
    state.attributes = {}
    event = MagicMock()
    event.data = {"entity_id": entity_id, "new_state": state}
    return event


def test_trigger_excluded_entity_not_enqueued() -> None:
    """An entity excluded for its mapped anomaly type never enqueues."""
    engine = _make_engine(
        {
            CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS: {
                "camera_entry_unsecured": ["camera.map_home"]
            }
        }
    )

    engine._on_state_changed(_state_change_event("camera.map_home"))
    assert engine._trigger_scheduler.queue_depth == 0

    engine._on_state_changed(_state_change_event("camera.driveway"))
    assert engine._trigger_scheduler.queue_depth == 1


def test_trigger_glob_excluded_entity_not_enqueued() -> None:
    """Glob exclusion entries suppress triggers for all matching entities."""
    engine = _make_engine(
        {
            CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS: {
                "camera_entry_unsecured": ["camera.map_*"]
            }
        }
    )

    engine._on_state_changed(_state_change_event("camera.map_alice_google"))
    engine._on_state_changed(_state_change_event("camera.map_bob_mapbox"))
    assert engine._trigger_scheduler.queue_depth == 0

    engine._on_state_changed(_state_change_event("camera.driveway"))
    assert engine._trigger_scheduler.queue_depth == 1


def test_trigger_wildcard_exclusion_suppresses_all_types() -> None:
    """The "*" type key suppresses event triggers for the listed entities."""
    engine = _make_engine({CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS: {"*": ["person.bob"]}})

    engine._on_state_changed(_state_change_event("person.bob"))
    assert engine._trigger_scheduler.queue_depth == 0

    engine._on_state_changed(_state_change_event("person.alice"))
    assert engine._trigger_scheduler.queue_depth == 1


def test_trigger_exclusion_for_other_type_still_enqueues() -> None:
    """An exclusion for an unrelated anomaly type does not suppress triggers."""
    engine = _make_engine(
        {
            CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS: {
                "appliance_power_duration": ["camera.map_home"]
            }
        }
    )

    engine._on_state_changed(_state_change_event("camera.map_home"))
    assert engine._trigger_scheduler.queue_depth == 1
