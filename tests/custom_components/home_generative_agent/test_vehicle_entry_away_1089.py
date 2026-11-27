# ruff: noqa: S101
"""A car's doors, windows and locks away from home are not house entries (#1089)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant.common import MockConfigEntry

from custom_components.home_generative_agent.sentinel.dynamic_rules import (
    evaluate_dynamic_rules,
)
from custom_components.home_generative_agent.sentinel.rules.camera_entry_unsecured import (
    CameraEntryUnsecuredRule,
)
from custom_components.home_generative_agent.sentinel.rules.open_entry_while_away import (
    OpenEntryWhileAwayRule,
)
from custom_components.home_generative_agent.snapshot.builder import (
    async_build_full_state_snapshot,
)
from custom_components.home_generative_agent.snapshot.schema import (
    CameraActivity,
    FullStateSnapshot,
    SnapshotEntity,
    validate_snapshot,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

NOW = "2026-02-01T00:00:00+00:00"
CAR_DOORS = "binary_sensor.911_carrera_doors_and_lids"
CAR_WINDOW = "binary_sensor.911_carrera_window"
CAR_LOCK = "lock.911_carrera_door_lock"
HOUSE_DOOR = "binary_sensor.front_door"
HOUSE_LOCK = "lock.front_door"
CAMERA = "camera.driveway"


def _entity(
    entity_id: str, state: str, *, device_class: str | None = None, away: bool = False
) -> SnapshotEntity:
    entity: SnapshotEntity = {
        "entity_id": entity_id,
        "domain": entity_id.split(".", 1)[0],
        "state": state,
        "friendly_name": entity_id,
        "area": "Driveway",
        "attributes": {"device_class": device_class} if device_class else {},
        "last_changed": NOW,
        "last_updated": NOW,
    }
    if away:
        entity["device_away_from_home"] = True
    return entity


def _snapshot(
    *,
    away: bool,
    anyone_home: bool = False,
    is_night: bool = True,
    house: bool = False,
    camera: bool = False,
) -> FullStateSnapshot:
    """Car open and unlocked; optionally a house door open and unlocked too."""
    entities = [
        _entity(CAR_DOORS, "on", device_class="opening", away=away),
        _entity(CAR_WINDOW, "on", device_class="window", away=away),
        _entity(CAR_LOCK, "unlocked", away=away),
    ]
    if house:
        entities += [
            _entity(HOUSE_DOOR, "on", device_class="door"),
            _entity(HOUSE_LOCK, "unlocked"),
        ]
    activity: list[CameraActivity] = []
    if camera:
        activity.append(
            {
                "camera_entity_id": CAMERA,
                "area": "Driveway",
                "last_activity": NOW,
                "motion_entities": [],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": [],
                "latest_path": None,
            }
        )
    return validate_snapshot(
        {
            "schema_version": 1,
            "generated_at": NOW,
            "entities": entities,
            "camera_activity": activity,
            "derived": {
                "now": NOW,
                "timezone": "UTC",
                "is_night": is_night,
                "anyone_home": anyone_home,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
        }
    )


def _open_entry(snapshot: FullStateSnapshot) -> list[str]:
    return sorted(
        e
        for f in OpenEntryWhileAwayRule().evaluate(snapshot)
        for e in f.triggering_entities
    )


def _camera_unsecured(
    snapshot: FullStateSnapshot, links: dict[str, list[str]] | None = None
) -> list[list[str]]:
    return [
        f.evidence["unsecured_entities"]
        for f in CameraEntryUnsecuredRule(camera_entry_links=links).evaluate(snapshot)
    ]


def test_open_entry_while_away_skips_an_away_car() -> None:
    """The box's case: the car's doors-and-lids sensor while it is out."""
    assert _open_entry(_snapshot(away=True)) == []
    assert _open_entry(_snapshot(away=True, house=True)) == [HOUSE_DOOR]


def test_open_entry_while_away_still_alerts_on_a_car_at_home() -> None:
    assert _open_entry(_snapshot(away=False)) == [CAR_DOORS, CAR_WINDOW]


def test_camera_entry_unsecured_skips_an_away_car() -> None:
    assert _camera_unsecured(_snapshot(away=True, camera=True)) == []
    assert _camera_unsecured(_snapshot(away=True, camera=True, house=True)) == [
        [HOUSE_DOOR, HOUSE_LOCK]
    ]


def test_camera_entry_unsecured_skips_an_away_car_linked_from_another_area() -> None:
    snapshot = _snapshot(away=True, camera=True)
    for entity in snapshot["entities"]:
        entity["area"] = "Garage"
    assert _camera_unsecured(snapshot, {CAMERA: [CAR_DOORS, CAR_LOCK]}) == []


def test_camera_entry_unsecured_still_alerts_on_a_car_at_home() -> None:
    assert _camera_unsecured(_snapshot(away=False, camera=True)) == [
        [CAR_DOORS, CAR_WINDOW, CAR_LOCK]
    ]


def _rule(template_id: str, **params: Any) -> dict[str, Any]:
    return {
        "rule_id": f"rule_{template_id}",
        "template_id": template_id,
        "params": params,
        "severity": "high",
    }


_ENTRIES = [CAR_DOORS, CAR_WINDOW, HOUSE_DOOR]

# (rule, anyone_home): every lock and entry template, aimed at the car and at
# the house so each test shows the car is the only thing skipped.
_TEMPLATE_CASES = [
    pytest.param(
        [
            _rule("unlocked_lock_when_home", lock_entity_id=CAR_LOCK),
            _rule("unlocked_lock_when_home", lock_entity_id=HOUSE_LOCK),
        ],
        True,
        id="unlocked_lock_when_home",
    ),
    pytest.param(
        [
            _rule("unlocked_lock_while_away", lock_entity_id=CAR_LOCK),
            _rule("unlocked_lock_while_away", lock_entity_id=HOUSE_LOCK),
        ],
        False,
        id="unlocked_lock_while_away",
    ),
    *(
        pytest.param([_rule(template, entry_entity_ids=_ENTRIES)], home, id=template)
        for template, home in (
            ("open_entry_when_home", True),
            ("open_entry_while_away", False),
            ("open_entry_at_night_when_home", True),
            ("open_entry_at_night_while_away", False),
            ("open_entry_at_night", True),
        )
    ),
    pytest.param(
        [_rule("multiple_entries_open_count", entry_entity_ids=_ENTRIES, min_count=1)],
        True,
        id="multiple_entries_open_count",
    ),
]


def _alarm_disarmed(snapshot: FullStateSnapshot) -> FullStateSnapshot:
    alarm = _entity("alarm_control_panel.home", "disarmed")
    snapshot["entities"].append(alarm)
    return snapshot


def _triggered(snapshot: FullStateSnapshot, rules: list[dict[str, Any]]) -> list[str]:
    return sorted(
        e
        for f in evaluate_dynamic_rules(snapshot, rules)
        for e in f.triggering_entities
        if not e.startswith("alarm_control_panel.")
    )


@pytest.mark.parametrize(("rules", "anyone_home"), _TEMPLATE_CASES)
def test_lock_and_entry_templates_skip_an_away_car(
    rules: list[dict[str, Any]],
    anyone_home: bool,  # noqa: FBT001
) -> None:
    """A rule approved for the car's own lock or doors skips it while it is out."""
    away = _triggered(_snapshot(away=True, anyone_home=anyone_home, house=True), rules)
    home = _triggered(_snapshot(away=False, anyone_home=anyone_home, house=True), rules)
    assert not {CAR_DOORS, CAR_WINDOW, CAR_LOCK} & set(away)
    assert {HOUSE_DOOR, HOUSE_LOCK} & set(away)
    assert {CAR_DOORS, CAR_WINDOW, CAR_LOCK} & set(home)


def test_alarm_disarmed_open_entry_skips_an_away_car() -> None:
    rules = [
        _rule(
            "alarm_disarmed_open_entry",
            alarm_entity_id="alarm_control_panel.home",
            entry_entity_ids=_ENTRIES,
        )
    ]
    away = _alarm_disarmed(_snapshot(away=True, house=True))
    home = _alarm_disarmed(_snapshot(away=False, house=True))
    assert _triggered(away, rules) == [HOUSE_DOOR]
    assert _triggered(home, rules) == [CAR_DOORS, CAR_WINDOW, HOUSE_DOOR]


def test_open_any_window_at_night_while_away_skips_an_away_car() -> None:
    rules = [_rule("open_any_window_at_night_while_away")]
    assert _triggered(_snapshot(away=True), rules) == []
    assert _triggered(_snapshot(away=False), rules) == [CAR_WINDOW]


def test_multiple_entries_open_count_does_not_count_an_away_car() -> None:
    """Two car openings and one house door are one open entry, not three."""
    rules = [
        _rule("multiple_entries_open_count", entry_entity_ids=_ENTRIES, min_count=2)
    ]
    assert _triggered(_snapshot(away=True, house=True), rules) == []
    assert _triggered(_snapshot(away=False, house=True), rules) == _ENTRIES


async def test_the_snapshot_marks_a_cars_door_sensor_away(hass: HomeAssistant) -> None:
    """The car's doors-and-lids sensor carries the flag, not just its lock."""
    owner = MockConfigEntry(domain="porscheconnect")
    owner.add_to_hass(hass)
    device = dr.async_get(hass).async_get_or_create(
        config_entry_id=owner.entry_id, identifiers={("porscheconnect", "vin")}
    )
    registry = er.async_get(hass)
    for domain, object_id in (
        ("binary_sensor", "911_carrera_doors_and_lids"),
        ("device_tracker", "911_carrera"),
    ):
        registry.async_get_or_create(
            domain,
            "porscheconnect",
            f"vin-{object_id}",
            suggested_object_id=object_id,
            device_id=device.id,
        )
    hass.states.async_set(CAR_DOORS, "on", {"device_class": "opening"})
    hass.states.async_set(
        "device_tracker.911_carrera", "not_home", {"source_type": "gps", "in_zones": []}
    )
    snapshot = await async_build_full_state_snapshot(hass)
    doors = next(e for e in snapshot["entities"] if e["entity_id"] == CAR_DOORS)
    assert doors.get("device_away_from_home") is True
    snapshot["derived"]["anyone_home"] = False
    assert _open_entry(snapshot) == []


@pytest.mark.parametrize(
    ("entity_id", "target_state"),
    [(CAR_LOCK, "unlocked"), (CAR_DOORS, "on"), (CAR_WINDOW, "on")],
)
def test_lock_and_entry_duration_rules_skip_an_away_car(
    entity_id: str, target_state: str
) -> None:
    """'Lock unlocked too long' / 'entry open too long' are house-entry rules."""
    rules = [
        _rule(
            "entity_state_duration",
            entity_id=entity_id,
            target_state=target_state,
            threshold_hours=1,
        )
    ]
    snapshot_away = _snapshot(away=True)
    snapshot_home = _snapshot(away=False)
    for snapshot in (snapshot_away, snapshot_home):
        snapshot["derived"]["now"] = "2026-02-01T03:00:00+00:00"
    assert _triggered(snapshot_away, rules) == []
    assert _triggered(snapshot_home, rules) == [entity_id]


def test_a_duration_rule_on_another_car_sensor_still_fires_while_away() -> None:
    """Only locks and door/window sensors are entries; a car's other sensors are not."""
    snapshot = _snapshot(away=True)
    snapshot["derived"]["now"] = "2026-02-01T03:00:00+00:00"
    snapshot["entities"].append(
        _entity("binary_sensor.911_carrera_charging", "on", away=True)
    )
    rules = [
        _rule(
            "entity_state_duration",
            entity_id="binary_sensor.911_carrera_charging",
            target_state="on",
            threshold_hours=1,
        )
    ]
    assert _triggered(snapshot, rules) == ["binary_sensor.911_carrera_charging"]


def test_an_away_car_does_not_stand_in_as_camera_activity() -> None:
    """A camera with no activity of its own must not borrow an away car's updates."""
    snapshot = _snapshot(away=True, camera=True)
    snapshot["camera_activity"][0]["last_activity"] = None
    snapshot["entities"].append(_entity(HOUSE_LOCK, "unlocked"))
    for entity in snapshot["entities"]:
        if entity["entity_id"] == HOUSE_LOCK:
            entity["last_changed"] = "2026-01-31T00:00:00+00:00"
    assert _camera_unsecured(snapshot) == []
    for entity in snapshot["entities"]:
        entity.pop("device_away_from_home", None)
    assert _camera_unsecured(snapshot) == [
        [CAR_DOORS, CAR_WINDOW, CAR_LOCK, HOUSE_LOCK]
    ]


def test_an_away_car_door_does_not_change_the_open_count_alert_id() -> None:
    """A car door opening and closing on a drive must not re-mint the house alert."""
    second_house_door = "binary_sensor.back_door"
    rules = [
        _rule(
            "multiple_entries_open_count",
            entry_entity_ids=[*_ENTRIES, second_house_door],
            min_count=2,
        )
    ]
    ids = []
    for car_state in ("on", "off"):
        snapshot = _snapshot(away=True, house=True)
        snapshot["entities"].append(
            _entity(second_house_door, "on", device_class="door")
        )
        for entity in snapshot["entities"]:
            if entity["entity_id"] in {CAR_DOORS, CAR_WINDOW}:
                entity["state"] = car_state
        findings = evaluate_dynamic_rules(snapshot, rules)
        assert len(findings) == 1
        ids.append(findings[0].anomaly_id)
    assert ids[0] == ids[1]


@pytest.mark.parametrize(
    ("entity_id", "device_class"),
    [
        ("binary_sensor.911_carrera_door_lock_state", "lock"),
        ("binary_sensor.911_carrera_rear_door", None),
    ],
)
def test_a_duration_rule_on_a_car_lock_sensor_or_unclassed_door_skips(
    entity_id: str, device_class: str | None
) -> None:
    """'Entry open too long' picks entries by id, not class: both shapes skip."""
    rules = [
        _rule(
            "entity_state_duration",
            entity_id=entity_id,
            target_state="on",
            threshold_hours=1,
        )
    ]
    results = []
    for away in (True, False):
        snapshot = _snapshot(away=away)
        snapshot["derived"]["now"] = "2026-02-01T03:00:00+00:00"
        snapshot["entities"].append(
            _entity(entity_id, "on", device_class=device_class, away=away)
        )
        results.append(_triggered(snapshot, rules))
    assert results == [[], [entity_id]]


def test_the_car_leaving_does_not_change_a_house_only_open_count_alert_id() -> None:
    """Two house windows open all afternoon: the car leaving is not a new alert."""
    back_door = "binary_sensor.back_door"
    rules = [
        _rule(
            "multiple_entries_open_count",
            entry_entity_ids=[*_ENTRIES, back_door],
            min_count=2,
        )
    ]
    ids = []
    for away in (False, True):
        snapshot = _snapshot(away=away, house=True)
        snapshot["entities"].append(_entity(back_door, "on", device_class="door"))
        for entity in snapshot["entities"]:
            if entity["entity_id"] in {CAR_DOORS, CAR_WINDOW}:
                entity["state"] = "off"
        findings = evaluate_dynamic_rules(snapshot, rules)
        assert len(findings) == 1
        ids.append(findings[0].anomaly_id)
    assert ids[0] == ids[1]


def test_an_unhashable_device_class_does_not_break_duration_rules() -> None:
    """A customized list-valued device_class must not raise in the entry check."""
    snapshot = _snapshot(away=True)
    snapshot["derived"]["now"] = "2026-02-01T03:00:00+00:00"
    odd = _entity("binary_sensor.911_carrera_odd", "on", away=True)
    odd["attributes"]["device_class"] = ["door"]
    snapshot["entities"].append(odd)
    rules = [
        _rule(
            "entity_state_duration",
            entity_id="binary_sensor.911_carrera_odd",
            target_state="on",
            threshold_hours=1,
        )
    ]
    assert _triggered(snapshot, rules) == ["binary_sensor.911_carrera_odd"]
