# ruff: noqa: S101
"""Tests for discovery snapshot reducer."""

from __future__ import annotations

import json
from typing import Any

from custom_components.home_generative_agent.snapshot.discovery_reducer import (
    _MAX_BASELINE_ENTITIES,
    _MAX_ENTITIES,
    _MAX_RECOGNIZED_PEOPLE,
    _MAX_SUMMARY_CHARS,
    _SECOND_PASS_SUMMARY_CHARS,
    _TOKEN_BUDGET_CHARS,
    _apply_budget_trim,
    _truncate_iso,
    reduce_snapshot_for_discovery,
)
from custom_components.home_generative_agent.snapshot.schema import (
    FullStateSnapshot,
    validate_snapshot,
)


def _make_snapshot(**overrides: Any) -> FullStateSnapshot:
    """Build a minimal valid snapshot with optional overrides."""
    base: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": "2025-01-01T00:00:00+00:00",
        "entities": [],
        "camera_activity": [],
        "derived": {
            "now": "2025-01-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    }
    base.update(overrides)
    return validate_snapshot(base)


def _make_entity(  # noqa: PLR0913
    entity_id: str,
    domain: str,
    state: str,
    area: str | None = None,
    device_class: str | None = None,
    last_changed: str | None = None,
) -> dict[str, Any]:
    """Build a minimal entity dict."""
    attrs: dict[str, str] = {}
    if device_class:
        attrs["device_class"] = device_class
    ts = last_changed or "2025-01-01T00:00:00+00:00"
    return {
        "entity_id": entity_id,
        "domain": domain,
        "state": state,
        "friendly_name": entity_id,
        "area": area,
        "attributes": attrs,
        "last_changed": ts,
        "last_updated": ts,
    }


def _make_camera(
    camera_id: str,
    area: str | None = None,
    summary: str | None = None,
    people: list[str] | None = None,
    last_activity: str | None = None,
) -> dict[str, Any]:
    """Build a minimal camera activity dict."""
    return {
        "camera_entity_id": camera_id,
        "area": area,
        "last_activity": last_activity or "2025-01-01T00:00:00+00:00",
        "motion_entities": [],
        "vmd_entities": [],
        "snapshot_summary": summary,
        "recognized_people": people or [],
        "latest_path": None,
    }


# --- Domain / device_class filtering ---


def test_reduce_snapshot_filters_entities() -> None:
    snapshot = _make_snapshot(
        entities=[
            _make_entity(
                "light.kitchen",
                "light",
                "on",
                area="Kitchen",
            ),
            _make_entity(
                "binary_sensor.front_door",
                "binary_sensor",
                "on",
                area="Front",
                device_class="door",
            ),
            _make_entity(
                "sensor.washer_power",
                "sensor",
                "250",
                area="Laundry",
                device_class="power",
            ),
        ],
        camera_activity=[
            _make_camera(
                "camera.front",
                area="Front",
                summary="Person at door",
                people=["Alex"],
            ),
        ],
    )

    reduced = reduce_snapshot_for_discovery(snapshot)

    # light.kitchen should be filtered out; 2 entities remain
    assert len(reduced["entities"]) == 2
    all_ids: list[str] = []
    for group in reduced["entities"]:
        all_ids.extend(group["entity_ids"])
    assert "binary_sensor.front_door" in all_ids
    assert "sensor.washer_power" in all_ids
    assert "light.kitchen" not in all_ids
    # Camera preserved
    assert reduced["camera_activity"][0]["camera_entity_id"] == "camera.front"


def test_sensor_without_allowed_device_class_excluded() -> None:
    snapshot = _make_snapshot(
        entities=[
            _make_entity(
                "sensor.temperature",
                "sensor",
                "22.5",
                device_class="temperature",
            ),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert len(reduced["entities"]) == 0


def test_binary_sensor_without_allowed_device_class_excluded() -> None:
    snapshot = _make_snapshot(
        entities=[
            _make_entity(
                "binary_sensor.smoke",
                "binary_sensor",
                "off",
                device_class="smoke",
            ),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert len(reduced["entities"]) == 0


# --- Grouping ---


def test_entities_with_same_domain_area_state_are_grouped() -> None:
    snapshot = _make_snapshot(
        entities=[
            _make_entity(
                "binary_sensor.kitchen_window_1",
                "binary_sensor",
                "off",
                area="Kitchen",
                device_class="window",
            ),
            _make_entity(
                "binary_sensor.kitchen_window_2",
                "binary_sensor",
                "off",
                area="Kitchen",
                device_class="window",
            ),
            _make_entity(
                "binary_sensor.kitchen_window_3",
                "binary_sensor",
                "off",
                area="Kitchen",
                device_class="window",
            ),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert len(reduced["entities"]) == 1
    group = reduced["entities"][0]
    assert len(group["entity_ids"]) == 3
    assert "binary_sensor.kitchen_window_1" in group["entity_ids"]
    assert group["state"] == "off"
    assert group["area"] == "Kitchen"
    assert group["device_class"] == "window"


def test_entities_with_different_states_not_grouped() -> None:
    snapshot = _make_snapshot(
        entities=[
            _make_entity(
                "binary_sensor.kitchen_window",
                "binary_sensor",
                "off",
                area="Kitchen",
                device_class="window",
            ),
            _make_entity(
                "binary_sensor.kitchen_door",
                "binary_sensor",
                "on",
                area="Kitchen",
                device_class="door",
            ),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert len(reduced["entities"]) == 2


def test_entities_with_different_areas_not_grouped() -> None:
    snapshot = _make_snapshot(
        entities=[
            _make_entity(
                "binary_sensor.kitchen_window",
                "binary_sensor",
                "off",
                area="Kitchen",
                device_class="window",
            ),
            _make_entity(
                "binary_sensor.bedroom_window",
                "binary_sensor",
                "off",
                area="Bedroom",
                device_class="window",
            ),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert len(reduced["entities"]) == 2


# --- Timestamp truncation ---


def test_truncate_iso_to_minute() -> None:
    assert _truncate_iso("2026-02-22T05:56:11.988456+00:00") == "2026-02-22T05:56"
    assert _truncate_iso("2026-02-22T05:56:11") == "2026-02-22T05:56"
    assert _truncate_iso(None) is None
    assert _truncate_iso("") is None


def test_timestamps_truncated_in_output() -> None:
    snapshot = _make_snapshot(
        entities=[
            _make_entity(
                "lock.front_door",
                "lock",
                "locked",
                area="Front",
                last_changed="2025-01-01T12:34:56.789+00:00",
            ),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    group = reduced["entities"][0]
    assert group["last_changed"] == "2025-01-01T12:34"


def test_derived_timestamps_truncated() -> None:
    snapshot = _make_snapshot()
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert reduced["derived"]["now"] == "2025-01-01T00:00"


# --- schema_version and generated_at removed from output ---


def test_no_schema_version_or_generated_at_in_output() -> None:
    snapshot = _make_snapshot()
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert "schema_version" not in reduced
    assert "generated_at" not in reduced


# --- Camera summary truncation ---


def test_camera_summary_truncated() -> None:
    long_summary = "A" * 300
    snapshot = _make_snapshot(
        camera_activity=[
            _make_camera("camera.front", summary=long_summary),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    cam = reduced["camera_activity"][0]
    # +3 for "..."
    assert len(cam["snapshot_summary"]) == _MAX_SUMMARY_CHARS + 3
    assert cam["snapshot_summary"].endswith("...")


def test_camera_empty_fields_omitted() -> None:
    snapshot = _make_snapshot(
        camera_activity=[
            _make_camera(
                "camera.front",
                summary=None,
                people=[],
            ),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    cam = reduced["camera_activity"][0]
    assert "snapshot_summary" not in cam
    assert "recognized_people" not in cam


# --- Anomaly prioritization ---


def test_anomalous_states_prioritized_over_normal() -> None:
    """When capped, entities with anomalous states survive."""
    normal_entities = [
        _make_entity(
            f"binary_sensor.window_{i}",
            "binary_sensor",
            "off",
            area="Kitchen",
            device_class="window",
        )
        for i in range(_MAX_ENTITIES + 5)
    ]
    anomalous = _make_entity(
        "lock.front_door",
        "lock",
        "unlocked",
        area="Front",
    )
    entities = [*normal_entities, anomalous]

    snapshot = _make_snapshot(entities=entities)
    reduced = reduce_snapshot_for_discovery(snapshot)

    all_ids: list[str] = []
    for group in reduced["entities"]:
        all_ids.extend(group["entity_ids"])
    assert "lock.front_door" in all_ids


# --- Domain field dropped ---


def test_domain_not_in_grouped_output() -> None:
    snapshot = _make_snapshot(
        entities=[
            _make_entity(
                "lock.front_door",
                "lock",
                "locked",
                area="Front",
            ),
        ],
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    group = reduced["entities"][0]
    assert "domain" not in group


# --- timezone dropped from derived ---


def test_timezone_absent_from_derived() -> None:
    snapshot = _make_snapshot()
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert "timezone" not in reduced["derived"]


# --- baseline_ready_entities trimming ---


def test_baseline_ready_entities_intersected_with_filtered() -> None:
    """Only entity IDs that survived domain filtering appear in baseline list."""
    snapshot = _make_snapshot(
        entities=[
            _make_entity("lock.front_door", "lock", "locked", area="Front"),
            _make_entity("light.kitchen", "light", "on", area="Kitchen"),
        ],
    )
    # baseline_ready_entities is injected after schema validation in production.
    snapshot["derived"]["baseline_ready_entities"] = [  # type: ignore[typeddict-unknown-key]
        "lock.front_door",
        "light.kitchen",
    ]
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert reduced["derived"]["baseline_ready_entities"] == ["lock.front_door"]


def test_baseline_ready_entities_capped() -> None:
    """baseline_ready_entities is capped at _MAX_BASELINE_ENTITIES."""
    lock_entities = [
        _make_entity(f"lock.door_{i}", "lock", "locked") for i in range(60)
    ]
    snapshot = _make_snapshot(entities=lock_entities)
    # Inject baseline list that covers all 60 locks (all survive domain filter).
    snapshot["derived"]["baseline_ready_entities"] = [  # type: ignore[typeddict-unknown-key]
        f"lock.door_{i}" for i in range(60)
    ]
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert len(reduced["derived"]["baseline_ready_entities"]) <= _MAX_BASELINE_ENTITIES


# --- Camera activity cap ---


def test_camera_activity_capped_at_max() -> None:
    cameras = [_make_camera(f"camera.cam_{i}") for i in range(30)]
    snapshot = _make_snapshot(camera_activity=cameras)
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert len(reduced["camera_activity"]) <= 20


# --- Token budget / second-pass trim ---


def _make_large_snapshot() -> Any:
    """Build a synthetic large snapshot that exercises the budget gate."""
    entities = [
        _make_entity(
            f"binary_sensor.window_{i}",
            "binary_sensor",
            "off",
            area=f"Room{i % 10}",
            device_class="window",
            last_changed="2025-01-01T00:00:00+00:00",
        )
        for i in range(_MAX_ENTITIES)
    ]
    cameras = [
        _make_camera(
            f"camera.cam_{i}",
            area=f"Room{i}",
            summary="X" * _MAX_SUMMARY_CHARS,
        )
        for i in range(20)
    ]
    snapshot = _make_snapshot(entities=entities, camera_activity=cameras)
    # Inject fields added post-validation in production.
    snapshot["derived"]["baseline_ready_entities"] = [  # type: ignore[typeddict-unknown-key]
        f"binary_sensor.window_{i}" for i in range(_MAX_ENTITIES)
    ]
    snapshot["derived"]["last_motion_by_area"] = {  # type: ignore[typeddict-item]
        f"Room{i}": "2025-01-01T00:00:00" for i in range(20)
    }
    return snapshot


def test_reduced_snapshot_stays_within_token_budget() -> None:
    """The serialised reduced snapshot never exceeds _TOKEN_BUDGET_CHARS."""
    snapshot = _make_large_snapshot()
    reduced = reduce_snapshot_for_discovery(snapshot)
    char_count = len(json.dumps(reduced, default=str, separators=(",", ":")))
    assert char_count <= _TOKEN_BUDGET_CHARS, (
        f"Snapshot too large: {char_count} chars (budget {_TOKEN_BUDGET_CHARS})"
    )


def test_second_pass_trims_last_changed_when_over_budget() -> None:
    """_apply_budget_trim strips last_changed from groups when over budget."""
    # Build a result dict that is guaranteed to exceed _TOKEN_BUDGET_CHARS.
    big_groups = [
        {
            "entity_ids": [f"binary_sensor.window_{i}"],
            "state": "off",
            "area": f"Room{i}",
            "device_class": "window",
            "last_changed": "2025-01-01T00:00",
        }
        for i in range(500)
    ]
    oversized: dict[str, Any] = {
        "entities": big_groups,
        "camera_activity": [],
        "derived": {
            "now": "2025-01-01T00:00",
            "is_night": False,
            "anyone_home": True,
            "people_home": [],
            "people_away": [],
            "last_motion_by_area": {},
        },
    }
    assert len(json.dumps(oversized, separators=(",", ":"))) > _TOKEN_BUDGET_CHARS

    trimmed = _apply_budget_trim(oversized)
    for group in trimmed["entities"]:
        assert "last_changed" not in group
    # The trim may or may not reach budget depending on data volume; the key
    # invariant is that last_changed was removed (pass 1 fired).


def test_third_pass_truncates_camera_summaries_to_second_pass_chars() -> None:
    """_apply_budget_trim truncates summaries to _SECOND_PASS_SUMMARY_CHARS in pass 2."""
    # Build a dict that is over budget after pass 1 (no last_changed) but fits
    # under budget once camera summaries are shortened to _SECOND_PASS_SUMMARY_CHARS.
    # Use few entities (no last_changed to strip) and many cameras with long summaries.
    cameras = [
        {"camera_entity_id": f"camera.cam_{i}", "snapshot_summary": "X" * 300}
        for i in range(100)
    ]
    over_after_pass1: dict[str, Any] = {
        "entities": [],
        "camera_activity": cameras,
        "derived": {"now": "2025-01-01T00:00"},
    }
    assert (
        len(json.dumps(over_after_pass1, separators=(",", ":"))) > _TOKEN_BUDGET_CHARS
    )

    trimmed = _apply_budget_trim(over_after_pass1)
    for cam in trimmed["camera_activity"]:
        summary = cam.get("snapshot_summary", "")
        if summary:
            assert len(summary) <= _SECOND_PASS_SUMMARY_CHARS + 3  # +3 for "..."


def test_fourth_pass_drops_camera_summaries_when_still_over_budget() -> None:
    """_apply_budget_trim drops all summaries if still over budget after passes 1 and 2."""
    # Force over budget even after truncating summaries to _SECOND_PASS_SUMMARY_CHARS:
    # use many cameras so the total camera payload stays large even with short summaries.
    cameras = [
        {
            "camera_entity_id": f"camera.cam_{i:04d}",
            "area": f"VeryLongAreaNameThatTakesUpSpace_{i:04d}",
            "snapshot_summary": "Y" * 300,
            "recognized_people": [f"Person{j:04d}" for j in range(20)],
        }
        for i in range(200)
    ]
    still_over: dict[str, Any] = {
        "entities": [],
        "camera_activity": cameras,
        "derived": {"now": "2025-01-01T00:00"},
    }
    assert len(json.dumps(still_over, separators=(",", ":"))) > _TOKEN_BUDGET_CHARS

    trimmed = _apply_budget_trim(still_over)
    for cam in trimmed["camera_activity"]:
        assert "snapshot_summary" not in cam


def test_fourth_pass_caps_recognized_people_when_still_over_budget() -> None:
    """_apply_budget_trim caps recognized_people per camera in pass 4."""
    # Construct a payload that is over budget even after dropping all summaries:
    # many cameras with large recognized_people lists and no summaries.
    cameras = [
        {
            "camera_entity_id": f"camera.cam_{i:04d}",
            "area": f"VeryLongAreaNameThatTakesUpSpace_{i:04d}",
            "recognized_people": [f"PersonWithLongName{j:04d}" for j in range(50)],
        }
        for i in range(200)
    ]
    over_without_summaries: dict[str, Any] = {
        "entities": [],
        "camera_activity": cameras,
        "derived": {"now": "2025-01-01T00:00"},
    }
    assert (
        len(json.dumps(over_without_summaries, separators=(",", ":")))
        > _TOKEN_BUDGET_CHARS
    )

    trimmed = _apply_budget_trim(over_without_summaries)
    for cam in trimmed["camera_activity"]:
        people = cam.get("recognized_people", [])
        assert len(people) <= _MAX_RECOGNIZED_PEOPLE
