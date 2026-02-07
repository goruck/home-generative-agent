# ruff: noqa: S101
"""Tests for dynamic rules evaluation and registry."""

from __future__ import annotations

import pytest

from custom_components.home_generative_agent.sentinel.dynamic_rules import (
    evaluate_dynamic_rules,
)
from custom_components.home_generative_agent.sentinel.rule_registry import RuleRegistry


def _base_entity(entity_id: str, domain: str, state: str) -> dict:
    return {
        "entity_id": entity_id,
        "domain": domain,
        "state": state,
        "friendly_name": entity_id,
        "area": None,
        "attributes": {},
        "last_changed": "2026-02-01T00:00:00+00:00",
        "last_updated": "2026-02-01T00:00:00+00:00",
    }


def _snapshot(entities: list[dict], camera_activity: list[dict], derived: dict) -> dict:
    return {
        "schema_version": 1,
        "generated_at": "2026-02-01T00:00:00+00:00",
        "entities": entities,
        "camera_activity": camera_activity,
        "derived": derived,
    }


def test_dynamic_rule_alarm_disarmed_open_entry() -> None:
    snapshot = _snapshot(
        [
            _base_entity(
                "alarm_control_panel.home_alarm", "alarm_control_panel", "disarmed"
            ),
            _base_entity("binary_sensor.front_door", "binary_sensor", "on"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "alarm_rule_1",
            "template_id": "alarm_disarmed_open_entry",
            "params": {
                "alarm_entity_id": "alarm_control_panel.home_alarm",
                "entry_entity_ids": ["binary_sensor.front_door"],
            },
            "severity": "high",
            "confidence": 0.6,
            "is_sensitive": True,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "alarm_rule_1"
    assert "binary_sensor.front_door" in findings[0].triggering_entities


def test_dynamic_rule_unlocked_lock_when_home() -> None:
    snapshot = _snapshot(
        [
            _base_entity("lock.garage_door", "lock", "unlocked"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "lock_rule_1",
            "template_id": "unlocked_lock_when_home",
            "params": {"lock_entity_id": "lock.garage_door"},
            "severity": "medium",
            "confidence": 0.5,
            "is_sensitive": True,
            "suggested_actions": ["lock_entity"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "lock_rule_1"


def test_dynamic_rule_motion_without_camera_activity() -> None:
    snapshot = _snapshot(
        [
            _base_entity("binary_sensor.front_motion", "binary_sensor", "on"),
        ],
        [
            {
                "camera_entity_id": "camera.front",
                "area": None,
                "last_activity": None,
                "motion_entities": [],
                "vmd_entities": [],
                "snapshot_summary": None,
                "recognized_people": [],
                "latest_path": None,
            }
        ],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": True,
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "motion_rule_1",
            "template_id": "motion_without_camera_activity",
            "params": {
                "motion_entity_ids": ["binary_sensor.front_motion"],
                "camera_entity_id": "camera.front",
            },
            "severity": "low",
            "confidence": 0.4,
            "is_sensitive": False,
            "suggested_actions": ["check_camera"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "motion_rule_1"


def test_dynamic_rule_open_entry_at_night_when_home() -> None:
    snapshot = _snapshot(
        [
            _base_entity("binary_sensor.playroom_window", "binary_sensor", "on"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": True,
            "last_motion_by_area": {},
        },
    )
    rules = [
        {
            "rule_id": "entry_rule_1",
            "template_id": "open_entry_at_night_when_home",
            "params": {"entry_entity_ids": ["binary_sensor.playroom_window"]},
            "severity": "medium",
            "confidence": 0.7,
            "is_sensitive": True,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "entry_rule_1"


def test_dynamic_rule_open_any_window_at_night_while_away() -> None:
    snapshot = _snapshot(
        [
            _base_entity("binary_sensor.playroom_window", "binary_sensor", "on"),
            _base_entity("binary_sensor.front_door", "binary_sensor", "off"),
        ],
        [],
        {
            "now": "2026-02-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": True,
            "anyone_home": False,
            "last_motion_by_area": {},
        },
    )
    snapshot["entities"][0]["attributes"]["device_class"] = "window"
    rules = [
        {
            "rule_id": "entry_rule_2",
            "template_id": "open_any_window_at_night_while_away",
            "params": {"entry_selector": "window"},
            "severity": "high",
            "confidence": 0.7,
            "is_sensitive": True,
            "suggested_actions": ["close_entry"],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert len(findings) == 1
    assert findings[0].type == "entry_rule_2"


@pytest.mark.asyncio
async def test_rule_registry_add_duplicate(hass) -> None:
    registry = RuleRegistry(hass)
    await registry.async_load()
    rule = {"rule_id": "rule_1", "template_id": "alarm_disarmed_open_entry"}
    assert await registry.async_add_rule(rule)
    assert not await registry.async_add_rule(rule)
