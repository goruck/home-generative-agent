# ruff: noqa: S101
"""
README community-dashboard recipes must reference real sensor attributes.

The ``entity-attributes-card`` recipes in README's Community Dashboards
section address rows as ``key: sensor.<entity>.<attribute>``. The card
silently omits any row whose key does not resolve — discussion #513's
per-camera recipe originally shipped three dead keys that nobody noticed
until PR #528 corrected them. Issue #538 added a second recipe of this
card type (the compact Sentinel health variant), so the documented keys
are pinned here: renaming or removing a sensor attribute must fail this
test and prompt a README update instead of a silently thinner card.

The checks run behavior-level, against the attribute dictionaries the
real sensor classes actually build, not against source-text greps — a
grep accepts docstring hits and producer-side literals that never reach
the sensor. A separate losslessness check pins the regex extractor
itself: every ``- key:`` row in README must parse, so reformatting a
recipe (quoted keys, wildcards, renamed entities) cannot silently drop
rows from coverage.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from custom_components.home_generative_agent.core.recognized_sensor import (
    RecognizedPeopleSensor,
)
from custom_components.home_generative_agent.sentinel.trigger_scheduler import (
    SentinelTriggerScheduler,
)

from .test_sentinel_health_sensor import _make_sensor, _mock_async_write_ha_state

_README = Path(__file__).resolve().parents[3] / "README.md"
_KEY_LINE = re.compile(r"^\s*- key: sensor\.(\w+)\.(\w+)\s*$", re.MULTILINE)
_ANY_KEY_LINE = re.compile(r"^\s*- key:", re.MULTILINE)

_SENTINEL_RECIPE_KEYS = {
    "triggers_excluded",
    "baseline_rules_waiting",
    "last_run_start",
    "run_duration_ms",
    "active_rule_count",
    "triggers_dropped_incoming",
    "triggers_ttl_expired",
    "triggers_dropped_queued",
}
_RECOGNIZED_RECIPE_KEYS = {"recognized_people", "count", "summary", "last_event"}


def _documented_keys() -> list[tuple[str, str]]:
    """Extract (entity, attribute) pairs from README entity-attributes-card rows."""
    return _KEY_LINE.findall(_README.read_text(encoding="utf-8"))


def test_key_extraction_is_lossless() -> None:
    """Every ``- key:`` row in README parses into the (entity, attribute) form."""
    text = _README.read_text(encoding="utf-8")
    assert len(_ANY_KEY_LINE.findall(text)) == len(_KEY_LINE.findall(text)), (
        "A README '- key:' row no longer matches the extractor pattern "
        "'- key: sensor.<entity>.<attribute>' — it has silently dropped out "
        "of parity coverage. Update the recipe or the extractor."
    )


def test_every_documented_entity_is_covered() -> None:
    """Each documented entity has a behavior-level check in this module."""
    unknown = {
        entity
        for entity, _ in _documented_keys()
        if entity != "sentinel_health" and not entity.endswith("_recognized_people")
    }
    assert not unknown, (
        f"README documents entity-attributes-card keys for {sorted(unknown)} "
        "but this module has no parity check for them — add one."
    )


@pytest.mark.asyncio
async def test_sentinel_recipe_keys_are_real_sensor_attributes() -> None:
    """
    Every documented sentinel_health key exists on a refreshed health sensor.

    The scheduler stats come from a real ``SentinelTriggerScheduler`` so the
    three ``triggers_*`` counters are the genuine post-first-run attribute
    names, not a hand-copied list that could drift alongside the README.
    """
    documented = {
        attr for entity, attr in _documented_keys() if entity == "sentinel_health"
    }
    assert documented >= _SENTINEL_RECIPE_KEYS, (
        "The issue #538 compact variant lost documented rows: "
        f"{sorted(_SENTINEL_RECIPE_KEYS - documented)}"
    )

    sensor = _make_sensor(run_stats={"scheduler": SentinelTriggerScheduler().stats})
    _mock_async_write_ha_state(sensor)
    await sensor._async_refresh()

    missing = documented - set(sensor._attrs)
    assert not missing, (
        f"README documents sensor.sentinel_health attributes {sorted(missing)} "
        "that the sensor no longer exposes — entity-attributes-card would "
        "silently drop those rows. Update the README recipe."
    )


def test_recognized_recipe_keys_are_real_sensor_attributes() -> None:
    """Every documented recognized-people key exists on the per-camera sensor."""
    documented = {
        attr
        for entity, attr in _documented_keys()
        if entity.endswith("_recognized_people")
    }
    assert documented >= _RECOGNIZED_RECIPE_KEYS, (
        "The per-camera recipe lost documented rows: "
        f"{sorted(_RECOGNIZED_RECIPE_KEYS - documented)}"
    )

    sensor = RecognizedPeopleSensor(MagicMock(), "camera.test_cam")
    missing = documented - set(sensor._attrs)
    assert not missing, (
        f"README documents recognized-people attributes {sorted(missing)} "
        "that the sensor no longer exposes — entity-attributes-card would "
        "silently drop those rows. Update the README recipe."
    )
