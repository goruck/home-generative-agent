# ruff: noqa: S101
"""
A motion push names its sensor in the subtitle (issue #897).

The subtitle was the rule's label, "Motion at night, alarm disarmed", and
named the sensor only in the body (since #731). A collapsed push shows title
and subtitle only, so two pushes about two sensors looked the same.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
from custom_components.home_generative_agent.sentinel.notifier import _build_subtitle

from .test_sentinel_notifier import _entry_snapshot

BACKYARD = "binary_sensor.backyard_motion"
HALL = "binary_sensor.hall_motion"


def _motion_finding(
    template_id: str, states: dict[str, str] | None = None
) -> AnomalyFinding:
    states = states or {BACKYARD: "on"}
    return AnomalyFinding(
        anomaly_id="m1",
        type="candidate_motion_rule",
        severity="high",
        confidence=0.7,
        triggering_entities=[*states, "alarm_control_panel.home"],
        evidence={
            "rule_id": "candidate_motion_rule",
            "template_id": template_id,
            "alarm_entity_id": "alarm_control_panel.home",
            "motion_entity_ids": list(states),
            "motion_states": states,
        },
        suggested_actions=[],
        is_sensitive=True,
    )


def _snapshot() -> Any:
    return cast(
        "Any",
        _entry_snapshot(
            {
                "entity_id": BACKYARD,
                "domain": "binary_sensor",
                "state": "on",
                "friendly_name": "Backyard",
                "area": "Backyard",
                "attributes": {"device_class": "motion"},
                "last_changed": "2026-11-02T03:00:00+00:00",
                "last_updated": "2026-11-02T03:00:00+00:00",
            }
        ),
    )


@pytest.mark.parametrize(
    ("template_id", "subtitle"),
    [
        (
            "motion_detected_at_night_while_alarm_disarmed",
            "Backyard: motion at night, alarm disarmed",
        ),
        (
            "motion_while_alarm_disarmed_and_home_present",
            "Backyard: motion, alarm disarmed, someone home",
        ),
        ("motion_detected_at_night_while_away", "Backyard: motion at night while away"),
        ("motion_detected_while_away", "Backyard: motion while away"),
    ],
)
def test_the_subtitle_names_the_sensor(template_id: str, subtitle: str) -> None:
    assert _build_subtitle(_motion_finding(template_id), None, _snapshot()) == subtitle


def test_every_sensor_that_is_on_is_named() -> None:
    """
    A second push, for another sensor that turned on meanwhile, reads differently.

    Naming only the first sensor in the rule's order repeated the first push.
    """
    first = _motion_finding(
        "motion_detected_at_night_while_alarm_disarmed",
        {HALL: "on", BACKYARD: "off"},
    )
    second = _motion_finding(
        "motion_detected_at_night_while_alarm_disarmed",
        {HALL: "on", BACKYARD: "on"},
    )
    assert _build_subtitle(first, None, _snapshot()) == (
        "Hall Motion: motion at night, alarm disarmed"
    )
    assert _build_subtitle(second, None, _snapshot()) == (
        "Hall Motion, Backyard: motion at night, alarm disarmed"
    )


def test_more_than_two_sensors_are_counted() -> None:
    states = {HALL: "on", BACKYARD: "on", "binary_sensor.porch_motion": "on"}
    finding = _motion_finding("motion_detected_while_away", states)
    assert _build_subtitle(finding, None, _snapshot()) == (
        "Hall Motion, Backyard +1: motion while away"
    )


def test_motion_without_camera_activity_names_its_sensor() -> None:
    finding = AnomalyFinding(
        anomaly_id="m2",
        type="candidate_camera_rule",
        severity="medium",
        confidence=0.6,
        triggering_entities=[BACKYARD, "camera.backyard"],
        evidence={
            "rule_id": "candidate_camera_rule",
            "template_id": "motion_without_camera_activity",
            "motion_entity_id": BACKYARD,
            "motion_state": "on",
        },
        suggested_actions=[],
        is_sensitive=False,
    )
    assert _build_subtitle(finding, None, _snapshot()) == (
        "Backyard: motion, no camera activity"
    )


def test_a_motion_finding_with_no_sensor_keeps_the_label() -> None:
    finding = _motion_finding("motion_detected_at_night_while_alarm_disarmed")
    finding.evidence["motion_entity_ids"] = []
    assert _build_subtitle(finding, None, _snapshot()) == (
        "Motion at night, alarm disarmed"
    )
