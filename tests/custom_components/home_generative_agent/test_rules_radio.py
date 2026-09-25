# ruff: noqa: S101
"""Tests for the radio rules against synthetic snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from custom_components.home_generative_agent.sentinel.rules.network_common import (
    TRUST_ONE_ACTION,
    TRUST_SEVERAL_ACTION,
)
from custom_components.home_generative_agent.sentinel.rules.radio_coordinator_update_pending import (
    RadioCoordinatorUpdatePendingRule,
)
from custom_components.home_generative_agent.sentinel.rules.radio_new_device_joined import (
    RadioNewDeviceJoinedRule,
)
from custom_components.home_generative_agent.sentinel.rules.zigbee_permit_join_open import (
    ZigbeePermitJoinOpenRule,
)
from custom_components.home_generative_agent.sentinel.rules.zwave_inclusion_active import (
    ZwaveInclusionActiveRule,
)
from custom_components.home_generative_agent.sentinel.rules.zwave_insecure_security_class import (
    ZwaveInsecureSecurityClassRule,
)
from custom_components.home_generative_agent.snapshot.schema import validate_snapshot

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


def _device(  # noqa: PLR0913
    device_id: str,
    protocol: str = "zigbee",
    *,
    name: str | None = "Device",
    security: bool = False,
    security_class: str | None = None,
    manufacturer: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    device: dict[str, Any] = {
        "device_id": device_id,
        "protocol": protocol,
        "platform": {"zigbee": "zha", "zwave": "zwave_js"}.get(protocol, protocol),
        "name": name,
        "is_security_device": security,
        "manufacturer": manufacturer,
        "model": model,
    }
    if security_class is not None:
        device["security_class"] = security_class
    return device


def _snapshot(  # noqa: PLR0913
    *,
    devices: list[dict[str, Any]] | None = None,
    new_devices: list[str] | None = None,
    posture: dict[str, Any] | None = None,
    entities: list[dict[str, Any]] | None = None,
    anyone_home: bool = True,
    is_night: bool = False,
) -> FullStateSnapshot:
    radio: dict[str, Any] = {
        "capabilities": [],
        "devices": devices or [],
        "posture": posture or {},
    }
    if new_devices is not None:
        radio["new_devices"] = new_devices
    return validate_snapshot(
        {
            "schema_version": 2,
            "generated_at": "2026-09-13T12:00:00+00:00",
            "entities": entities or [],
            "camera_activity": [],
            "derived": {
                "now": "2026-09-13T12:00:00+00:00",
                "timezone": "UTC",
                "is_night": is_night,
                "anyone_home": anyone_home,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
            "network": {
                "capabilities": [],
                "sources": {},
                "clients": [],
                "posture": {},
                "ha_security": {},
                "counters": {},
                "radio": radio,
            },
        }
    )


def _only(findings: list[AnomalyFinding]) -> AnomalyFinding:
    assert len(findings) == 1, [f.evidence for f in findings]
    return findings[0]


# ---------------------------------------------------------------------------
# radio_new_device_joined
# ---------------------------------------------------------------------------


def test_new_device_severity_by_occupancy_night_and_security() -> None:
    rule = RadioNewDeviceJoinedRule()
    plain = [_device("a"), _device("b", name="Bulb")]
    lock = [_device("a"), _device("b", name="Lock", security=True)]
    assert (
        _only(
            rule.evaluate(_snapshot(devices=plain, new_devices=["zigbee:b"]))
        ).severity
        == "low"
    )
    assert (
        _only(
            rule.evaluate(
                _snapshot(devices=plain, new_devices=["zigbee:b"], is_night=True)
            )
        ).severity
        == "medium"
    )
    assert (
        _only(
            rule.evaluate(
                _snapshot(devices=plain, new_devices=["zigbee:b"], anyone_home=False)
            )
        ).severity
        == "medium"
    )
    finding = _only(
        rule.evaluate(
            _snapshot(devices=lock, new_devices=["zigbee:b"], anyone_home=False)
        )
    )
    assert finding.severity == "high"
    assert "while nobody is home" in finding.evidence["summary"]
    assert finding.is_sensitive


def test_new_device_summary_identity_and_actions() -> None:
    rule = RadioNewDeviceJoinedRule()
    devices = [
        _device("a"),
        _device("b", name="Front Lock", manufacturer="Schlage", model="BE468"),
        _device("c", "zwave", name=None),
    ]
    finding = _only(
        rule.evaluate(_snapshot(devices=devices, new_devices=["zigbee:b", "zwave:c"]))
    )
    summary = finding.evidence["summary"]
    assert summary.startswith("New radio devices joined: ")
    assert "Front Lock (Zigbee, Schlage BE468)" in summary
    assert "Unnamed device (Z-Wave)" in summary
    assert finding.evidence["device_keys"] == ["zigbee:b", "zwave:c"]
    assert finding.evidence["device_ids"] == ["b", "c"]
    assert not finding.triggering_entities
    # Two devices: the push offers no Trust button, so the action names the
    # service; one device keeps the tap.
    assert finding.suggested_actions[-1] == TRUST_SEVERAL_ACTION
    alone = _only(rule.evaluate(_snapshot(devices=devices, new_devices=["zigbee:b"])))
    assert alone.suggested_actions[-1] == TRUST_ONE_ACTION
    # Display fields do not change the identity; the device set does.
    renamed = [_device("a"), _device("b", name="Renamed"), _device("c", "zwave")]
    again = _only(
        rule.evaluate(_snapshot(devices=renamed, new_devices=["zigbee:b", "zwave:c"]))
    )
    assert again.anomaly_id == finding.anomaly_id


def test_new_device_caps_long_lists_and_ignores_stale_keys() -> None:
    rule = RadioNewDeviceJoinedRule()
    devices = [_device(f"d{i:02d}", name=f"Sensor {i:02d}") for i in range(14)]
    keys = [f"zigbee:d{i:02d}" for i in range(14)]
    finding = _only(rule.evaluate(_snapshot(devices=devices, new_devices=keys)))
    assert "and 4 more" in finding.evidence["summary"]
    # A key whose device is gone produces nothing.
    assert rule.evaluate(_snapshot(devices=[], new_devices=["zigbee:gone"])) == []
    assert rule.evaluate(_snapshot(devices=devices, new_devices=[])) == []
    assert rule.evaluate(_snapshot(devices=devices)) == []


# ---------------------------------------------------------------------------
# zwave_insecure_security_class
# ---------------------------------------------------------------------------


def test_zwave_security_devices_without_s2_are_high() -> None:
    rule = ZwaveInsecureSecurityClassRule()
    devices = [
        _device("lock", "zwave", name="Front Lock", security=True, security_class="s0"),
        _device("gate", "zwave", name="Gate", security=True, security_class="none"),
        _device(
            "safe", "zwave", name="Back Lock", security=True, security_class="s2_access"
        ),
        _device("sensor", "zwave", name="Motion", security_class="none"),
        _device("plug", "zwave", name="Old Plug", security_class="s0"),
        _device("zig", "zigbee", name="Zigbee Lock", security=True),
        _device("pending", "zwave", name="Interviewing", security=True),
    ]
    finding = _only(rule.evaluate(_snapshot(devices=devices)))
    assert finding.severity == "high"
    summary = finding.evidence["summary"]
    assert "Front Lock (legacy S0 security)" in summary
    assert "Gate (no security)" in summary
    assert "Old Plug" in summary
    for absent in ("Back Lock", "Motion", "Zigbee Lock", "Interviewing"):
        assert absent not in summary
    assert finding.evidence["devices"] == {"gate": "none", "lock": "s0", "plug": "s0"}


def test_zwave_only_s0_non_security_devices_is_low_and_none_is_quiet() -> None:
    rule = ZwaveInsecureSecurityClassRule()
    low = _only(
        rule.evaluate(
            _snapshot(
                devices=[_device("plug", "zwave", name="Plug", security_class="s0")]
            )
        )
    )
    assert low.severity == "low"
    quiet = [_device("sensor", "zwave", name="Motion", security_class="none")]
    assert rule.evaluate(_snapshot(devices=quiet)) == []
    assert rule.cooldown_minutes > 0


# ---------------------------------------------------------------------------
# zigbee_permit_join_open / zwave_inclusion_active
# ---------------------------------------------------------------------------


def test_permit_join_open_severity_and_exclusions() -> None:
    posture = {
        "zigbee_permit_join": True,
        "zigbee_permit_join_entity_ids": ["switch.zigbee2mqtt_bridge_permit_join"],
    }
    rule = ZigbeePermitJoinOpenRule()
    home = _only(rule.evaluate(_snapshot(posture=posture)))
    assert home.severity == "medium"
    assert home.triggering_entities == ["switch.zigbee2mqtt_bridge_permit_join"]
    away = _only(rule.evaluate(_snapshot(posture=posture, anyone_home=False)))
    assert away.severity == "high"
    assert rule.evaluate(_snapshot(posture={"zigbee_permit_join": False})) == []
    excluded = ZigbeePermitJoinOpenRule(is_entity_excluded=lambda _e, _t: True)
    assert excluded.evaluate(_snapshot(posture=posture)) == []


def test_zwave_inclusion_active() -> None:
    rule = ZwaveInclusionActiveRule()
    assert rule.evaluate(_snapshot(posture={"zwave_inclusion_active": False})) == []
    assert (
        _only(
            rule.evaluate(_snapshot(posture={"zwave_inclusion_active": True}))
        ).severity
        == "medium"
    )
    assert (
        _only(
            rule.evaluate(
                _snapshot(posture={"zwave_inclusion_active": True}, anyone_home=False)
            )
        ).severity
        == "high"
    )


# ---------------------------------------------------------------------------
# radio_coordinator_update_pending
# ---------------------------------------------------------------------------


def test_coordinator_update_lists_versions_and_honors_exclusions() -> None:
    entity = {
        "entity_id": "update.zbt_1_firmware",
        "domain": "update",
        "state": "on",
        "friendly_name": "ZBT-1 firmware",
        "area": None,
        "attributes": {"installed_version": "7.4.4", "latest_version": "7.4.5"},
        "last_changed": "2026-09-13T11:00:00+00:00",
        "last_updated": "2026-09-13T11:00:00+00:00",
    }
    posture = {"coordinator_update_pending": ["update.zbt_1_firmware"]}
    finding = _only(
        RadioCoordinatorUpdatePendingRule().evaluate(
            _snapshot(posture=posture, entities=[entity])
        )
    )
    assert finding.severity == "medium"
    assert "ZBT-1 firmware (7.4.4 -> 7.4.5)" in finding.evidence["summary"]
    assert finding.triggering_entities == ["update.zbt_1_firmware"]
    assert (
        RadioCoordinatorUpdatePendingRule(
            is_entity_excluded=lambda _e, _t: True
        ).evaluate(_snapshot(posture=posture, entities=[entity]))
        == []
    )
    assert (
        RadioCoordinatorUpdatePendingRule().evaluate(
            _snapshot(posture={"coordinator_update_pending": []})
        )
        == []
    )
