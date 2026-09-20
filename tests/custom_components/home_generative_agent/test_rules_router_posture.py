# ruff: noqa: S101
"""The router posture rules and the guest-idle derivation that feeds one of them."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import pytest

from custom_components.home_generative_agent.sentinel.rules.network_common import (
    POSTURE_COOLDOWN_MINUTES,
)
from custom_components.home_generative_agent.sentinel.rules.network_router_posture import (
    NetworkDdnsEnabledRule,
    NetworkGuestNetworkIdleRule,
    NetworkProtectionDisabledRule,
    NetworkWpa3DisabledRule,
)
from custom_components.home_generative_agent.snapshot.network import (
    AdapterResult,
    NetworkBuildContext,
    derive_guest_idle,
    merge_adapter_results,
    posture_cap,
)
from custom_components.home_generative_agent.snapshot.schema import validate_snapshot

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import (
        AnomalyFinding,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


def _snapshot(posture: dict[str, Any]) -> FullStateSnapshot:
    caps = sorted(posture_cap(k) for k in posture if not k.endswith("_entity_id"))
    return validate_snapshot(
        {
            "schema_version": 2,
            "generated_at": NOW.isoformat(),
            "entities": [],
            "camera_activity": [],
            "derived": {
                "now": NOW.isoformat(),
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": True,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
            "network": {
                "capabilities": caps,
                "sources": dict.fromkeys(caps, "eero_runtime"),
                "clients": [],
                "posture": posture,
                "ha_security": {},
                "counters": {},
            },
        }
    )


def _only(findings: list[AnomalyFinding]) -> AnomalyFinding:
    assert len(findings) == 1
    return findings[0]


ALL = [
    NetworkGuestNetworkIdleRule(),
    NetworkWpa3DisabledRule(),
    NetworkProtectionDisabledRule(),
    NetworkDdnsEnabledRule(),
]


def test_every_rule_is_a_standing_posture_rule() -> None:
    for rule in ALL:
        assert rule.cooldown_minutes == POSTURE_COOLDOWN_MINUTES
        assert all(cap.startswith("network.posture.") for cap in rule.requires)
        assert rule.evaluate(_snapshot({})) == []


def test_guest_network_idle_fires_at_the_threshold() -> None:
    rule = NetworkGuestNetworkIdleRule(idle_days=7)
    assert rule.requires == {
        posture_cap("guest_network_enabled"),
        posture_cap("guest_network_idle_days"),
    }
    quiet = {"guest_network_enabled": True, "guest_network_idle_days": 6}
    assert rule.evaluate(_snapshot(quiet)) == []
    off = {"guest_network_enabled": False, "guest_network_idle_days": 30}
    assert rule.evaluate(_snapshot(off)) == []
    idle = {
        "guest_network_enabled": True,
        "guest_network_idle_days": 9,
        "guest_network_enabled_entity_id": "switch.kro_guest_network",
    }
    finding = _only(rule.evaluate(_snapshot(idle)))
    assert finding.severity == "low"
    assert finding.is_sensitive
    assert "no guest has connected for 9 days" in finding.evidence["summary"]
    assert finding.evidence["idle_days"] == 9
    assert finding.triggering_entities == ["switch.kro_guest_network"]
    # A standing condition: the day count is display, not identity.
    later = _only(rule.evaluate(_snapshot({**idle, "guest_network_idle_days": 10})))
    assert later.anomaly_id == finding.anomaly_id
    assert all("." not in action for action in finding.suggested_actions)


def test_wpa3_protection_and_ddns_fire_on_the_weak_value_only() -> None:
    assert NetworkWpa3DisabledRule().evaluate(_snapshot({"wpa3_enabled": True})) == []
    wpa3 = _only(NetworkWpa3DisabledRule().evaluate(_snapshot({"wpa3_enabled": False})))
    assert wpa3.severity == "low"
    assert "WPA3 is turned off" in wpa3.evidence["summary"]

    protection = NetworkProtectionDisabledRule()
    assert protection.evaluate(_snapshot({"malware_blocking_enabled": True})) == []
    off = _only(
        protection.evaluate(
            _snapshot({"malware_blocking_enabled": False, "ad_blocking_enabled": False})
        )
    )
    assert off.severity == "medium"
    assert "Ad blocking is off as well" in off.evidence["summary"]
    only_malware = _only(
        protection.evaluate(
            _snapshot({"malware_blocking_enabled": False, "ad_blocking_enabled": True})
        )
    )
    assert "Ad blocking" not in only_malware.evidence["summary"]
    # Ad blocking alone being off is not a security finding.
    assert protection.evaluate(_snapshot({"ad_blocking_enabled": False})) == []

    assert NetworkDdnsEnabledRule().evaluate(_snapshot({"ddns_enabled": False})) == []
    ddns = _only(NetworkDdnsEnabledRule().evaluate(_snapshot({"ddns_enabled": True})))
    assert ddns.severity == "low"
    assert "Dynamic DNS is on" in ddns.evidence["summary"]


def test_the_settings_switch_is_the_triggering_entity() -> None:
    """The engine's exclusion filter works on triggering entities; the rule adds none of its own."""
    posture = {"wpa3_enabled": False, "wpa3_enabled_entity_id": "switch.kro_wpa3"}
    finding = _only(NetworkWpa3DisabledRule().evaluate(_snapshot(posture)))
    assert finding.triggering_entities == ["switch.kro_wpa3"]
    bare = _only(NetworkWpa3DisabledRule().evaluate(_snapshot({"wpa3_enabled": False})))
    assert bare.triggering_entities == []


# ---------------------------------------------------------------------------
# derive_guest_idle
# ---------------------------------------------------------------------------


TODAY = NOW.replace(hour=0, minute=0, second=0, microsecond=0)


def _memory(active_days_ago: int, observed_days_ago: int = 0) -> dict[str, Any]:
    return {
        "guest_network_last_active": (
            TODAY - timedelta(days=active_days_ago)
        ).isoformat(),
        "guest_network_last_observed": (
            TODAY - timedelta(days=observed_days_ago)
        ).isoformat(),
    }


def _derive(
    posture: dict[str, Any],
    previous: dict[str, Any] | None,
    *,
    second: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results = [AdapterResult(name="eero_runtime", posture=posture)]
    if second is not None:
        results.insert(0, AdapterResult(name="eero", posture=second))
    section = merge_adapter_results(results)
    derive_guest_idle(section, NetworkBuildContext(previous_posture=previous), NOW)
    out = cast("dict[str, Any]", section["posture"])
    out["_caps"] = section["capabilities"]
    return out


IDLE = {"guest_network_enabled": True, "guest_client_count": 0}


def test_first_observation_starts_the_clock_and_judges_nothing() -> None:
    out = _derive(IDLE, None)
    # Days, not instants: the memory changes (and is written) at most daily.
    assert out["guest_network_last_active"] == TODAY.isoformat()
    assert out["guest_network_last_observed"] == TODAY.isoformat()
    assert "guest_network_idle_days" not in out
    assert posture_cap("guest_network_idle_days") not in out["_caps"]
    # The memory keys are never capabilities.
    assert posture_cap("guest_network_last_active") not in out["_caps"]
    assert posture_cap("guest_network_last_observed") not in out["_caps"]


def test_idle_days_count_from_the_remembered_day_while_watched() -> None:
    out = _derive(IDLE, _memory(9, 1))
    assert out["guest_network_idle_days"] == 9
    assert out["guest_network_last_active"] == (TODAY - timedelta(days=9)).isoformat()
    assert out["guest_network_last_observed"] == TODAY.isoformat()
    assert posture_cap("guest_network_idle_days") in out["_caps"]


@pytest.mark.parametrize(
    "posture",
    [
        {"guest_network_enabled": True, "guest_client_count": 2},  # a guest is on
        {"guest_network_enabled": False, "guest_client_count": 0},  # network is off
    ],
)
def test_the_clock_restarts_when_in_use_or_off(posture: dict[str, Any]) -> None:
    out = _derive(posture, _memory(30, 1))
    assert out["guest_network_last_active"] == TODAY.isoformat()
    assert out.get("guest_network_idle_days", 0) == 0


def test_an_unwatched_stretch_is_not_idle_time() -> None:
    """Ten days with no observation proves nothing about the guests in between."""
    out = _derive(IDLE, _memory(30, 10))
    assert out["guest_network_last_active"] == TODAY.isoformat()
    assert "guest_network_idle_days" not in out
    # A short gap (one missed day) keeps the clock.
    assert _derive(IDLE, _memory(30, 2))["guest_network_idle_days"] == 30


def test_a_run_without_an_observation_leaves_the_memory_alone() -> None:
    """A read gap neither restarts the clock nor advances it."""
    for posture in (
        {"upnp_enabled": True},  # no guest network state at all
        {"guest_network_enabled": True},  # on, but no guest count to go by
    ):
        out = _derive(posture, _memory(6, 0))
        assert "guest_network_last_active" not in out
        assert "guest_network_last_observed" not in out
        assert "guest_network_idle_days" not in out


def test_state_and_count_from_different_sources_are_not_an_observation() -> None:
    """Two sources can cover different networks; mixing them fabricates idleness."""
    out = _derive(
        {"guest_network_enabled": True},  # runtime: enabled, count withheld
        _memory(10, 0),
        second={"guest_client_count": 0},  # entity tier: another network's zero
    )
    assert "guest_network_idle_days" not in out
    assert "guest_network_last_active" not in out


def test_a_future_or_garbled_memory_restarts_the_clock() -> None:
    future = {
        "guest_network_last_active": "2099-01-01T00:00:00+00:00",
        "guest_network_last_observed": TODAY.isoformat(),
    }
    out = _derive(IDLE, future)
    assert out["guest_network_last_active"] == TODAY.isoformat()
    assert "guest_network_idle_days" not in out
    garbled = {
        "guest_network_last_active": "garbage",
        "guest_network_last_observed": "x",
    }
    out = _derive(IDLE, garbled)
    assert out["guest_network_last_active"] == TODAY.isoformat()
    assert "guest_network_idle_days" not in out
