# ruff: noqa: S101
"""The network_unknown_device_joined rule."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from custom_components.home_generative_agent.sentinel.rules.network_unknown_device_joined import (
    NetworkUnknownDeviceJoinedRule,
    describe_client,
)
from custom_components.home_generative_agent.snapshot.network import (
    CAP_CLIENTS,
    CAP_NEW_CLIENTS,
)
from custom_components.home_generative_agent.snapshot.schema import validate_snapshot

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import (
        AnomalyFinding,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

NOW = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)


def _client(
    key: str,
    *,
    connected: bool = True,
    first_seen: datetime | None = NOW - timedelta(minutes=10),
    **extra: Any,
) -> dict[str, Any]:
    client: dict[str, Any] = {
        "key": key,
        "connected": connected,
        "name": f"Device {key}",
        "manufacturer": "Apple",
        "connection_type": "wireless",
        "ip": "192.168.1.23",
        "tracker_entity_id": f"device_tracker.{key}",
        **extra,
    }
    if first_seen is not None:
        client["first_seen"] = first_seen.isoformat()
    return client


def _snapshot(
    clients: list[dict[str, Any]],
    new_clients: list[str] | None,
    *,
    anyone_home: bool = True,
    is_night: bool = False,
) -> FullStateSnapshot:
    caps = [CAP_CLIENTS] + ([CAP_NEW_CLIENTS] if new_clients is not None else [])
    network: dict[str, Any] = {
        "capabilities": caps,
        "sources": dict.fromkeys(caps, "generic_router_tracker"),
        "clients": clients,
        "posture": {},
        "ha_security": {},
        "counters": {},
    }
    if new_clients is not None:
        network["new_clients"] = new_clients
    return validate_snapshot(
        {
            "schema_version": 2,
            "generated_at": NOW.isoformat(),
            "entities": [],
            "camera_activity": [],
            "derived": {
                "now": NOW.isoformat(),
                "timezone": "UTC",
                "is_night": is_night,
                "anyone_home": anyone_home,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
            "network": network,
        }
    )


def _only(findings: list[AnomalyFinding]) -> AnomalyFinding:
    assert len(findings) == 1
    return findings[0]


def test_rule_declares_client_capabilities() -> None:
    assert NetworkUnknownDeviceJoinedRule.requires == {CAP_CLIENTS, CAP_NEW_CLIENTS}
    assert NetworkUnknownDeviceJoinedRule.cooldown_minutes == 0


def test_nothing_new_means_no_finding() -> None:
    rule = NetworkUnknownDeviceJoinedRule(grace_minutes=0)
    assert rule.evaluate(_snapshot([_client("a")], [])) == []
    assert rule.evaluate(_snapshot([_client("a")], None)) == []
    # A key the section no longer carries is ignored.
    assert rule.evaluate(_snapshot([_client("a")], ["gone"])) == []


def test_grace_period_waits_for_the_inventory_row() -> None:
    rule = NetworkUnknownDeviceJoinedRule(grace_minutes=5)
    # Not recorded yet: the commit after this run starts the clock.
    assert rule.evaluate(_snapshot([_client("a", first_seen=None)], ["a"])) == []
    # Recorded two minutes ago: still inside the grace period.
    recent = _client("a", first_seen=NOW - timedelta(minutes=2))
    assert rule.evaluate(_snapshot([recent], ["a"])) == []
    # Recorded ten minutes ago: reported.
    assert len(rule.evaluate(_snapshot([_client("a")], ["a"]))) == 1
    # Grace 0 reports on first sight.
    zero = NetworkUnknownDeviceJoinedRule(grace_minutes=0)
    assert len(zero.evaluate(_snapshot([_client("a", first_seen=None)], ["a"]))) == 1


def test_only_connected_and_unexcluded_clients_are_reported() -> None:
    rule = NetworkUnknownDeviceJoinedRule(
        grace_minutes=0,
        is_entity_excluded=lambda _rule, entity_id: entity_id.endswith(".c"),
    )
    clients = [_client("a"), _client("b", connected=False), _client("c")]
    finding = _only(rule.evaluate(_snapshot(clients, ["a", "b", "c"])))
    assert finding.evidence["client_keys"] == ["a"]
    assert finding.triggering_entities == ["device_tracker.a"]


def test_severity_by_occupancy_night_and_known_random_address() -> None:
    rule = NetworkUnknownDeviceJoinedRule(grace_minutes=0)
    home = _only(rule.evaluate(_snapshot([_client("a")], ["a"])))
    assert home.severity == "medium"
    assert "while nobody is home" not in home.evidence["summary"]
    away = _only(rule.evaluate(_snapshot([_client("a")], ["a"], anyone_home=False)))
    assert away.severity == "high"
    assert "while nobody is home" in away.evidence["summary"]
    night = _only(rule.evaluate(_snapshot([_client("a")], ["a"], is_night=True)))
    assert night.severity == "high"
    assert "at night" in night.evidence["summary"]
    rotated = _client("a", mac_randomized=True, hostname_trusted=True)
    low = _only(rule.evaluate(_snapshot([rotated], ["a"], anyone_home=False)))
    assert low.severity == "low"
    assert low.evidence["randomized_known"] == ["a"]
    # One unknown device among rotated ones keeps the full severity.
    mixed = _only(rule.evaluate(_snapshot([rotated, _client("b")], ["a", "b"])))
    assert mixed.severity == "medium"


def test_identity_display_and_actions() -> None:
    rule = NetworkUnknownDeviceJoinedRule(grace_minutes=0)
    first = _only(rule.evaluate(_snapshot([_client("a"), _client("b")], ["a", "b"])))
    # Identity is the set of client keys; names, addresses, and the inventory
    # keys the Trust button uses are display only.
    renamed = [_client("a", name="Other", ip="10.0.0.9"), _client("b")]
    again = _only(rule.evaluate(_snapshot(renamed, ["a", "b"])))
    assert first.anomaly_id == again.anomaly_id
    assert first.evidence["device_ids"] == ["router:a", "router:b"]
    assert first.evidence["names"] == [
        "Device a (Apple, wireless, 192.168.1.23)",
        "Device b (Apple, wireless, 192.168.1.23)",
    ]
    assert first.evidence["summary"] == (
        "New devices on the network: Device a (Apple, wireless, 192.168.1.23), "
        "Device b (Apple, wireless, 192.168.1.23)."
    )
    assert first.is_sensitive
    assert first.suggested_actions[-1] == "Tap Trust device if you recognize it"
    assert all("." not in action for action in first.suggested_actions)
    assert first.triggering_entities == ["device_tracker.a", "device_tracker.b"]


def test_describe_client_falls_back_gracefully() -> None:
    assert describe_client({"hostname": "nas"}) == "nas"
    assert describe_client({"manufacturer": "Acme"}) == "Unnamed device (Acme)"
    assert describe_client({"name": "TV", "ip": "10.0.0.2"}) == "TV (10.0.0.2)"
