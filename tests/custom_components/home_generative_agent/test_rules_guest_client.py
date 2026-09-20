# ruff: noqa: S101
"""The network_guest_client_present rule."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from custom_components.home_generative_agent.sentinel.rules.network_common import (
    POSTURE_COOLDOWN_MINUTES,
)
from custom_components.home_generative_agent.sentinel.rules.network_guest_client_present import (
    NetworkGuestClientPresentRule,
)
from custom_components.home_generative_agent.sentinel.rules.network_unknown_device_joined import (
    describe_client,
)
from custom_components.home_generative_agent.snapshot.network import (
    CAP_CLIENTS,
    CAP_GUEST_CLIENTS,
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

NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


def _guest(key: str, **extra: Any) -> dict[str, Any]:
    """Return a connected, untrusted guest client first seen three days ago."""
    client: dict[str, Any] = {
        "key": key,
        "connected": True,
        "is_guest": True,
        "trusted": False,
        "name": f"Phone {key}",
        "connection_type": "wireless",
        "tracker_entity_id": f"device_tracker.{key}",
        "first_seen": (NOW - timedelta(days=3)).isoformat(),
    }
    client.update(extra)
    return {k: v for k, v in client.items() if v is not None}


def _snapshot(
    clients: list[dict[str, Any]],
    *,
    people_home: list[str] | None = None,
    people_away: list[str] | None = None,
    is_night: bool = False,
    new_clients: list[str] | None = None,
) -> FullStateSnapshot:
    """Return a snapshot; by default one tracked person, away."""
    home = people_home or []
    away = ["person.sam"] if people_away is None else people_away
    caps = [CAP_CLIENTS, CAP_NEW_CLIENTS, CAP_GUEST_CLIENTS]
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
                "anyone_home": bool(home),
                "people_home": home,
                "people_away": away,
                "last_motion_by_area": {},
            },
            "network": {
                "capabilities": caps,
                "sources": dict.fromkeys(caps, "eero_runtime"),
                "clients": clients,
                "new_clients": new_clients or [],
                "posture": {},
                "ha_security": {},
                "counters": {},
            },
        }
    )


def _only(findings: list[AnomalyFinding]) -> AnomalyFinding:
    assert len(findings) == 1
    return findings[0]


def test_rule_declares_its_capabilities_and_the_daily_floor() -> None:
    assert NetworkGuestClientPresentRule.requires == {
        CAP_CLIENTS,
        CAP_NEW_CLIENTS,
        CAP_GUEST_CLIENTS,
    }
    assert NetworkGuestClientPresentRule.cooldown_minutes == POSTURE_COOLDOWN_MINUTES


def test_a_guest_while_someone_is_home_is_what_the_network_is_for() -> None:
    rule = NetworkGuestClientPresentRule()
    home = {"people_home": ["person.sam"], "people_away": []}
    assert rule.evaluate(_snapshot([_guest("a")], **home)) == []
    # Night is not a trigger: the derived flag follows the sun, and an
    # evening or overnight visitor is the ordinary case.
    assert rule.evaluate(_snapshot([_guest("a")], is_night=True, **home)) == []


def test_an_install_without_person_tracking_is_never_away() -> None:
    # ``anyone_home`` is False there, which must not read as "nobody is home".
    rule = NetworkGuestClientPresentRule()
    assert rule.evaluate(_snapshot([_guest("a")], people_away=[])) == []


def test_untrusted_guest_while_nobody_is_home_is_medium() -> None:
    finding = _only(NetworkGuestClientPresentRule().evaluate(_snapshot([_guest("a")])))
    assert finding.type == "network_guest_client_present"
    assert finding.severity == "medium"
    assert finding.is_sensitive
    assert finding.triggering_entities == ["device_tracker.a"]
    assert finding.evidence["client_keys"] == ["a"]
    # Inventory keys, so the Trust button resolves them.
    assert finding.evidence["device_ids"] == ["router:a"]
    assert finding.evidence["summary"] == (
        "1 guest Wi-Fi device you have not trusted is connected while nobody "
        "is home: Phone a (wireless, guest Wi-Fi)."
    )


def test_names_and_keys_line_up_whatever_the_client_order() -> None:
    finding = _only(
        NetworkGuestClientPresentRule().evaluate(_snapshot([_guest("b"), _guest("a")]))
    )
    assert finding.evidence["client_keys"] == ["a", "b"]
    assert finding.evidence["device_ids"] == ["router:a", "router:b"]
    assert [n.split(" (")[0] for n in finding.evidence["names"]] == [
        "Phone a",
        "Phone b",
    ]
    assert finding.evidence["summary"].startswith(
        "2 guest Wi-Fi devices you have not trusted are connected while nobody"
    )


def test_only_connected_untrusted_guests_are_reported() -> None:
    clients = [
        _guest("main", is_guest=False),
        _guest("unflagged", is_guest=None),
        _guest("offline", connected=False),
        # Parked on the guest network on purpose, or a visitor the owner
        # vouched for.
        _guest("trusted", trusted=True),
        # No inventory verdict yet: the unknown-device rule's business.
        _guest("no_row", trusted=None),
        # The registry vouches for it this run; the row catches up after it.
        _guest("auto", auto_trust=True),
        # A rotated random address on a device a trusted row already names.
        _guest("rotated", mac_randomized=True, hostname_trusted=True),
        _guest("stranger"),
    ]
    finding = _only(NetworkGuestClientPresentRule().evaluate(_snapshot(clients)))
    assert finding.evidence["client_keys"] == ["stranger"]


def test_the_first_day_belongs_to_the_unknown_device_rule() -> None:
    rule = NetworkGuestClientPresentRule()
    # Announced earlier today: a second push would be noise.
    fresh = _guest("a", first_seen=(NOW - timedelta(hours=23)).isoformat())
    assert rule.evaluate(_snapshot([fresh])) == []
    day_old = _guest("a", first_seen=(NOW - timedelta(days=1)).isoformat())
    assert len(rule.evaluate(_snapshot([day_old]))) == 1
    # A row without a readable first sighting is not judged.
    assert rule.evaluate(_snapshot([_guest("a", first_seen=None)])) == []
    assert rule.evaluate(_snapshot([_guest("a", first_seen="garbled")])) == []


def test_an_unsettled_new_device_alert_does_not_hide_the_client_forever() -> None:
    # Excluded from the unknown-device rule only, or held by quiet hours for
    # days: the key stays in new_clients, and after a day this rule judges it.
    finding = _only(
        NetworkGuestClientPresentRule().evaluate(
            _snapshot([_guest("a")], new_clients=["a"])
        )
    )
    assert finding.evidence["client_keys"] == ["a"]


def test_identity_is_the_client_set() -> None:
    rule = NetworkGuestClientPresentRule()
    first = _only(rule.evaluate(_snapshot([_guest("a")])))
    again = _only(rule.evaluate(_snapshot([_guest("a", ip="192.168.4.9")])))
    other = _only(rule.evaluate(_snapshot([_guest("b")])))
    assert first.anomaly_id == again.anomaly_id
    assert first.anomaly_id != other.anomaly_id


def test_excluded_tracker_is_skipped() -> None:
    calls: list[tuple[str, str]] = []

    def excluded(entity_id: str, rule_id: str) -> bool:
        calls.append((entity_id, rule_id))
        return entity_id == "device_tracker.a"

    rule = NetworkGuestClientPresentRule(is_entity_excluded=excluded)
    finding = _only(rule.evaluate(_snapshot([_guest("a"), _guest("b")])))
    assert finding.evidence["client_keys"] == ["b"]
    assert ("device_tracker.a", "network_guest_client_present") in calls


def test_no_dhcp_hostname_or_key_leaks_into_the_summary() -> None:
    client = _guest("a", name=None, hostname="Annas-iPhone", manufacturer="Apple")
    finding = _only(NetworkGuestClientPresentRule().evaluate(_snapshot([client])))
    assert "Annas" not in finding.evidence["summary"]
    assert "Apple a" in finding.evidence["summary"]


def test_unknown_device_copy_names_the_guest_network() -> None:
    assert describe_client(_guest("a")) == "Phone a (wireless, guest Wi-Fi)"
    assert describe_client(_guest("a", is_guest=False)) == "Phone a (wireless)"
