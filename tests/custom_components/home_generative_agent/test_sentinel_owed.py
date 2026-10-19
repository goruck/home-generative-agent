# ruff: noqa: S101
"""Change alerts owed from autonomy level 0 (#799): what is kept and how it is told."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from custom_components.home_generative_agent.sentinel import owed
from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    Severity,
    anomaly_finding_from_dict,
)

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)


def _finding(  # noqa: PLR0913
    kind: str,
    evidence: dict[str, Any],
    *,
    minute: int = 0,
    severity: Severity = "medium",
    anomaly_id: str | None = None,
    entities: list[str] | None = None,
) -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id=anomaly_id or f"{kind}:{sorted(evidence.items())}",
        type=kind,
        severity=severity,
        confidence=0.9,
        triggering_entities=entities or [],
        evidence=evidence,
        suggested_actions=["Check it"],
        is_sensitive=True,
        detected_at=NOW + timedelta(minutes=minute),
    )


def _admins(*names: str, tokens: tuple[str, ...] = (), **kw: Any) -> AnomalyFinding:
    return _finding(
        "ha_new_admin_or_token",
        {
            "new_admin_users": list(names),
            "new_long_lived_tokens": list(tokens),
            "summary": f"New admins: {', '.join(names)}.",
        },
        **kw,
    )


def _clients(*keys: str, **kw: Any) -> AnomalyFinding:
    return _finding(
        "network_unknown_device_joined",
        {
            "client_keys": list(keys),
            "device_ids": [f"router:{k}" for k in keys],
            "summary": f"New clients: {', '.join(keys)}.",
        },
        **kw,
    )


def test_an_account_alert_is_covered_only_by_one_naming_every_account() -> None:
    owed_alert = _admins("eve", tokens=("t1",))
    assert owed.covers(_admins("eve", "bob", tokens=("t1",)), owed_alert)
    assert not owed.covers(_admins("eve"), owed_alert)
    assert not owed.covers(_admins("bob", tokens=("t1",)), owed_alert)


def test_a_client_alert_is_covered_by_a_superset_not_another_client() -> None:
    assert owed.covers(_clients("k1", "k2"), _clients("k1"))
    assert not owed.covers(_clients("k2"), _clients("k1"))
    assert not owed.covers(_admins("k1"), _clients("k1"))


def test_a_stored_string_is_not_read_as_a_list_of_items() -> None:
    """A malformed saved value never matches by its characters."""
    broken = _finding("network_unknown_device_joined", {"client_keys": "k1"})
    assert not owed.covers(broken, _clients("k"))
    assert not owed.covers(broken, _clients("1"))
    assert owed.covers(_clients("k1"), broken)  # names nothing, so nothing lost


def test_a_growing_set_keeps_one_entry() -> None:
    store: dict[str, list[dict[str, Any]]] = {}
    assert owed.record(store, _clients("k1"))
    assert owed.record(store, _clients("k1", "k2", minute=1))
    assert not owed.record(store, _clients("k1", minute=2))
    members = owed.read_members(store["network_unknown_device_joined"])
    assert [m.evidence["client_keys"] for m in members] == [["k1", "k2"]]


def test_one_kind_never_pushes_out_another(caplog: pytest.LogCaptureFixture) -> None:
    """MAC churn fills its own kind only, and cannot bury the first change."""
    store: dict[str, list[dict[str, Any]]] = {}
    owed.record(store, _admins("eve"))
    for n in range(owed.MAX_OWED_PER_KIND + 5):
        owed.record(store, _clients(f"k{n:02d}", minute=n))
    assert len(store["network_unknown_device_joined"]) == owed.MAX_OWED_PER_KIND
    kept = owed.read_members(store["network_unknown_device_joined"])
    assert kept[0].evidence["client_keys"] == ["k00"]
    assert len(store["ha_new_admin_or_token"]) == 1
    assert caplog.text.count("are counted, not described") == 1
    sent = owed.replay("network_unknown_device_joined", kept)
    # Every client is still named in the evidence and counted in the note.
    assert len(sent.evidence["client_keys"]) == owed.MAX_OWED_PER_KIND + 5
    assert ": 25 devices;" in sent.evidence["summary"]
    assert "5 more devices seen are not described here." in sent.evidence["summary"]
    assert "owed_not_listed" not in sent.evidence


def test_an_unlisted_client_seen_again_is_not_counted_again() -> None:
    """A rule re-fires every run at level 0: the count is of clients, not runs."""
    store: dict[str, list[dict[str, Any]]] = {}
    for n in range(owed.MAX_OWED_PER_KIND + 1):
        owed.record(store, _clients(f"k{n:02d}", minute=n))
    before = [dict(m) for m in store["network_unknown_device_joined"]]
    assert not owed.record(store, _clients("k20", minute=99))
    assert store["network_unknown_device_joined"] == before


def test_a_superset_at_capacity_replaces_not_counts() -> None:
    store: dict[str, list[dict[str, Any]]] = {}
    for n in range(owed.MAX_OWED_PER_KIND):
        owed.record(store, _clients(f"k{n:02d}", minute=n))
    assert owed.record(store, _clients("k00", "new", minute=99))
    kept = owed.read_members(store["network_unknown_device_joined"])
    assert len(kept) == owed.MAX_OWED_PER_KIND
    assert any(m.evidence["client_keys"] == ["k00", "new"] for m in kept)
    assert all("owed_not_listed" not in m.evidence for m in kept)


def test_a_posture_kind_keeps_each_change() -> None:
    """An IP that went A to B and back, then to C, changed twice."""
    store: dict[str, list[dict[str, Any]]] = {}
    b = _finding(
        "network_public_ip_changed", {"public_ip_key": "b", "summary": "IP changed."}
    )
    c = _finding(
        "network_public_ip_changed",
        {"public_ip_key": "c", "summary": "IP changed."},
        minute=60,
    )
    assert owed.record(store, b)
    assert not owed.record(store, b)
    assert owed.record(store, c)
    members = owed.read_members(store["network_public_ip_changed"])
    assert [m.evidence["public_ip_key"] for m in members] == ["b", "c"]
    sent = owed.replay("network_public_ip_changed", members)
    # Told once, across both sightings.
    assert sent.evidence["summary"].count("IP changed.") == 1
    assert " to " in sent.evidence["summary"].split(")", 1)[0]
    assert owed.covers(c, members[1])
    assert not owed.covers(c, members[0])


def test_the_owed_kinds_match_the_change_alert_rules() -> None:
    from custom_components.home_generative_agent.sentinel.rules.network_common import (  # noqa: PLC0415
        CHANGE_ALERT_RULE_TYPES,
    )

    assert set(owed.ITEM_KEYS) <= CHANGE_ALERT_RULE_TYPES
    assert CHANGE_ALERT_RULE_TYPES - set(owed.ITEM_KEYS) == {
        "network_public_ip_changed",
        "network_upnp_port_mapping_added",
    }
    assert set(owed.INVENTORY_ITEM_KEYS) <= set(owed.ITEM_KEYS)


@pytest.mark.parametrize(
    "extra", [{"owed_not_listed": "many"}, {"owed_first_seen": "2026-13-45T99:00"}]
)
def test_bad_saved_extras_do_not_break_a_replay(extra: dict[str, Any]) -> None:
    member = _clients("k1")
    member = replace(member, evidence={**member.evidence, **extra})
    sent = owed.replay("network_unknown_device_joined", [member])
    assert sent.evidence["client_keys"] == ["k1"]


def test_a_superset_keeps_the_first_sighting_and_top_severity() -> None:
    """A device that joined while nobody was home stays high when another joins."""
    store: dict[str, list[dict[str, Any]]] = {}
    owed.record(store, _clients("k1", minute=0, severity="high"))
    owed.record(store, _clients("k1", "k2", minute=90, severity="medium"))
    (member,) = owed.read_members(store["network_unknown_device_joined"])
    assert member.severity == "high"
    sent = owed.replay("network_unknown_device_joined", [member])
    assert " to " in sent.evidence["summary"].split(")", 1)[0]
    assert "owed_first_seen" not in sent.evidence


def test_a_later_sighting_raises_the_owed_severity() -> None:
    store: dict[str, list[dict[str, Any]]] = {}
    owed.record(store, _clients("k1", "k2", severity="low"))
    assert owed.record(store, _clients("k1", minute=5, severity="high"))
    (member,) = owed.read_members(store["network_unknown_device_joined"])
    assert member.severity == "high"
    assert not owed.record(store, _clients("k2", minute=6, severity="medium"))


def test_a_covered_finding_still_drops_unreadable_rows() -> None:
    store = {"network_unknown_device_joined": [_clients("k1").as_dict(), {"bogus": 1}]}
    assert owed.record(store, _clients("k1", minute=1))
    assert len(store["network_unknown_device_joined"]) == 1


def test_what_was_told_is_taken_out_of_an_owed_finding() -> None:
    member = _clients("k1", "k2")
    rest = owed.without(member, {"client_keys": {"k1"}})
    assert rest is not None
    assert rest.evidence["client_keys"] == ["k2"]
    assert owed.without(member, {"client_keys": {"k1", "k2"}}) is None
    assert owed.without(member, {}) is member
    posture = _finding("network_public_ip_changed", {"public_ip_key": "c"})
    assert owed.without(posture, {}) is posture


def test_a_replay_points_to_the_trust_service_not_the_button() -> None:
    from custom_components.home_generative_agent.sentinel.rules.network_common import (  # noqa: PLC0415
        TRUST_ONE_ACTION,
        TRUST_SEVERAL_ACTION,
    )

    member = replace(_clients("k1"), suggested_actions=[TRUST_ONE_ACTION])
    sent = owed.replay("network_unknown_device_joined", [member])
    assert sent.suggested_actions == [TRUST_SEVERAL_ACTION]


def test_a_replay_names_every_item_with_the_note_first() -> None:
    members = [
        _clients("k1", minute=0, severity="low"),
        _clients("k2", minute=90, severity="high", entities=["device_tracker.x"]),
    ]
    sent = owed.replay("network_unknown_device_joined", members)
    assert sent.evidence["client_keys"] == ["k1", "k2"]
    assert sent.severity == "high"
    assert sent.triggering_entities == ["device_tracker.x"]
    assert sent.detected_at == members[1].detected_at
    summary = sent.evidence["summary"]
    assert summary.startswith("Seen while Sentinel was at autonomy level 0 (")
    assert " to " in summary.split(")", 1)[0]
    assert "New clients: k1. New clients: k2." in summary
    assert summary.endswith(
        "Trust the ones you recognize with the Sentinel trust device service."
    )
    assert ": 2 devices;" in summary
    # No Trust device button on a replay.
    assert "device_ids" not in sent.evidence
    assert sent.evidence[owed.SEEN_WHILE_SILENCED] is True


def test_a_posture_replay_keeps_its_identity() -> None:
    latest = _finding(
        "network_public_ip_changed",
        {"public_ip_key": "c", "summary": "IP changed."},
        anomaly_id="ip-c",
    )
    sent = owed.replay("network_public_ip_changed", [latest])
    assert sent.anomaly_id == "ip-c"
    assert sent.evidence["public_ip_key"] == "c"
    assert " to " not in sent.evidence["summary"].split(")", 1)[0]


@pytest.mark.parametrize(
    "change",
    [
        {"severity": "bogus"},
        {"detected_at": "not-a-date"},
        {"triggering_entities": "sensor.gw_ip"},
        {"suggested_actions": "Check it"},
        {"evidence": ["summary"]},
        {"anomaly_id": None, "type": None, "confidence": "x"},
    ],
)
def test_an_unreadable_stored_finding_is_skipped(
    change: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    good = _clients("k1").as_dict()
    members = owed.read_members([good, {**good, **change}])
    assert len(members) == 1
    assert "unreadable owed Sentinel change alert" in caplog.text


def test_a_stored_finding_round_trips() -> None:
    finding = _admins("eve", tokens=("t1",))
    assert anomaly_finding_from_dict(finding.as_dict()) == finding


def test_a_held_back_replay_does_not_say_sentinel_was_silenced() -> None:
    held = owed.held_back(_clients("k1"))
    sent = owed.replay("network_unknown_device_joined", [held])
    assert sent.evidence["summary"].startswith("Seen while its alert was held back (")
    assert "owed_held_back" not in sent.evidence
    silenced = owed.replay("network_unknown_device_joined", [_clients("k1")])
    assert silenced.evidence["summary"].startswith(
        "Seen while Sentinel was at autonomy level 0 ("
    )
