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
    assert caplog.text.count("later ones are added to the latest") == 1
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


def test_a_replay_has_an_id_of_its_own() -> None:
    """Never a live alert's: the notifier would drop it as a repeat."""
    latest = _finding(
        "network_public_ip_changed",
        {"public_ip_key": "c", "summary": "IP changed."},
        anomaly_id="ip-c",
    )
    sent = owed.replay("network_public_ip_changed", [latest])
    assert sent.anomaly_id != "ip-c"
    assert sent.evidence["public_ip_key"] == "c"
    assert " to " not in sent.evidence["summary"].split(")", 1)[0]
    devices = owed.replay("network_unknown_device_joined", [_clients("k1")])
    assert devices.anomaly_id != _clients("k1").anomaly_id


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


def test_a_mixed_replay_names_both_ways_it_was_held() -> None:
    sent = owed.replay(
        "network_unknown_device_joined",
        [_clients("k1"), owed.held_back(_clients("k2", minute=5))],
    )
    assert sent.evidence["summary"].startswith(
        "Seen while Sentinel was at autonomy level 0 or its alert was held back ("
    )


def test_posture_past_the_cap_keeps_the_latest(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store: dict[str, list[dict[str, Any]]] = {}
    for n in range(owed.MAX_OWED_PER_KIND + 2):
        owed.record(
            store,
            _finding(
                "network_public_ip_changed", {"public_ip_key": f"ip{n}"}, minute=n
            ),
        )
    members = owed.read_members(store["network_public_ip_changed"])
    assert len(members) == owed.MAX_OWED_PER_KIND
    assert members[0].evidence["public_ip_key"] == "ip0"
    assert members[-1].evidence["public_ip_key"] == f"ip{owed.MAX_OWED_PER_KIND + 1}"
    assert "keeping the first and the latest" in caplog.text


def test_unlisted_items_are_bounded() -> None:
    store: dict[str, list[dict[str, Any]]] = {}
    for n in range(owed.MAX_OWED_PER_KIND + owed.MAX_UNLISTED_ITEMS + 10):
        owed.record(store, _clients(f"k{n:04d}", minute=n))
    members = owed.read_members(store["network_unknown_device_joined"])
    unlisted = members[-1].evidence["owed_not_listed"]
    assert len(unlisted) == owed.MAX_UNLISTED_ITEMS


def test_the_posture_cap_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    store: dict[str, list[dict[str, Any]]] = {}
    for n in range(owed.MAX_OWED_PER_KIND + 5):
        owed.record(
            store,
            _finding(
                "network_public_ip_changed", {"public_ip_key": f"ip{n}"}, minute=n
            ),
        )
    assert caplog.text.count("keeping the first and the latest") == 1
    sent = owed.replay(
        "network_public_ip_changed",
        owed.read_members(store["network_public_ip_changed"]),
    )
    assert "owed_capped" not in sent.evidence


def test_a_full_kind_reports_no_change_it_did_not_make() -> None:
    """Past the unlisted bound, only the first overflow is saved (warns once)."""
    store: dict[str, list[dict[str, Any]]] = {}
    for n in range(owed.MAX_OWED_PER_KIND + owed.MAX_UNLISTED_ITEMS):
        owed.record(store, _clients(f"k{n:04d}", minute=n))
    assert owed.record(store, _clients("knew", minute=9999))  # marks it full
    assert not owed.record(store, _clients("knew2", minute=9999))
    told = owed.replay(
        "network_unknown_device_joined",
        owed.read_members(store["network_unknown_device_joined"]),
    )
    assert "knew" not in told.evidence["client_keys"]


def _upnp(count: int, added: int, **kw: Any) -> AnomalyFinding:
    return _finding(
        "network_upnp_port_mapping_added",
        {"count": count, "added": added, "summary": "Mapping opened."},
        entities=["sensor.gw_upnp"],
        **kw,
    )


def test_more_mappings_are_counted_from_the_same_baseline() -> None:
    """The baseline is the count less the mappings added, not a display value."""
    assert owed.covers(_upnp(4, 2), _upnp(3, 1))  # both from 2
    assert not owed.covers(_upnp(4, 1), _upnp(3, 1))  # from 3, not 2
    assert not owed.covers(_upnp(3, 1), _upnp(4, 2))  # fewer
    assert not owed.covers(_upnp(True, 1), _upnp(1, 1))  # type: ignore[arg-type]  # noqa: FBT003


def test_a_change_seen_at_level_zero_stays_level_zero_once_merged() -> None:
    """A later held-back sighting does not make a level-0 one 'held back'."""
    store: dict[str, list[dict[str, Any]]] = {}
    owed.record(store, _clients("k1"))
    owed.record(store, owed.held_back(_clients("k1", "k2", minute=5)))
    (member,) = owed.read_members(store["network_unknown_device_joined"])
    assert not owed.is_held_back(member)
    store = {}
    owed.record(store, owed.held_back(_clients("k1")))
    owed.record(store, owed.held_back(_clients("k1", "k2", minute=5)))
    (member,) = owed.read_members(store["network_unknown_device_joined"])
    assert owed.is_held_back(member)


def test_the_same_change_made_again_gets_a_new_replay_id() -> None:
    """A mapping reopened later is a new replay, not a repeat the notifier drops."""
    first = _upnp(3, 1, anomaly_id="m")
    again = replace(first, detected_at=NOW + timedelta(hours=2))
    kind = "network_upnp_port_mapping_added"
    assert (
        owed.replay(kind, [first]).anomaly_id == owed.replay(kind, [first]).anomaly_id
    )
    assert (
        owed.replay(kind, [first]).anomaly_id != owed.replay(kind, [again]).anomaly_id
    )


def test_a_replay_names_the_owed_findings_it_told() -> None:
    sent = owed.replay(
        "network_unknown_device_joined",
        [_clients("k1", anomaly_id="a"), _clients("k2", anomaly_id="b", minute=5)],
    )
    assert sent.evidence[owed.MEMBER_IDS] == ["a", "b"]


def test_only_keeps_the_items_a_live_alert_names() -> None:
    member = _clients("k1", "k2")
    part = owed.only(member, {"client_keys": {"k2", "k9"}})
    assert part is not None
    assert part.evidence["client_keys"] == ["k2"]
    assert owed.only(member, {"client_keys": {"k9"}}) is None
    ip = _finding("network_public_ip_changed", {"public_ip_key": "b"})
    assert owed.only(ip, {}) is None
    assert owed.merged({"a": {"1"}}, {"a": {"2"}, "b": {"3"}}) == {
        "a": {"1", "2"},
        "b": {"3"},
    }


def _labelled_clients(*keys: str, **kw: Any) -> AnomalyFinding:
    """Clients as the rule reports them since #802: a label and tracker each."""
    base = _clients(*keys, **kw)
    return replace(
        base,
        evidence={
            **base.evidence,
            owed.ITEM_LABELS_KEY: {k: f"Phone {k}" for k in keys},
            owed.ITEM_ENTITIES_KEY: {k: f"device_tracker.{k}" for k in keys},
        },
        triggering_entities=[f"device_tracker.{k}" for k in keys],
    )


def test_a_replay_names_only_the_items_still_owed() -> None:
    """A device told meanwhile is not named again, nor cools the rest (#802)."""
    rest = owed.without(_labelled_clients("k1", "k2"), {"client_keys": {"k1"}})
    assert rest is not None
    assert rest.triggering_entities == ["device_tracker.k2"]
    assert rest.evidence[owed.ITEM_LABELS_KEY] == {"k2": "Phone k2"}

    sent = owed.replay("network_unknown_device_joined", [rest])

    summary = sent.evidence["summary"]
    assert "New device on the network: Phone k2." in summary
    assert "k1" not in summary
    assert sent.triggering_entities == ["device_tracker.k2"]
    # The per-item maps are owed bookkeeping, not part of the alert.
    assert owed.ITEM_LABELS_KEY not in sent.evidence
    assert owed.ITEM_ENTITIES_KEY not in sent.evidence


def test_a_replay_lists_every_described_item_across_members() -> None:
    members = [_labelled_clients("k1", minute=0), _labelled_clients("k2", minute=5)]
    sent = owed.replay("network_unknown_device_joined", members)
    assert "New devices on the network: Phone k1, Phone k2." in sent.evidence["summary"]


def test_an_alert_owed_before_labels_keeps_its_summary() -> None:
    """An untouched alert owed before #802 is told by its stored summary."""
    members = [_labelled_clients("k1"), _clients("k2", minute=5)]
    sent = owed.replay("network_unknown_device_joined", members)
    summary = sent.evidence["summary"]
    assert "New device on the network: Phone k1. New clients: k2." in summary
    assert "not described" not in summary


def test_a_trimmed_alert_owed_before_labels_is_counted_not_named() -> None:
    """Its stored summary would name a told device: the rest is only counted."""
    rest = owed.without(_clients("k1", "k2"), {"client_keys": {"k1"}})
    assert rest is not None
    assert "summary" not in rest.evidence
    summary = owed.replay("network_unknown_device_joined", [rest]).evidence["summary"]
    assert "k1" not in summary
    assert "since. 1 more device seen is not described here." in summary


def test_an_overflow_only_replay_names_no_told_device() -> None:
    """Only absorbed items left, without labels: counted, never the old text."""
    member = replace(
        _labelled_clients("k1"),
        evidence={
            **_labelled_clients("k1").evidence,
            "client_keys": ["k1", "late"],
            "owed_not_listed": ["late"],
        },
    )
    rest = owed.without(member, {"client_keys": {"k1"}})
    assert rest is not None
    summary = owed.replay("network_unknown_device_joined", [rest]).evidence["summary"]
    assert "Phone k1" not in summary
    assert "since. 1 more device seen is not described here." in summary


def test_a_replay_says_what_raised_the_severity() -> None:
    member = replace(
        _labelled_clients("k1"),
        evidence={
            **_labelled_clients("k1").evidence,
            "seen_when": "while nobody was home",
        },
    )
    sent = owed.replay("network_unknown_device_joined", [member])
    assert "It joined while nobody was home." in sent.evidence["summary"]
    assert "seen_when" not in sent.evidence


def test_a_replay_summary_drops_control_characters() -> None:
    member = replace(
        _labelled_clients("k1"),
        evidence={
            **_labelled_clients("k1").evidence,
            owed.ITEM_LABELS_KEY: {"k1": "Phone\u202ek1"},
        },
    )
    sent = owed.replay("network_unknown_device_joined", [member])
    assert "\u202e" not in sent.evidence["summary"]
    assert "Phonek1" in sent.evidence["summary"]


def test_an_account_replay_names_what_is_left_without_labels() -> None:
    """Accounts and tokens are their own labels: the told one is not named."""
    rest = owed.without(
        _admins("alice", tokens=("laptop",)), {"new_admin_users": {"alice"}}
    )
    assert rest is not None
    sent = owed.replay("ha_new_admin_or_token", [rest])
    summary = sent.evidence["summary"]
    assert "New long-lived access token: laptop." in summary
    assert "alice" not in summary


def test_only_keeps_the_maps_and_entities_of_its_items() -> None:
    part = owed.only(_labelled_clients("k1", "k2"), {"client_keys": {"k2"}})
    assert part is not None
    assert part.evidence[owed.ITEM_ENTITIES_KEY] == {"k2": "device_tracker.k2"}
    assert part.triggering_entities == ["device_tracker.k2"]


def test_an_entity_without_an_item_stays_on_the_rest() -> None:
    """Entities not tied to an item are kept: only a told item's entity goes."""
    member = replace(
        _labelled_clients("k1", "k2"),
        triggering_entities=["device_tracker.k1", "device_tracker.k2", "sensor.x"],
    )
    rest = owed.without(member, {"client_keys": {"k1"}})
    assert rest is not None
    assert rest.triggering_entities == ["device_tracker.k2", "sensor.x"]


def test_excluded_items_follow_their_tracker() -> None:
    member = _labelled_clients("k1", "k2")
    gone = owed.excluded_items(member, lambda entity: entity == "device_tracker.k1")
    assert gone == {"client_keys": {"k1"}}
    rest = owed.without(member, gone)
    assert rest is not None
    assert rest.evidence["client_keys"] == ["k2"]
    assert owed.excluded_items(member, lambda _entity: False) == {}
    assert owed.excluded_items(_clients("k3"), lambda _entity: True) == {}


def test_absorbed_items_bring_their_maps() -> None:
    """Items past the cap keep their trackers, so they are excluded alike."""
    store: dict[str, list[dict[str, Any]]] = {}
    for n in range(owed.MAX_OWED_PER_KIND):
        owed.record(store, _labelled_clients(f"k{n}", minute=n))
    owed.record(store, _labelled_clients("late", minute=60))
    latest = owed.read_members(store["network_unknown_device_joined"])[-1]
    assert latest.evidence[owed.ITEM_ENTITIES_KEY]["late"] == "device_tracker.late"
    # Not a triggering entity: its cooldown must not hold back the whole kind.
    assert "device_tracker.late" not in latest.triggering_entities
    # Described when told, since its rule gave a description.
    summary = owed.replay(
        "network_unknown_device_joined",
        owed.read_members(store["network_unknown_device_joined"]),
    ).evidence["summary"]
    assert "not described" not in summary


def test_the_owed_count_is_items_for_item_kinds() -> None:
    stored = [_clients("k1", "k2").as_dict(), _clients("k2", "k3", minute=5).as_dict()]
    assert owed.owed_count("network_unknown_device_joined", stored) == 3
    posture = [_finding("network_public_ip_changed", {"public_ip_key": "c"}).as_dict()]
    assert owed.owed_count("network_public_ip_changed", posture) == 1


def _seen(member: AnomalyFinding, seen_when: str) -> AnomalyFinding:
    return replace(member, evidence={**member.evidence, "seen_when": seen_when})


def test_a_merge_keeps_what_raised_the_severity() -> None:
    """A daytime superset replacing an away sighting still says why it is high."""
    store: dict[str, list[dict[str, Any]]] = {}
    owed.record(
        store,
        _seen(_labelled_clients("k1", severity="high"), "while nobody was home"),
    )
    owed.record(store, _seen(_labelled_clients("k1", "k2", minute=5), ""))
    (member,) = owed.read_members(store["network_unknown_device_joined"])
    assert member.severity == "high"
    sent = owed.replay("network_unknown_device_joined", [member])
    assert "At least one joined while nobody was home." in sent.evidence["summary"]


def test_a_raised_holder_takes_the_new_circumstance() -> None:
    store: dict[str, list[dict[str, Any]]] = {}
    owed.record(store, _seen(_labelled_clients("k1", "k2", severity="low"), ""))
    owed.record(store, _seen(_labelled_clients("k1", severity="medium"), "at night"))
    (member,) = owed.read_members(store["network_unknown_device_joined"])
    assert member.severity == "medium"
    assert member.evidence["seen_when"] == "at night"


def test_an_absorbed_away_sighting_is_kept() -> None:
    store: dict[str, list[dict[str, Any]]] = {}
    for n in range(owed.MAX_OWED_PER_KIND):
        owed.record(store, _seen(_labelled_clients(f"k{n}", minute=n), ""))
    owed.record(
        store,
        _seen(
            _labelled_clients("late", minute=60, severity="high"),
            "while nobody was home",
        ),
    )
    latest = owed.read_members(store["network_unknown_device_joined"])[-1]
    assert latest.severity == "high"
    assert latest.evidence["seen_when"] == "while nobody was home"


def test_an_account_and_a_token_read_as_one_sentence() -> None:
    sent = owed.replay("ha_new_admin_or_token", [_admins("alice", tokens=("laptop",))])
    assert (
        "New administrator account: alice; new long-lived access token: laptop."
        in sent.evidence["summary"]
    )


def test_a_trimmed_merge_drops_a_circumstance_it_can_no_longer_vouch_for() -> None:
    """The away device told meanwhile: the rest is not said to have joined away."""
    store: dict[str, list[dict[str, Any]]] = {}
    owed.record(store, _seen(_labelled_clients("a", "b"), ""))
    owed.record(
        store, _seen(_labelled_clients("b", severity="high"), "while nobody was home")
    )
    (member,) = owed.read_members(store["network_unknown_device_joined"])
    assert member.evidence["seen_when"] == "while nobody was home"
    rest = owed.without(member, {"client_keys": {"b"}})
    assert rest is not None
    summary = owed.replay("network_unknown_device_joined", [rest]).evidence["summary"]
    assert "nobody was home" not in summary


def test_an_equal_severity_sighting_still_adds_its_circumstance() -> None:
    store: dict[str, list[dict[str, Any]]] = {}
    owed.record(store, _seen(_labelled_clients("a", "b", severity="high"), "at night"))
    owed.record(
        store, _seen(_labelled_clients("a", severity="high"), "while nobody was home")
    )
    (member,) = owed.read_members(store["network_unknown_device_joined"])
    assert member.evidence["seen_when"] == "while nobody was home"


def test_every_account_and_token_is_named() -> None:
    """Unlike devices, accounts and tokens are not capped: none can hide."""
    tokens = tuple(f"t{n:02d}" for n in range(12))
    sent = owed.replay("ha_new_admin_or_token", [_admins(tokens=tokens)])
    summary = sent.evidence["summary"]
    assert all(t in summary for t in tokens)
    assert "more" not in summary.split("tokens:", 1)[1]


def test_it_joined_counts_the_undescribed_too() -> None:
    member = _seen(_labelled_clients("a"), "while nobody was home")
    member = replace(
        member,
        evidence={
            **member.evidence,
            "client_keys": ["a", "x", "y"],
            "owed_not_listed": ["x", "y"],
        },
    )
    summary = owed.replay("network_unknown_device_joined", [member]).evidence["summary"]
    assert "At least one joined while nobody was home." in summary
    assert "It joined" not in summary
