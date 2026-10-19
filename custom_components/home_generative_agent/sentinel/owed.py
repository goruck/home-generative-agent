"""
Change alerts held back, owed until one reaches the user (#799, #801).

A change alert (a new admin or token, a new radio device or router client, a
public IP or UPnP port-mapping change) held at autonomy level 0, or held back
by a cooldown, quiet hours or a pending prompt at any level, is kept here, so
it is still told once it can be, even if the change was undone meanwhile. One
entry per kind of change, so a busy kind (a rotating IP, MAC churn on guest
Wi-Fi) neither floods the user afterwards nor pushes out another kind:

- Posture kinds (a public IP, a UPnP mapping count) keep each change seen,
  as its own value (more port mappings from the same baseline replace fewer),
  up to ``MAX_OWED_PER_KIND``: the first ones and the latest.
- Item kinds (devices, clients, accounts) keep up to ``MAX_OWED_PER_KIND``
  findings, replacing any a later one names in full, and are told in one
  alert listing every item.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from homeassistant.util import dt as dt_util

from .models import (
    SEVERITY_RANK,
    TRUST_DEVICE_IDS_KEY,
    AnomalyFinding,
    anomaly_finding_from_dict,
    build_anomaly_id,
)
from .network_inventory import client_key
from .rules.network_common import TRUST_ONE_ACTION, TRUST_SEVERAL_ACTION, listed

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

LOGGER = logging.getLogger(__name__)

# Past this, a kind keeps no more findings: a posture kind replaces its latest
# with the newest, and an item kind adds later items to its latest, unlisted,
# up to MAX_UNLISTED_ITEMS; churn never pushes out the first seen, nor another
# kind's.
MAX_OWED_PER_KIND = 20
# Past this many unlisted items, more are not kept.
MAX_UNLISTED_ITEMS = 500

# An owed kind whose rule has not run for this many days is logged once, and
# dropped (audited) after the second: its rule may never run again (network
# checks turned off, the integration removed), and a replay months later would
# describe a home that has moved on (#802). A rule that runs but is held (level
# 0, a cooldown) does not count as idle.
OWED_IDLE_WARN_DAYS = 7
OWED_IDLE_EXPIRE_DAYS = 30

# The list evidence an item kind names its items by. A finding of the kind
# still reports an owed one while it names every item that one named.
ITEM_KEYS: dict[str, tuple[str, ...]] = {
    "ha_new_admin_or_token": ("new_admin_users", "new_long_lived_tokens"),
    "radio_new_device_joined": ("device_keys",),
    "network_unknown_device_joined": ("client_keys",),
}

# The item key naming device inventory rows, for the kinds that have them.
INVENTORY_ITEM_KEYS: dict[str, str] = {
    "radio_new_device_joined": "device_keys",
    "network_unknown_device_joined": "client_keys",
}

# What an item kind's items are called in a replay's note.
_ITEM_NOUNS: dict[str, tuple[str, str]] = {
    "ha_new_admin_or_token": ("account or token", "accounts or tokens"),
    "radio_new_device_joined": ("device", "devices"),
    "network_unknown_device_joined": ("device", "devices"),
}

# How a replay lists each item key's items when it can describe every one
# (#802): (singular, plural) heading. An account or token is its own label;
# devices and clients are described by ITEM_LABELS from their rule.
_ITEM_HEADINGS: dict[str, tuple[str, str]] = {
    "new_admin_users": ("New administrator account", "New administrator accounts"),
    "new_long_lived_tokens": (
        "New long-lived access token",
        "New long-lived access tokens",
    ),
    "device_keys": ("New radio device", "New radio devices"),
    "client_keys": ("New device on the network", "New devices on the network"),
}
_SELF_LABELLED_KEYS = frozenset({"new_admin_users", "new_long_lived_tokens"})

# Display evidence an item kind's rule sets (#802): how each item is described
# in a summary, and the entity each item is tracked by, if any. A replay for
# part of an owed alert is told, cooled down and excluded by its own items.
ITEM_LABELS = "item_labels"
ITEM_ENTITIES = "item_entities"

# Evidence that picks a push's Trust device button. A replay describes the
# past: the device may be gone, and one still present gets its live alert.
_TRUST_EVIDENCE_KEYS = (TRUST_DEVICE_IDS_KEY, "security_device_ids")

# Kept on an owed finding only: when the change was first seen, and the items
# of later findings past the bound (told, but not described).
_FIRST_SEEN = "owed_first_seen"
# The latest member of a posture kind past the bound, replaced by each new one.
_CAPPED = "owed_capped"
_NOT_LISTED = "owed_not_listed"

# Marks a replay in its evidence (held at level 0 or held back), so the audit
# trail tells it from a live alert and a replay held again is not owed a
# second time. The key name predates #801.
SEEN_WHILE_SILENCED = "seen_while_silenced"
# Kept on an owed finding held back above level 0 (a cooldown, quiet hours,
# a pending prompt), not silenced: its replay says so (#801).
_HELD_BACK = "owed_held_back"
# Set on the latest member of an item kind once MAX_UNLISTED_ITEMS is reached.
_FULL = "owed_full"
# On a replay: the ids of the owed findings it told. A live finding with one
# of them is the same change, told by the replay.
MEMBER_IDS = "owed_member_ids"


def items(finding: AnomalyFinding, key: str) -> set[str]:
    """Return the items ``finding`` names under ``key``; none for a bad value."""
    value: object = finding.evidence.get(key)
    if not isinstance(value, list):
        return set()
    return {str(item) for item in cast("list[object]", value)}


def _item_map(finding: AnomalyFinding, key: str) -> dict[str, str]:
    """Return a per-item display map (ITEM_LABELS, ITEM_ENTITIES); none if bad."""
    value: object = finding.evidence.get(key)
    if not isinstance(value, dict):
        return {}
    return {
        str(item): str(text)
        for item, text in cast("dict[object, object]", value).items()
        if text
    }


def _restricted(
    member: AnomalyFinding, keep: dict[str, list[str]], unlisted: list[str]
) -> AnomalyFinding:
    """
    Return ``member`` naming only the items in ``keep``.

    Its per-item labels and entities follow, and so do its triggering
    entities: an entity tracking an item no longer named goes, so that
    device's cooldown or exclusion no longer holds the rest (#802).
    """
    names = {item for kept in keep.values() for item in kept} | set(unlisted)
    labels = _item_map(member, ITEM_LABELS)
    entities = _item_map(member, ITEM_ENTITIES)
    dropped = {entity for item, entity in entities.items() if item not in names}
    kept_entities = {entities[item] for item in names if item in entities}
    evidence = {**member.evidence, **keep, _NOT_LISTED: unlisted}
    if labels:
        evidence[ITEM_LABELS] = {i: t for i, t in labels.items() if i in names}
    if entities:
        evidence[ITEM_ENTITIES] = {i: e for i, e in entities.items() if i in names}
    return replace(
        member,
        evidence=evidence,
        triggering_entities=[
            e
            for e in member.triggering_entities
            if e not in dropped or e in kept_entities
        ],
    )


def idle_since(last_ran: str | None, members: list[AnomalyFinding]) -> datetime:
    """
    Return when an owed kind's rule was last seen running.

    The day saved when it last ran while owed, or else (owed by a build
    before #802) when its oldest alert was first seen.
    """
    try:
        day = dt_util.parse_date(last_ran) if last_ran else None
    except ValueError:
        day = None
    if day is not None:
        # The day was saved in UTC: its UTC midnight.
        return datetime(day.year, day.month, day.day, tzinfo=UTC)
    return min((_first_seen(m) for m in members), default=dt_util.utcnow())


def excluded_items(
    member: AnomalyFinding, is_excluded: Callable[[str], bool]
) -> dict[str, set[str]]:
    """Return the items of ``member`` whose tracking entity ``is_excluded``."""
    keys = ITEM_KEYS.get(member.type, ())
    gone = {
        item
        for item, entity in _item_map(member, ITEM_ENTITIES).items()
        if is_excluded(entity)
    }
    return dict.fromkeys(keys, gone) if gone else {}


def told_items(kind: str, findings: Iterable[AnomalyFinding]) -> dict[str, set[str]]:
    """Return the items of ``kind`` that ``findings`` reported, by key."""
    return {
        key: {item for f in findings for item in items(f, key)}
        for key in ITEM_KEYS.get(kind, ())
    }


def inventory_key(kind: str, item: str) -> str:
    """Return the device inventory row key for an item of ``kind``."""
    return client_key(item) if kind == "network_unknown_device_joined" else item


def covers(live: AnomalyFinding, owed: AnomalyFinding) -> bool:
    """Return True when ``live`` reports everything ``owed`` reported."""
    if live.type != owed.type:
        return False
    keys = ITEM_KEYS.get(owed.type)
    if keys is None:
        # A posture change is one value (an address, a mapping count): another
        # value is another change, even from the same baseline (an IP that
        # went A to B and back, then to C, changed twice). Except more port
        # mappings counted from the same baseline, which include the fewer.
        return live.anomaly_id == owed.anomaly_id or _more_mappings(live, owed)
    return all(items(owed, key) <= items(live, key) for key in keys)


def _upnp_counts(finding: AnomalyFinding) -> tuple[int, int] | None:
    """Return a UPnP finding's (baseline, count), or None when not both known."""
    count, added = finding.evidence.get("count"), finding.evidence.get("added")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (count, added)):
        return None
    return cast("int", count) - cast("int", added), cast("int", count)


def _more_mappings(live: AnomalyFinding, owed: AnomalyFinding) -> bool:
    """Return True for a UPnP count at least as high from the same baseline."""
    if owed.type != "network_upnp_port_mapping_added":
        return False
    live_counts, owed_counts = _upnp_counts(live), _upnp_counts(owed)
    return (
        live_counts is not None
        and owed_counts is not None
        and live_counts[0] == owed_counts[0]
        and live.triggering_entities == owed.triggering_entities
        and live_counts[1] >= owed_counts[1]
    )


def read_members(stored: object) -> list[AnomalyFinding]:
    """Return the findings stored for one kind, oldest first, skipping bad ones."""
    members: list[AnomalyFinding] = []
    for data in cast("list[object]", stored) if isinstance(stored, list) else []:
        try:
            members.append(anomaly_finding_from_dict(cast("dict[str, Any]", data)))
        except (KeyError, TypeError, ValueError):
            LOGGER.warning("Dropping an unreadable owed Sentinel change alert.")
    return sorted(members, key=lambda finding: finding.detected_at)


def _first_seen(finding: AnomalyFinding) -> datetime:
    stored = finding.evidence.get(_FIRST_SEEN)
    try:
        parsed = dt_util.parse_datetime(stored) if isinstance(stored, str) else None
    except ValueError:
        parsed = None
    return dt_util.as_utc(parsed) if parsed else finding.detected_at


def held_back(finding: AnomalyFinding) -> AnomalyFinding:
    """Return ``finding`` marked as held back above level 0, not silenced."""
    return replace(finding, evidence={**finding.evidence, _HELD_BACK: True})


def is_held_back(finding: AnomalyFinding) -> bool:
    """Return True for an owed finding held back above level 0 only."""
    return bool(finding.evidence.get(_HELD_BACK))


def _rank(finding: AnomalyFinding) -> int:
    return SEVERITY_RANK[finding.severity]


def _carry(finding: AnomalyFinding, older: list[AnomalyFinding]) -> AnomalyFinding:
    """Return ``finding`` with what ``older`` kept: severity, sightings."""
    if not older:
        return finding
    first = min([finding, *older], key=_first_seen)
    top = max([finding, *older], key=_rank)
    unlisted = sorted({i for m in [finding, *older] for i in items(m, _NOT_LISTED)})
    evidence = {**finding.evidence, _FIRST_SEEN: _first_seen(first).isoformat()}
    if unlisted:
        evidence[_NOT_LISTED] = unlisted
    # Held back only if every sighting was: one seen at level 0 says so.
    if all(is_held_back(m) for m in [finding, *older]):
        evidence[_HELD_BACK] = True
    else:
        evidence.pop(_HELD_BACK, None)
    return replace(finding, severity=top.severity, evidence=evidence)


def _absorb(latest: AnomalyFinding, finding: AnomalyFinding) -> AnomalyFinding:
    """Add ``finding``'s items to ``latest`` as unlisted (the kind is full)."""
    keys = ITEM_KEYS[latest.type]
    new = {i for key in keys for i in items(finding, key) - items(latest, key)}
    if not new:
        return latest
    if len(items(latest, _NOT_LISTED)) + len(new) > MAX_UNLISTED_ITEMS:
        # Bounded, so churn cannot grow the saved state without end; the
        # first ones are kept and named.
        if latest.evidence.get(_FULL):
            return latest
        LOGGER.warning(
            "Sentinel holds %d unlisted owed %s items; later ones are not kept.",
            MAX_UNLISTED_ITEMS,
            latest.type,
        )
        return replace(latest, evidence={**latest.evidence, _FULL: True})
    evidence = {
        **latest.evidence,
        **{key: sorted(items(latest, key) | items(finding, key)) for key in keys},
        _NOT_LISTED: sorted(items(latest, _NOT_LISTED) | new),
    }
    for key in (ITEM_LABELS, ITEM_ENTITIES):
        merged_map = {**_item_map(finding, key), **_item_map(latest, key)}
        if merged_map:
            evidence[key] = merged_map
    top = max(latest, finding, key=_rank)
    return replace(
        latest,
        severity=top.severity,
        evidence=evidence,
        triggering_entities=_unique(
            [*latest.triggering_entities, *finding.triggering_entities]
        ),
    )


def record(owed: dict[str, list[dict[str, Any]]], finding: AnomalyFinding) -> bool:
    """
    Owe ``finding``; return True when what is owed changed.

    For an item kind, a later finding naming every item an owed one named
    replaces it, but keeps its first sighting and highest severity (a device
    that joined while nobody was home stays high). Past ``MAX_OWED_PER_KIND``
    a kind keeps no more findings: an item kind adds their items to the latest
    kept, as unlisted, and a posture kind drops them; churn never pushes out
    the change seen first.
    """
    stored = owed.get(finding.type, [])
    members = read_members(stored)
    changed = (
        _record_items(members, finding)
        if finding.type in ITEM_KEYS
        else _record_posture(members, finding)
    )
    if not changed and len(members) == len(stored):
        return False
    owed[finding.type] = [member.as_dict() for member in members]
    return True


def _record_posture(members: list[AnomalyFinding], finding: AnomalyFinding) -> bool:
    if any(covers(m, finding) for m in members):
        return False
    older = [m for m in members if covers(finding, m)]
    if older:
        # More port mappings from the same baseline: the new count reports
        # the earlier one too.
        members[:] = [m for m in members if m not in older]
        members.append(_carry(finding, older))
        return True
    if len(members) >= MAX_OWED_PER_KIND:
        # Keep the first changes and the latest; churn (a LAN host opening
        # and closing mappings) cannot make the latest go untold.
        if not members[-1].evidence.get(_CAPPED):
            LOGGER.warning(
                "Sentinel holds %d owed %s alerts; keeping the first and the latest.",
                MAX_OWED_PER_KIND,
                finding.type,
            )
        members[-1] = replace(finding, evidence={**finding.evidence, _CAPPED: True})
        return True
    members.append(finding)
    return True


def _record_items(members: list[AnomalyFinding], finding: AnomalyFinding) -> bool:
    holder = next((m for m in members if covers(m, finding)), None)
    if holder is not None:
        if _rank(finding) <= _rank(holder):
            return False
        members[members.index(holder)] = replace(holder, severity=finding.severity)
        return True
    older = [m for m in members if covers(finding, m)]
    if not older and len(members) >= MAX_OWED_PER_KIND:
        if not items(members[-1], _NOT_LISTED):
            LOGGER.warning(
                "Sentinel holds %d owed %s alerts; later ones are counted, "
                "not described.",
                MAX_OWED_PER_KIND,
                finding.type,
            )
        absorbed = _absorb(members[-1], finding)
        if absorbed is members[-1]:
            return False  # nothing new, or past MAX_UNLISTED_ITEMS
        members[-1] = absorbed
        return True
    members[:] = [m for m in members if m not in older]
    members.append(_carry(finding, older))
    return True


def without(member: AnomalyFinding, told: dict[str, set[str]]) -> AnomalyFinding | None:
    """
    Return ``member`` less the items already ``told``, or None if none are left.

    An owed finding partly reported by a live alert (or a device trusted
    since) is replayed for the rest alone. A posture finding is returned whole.
    """
    keys = ITEM_KEYS.get(member.type)
    if keys is None:
        return member
    left = {key: sorted(items(member, key) - told.get(key, set())) for key in keys}
    if not any(left.values()):
        return None
    if all(len(left[key]) == len(items(member, key)) for key in keys):
        return member
    every = {i for told_set in told.values() for i in told_set}
    return _restricted(member, left, sorted(items(member, _NOT_LISTED) - every))


def only(member: AnomalyFinding, told: dict[str, set[str]]) -> AnomalyFinding | None:
    """
    Return ``member`` with only the items in ``told``, or None if it has none.

    The part a waiting live alert names stays owed apart, settled with it. A
    posture finding has no items: None.
    """
    keys = ITEM_KEYS.get(member.type)
    if keys is None:
        return None
    kept = {key: sorted(items(member, key) & told.get(key, set())) for key in keys}
    if not any(kept.values()):
        return None
    every = {i for told_set in told.values() for i in told_set}
    return _restricted(member, kept, sorted(items(member, _NOT_LISTED) & every))


def merged(*told: dict[str, set[str]]) -> dict[str, set[str]]:
    """Return the union of item maps, by key."""
    out: dict[str, set[str]] = {}
    for one in told:
        for key, values in one.items():
            out[key] = out.get(key, set()) | values
    return out


def _replay_id(kind: str, entities: list[str], members: list[AnomalyFinding]) -> str:
    """
    Return an id for a replay of its own, never a live alert's.

    The notifier drops a repeat of an id it pushed lately, and a pending prompt
    is keyed by it: a replay sharing a live alert's id (a port mapping reopened
    with the same count) would be dropped as told while never shown.
    """
    # With when each was first seen: the same change made again later (a
    # mapping reopened) is told again, not dropped as the earlier replay.
    return build_anomaly_id(
        kind,
        entities,
        {
            "owed_replay": sorted(
                f"{m.anomaly_id}@{_first_seen(m).isoformat()}" for m in members
            )
        },
    )


def _minute(when: datetime) -> str:
    return dt_util.as_local(when).strftime("%Y-%m-%d %H:%M")


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _trust_free(action: str) -> str:
    # The push has no Trust device button; the service still trusts a device.
    return TRUST_SEVERAL_ACTION if action == TRUST_ONE_ACTION else action


def _described(kind: str, members: list[AnomalyFinding]) -> str | None:
    """
    Return a summary naming each item ``members`` still list, or None.

    Built from the items themselves, not the summaries the findings were
    seen with, so items told meanwhile (a live alert, a trust) are not named
    again (#802). None when an item has no description (an alert owed by a
    build before #802): the stored summaries are used then.
    """
    labels: dict[str, str] = {}
    for member in members:
        labels.update(_item_map(member, ITEM_LABELS))
    unlisted = {i for m in members for i in items(m, _NOT_LISTED)}
    parts: list[str] = []
    for key in ITEM_KEYS[kind]:
        named = sorted({i for m in members for i in items(m, key)} - unlisted)
        if not named:
            continue
        if key in _SELF_LABELLED_KEYS:
            described = named
        elif all(item in labels for item in named):
            described = sorted(labels[item] for item in named)
        else:
            return None
        singular, plural = _ITEM_HEADINGS[key]
        heading = singular if len(named) == 1 else plural
        parts.append(f"{heading}: {listed(described)}")
    if not parts:
        return None
    return "; ".join(parts) + "."


def replay(kind: str, members: list[AnomalyFinding]) -> AnomalyFinding:
    """
    Return the one alert that tells ``members`` (oldest first), marked as old.

    The note comes first, with how many items it names: a push keeps only the
    start of a long summary.
    """
    latest = members[-1]
    keys = ITEM_KEYS.get(kind)
    evidence: dict[str, Any]
    count = ""
    entities = sorted({e for m in members for e in m.triggering_entities})
    summary = " ".join(
        _unique(str(m.evidence.get("summary") or "").strip() for m in members)
    )
    if keys is None:
        evidence = dict(latest.evidence)
        anomaly_id = _replay_id(kind, entities, members)
    else:
        evidence = {
            key: sorted({item for m in members for item in items(m, key)})
            for key in keys
        }
        summary = _described(kind, members) or summary
        singular, plural = _ITEM_NOUNS[kind]
        unlisted = len({i for m in members for i in items(m, _NOT_LISTED)})
        if unlisted:
            noun = singular if unlisted == 1 else plural
            summary += f" {unlisted} more {noun} seen are not described here."
        if kind in INVENTORY_ITEM_KEYS:
            summary += f" {TRUST_SEVERAL_ACTION}."
        total = sum(len(named) for named in evidence.values())
        count = f": {total} {singular if total == 1 else plural}"
        anomaly_id = _replay_id(kind, entities, members)
    for key in (
        *_TRUST_EVIDENCE_KEYS,
        _FIRST_SEEN,
        _NOT_LISTED,
        _HELD_BACK,
        _CAPPED,
        _FULL,
        ITEM_LABELS,
        ITEM_ENTITIES,
    ):
        evidence.pop(key, None)
    first = min(_first_seen(m) for m in members)
    seen = _minute(first)
    if _minute(latest.detected_at) != seen:
        seen = f"{seen} to {_minute(latest.detected_at)}"
    held = [is_held_back(m) for m in members]
    if all(held):
        where = "its alert was held back"
    elif any(held):
        where = "Sentinel was at autonomy level 0 or its alert was held back"
    else:
        where = "Sentinel was at autonomy level 0"
    note = f"Seen while {where} ({seen}){count}; it may have been undone since."
    evidence["summary"] = f"{note} {summary}".strip()
    evidence[SEEN_WHILE_SILENCED] = True
    evidence[MEMBER_IDS] = sorted({m.anomaly_id for m in members})
    return AnomalyFinding(
        anomaly_id=anomaly_id,
        type=kind,
        severity=max(members, key=_rank).severity,
        confidence=max(m.confidence for m in members),
        triggering_entities=entities,
        evidence=evidence,
        suggested_actions=_unique(
            _trust_free(a) for m in members for a in m.suggested_actions
        ),
        is_sensitive=True,
        detected_at=latest.detected_at,
    )
