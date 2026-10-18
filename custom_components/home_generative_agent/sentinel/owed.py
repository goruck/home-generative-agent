"""
Change alerts held at autonomy level 0, owed until one reaches the user (#799).

A change alert (a new admin or token, a new radio device or router client, a
public IP or UPnP port-mapping change) seen while Sentinel is silenced is kept
here, so it is still told once the level rises even if the change was undone
meanwhile. One entry per kind of change, so a busy kind (a rotating IP, MAC
churn on guest Wi-Fi) neither floods the user afterwards nor pushes out
another kind:

- Posture kinds (a public IP, a UPnP mapping count) keep each change seen,
  as its own value, up to ``MAX_OWED_PER_KIND``.
- Item kinds (devices, clients, accounts) keep up to ``MAX_OWED_PER_KIND``
  findings, replacing any a later one names in full, and are told in one
  alert listing every item.
"""

from __future__ import annotations

import logging
from dataclasses import replace
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
from .rules.network_common import TRUST_ONE_ACTION, TRUST_SEVERAL_ACTION

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

LOGGER = logging.getLogger(__name__)

# Past this, a kind keeps no more findings: the items of later ones are added
# to the latest kept, unlisted, so churn never pushes out the first seen; and
# never another kind's.
MAX_OWED_PER_KIND = 20

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

# Evidence that picks a push's Trust device button. A replay describes the
# past: the device may be gone, and one still present gets its live alert.
_TRUST_EVIDENCE_KEYS = (TRUST_DEVICE_IDS_KEY, "security_device_ids")

# Kept on an owed finding only: when the change was first seen, and the items
# of later findings past the bound (told, but not described).
_FIRST_SEEN = "owed_first_seen"
_NOT_LISTED = "owed_not_listed"

# Marks a replay in its evidence, so the audit trail tells it from a live
# alert and a replay held again by level 0 is not owed a second time.
SEEN_WHILE_SILENCED = "seen_while_silenced"


def items(finding: AnomalyFinding, key: str) -> set[str]:
    """Return the items ``finding`` names under ``key``; none for a bad value."""
    value: object = finding.evidence.get(key)
    if not isinstance(value, list):
        return set()
    return {str(item) for item in cast("list[object]", value)}


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
        # went A to B and back, then to C, changed twice).
        return live.anomaly_id == owed.anomaly_id
    return all(items(owed, key) <= items(live, key) for key in keys)


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
    return replace(finding, severity=top.severity, evidence=evidence)


def _absorb(latest: AnomalyFinding, finding: AnomalyFinding) -> AnomalyFinding:
    """Add ``finding``'s items to ``latest`` as unlisted (the kind is full)."""
    keys = ITEM_KEYS[latest.type]
    new = {i for key in keys for i in items(finding, key) - items(latest, key)}
    evidence = {
        **latest.evidence,
        **{key: sorted(items(latest, key) | items(finding, key)) for key in keys},
        _NOT_LISTED: sorted(items(latest, _NOT_LISTED) | new),
    }
    top = max(latest, finding, key=_rank)
    return replace(latest, severity=top.severity, evidence=evidence)


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
    if len(members) >= MAX_OWED_PER_KIND:
        LOGGER.debug("Owed %s alerts are full; not kept.", finding.type)
        return False
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
                "Sentinel holds %d %s alerts from autonomy level 0; later ones "
                "are counted, not described.",
                MAX_OWED_PER_KIND,
                finding.type,
            )
        members[-1] = _absorb(members[-1], finding)
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
    unlisted = sorted(items(member, _NOT_LISTED) - every)
    return replace(member, evidence={**member.evidence, **left, _NOT_LISTED: unlisted})


def _minute(when: datetime) -> str:
    return dt_util.as_local(when).strftime("%Y-%m-%d %H:%M")


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _trust_free(action: str) -> str:
    # The push has no Trust device button; the service still trusts a device.
    return TRUST_SEVERAL_ACTION if action == TRUST_ONE_ACTION else action


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
        anomaly_id = latest.anomaly_id
    else:
        evidence = {
            key: sorted({item for m in members for item in items(m, key)})
            for key in keys
        }
        singular, plural = _ITEM_NOUNS[kind]
        unlisted = len({i for m in members for i in items(m, _NOT_LISTED)})
        if unlisted:
            noun = singular if unlisted == 1 else plural
            summary += f" {unlisted} more {noun} seen are not described here."
        if kind in INVENTORY_ITEM_KEYS:
            summary += f" {TRUST_SEVERAL_ACTION}."
        total = sum(len(named) for named in evidence.values())
        count = f": {total} {singular if total == 1 else plural}"
        anomaly_id = build_anomaly_id(kind, entities, evidence)
    for key in (*_TRUST_EVIDENCE_KEYS, _FIRST_SEEN, _NOT_LISTED):
        evidence.pop(key, None)
    first = min(_first_seen(m) for m in members)
    seen = _minute(first)
    if _minute(latest.detected_at) != seen:
        seen = f"{seen} to {_minute(latest.detected_at)}"
    note = (
        f"Seen while Sentinel was at autonomy level 0 ({seen}){count}; "
        "it may have been undone since."
    )
    evidence["summary"] = f"{note} {summary}".strip()
    evidence[SEEN_WHILE_SILENCED] = True
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
