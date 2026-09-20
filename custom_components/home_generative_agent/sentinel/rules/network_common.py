"""
Shared helpers for the network / Home Assistant security rules.

These rules read ``snapshot["network"]`` (see ``snapshot/network.py``) and
declare the capability paths they need in ``requires``; the engine skips a
rule whose capabilities are missing and reports it as inactive. Every finding
carries an English ``summary`` in its evidence that states the exact facts,
because the generic notification fallback ("<type>: Unknown entity") has no
entity to name for most of these findings.
"""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING, Any

from custom_components.home_generative_agent.const import (
    SENTINEL_POSTURE_RULE_COOLDOWN_MINUTES,
)
from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    Severity,
    build_anomaly_id,
    hashable_evidence,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
        NetworkSnapshot,
    )

# Type keys of every rule in this family; the notifier renders their
# ``summary`` evidence verbatim instead of the generic fallback copy.
NETWORK_RULE_TYPES: frozenset[str] = frozenset(
    {
        "ha_sensitive_entity_exposed_without_pin",
        "ha_new_admin_or_token",
        "ha_long_lived_token_stale",
        "ha_failed_logins",
        "ha_cloud_remote_ui_enabled",
        "ha_addon_exposed_port",
        "ha_addon_unprotected",
        "ha_webhook_automation_public",
        "ha_trusted_networks_bypass_login",
        "ha_http_proxy_misconfigured",
        "security_device_unavailable",
        "network_unconfigured_discovered_device",
        "network_router_update_pending",
        "network_upnp_enabled",
        "network_public_ip_changed",
        "network_upnp_port_mapping_added",
        "radio_new_device_joined",
        "network_unknown_device_joined",
        "network_guest_network_idle",
        "network_wpa3_disabled",
        "network_protection_disabled",
        "network_ddns_enabled",
        "zwave_insecure_security_class",
        "zigbee_permit_join_open",
        "zwave_inclusion_active",
        "radio_coordinator_update_pending",
    }
)

POSTURE_COOLDOWN_MINUTES = SENTINEL_POSTURE_RULE_COOLDOWN_MINUTES


def network_section(snapshot: FullStateSnapshot) -> NetworkSnapshot | None:
    """Return the network section, or None for a pre-v2 snapshot."""
    return snapshot.get("network")


def ha_security(snapshot: FullStateSnapshot) -> dict[str, Any]:
    """Return the ``ha_security`` mapping (empty when absent)."""
    section = network_section(snapshot)
    return dict(section.get("ha_security", {})) if section else {}


def posture(snapshot: FullStateSnapshot) -> dict[str, Any]:
    """Return the ``posture`` mapping (empty when absent)."""
    section = network_section(snapshot)
    return dict(section.get("posture", {})) if section else {}


def clients(snapshot: FullStateSnapshot) -> list[dict[str, Any]]:
    """Return the router client list (empty when absent)."""
    section = network_section(snapshot)
    return [dict(c) for c in (section.get("clients") or [])] if section else []


def new_clients(snapshot: FullStateSnapshot) -> set[str]:
    """Return the client keys the device inventory did not know before this run."""
    section = network_section(snapshot)
    return set(section.get("new_clients") or []) if section else set()


def radio(snapshot: FullStateSnapshot) -> dict[str, Any]:
    """Return the ``radio`` mapping (empty when absent)."""
    section = network_section(snapshot)
    return dict(section.get("radio") or {}) if section else {}


def radio_posture(snapshot: FullStateSnapshot) -> dict[str, Any]:
    """Return the radio ``posture`` mapping (empty when absent)."""
    return dict(radio(snapshot).get("posture") or {})


def is_night(snapshot: FullStateSnapshot) -> bool:
    """Return the derived night flag."""
    return bool(snapshot["derived"].get("is_night", False))


PROTOCOL_LABELS: dict[str, str] = {
    "zigbee": "Zigbee",
    "zwave": "Z-Wave",
    "bluetooth": "Bluetooth",
    "matter": "Matter",
}

# Longest list of names spelled out in one summary before "and N more".
MAX_LISTED_ITEMS = 10


def listed(names: Sequence[str], limit: int = MAX_LISTED_ITEMS) -> str:
    """Join *names*, capping the list with an "and N more" tail."""
    shown = list(names[:limit])
    if len(names) > limit:
        shown.append(f"and {len(names) - limit} more")
    return ", ".join(shown)


def anyone_home(snapshot: FullStateSnapshot) -> bool:
    """Return the derived occupancy flag."""
    return bool(snapshot["derived"].get("anyone_home", False))


def noun(count: int, singular: str, plural_form: str | None = None) -> str:
    """Return the singular or plural noun for *count* without the number."""
    return singular if count == 1 else (plural_form or f"{singular}s")


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    """Return ``"1 token"`` / ``"3 tokens"`` style phrases."""
    return f"{count} {noun(count, singular, plural_form)}"


def _printable(text: str) -> str:
    """Drop control and format characters (bidi overrides, zero-width joiners)."""
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")


def make_finding(  # noqa: PLR0913
    rule_id: str,
    *,
    severity: Severity,
    evidence: dict[str, Any],
    summary: str,
    suggested_actions: Sequence[str],
    triggering_entities: Sequence[str] = (),
    confidence: float = 0.9,
    display: Mapping[str, Any] | None = None,
) -> AnomalyFinding:
    """
    Build a sensitive finding for a network / HA-security rule.

    ``evidence`` is the finding's identity: only keys that name *what* is
    wrong belong there. ``display`` carries figures that change from cycle to
    cycle (days idle, minutes offline, versions) and ``summary`` the rendered
    sentence; both are attached for the notifier and the audit trail but
    excluded from the anomaly-id hash, so pending-prompt and snooze identity
    survive a changing count. ``suggested_actions`` must not contain a dot:
    the engine treats ``a.b`` strings as ``domain.service`` calls.
    """
    entities = list(triggering_entities)
    actions = list(suggested_actions)
    if any("." in action for action in actions):
        msg = f"{rule_id}: suggested actions must not contain '.'"
        raise ValueError(msg)
    full_evidence = {**evidence, **(display or {}), "summary": _printable(summary)}
    return AnomalyFinding(
        anomaly_id=build_anomaly_id(rule_id, entities, hashable_evidence(evidence)),
        type=rule_id,
        severity=severity,
        confidence=confidence,
        triggering_entities=entities,
        evidence=full_evidence,
        suggested_actions=actions,
        is_sensitive=True,
    )
