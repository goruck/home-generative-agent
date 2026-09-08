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

from typing import TYPE_CHECKING, Any

from custom_components.home_generative_agent.const import (
    SENTINEL_POSTURE_RULE_COOLDOWN_MINUTES,
)
from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    Severity,
    build_anomaly_id,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def anyone_home(snapshot: FullStateSnapshot) -> bool:
    """Return the derived occupancy flag."""
    return bool(snapshot["derived"].get("anyone_home", False))


def is_night(snapshot: FullStateSnapshot) -> bool:
    """Return the derived night flag."""
    return bool(snapshot["derived"].get("is_night", False))


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    """Return ``"1 token"`` / ``"3 tokens"`` style phrases."""
    word = singular if count == 1 else (plural_form or f"{singular}s")
    return f"{count} {word}"


def make_finding(  # noqa: PLR0913
    rule_id: str,
    *,
    severity: Severity,
    evidence: dict[str, Any],
    summary: str,
    suggested_actions: Sequence[str],
    triggering_entities: Sequence[str] = (),
    confidence: float = 0.9,
) -> AnomalyFinding:
    """
    Build a sensitive finding for a network / HA-security rule.

    ``summary`` is display-only (excluded from the anomaly-id hash) so a
    changing count in the sentence never breaks snooze or cooldown identity.
    """
    entities = list(triggering_entities)
    full_evidence = {**evidence, "summary": summary}
    return AnomalyFinding(
        anomaly_id=build_anomaly_id(rule_id, entities, full_evidence),
        type=rule_id,
        severity=severity,
        confidence=confidence,
        triggering_entities=entities,
        evidence=full_evidence,
        suggested_actions=list(suggested_actions),
        is_sensitive=True,
    )
