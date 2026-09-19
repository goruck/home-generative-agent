"""
On-demand report over the network / Home Assistant security rules.

The Sentinel engine runs these rules on its detection interval and the results
reach the user as notifications, audit rows, and health-sensor attributes. The
``audit_home_security`` agent tool and the ``run_network_audit`` service need
the same facts right now, in one structure, so a user asking "is my home
secure?" gets the live answer rather than whatever the cooldown floor last
let through. ``build_report`` turns one evaluation into that structure; the
engine owns the evaluation itself (``SentinelEngine.async_audit_network``).

Every string here is deterministic. The report states which checks could
not run and why so the agent never claims a check passed when the home
cannot provide the fact it reads (docs/network-security-plan.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

from custom_components.home_generative_agent.snapshot.network import (
    CAP_CLIENTS,
    CAP_NEW_CLIENTS,
    ha_cap,
    posture_cap,
    radio_cap,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from datetime import datetime

    from .models import AnomalyFinding, Severity

AuditStatus = Literal["ok", "disabled", "unavailable"]

_SEVERITY_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}

PRIVACY_NOTES: tuple[str, ...] = (
    (
        "Every fact comes from Home Assistant itself (its auth store, registries, "
        "config flows, Supervisor, runtime settings, its SSDP discovery cache, "
        "the Zigbee, Z-Wave, Bluetooth, and UPnP/IGD integrations, and the "
        "device trackers and settings of a router integration when one is set "
        "up); nothing scanned the network."
    ),
    (
        "Router clients are identified by a per-install pseudonym of their MAC "
        "address; no MAC, IP address, or hostname reaches a language model, and "
        "the device inventory stores the pseudonym and the device's name only. "
        "Router settings are read from eero in this version; checks other "
        "routers would provide are listed as not run rather than assumed to pass."
    ),
    (
        "Radio checks cover configuration only. Attacks on the radio itself (key "
        "sniffing while a device joins, Z-Wave S0 downgrade, Bluetooth pairing "
        "exploits, jamming, or replay) need RF monitoring Home Assistant does not "
        "do, and are not detected."
    ),
)

# Why a capability can be missing on this version, most specific prefix
# first. The reason names the integration or install type that would provide
# it so the user is told what they are not seeing instead of shown a clean
# bill of health.
_CAPABILITY_REASONS: tuple[tuple[str, str], ...] = (
    (
        posture_cap("router_update_pending"),
        (
            "no router integration exposes a firmware update entity (FRITZ!Box, "
            "UniFi, eero, ASUSWRT, TP-Link Omada, Keenetic, MikroTik, NETGEAR, "
            "Freebox, or UPnP)"
        ),
    ),
    (
        ha_cap("addons_"),
        (
            "the Supervisor is not available; add-ons exist only on Home Assistant "
            "OS and Supervised installs"
        ),
    ),
    (
        ha_cap("cloud_remote_ui_enabled"),
        "Home Assistant Cloud is not loaded",
    ),
    (
        ha_cap("new_"),
        (
            "the user and token inventory is not available, so changes cannot be "
            "compared against a previous state"
        ),
    ),
    (
        ha_cap(""),
        (
            "Home Assistant did not return this fact on this version; the log "
            "names the read that failed"
        ),
    ),
    (
        radio_cap("posture.zigbee_permit_join"),
        (
            "needs a Zigbee2MQTT bridge; ZHA does not report whether it is accepting "
            "new devices (its radio library keeps no record of an open join window)"
        ),
    ),
    (
        radio_cap("posture.zwave_inclusion_active"),
        "the Z-Wave JS integration is not loaded or not connected to its server",
    ),
    (
        radio_cap("devices.security_class"),
        "the Z-Wave JS integration is not loaded or not connected to its server",
    ),
    (
        radio_cap("posture.coordinator_update_pending"),
        (
            "no Zigbee or Z-Wave coordinator, Home Assistant radio stick, or "
            "Bluetooth proxy exposes a firmware update entity"
        ),
    ),
    (
        radio_cap("new_devices"),
        (
            "the device inventory is not available, so new devices cannot be "
            "compared against a previous state"
        ),
    ),
    (
        radio_cap(""),
        (
            "Home Assistant did not return the radio device list; the log names "
            "the read that failed"
        ),
    ),
    (
        posture_cap("upnp_enabled"),
        (
            "no UPnP gateway has announced itself to Home Assistant's SSDP discovery "
            "and no UPnP/IGD integration is answering; UPnP is off on the router, "
            "or Home Assistant cannot see the router's network segment"
        ),
    ),
    (
        posture_cap("upnp_port_mappings_added"),
        (
            "needs the UPnP/IGD integration's port-mapping count sensor (disabled "
            "by default) and a previous Sentinel run to compare with"
        ),
    ),
    (
        posture_cap("public_ip_changed"),
        (
            "needs the UPnP/IGD integration's external IP sensor and a previous "
            "Sentinel run to compare with"
        ),
    ),
    (
        CAP_NEW_CLIENTS,
        (
            "needs the Sentinel device inventory to have compared this run's "
            "router clients"
        ),
    ),
    (
        CAP_CLIENTS,
        (
            "needs a router integration whose device trackers report "
            "source_type router (FRITZ!Box, UniFi, eero, ASUSWRT, Nmap, ...); "
            "none is set up"
        ),
    ),
    (
        posture_cap(""),
        (
            "needs a router or DNS integration that exposes this setting; this "
            "version reads it from eero, and other routers follow"
        ),
    ),
)


class NetworkAuditReport(TypedDict):
    """What the on-demand audit returns; serializable as a service response."""

    status: AuditStatus
    generated_at: str
    findings: list[dict[str, Any]]
    checks_run: list[str]
    checks_not_run: dict[str, str]
    capabilities: list[str]
    missing_capabilities: dict[str, str]
    notes: list[str]
    privacy_notes: list[str]
    # Device inventory counts (trusted / untrusted, by source); None when the
    # inventory is not available.
    inventory: dict[str, Any] | None


def capability_reason(capability: str) -> str:
    """Return the deterministic reason a capability is missing."""
    for prefix, reason in _CAPABILITY_REASONS:
        if capability.startswith(prefix):
            return reason
    return "not provided by any adapter in this version"


def _finding_entry(finding: AnomalyFinding) -> dict[str, Any]:
    """Serialize one finding for the report, summary first."""
    payload = finding.as_dict()
    evidence = dict(payload["evidence"])
    summary = evidence.pop("summary", "")
    return {
        "type": payload["type"],
        "severity": payload["severity"],
        "summary": summary,
        "suggested_actions": payload["suggested_actions"],
        "triggering_entities": payload["triggering_entities"],
        "detected_at": payload["detected_at"],
        "evidence": evidence,
    }


def sort_findings(findings: Iterable[AnomalyFinding]) -> list[AnomalyFinding]:
    """Order findings by severity (high first), then by type for stability."""

    def _key(finding: AnomalyFinding) -> tuple[int, str]:
        severity: Severity = finding.severity
        return (_SEVERITY_ORDER.get(severity, len(_SEVERITY_ORDER)), finding.type)

    return sorted(findings, key=_key)


def build_report(  # noqa: PLR0913
    *,
    now: datetime,
    findings: Iterable[AnomalyFinding],
    checks_run: Iterable[str],
    inactive_rules: Mapping[str, list[str]],
    capabilities: Iterable[str],
    failed_rules: Iterable[str] = (),
    notes: Iterable[str] = (),
    inventory: Mapping[str, Any] | None = None,
) -> NetworkAuditReport:
    """
    Assemble the report for one completed evaluation.

    ``inactive_rules`` maps a rule to the capability paths it lacks;
    ``failed_rules`` names rules whose evaluation raised. Both land in
    ``checks_not_run`` with a reason, so the caller never has to encode a
    failure as an empty capability list.
    """
    missing: dict[str, str] = {}
    not_run: dict[str, str] = {}
    for rule_id, paths in sorted(inactive_rules.items()):
        reasons: list[str] = []
        for path in paths:
            reason = capability_reason(path)
            missing[path] = reason
            if reason not in reasons:
                reasons.append(reason)
        not_run[rule_id] = "; ".join(reasons) or "required snapshot data missing"
    for rule_id in sorted(failed_rules):
        not_run[rule_id] = (
            "the check raised an error while evaluating; see the Home Assistant log"
        )
    return {
        "status": "ok",
        "generated_at": now.isoformat(),
        "findings": [_finding_entry(f) for f in sort_findings(findings)],
        "checks_run": sorted(checks_run),
        "checks_not_run": not_run,
        "capabilities": sorted(capabilities),
        "missing_capabilities": missing,
        "notes": list(notes),
        "privacy_notes": list(PRIVACY_NOTES),
        "inventory": _inventory_counts(inventory),
    }


def _inventory_counts(inventory: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Keep only counts: the report never lists device names."""
    if inventory is None:
        return None
    return {
        "trusted": int(inventory.get("trusted", 0)),
        "untrusted": int(inventory.get("untrusted", 0)),
        "by_source": dict(inventory.get("by_source") or {}),
    }


def empty_report(status: AuditStatus, now: datetime, note: str) -> NetworkAuditReport:
    """Return a report for an audit that could not evaluate anything."""
    return {
        "status": status,
        "generated_at": now.isoformat(),
        "findings": [],
        "checks_run": [],
        "checks_not_run": {},
        "capabilities": [],
        "missing_capabilities": {},
        "notes": [note],
        "privacy_notes": list(PRIVACY_NOTES),
        "inventory": None,
    }


def summarize(report: NetworkAuditReport) -> str:
    """One-line English digest: counts by severity plus checks run / not run."""
    counts = {"high": 0, "medium": 0, "low": 0}
    for finding in report["findings"]:
        severity = str(finding.get("severity", ""))
        if severity in counts:
            counts[severity] += 1
    total = len(report["findings"])
    if total == 0:
        head = "No findings"
    else:
        parts = [f"{n} {sev}" for sev, n in counts.items() if n]
        head = f"{total} finding{'s' if total != 1 else ''} ({', '.join(parts)})"
    ran = len(report["checks_run"])
    skipped = len(report["checks_not_run"])
    tail = f"{ran} check{'s' if ran != 1 else ''} ran"
    if skipped:
        tail += f", {skipped} could not run"
    return f"{head}; {tail}."
