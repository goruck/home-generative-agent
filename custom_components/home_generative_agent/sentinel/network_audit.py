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
    ha_cap,
    posture_cap,
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
        "config flows, Supervisor, and runtime settings); nothing scanned the "
        "network."
    ),
    (
        "Checks that need a router, DNS, or radio integration are not part of this "
        "version and are listed as not run rather than assumed to pass."
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
        CAP_CLIENTS,
        "needs a router integration adapter, which this version does not include",
    ),
    (
        posture_cap(""),
        (
            "needs a router or DNS integration adapter, which this version does not "
            "include"
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
    notes: Iterable[str] = (),
) -> NetworkAuditReport:
    """Assemble the report for one completed evaluation."""
    missing: dict[str, str] = {}
    not_run: dict[str, str] = {}
    for rule_id, paths in sorted(inactive_rules.items()):
        reasons: list[str] = []
        for path in paths:
            reason = capability_reason(path)
            missing[path] = reason
            if reason not in reasons:
                reasons.append(reason)
        not_run[rule_id] = "; ".join(reasons) or (
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
