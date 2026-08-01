"""Audit models for sentinel events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AuditRecord:
    """
    Persistent audit record.

    Version history
    ---------------
    v1 (original): snapshot_ref, finding, notification, user_response,
        action_outcome
    v2 (issue #254): adds data_quality, trigger_source,
        suppression_reason_code, triage_confidence, canary_would_execute,
        execution_id, rule_version, autonomy_level_at_decision, version
    v2 (issue #262): adds triage_decision, triage_reason_code
        (backward-compatible additions; no version bump needed)
    """

    snapshot_ref: dict[str, Any]
    finding: dict[str, Any]
    notification: dict[str, Any]
    user_response: dict[str, Any] | None
    action_outcome: dict[str, Any] | None

    # --- v2 fields (Section 12) ---
    data_quality: dict[str, Any] | None = field(default=None)
    trigger_source: str | None = field(default=None)
    suppression_reason_code: str | None = field(default=None)
    triage_confidence: float | None = field(default=None)
    canary_would_execute: bool | None = field(default=None)
    execution_id: str | None = field(default=None)
    rule_version: str | None = field(default=None)
    autonomy_level_at_decision: str | None = field(default=None)
    action_policy_path: str | None = field(default=None)

    # --- triage fields (Issue #262) ---
    triage_decision: str | None = field(default=None)
    triage_reason_code: str | None = field(default=None)

    # Schema version — increment when new fields are added
    version: int = field(default=2)
