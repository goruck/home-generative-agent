"""Audit models for sentinel events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuditRecord:
    """Persistent audit record."""

    snapshot_ref: dict[str, Any]
    finding: dict[str, Any]
    notification: dict[str, Any]
    user_response: dict[str, Any] | None
    action_outcome: dict[str, Any] | None
