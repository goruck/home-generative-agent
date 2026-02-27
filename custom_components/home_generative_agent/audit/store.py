"""Audit storage for sentinel events."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .models import AuditRecord

STORE_VERSION = 1
STORE_KEY = "home_generative_agent_audit"
MAX_RECORDS = 200

# v2 fields that must be present after migration; maps field name -> default
_V2_FIELD_DEFAULTS: dict[str, Any] = {
    "data_quality": None,
    "trigger_source": None,
    "suppression_reason_code": None,
    "triage_confidence": None,
    "canary_would_execute": None,
    "execution_id": None,
    "rule_version": None,
    "autonomy_level_at_decision": None,
}


def _snapshot_ref(snapshot: FullStateSnapshot) -> dict[str, Any]:
    payload = json.dumps(snapshot, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return {
        "schema_version": snapshot.get("schema_version"),
        "generated_at": snapshot.get("generated_at"),
        "snapshot_hash": digest,
    }


def _now_iso() -> str:
    return dt_util.as_utc(dt_util.utcnow()).isoformat()


def _migrate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Backfill missing v2 fields into a raw record dict (in-place, returns it)."""
    record_version = record.get("version", 1)
    if record_version < 2:  # noqa: PLR2004
        for field_name, default in _V2_FIELD_DEFAULTS.items():
            record.setdefault(field_name, default)
        record["version"] = 2
    return record


class AuditStore:
    """Persistent audit store for sentinel activity."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the audit store."""
        self._store = Store(hass, STORE_VERSION, STORE_KEY)
        self._records: list[dict[str, Any]] = []

    async def async_load(self) -> None:
        """Load audit records from storage, migrating old records on the fly."""
        try:
            data = await self._store.async_load()
        except (HomeAssistantError, OSError, ValueError):
            return
        if isinstance(data, list):
            self._records = [_migrate_record(r) for r in data]

    async def async_save(self) -> None:
        """Persist audit records to storage."""
        try:
            await self._store.async_save(self._records)
        except (HomeAssistantError, OSError, ValueError):
            return

    async def async_append_finding(  # noqa: PLR0913
        self,
        snapshot: FullStateSnapshot,
        finding: AnomalyFinding,
        explanation: str | None,
        *,
        suppression_reason_code: str | None = None,
        trigger_source: str | None = None,
        data_quality: dict[str, Any] | None = None,
        triage_confidence: float | None = None,
        canary_would_execute: bool | None = None,
        execution_id: str | None = None,
        rule_version: str | None = None,
        autonomy_level_at_decision: str | None = None,
        action_policy_path: str | None = None,
    ) -> None:
        """Append a finding audit record."""
        record = AuditRecord(
            snapshot_ref=_snapshot_ref(snapshot),
            finding=finding.as_dict(),
            notification={
                "explanation": explanation,
                "notified_at": _now_iso(),
            },
            user_response=None,
            action_outcome=None,
            # v2 fields
            data_quality=data_quality,
            trigger_source=trigger_source,
            suppression_reason_code=suppression_reason_code,
            triage_confidence=triage_confidence,
            canary_would_execute=canary_would_execute,
            execution_id=execution_id,
            rule_version=rule_version,
            autonomy_level_at_decision=autonomy_level_at_decision,
            action_policy_path=action_policy_path,
        )
        self._records.append(record.__dict__)
        if len(self._records) > MAX_RECORDS:
            self._records = self._records[-MAX_RECORDS:]
        await self.async_save()

    async def async_update_response(
        self, anomaly_id: str, response: dict[str, Any], outcome: dict[str, Any] | None
    ) -> None:
        """Update the latest record for an anomaly with user response."""
        for record in reversed(self._records):
            finding = record.get("finding", {})
            if finding.get("anomaly_id") == anomaly_id:
                record["user_response"] = response
                record["action_outcome"] = outcome
                record.setdefault("notification", {})["responded_at"] = _now_iso()
                await self.async_save()
                return

    async def async_get_latest(self, limit: int) -> list[dict[str, Any]]:
        """Return the latest audit records, newest first."""
        if limit <= 0:
            return []
        return list(reversed(self._records[-limit:]))
