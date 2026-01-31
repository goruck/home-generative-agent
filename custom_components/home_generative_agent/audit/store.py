"""Audit storage for sentinel events."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .models import AuditRecord
from ..sentinel.models import AnomalyFinding

STORE_VERSION = 1
STORE_KEY = "home_generative_agent_audit"
MAX_RECORDS = 200


def _snapshot_ref(snapshot: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(snapshot, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return {
        "schema_version": snapshot.get("schema_version"),
        "generated_at": snapshot.get("generated_at"),
        "snapshot_hash": digest,
    }


def _now_iso() -> str:
    return dt_util.as_utc(dt_util.utcnow()).isoformat()


class AuditStore:
    """Persistent audit store for sentinel activity."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._store = Store(hass, STORE_VERSION, STORE_KEY)
        self._records: list[dict[str, Any]] = []

    async def async_load(self) -> None:
        """Load audit records from storage."""
        try:
            data = await self._store.async_load()
        except (HomeAssistantError, OSError, ValueError):
            return
        if isinstance(data, list):
            self._records = list(data)

    async def async_save(self) -> None:
        """Persist audit records to storage."""
        try:
            await self._store.async_save(self._records)
        except (HomeAssistantError, OSError, ValueError):
            return

    async def async_append_finding(
        self, snapshot: dict[str, Any], finding: AnomalyFinding, explanation: str | None
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
