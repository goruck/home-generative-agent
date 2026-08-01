"""Audit storage for sentinel events."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.sentinel.models import (
        AnomalyFinding,
        CompoundFinding,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .models import AuditRecord

LOGGER = logging.getLogger(__name__)

STORE_VERSION = 1
STORE_KEY = "home_generative_agent_audit"

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
    # Issue #262 additions (backward-compatible; same version bucket)
    "triage_decision": None,
    "triage_reason_code": None,
    # action_policy_path was present in async_append_finding but missing from
    # _V2_FIELD_DEFAULTS, so v1→v2 migration did not backfill it.
    "action_policy_path": None,
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


def _is_evictable(record: dict[str, Any]) -> bool:
    """
    Return True when *record* may be evicted to make room for a newer entry.

    Records explicitly marked as user-facing (``suppression_reason_code ==
    "not_suppressed"``) are preserved in preference to suppressed records.
    Records missing the field (e.g. migrated v1 records) are treated as
    evictable — we don't know whether they were user-facing.
    """
    return record.get("suppression_reason_code") != "not_suppressed"


def _relabel_downstream_suppressed(record: dict[str, Any]) -> dict[str, Any]:
    """
    Relabel records mislabeled ``not_suppressed`` before the reason-code fix.

    Findings suppressed by LLM triage or blocked by execution policy were
    historically audited as ``not_suppressed`` even though they never reached
    the user, making them permanently non-evictable and inflating notified
    KPIs.  A genuinely notified record can never carry
    ``triage_decision == "suppress"`` or ``action_policy_path == "blocked"``
    (both engine branches return before notifying), so the relabel is
    deterministic.  Idempotent; runs on every load because affected records
    are already ``version: 2``.
    """
    if record.get("suppression_reason_code") != "not_suppressed":
        return record
    if record.get("triage_decision") == "suppress":
        record["suppression_reason_code"] = "triage_suppressed"
    elif record.get("action_policy_path") == "blocked":
        record["suppression_reason_code"] = "policy_blocked"
    return record


def _migrate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Backfill missing v2 fields into a raw record dict (in-place, returns it)."""
    record_version = record.get("version", 1)
    if record_version < 2:  # noqa: PLR2004
        for field_name, default in _V2_FIELD_DEFAULTS.items():
            record.setdefault(field_name, default)
        record["version"] = 2
    return _relabel_downstream_suppressed(record)


class AuditStore:
    """Persistent audit store for sentinel activity."""

    def __init__(self, hass: HomeAssistant, max_records: int = 500) -> None:
        """Initialize the audit store."""
        self._store = Store(hass, STORE_VERSION, STORE_KEY)
        self._records: list[dict[str, Any]] = []
        self._max_records = max_records

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
        finding: AnomalyFinding | CompoundFinding,
        explanation: str | None,
        *,
        suppression_reason_code: str | None = None,
        trigger_source: str | None = None,
        data_quality: dict[str, Any] | None = None,
        triage_confidence: float | None = None,
        triage_decision: str | None = None,
        triage_reason_code: str | None = None,
        canary_would_execute: bool | None = None,
        execution_id: str | None = None,
        rule_version: str | None = None,
        autonomy_level_at_decision: str | None = None,
        action_policy_path: str | None = None,
        action_outcome: dict[str, Any] | None = None,
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
            action_outcome=action_outcome,
            # v2 fields
            data_quality=data_quality,
            trigger_source=trigger_source,
            suppression_reason_code=suppression_reason_code,
            triage_confidence=triage_confidence,
            triage_decision=triage_decision,
            triage_reason_code=triage_reason_code,
            canary_would_execute=canary_would_execute,
            execution_id=execution_id,
            rule_version=rule_version,
            autonomy_level_at_decision=autonomy_level_at_decision,
            action_policy_path=action_policy_path,
        )
        self._records.append(record.__dict__)
        if len(self._records) > self._max_records:
            self._evict_one()
        await self.async_save()

    def _evict_one(self) -> None:
        """
        Evict one record to keep the store within *_max_records*.

        Eviction priority: prefer dropping the oldest evictable (suppressed)
        record before touching any ``not_suppressed`` record so that
        user-facing KPI history is preserved during high-volume periods.

        # O(n) scan; acceptable at typical store sizes (<=500 records).
        """
        evict_idx = next(
            (i for i, r in enumerate(self._records) if _is_evictable(r)),
            None,
        )
        if evict_idx is not None:
            self._records.pop(evict_idx)
        else:
            # Last resort: every record is not_suppressed.  Evict oldest overall.
            LOGGER.warning(
                "Audit store at capacity (%d records) with no evictable records; "
                "evicting oldest not_suppressed record. "
                "Consider increasing audit_hot_max_records.",
                self._max_records,
            )
            self._records.pop(0)

    async def async_update_response(
        self, anomaly_id: str, response: dict[str, Any], outcome: dict[str, Any] | None
    ) -> None:
        """
        Update the latest record for an anomaly with user response.

        Matches simple findings by ``finding.anomaly_id`` and compound findings
        by any ``finding.constituent_findings[].anomaly_id``, since compound
        notifications are dispatched using the highest-confidence constituent's
        anomaly_id.

        Prefers the newest ``not_suppressed`` match: ``anomaly_id`` is a stable
        content hash, so a re-fired finding that was suppressed downstream
        (triage/policy) can produce a newer record with the same id.  User
        responses always originate from a notification, which only
        ``not_suppressed`` records represent; landing the response there keeps
        it KPI-visible and eviction-protected.  Records missing the reason
        code (v1) may have been notified, so they are preferred over records
        explicitly marked suppressed.  Falls back to the newest match of any
        kind so responses are never dropped.
        """
        notified_match: dict[str, Any] | None = None
        unknown_match: dict[str, Any] | None = None
        suppressed_match: dict[str, Any] | None = None
        for record in reversed(self._records):
            finding = record.get("finding", {})
            if finding.get("anomaly_id") == anomaly_id or any(
                cf.get("anomaly_id") == anomaly_id
                for cf in finding.get("constituent_findings", [])
            ):
                reason_code = record.get("suppression_reason_code")
                if reason_code == "not_suppressed":
                    notified_match = record
                    break
                if reason_code is None:
                    unknown_match = unknown_match or record
                else:
                    suppressed_match = suppressed_match or record
        match = notified_match or unknown_match or suppressed_match
        if match is None:
            return
        match["user_response"] = response
        match["action_outcome"] = outcome
        match.setdefault("notification", {})["responded_at"] = _now_iso()
        await self.async_save()

    async def async_get_latest(self, limit: int) -> list[dict[str, Any]]:
        """Return the latest audit records, newest first."""
        if limit <= 0:
            return []
        return list(reversed(self._records[-limit:]))
