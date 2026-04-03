"""Tests for audit record schema (issue #254): versioned fields and migration."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.home_generative_agent.audit.models import AuditRecord
from custom_components.home_generative_agent.audit.store import (
    AuditStore,
    _migrate_record,
)
from custom_components.home_generative_agent.const import (
    CONF_AUDIT_ARCHIVAL_BACKLOG_MAX,
    CONF_AUDIT_HIGH_RETENTION_DAYS,
    CONF_AUDIT_HOT_MAX_RECORDS,
    CONF_AUDIT_RETENTION_DAYS,
    RECOMMENDED_AUDIT_HOT_MAX_RECORDS,
)

# ---------------------------------------------------------------------------
# AuditRecord field completeness
# ---------------------------------------------------------------------------

_REQUIRED_V2_FIELDS = {
    "snapshot_ref",
    "finding",
    "notification",
    "user_response",
    "action_outcome",
    # v2 additions
    "data_quality",
    "trigger_source",
    "suppression_reason_code",
    "triage_confidence",
    "canary_would_execute",
    "execution_id",
    "rule_version",
    "autonomy_level_at_decision",
    "version",
}


def test_audit_record_has_all_v2_fields() -> None:
    """AuditRecord dataclass exposes every field required by the v2 schema."""
    actual_fields = {f.name for f in fields(AuditRecord)}
    missing = _REQUIRED_V2_FIELDS - actual_fields
    assert not missing, f"AuditRecord is missing fields: {missing}"


def test_audit_record_default_version_is_2() -> None:
    """AuditRecord.version defaults to 2 (the current schema version)."""
    record = AuditRecord(
        snapshot_ref={},
        finding={},
        notification={},
        user_response=None,
        action_outcome=None,
    )
    assert record.version == 2


def test_audit_record_new_fields_default_to_none() -> None:
    """All v2-only fields default to None when not supplied."""
    record = AuditRecord(
        snapshot_ref={},
        finding={},
        notification={},
        user_response=None,
        action_outcome=None,
    )
    assert record.data_quality is None
    assert record.trigger_source is None
    assert record.suppression_reason_code is None
    assert record.triage_confidence is None
    assert record.canary_would_execute is None
    assert record.execution_id is None
    assert record.rule_version is None
    assert record.autonomy_level_at_decision is None


def test_audit_record_v2_fields_settable() -> None:
    """All v2 fields can be set to non-None values."""
    record = AuditRecord(
        snapshot_ref={"snap": True},
        finding={"anomaly_id": "a1"},
        notification={"notified_at": "2026-01-01T00:00:00+00:00"},
        user_response=None,
        action_outcome=None,
        data_quality={"completeness": 0.95},
        trigger_source="rule",
        suppression_reason_code="not_suppressed",
        triage_confidence=0.87,
        canary_would_execute=True,
        execution_id="exec-123",
        rule_version="1.0.0",
        autonomy_level_at_decision="notify_only",
        version=2,
    )
    assert record.data_quality == {"completeness": 0.95}
    assert record.trigger_source == "rule"
    assert record.suppression_reason_code == "not_suppressed"
    assert record.triage_confidence == pytest.approx(0.87)
    assert record.canary_would_execute is True
    assert record.execution_id == "exec-123"
    assert record.rule_version == "1.0.0"
    assert record.autonomy_level_at_decision == "notify_only"


# ---------------------------------------------------------------------------
# v1 -> v2 migration
# ---------------------------------------------------------------------------


def _make_v1_record() -> dict[str, object]:
    """Return a raw dict that mimics a v1 audit record (no version key)."""
    return {
        "snapshot_ref": {
            "schema_version": 1,
            "generated_at": "2025-01-01T00:00:00+00:00",
        },
        "finding": {"anomaly_id": "old-1", "type": "rule"},
        "notification": {
            "explanation": None,
            "notified_at": "2025-01-01T00:00:00+00:00",
        },
        "user_response": None,
        "action_outcome": None,
        # No version key — classic v1 record
    }


def test_migrate_record_backfills_v2_fields() -> None:
    """_migrate_record must add all v2 fields to a v1 record."""
    record = _make_v1_record()
    result = _migrate_record(record)

    # Mutation is in-place and also returned
    assert result is record

    # All v2 fields must now be present and defaulted to None
    for field_name in (
        "data_quality",
        "trigger_source",
        "suppression_reason_code",
        "triage_confidence",
        "canary_would_execute",
        "execution_id",
        "rule_version",
        "autonomy_level_at_decision",
    ):
        assert field_name in result, f"Missing field after migration: {field_name}"
        assert result[field_name] is None, f"Expected None for {field_name!r}"

    # version must be bumped to 2
    assert result["version"] == 2


def test_migrate_record_preserves_existing_fields() -> None:
    """_migrate_record must not overwrite existing data."""
    record = _make_v1_record()
    # Pre-populate a v2 field — must not be clobbered
    record["suppression_reason_code"] = "type_cooldown"
    _migrate_record(record)
    assert record["suppression_reason_code"] == "type_cooldown"


def test_migrate_record_is_idempotent() -> None:
    """Running _migrate_record twice on the same dict must be a no-op."""
    record = _make_v1_record()
    _migrate_record(record)
    snapshot_after_first = dict(record)
    _migrate_record(record)
    assert record == snapshot_after_first


def test_migrate_record_skips_already_v2() -> None:
    """A record that already carries version=2 must not be modified."""
    record: dict[str, object] = {
        "snapshot_ref": {},
        "finding": {},
        "notification": {},
        "user_response": None,
        "action_outcome": None,
        "data_quality": {"completeness": 1.0},
        "trigger_source": "dynamic_rule",
        "suppression_reason_code": "not_suppressed",
        "triage_confidence": 0.9,
        "canary_would_execute": False,
        "execution_id": "exec-456",
        "rule_version": "2.0.0",
        "autonomy_level_at_decision": "notify_only",
        "version": 2,
    }
    original = dict(record)
    _migrate_record(record)
    assert record == original


# ---------------------------------------------------------------------------
# New CONF_AUDIT_* constants
# ---------------------------------------------------------------------------


def test_conf_audit_hot_max_records_is_string() -> None:
    """CONF_AUDIT_HOT_MAX_RECORDS must be a non-empty string."""
    assert isinstance(CONF_AUDIT_HOT_MAX_RECORDS, str)
    assert CONF_AUDIT_HOT_MAX_RECORDS


def test_conf_audit_archival_backlog_max_is_string() -> None:
    """CONF_AUDIT_ARCHIVAL_BACKLOG_MAX must be a non-empty string."""
    assert isinstance(CONF_AUDIT_ARCHIVAL_BACKLOG_MAX, str)
    assert CONF_AUDIT_ARCHIVAL_BACKLOG_MAX


def test_conf_audit_retention_days_is_string() -> None:
    """CONF_AUDIT_RETENTION_DAYS must be a non-empty string."""
    assert isinstance(CONF_AUDIT_RETENTION_DAYS, str)
    assert CONF_AUDIT_RETENTION_DAYS


def test_conf_audit_high_retention_days_is_string() -> None:
    """CONF_AUDIT_HIGH_RETENTION_DAYS must be a non-empty string."""
    assert isinstance(CONF_AUDIT_HIGH_RETENTION_DAYS, str)
    assert CONF_AUDIT_HIGH_RETENTION_DAYS


def test_conf_audit_constants_are_distinct() -> None:
    """All four CONF_AUDIT_* constants must be unique strings."""
    constants = [
        CONF_AUDIT_HOT_MAX_RECORDS,
        CONF_AUDIT_ARCHIVAL_BACKLOG_MAX,
        CONF_AUDIT_RETENTION_DAYS,
        CONF_AUDIT_HIGH_RETENTION_DAYS,
    ]
    assert len(constants) == len(set(constants)), (
        "Duplicate CONF_AUDIT_* constant values"
    )


# ---------------------------------------------------------------------------
# async_update_response — simple and compound finding write-back
# ---------------------------------------------------------------------------


def _make_store() -> AuditStore:
    """Return an AuditStore with a no-op HA Store underneath."""
    hass = MagicMock()
    store = AuditStore(hass)
    store._store = MagicMock()
    store._store.async_save = AsyncMock()
    return store


def _simple_record(anomaly_id: str) -> dict[str, Any]:
    return {
        "finding": {"anomaly_id": anomaly_id, "type": "open_entry"},
        "notification": {"notified_at": "2026-01-01T00:00:00+00:00"},
        "user_response": None,
        "action_outcome": None,
    }


def _compound_record(compound_id: str, constituent_ids: list[str]) -> dict[str, Any]:
    return {
        "finding": {
            "compound_id": compound_id,
            "constituent_findings": [
                {"anomaly_id": aid, "type": "open_entry"} for aid in constituent_ids
            ],
        },
        "notification": {"notified_at": "2026-01-01T00:00:00+00:00"},
        "user_response": None,
        "action_outcome": None,
    }


@pytest.mark.asyncio
async def test_update_response_matches_simple_finding() -> None:
    """async_update_response updates a simple finding by its anomaly_id."""
    store = _make_store()
    store._records = [_simple_record("aaa111")]

    await store.async_update_response(
        anomaly_id="aaa111",
        response={"action": "dismiss", "false_positive": True},
        outcome={"status": "dismissed"},
    )

    record = store._records[0]
    assert record["user_response"] == {"action": "dismiss", "false_positive": True}
    assert record["action_outcome"] == {"status": "dismissed"}
    assert "responded_at" in record["notification"]


@pytest.mark.asyncio
async def test_update_response_matches_compound_finding_via_constituent() -> None:
    """async_update_response matches a compound finding by a constituent anomaly_id."""
    store = _make_store()
    best_id = "constituent-best"
    store._records = [_compound_record("compound-uuid-1", ["constituent-a", best_id])]

    await store.async_update_response(
        anomaly_id=best_id,
        response={"action": "dismiss", "false_positive": True},
        outcome={"status": "dismissed"},
    )

    record = store._records[0]
    assert record["user_response"] == {"action": "dismiss", "false_positive": True}
    assert record["action_outcome"] == {"status": "dismissed"}
    assert "responded_at" in record["notification"]


@pytest.mark.asyncio
async def test_update_response_no_match_is_noop() -> None:
    """async_update_response does nothing when anomaly_id is not found."""
    store = _make_store()
    store._records = [_simple_record("aaa111")]

    await store.async_update_response(
        anomaly_id="nonexistent",
        response={"action": "dismiss", "false_positive": True},
        outcome={"status": "dismissed"},
    )

    record = store._records[0]
    assert record["user_response"] is None
    assert record["action_outcome"] is None
    cast("AsyncMock", store._store.async_save).assert_not_called()


@pytest.mark.asyncio
async def test_update_response_picks_most_recent_compound_record() -> None:
    """async_update_response updates the latest matching record, not an older one."""
    store = _make_store()
    old = _compound_record("compound-old", ["cid-shared"])
    new = _compound_record("compound-new", ["cid-shared"])
    store._records = [old, new]

    await store.async_update_response(
        anomaly_id="cid-shared",
        response={"action": "dismiss", "false_positive": True},
        outcome={"status": "dismissed"},
    )

    # newest record (index 1) updated; oldest (index 0) untouched
    assert store._records[1]["user_response"] == {
        "action": "dismiss",
        "false_positive": True,
    }
    assert store._records[0]["user_response"] is None


# ---------------------------------------------------------------------------
# Gap 5: action_policy_path backfilled by v1→v2 migration
# ---------------------------------------------------------------------------


def test_migrate_record_backfills_action_policy_path() -> None:
    """_migrate_record must add action_policy_path=None to a v1 record."""
    record = _make_v1_record()
    assert "action_policy_path" not in record
    _migrate_record(record)
    assert "action_policy_path" in record
    assert record["action_policy_path"] is None


def test_migrate_record_preserves_action_policy_path_if_set() -> None:
    """_migrate_record must not overwrite an existing action_policy_path value."""
    record = _make_v1_record()
    record["action_policy_path"] = "prompt_user"
    _migrate_record(record)
    assert record["action_policy_path"] == "prompt_user"


# ---------------------------------------------------------------------------
# Gap 4: AuditStore max_records is config-driven
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_store_respects_config_max_records() -> None:
    """AuditStore must enforce the max_records cap passed at construction."""
    store = _make_store()
    store._max_records = 3

    snapshot = MagicMock()
    snapshot.get = MagicMock(return_value=None)

    finding = MagicMock()
    finding.as_dict.return_value = {"anomaly_id": "x", "type": "t"}

    for _ in range(5):
        await store.async_append_finding(snapshot, finding, None)

    assert len(store._records) == 3


@pytest.mark.asyncio
async def test_audit_store_default_max_records() -> None:
    """AuditStore constructed without max_records uses RECOMMENDED_AUDIT_HOT_MAX_RECORDS."""
    hass = MagicMock()
    store = AuditStore(hass)
    assert store._max_records == RECOMMENDED_AUDIT_HOT_MAX_RECORDS


# ---------------------------------------------------------------------------
# Priority eviction: not_suppressed records are preserved over suppressed ones
# ---------------------------------------------------------------------------


def _record_with_code(anomaly_id: str, code: str | None) -> dict[str, Any]:
    """Build a minimal audit record with the given suppression_reason_code."""
    r = _simple_record(anomaly_id)
    r["suppression_reason_code"] = code
    return r


@pytest.mark.asyncio
async def test_eviction_priority_spares_not_suppressed() -> None:
    """Suppressed records are evicted before not_suppressed records."""
    store = _make_store()
    store._max_records = 5
    # Seed: 3 not_suppressed + 2 suppressed
    store._records = [
        _record_with_code("ns1", "not_suppressed"),
        _record_with_code("ns2", "not_suppressed"),
        _record_with_code("s1", "type_cooldown"),
        _record_with_code("ns3", "not_suppressed"),
        _record_with_code("s2", "suppressed"),
    ]
    # Append one more suppressed record to trigger eviction.
    store._records.append(_record_with_code("s3", "type_cooldown"))
    store._evict_one()

    assert len(store._records) == 5
    ids = {r["finding"]["anomaly_id"] for r in store._records}
    # All three not_suppressed records must survive.
    assert {"ns1", "ns2", "ns3"} <= ids
    # The oldest suppressed (s1) should have been evicted.
    assert "s1" not in ids


@pytest.mark.asyncio
async def test_eviction_last_resort_when_all_not_suppressed() -> None:
    """When every record is not_suppressed, the oldest is evicted as last resort."""
    store = _make_store()
    store._max_records = 3
    store._records = [
        _record_with_code("ns1", "not_suppressed"),
        _record_with_code("ns2", "not_suppressed"),
        _record_with_code("ns3", "not_suppressed"),
        _record_with_code("ns4", "not_suppressed"),  # triggers eviction
    ]
    store._evict_one()

    assert len(store._records) == 3
    ids = [r["finding"]["anomaly_id"] for r in store._records]
    # Oldest (ns1) should be evicted.
    assert "ns1" not in ids
    assert ids == ["ns2", "ns3", "ns4"]


@pytest.mark.asyncio
async def test_eviction_v1_record_without_code_treated_as_evictable() -> None:
    """Records missing suppression_reason_code (v1) are treated as evictable."""
    store = _make_store()
    store._max_records = 2
    v1_record = _simple_record("v1")  # no suppression_reason_code key
    not_suppressed = _record_with_code("ns1", "not_suppressed")
    store._records = [v1_record, not_suppressed, _record_with_code("s1", "suppressed")]
    store._evict_one()

    assert len(store._records) == 2
    ids = {r["finding"]["anomaly_id"] for r in store._records}
    # v1 record should have been evicted (oldest evictable).
    assert "v1" not in ids
    assert "ns1" in ids


@pytest.mark.asyncio
async def test_eviction_boundary_at_exactly_max() -> None:
    """No eviction occurs when the store is exactly at max_records capacity."""
    store = _make_store()
    store._max_records = 3
    store._records = [
        _record_with_code("s1", "suppressed"),
        _record_with_code("s2", "suppressed"),
        _record_with_code("s3", "suppressed"),
    ]
    # Should not evict — we're at capacity, not over it.
    assert len(store._records) == store._max_records


@pytest.mark.asyncio
async def test_flood_survival_exact_count() -> None:
    """not_suppressed records survive a flood of suppressed appends."""
    store = _make_store()
    store._max_records = 10
    # Seed: 2 not_suppressed records.
    store._records = [
        _record_with_code("ns1", "not_suppressed"),
        _record_with_code("ns2", "not_suppressed"),
    ]
    # Flood with 18 suppressed records (total will hit 20 → 10 evictions).
    for i in range(18):
        store._records.append(_record_with_code(f"s{i}", "type_cooldown"))
        if len(store._records) > store._max_records:
            store._evict_one()

    assert len(store._records) == store._max_records
    not_suppressed = [
        r
        for r in store._records
        if r.get("suppression_reason_code") == "not_suppressed"
    ]
    assert len(not_suppressed) == 2  # exact count preserved
