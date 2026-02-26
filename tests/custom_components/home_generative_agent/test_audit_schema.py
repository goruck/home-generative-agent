# ruff: noqa: S101
"""Tests for audit record schema (issue #254): versioned fields and migration."""

from __future__ import annotations

from dataclasses import fields

import pytest

from custom_components.home_generative_agent.audit.models import AuditRecord
from custom_components.home_generative_agent.audit.store import (
    _migrate_record,
)
from custom_components.home_generative_agent.const import (
    CONF_AUDIT_ARCHIVAL_BACKLOG_MAX,
    CONF_AUDIT_HIGH_RETENTION_DAYS,
    CONF_AUDIT_HOT_MAX_RECORDS,
    CONF_AUDIT_RETENTION_DAYS,
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
