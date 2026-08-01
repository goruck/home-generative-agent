# ruff: noqa: S101
"""Tests for ProposalStore.cleanup_unsupported_ttl (Bug 3 fix)."""

from __future__ import annotations

import datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.home_generative_agent.sentinel.proposal_store import (
    _UNSUPPORTED_TTL_DAYS,
    ProposalStore,
)


def _make_proposal(
    candidate_id: str,
    status: str,
    created_at: str | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {"candidate_id": candidate_id, "status": status}
    if created_at is not None:
        record["created_at"] = created_at
    return record


def _store_with_records(records: list[dict[str, Any]]) -> ProposalStore:
    """Return a ProposalStore pre-loaded with records (no HA hass needed)."""
    store = object.__new__(ProposalStore)
    store._records = list(records)
    store._store = AsyncMock()
    store._store.async_save = AsyncMock()
    return store


def _days_ago(days: float) -> str:
    dt = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Happy path: old unsupported record is removed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_removes_old_unsupported() -> None:
    old = _make_proposal("c1", "unsupported", _days_ago(_UNSUPPORTED_TTL_DAYS + 1))
    store = _store_with_records([old])
    removed = await store.cleanup_unsupported_ttl()
    assert removed == 1
    assert store._records == []


@pytest.mark.asyncio
async def test_cleanup_keeps_recent_unsupported() -> None:
    recent = _make_proposal("c2", "unsupported", _days_ago(_UNSUPPORTED_TTL_DAYS - 1))
    store = _store_with_records([recent])
    removed = await store.cleanup_unsupported_ttl()
    assert removed == 0
    assert len(store._records) == 1


# ---------------------------------------------------------------------------
# Non-unsupported records are never touched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_ignores_non_unsupported_statuses() -> None:
    draft = _make_proposal("c3", "draft", _days_ago(999))
    rejected = _make_proposal("c4", "rejected", _days_ago(999))
    approved = _make_proposal("c5", "approved", _days_ago(999))
    store = _store_with_records([draft, rejected, approved])
    removed = await store.cleanup_unsupported_ttl()
    assert removed == 0
    assert len(store._records) == 3


# ---------------------------------------------------------------------------
# Defensive: missing or malformed created_at leaves record in place
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_keeps_record_with_missing_created_at() -> None:
    no_date = _make_proposal("c6", "unsupported", None)
    store = _store_with_records([no_date])
    removed = await store.cleanup_unsupported_ttl()
    assert removed == 0
    assert len(store._records) == 1


@pytest.mark.asyncio
async def test_cleanup_keeps_record_with_malformed_created_at() -> None:
    bad_date = _make_proposal("c7", "unsupported", "not-a-date")
    store = _store_with_records([bad_date])
    removed = await store.cleanup_unsupported_ttl()
    assert removed == 0
    assert len(store._records) == 1


# ---------------------------------------------------------------------------
# Mixed batch: only expired unsupported records are removed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_mixed_batch() -> None:
    old_unsupported = _make_proposal(
        "c8", "unsupported", _days_ago(_UNSUPPORTED_TTL_DAYS + 5)
    )
    recent_unsupported = _make_proposal(
        "c9", "unsupported", _days_ago(_UNSUPPORTED_TTL_DAYS - 1)
    )
    draft = _make_proposal("c10", "draft", _days_ago(999))
    store = _store_with_records([old_unsupported, recent_unsupported, draft])
    removed = await store.cleanup_unsupported_ttl()
    assert removed == 1
    remaining_ids = [r["candidate_id"] for r in store._records]
    assert "c8" not in remaining_ids
    assert "c9" in remaining_ids
    assert "c10" in remaining_ids


# ---------------------------------------------------------------------------
# Save is called only when records are actually removed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_calls_save_only_when_removed() -> None:
    old = _make_proposal("c11", "unsupported", _days_ago(_UNSUPPORTED_TTL_DAYS + 1))
    store = _store_with_records([old])
    with patch.object(store, "async_save", new_callable=AsyncMock) as mock_save:
        await store.cleanup_unsupported_ttl()
    mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_does_not_call_save_when_nothing_removed() -> None:
    recent = _make_proposal("c12", "draft", _days_ago(999))
    store = _store_with_records([recent])
    with patch.object(store, "async_save", new_callable=AsyncMock) as mock_save:
        await store.cleanup_unsupported_ttl()
    mock_save.assert_not_awaited()


# ---------------------------------------------------------------------------
# Naive (no-tz) created_at is treated as UTC and still cleaned up
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_naive_datetime_treated_as_utc() -> None:
    naive_old = (
        datetime.datetime.now(datetime.UTC)
        - datetime.timedelta(days=_UNSUPPORTED_TTL_DAYS + 2)
    ).replace(tzinfo=None)
    record = _make_proposal("c13", "unsupported", naive_old.isoformat())
    store = _store_with_records([record])
    removed = await store.cleanup_unsupported_ttl()
    assert removed == 1


# ---------------------------------------------------------------------------
# approved_at stamp (Fix 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approved_at_stamped_on_first_approval() -> None:
    """async_update_status stamps approved_at when status=approved for the first time."""
    record: dict = {"candidate_id": "p1", "status": "draft"}
    store = _store_with_records([record])

    updated = await store.async_update_status("p1", "approved")

    assert updated is True
    assert "approved_at" in store._records[0]
    # Must be a valid ISO-8601 timestamp.
    parsed = datetime.datetime.fromisoformat(store._records[0]["approved_at"])
    assert parsed.tzinfo is not None  # timezone-aware


@pytest.mark.asyncio
async def test_approved_at_not_overwritten_on_re_approval() -> None:
    """Re-approving a proposal does not reset approved_at."""
    original_ts = "2026-01-01T00:00:00+00:00"
    record: dict = {
        "candidate_id": "p2",
        "status": "approved",
        "approved_at": original_ts,
    }
    store = _store_with_records([record])

    await store.async_update_status("p2", "approved")

    assert store._records[0]["approved_at"] == original_ts


@pytest.mark.asyncio
async def test_non_approved_status_does_not_set_approved_at() -> None:
    """Setting status to something other than 'approved' must not write approved_at."""
    record: dict = {"candidate_id": "p3", "status": "draft"}
    store = _store_with_records([record])

    await store.async_update_status("p3", "unsupported")

    assert "approved_at" not in store._records[0]
