# ruff: noqa: S101
"""
Tests for the no-database guard (#615).

An entry with no Database subentry loads on in-memory storage. That must be
visible (a log warning and a repair issue), and the two face-enrollment
paths must say what is missing instead of dereferencing a ``None`` gallery.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.const import DOMAIN
from custom_components.home_generative_agent.core.database_guard import (
    ISSUE_DATABASE_NOT_CONFIGURED,
    async_clear_database_issue,
    async_sync_database_issue,
)

from .test_fallback_setup import _fallback_setup_data, _patch_setup_dependencies


def _issue(hass: Any, entry_id: str) -> ir.IssueEntry | None:
    return ir.async_get(hass).async_get_issue(
        DOMAIN, f"{ISSUE_DATABASE_NOT_CONFIGURED}_{entry_id}"
    )


async def _setup_without_database(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> MockConfigEntry:
    """Set up through the fallback harness, which resolves no database URI."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True
    _patch_setup_dependencies(hass, monkeypatch, _fallback_setup_data())
    result = await cast("Any", hga_component).async_setup_entry(hass, entry)
    assert result is True
    assert entry.runtime_data.person_gallery is None
    return entry


@pytest.mark.asyncio
async def test_setup_without_database_warns_and_raises_repair_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The silent fallback branch must log and surface a repair issue."""
    entry = await _setup_without_database(hass, monkeypatch)

    assert "No database is configured for this entry" in caplog.text
    issue = _issue(hass, entry.entry_id)
    assert issue is not None
    assert issue.severity == ir.IssueSeverity.WARNING
    assert issue.translation_key == ISSUE_DATABASE_NOT_CONFIGURED


@pytest.mark.asyncio
async def test_enroll_person_service_reports_missing_database(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The service must raise a clear error, not AttributeError on None."""
    await _setup_without_database(hass, monkeypatch)

    async def _fake_read(*_args: Any, **_kwargs: Any) -> bytes:
        return b"jpeg-bytes"

    monkeypatch.setattr(hga_component, "_read_enroll_image_bytes", _fake_read)

    with pytest.raises(HomeAssistantError, match="configured database"):
        await hass.services.async_call(
            DOMAIN,
            "enroll_person",
            {"name": "Alice", "file_path": "/media/faces/alice.jpg"},
            blocking=True,
        )


@pytest.mark.asyncio
async def test_sync_database_issue_clears_once_configured(hass: Any) -> None:
    """Idempotent raise/clear so a reload with a database drops the issue."""
    async_sync_database_issue(hass, "entry-a", configured=False)
    async_sync_database_issue(hass, "entry-a", configured=False)
    assert _issue(hass, "entry-a") is not None

    async_sync_database_issue(hass, "entry-a", configured=True)
    assert _issue(hass, "entry-a") is None

    # Clearing an issue that does not exist is a no-op.
    async_sync_database_issue(hass, "entry-a", configured=True)
    assert _issue(hass, "entry-a") is None


@pytest.mark.asyncio
async def test_issue_is_per_entry_and_removed_with_the_entry(hass: Any) -> None:
    """Two entries do not clobber each other's issue; removal drops it."""
    async_sync_database_issue(hass, "entry-a", configured=False)
    async_sync_database_issue(hass, "entry-b", configured=False)

    async_clear_database_issue(hass, "entry-a")
    assert _issue(hass, "entry-a") is None
    assert _issue(hass, "entry-b") is not None

    entry_b = MockConfigEntry(domain=DOMAIN, data={}, entry_id="entry-b")
    await cast("Any", hga_component).async_remove_entry(hass, entry_b)
    assert _issue(hass, "entry-b") is None
