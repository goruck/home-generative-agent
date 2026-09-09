"""
Surface an entry that is running without its PostgreSQL database.

A config entry with no Database subentry loads successfully: setup falls back
to in-memory conversation state, a null long-term store, and no person
gallery. Nothing else fails loudly, so the user sees a working chat and only
discovers the gap when memory does not persist across restarts, semantic
recall returns nothing, or face enrollment fails.

The Advanced setup wizard makes that state easy to reach: it writes the
feature subentries *before* its final Database step, so closing the dialog at
a "cannot connect" error leaves every feature configured and no database
(#615). This module raises a repair issue for that state and clears it once a
database is configured, so the cause is visible in Settings → Repairs instead
of only in a warning buried in the log.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.core import callback
from homeassistant.helpers import issue_registry as ir

from ..const import DOMAIN  # noqa: TID252

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

ISSUE_DATABASE_NOT_CONFIGURED = "database_not_configured"


def _issue_id(entry_id: str) -> str:
    """Per-entry issue id so two config entries cannot clobber each other."""
    return f"{ISSUE_DATABASE_NOT_CONFIGURED}_{entry_id}"


@callback
def async_sync_database_issue(
    hass: HomeAssistant, entry_id: str, *, configured: bool
) -> None:
    """
    Raise or clear the repair issue for an entry with no database.

    Idempotent: safe to call on every setup and reload. Deleting an issue that
    does not exist is a no-op in the issue registry, so the clear path needs no
    existence check.
    """
    issue_id = _issue_id(entry_id)

    if configured:
        ir.async_delete_issue(hass, DOMAIN, issue_id)
        return

    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=False,
        severity=ir.IssueSeverity.WARNING,
        translation_key=ISSUE_DATABASE_NOT_CONFIGURED,
    )


@callback
def async_clear_database_issue(hass: HomeAssistant, entry_id: str) -> None:
    """Drop the issue when the entry itself is removed."""
    ir.async_delete_issue(hass, DOMAIN, _issue_id(entry_id))
