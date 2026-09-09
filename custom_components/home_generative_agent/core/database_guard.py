"""
Surface an entry that is running without its PostgreSQL database.

A config entry with no Database subentry loads successfully: setup falls back
to in-memory conversation state, a null long-term store, and no person
gallery. Nothing else fails loudly, so the user sees a working chat and only
discovers the gap when memory does not persist across restarts, semantic
recall returns nothing, or face enrollment fails.

Two paths lead there (#615): ``+ Setup`` was never run after the entry was
created, or the Advanced wizard was closed at its final Database step (it
writes the feature subentries *before* that step, so a "cannot connect" error
dismissed there leaves every feature configured and no database). This module
raises a repair issue for that state and clears it once a database is
configured, so the cause is visible in Settings → Repairs instead of only in a
warning buried in the log.

The issue is deliberately not persistent (the registry default): the entry
re-syncs it on every setup, and a persistent issue would outlive an
uninstall-before-remove, where Home Assistant never reaches the integration's
``async_remove_entry`` hook (``ConfigEntry.async_remove`` returns on
``IntegrationNotFound`` first).
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
    hass: HomeAssistant, entry_id: str, *, database_missing: bool
) -> None:
    """
    Raise or clear the repair issue for an entry with no database.

    Idempotent: safe to call on every setup and reload. Deleting an issue that
    does not exist is a no-op in the issue registry, and re-creating an
    unchanged issue neither writes storage nor fires an event, so repeated
    setups are quiet.
    """
    issue_id = _issue_id(entry_id)

    if not database_missing:
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
    """
    Drop the issue outside of setup: on unload and on entry removal.

    An unloaded (disabled, reloading, or failed) entry is not "running without
    a database", so the issue must not stand while it is down; the next
    successful setup re-raises it if the database is still missing.
    """
    ir.async_delete_issue(hass, DOMAIN, _issue_id(entry_id))
