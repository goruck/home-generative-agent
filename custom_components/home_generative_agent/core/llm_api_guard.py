"""
Raise a repair issue when every selected LLM API needs an admin user.

Home Assistant 2026.11's **Home Assistant** (``homeassistant``) API loads only
for a turn whose user is an admin (#918). With nothing else selected, a voice
satellite, an automation and every user who is not an admin get "No LLM APIs
could be loaded". The Options form refuses such a selection (#998), but an
install that saved one earlier still has it, and nothing at startup says why
those conversations fail (#1037).

Ids that are not registered do not count: an MCP server that is down at
startup unregisters its API for a while (#568), and that is not a setting to
change. A selection that mixes in an API everyone can load also does not
count, since that one still loads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.core import callback
from homeassistant.helpers import issue_registry as ir

from ..agent.helpers import (  # noqa: TID252
    active_llm_api_ids,
    admin_only_selection,
    sanitize_tool_text,
)
from ..const import DOMAIN  # noqa: TID252

if TYPE_CHECKING:
    from collections.abc import Mapping

    from homeassistant.core import HomeAssistant

# The id predates #918, when such an API was not loaded at all: kept so an
# issue raised by an earlier release is replaced, not left beside a new one.
ISSUE_LLM_API_UNSUPPORTED = "llm_api_unsupported_selection"
ISSUE_TRANSLATION_KEY = "llm_api_admin_only_selection"


def _issue_id(entry_id: str) -> str:
    """Per-entry issue id so two config entries cannot clobber each other."""
    return f"{ISSUE_LLM_API_UNSUPPORTED}_{entry_id}"


def admin_only_api_names(hass: HomeAssistant, options: Mapping[str, Any]) -> list[str]:
    """Return the selected APIs' names when every one needs an admin, else []."""
    apis = admin_only_selection(hass, active_llm_api_ids(options))
    # MCP API names are remote server text, sanitized as the Options form does.
    return [sanitize_tool_text(api.name) for api in apis]


@callback
def async_sync_llm_api_issue(
    hass: HomeAssistant, entry_id: str, options: Mapping[str, Any]
) -> None:
    """Raise or clear the repair issue for an admin-only LLM API selection."""
    issue_id = _issue_id(entry_id)
    names = admin_only_api_names(hass, options)
    if not names:
        ir.async_delete_issue(hass, DOMAIN, issue_id)
        return
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key=ISSUE_TRANSLATION_KEY,
        translation_placeholders={"apis": ", ".join(sorted(names))},
    )


@callback
def async_clear_llm_api_issue(hass: HomeAssistant, entry_id: str) -> None:
    """Drop the issue on unload and removal; the next setup re-checks it."""
    ir.async_delete_issue(hass, DOMAIN, _issue_id(entry_id))
