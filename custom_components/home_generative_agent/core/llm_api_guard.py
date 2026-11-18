"""
Raise a repair issue when every selected LLM API is one this integration skips.

The Options form refuses a **Control Home Assistant** selection that holds only
APIs this integration does not load (#998), but an install that saved one
before the form hid them still has it. Every conversation then fails with "No
LLM APIs could be loaded", and nothing at startup says why (#1037). Today the
only such API is Home Assistant 2026.11's admin-only **Home Assistant**
(``homeassistant``) API.

Ids that are not registered do not count: an MCP server that is down at
startup unregisters its API for a while (#568), and that is not a setting to
change. A selection that mixes a supported API with an unsupported one also
does not count, since the supported one still loads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.core import callback
from homeassistant.helpers import issue_registry as ir

from ..agent.helpers import (  # noqa: TID252
    active_llm_api_ids,
    sanitize_tool_text,
    unsupported_only_apis,
)
from ..const import DOMAIN  # noqa: TID252

if TYPE_CHECKING:
    from collections.abc import Mapping

    from homeassistant.core import HomeAssistant

ISSUE_LLM_API_UNSUPPORTED = "llm_api_unsupported_selection"


def _issue_id(entry_id: str) -> str:
    """Per-entry issue id so two config entries cannot clobber each other."""
    return f"{ISSUE_LLM_API_UNSUPPORTED}_{entry_id}"


def unsupported_only_selection(
    hass: HomeAssistant, options: Mapping[str, Any]
) -> list[str]:
    """Return the selected APIs' names when none of them can be loaded, else []."""
    apis = unsupported_only_apis(hass, active_llm_api_ids(options))
    # MCP API names are remote server text, sanitized as the Options form does.
    return [sanitize_tool_text(api.name) for api in apis]


@callback
def async_sync_llm_api_issue(
    hass: HomeAssistant, entry_id: str, options: Mapping[str, Any]
) -> None:
    """Raise or clear the repair issue for an unloadable LLM API selection."""
    issue_id = _issue_id(entry_id)
    names = unsupported_only_selection(hass, options)
    if not names:
        ir.async_delete_issue(hass, DOMAIN, issue_id)
        return
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key=ISSUE_LLM_API_UNSUPPORTED,
        translation_placeholders={"apis": ", ".join(sorted(names))},
    )


@callback
def async_clear_llm_api_issue(hass: HomeAssistant, entry_id: str) -> None:
    """Drop the issue on unload and removal; the next setup re-checks it."""
    ir.async_delete_issue(hass, DOMAIN, _issue_id(entry_id))
