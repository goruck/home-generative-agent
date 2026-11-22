# ruff: noqa: S101
"""
Tests for the repair issue on an LLM API selection the agent cannot load.

The Options form refuses a selection of only admin-only APIs (#998), but an
install that saved one before then fails every conversation with "No LLM APIs
could be loaded" and nothing says why (#1037).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers import llm
from pytest_homeassistant.common import MockConfigEntry

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.const import DOMAIN
from custom_components.home_generative_agent.core.llm_api_guard import (
    ISSUE_LLM_API_UNSUPPORTED,
    ISSUE_TRANSLATION_KEY,
    async_clear_llm_api_issue,
    async_sync_llm_api_issue,
)

from .test_fallback_setup import _fallback_setup_data, _patch_setup_dependencies

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

ENTRY_ID = "entry-1"
ISSUE_ID = f"{ISSUE_LLM_API_UNSUPPORTED}_{ENTRY_ID}"


def _registry() -> Any:
    """Register Assist plus HA 2026.11's admin-only "homeassistant" API."""
    return patch(
        "custom_components.home_generative_agent.agent.helpers.llm.async_get_apis",
        return_value=[
            SimpleNamespace(id=llm.LLM_API_ASSIST, name="Assist"),
            SimpleNamespace(
                id="homeassistant", name="Home Assistant", requires_admin=True
            ),
        ],
    )


def _issue(hass: HomeAssistant) -> ir.IssueEntry | None:
    return ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_ID)


@pytest.mark.asyncio
async def test_a_selection_of_only_admin_apis_raises_the_issue(
    hass: HomeAssistant,
) -> None:
    """Every turn fails on this selection, so setup says why."""
    with _registry():
        async_sync_llm_api_issue(hass, ENTRY_ID, {CONF_LLM_HASS_API: ["homeassistant"]})

    issue = _issue(hass)
    assert issue is not None
    assert issue.severity == ir.IssueSeverity.ERROR
    assert issue.translation_key == ISSUE_TRANSLATION_KEY
    assert issue.translation_placeholders == {"apis": "Home Assistant"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        pytest.param(
            {CONF_LLM_HASS_API: [llm.LLM_API_ASSIST, "homeassistant"]},
            id="with-a-supported",
        ),
        # Often a transient MCP outage, not a setting to change (#568).
        pytest.param(
            {CONF_LLM_HASS_API: ["homeassistant", "mcp-gone"]},
            id="with-an-unregistered",
        ),
        pytest.param({CONF_LLM_HASS_API: ["mcp-gone"]}, id="only-unregistered"),
        pytest.param({CONF_LLM_HASS_API: []}, id="empty"),
        # An absent key means Assist.
        pytest.param({}, id="absent"),
    ],
)
async def test_a_loadable_or_unknown_selection_clears_the_issue(
    hass: HomeAssistant, options: dict[str, Any]
) -> None:
    """Only a selection the agent can never load is reported."""
    with _registry():
        async_sync_llm_api_issue(hass, ENTRY_ID, {CONF_LLM_HASS_API: ["homeassistant"]})
        assert _issue(hass) is not None

        async_sync_llm_api_issue(hass, ENTRY_ID, options)

    assert _issue(hass) is None


@pytest.mark.asyncio
async def test_unload_drops_the_issue(hass: HomeAssistant) -> None:
    """An unloaded entry is not failing turns; the next setup re-checks."""
    with _registry():
        async_sync_llm_api_issue(hass, ENTRY_ID, {CONF_LLM_HASS_API: ["homeassistant"]})

    async_clear_llm_api_issue(hass, ENTRY_ID)

    assert _issue(hass) is None


def test_the_issue_has_a_title_and_description() -> None:
    """A translation_key with no strings shows an empty repair card."""
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).parents[3] / "custom_components" / DOMAIN
    for name in ("strings.json", "translations/en.json"):
        issues = json.loads((root / name).read_text(encoding="utf-8"))["issues"]
        entry = issues[ISSUE_TRANSLATION_KEY]
        assert entry["title"]
        assert "{apis}" in entry["description"]


@pytest.mark.asyncio
async def test_setup_reports_an_unloadable_llm_api_selection(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup raises the issue, and unload drops it until the next setup."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    data = _fallback_setup_data()
    data.options[CONF_LLM_HASS_API] = ["homeassistant"]
    _patch_setup_dependencies(hass, monkeypatch, data)
    issue_id = f"{ISSUE_LLM_API_UNSUPPORTED}_{entry.entry_id}"

    with _registry():
        assert await cast("Any", hga_component).async_setup_entry(hass, entry)
    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id) is not None

    assert await cast("Any", hga_component).async_unload_entry(hass, entry) is True
    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id) is None
