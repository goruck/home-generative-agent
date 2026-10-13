# ruff: noqa: S101
"""Saving the Database settings reloads an entry that is not running (#748)."""

from __future__ import annotations

from typing import Any

import pytest
from homeassistant.config_entries import ConfigEntryState, ConfigSubentryData
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.const import DOMAIN
from custom_components.home_generative_agent.flows.pgvector_db_subentry_flow import (
    PgVectorDbSubentryFlow,
)

_OLD = {"host": "db", "password": "old"}
_NEW = {"host": "db", "password": "new"}


def _flow_for(hass: Any, state: ConfigEntryState) -> tuple[Any, Any, list[str]]:
    """Return a reconfigure flow on an entry in ``state``, and its reloads."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={},
        subentries_data=[
            ConfigSubentryData(
                data=_OLD, subentry_type="database", title="Database", unique_id=None
            )
        ],
    )
    entry.add_to_hass(hass)
    entry.mock_state(hass, state)
    subentry = next(iter(entry.subentries.values()))

    flow = PgVectorDbSubentryFlow()
    flow.hass = hass
    flow.handler = (entry.entry_id, "database")
    flow.flow_id = "flow"
    flow.context = {"source": "reconfigure", "subentry_id": subentry.subentry_id}

    reloads: list[str] = []
    hass.config_entries.async_schedule_reload = reloads.append
    return flow, subentry, reloads


@pytest.mark.parametrize(
    "state", [ConfigEntryState.SETUP_RETRY, ConfigEntryState.SETUP_ERROR]
)
async def test_fixing_settings_reloads_an_entry_that_is_down(
    hass: Any, state: ConfigEntryState
) -> None:
    """The fix applies at once, not at the next retry up to 10 minutes away."""
    flow, subentry, reloads = _flow_for(hass, state)

    result = flow._async_update_and_reload_if_down(subentry, _NEW)

    assert result["reason"] == "reconfigure_successful"
    assert flow._get_entry().subentries[subentry.subentry_id].data == _NEW
    assert reloads == [flow._get_entry().entry_id]


async def test_saving_on_a_loaded_entry_leaves_the_reload_to_its_listener(
    hass: Any,
) -> None:
    """A running entry reloads through its update listener, as before."""
    flow, subentry, reloads = _flow_for(hass, ConfigEntryState.LOADED)

    flow._async_update_and_reload_if_down(subentry, _NEW)

    assert reloads == []
