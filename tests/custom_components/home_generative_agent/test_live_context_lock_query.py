# ruff: noqa: S101
"""GetLiveContext lock-state queries must reach HA with arguments it accepts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from homeassistant.components.homeassistant.llm import GetLiveContextTool

from custom_components.home_generative_agent.agent.graph import _prepare_tool_args


def _ctx() -> Any:
    return cast(
        "Any",
        SimpleNamespace(
            hass=None,
            config={"configurable": {}},
            critical_actions=[],
            pin_enabled=False,
            pin_hash=None,
            pin_salt=None,
            pending_actions={},
        ),
    )


@pytest.mark.parametrize(
    "args",
    [
        {"name": "Garage Door Lock", "domain": "lock"},
        {"name": "Garage Door Lock", "domain": ["lock"]},
        {
            "name": "Garage Door Lock",
            "domain": "lock",
            "entity_id": "lock.garage_door_lock",
        },
    ],
)
def test_lock_state_query_passes_ha_schema(args: dict[str, Any]) -> None:
    """
    Field failure 2026-10-01: every "is the garage door lock locked?" looped.

    The lock rewriting for lock/unlock commands ran on GetLiveContext too and
    added service/entity_id keys, which HA rejects (ExtraKeysInvalid), so all
    15 lock queries in the log failed while every other entity worked.
    """
    redirect, prepared = _prepare_tool_args(
        "homeassistant__GetLiveContext", dict(args), _ctx(), {"id": "1"}
    )

    assert redirect is None
    assert set(prepared) <= {"name", "domain", "area"}
    GetLiveContextTool.parameters(prepared)  # raises if HA would reject it
    assert prepared["name"] == "Garage Door Lock"


def test_other_live_context_queries_are_unchanged() -> None:
    """Non-lock filters keep their values (only unknown keys are dropped)."""
    _, prepared = _prepare_tool_args(
        "homeassistant__GetLiveContext",
        {"name": "Back Porch Light", "domain": "light", "area": "Family Room"},
        _ctx(),
        {"id": "1"},
    )
    GetLiveContextTool.parameters(prepared)
    assert prepared["name"] == "Back Porch Light"
    assert prepared["area"] == "Family Room"
    assert prepared["domain"] in ("light", ["light"])
