# ruff: noqa: S101
"""HA 2026.10 wraps tool call results in ``llm.ToolResult`` (issue #735)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from homeassistant.helpers import llm

from custom_components.home_generative_agent.agent import graph
from custom_components.home_generative_agent.agent.helpers import tool_call_data
from custom_components.home_generative_agent.agent.tools import confirm_sensitive_action
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
)
from custom_components.home_generative_agent.core.utils import hash_pin


@dataclass(slots=True)
class _ToolResult:
    """Same shape as HA 2026.10's ``llm.ToolResult``."""

    data: Any
    error: bool = False


@pytest.fixture
def tool_result_cls(monkeypatch: pytest.MonkeyPatch) -> type:
    """Return ``llm.ToolResult``, installing a stand-in on HA < 2026.10."""
    existing = getattr(llm, "ToolResult", None)
    if existing is not None:
        return existing
    monkeypatch.setattr(llm, "ToolResult", _ToolResult, raising=False)
    return _ToolResult


class _FakeAPI:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[str] = []

    async def async_call_tool(self, tool_input: Any) -> Any:
        self.calls.append(tool_input.tool_name)
        return self.result


def test_tool_call_data_unwraps_a_tool_result(tool_result_cls: type) -> None:
    data = {"success": True, "result": "Lights: on"}

    unwrapped = tool_call_data(tool_result_cls(data=data))

    assert unwrapped.data == data
    assert unwrapped.error is False


def test_tool_call_data_keeps_the_error_flag(tool_result_cls: type) -> None:
    unwrapped = tool_call_data(tool_result_cls(data={"error": "x"}, error=True))

    assert unwrapped.error is True


def test_tool_call_data_passes_a_pre_2026_10_dict_through() -> None:
    data = {"success": True}

    assert tool_call_data(data) == (data, False)


def _ctx(api: _FakeAPI) -> graph.ToolExecutionContext:
    return graph.ToolExecutionContext(
        hass=MagicMock(),
        store=MagicMock(),
        config=cast("Any", {"configurable": {}}),
        ha_llm_api=api,
        critical_actions=[],
        pin_enabled=False,
        pin_hash=None,
        pin_salt=None,
        pending_actions={},
        state=cast("Any", {}),
    )


async def test_ha_tool_result_reaches_the_model_as_its_data(
    tool_result_cls: type,
) -> None:
    """
    Field case: every HA tool call failed on 2026.10.

    "Tool homeassistant__GetLiveContext raised during gather: TypeError('Object
    of type ToolResult is not JSON serializable')".
    """
    data = {"success": True, "result": "Kitchen light: on"}
    api = _FakeAPI(tool_result_cls(data=data))

    message = await graph._run_ha_tool("GetLiveContext", {}, _ctx(api), {"id": "call1"})

    assert api.calls == ["GetLiveContext"]
    assert json.loads(cast("str", message.content)) == data
    assert message.status == "success"


async def test_ha_tool_error_result_is_an_error_message(
    tool_result_cls: type,
) -> None:
    api = _FakeAPI(tool_result_cls(data={"error": "Boom"}, error=True))

    message = await graph._run_ha_tool("GetLiveContext", {}, _ctx(api), {"id": "call1"})

    assert message.status == "error"


async def test_pin_confirmed_action_reports_completion(
    tool_result_cls: type,
) -> None:
    """
    The confirmed action ran; its reply must not crash on serialization.

    Before the fix the unlock happened and the user got an error instead.
    """
    hashed, salt = hash_pin("1234")
    api = _FakeAPI(tool_result_cls(data={"success": True, "result": "Unlocked"}))
    config = {
        "configurable": {
            "options": {
                CONF_CRITICAL_ACTION_PIN_HASH: hashed,
                CONF_CRITICAL_ACTION_PIN_SALT: salt,
            },
            "pending_actions": {
                "aid": {
                    "tool_name": "HassTurnOff",
                    "tool_args": {"domain": ["lock"], "name": "Front Door"},
                    "user": "u1",
                }
            },
            "user_id": "u1",
            "ha_llm_api": api,
            "hass": None,
        }
    }
    confirm = cast("Any", confirm_sensitive_action).coroutine

    reply = json.loads(await confirm("aid", "1234", config=config, store=None))

    assert api.calls == ["HassTurnOff"]
    assert reply["status"] == "completed"
    assert reply["result"] == {"success": True, "result": "Unlocked"}
