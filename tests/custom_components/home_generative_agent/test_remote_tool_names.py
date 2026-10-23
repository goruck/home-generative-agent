# ruff: noqa: S101
"""
A remote Home Assistant's tools reach the agent under their server's name (#843).

This Home Assistant and a remote one reached over MCP offer their tools under
the same names (`intent__HassTurnOn`). The agent is given one tool per name,
first-seen wins, so which house a command reached depended on ranking. A
remote Home Assistant's tools are now named after their server
(`upstairs_ha__intent__HassTurnOn`), so the agent sees both, and each call is
sent to its API under the tool's own name, on the normal path and when a PIN
confirms it.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.helpers import llm
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent.graph import (
    RawTool,
    _format_and_dedupe_tools,
    _prepare_tool_args,
)
from custom_components.home_generative_agent.agent.helpers import (
    PinReply,
    remote_tool_name,
)
from custom_components.home_generative_agent.agent.indirect_pin import (
    api_tool_name,
    remote_api_names,
    tool_origin,
)
from custom_components.home_generative_agent.agent.tools import (
    confirm_sensitive_action,
)
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
    RECOMMENDED_CRITICAL_ACTIONS,
)
from custom_components.home_generative_agent.core.utils import hash_pin

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

confirm_tool = cast("Any", confirm_sensitive_action).coroutine

TURN_ON = "intent__HassTurnOn"
REMOTE = "mcp-upstairs"
LOCAL = llm.LLM_API_ASSIST
REMOTE_TURN_ON = "upstairs_ha__intent__HassTurnOn"


class _RemoteTool(llm.Tool):
    """An MCP server's tool."""

    integration = "mcp"

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = None
        self.parameters = cast("Any", None)

    async def async_call(self, hass: Any, tool_input: Any, llm_context: Any) -> Any:
        raise NotImplementedError


def _local_intent(name: str) -> llm.Tool:
    tool = object.__new__(llm.IntentTool)
    tool.name = name
    return tool


class _Instance:
    """One API's instance, recording the names it is called with."""

    def __init__(self, name: str, tools: list[llm.Tool]) -> None:
        self.api = SimpleNamespace(name=name, id=name)
        self.tools = tools
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def async_call_tool(self, tool_input: Any) -> dict[str, object]:
        if tool_input.tool_name not in {t.name for t in self.tools}:
            msg = f"Tool {tool_input.tool_name} not found"
            raise ValueError(msg)
        self.calls.append((tool_input.tool_name, dict(tool_input.tool_args)))
        return {"success": True}


class _Wrapper:
    """Stand-in for MultiLLMAPI (conversation.py needs hassil to import)."""

    def __init__(self) -> None:
        self.apis: dict[str, Any] = {
            LOCAL: _Instance("Assist", [_local_intent(TURN_ON)]),
            REMOTE: _Instance("Upstairs HA", [_RemoteTool(TURN_ON)]),
        }
        self.routing_map = {TURN_ON: LOCAL, REMOTE_TURN_ON: REMOTE}
        self.remote_home_assistants = frozenset({REMOTE})


def _wrapper() -> _Wrapper:
    return _Wrapper()


def _raw(name: str, api_id: str) -> RawTool:
    return RawTool(
        name=name,
        api_id=api_id,
        description="Turns on a device",
        parameters="{}",
        is_actuation=True,
    )


@pytest.mark.parametrize(
    ("api_name", "name", "expected"),
    [
        ("Upstairs HA", TURN_ON, REMOTE_TURN_ON),
        ("1st floor", "script__open", "r_1st_floor__script__open"),
        ("", "GetLiveContext", "remote__GetLiveContext"),
        ("Cabin", "x.y-z", None),
        ("A very long server name indeed for this", "light__HassLightSet", None),
        ("Cabin", "x" * 90, None),
    ],
)
def test_a_remote_tool_name_fits_every_provider(
    api_name: str, name: str, expected: str | None
) -> None:
    """Letters, digits and `_`, a letter first, at most 64 characters."""
    result = remote_tool_name(api_name, name)
    assert re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", result)
    if expected is not None:
        assert result == expected
    # The tool's own end survives, so the PIN check reads its base name.
    if name == "light__HassLightSet":
        assert result.endswith("__light__HassLightSet")
    assert remote_tool_name(api_name, name) == result  # the same every time


def test_both_houses_tools_reach_the_agent() -> None:
    """The remote tool is offered beside this house's, under its own name."""
    selected, routing = _format_and_dedupe_tools(
        [_raw(TURN_ON, LOCAL), _raw(TURN_ON, REMOTE)],
        remote_apis={REMOTE: "Upstairs HA"},
    )
    names = {t["function"]["name"]: t["function"] for t in selected}
    assert set(names) == {TURN_ON, REMOTE_TURN_ON}
    assert routing == {TURN_ON: LOCAL, REMOTE_TURN_ON: REMOTE}
    assert names[REMOTE_TURN_ON]["description"].startswith(
        "[Upstairs HA, another Home Assistant]"
    )
    assert names[TURN_ON]["description"] == "Turns on a device"


def test_without_a_remote_home_assistant_names_are_unchanged() -> None:
    """Another MCP server's tools keep their names."""
    selected, routing = _format_and_dedupe_tools(
        [_raw("get_current_time", "mcp-time")], remote_apis={}
    )
    assert [t["function"]["name"] for t in selected] == ["get_current_time"]
    assert routing == {"get_current_time": "mcp-time"}


def test_the_renamed_tool_is_read_as_the_remote_one() -> None:
    """The PIN check finds the remote tool behind its new name."""
    api = _wrapper()
    origin = tool_origin(api, REMOTE_TURN_ON)
    assert origin.remote is True
    assert origin.api_id == REMOTE
    assert origin.tool_name == TURN_ON
    assert tool_origin(api, TURN_ON).remote is False
    assert api_tool_name(api, REMOTE, REMOTE_TURN_ON) == TURN_ON
    assert api_tool_name(api, LOCAL, TURN_ON) == TURN_ON
    assert remote_api_names(api) == {REMOTE: "Upstairs HA"}


async def test_a_confirmed_remote_call_uses_the_tools_own_name(
    hass: HomeAssistant,
) -> None:
    """Held under the agent's name, sent under the remote API's."""
    assert await async_setup_component(hass, "homeassistant", {})
    hass.states.async_set("light.porch", "off", {"friendly_name": "Porch"})
    async_expose_entity(hass, "conversation", "light.porch", should_expose=True)
    api = _wrapper()
    ctx = cast(
        "Any",
        SimpleNamespace(
            hass=hass,
            config={"configurable": {"thread_id": "t1"}},
            critical_actions=RECOMMENDED_CRITICAL_ACTIONS,
            pin_enabled=True,
            pin_hash="hash",
            pin_salt="salt",
            pending_actions={},
            ha_llm_api=api,
        ),
    )
    redirect, _ = _prepare_tool_args(
        REMOTE_TURN_ON, {"name": "Porch"}, ctx, {"id": "c1"}
    )
    assert redirect is not None  # every remote change asks for the PIN (#840)
    ((action_id, action),) = ctx.pending_actions.items()
    action.update(user="u1", thread_id="conv1", reply_turn="turn2")
    hashed, salt = hash_pin("1234")
    # The reply's turn offers other tools: the held call still goes home.
    api.routing_map = {}
    result = await confirm_tool(
        action_id,
        "1234",
        config={
            "configurable": {
                "options": {
                    CONF_CRITICAL_ACTION_PIN_HASH: hashed,
                    CONF_CRITICAL_ACTION_PIN_SALT: salt,
                },
                "pending_actions": ctx.pending_actions,
                "user_id": "u1",
                "thread_id": "conv1",
                "pin_reply": PinReply("1234", turn_id="turn2"),
                "ha_llm_api": api,
                "hass": hass,
            }
        },
        store=None,
    )
    assert json.loads(result).get("status") == "completed"
    local, remote = (cast("_Instance", api.apis[k]) for k in (LOCAL, REMOTE))
    assert remote.calls == [(TURN_ON, {"name": "Porch"})]
    assert local.calls == []
