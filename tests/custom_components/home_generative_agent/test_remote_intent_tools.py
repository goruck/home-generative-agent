# ruff: noqa: S101
"""
A remote Home Assistant's intent and script tools are held for the PIN (#840).

A remote Home Assistant reached over MCP offers its intents and scripts under
the names this one uses (`intent__HassTurnOn`, `script__open_door`). The gate
read those names as this Home Assistant's own tools and looked the targets up
here: `HassTurnOn {"name": "Front Door"}` matched nothing here, so it ran on
the remote house with no PIN. A tool is now read as this Home Assistant's own
only when the API that runs it holds an intent or action tool; otherwise any
call that can change something is held, its arguments go as given, the PIN
line names the API, and the confirmed call goes back to that API.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.helpers import llm
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent.graph import (
    _is_critical_action,
    _prepare_tool_args,
)
from custom_components.home_generative_agent.agent.helpers import (
    BUILTIN_INTENT,
    REMOTE_TOOL,
    PinReply,
)
from custom_components.home_generative_agent.agent.indirect_pin import (
    tool_args_kind,
    tool_origin,
)
from custom_components.home_generative_agent.agent.pin_messages import (
    describe_pending_action,
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
REMOTE = "mcp-remote"
LOCAL = "assist"


class _RemoteTool(llm.Tool):
    """A tool another API runs, such as an MCP server's."""

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


def _local_script(name: str) -> llm.Tool:
    tool = object.__new__(llm.ActionTool)
    tool.name = name
    return tool


class _Instance:
    """One API's instance: its tools, recording the calls it runs."""

    def __init__(self, name: str, tools: list[llm.Tool]) -> None:
        self.api = SimpleNamespace(name=name, id=name)
        self.tools = tools
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def async_call_tool(self, tool_input: Any) -> dict[str, object]:
        self.calls.append((tool_input.tool_name, dict(tool_input.tool_args)))
        return {"success": True}


class _Wrapper:
    """Stand-in for MultiLLMAPI: routes a call by the routing map."""

    def __init__(self, apis: dict[str, _Instance], routing: dict[str, str]) -> None:
        self.apis = apis
        self.routing_map = routing

    async def async_call_tool(self, tool_input: Any) -> Any:
        return await self.apis[self.routing_map[tool_input.tool_name]].async_call_tool(
            tool_input
        )


def _api(remote_tools: list[str], local_tools: list[llm.Tool] | None = None) -> Any:
    local = _Instance(LOCAL, local_tools or [])
    remote = _Instance("Upstairs HA", [_RemoteTool(n) for n in remote_tools])
    routing = {t.name: LOCAL for t in local.tools}
    routing.update(dict.fromkeys(remote_tools, REMOTE))
    return _Wrapper({LOCAL: local, REMOTE: remote}, routing)


@pytest.fixture
async def home(hass: HomeAssistant) -> HomeAssistant:
    """Set up a light named Garage Door here; the remote door is unseen."""
    assert await async_setup_component(hass, "homeassistant", {})
    hass.states.async_set("light.garage_door", "off", {"friendly_name": "Garage Door"})
    hass.states.async_set(
        "alarm_control_panel.home", "armed_away", {"friendly_name": "Home"}
    )
    for entity_id in ("light.garage_door", "alarm_control_panel.home"):
        async_expose_entity(hass, "conversation", entity_id, should_expose=True)
    return hass


def _ctx(hass: HomeAssistant, api: Any) -> Any:
    return cast(
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


async def _answer_pin(home: HomeAssistant, ctx: Any, api: Any) -> str:
    """Answer the one held action with the right PIN."""
    ((action_id, action),) = ctx.pending_actions.items()
    action.update(user="u1", thread_id="conv1", reply_turn="turn2")
    hashed, salt = hash_pin("1234")
    config = {
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
            "hass": home,
        }
    }
    return await confirm_tool(action_id, "1234", config=config, store=None)


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        (TURN_ON, {"name": "Front Door"}),
        ("HassTurnOn", {"name": "Front Door"}),  # a remote HA before 2026.9
        (TURN_ON, {"area": "Garage"}),
        (TURN_ON, {"name": "Garage Door"}),  # a light here, a door there
        (TURN_ON, {"name": "Front Door", "domain": "cover"}),
        ("intent__HassSetPosition", {"name": "Front Door", "position": 100}),
        ("intent__HassLightSet", {"name": "Hall", "brightness": 50}),
        ("script__open_door", {}),
        ("intent_script__OpenDoor", {}),
    ],
)
async def test_a_remote_change_is_held(
    home: HomeAssistant, tool_name: str, args: dict[str, Any]
) -> None:
    """Nothing here says what the remote call acts on, so it is held."""
    api = _api([tool_name])
    ctx = _ctx(home, api)
    redirect, out = _prepare_tool_args(tool_name, dict(args), ctx, {"id": "c1"})
    assert redirect is not None
    assert "requires_pin" in str(redirect.content)
    assert out == args  # as given: no lock filled in, no domain list
    (action,) = ctx.pending_actions.values()
    assert action["remote"] is True
    assert action["args_kind"] == REMOTE_TOOL
    line = describe_pending_action(home, action)
    assert 'through "Upstairs HA"' in line
    # The model's words are not repeated, and this house's devices not named.
    assert "Front Door" not in line
    assert "Garage Door" not in line


@pytest.mark.parametrize(
    "tool_name", ["intent__HassGetState", "HassGetWeather", "intent__HassNevermind"]
)
async def test_a_remote_read_is_not_held(home: HomeAssistant, tool_name: str) -> None:
    """A read-only intent changes nothing, wherever it runs."""
    ctx = _ctx(home, _api([tool_name]))
    redirect, _ = _prepare_tool_args(tool_name, {"name": "Front Door"}, ctx, {})
    assert redirect is None
    assert not ctx.pending_actions


async def test_a_remote_alarm_call_is_held_not_sent_to_the_alarm_tool(
    home: HomeAssistant,
) -> None:
    """The alarm tool acts on this house's panels, not the remote one's."""
    ctx = _ctx(home, _api(["intent__HassTurnOff"]))
    redirect, _ = _prepare_tool_args(
        "intent__HassTurnOff",
        {"domain": "alarm_control_panel", "name": "Home"},
        ctx,
        {"id": "c1"},
    )
    assert redirect is not None
    assert "requires_pin" in str(redirect.content)


async def test_this_home_assistants_own_intent_is_read_as_before(
    home: HomeAssistant,
) -> None:
    """A local intent tool is screened against this house, as before."""
    api = _api([], [_local_intent(TURN_ON)])
    assert tool_origin(api, TURN_ON).remote is False
    assert tool_args_kind(home, TURN_ON, remote=False) == BUILTIN_INTENT
    ctx = _ctx(home, api)
    redirect, _ = _prepare_tool_args(TURN_ON, {"name": "Garage Door"}, ctx, {})
    assert redirect is None  # a light here
    assert not ctx.pending_actions


async def test_a_local_script_tool_is_not_remote() -> None:
    """A script's own tool is an action tool."""
    api = _api([], [_local_script("script__open_door")])
    assert tool_origin(api, "script__open_door").remote is False


async def test_an_unrelated_remote_tool_is_not_affected() -> None:
    """Only names Home Assistant gives its own tools are read as remote ones."""
    api = _api(["get_current_time"])
    origin = tool_origin(api, "get_current_time")
    assert origin.api_id == REMOTE
    assert origin.remote is False


async def test_no_routing_reads_the_name_as_before(home: HomeAssistant) -> None:
    """Without an API to read, nothing is known about the tool."""
    assert tool_origin(None, TURN_ON).remote is False
    assert tool_origin(SimpleNamespace(), TURN_ON).remote is False
    assert not _is_critical_action(
        {"name": "Front Door"}, RECOMMENDED_CRITICAL_ACTIONS, TURN_ON, home
    )
    assert _is_critical_action(
        {"name": "Front Door"}, RECOMMENDED_CRITICAL_ACTIONS, TURN_ON, home, remote=True
    )


async def test_the_confirmed_call_goes_to_the_api_it_was_held_for(
    home: HomeAssistant,
) -> None:
    """Whatever the reply's turn routes the name to, the held API runs it."""
    api = _api([TURN_ON])
    ctx = _ctx(home, api)
    args = {"name": "Front Door", "domain": "lock"}
    _prepare_tool_args(TURN_ON, dict(args), ctx, {"id": "c1"})
    # The reply's turn offers this house's tool under the same name.
    api.apis[LOCAL].tools = [_local_intent(TURN_ON)]
    api.routing_map = {TURN_ON: LOCAL}
    result = await _answer_pin(home, ctx, api)
    assert json.loads(result).get("status") == "completed"
    assert api.apis[REMOTE].calls == [(TURN_ON, args)]
    assert api.apis[LOCAL].calls == []


async def test_a_held_call_whose_api_is_gone_is_dropped(home: HomeAssistant) -> None:
    """With the remote API removed, nothing runs and the request is dropped."""
    api = _api([TURN_ON])
    ctx = _ctx(home, api)
    _prepare_tool_args(TURN_ON, {"name": "Front Door"}, ctx, {"id": "c1"})
    del api.apis[REMOTE]
    api.apis[LOCAL].tools = [_local_intent(TURN_ON)]
    api.routing_map = {TURN_ON: LOCAL}
    result = await _answer_pin(home, ctx, api)
    assert "completed" not in result
    assert api.apis[LOCAL].calls == []
    assert not ctx.pending_actions


async def test_a_local_held_call_goes_to_the_local_api(home: HomeAssistant) -> None:
    """A held local call is confirmed on this house, not a remote look-alike."""
    hass = home
    hass.states.async_set("lock.front_door", "locked", {"friendly_name": "Front Door"})
    async_expose_entity(hass, "conversation", "lock.front_door", should_expose=True)
    api = _api([], [_local_intent("intent__HassTurnOff")])
    ctx = _ctx(hass, api)
    redirect, _ = _prepare_tool_args(
        "intent__HassTurnOff", {"name": "Front Door"}, ctx, {"id": "c1"}
    )
    assert redirect is not None
    (action,) = ctx.pending_actions.values()
    assert action["remote"] is False
    assert action["api_id"] == LOCAL
    # The reply's turn routes the name to a remote API instead.
    api.apis[REMOTE].tools = [_RemoteTool("intent__HassTurnOff")]
    api.routing_map = {"intent__HassTurnOff": REMOTE}
    result = await _answer_pin(hass, ctx, api)
    assert json.loads(result).get("status") == "completed"
    assert api.apis[REMOTE].calls == []
    assert [name for name, _ in api.apis[LOCAL].calls] == ["intent__HassTurnOff"]
