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
    ToolOrigin,
    is_remote_home_assistant,
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

    def __init__(self, name: str, *, read_only: bool = False) -> None:
        self.name = name
        self.description = None
        self.parameters = cast("Any", None)
        self.annotations = llm.ToolAnnotations(read_only=read_only)

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


# A tool only a Home Assistant offers: its server is a remote Home Assistant.
HA_MARKER = "intent__HassNevermind"


def _api(
    remote_tools: list[str],
    local_tools: list[llm.Tool] | None = None,
    *,
    ha_server: bool = True,
    read_only: tuple[str, ...] = (),
) -> Any:
    local = _Instance(LOCAL, local_tools or [])
    names = list(remote_tools)
    if ha_server and HA_MARKER not in names:
        names.append(HA_MARKER)
    remote = _Instance(
        "Upstairs HA", [_RemoteTool(n, read_only=n in read_only) for n in names]
    )
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
        # Named like a read-only intent, but a script may do anything.
        ("intent_script__HassGetState", {}),
        ("script__hassgetstate", {}),
        # Names that say nothing, on a remote Home Assistant's server.
        ("open_garage_door", {}),  # a script's tool before 2026.9
        ("OpenGarage", {}),  # an intent script's tool before 2026.9
        ("media_player__play_media", {"media_id": "x"}),
        ("my_integration__open_gate", {}),
        # The remote side runs the command when the timer ends.
        (
            "intent__HassStartTimer",
            {"seconds": 1, "conversation_command": "open the garage door"},
        ),
        # Not marked read-only, whatever its name: no shortcut past the check.
        ("GetLiveContext", {}),
        ("intent_script__GetLiveContext", {}),
        # Changes on the remote side, though this house lets its own through.
        ("todo__HassListAddItem", {"name": "Groceries", "item": "Milk"}),
        ("todo__HassListRemoveItem", {"name": "Groceries", "item": "Milk"}),
        ("assist_satellite__HassBroadcast", {"message": "Hello"}),
        ("intent__HassCancelAllTimers", {}),
        ("intent__HassStartTimer", {"minutes": 5}),
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
    assert ' on "Upstairs HA"' in line
    assert line.startswith("run ")
    # This house's devices are not named: the light here is not the door.
    assert "light.garage_door" not in line


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("intent__HassGetState", {"name": "Front Door"}),
        ("HassGetWeather", {}),
        ("intent__HassNevermind", {}),
        ("intent__HassTimerStatus", {}),
        # Marked read-only by the remote server.
        ("GetLiveContext", {}),
        ("calendar__get_events", {"calendar": "Home"}),
    ],
)
async def test_a_remote_read_is_not_held(
    home: HomeAssistant, tool_name: str, args: dict[str, Any]
) -> None:
    """A read-only intent or tool changes nothing, wherever it runs."""
    api = _api([tool_name], read_only=("GetLiveContext", "calendar__get_events"))
    ctx = _ctx(home, api)
    redirect, _ = _prepare_tool_args(tool_name, dict(args), ctx, {})
    assert redirect is None
    assert not ctx.pending_actions


async def test_a_read_only_mark_does_not_clear_a_script(home: HomeAssistant) -> None:
    """A script marked read-only by the server is held all the same."""
    api = _api(["script__open_door"], read_only=("script__open_door",))
    ctx = _ctx(home, api)
    redirect, _ = _prepare_tool_args("script__open_door", {}, ctx, {"id": "c1"})
    assert redirect is not None


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


async def test_another_mcp_servers_tool_is_not_affected() -> None:
    """A server offering no Home Assistant tool is not a remote one."""
    api = _api(["get_current_time", "open_garage_door"], ha_server=False)
    for name in ("get_current_time", "open_garage_door"):
        origin = tool_origin(api, name)
        assert origin.api_id == REMOTE
        assert origin.remote is False


async def test_this_home_assistants_api_is_never_remote() -> None:
    """Its own intent and script tools make no API a remote one."""
    tools = [
        _local_intent(TURN_ON),
        _local_script("script__open_door"),
        _RemoteTool("GetLiveContext"),  # HA's own, neither intent nor action
        _RemoteTool("calendar__get_events"),
    ]
    api = _api([], tools)
    for tool in tools:
        assert tool_origin(api, tool.name).remote is False


async def test_an_unrouted_call_is_read_as_the_fallback_sends_it(
    home: HomeAssistant,
) -> None:
    """With no routing entry, any API offering the name counts (#840 review)."""
    api = _api([TURN_ON], [_local_intent(TURN_ON)])
    api.routing_map = {}
    origin = tool_origin(api, TURN_ON)
    assert origin.remote is True
    assert origin.api_id == REMOTE
    ctx = _ctx(home, api)
    redirect, _ = _prepare_tool_args(TURN_ON, {"name": "Front Door"}, ctx, {"id": "c1"})
    assert redirect is not None


async def test_no_routing_reads_the_name_as_before(home: HomeAssistant) -> None:
    """Without an API to read, nothing is known about the tool."""
    assert tool_origin(None, TURN_ON).remote is False
    assert tool_origin(SimpleNamespace(), TURN_ON).remote is False
    assert not _is_critical_action(
        {"name": "Front Door"}, RECOMMENDED_CRITICAL_ACTIONS, TURN_ON, home
    )
    assert _is_critical_action(
        {"name": "Front Door"},
        RECOMMENDED_CRITICAL_ACTIONS,
        TURN_ON,
        home,
        origin=ToolOrigin(remote=True),
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


async def test_a_request_sent_to_another_home_assistant_is_another_request(
    home: HomeAssistant,
) -> None:
    """A PIN asked for one remote Home Assistant never runs on another."""
    down = _Instance("Downstairs HA", [_RemoteTool(TURN_ON), _RemoteTool(HA_MARKER)])
    api = _api([TURN_ON])
    api.apis["mcp-down"] = down
    ctx = _ctx(home, api)
    ctx.config = {
        "configurable": {
            "thread_id": "t1",
            "pin_reply": PinReply("open the garage", turn_id="t1"),
        }
    }
    redirect, _ = _prepare_tool_args(TURN_ON, {"name": "Garage"}, ctx, {"id": "c1"})
    assert redirect is not None
    ((_, held),) = ctx.pending_actions.items()
    held.update(reply_turn="turn2", disclosed=True)
    # The reply's turn routes the name to the other server, and the model
    # issues the call again.
    api.routing_map = {TURN_ON: "mcp-down"}
    ctx.config = {
        "configurable": {
            "thread_id": "t1",
            "pin_reply": PinReply("1234", turn_id="turn2"),
        }
    }
    _prepare_tool_args(TURN_ON, {"name": "Garage"}, ctx, {"id": "c2"})
    for action in ctx.pending_actions.values():
        if action.get("api_id") == "mcp-down":
            # Asked for anew: it does not take over the first one's prompt.
            assert action.get("disclosed") is not True
            assert action.get("reply_turn") != "turn2"
    assert down.calls == []


async def test_the_line_shows_what_the_call_sends(home: HomeAssistant) -> None:
    """The user sees the arguments the remote side gets (#840 review)."""
    ctx = _ctx(home, _api(["intent__HassTurnOff"]))
    _prepare_tool_args(
        "intent__HassTurnOff",
        {"name": "Front Door", "domain": ["lock"]},
        ctx,
        {"id": "c1"},
    )
    (action,) = ctx.pending_actions.values()
    assert describe_pending_action(home, action) == (
        'run HassTurnOff (name Front Door; domain lock) on "Upstairs HA"'
    )


async def test_the_line_drops_quotes_and_caps_the_agents_words(
    home: HomeAssistant,
) -> None:
    """Text the agent wrote cannot close the server's name or run on."""
    action = {
        "tool_name": TURN_ON,
        "tool_args": {
            "name": 'Porch" on "Cabin\u201d\n',
            "data": {"entity_id": "lock.x"},
            "area": "x" * 200,
        },
        "remote": True,
        "api_name": "Upstairs HA",
    }
    line = describe_pending_action(home, action)
    assert line.startswith("run HassTurnOn (name Porch on Cabin; data \u2026; area ")
    assert line.endswith(' on "Upstairs HA"')
    assert line.count('"') == 2
    assert "\n" not in line
    assert len(line) < 220


async def test_hiding_a_remote_servers_intents_keeps_it_remote() -> None:
    """The server is judged before the user's exclusions (#840 review)."""
    api = _api(["open_garage_door"], ha_server=False)
    # Judged on its full list, which offered an intent the user then hid.
    api.remote_home_assistants = frozenset({REMOTE})
    assert tool_origin(api, "open_garage_door").remote is True
    api.remote_home_assistants = frozenset()
    assert tool_origin(api, "open_garage_door").remote is False


def test_a_server_is_judged_by_every_tool_it_offers() -> None:
    """HA-only tools mark a remote Home Assistant; HA's own API never is one."""
    remote = _Instance("Cabin", [_RemoteTool("GetDateTime"), _RemoteTool("play")])
    assert is_remote_home_assistant(REMOTE, remote) is True
    plain = _Instance("Time", [_RemoteTool("get_current_time")])
    assert is_remote_home_assistant(REMOTE, plain) is False
    # A plain tool named like an intent in this house's own Assist API.
    assist = _Instance(LOCAL, [_RemoteTool("custom__HassPing")])
    assert is_remote_home_assistant(llm.LLM_API_ASSIST, assist) is False


async def test_an_unrouted_name_on_several_apis_is_held(home: HomeAssistant) -> None:
    """Which API runs it is not known, so a read-only mark does not count."""
    api = _api(["calendar__get_events"], read_only=("calendar__get_events",))
    api.apis[LOCAL].tools = [_RemoteTool("calendar__get_events")]
    api.routing_map = {}
    origin = tool_origin(api, "calendar__get_events")
    assert origin.remote is True
    assert origin.read_only is False
    ctx = _ctx(home, api)
    redirect, _ = _prepare_tool_args("calendar__get_events", {}, ctx, {"id": "c1"})
    assert redirect is not None
