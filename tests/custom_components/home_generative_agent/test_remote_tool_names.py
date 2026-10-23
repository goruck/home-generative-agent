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

import homeassistant.util.dt as dt_util
import pytest
import voluptuous as vol
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.helpers import llm
from homeassistant.setup import async_setup_component
from langchain_core.messages import HumanMessage

from custom_components.home_generative_agent.agent.graph import (
    RawTool,
    _append_history_tool,
    _append_included_tools,
    _format_and_dedupe_tools,
    _get_held_action_tools,
    _get_rag_retrieved_tools,
    _get_tool_by_name,
    _is_critical_action,
    _open_state_args,
    _open_state_live_context_normalization_context,
    _prepare_tool_args,
    _retrieve_tools,
)
from custom_components.home_generative_agent.agent.helpers import (
    PinReply,
    remote_tool_name,
)
from custom_components.home_generative_agent.agent.indirect_pin import (
    api_tool_name,
    build_remote_aliases,
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
        self.custom_serializer = None
        self.api_prompt = ""
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
        self.tool_aliases: dict[tuple[str, str], str] | None = None


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


# ----- Review cycle 1: every step that binds a tool keeps both houses apart -----

LIVE_CTX = "homeassistant__GetLiveContext"


class _Item:
    """A tool index row."""

    def __init__(self, name: str, api_id: str, score: float = 0.9) -> None:
        self.value = {
            "name": name,
            "api_id": api_id,
            "description": f"{name} of {api_id}",
            "parameters": "{}",
            "is_actuation": "Hass" in name,
        }
        self.score = score


class _Store:
    """A tool index: search returns `found`; get reads every row by key."""

    def __init__(self, found: list[_Item], rows: list[_Item]) -> None:
        self.found = found
        self.rows = {f"{r.value['api_id']}::{r.value['name']}": r for r in rows}

    async def asearch(self, *_: Any, **__: Any) -> list[_Item]:
        return self.found

    async def aget(self, _namespace: Any, key: str) -> _Item | None:
        return self.rows.get(key)


def _two_houses(local: list[str], remote: list[str]) -> _Wrapper:
    api = _Wrapper()
    api.apis[LOCAL].tools = [_local_intent(n) for n in local]
    api.apis[REMOTE].tools = [_RemoteTool(n) for n in remote]
    api.routing_map = {}
    api.tool_aliases = build_remote_aliases(api)
    return api


def _retrieval_config(api: Any, **configurable: Any) -> Any:
    return {
        "configurable": {
            "options": {
                "llm_hass_api": [LOCAL, REMOTE],
                "tool_relevance_threshold": 0.1,
                "tool_retrieval_limit": 5,
            },
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": api,
            **configurable,
        }
    }


def _state(text: str) -> Any:
    return {
        "messages": [HumanMessage(content=text)],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }


async def test_search_keeps_both_houses_namesakes() -> None:
    """Both houses' HassTurnOn reach the agent, whichever ranks first."""
    names = [TURN_ON, "intent__HassTurnOff"]
    api = _two_houses(names, names)
    rows = [_Item(n, a) for n in names for a in (REMOTE, LOCAL)]
    result = await _retrieve_tools(
        _state("turn on the porch light"),
        _retrieval_config(api),
        store=cast("Any", _Store(rows, rows)),
    )
    routing = result["tool_routing_map"]
    assert routing.get(TURN_ON) == LOCAL
    assert routing.get(REMOTE_TURN_ON) == REMOTE


async def test_this_houses_live_context_is_bound_beside_a_remote_one() -> None:
    """A remote GetLiveContext does not stand in for this house's own."""
    api = _two_houses([LIVE_CTX], [LIVE_CTX])
    remote_ctx, local_ctx = _Item(LIVE_CTX, REMOTE), _Item(LIVE_CTX, LOCAL)
    result = await _retrieve_tools(
        _state("is the garage door open"),
        _retrieval_config(api),
        store=cast("Any", _Store([remote_ctx], [remote_ctx, local_ctx])),
    )
    assert result["tool_routing_map"].get(LIVE_CTX) == LOCAL


async def test_this_houses_turn_on_is_bound_beside_a_remote_one() -> None:
    """A remote turn off does not stand in for this house's turn on (#795)."""
    off = "intent__HassTurnOff"
    api = _two_houses([TURN_ON, off], [TURN_ON, off])
    found = [_Item(off, REMOTE)]
    rows = [_Item(n, a) for n in (TURN_ON, off) for a in (REMOTE, LOCAL)]
    result = await _retrieve_tools(
        _state("open the study shade"),
        _retrieval_config(api),
        store=cast("Any", _Store(found, rows)),
    )
    assert result["tool_routing_map"].get(TURN_ON) == LOCAL


async def test_a_held_remote_tool_is_bound_from_its_own_house() -> None:
    """The PIN reply turn offers the held remote tool, not this house's."""
    api = _two_houses([TURN_ON], [TURN_ON])
    rows = [_Item(TURN_ON, LOCAL), _Item(TURN_ON, REMOTE)]
    cfg = {
        "thread_id": "t1",
        "pending_actions": {
            "a1": {
                "tool_name": REMOTE_TURN_ON,
                "api_id": REMOTE,
                "api_tool_name": TURN_ON,
                "thread_id": "t1",
                "created_at": dt_util.utcnow().isoformat(),
            }
        },
    }
    tools = await _get_held_action_tools(
        cast("Any", _Store([], rows)), _retrieval_config(api, **cfg), cfg
    )
    assert [(t["name"], t["api_id"]) for t in tools] == [(TURN_ON, REMOTE)]


async def test_an_always_included_remote_namesake_is_bound() -> None:
    """A pinned remote tool is not shadowed by this house's namesake."""
    api = _two_houses([TURN_ON], [TURN_ON])
    rows = [_Item(TURN_ON, LOCAL), _Item(TURN_ON, REMOTE)]
    config = _retrieval_config(api)
    config["configurable"]["options"]["tool_inclusions"] = {REMOTE: [TURN_ON]}
    candidates = await _append_included_tools(
        [cast("Any", dict(_Item(TURN_ON, LOCAL).value))],
        cast("Any", _Store([], rows)),
        config,
        None,
    )
    assert {(t["name"], t["api_id"]) for t in candidates} == {
        (TURN_ON, LOCAL),
        (TURN_ON, REMOTE),
    }


def test_two_servers_with_the_same_name_get_distinct_names(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each remote Home Assistant is "home-assistant" until renamed."""
    api = _Wrapper()
    api.apis[REMOTE].api.name = "home-assistant"
    api.apis["mcp-cabin"] = _Instance("home-assistant", [_RemoteTool(TURN_ON)])
    api.remote_home_assistants = frozenset({REMOTE, "mcp-cabin"})
    aliases = build_remote_aliases(api)
    first, second = aliases[(REMOTE, TURN_ON)], aliases[("mcp-cabin", TURN_ON)]
    assert first != second
    assert first.endswith("__intent__HassTurnOn")
    assert second.endswith("__intent__HassTurnOn")
    assert "read alike" in caplog.text


def test_a_remote_name_never_takes_one_of_this_houses() -> None:
    """A server named "Intent" offering HassTurnOn is not intent__HassTurnOn."""
    api = _Wrapper()
    api.apis[REMOTE] = _Instance("Intent", [_RemoteTool("HassTurnOn")])
    alias = build_remote_aliases(api)[(REMOTE, "HassTurnOn")]
    assert alias != TURN_ON
    assert alias.endswith("__HassTurnOn")


async def test_a_cut_short_remote_script_is_still_held(hass: HomeAssistant) -> None:
    """The PIN check reads the tool's own name, not the cut-short one."""
    long_script = "script__" + "a" * 60
    api = _Wrapper()
    api.apis[REMOTE] = _Instance("abcdefghijklmnop", [_RemoteTool(long_script)])
    api.apis[REMOTE].tools[0].annotations = llm.ToolAnnotations(read_only=True)
    api.tool_aliases = build_remote_aliases(api)
    alias = api.tool_aliases[(REMOTE, long_script)]
    assert "script__" not in alias  # cut short to fit 64 characters
    api.routing_map = {alias: REMOTE}
    origin = tool_origin(api, alias)
    assert origin.tool_name == long_script
    assert _is_critical_action(
        {}, RECOMMENDED_CRITICAL_ACTIONS, alias, hass, origin=origin
    )


def test_a_remote_live_context_call_keeps_its_arguments() -> None:
    """The open-state rewrite is for this house's GetLiveContext only."""
    api = _two_houses([LIVE_CTX], [LIVE_CTX])
    remote_name = build_remote_aliases(api)[(REMOTE, LIVE_CTX)]
    api.routing_map = {remote_name: REMOTE, LIVE_CTX: LOCAL}
    args = {"name": "Window", "area": "Bedroom"}
    assert _open_state_args(api, remote_name, dict(args), "x", force_broad=True) == (
        True,
        args,
    )


def test_the_server_name_in_a_description_is_one_clean_line() -> None:
    """A server's name cannot break the description it is put in."""
    selected, _ = _format_and_dedupe_tools(
        [_raw(TURN_ON, REMOTE)], remote_apis={REMOTE: "Up\nstairs\u202e HA"}
    )
    description = selected[0]["function"]["description"]
    assert description.startswith("[Up stairs? HA, another Home Assistant]")


# ----- Review cycle 2 -----


async def test_a_remote_only_setup_still_gets_turn_on_and_live_context() -> None:
    """With no Assist API, the remote house's own are force-bound, renamed."""
    names = [TURN_ON, "intent__HassTurnOff", "intent__HassMediaPause", LIVE_CTX]
    api = _two_houses([], names)
    del api.apis[LOCAL]
    api.tool_aliases = build_remote_aliases(api)
    rows = [_Item(n, REMOTE) for n in names]
    config = _retrieval_config(api)
    config["configurable"]["options"]["llm_hass_api"] = [REMOTE]
    result = await _retrieve_tools(
        _state("open the study shade"),
        config,
        store=cast("Any", _Store([_Item("intent__HassMediaPause", REMOTE)], rows)),
    )
    routing = result["tool_routing_map"]
    assert routing.get(REMOTE_TURN_ON) == REMOTE
    assert routing.get(f"upstairs_ha__{LIVE_CTX}") == REMOTE


async def test_two_plain_servers_namesakes_take_one_slot() -> None:
    """Without a remote Home Assistant, a shared name still takes one slot."""
    api = _Wrapper()
    api.remote_home_assistants = frozenset()
    shared = ["search", "fetch", "a", "b", "c", "d"]
    api.apis = {
        "mcp-x": _Instance("X", [_RemoteTool(n) for n in shared]),
        "mcp-y": _Instance("Y", [_RemoteTool(n) for n in shared]),
    }
    found = [_Item(n, a, 0.9 - i / 100) for i, n in enumerate(shared) for a in api.apis]
    config = _retrieval_config(api)
    config["configurable"]["options"]["llm_hass_api"] = list(api.apis)
    tools, _ = await _get_rag_retrieved_tools(
        cast("Any", _Store(found, found)), config, "query", set(api.apis), None
    )
    assert len({t["name"] for t in tools}) == len(tools) == 5


async def test_a_held_call_runs_by_its_own_name_after_a_rename(
    hass: HomeAssistant,
) -> None:
    """A second server named alike changes the alias; the held call still runs."""
    assert await async_setup_component(hass, "homeassistant", {})
    api = _wrapper()
    api.tool_aliases = build_remote_aliases(api)
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
    assert redirect is not None
    ((action_id, action),) = ctx.pending_actions.items()
    action.update(user="u1", thread_id="conv1", reply_turn="turn2")
    # Before the reply, a second "Upstairs HA" joins: every alias changes.
    api.apis["mcp-upstairs-2"] = _Instance("Upstairs HA", [_RemoteTool(TURN_ON)])
    api.remote_home_assistants = frozenset({REMOTE, "mcp-upstairs-2"})
    api.tool_aliases = build_remote_aliases(api)
    assert api.tool_aliases[(REMOTE, TURN_ON)] != REMOTE_TURN_ON
    hashed, salt = hash_pin("1234")
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
    assert cast("_Instance", api.apis[REMOTE]).calls == [(TURN_ON, {"name": "Porch"})]


async def test_hgas_own_history_tool_is_bound_beside_a_remote_namesake() -> None:
    """A remote server's get_entity_history does not stand in for HGA's."""
    api = _two_houses([], ["get_entity_history"])
    local = _Item("get_entity_history", "hga_local")
    remote = _Item("get_entity_history", REMOTE)
    config = _retrieval_config(api)
    config["configurable"]["options"]["llm_hass_api"] = [REMOTE]
    candidates = await _append_history_tool(
        [cast("Any", dict(remote.value))],
        cast("Any", _Store([], [local, remote])),
        config,
        "what happened to the porch light yesterday",
        [HumanMessage(content="what happened to the porch light yesterday")],
        None,
    )
    assert ("get_entity_history", "hga_local") in {
        (t["name"], t["api_id"]) for t in candidates
    }


def test_a_renamed_tool_never_lands_on_a_taken_name() -> None:
    """Even the hashed name is retried until it is free."""
    taken = remote_tool_name("Intent", "HassTurnOn", REMOTE)
    api = _Wrapper()
    api.apis[LOCAL].tools = [_local_intent(TURN_ON), _local_intent(taken)]
    api.apis[REMOTE] = _Instance("Intent", [_RemoteTool("HassTurnOn")])
    alias = build_remote_aliases(api)[(REMOTE, "HassTurnOn")]
    assert alias not in {TURN_ON, taken}


def test_the_open_state_rewrite_takes_this_houses_call() -> None:
    """A remote GetLiveContext call first does not take this house's turn."""
    api = _two_houses([LIVE_CTX], [LIVE_CTX])
    remote_name = build_remote_aliases(api)[(REMOTE, LIVE_CTX)]
    api.routing_map = {remote_name: REMOTE, LIVE_CTX: LOCAL}
    remote_call = {"name": remote_name, "args": {"name": "Window"}, "id": "r"}
    local_call = {"name": LIVE_CTX, "args": {"name": "Window"}, "id": "l"}
    _, first = _open_state_live_context_normalization_context(
        [HumanMessage(content="which windows are open?")],
        [remote_call, local_call],
        api,
    )
    assert first is local_call


async def test_a_tool_is_found_by_its_exact_name_first() -> None:
    """script__morning and intent_script__morning share a base name."""
    api = _Wrapper()
    api.remote_home_assistants = frozenset()
    api.apis = {
        "mcp-x": _Instance(
            "X",
            [
                cast(
                    "Any",
                    SimpleNamespace(name=n, description=n, parameters=vol.Schema({})),
                )
                for n in ("script__morning", "intent_script__morning")
            ],
        )
    }
    found = await _get_tool_by_name(
        cast("Any", _Store([], [])),
        _retrieval_config(api),
        "intent_script__morning",
        {"mcp-x"},
        None,
    )
    assert found is not None
    assert found["name"] == "intent_script__morning"
