# ruff: noqa: S101
"""
A delayed command reaches the timer tool once, with the delay given once.

Field test 2026-11-27, on a voice satellite:

- "Unlock the playroom door lock in one minute." The model gave the timer the
  whole sentence as its command. When the timer ended, Home Assistant's own
  agent read "in one minute" as a new delayed command and started a second
  timer, so the door unlocked after two minutes.
- "Lock the garage door lock in 30 seconds." Ranking bound the device tools
  and left the timer tool out. The model's call to it was rejected as not
  retrieved, and it offered to lock the door at once instead.
"""

from __future__ import annotations

import ast
import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent import graph
from custom_components.home_generative_agent.agent.graph import (
    State,
    _query_wants_delayed_command,
    _retrieve_tools,
    _run_ha_tool,
)
from custom_components.home_generative_agent.agent.helpers import (
    command_sets_its_own_delay,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from homeassistant.core import HomeAssistant
    from langchain_core.runnables import RunnableConfig

TIMER = "intent__HassStartTimer"


@pytest.mark.parametrize(
    "query",
    [
        "Lock the garage door lock in 30 seconds.",
        "Unlock the playroom door lock in one minute.",
        "turn off the fan after an hour",
        "close the garage in a few minutes",
        "turn on the porch light in half an hour",
        "open the shade in 5 min",
        "turn off the heater in 1.5 hours",
    ],
)
def test_delay_phrasings_are_read(query: str) -> None:
    """A request to act after a delay asks for the timer tool."""
    assert _query_wants_delayed_command(query)


@pytest.mark.parametrize(
    "query",
    [
        "Lock the garage door lock.",
        "turn on the light in the kitchen",
        "what's the temperature in the living room",
        "turn on the fan every 5 minutes",
        "set the thermostat to 70 in one room",
    ],
)
def test_requests_without_a_delay_are_not(query: str) -> None:
    """A request with no delay leaves the timer tool to ranking."""
    assert not _query_wants_delayed_command(query)


def _item(name: str, *, score: float = 0.5, is_actuation: bool = False) -> MagicMock:
    item = MagicMock()
    item.value = {
        "name": name,
        "api_id": "assist",
        "description": f"{name} description",
        "parameters": "{}",
        "is_actuation": is_actuation,
    }
    item.score = score
    return item


def _store() -> MagicMock:
    """Ranking finds the device tools for the command's words, never the timer."""
    store = MagicMock()
    store.asearch = AsyncMock(
        return_value=[
            _item("intent__HassTurnOn", score=0.62, is_actuation=True),
            _item("intent__HassTurnOff", score=0.61, is_actuation=True),
            _item("vacuum__HassVacuumStart", score=0.55, is_actuation=True),
        ]
    )
    timer = _item(TIMER)

    async def aget(_namespace: Any, key: str = "", **_kwargs: Any) -> Any:
        return timer if key == f"assist::{TIMER}" else None

    store.aget = AsyncMock(side_effect=aget)
    return store


def _state(text: str) -> State:
    return {
        "messages": [MagicMock(content=text)],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }


def _config(*live: str, **options: Any) -> RunnableConfig:
    api = SimpleNamespace(
        tools=[
            SimpleNamespace(
                name=name,
                description=f"{name} live",
                parameters={"type": "object", "properties": {}},
            )
            for name in live
        ],
        custom_serializer=None,
    )
    return {
        "configurable": {
            "options": {
                "llm_hass_api": ["assist"],
                "tool_relevance_threshold": 0.15,
                **options,
            },
            "tool_index_ready": True,
            "langchain_tools": {},
            "ha_llm_api": MagicMock(apis={"assist": api}),
        }
    }


_SATELLITE_TOOLS = (
    "intent__HassTurnOn",
    "intent__HassTurnOff",
    "vacuum__HassVacuumStart",
    TIMER,
)


@pytest.mark.asyncio
async def test_timer_tool_is_bound_for_a_delayed_command() -> None:
    """The field repro: the timer tool is bound beside the device tools."""
    result = await _retrieve_tools(
        _state("Lock the garage door lock in 30 seconds."),
        _config(*_SATELLITE_TOOLS),
        store=_store(),
    )

    assert result["tool_routing_map"][TIMER] == "assist"
    # Bound outside the limit: nothing ranking chose is evicted.
    assert "intent__HassTurnOn" in result["tool_routing_map"]
    assert "intent__HassTurnOff" in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_timer_tool_is_not_bound_without_a_delay() -> None:
    """A plain command does not widen the tool list."""
    result = await _retrieve_tools(
        _state("Lock the garage door lock."),
        _config(*_SATELLITE_TOOLS),
        store=_store(),
    )

    assert TIMER not in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_timer_tool_is_not_bound_where_it_is_not_live() -> None:
    """Home Assistant offers timers to timer-capable satellites alone."""
    result = await _retrieve_tools(
        _state("Lock the garage door lock in 30 seconds."),
        _config("intent__HassTurnOn", "intent__HassTurnOff"),
        store=_store(),
    )

    assert TIMER not in result["tool_routing_map"]


@pytest.mark.asyncio
async def test_excluded_timer_tool_is_not_bound() -> None:
    """An exclusion of the timer tool keeps it out."""
    result = await _retrieve_tools(
        _state("Lock the garage door lock in 30 seconds."),
        _config(*_SATELLITE_TOOLS, tool_exclusions={"assist": [TIMER]}),
        store=_store(),
    )

    assert TIMER not in result["tool_routing_map"]


_CONVERSATION_MODULES = (
    "homeassistant.components.conversation",
    "hassil",
    "home_assistant_intents",
)


def _is_conversation_module(name: str) -> bool:
    return any(name == m or name.startswith(f"{m}.") for m in _CONVERSATION_MODULES)


@contextmanager
def _real_conversation_modules() -> Iterator[None]:
    """
    Load Home Assistant's real conversation component inside the block.

    Other test modules stub it in `sys.modules` (test_conversation_units,
    test_conversation_stream), and a stub has no agent to read a command.
    The stubs are put back afterwards, and the real modules taken out. An
    import also sets the module as an attribute of its parent package
    (`homeassistant.components.conversation`), so that is put back too:
    otherwise `from homeassistant.components import conversation` would
    reach the module this test loaded, not the one `sys.modules` holds.
    """
    from homeassistant import components  # noqa: PLC0415

    missing = object()
    saved = {n: m for n, m in sys.modules.items() if _is_conversation_module(n)}
    saved_attr = getattr(components, "conversation", missing)
    for name in saved:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in [n for n in sys.modules if _is_conversation_module(n)]:
            del sys.modules[name]
        sys.modules.update(saved)
        if saved_attr is missing:
            if hasattr(components, "conversation"):
                delattr(components, "conversation")
        else:
            components.conversation = saved_attr  # type: ignore[attr-defined]


@pytest.fixture
def real_conversation() -> Iterator[None]:
    """Run the test with Home Assistant's real conversation component."""
    with _real_conversation_modules():
        yield


def test_module_swap_leaves_import_state_as_it_found_it() -> None:
    """Afterwards, the package attribute and `sys.modules` are as they were."""
    from homeassistant import components  # noqa: PLC0415

    name = "homeassistant.components.conversation"
    importlib.import_module(name)
    before = sys.modules[name]
    attr_before = getattr(components, "conversation", None)

    with _real_conversation_modules():
        # As Home Assistant's loader imports it: through `sys.modules`, which
        # also points the package attribute at the fresh module.
        fresh = importlib.import_module(name)

        assert fresh is not before
        assert getattr(components, "conversation", None) is fresh

    assert sys.modules[name] is before
    assert getattr(components, "conversation", None) is attr_before


@pytest.fixture
async def home(
    real_conversation: None,  # noqa: ARG001
    hass: HomeAssistant,
) -> HomeAssistant:
    """Set up Home Assistant's own agent with an exposed lock."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})
    assert await async_setup_component(hass, "intent", {})
    hass.states.async_set("lock.front_door", "locked", {"friendly_name": "Front Door"})
    async_expose_entity(hass, "conversation", "lock.front_door", should_expose=True)
    return hass


async def test_command_naming_its_delay_is_read_as_a_timer(home: HomeAssistant) -> None:
    """Home Assistant's agent would start a second timer for this command."""
    assert await command_sets_its_own_delay(
        home,
        "Unlock the front door in one minute.",
        language="en",
        device_id=None,
    )


@pytest.mark.parametrize("command", ["Unlock the front door", ""])
async def test_command_alone_is_not(home: HomeAssistant, command: str) -> None:
    """The action alone runs once, when the timer ends."""
    assert not await command_sets_its_own_delay(
        home, command, language="en", device_id=None
    )


async def test_unreadable_command_is_left_to_the_gate(home: HomeAssistant) -> None:
    """A failure reading the command answers False: the PIN gate still holds it."""
    with patch(
        "homeassistant.components.conversation.default_agent.DefaultAgent."
        "async_recognize_intent",
        side_effect=RuntimeError("boom"),
    ):
        assert not await command_sets_its_own_delay(
            home, "Unlock the front door in one minute.", language="en", device_id=None
        )


def _ctx(hass: HomeAssistant) -> MagicMock:
    ha_llm_api = MagicMock()
    ha_llm_api.apis = {
        "assist": SimpleNamespace(
            llm_context=SimpleNamespace(language="en", device_id=None)
        )
    }
    ha_llm_api.async_call_tool = AsyncMock()
    ctx = MagicMock()
    ctx.hass = hass
    ctx.ha_llm_api = ha_llm_api
    return ctx


def _origin(*, remote: bool) -> SimpleNamespace:
    return SimpleNamespace(remote=remote, ambiguous=False)


async def test_nested_delay_is_refused_before_the_pin(home: HomeAssistant) -> None:
    """The model is told to drop the delay; nothing is held or run."""
    ctx = _ctx(home)
    prepare = MagicMock()
    with (
        patch.object(graph, "tool_origin", return_value=_origin(remote=False)),
        patch.object(graph, "_prepare_tool_args", prepare),
    ):
        reply = await _run_ha_tool(
            TIMER,
            {
                "minutes": 1,
                "conversation_command": "Unlock the front door in one minute.",
            },
            ctx,
            {"id": "call-1"},
        )

    assert reply.status == "error"
    assert "names its own delay" in str(reply.content)
    assert "conversation_command" in str(reply.content)
    prepare.assert_not_called()
    ctx.ha_llm_api.async_call_tool.assert_not_called()


async def test_command_alone_goes_on_to_the_gate(home: HomeAssistant) -> None:
    """A command without its delay reaches the PIN gate as before."""
    ctx = _ctx(home)
    held = MagicMock(content=json.dumps({"status": "requires_pin"}))
    prepare = MagicMock(return_value=(held, {}))
    with (
        patch.object(graph, "tool_origin", return_value=_origin(remote=False)),
        patch.object(graph, "_prepare_tool_args", prepare),
    ):
        reply = await _run_ha_tool(
            TIMER,
            {"minutes": 1, "conversation_command": "Unlock the front door"},
            ctx,
            {"id": "call-1"},
        )

    assert reply is held
    prepare.assert_called_once()


async def test_remote_delayed_command_is_left_to_the_gate(home: HomeAssistant) -> None:
    """A remote Home Assistant runs the command on its own agent: not read here."""
    ctx = _ctx(home)
    held = MagicMock()
    prepare = MagicMock(return_value=(held, {}))
    reader = AsyncMock(return_value=True)
    with (
        patch.object(graph, "tool_origin", return_value=_origin(remote=True)),
        patch.object(graph, "_prepare_tool_args", prepare),
        patch.object(graph, "command_sets_its_own_delay", reader),
    ):
        reply = await _run_ha_tool(
            TIMER,
            {
                "minutes": 1,
                "conversation_command": "Unlock the front door in one minute.",
            },
            ctx,
            {"id": "call-1"},
        )

    assert reply is held
    reader.assert_not_called()


# Review of PR #1099.


@pytest.mark.parametrize(
    "query",
    [
        "turn off the fan in eleven minutes",
        "turn off the fan in twenty one minutes",
        "lock the door in forty five seconds",
        "close the garage in twenty-five minutes",
        "open the shade in 30-seconds",
        "lock the door in 10 more minutes",
        "turn off the heater in two and a half hours",
        "unlock the door in twelve minutes",
    ],
)
def test_more_spoken_delays_are_read(query: str) -> None:
    """Number words and phrasings Home Assistant's own sentences accept."""
    assert _query_wants_delayed_command(query)


@pytest.mark.asyncio
async def test_timer_tool_already_ranked_is_not_fetched_again() -> None:
    """A timer ranking already bound is kept once, with no extra lookup."""
    store = _store()
    store.asearch = AsyncMock(
        return_value=[
            _item("intent__HassTurnOn", score=0.62, is_actuation=True),
            _item(TIMER, score=0.6),
        ]
    )
    result = await _retrieve_tools(
        _state("Lock the garage door lock in 30 seconds."),
        _config(*_SATELLITE_TOOLS),
        store=store,
    )

    assert result["tool_routing_map"][TIMER] == "assist"
    timer_lookups = [
        call
        for call in store.aget.call_args_list
        if str(call.kwargs.get("key", "")).endswith(TIMER)
    ]
    assert not timer_lookups


@pytest.mark.parametrize(
    "command", ["start a timer for 5 minutes", "set a five minute timer"]
)
async def test_command_that_sets_a_timer_is_not_a_doubled_delay(
    home: HomeAssistant, command: str
) -> None:
    """
    "In ten minutes, start a five-minute timer" is a real request.

    Home Assistant reads the command as a timer with no command of its own,
    so it rings once; only a timer that carries a command doubles the delay.
    """
    assert not await command_sets_its_own_delay(
        home, command, language="en", device_id=None
    )


async def test_sentence_trigger_runs_instead_of_a_timer(home: HomeAssistant) -> None:
    """A user's sentence trigger wins over the built-in timer sentence, as in HA."""
    assert await async_setup_component(
        home,
        "automation",
        {
            "automation": {
                "triggers": {
                    "trigger": "conversation",
                    "command": "unlock the front door in one minute",
                },
                "actions": [],
            }
        },
    )
    await home.async_block_till_done()

    assert not await command_sets_its_own_delay(
        home, "unlock the front door in one minute", language="en", device_id=None
    )


async def test_unreadable_command_is_warned_about_once(
    home: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """A failing check is said once in the log, not hidden at debug level."""
    helpers_module = sys.modules[command_sets_its_own_delay.__module__]
    with (
        patch.object(helpers_module, "_DELAY_CHECK_WARNED", new=False),
        patch(
            "homeassistant.components.conversation.default_agent.DefaultAgent."
            "async_recognize_intent",
            side_effect=RuntimeError("boom"),
        ),
    ):
        for _ in range(2):
            assert not await command_sets_its_own_delay(
                home,
                "Unlock the front door in one minute.",
                language="en",
                device_id=None,
            )

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "delayed command" in warnings[0].getMessage()


async def test_scheduled_command_is_left_to_the_gate(home: HomeAssistant) -> None:
    """The refusal names a timer's arguments, so it is for the timer alone."""
    ctx = _ctx(home)
    held = MagicMock()
    prepare = MagicMock(return_value=(held, {}))
    with (
        patch.object(graph, "tool_origin", return_value=_origin(remote=False)),
        patch.object(graph, "_prepare_tool_args", prepare),
    ):
        reply = await _run_ha_tool(
            "intent__schedule_command",
            {"command": "Unlock the front door in one minute.", "minutes": 1},
            ctx,
            {"id": "call-1"},
        )

    assert reply is held


async def test_ambiguous_origin_is_left_to_the_gate(home: HomeAssistant) -> None:
    """A tool whose API cannot be told is never read as this house's."""
    ctx = _ctx(home)
    held = MagicMock()
    prepare = MagicMock(return_value=(held, {}))
    reader = AsyncMock(return_value=True)
    with (
        patch.object(
            graph,
            "tool_origin",
            return_value=SimpleNamespace(remote=False, ambiguous=True),
        ),
        patch.object(graph, "_prepare_tool_args", prepare),
        patch.object(graph, "command_sets_its_own_delay", reader),
    ):
        reply = await _run_ha_tool(
            TIMER,
            {"minutes": 1, "conversation_command": "Unlock the front door"},
            ctx,
            {"id": "call-1"},
        )

    assert reply is held
    reader.assert_not_called()


async def test_command_is_read_in_the_routed_apis_language(
    home: HomeAssistant,
) -> None:
    """The language and device come from the API the timer is sent to."""
    ctx = _ctx(home)
    ctx.ha_llm_api.apis = {
        "other_mcp": SimpleNamespace(
            llm_context=SimpleNamespace(language="fr", device_id="mcp-device")
        ),
        "assist": SimpleNamespace(
            llm_context=SimpleNamespace(language="de", device_id="satellite")
        ),
    }
    ctx.ha_llm_api.routing_map = {TIMER: "assist"}
    reader = AsyncMock(return_value=False)
    with (
        patch.object(graph, "tool_origin", return_value=_origin(remote=False)),
        patch.object(graph, "_prepare_tool_args", MagicMock(return_value=(None, {}))),
        patch.object(graph, "command_sets_its_own_delay", reader),
    ):
        await _run_ha_tool(
            TIMER,
            {"minutes": 1, "conversation_command": "Schließe die Haustür ab"},
            ctx,
            {"id": "call-1"},
        )

    assert reader.await_args is not None
    assert reader.await_args.kwargs == {"language": "de", "device_id": "satellite"}


# Review of PR #1099, cycle 2.


@pytest.mark.parametrize(
    "query",
    [
        "lock the door in 5m",
        "turn off the fan in 30s",
        "close the garage in 1h",
        "unlock the door 10 minutes from now",
        "turn on the porch light twenty minutes from now",
    ],
)
def test_short_and_from_now_delays_are_read(query: str) -> None:
    """Typed units and "N minutes from now" name a delay too."""
    assert _query_wants_delayed_command(query)


@pytest.mark.parametrize(
    "query",
    ["is the door locked now", "turn on room 5s light", "dim it to 5m"],
)
def test_short_units_need_a_delay_phrase(query: str) -> None:
    """A bare letter after a number is a delay only after "in" or "after"."""
    assert not _query_wants_delayed_command(query)


_CONVERSATION_SOURCE = Path(graph.__file__).resolve().parents[1] / "conversation.py"


def test_agent_llm_context_names_no_agent() -> None:
    """
    Tripwire: the doubled-delay check reads commands on HA's own agent.

    Home Assistant runs a timer's command on the agent the LLM context names,
    and this integration names none, so it runs on Home Assistant's own agent
    (`command_sets_its_own_delay`). Naming one moves it to this agent: update
    that check and the PIN hold (TODOS.md) in the same change, then this test.
    """
    tree = ast.parse(_CONVERSATION_SOURCE.read_text(encoding="utf-8"))
    contexts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "LLMContext"
    ]

    assert contexts
    assert all(
        keyword.arg != "agent_id" for call in contexts for keyword in call.keywords
    )


_NESTED_CALL = {
    "minutes": 1,
    "conversation_command": "Unlock the front door in one minute.",
}


async def test_refusal_leaves_a_pending_pin_alone(home: HomeAssistant) -> None:
    """A PIN already asked for stays pending; the refused call adds nothing."""
    ctx = _ctx(home)
    pending = {"held-1": {"tool_name": "intent__HassTurnOff"}}
    ctx.pending_actions = pending
    prepare = MagicMock()
    with (
        patch.object(graph, "tool_origin", return_value=_origin(remote=False)),
        patch.object(graph, "_prepare_tool_args", prepare),
    ):
        reply = await _run_ha_tool(TIMER, dict(_NESTED_CALL), ctx, {"id": "c1"})

    assert reply.status == "error"
    assert pending == {"held-1": {"tool_name": "intent__HassTurnOff"}}
    prepare.assert_not_called()


async def test_repeated_nested_delay_is_refused_each_time(home: HomeAssistant) -> None:
    """Each retry with the delay still in the command is refused; nothing is held."""
    ctx = _ctx(home)
    prepare = MagicMock()
    with (
        patch.object(graph, "tool_origin", return_value=_origin(remote=False)),
        patch.object(graph, "_prepare_tool_args", prepare),
    ):
        replies = [
            await _run_ha_tool(TIMER, dict(_NESTED_CALL), ctx, {"id": f"c{n}"})
            for n in range(2)
        ]

    assert [r.status for r in replies] == ["error", "error"]
    prepare.assert_not_called()
    ctx.ha_llm_api.async_call_tool.assert_not_called()


# Review of PR #1099, cycle 3.


@pytest.mark.parametrize(
    "query",
    [
        "turn off the oven in 2h30m",
        "lock the door in 1h30",
        "turn off the fan in one hundred seconds",
        "close the garage in a hundred and twenty seconds",
    ],
)
def test_compact_and_hundreds_delays_are_read(query: str) -> None:
    """Typed hours-and-minutes and spoken hundreds name a delay too."""
    assert _query_wants_delayed_command(query)


_SCRIPT_TIMER = "intent_script__HassStartTimer"


@pytest.mark.asyncio
async def test_intent_script_named_like_the_timer_is_not_the_timer() -> None:
    """A ranked intent script of that name does not stand in for the timer."""
    store = _store()
    store.asearch = AsyncMock(
        return_value=[
            _item("intent__HassTurnOn", score=0.62, is_actuation=True),
            _item(_SCRIPT_TIMER, score=0.6),
        ]
    )
    result = await _retrieve_tools(
        _state("Lock the garage door lock in 30 seconds."),
        _config(*_SATELLITE_TOOLS, _SCRIPT_TIMER),
        store=store,
    )

    assert result["tool_routing_map"][TIMER] == "assist"


@pytest.mark.asyncio
async def test_excluding_the_script_keeps_the_timer() -> None:
    """Excluding an intent script named HassStartTimer leaves the timer bound."""
    result = await _retrieve_tools(
        _state("Lock the garage door lock in 30 seconds."),
        _config(
            *_SATELLITE_TOOLS,
            _SCRIPT_TIMER,
            tool_exclusions={"assist": [_SCRIPT_TIMER]},
        ),
        store=_store(),
    )

    assert result["tool_routing_map"][TIMER] == "assist"
