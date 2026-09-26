# ruff: noqa: S101
"""Tests for correction rounds: a refused automation must not eat its retry."""

from __future__ import annotations

from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from custom_components.home_generative_agent.agent import graph as graph_module
from custom_components.home_generative_agent.agent.automation_targets import (
    AUTOMATION_REFUSAL_PREFIX,
)
from custom_components.home_generative_agent.agent.graph import (
    _MAX_ACTION_ROUNDS,
    _MAX_CORRECTION_ROUNDS,
    _is_automation_refusal,
    _next_round_counters,
    _should_continue,
    _tool_loop_guard,
)

REFUSAL = (
    f"{AUTOMATION_REFUSAL_PREFIX}: it refers to entities or services that do not "
    "exist in Home Assistant, so it would never run. Correct these and call the "
    "tool again with the full automation:\n- Entity 'binary_sensor.x' does not exist."
)


def _refusal(name: str = "add_automation") -> ToolMessage:
    return ToolMessage(content=REFUSAL, name=name, tool_call_id="t1")


def _added() -> ToolMessage:
    return ToolMessage(
        content="Added automation 01X", name="add_automation", tool_call_id="t2"
    )


def _state(**counters: int) -> Any:
    return {
        "messages": [],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
        **counters,
    }


def test_refusal_detection_needs_the_tool_name_and_the_prefix() -> None:
    assert _is_automation_refusal(_refusal())
    # HA may namespace the tool; the base name still counts.
    assert _is_automation_refusal(_refusal("hga_local__add_automation"))
    assert not _is_automation_refusal(_added())
    assert not _is_automation_refusal(
        ToolMessage(
            content=REFUSAL, name="homeassistant__GetLiveContext", tool_call_id="t"
        )
    )
    assert not _is_automation_refusal(AIMessage(content=REFUSAL))


def test_refusal_round_is_a_correction_round_until_the_budget_is_spent() -> None:
    state = _state(action_rounds=2, correction_rounds=0)

    first = _next_round_counters(state, [_refusal()])
    assert first == {"action_rounds": 2, "correction_rounds": 1}

    second = _next_round_counters(_state(**first), [_refusal()])
    assert second == {"action_rounds": 2, "correction_rounds": 2}

    # Budget spent: a third refusal is an action round like any other.
    third = _next_round_counters(_state(**second), [_refusal()])
    assert third == {"action_rounds": 3, "correction_rounds": 2}
    assert _MAX_CORRECTION_ROUNDS == 2


def test_mixed_or_successful_rounds_are_action_rounds() -> None:
    state = _state(action_rounds=0, correction_rounds=0)

    lookup = ToolMessage(
        content="{}", name="homeassistant__GetLiveContext", tool_call_id="t"
    )
    assert _next_round_counters(state, [lookup]) == {
        "action_rounds": 1,
        "correction_rounds": 0,
    }
    assert _next_round_counters(state, [_refusal(), lookup]) == {
        "action_rounds": 1,
        "correction_rounds": 0,
    }
    assert _next_round_counters(state, [_added()]) == {
        "action_rounds": 1,
        "correction_rounds": 0,
    }
    assert _next_round_counters(state, []) == {
        "action_rounds": 1,
        "correction_rounds": 0,
    }


def test_field_turn_now_reaches_the_corrected_call() -> None:
    """Two lookups, one refusal, then the corrected call: the guard must not fire."""
    counters = {"action_rounds": 0, "correction_rounds": 0}
    lookup = ToolMessage(
        content="{}", name="homeassistant__GetLiveContext", tool_call_id="t"
    )
    for responses in ([lookup], [lookup], [_refusal()]):
        counters = _next_round_counters(_state(**counters), responses)

    retry = AIMessage(
        content="", tool_calls=[{"name": "add_automation", "args": {}, "id": "c"}]
    )
    state = _state(**counters)
    state["messages"] = [HumanMessage(content="always notify me"), retry]

    assert counters["action_rounds"] == 2
    assert _should_continue(cast("Any", state)) == "action"


def test_guard_still_fires_at_the_action_limit() -> None:
    retry = AIMessage(
        content="", tool_calls=[{"name": "add_automation", "args": {}, "id": "c"}]
    )
    state = _state(
        action_rounds=_MAX_ACTION_ROUNDS, correction_rounds=_MAX_CORRECTION_ROUNDS
    )
    state["messages"] = [retry]

    assert _should_continue(cast("Any", state)) == "tool_loop_guard"


@pytest.mark.asyncio
async def test_guard_message_repeats_the_last_refusal() -> None:
    state = _state(action_rounds=3)
    state["messages"] = [
        HumanMessage(content="always notify me"),
        AIMessage(
            content="", tool_calls=[{"name": "add_automation", "args": {}, "id": "a"}]
        ),
        _refusal(),
        AIMessage(
            content="", tool_calls=[{"name": "add_automation", "args": {}, "id": "b"}]
        ),
    ]

    result = await _tool_loop_guard(cast("Any", state))

    text = result["messages"][0].content
    assert text.startswith("I wasn't able to complete this request")
    assert "The last problem was:" in text
    assert "binary_sensor.x" in text


@pytest.mark.asyncio
async def test_guard_message_is_generic_after_an_ordinary_round() -> None:
    state = _state(action_rounds=3)
    state["messages"] = [
        AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "a"}]),
        ToolMessage(
            content="{}", name="homeassistant__GetLiveContext", tool_call_id="a"
        ),
        AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "b"}]),
    ]

    result = await _tool_loop_guard(cast("Any", state))

    assert "The last problem was" not in result["messages"][0].content


def test_graph_module_exports_the_counters() -> None:
    assert graph_module._MAX_ACTION_ROUNDS == 3
