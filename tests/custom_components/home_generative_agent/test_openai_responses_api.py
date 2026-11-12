# ruff: noqa: S101
"""OpenAI models on the Responses API: routing, effort, the request and the reply."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai.chat_models import base as lc_openai

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.core.openai_client import (
    drop_reasoning_items,
)
from custom_components.home_generative_agent.core.utils import (
    openai_api_configurable,
    openai_uses_responses_api,
    thinking_configurable,
)


@pytest.mark.parametrize(
    ("model", "responses"),
    [
        ("gpt-6-astra", True),
        ("gpt-6.1-sol", True),
        ("gpt-6.1-terra", True),
        ("gpt-6-luna", True),
        ("GPT-6-Luna", True),
        ("gpt-5.4", True),
        ("gpt-5.5-mini", True),
        ("gpt-5.6-sol", True),
        ("gpt-5.6-terra", True),
        ("gpt-5-pro", True),
        ("gpt-5.2-pro", True),
        ("gpt-5-codex", True),
        # Chat Completions, as before.
        ("gpt-5", False),
        ("gpt-5-mini", False),
        ("gpt-5-nano", False),
        ("gpt-5.1", False),
        ("gpt-5.3-chat-latest", False),
        ("gpt-4o", False),
        ("gpt-4.1-nano", False),
        ("o4-mini", False),
        ("", False),
        (None, False),
    ],
)
def test_models_that_need_the_responses_api(model: Any, *, responses: bool) -> None:
    assert openai_uses_responses_api(model) is responses


def test_a_responses_model_sends_no_sampling_values() -> None:
    """OpenAI's reasoning models take only the default temperature and top_p."""
    assert openai_api_configurable("gpt-6-luna") == {
        "use_responses_api": True,
        "temperature": None,
        "top_p": None,
    }
    assert openai_api_configurable("gpt-5") == {"use_responses_api": False}


@pytest.mark.parametrize(
    ("model", "reasoning", "expected"),
    [
        # Off: none where the model takes it, else its lowest level.
        ("gpt-6-luna", False, {"reasoning_effort": "none"}),
        ("gpt-6.1-terra", False, {"reasoning_effort": "none"}),
        ("gpt-6-astra", False, {"reasoning_effort": "low"}),
        ("gpt-6.1-sol", False, {"reasoning_effort": "low"}),
        # GPT-6 has no minimal.
        ("gpt-6-luna", "minimal", {"reasoning_effort": "low"}),
        ("gpt-6-astra", "xhigh", {"reasoning_effort": "xhigh"}),
        ("gpt-6.1-sol", "max", {"reasoning_effort": "max"}),
        ("gpt-6-luna", "medium", {"reasoning_effort": "medium"}),
        ("gpt-6-luna", True, {}),
        ("gpt-6-luna", None, {}),
        # Earlier models: as before, and no GPT-6 levels.
        ("gpt-5", "minimal", {"reasoning_effort": "minimal"}),
        ("gpt-5", False, {}),
        ("gpt-5", "xhigh", {}),
        ("gpt-4o", False, {}),
    ],
)
def test_reasoning_effort_per_model(model: str, reasoning: Any, expected: dict) -> None:
    assert (
        thinking_configurable(provider_type="openai", reasoning=reasoning, model=model)
        == expected
    )


def _sse(events: list[dict[str, Any]]) -> bytes:
    lines = []
    for number, event in enumerate(events):
        numbered = {**event, "sequence_number": number}
        lines.append(f"event: {event['type']}\ndata: {json.dumps(numbered)}\n\n")
    return "".join(lines).encode()


_REASONING = {
    "id": "rs_1",
    "type": "reasoning",
    "summary": [],
    "encrypted_content": "ENC",
}
_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Turning it on.", "annotations": []}],
}
_CALL = {
    "id": "fc_1",
    "type": "function_call",
    "call_id": "call_1",
    "name": "HassTurnOn",
    "arguments": '{"name": "Kitchen light"}',
    "status": "completed",
}


def _response(output: list[dict[str, Any]], status: str) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "gpt-6-luna",
        "status": status,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": output,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 3},
        },
    }


# A recorded-shape stream: reasoning, then text in two deltas, then a tool call.
_STREAM = [
    {"type": "response.created", "response": _response([], "in_progress")},
    {"type": "response.output_item.added", "output_index": 0, "item": _REASONING},
    {"type": "response.output_item.done", "output_index": 0, "item": _REASONING},
    {
        "type": "response.output_item.added",
        "output_index": 1,
        "item": {**_MESSAGE, "status": "in_progress", "content": []},
    },
    {
        "type": "response.content_part.added",
        "item_id": "msg_1",
        "output_index": 1,
        "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    },
    *(
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 1,
            "content_index": 0,
            "delta": delta,
            "logprobs": [],
        }
        for delta in ("Turning ", "it on.")
    ),
    {
        "type": "response.output_text.done",
        "item_id": "msg_1",
        "output_index": 1,
        "content_index": 0,
        "text": "Turning it on.",
        "logprobs": [],
    },
    {
        "type": "response.content_part.done",
        "item_id": "msg_1",
        "output_index": 1,
        "content_index": 0,
        "part": _MESSAGE["content"][0],
    },
    {"type": "response.output_item.done", "output_index": 1, "item": _MESSAGE},
    {
        "type": "response.output_item.added",
        "output_index": 2,
        "item": {**_CALL, "arguments": "", "status": "in_progress"},
    },
    {
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_1",
        "output_index": 2,
        "delta": _CALL["arguments"],
    },
    {
        "type": "response.function_call_arguments.done",
        "item_id": "fc_1",
        "output_index": 2,
        "name": "HassTurnOn",
        "arguments": _CALL["arguments"],
    },
    {"type": "response.output_item.done", "output_index": 2, "item": _CALL},
    {
        "type": "response.completed",
        "response": _response([_REASONING, _MESSAGE, _CALL], "completed"),
    },
]


_CHAT_CHUNK = {
    "id": "chatcmpl-1",
    "object": "chat.completion.chunk",
    "created": 1,
    "model": "gpt-5",
    "choices": [
        {
            "index": 0,
            "delta": {"role": "assistant", "content": "Hi."},
            "finish_reason": None,
        }
    ],
}
_CHAT_COMPLETIONS_STREAM = (
    f"data: {json.dumps(_CHAT_CHUNK)}\n\n"
    f"data: {json.dumps({**_CHAT_CHUNK, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    "data: [DONE]\n\n"
).encode()


@tool
def HassTurnOn(name: str) -> str:  # noqa: N802 - HA's intent tool name
    """Turn on a device."""
    return name


def _client(
    requests: list[httpx.Request], stream: list[dict[str, Any]] | None = None
) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                content=_CHAT_COMPLETIONS_STREAM,
                headers={"content-type": "text/event-stream"},
            )
        if not json.loads(request.content).get("stream"):
            return httpx.Response(
                200, json=_response([_REASONING, _MESSAGE, _CALL], "completed")
            )
        return httpx.Response(
            200,
            content=_sse(_STREAM if stream is None else stream),
            headers={"content-type": "text/event-stream"},
        )

    return hga_component.build_openai_chat(
        "sk-test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _configured(model: Any, name: str, reasoning: Any = None) -> Any:
    return model.with_config(
        config={
            "configurable": {
                "model_name": name,
                "temperature": 0.7,
                "top_p": 1.0,
                **openai_api_configurable(name),
                **thinking_configurable(
                    provider_type="openai", reasoning=reasoning, model=name
                ),
            }
        }
    )


async def test_a_gpt6_tool_turn_streams_through_the_responses_api() -> None:
    requests: list[httpx.Request] = []
    model = _configured(_client(requests), "gpt-6-luna", reasoning=False).bind_tools(
        [HassTurnOn]
    )

    chunks = [
        chunk async for chunk in model.astream([HumanMessage("turn on the kitchen")])
    ]

    (request,) = requests
    assert request.url.path.endswith("/responses")
    body = json.loads(request.content)
    assert body["model"] == "gpt-6-luna"
    assert body["store"] is False
    assert body["reasoning"]["effort"] == "none"
    assert "temperature" not in body
    assert "top_p" not in body
    assert [t["name"] for t in body["tools"]] == ["HassTurnOn"]

    # Each streamed text delta reads as text for the chat log and TTS.
    texts = [
        text
        for chunk in chunks
        if (text := _text(chunk.content))  # what _normalize_ai_content keeps
    ]
    assert "".join(texts) == "Turning it on."

    reply = chunks[0]
    for chunk in chunks[1:]:
        reply += chunk
    assert [(c["name"], c["args"]) for c in reply.tool_calls] == [
        ("HassTurnOn", {"name": "Kitchen light"})
    ]
    assert all(
        isinstance(block, str) or block.get("type") == "text"
        for block in (reply.content if isinstance(reply.content, list) else [])
    )


def _text(content: Any) -> str:
    """Mirror conversation._normalize_ai_content (text and text_delta blocks)."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") in ("text", "text_delta")
    )


@pytest.mark.parametrize(
    ("name", "reasoning", "expected"),
    [
        ("gpt-5", "low", {"reasoning_effort": "low"}),
        ("gpt-4.1", None, {"temperature": 0.7}),
    ],
)
async def test_earlier_models_stay_on_chat_completions(
    name: str, reasoning: Any, expected: dict[str, Any]
) -> None:
    requests: list[httpx.Request] = []
    model = _configured(_client(requests), name, reasoning)
    reply = await model.ainvoke([HumanMessage("hello")])
    assert reply.content == "Hi."
    (request,) = requests
    assert request.url.path.endswith("/chat/completions")
    body = json.loads(request.content)
    assert body["model"] == name
    assert body["store"] is False
    for key, value in expected.items():
        assert body[key] == value


def _gpt6_reply() -> AIMessage:
    """Return a completed Responses reply, as the production client builds it."""
    from openai.types.responses import Response  # noqa: PLC0415

    result = lc_openai._construct_lc_result_from_responses_api(
        Response.model_validate(_response([_REASONING, _MESSAGE, _CALL], "completed")),
        # The output format the production client is built with.
        output_version=cast(
            "Any",
            hga_component.build_openai_chat(
                "sk-test", http_client=None, http_async_client=None
            ),
        ).default.output_version,
    )
    message = result.generations[0].message
    assert isinstance(message, AIMessage)
    return message


def _history() -> list[Any]:
    return [
        HumanMessage("turn on the kitchen light"),
        _gpt6_reply(),
        ToolMessage("done", tool_call_id="call_1"),
        HumanMessage("thanks"),
    ]


def _shifted(event: dict[str, Any], by: int) -> dict[str, Any]:
    if "output_index" not in event:
        return event
    return {**event, "output_index": event["output_index"] + by}


_REASONING_2 = {**_REASONING, "id": "rs_2", "encrypted_content": "ENC2"}
# GPT-6 often returns several reasoning items in one reply (seen live): the v0
# format holds one, so streaming concatenates them ("ENCENC2"), and OpenAI
# rejects that on replay with 400 invalid_encrypted_content.
_STREAM_TWO_REASONING = [
    _STREAM[0],
    *_STREAM[1:3],
    {"type": "response.output_item.added", "output_index": 1, "item": _REASONING_2},
    {"type": "response.output_item.done", "output_index": 1, "item": _REASONING_2},
    *(_shifted(event, 1) for event in _STREAM[3:-1]),
    {
        "type": "response.completed",
        "response": _response([_REASONING, _REASONING_2, _MESSAGE, _CALL], "completed"),
    },
]


async def test_the_next_openai_turn_sends_no_reasoning() -> None:
    """A tool turn's follow-up sends the call and its result, never the reasoning."""
    requests: list[httpx.Request] = []
    model = _configured(
        _client(requests, _STREAM_TWO_REASONING), "gpt-6-luna"
    ).bind_tools([HassTurnOn])
    first = [HumanMessage("turn on the kitchen")]
    chunks = [chunk async for chunk in model.astream(first)]
    reply = chunks[0]
    for chunk in chunks[1:]:
        reply += chunk
    # Merged, the v0 reply would hold one reasoning item with both items'
    # encrypted contents run together; it is not kept at all.
    assert "reasoning" not in reply.additional_kwargs
    assert [c["name"] for c in reply.tool_calls] == ["HassTurnOn"]

    history = [*first, reply, ToolMessage("done", tool_call_id="call_1")]
    _ = [chunk async for chunk in model.astream(history)]

    body = json.loads(requests[-1].content)
    types = [item.get("type") for item in body["input"]]
    assert "reasoning" not in types
    assert "function_call" in types
    assert "function_call_output" in types
    assert "ENC" not in requests[-1].content.decode()


async def test_a_replayed_history_with_reasoning_still_sends_none() -> None:
    """A checkpointed reply that kept its reasoning item does not send it."""
    requests: list[httpx.Request] = []
    model = _configured(_client(requests), "gpt-6-luna").bind_tools([HassTurnOn])
    history = _history()
    assert history[1].additional_kwargs["reasoning"]["encrypted_content"] == "ENC"
    _ = [chunk async for chunk in model.astream(history)]
    body = json.loads(requests[-1].content)
    assert "reasoning" not in [item.get("type") for item in body["input"]]


async def test_a_non_streamed_reply_keeps_no_reasoning() -> None:
    requests: list[httpx.Request] = []
    model = _configured(_client(requests), "gpt-6-luna")
    reply = await model.ainvoke([HumanMessage("hi")], stream=False)
    assert json.loads(requests[-1].content).get("stream") is not True
    assert "reasoning" not in reply.additional_kwargs
    assert reply.tool_calls


def test_drop_reasoning_items_leaves_other_payloads_alone() -> None:
    chat = {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}
    assert drop_reasoning_items(chat) is chat
    text = {"model": "gpt-6-luna", "input": "hi"}
    assert drop_reasoning_items(text) is text
    responses = {
        "model": "gpt-6-luna",
        "input": [{"type": "reasoning", "id": "rs_1"}, {"role": "user"}],
    }
    assert drop_reasoning_items(responses)["input"] == [{"role": "user"}]
    assert len(responses["input"]) == 2  # the caller's payload is not mutated


def test_a_gpt6_history_converts_for_every_fallback_provider() -> None:
    """The reply holds only text blocks, so another provider can take over."""
    from langchain_anthropic.chat_models import _format_messages  # noqa: PLC0415
    from langchain_google_genai.chat_models import (  # noqa: PLC0415
        _parse_chat_history,
    )
    from langchain_ollama import ChatOllama  # noqa: PLC0415

    history = _history()
    assert all(
        block.get("type") == "text"
        for block in history[1].content
        if isinstance(block, dict)
    )
    ollama = ChatOllama(model="qwen3:8b")._convert_messages_to_ollama_messages(history)
    assert ollama[1]["content"].strip() == "Turning it on."
    assert ollama[1]["tool_calls"][0]["function"]["name"] == "HassTurnOn"
    _, anthropic = _format_messages(history, model="claude-opus-4-7")
    assert [block["type"] for block in anthropic[1]["content"]] == [
        "text",
        "tool_use",
    ]
    _parse_chat_history(history)


def test_a_gpt6_history_replays_on_chat_completions() -> None:
    """Switching the model back to gpt-5 must not send Responses-only items."""
    replay = [lc_openai._convert_message_to_dict(message) for message in _history()]
    assistant = replay[1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["function"]["name"] == "HassTurnOn"
    assert "reasoning" not in assistant
    assert all(
        block.get("type") == "text"
        for block in assistant.get("content") or []
        if isinstance(block, dict)
    )


def test_a_chat_completions_history_replays_on_the_responses_api() -> None:
    """A thread started on gpt-5 continues on GPT-6."""
    history = [
        HumanMessage("turn on the kitchen light"),
        AIMessage(
            "",
            tool_calls=[
                {"name": "HassTurnOn", "args": {"name": "Kitchen"}, "id": "call_1"}
            ],
        ),
        ToolMessage("done", tool_call_id="call_1"),
        AIMessage("Done."),
        HumanMessage("thanks"),
    ]
    replay = lc_openai._construct_responses_api_input(history, store=False)
    kinds = [item.get("type", item.get("role")) for item in replay]
    assert "function_call" in kinds
    assert "function_call_output" in kinds
    assert not any(item.get("id") for item in replay if item.get("type") == "message")
