# ruff: noqa: S101
"""Regression tests for previously fixed issues."""

from __future__ import annotations

import asyncio
import logging
import ssl
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import psycopg
import pytest
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.dispatcher import async_dispatcher_send
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import ConfigurableField

import custom_components.home_generative_agent as hga_component
import custom_components.home_generative_agent.agent.graph as agent_graph
from custom_components.home_generative_agent.agent import tools as agent_tools
from custom_components.home_generative_agent.agent.graph import (
    _MAX_ACTION_ROUNDS,
    State,
    _search_memories,
    _should_continue,
    _tool_loop_guard,
)
from custom_components.home_generative_agent.const import SIGNAL_HGA_RECOGNIZED
from custom_components.home_generative_agent.core.image_entity import LastEventImage
from custom_components.home_generative_agent.core.person_gallery import PersonGalleryDAO
from custom_components.home_generative_agent.core.recognized_sensor import (
    RecognizedPeopleSensor,
)
from custom_components.home_generative_agent.core.video_analyzer import VideoAnalyzer

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


HistoryTool = Any
history_tool = cast(
    "HistoryTool", cast("Any", agent_tools.get_entity_history).coroutine
)


# ---------------------------------------------------------------------------
# _get_existing_entity_id — issue #414: ambiguous friendly name disambiguation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_existing_entity_id_prefers_canonical_over_numbered_duplicate(
    hass: HomeAssistant,
) -> None:
    """_get_existing_entity_id picks the un-suffixed entity when a _N duplicate exists (issue #414)."""
    hass.states.async_set("binary_sensor.haustur", "off", {"friendly_name": "Haustür"})
    hass.states.async_set(
        "binary_sensor.haustur_2", "off", {"friendly_name": "Haustür"}
    )

    result = await agent_tools._get_existing_entity_id("Haustür", hass, "binary_sensor")
    assert result == "binary_sensor.haustur"


@pytest.mark.asyncio
async def test_get_entity_history_returns_error_dict_on_genuinely_ambiguous_entity(
    hass: HomeAssistant,
) -> None:
    """get_entity_history returns {"error": ...} when the entity name is genuinely ambiguous (issue #414)."""
    hass.states.async_set(
        "binary_sensor.front_door_a", "off", {"friendly_name": "Front Door"}
    )
    hass.states.async_set(
        "binary_sensor.front_door_b", "off", {"friendly_name": "Front Door"}
    )

    config = {"configurable": {"hass": hass}}
    result = await history_tool(
        ["Front Door"],
        ["binary_sensor"],
        "2025-01-01T00:00:00+0000",
        "2025-01-02T00:00:00+0000",
        config=config,
    )

    assert "error" in result
    assert "Front Door" in result["error"]


@pytest.mark.asyncio
async def test_get_entity_history_pairs_zip_warns(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mismatched name/domain lengths should be paired best-effort."""
    calls: list[tuple[str, str]] = []

    async def _fake_get_existing_entity_id(
        name: str, hass_arg: object, domain: str
    ) -> str:
        _ = hass_arg
        calls.append((name, domain))
        return f"{domain}.{name.lower().replace(' ', '_')}"

    async def _fake_fetch(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {}

    monkeypatch.setattr(
        agent_tools, "_get_existing_entity_id", _fake_get_existing_entity_id
    )
    monkeypatch.setattr(agent_tools, "_fetch_data_from_history", _fake_fetch)
    monkeypatch.setattr(agent_tools, "_fetch_data_from_long_term_stats", _fake_fetch)

    config = {"configurable": {"hass": hass}}
    with caplog.at_level(logging.WARNING):
        result = await history_tool(
            ["Front Door", "Living Room Light"],
            ["binary_sensor"],
            "2025-01-01T00:00:00+0000",
            "2025-01-02T00:00:00+0000",
            config=config,
        )

    assert calls == [("Front Door", "binary_sensor")]
    assert any("pairing best-effort" in rec.message for rec in caplog.records)
    assert result == {}


def test_filter_data_total_increasing_empty(hass: HomeAssistant) -> None:
    """Return zero value when total_increasing data is non-numeric."""
    hass.states.async_set(
        "sensor.energy",
        "unknown",
        {
            "state_class": "total_increasing",
            "unit_of_measurement": "kWh",
        },
    )

    result = agent_tools._filter_data("sensor.energy", [{"state": "unknown"}], hass)
    assert result["value"] == 0.0
    assert result["units"] == "kWh"


@pytest.mark.asyncio
async def test_recognized_sensor_attributes_update(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """extra_state_attributes should reflect current people list."""
    sensor = RecognizedPeopleSensor(hass, "camera.test")
    monkeypatch.setattr(sensor, "async_write_ha_state", lambda: None)
    await sensor.async_added_to_hass()

    attrs_initial = dict(sensor.extra_state_attributes or {})
    assert attrs_initial is not None
    async_dispatcher_send(
        hass, SIGNAL_HGA_RECOGNIZED, "camera.test", ["Alice"], None, None, None
    )
    attrs_updated = sensor.extra_state_attributes
    assert attrs_updated is not None

    assert attrs_initial["count"] == 0
    assert attrs_updated["count"] == 1


def test_last_event_image_recognized_mapping(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SIGNAL_HGA_RECOGNIZED args should map to the right fields."""
    entity = LastEventImage(hass, "camera.test")
    monkeypatch.setattr(entity, "async_write_ha_state", lambda: None)

    latest_path = tmp_path / "latest.jpg"
    entity._on_recognized(
        "camera.test",
        ["Alice"],
        "Porch activity",
        "2025-01-01T00:00:00+0000",
        str(latest_path),
    )

    attrs = entity._attrs
    assert attrs["recognized_people"] == ["Alice"]
    assert attrs["summary"] == "Porch activity"
    assert attrs["last_event"] == "2025-01-01T00:00:00+0000"
    assert attrs["latest_path"] == str(latest_path)


@pytest.mark.asyncio
async def test_person_gallery_invalid_embedding(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid embeddings should not trigger DB inserts."""

    class _FakeResp:
        def __init__(self) -> None:
            self._payload: dict[str, object] = {"faces": [{"embedding": [0.1, 0.2]}]}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    class _FakeClient:
        async def post(self, *_args: object, **_kwargs: object) -> _FakeResp:
            return _FakeResp()

    dao = PersonGalleryDAO(cast("Any", object()), hass)
    dao._client = _FakeClient()  # type: ignore[assignment]

    async def _fail_add_person(*_args: object, **_kwargs: object) -> None:
        msg = "add_person should not be called"
        raise AssertionError(msg)

    monkeypatch.setattr(dao, "add_person", _fail_add_person)

    result = await dao.enroll_from_image("http://face-api", "Alice", b"img")
    assert result is False


@pytest.mark.asyncio
async def test_video_analyzer_recognize_faces_without_gallery(
    hass: HomeAssistant,
) -> None:
    """Face recognition should not crash when person_gallery is unavailable."""

    class _FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"faces": [{"embedding": [0.1, 0.2, 0.3]}]}

    class _FakeClient:
        async def post(self, *_args: object, **_kwargs: object) -> _FakeResp:
            return _FakeResp()

    runtime_data = SimpleNamespace(
        face_recognition=True,
        face_api_url="http://face-api",
        person_gallery=None,
    )
    entry = SimpleNamespace(runtime_data=runtime_data)
    analyzer = VideoAnalyzer(hass, cast("Any", entry))
    analyzer._httpx_client = _FakeClient()  # type: ignore[assignment]

    recognized = await analyzer.recognize_faces(b"not-an-image", "camera.test")
    assert [hit.name for hit in recognized] == ["Indeterminate"]
    assert recognized[0].embedding is None


def test_agent_tools_uses_direct_tool_node_injected_store_import() -> None:
    """Avoid importing langgraph.prebuilt package during startup."""
    source = Path(agent_tools.__file__).read_text(encoding="utf-8")
    assert "from langgraph.prebuilt.tool_node import InjectedStore" in source
    assert "from langgraph.prebuilt import InjectedStore" not in source


@pytest.mark.asyncio
async def test_invoke_model_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    _invoke_model raises HomeAssistantError when the LLM hangs under load.

    Regression: without asyncio.wait_for the model.ainvoke() call blocked
    indefinitely when the Ollama GPU was saturated by background VLM work,
    stalling astream_events and showing 'no response' in the chat UI.
    """
    monkeypatch.setattr(agent_graph, "_LLM_INVOKE_TIMEOUT_S", 0.05)

    async def _slow_ainvoke(*_args: object, **_kwargs: object) -> None:
        await asyncio.sleep(10)

    mock_model = MagicMock()
    mock_model.ainvoke = _slow_ainvoke

    with pytest.raises(HomeAssistantError, match="timed out"):
        await agent_graph._invoke_model(mock_model, [], {})


@pytest.mark.asyncio
async def test_invoke_model_returns_result_within_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_invoke_model returns the model result when the LLM responds in time."""
    monkeypatch.setattr(agent_graph, "_LLM_INVOKE_TIMEOUT_S", 5.0)

    expected = MagicMock()

    async def _fast_ainvoke(*_args: object, **_kwargs: object) -> MagicMock:
        return expected

    mock_model = MagicMock()
    mock_model.ainvoke = _fast_ainvoke

    result = await agent_graph._invoke_model(mock_model, [], {})
    assert result is expected


@pytest.mark.asyncio
async def test_invoke_model_drops_unsupported_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    _invoke_model retries once without temperature on an OpenAI 400 (issue #502).

    Regression: reasoning-style OpenAI models reject non-default temperature
    with unsupported_value; the chat graph must drop the param and retry on
    the same model instead of failing the turn.
    """
    monkeypatch.setattr(agent_graph, "_LLM_INVOKE_TIMEOUT_S", 5.0)

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(400, request=request)
    err = openai.BadRequestError(
        "Error code: 400",
        response=response,
        body={
            "message": "Unsupported value: 'temperature' does not support 0.2.",
            "type": "invalid_request_error",
            "param": "temperature",
            "code": "unsupported_value",
        },
    )
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(side_effect=[err, AIMessage(content="ok")])

    result = await agent_graph._invoke_model(mock_model, [], {})

    assert result.content == "ok"
    assert mock_model.ainvoke.await_count == 2
    retry_config = mock_model.ainvoke.await_args_list[1].args[1]
    assert retry_config["configurable"]["temperature"] is None


# ---------------------------------------------------------------------------
# _make_transient_tool_error — non-retryable tool timeout message
# ---------------------------------------------------------------------------


def test_make_transient_tool_error_status_and_content() -> None:
    """_make_transient_tool_error must produce a ToolMessage with status='error'."""
    msg = agent_graph._make_transient_tool_error("boom", "my_tool", "call-123")

    assert isinstance(msg, ToolMessage)
    assert msg.name == "my_tool"
    assert msg.tool_call_id == "call-123"
    assert msg.status == "error"
    # The content must contain the error description.
    assert "boom" in msg.content


def test_make_transient_tool_error_content_instructs_no_retry() -> None:
    """Transient error message must tell the LLM not to retry the tool."""
    msg = agent_graph._make_transient_tool_error("timeout", "some_tool", "id-1")
    # The template instructs the model not to retry.
    content = str(msg.content)
    assert "Do not retry" in content or "transient" in content.lower()


def test_ollama_httpx_client_kwargs_use_prebuilt_ssl_context() -> None:
    """Ollama chat and embedding clients must not build SSL contexts on-loop."""
    kwargs = cast("Any", hga_component)._ollama_httpx_client_kwargs()

    assert isinstance(kwargs["verify"], ssl.SSLContext)


# ---------------------------------------------------------------------------
# _call_model — fallback "Done." when Qwen3 thinking strips all content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_model_injects_done_fallback_after_empty_tool_response(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    _call_model must emit 'Done.' when the model returns empty content after a tool.

    Regression: Qwen3 extended-thinking strips <think> tokens so content becomes
    '' (eval_count: 9).  Without the fallback the user receives no reply at all.
    """
    empty_ai = AIMessage(content="", tool_calls=[])

    async def _fake_invoke_model(*_args: object, **_kwargs: object) -> AIMessage:
        return empty_ai

    async def _fake_trim(
        messages: list[object], *_args: object, **_kwargs: object
    ) -> list[object]:
        return messages

    mock_store = MagicMock()
    mock_store.asearch = AsyncMock(return_value=[])

    monkeypatch.setattr(agent_graph, "_invoke_model", _fake_invoke_model)
    monkeypatch.setattr(agent_graph, "_trim_messages_for_model", _fake_trim)

    state: dict[str, object] = {
        "messages": [
            HumanMessage(content="turn it off"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "HassTurnOff",
                        "args": {},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="action_done", tool_call_id="call-1"),
        ],
        "selected_tools": [],
        "summary": "",
        "tool_routing_map": {},
        "messages_to_remove": [],
        "chat_model_usage_metadata": {},
    }

    config: dict[str, object] = {
        "configurable": {
            "chat_model": MagicMock(),
            "user_id": "user-test",
            "hass": hass,
            "options": {},
            "chat_model_options": {},
            "prompt": "You are a helpful assistant.",
            "langchain_tools": {},
            "ha_llm_api": None,
            "pending_actions": {},
        }
    }

    result = await agent_graph._call_model(state, config, store=mock_store)  # type: ignore[arg-type]

    ai_msg = result["messages"]
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.content == "Done.", f"expected 'Done.' but got {ai_msg.content!r}"
    assert not ai_msg.tool_calls


@pytest.mark.asyncio
async def test_call_model_preserves_newlines_in_reply(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    _call_model must round-trip a multi-line reply unchanged (issue #628).

    extract_final used to collapse every whitespace run to a single space, so
    the graph-state copy of a formatted reply differed from the streamed copy
    by whitespace alone.  conversation.py read that as a mid-stream model
    fallback and replaced the streamed markdown with the flattened text.
    Leaked <think> blocks must still be stripped.
    """
    formatted = (
        "Here's your security audit:\n\n"
        "**High severity:**\n\n"
        "1. **Add-ons** - text.\n"
        "2. **Locks** - text.\n"
    )
    raw_reply = formatted

    async def _fake_invoke_model(*_args: object, **_kwargs: object) -> AIMessage:
        return AIMessage(content=raw_reply)

    async def _fake_trim(
        messages: list[object], *_args: object, **_kwargs: object
    ) -> list[object]:
        return messages

    mock_store = MagicMock()
    mock_store.asearch = AsyncMock(return_value=[])

    monkeypatch.setattr(agent_graph, "_invoke_model", _fake_invoke_model)
    monkeypatch.setattr(agent_graph, "_trim_messages_for_model", _fake_trim)

    state: dict[str, object] = {
        "messages": [HumanMessage(content="audit my home")],
        "selected_tools": [],
        "summary": "",
        "tool_routing_map": {},
        "messages_to_remove": [],
        "chat_model_usage_metadata": {},
    }

    config: dict[str, object] = {
        "configurable": {
            "chat_model": MagicMock(),
            "user_id": "user-test",
            "hass": hass,
            "options": {},
            "chat_model_options": {},
            "prompt": "You are a helpful assistant.",
            "langchain_tools": {},
            "ha_llm_api": None,
            "pending_actions": {},
        }
    }

    # No <think> block: the reply must survive byte-for-byte, trailing newline
    # included, or it differs from the streamed copy and replace_partial fires.
    result = await agent_graph._call_model(state, config, store=mock_store)  # type: ignore[arg-type]
    ai_msg = result["messages"]
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.content == formatted

    # Leaked reasoning is still stripped, along with the whitespace it leaves.
    raw_reply = f"<think>reasoning</think>\n\n{formatted}"
    result = await agent_graph._call_model(state, config, store=mock_store)  # type: ignore[arg-type]
    ai_msg = result["messages"]
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.content == formatted.strip()
    assert "<think>" not in ai_msg.content


@pytest.mark.asyncio
async def test_call_model_binds_tools_in_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool binding can lazily import LangChain modules, so keep it off-loop."""

    async def _fake_invoke_model(*_args: object, **_kwargs: object) -> AIMessage:
        return AIMessage(content="ok")

    async def _fake_trim(
        messages: list[object], *_args: object, **_kwargs: object
    ) -> list[object]:
        return messages

    class FakeHass:
        executor_calls = 0

        async def async_add_executor_job(self, target: Any) -> Any:
            self.executor_calls += 1
            return target()

    class FakeModel:
        bound_tools: list[object] | None = None
        config: dict[str, object] | None = None

        def with_config(self, *, config: dict[str, object]) -> FakeModel:
            self.config = config
            return self

        def bind_tools(self, tools: list[object]) -> FakeModel:
            self.bound_tools = tools
            return self

    fake_hass = FakeHass()
    model = FakeModel()
    selected_tools: list[object] = [object()]
    mock_store = MagicMock()
    mock_store.asearch = AsyncMock(return_value=[])

    monkeypatch.setattr(agent_graph, "_invoke_model", _fake_invoke_model)
    monkeypatch.setattr(agent_graph, "_trim_messages_for_model", _fake_trim)

    state: dict[str, object] = {
        "messages": [HumanMessage(content="turn on the lamp")],
        "selected_tools": selected_tools,
        "summary": "",
        "tool_routing_map": {},
        "messages_to_remove": [],
        "chat_model_usage_metadata": {},
    }
    config: dict[str, object] = {
        "configurable": {
            "chat_model": model,
            "user_id": "user-test",
            "hass": fake_hass,
            "options": {},
            "chat_model_options": {"reasoning": True},
            "prompt": "You are a helpful assistant.",
            "langchain_tools": {},
            "ha_llm_api": None,
            "pending_actions": {},
        }
    }

    result = await agent_graph._call_model(state, config, store=mock_store)  # type: ignore[arg-type]

    assert fake_hass.executor_calls == 1
    assert model.config == {"configurable": {"reasoning": False}}
    assert model.bound_tools == selected_tools
    assert result["messages"].content == "ok"


# ---------------------------------------------------------------------------
# _search_memories — issue #394: llama-server embedding incompatibility
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_memories_returns_semantic_results() -> None:
    """Happy path: semantic search returns store results unchanged."""
    store = MagicMock()
    expected = [MagicMock()]
    store.asearch = AsyncMock(return_value=expected)

    result = await _search_memories(store, "user-1", "what is the temperature?")

    store.asearch.assert_awaited_once_with(
        ("user-1", "memories"), query="what is the temperature?", limit=10
    )
    assert result is expected


@pytest.mark.asyncio
async def test_search_memories_falls_back_to_recency_on_attribute_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    AttributeError from embedding endpoint falls back to recency search (issue #394).

    llama-server returns a bare JSON list instead of {"data": [...]}, which causes
    the OpenAI SDK parser to raise AttributeError inside store.asearch.
    """
    store = MagicMock()
    fallback = [MagicMock()]
    store.asearch = AsyncMock(
        side_effect=[
            AttributeError("'list' object has no attribute 'data'"),
            fallback,
        ]
    )

    with caplog.at_level(logging.WARNING):
        result = await _search_memories(store, "user-1", "some query")

    assert result is fallback
    assert "incompatible response" in caplog.text
    # Second call must omit query= (recency-only)
    _, recency_kwargs = store.asearch.call_args_list[1]
    assert "query" not in recency_kwargs


@pytest.mark.asyncio
async def test_search_memories_returns_empty_when_both_searches_fail() -> None:
    """Both semantic and recency search failures return [] without propagating."""
    store = MagicMock()
    store.asearch = AsyncMock(
        side_effect=[
            AttributeError("bad embedding format"),
            RuntimeError("DB gone"),
        ]
    )

    result = await _search_memories(store, "user-1", "query")

    assert result == []


@pytest.mark.asyncio
async def test_search_memories_falls_back_to_recency_on_vector_dimension_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Vector dimension mismatches fall back to recency without a traceback."""
    store = MagicMock()
    fallback = [MagicMock()]
    store.asearch = AsyncMock(
        side_effect=[
            psycopg.DataError("different vector dimensions 1024 and 3072"),
            fallback,
        ]
    )

    with caplog.at_level(logging.WARNING):
        result = await _search_memories(store, "user-1", "some query")

    assert result is fallback
    assert "vector store data mismatch" in caplog.text
    assert "Unexpected memory store search failure" not in caplog.text
    _, recency_kwargs = store.asearch.call_args_list[1]
    assert "query" not in recency_kwargs


@pytest.mark.asyncio
async def test_search_memories_returns_empty_on_unexpected_error() -> None:
    """Unexpected exceptions are swallowed and return [] so the agent can still respond."""
    store = MagicMock()
    store.asearch = AsyncMock(side_effect=RuntimeError("unexpected failure"))

    result = await _search_memories(store, "user-1", "query")

    assert result == []


@pytest.mark.asyncio
async def test_openai_compatible_repeated_tool_calls_hit_loop_guard() -> None:
    """Loop guard routes to tool_loop_guard and emits a friendly message (issue #394)."""
    state: State = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "SomeTool", "args": {}, "id": "call_1"}],
            )
        ],
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
        "action_rounds": _MAX_ACTION_ROUNDS,
    }

    assert _should_continue(state) == "tool_loop_guard"

    guard_result = await _tool_loop_guard(state)
    response_text = guard_result["messages"][0].content
    assert "wasn't able to complete" in response_text


@pytest.mark.asyncio
async def test_sampling_rebind_recovers_tool_bound_chat_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The chat path recovers from a temperature 400 even with tools bound.

    Regression for the issue #502 review finding: bind_tools() collapses a
    configurable-fields model into a concrete RunnableBinding with sampling
    params baked in, so the invoke-time config drop is a no-op there. The
    graph must rebuild from the pre-bind model with sampling params nulled
    and re-bind the tools. Uses a real configurable_fields -> with_config ->
    bind_tools chain, not mocks, to prove the production shape works.
    """
    monkeypatch.setattr(agent_graph, "_LLM_INVOKE_TIMEOUT_S", 5.0)

    seen_temperatures: list[float | None] = []

    def _temp_400() -> openai.BadRequestError:
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(400, request=request)
        return openai.BadRequestError(
            "Error code: 400",
            response=response,
            body={
                "message": "Unsupported value: 'temperature' does not support 0.2.",
                "type": "invalid_request_error",
                "param": "temperature",
                "code": "unsupported_value",
            },
        )

    class _PickyChatModel(BaseChatModel):
        """Rejects any non-default temperature like OpenAI reasoning models."""

        temperature: float | None = None
        top_p: float | None = None

        @property
        def _llm_type(self) -> str:
            return "picky"

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self.bind(tools=list(tools))

        def _generate(
            self,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            seen_temperatures.append(self.temperature)
            if self.temperature is not None:
                raise _temp_400()
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="ok"))]
            )

    base_model = (
        _PickyChatModel()
        .configurable_fields(
            temperature=ConfigurableField(id="temperature"),
            top_p=ConfigurableField(id="top_p"),
        )
        .with_config(config={"configurable": {"temperature": 0.2}})
    )

    selected_tools = [{"type": "function", "function": {"name": "t"}}]
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn: fn())

    bound = agent_graph._bind_model_tools(
        base_model, selected_tools, disable_reasoning=False
    )

    result = await agent_graph._invoke_chat_model_with_sampling_rebind(
        hass,
        base_model,
        bound,
        [HumanMessage(content="hi")],
        {},
        selected_tools,
        disable_reasoning=False,
    )

    assert result.content == "ok"
    # Exactly two API calls: baked 0.2 rejected once, rebind with None succeeds.
    assert seen_temperatures == [0.2, None]


@pytest.mark.asyncio
async def test_schema_recovery_drops_tool_rejected_by_provider(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    A tool the provider refuses to validate is dropped, not the whole turn.

    Regression for issue #585: Anthropic validates every tool schema before
    it reads the conversation, so one bad schema 400s the request and the
    user hears an error. The provider names the offender only by position
    ("tools.1.custom.input_schema: ..."), so the graph must map that back to
    the bound tool list, drop it and retry.
    """
    monkeypatch.setattr(agent_graph, "_LLM_INVOKE_TIMEOUT_S", 5.0)

    seen_tool_sets: list[list[str]] = []

    class _SchemaPickyChatModel(BaseChatModel):
        """400s while a tool named HassStartTimer is bound."""

        # pydantic model field: the default is copied per instance.
        bound_tools: list[Any] = []  # noqa: RUF012

        @property
        def _llm_type(self) -> str:
            return "schema_picky"

        def bind_tools(self, tools: Any, **_kwargs: Any) -> Any:
            return self.model_copy(update={"bound_tools": list(tools)})

        def _generate(
            self,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            names = [t["function"]["name"] for t in self.bound_tools]
            seen_tool_sets.append(names)
            if "HassStartTimer" in names:
                index = names.index("HassStartTimer")
                msg = (
                    f"tools.{index}.custom.input_schema: input_schema does not "
                    "support oneOf, allOf, or anyOf at the top level"
                )
                raise ValueError(msg)
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="ok"))]
            )

    selected_tools = [
        {"type": "function", "function": {"name": "GetLiveContext"}},
        {"type": "function", "function": {"name": "HassStartTimer"}},
    ]
    base_model = _SchemaPickyChatModel()
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn: fn())

    bound = agent_graph._bind_model_tools(
        base_model, selected_tools, disable_reasoning=False
    )

    with caplog.at_level(logging.WARNING):
        result = await agent_graph._invoke_chat_model_with_schema_recovery(
            hass,
            base_model,
            bound,
            [HumanMessage(content="hi")],
            {},
            selected_tools,
            disable_reasoning=False,
        )

    assert result.content == "ok"
    assert seen_tool_sets == [
        ["GetLiveContext", "HassStartTimer"],
        ["GetLiveContext"],
    ], "exactly one retry, with only the rejected tool removed"
    assert "HassStartTimer" in caplog.text, "the log must name the dropped tool"
    assert selected_tools[1]["function"]["name"] == "HassStartTimer", (
        "the caller's tool list must not be mutated"
    )


@pytest.mark.asyncio
async def test_schema_recovery_names_the_tool_when_retries_run_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An unrecoverable schema rejection still names the tool it blames.

    The reporter of issue #585 lost hours to a 400 that identified the tool
    only as `tools.3`; with a retrieval limit of five that index points at a
    different tool every turn. Whatever else happens, the error must name it.
    """
    monkeypatch.setattr(agent_graph, "_LLM_INVOKE_TIMEOUT_S", 5.0)

    class _AlwaysRejects(BaseChatModel):
        """Blames position 0 no matter which tools are bound."""

        # pydantic model field: the default is copied per instance.
        bound_tools: list[Any] = []  # noqa: RUF012

        @property
        def _llm_type(self) -> str:
            return "always_rejects"

        def bind_tools(self, tools: Any, **_kwargs: Any) -> Any:
            return self.model_copy(update={"bound_tools": list(tools)})

        def _generate(
            self,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = "tools.0.custom.input_schema: unsupported schema"
            raise ValueError(msg)

    selected_tools = [
        {"type": "function", "function": {"name": f"Tool{i}"}} for i in range(5)
    ]
    base_model = _AlwaysRejects()
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn: fn())
    bound = agent_graph._bind_model_tools(
        base_model, selected_tools, disable_reasoning=False
    )

    with pytest.raises(HomeAssistantError, match="Tool3"):
        await agent_graph._invoke_chat_model_with_schema_recovery(
            hass,
            base_model,
            bound,
            [HumanMessage(content="hi")],
            {},
            selected_tools,
            disable_reasoning=False,
        )


@pytest.mark.asyncio
async def test_schema_recovery_passes_through_unrelated_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error that is not a tool-schema rejection is raised untouched."""
    monkeypatch.setattr(agent_graph, "_LLM_INVOKE_TIMEOUT_S", 5.0)

    class _Broken(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "broken"

        def bind_tools(self, tools: Any, **_kwargs: Any) -> Any:
            return self

        def _generate(
            self,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = "upstream connect error"
            raise ValueError(msg)

    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn: fn())
    selected_tools = [{"type": "function", "function": {"name": "GetLiveContext"}}]

    with pytest.raises(HomeAssistantError, match="upstream connect error"):
        await agent_graph._invoke_chat_model_with_schema_recovery(
            hass,
            _Broken(),
            _Broken(),
            [HumanMessage(content="hi")],
            {},
            selected_tools,
            disable_reasoning=False,
        )
    assert hass.async_add_executor_job.await_count == 0, "no rebind on a non-schema 400"


# ---------------------------------------------------------------------------
# Issue #617 — the system prompt's stable prefix must not change per turn
# ---------------------------------------------------------------------------


class _Mem:
    def __init__(self, key: str, value: str) -> None:
        self.key = key
        self.value = value


def test_build_system_message_puts_volatile_context_after_stable_prefix() -> None:
    """Time, memories and summary all land in the second block, in that order."""
    msg = agent_graph._build_system_message(
        "STABLE",
        "Current time is 15:09:20. Today's date is 2026-09-09.",
        agent_graph._format_memories([_Mem("k", "v")]),
        "earlier we talked",
    )
    assert msg.content == [
        {"type": "text", "text": "STABLE"},
        {
            "type": "text",
            "text": (
                "Current time is 15:09:20. Today's date is 2026-09-09.\n"
                "<memories>\n[k]: v\n</memories>\n"
                "<past_conversation_summary>\nearlier we talked\n"
                "</past_conversation_summary>"
            ),
        },
    ]


def test_build_system_message_stable_prefix_identical_across_turns() -> None:
    """The regression itself: two turns a second apart share the first block."""
    turn1 = agent_graph._build_system_message(
        "STABLE", "Current time is 15:09:20.", "", ""
    )
    turn2 = agent_graph._build_system_message(
        "STABLE", "Current time is 15:09:34.", "", ""
    )
    assert turn1.content[0] == turn2.content[0]
    assert turn1.content[1] != turn2.content[1]


def test_build_system_message_plain_string_without_any_volatile_context() -> None:
    assert agent_graph._build_system_message("STABLE", "", "", "").content == "STABLE"
    assert agent_graph._format_memories([]) == ""


@pytest.mark.asyncio
async def test_call_model_tool_loop_keys_memories_on_last_user_message(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Mid tool loop the memory search must use the request, not recency.

    A recency listing differs from the semantic result the first call used,
    so the <memories> block — and with it the cached prefix — changed between
    the call that requested the tool and the call that answers.
    """
    captured: dict[str, object] = {}

    async def _fake_invoke_model(
        _model: object, messages: list[object], *_args: object, **_kwargs: object
    ) -> AIMessage:
        captured["messages"] = messages
        return AIMessage(content="Done.")

    async def _fake_trim(
        messages: list[object], *_args: object, **_kwargs: object
    ) -> list[object]:
        return messages

    mock_store = MagicMock()
    mock_store.asearch = AsyncMock(return_value=[_Mem("pref", "likes it warm")])
    monkeypatch.setattr(agent_graph, "_invoke_model", _fake_invoke_model)
    monkeypatch.setattr(agent_graph, "_trim_messages_for_model", _fake_trim)

    state: dict[str, object] = {
        "messages": [
            HumanMessage(content="turn it off"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "HassTurnOff", "args": {}, "id": "c1", "type": "tool_call"}
                ],
            ),
            ToolMessage(content="action_done", tool_call_id="c1"),
        ],
        "selected_tools": [],
        "summary": "",
        "tool_routing_map": {},
        "messages_to_remove": [],
        "chat_model_usage_metadata": {},
    }
    config: dict[str, object] = {
        "configurable": {
            "chat_model": MagicMock(),
            "user_id": "user-test",
            "hass": hass,
            "options": {},
            "chat_model_options": {},
            "prompt": "STABLE",
            "prompt_volatile": "Current time is 15:09:20.",
            "langchain_tools": {},
            "ha_llm_api": None,
            "pending_actions": {},
        }
    }

    result = await agent_graph._call_model(state, config, store=mock_store)  # type: ignore[arg-type]

    query = mock_store.asearch.await_args.kwargs["query"]
    assert query is not None
    assert "turn it off" in query
    system = cast("list[object]", captured["messages"])[0]
    assert isinstance(system, SystemMessage)
    assert system.content[0] == {"type": "text", "text": "STABLE"}
    tail = cast("dict[str, Any]", system.content[1])
    assert "Current time is 15:09:20." in tail["text"]
    assert "[pref]: likes it warm" in tail["text"]
    # The memo the next call of this turn will reuse.
    memo = result["turn_memories"]
    assert memo["key"] == "turn it off"
    assert "[pref]: likes it warm" in memo["text"]


def _call_model_harness(
    monkeypatch: pytest.MonkeyPatch,
    hass: HomeAssistant,
    messages: list[object],
    *,
    turn_memories: dict[str, str] | None = None,
    search_result: list[object] | None = None,
) -> tuple[dict[str, object], dict[str, object], MagicMock, dict[str, object]]:
    """Build state/config/store for a _call_model call and capture its input."""
    captured: dict[str, object] = {}

    async def _fake_invoke_model(
        _model: object, msgs: list[object], *_args: object, **_kwargs: object
    ) -> AIMessage:
        captured["messages"] = msgs
        return AIMessage(content="ok")

    async def _fake_trim(
        msgs: list[object], *_args: object, **_kwargs: object
    ) -> list[object]:
        return msgs

    monkeypatch.setattr(agent_graph, "_invoke_model", _fake_invoke_model)
    monkeypatch.setattr(agent_graph, "_trim_messages_for_model", _fake_trim)
    mock_store = MagicMock()
    mock_store.asearch = AsyncMock(return_value=search_result or [])
    state: dict[str, object] = {
        "messages": messages,
        "selected_tools": [],
        "summary": "",
        "tool_routing_map": {},
        "messages_to_remove": [],
        "chat_model_usage_metadata": {},
    }
    if turn_memories is not None:
        state["turn_memories"] = turn_memories
    config: dict[str, object] = {
        "configurable": {
            "chat_model": MagicMock(),
            "user_id": "user-test",
            "hass": hass,
            "options": {},
            "chat_model_options": {},
            "prompt": "STABLE",
            "prompt_volatile": "Current time is 15:09:20.",
            "langchain_tools": {},
            "ha_llm_api": None,
            "pending_actions": {},
        }
    }
    return state, config, mock_store, captured


def _system_tail(captured: dict[str, object]) -> str:
    system = cast("list[object]", captured["messages"])[0]
    assert isinstance(system, SystemMessage)
    return cast("dict[str, Any]", system.content[1])["text"]


@pytest.mark.asyncio
async def test_call_model_reuses_memoized_memories_within_a_turn(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later call of the same turn must not search again (no extra embedding)."""
    memo = {"key": "turn it off", "text": "<memories>\n[k]: v\n</memories>"}
    state, config, store, captured = _call_model_harness(
        monkeypatch,
        hass,
        [
            HumanMessage(content="turn it off"),
            AIMessage(content="", tool_calls=[{"name": "T", "args": {}, "id": "c1"}]),
            ToolMessage(content="done", tool_call_id="c1"),
        ],
        turn_memories=memo,
    )
    result = await agent_graph._call_model(state, config, store=store)  # type: ignore[arg-type]
    store.asearch.assert_not_awaited()
    assert "[k]: v" in _system_tail(captured)
    assert result["turn_memories"] == memo


@pytest.mark.asyncio
async def test_call_model_ignores_memo_from_another_user_message(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A memo keyed on a different message is stale and must be recomputed."""
    state, config, store, _captured = _call_model_harness(
        monkeypatch,
        hass,
        [HumanMessage(content="what time is it")],
        turn_memories={"key": "turn it off", "text": "<memories>\n[k]: v\n</memories>"},
        search_result=[_Mem("new", "fresh")],
    )
    result = await agent_graph._call_model(state, config, store=store)  # type: ignore[arg-type]
    store.asearch.assert_awaited_once()
    assert "what time is it" in store.asearch.await_args.kwargs["query"]
    assert result["turn_memories"]["key"] == "what time is it"
    assert "[new]: fresh" in result["turn_memories"]["text"]


@pytest.mark.asyncio
async def test_call_model_memo_key_prefers_message_id(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With an id assigned by the reducer, the memo is keyed on it, not the text."""
    state, config, store, _captured = _call_model_harness(
        monkeypatch, hass, [HumanMessage(content="turn it off", id="msg-1")]
    )
    result = await agent_graph._call_model(state, config, store=store)  # type: ignore[arg-type]
    assert result["turn_memories"]["key"] == "msg-1"


@pytest.mark.asyncio
async def test_call_model_multimodal_user_message_embeds_text_only(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, config, store, _captured = _call_model_harness(
        monkeypatch,
        hass,
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "what is in this picture"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ]
            )
        ],
    )
    await agent_graph._call_model(state, config, store=store)  # type: ignore[arg-type]
    query = store.asearch.await_args.kwargs["query"]
    assert "what is in this picture" in query
    assert "AAAA" not in query


@pytest.mark.asyncio
async def test_call_model_attachment_only_user_message_lists_by_recency(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No text to embed → recency listing, never the bare template."""
    state, config, store, _captured = _call_model_harness(
        monkeypatch,
        hass,
        [
            HumanMessage(
                content=[{"type": "image_url", "image_url": {"url": "data:,x"}}]
            )
        ],
    )
    await agent_graph._call_model(state, config, store=store)  # type: ignore[arg-type]
    assert store.asearch.await_args.kwargs["query"] is None


@pytest.mark.asyncio
async def test_call_model_without_user_message_lists_by_recency(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, config, store, _captured = _call_model_harness(
        monkeypatch, hass, [AIMessage(content="hello")]
    )
    result = await agent_graph._call_model(state, config, store=store)  # type: ignore[arg-type]
    assert store.asearch.await_args.kwargs["query"] is None
    assert result["turn_memories"]["key"] == ""


def test_format_and_dedupe_tools_sorts_non_string_names_without_raising() -> None:
    """A corrupt index row with a non-str name must not abort the turn."""
    raw: list = [
        {
            "name": 7,
            "api_id": "a",
            "description": "seven",
            "parameters": "{}",
            "is_actuation": False,
        },
        {
            "name": "b",
            "api_id": "a",
            "description": "bee",
            "parameters": "{}",
            "is_actuation": False,
        },
    ]
    selected, _routing = agent_graph._format_and_dedupe_tools(raw)
    assert [t["function"]["name"] for t in selected] == [7, "b"]


def test_format_and_dedupe_tools_binds_in_name_order() -> None:
    """Same tool set, different retrieval order → identical bound array."""

    def _raw(*names: str) -> list:
        return [
            {
                "name": n,
                "api_id": "assist",
                "description": n,
                "parameters": "{}",
                "is_actuation": False,
            }
            for n in names
        ]

    first, _ = agent_graph._format_and_dedupe_tools(
        _raw("HassTurnOn", "GetLiveContext")
    )
    second, _ = agent_graph._format_and_dedupe_tools(
        _raw("GetLiveContext", "HassTurnOn")
    )
    assert first == second
    assert [t["function"]["name"] for t in first] == ["GetLiveContext", "HassTurnOn"]


def test_format_and_dedupe_tools_first_seen_still_owns_the_route() -> None:
    """Sorting happens after dedupe: the earlier api_id keeps the name."""
    raw: list = [
        {
            "name": "search",
            "api_id": "mcp-a",
            "description": "a",
            "parameters": "{}",
            "is_actuation": False,
        },
        {
            "name": "search",
            "api_id": "mcp-b",
            "description": "b",
            "parameters": "{}",
            "is_actuation": False,
        },
    ]
    selected, routing = agent_graph._format_and_dedupe_tools(raw)
    assert routing == {"search": "mcp-a"}
    assert [t["function"]["description"] for t in selected] == ["a"]
