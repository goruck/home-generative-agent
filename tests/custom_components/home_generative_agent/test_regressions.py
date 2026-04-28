# ruff: noqa: S101
"""Regression tests for previously fixed issues."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.dispatcher import async_dispatcher_send
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import custom_components.home_generative_agent.agent.graph as agent_graph
from custom_components.home_generative_agent.agent import tools as agent_tools
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
    assert recognized == ["Indeterminate"]


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

    async def _fake_camera(*_args: object, **_kwargs: object) -> list[object]:
        return []

    async def _fake_trim(
        messages: list[object], *_args: object, **_kwargs: object
    ) -> list[object]:
        return messages

    mock_store = MagicMock()
    mock_store.asearch = AsyncMock(return_value=[])

    monkeypatch.setattr(agent_graph, "_invoke_model", _fake_invoke_model)
    monkeypatch.setattr(agent_graph, "get_recent_camera_activity", _fake_camera)
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
