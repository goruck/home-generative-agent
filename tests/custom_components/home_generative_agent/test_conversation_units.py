# ruff: noqa: S101
"""
Unit tests for conversation.py helpers (MultiLLMAPI, _run_tool_index_background).

hassil is not installed in the test venv, so this module stubs the entire
homeassistant.components.conversation import chain before importing conversation.py.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.exceptions import HomeAssistantError

from custom_components.home_generative_agent.core.utils import (
    gather_store_puts_in_chunks,
)


def _stub_ha_conversation() -> None:
    """
    Stub homeassistant.components.conversation before conversation.py loads it.

    hassil and home_assistant_intents are not installed in the test venv.
    We mock the HA conversation module with just enough surface area for
    conversation.py to import cleanly and for MultiLLMAPI / _run_tool_index_background
    to be accessible.
    """
    if "homeassistant.components.conversation" in sys.modules:
        return

    # Build real (empty) base classes so class inheritance works.
    class _ConversationEntity:
        pass

    class _AbstractConversationAgent:
        pass

    class _ConversationResult:
        pass

    class _UserContent:
        pass

    class _AssistantContent:
        pass

    # conversation module
    conv_mod: Any = types.ModuleType("homeassistant.components.conversation")
    conv_mod.ConversationEntity = _ConversationEntity
    conv_mod.AbstractConversationAgent = _AbstractConversationAgent
    conv_mod.ConversationResult = _ConversationResult
    conv_mod.UserContent = _UserContent
    conv_mod.AssistantContent = _AssistantContent
    conv_mod.AssistantContentDeltaDict = dict
    conv_mod.ToolResultContentDeltaDict = dict
    conv_mod.DOMAIN = "conversation"
    conv_mod.async_set_agent = MagicMock()
    conv_mod.trace = MagicMock()

    # conversation.models submodule
    models_mod: Any = types.ModuleType("homeassistant.components.conversation.models")
    models_mod.AbstractConversationAgent = _AbstractConversationAgent
    conv_mod.models = models_mod

    sys.modules["homeassistant.components.conversation"] = conv_mod
    sys.modules["homeassistant.components.conversation.models"] = models_mod


def _ensure_content_classes() -> None:
    """
    Guarantee AssistantContent/UserContent exist on the loaded module.

    Suite ordering decides whether the real HA conversation module or another
    test file's import stub is in sys.modules; leaner stubs (e.g. the one in
    test_conversation_stream.py) omit the content classes. The integration
    resolves them at runtime through the module object, so adding them here
    keeps isinstance checks and test construction consistent.
    """
    conv: Any = sys.modules["homeassistant.components.conversation"]
    if not hasattr(conv, "AssistantContent"):

        class _StubAssistantContent:
            pass

        conv.AssistantContent = _StubAssistantContent
    if not hasattr(conv, "UserContent"):

        class _StubUserContent:
            pass

        conv.UserContent = _StubUserContent


_stub_ha_conversation()
_ensure_content_classes()

# These imports must come AFTER the stub so conversation.py loads cleanly.
from homeassistant.components import conversation as ha_conversation  # noqa: E402

from custom_components.home_generative_agent.conversation import (  # noqa: E402
    _STREAM_ERROR_REASON_MAX_CHARS,
    HGAConversationEntity,
    MultiLLMAPI,
    _get_stt_hallucination_exact_patterns,
    _get_stt_hallucination_patterns,
    _is_stt_hallucination,
    _recommit_final_assistant_content,
    _run_tool_index_background,
    _streaming_failure_content,
)

# ---------------------------------------------------------------------------
# MultiLLMAPI: empty routing_map fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_llm_api_empty_routing_map_iterates_apis() -> None:
    """With no routing entry, async_call_tool falls back to iterating all APIs."""
    api1 = MagicMock()
    api1.async_call_tool = AsyncMock(side_effect=HomeAssistantError("not mine"))
    api2 = MagicMock()
    api2.async_call_tool = AsyncMock(return_value={"result": "ok"})

    multi = MultiLLMAPI({"api1": api1, "api2": api2}, routing_map={})

    tool_input = MagicMock()
    tool_input.tool_name = "mystery_tool"

    result = await multi.async_call_tool(tool_input)

    assert result == {"result": "ok"}
    api1.async_call_tool.assert_called_once_with(tool_input)
    api2.async_call_tool.assert_called_once_with(tool_input)


@pytest.mark.asyncio
async def test_multi_llm_api_empty_routing_map_all_fail_raises() -> None:
    """With no routing entry and all APIs failing, HomeAssistantError is raised."""
    api1 = MagicMock()
    api1.async_call_tool = AsyncMock(side_effect=HomeAssistantError("nope"))

    multi = MultiLLMAPI({"api1": api1}, routing_map={})

    tool_input = MagicMock()
    tool_input.tool_name = "mystery_tool"

    with pytest.raises(HomeAssistantError, match="No routing target"):
        await multi.async_call_tool(tool_input)


@pytest.mark.asyncio
async def test_multi_llm_api_routes_to_correct_api() -> None:
    """With a populated routing_map, calls go directly to the mapped API."""
    api1 = MagicMock()
    api1.async_call_tool = AsyncMock(return_value="from_api1")
    api2 = MagicMock()
    api2.async_call_tool = AsyncMock(return_value="from_api2")

    multi = MultiLLMAPI(
        {"api1": api1, "api2": api2},
        routing_map={"tool_a": "api2"},
    )

    tool_input = MagicMock()
    tool_input.tool_name = "tool_a"

    result = await multi.async_call_tool(tool_input)

    assert result == "from_api2"
    api1.async_call_tool.assert_not_called()
    api2.async_call_tool.assert_called_once_with(tool_input)


# ---------------------------------------------------------------------------
# _run_tool_index_background failure path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_tool_index_background_failure_sets_flag() -> None:
    """Indexing failure sets tool_index_failed=True and resets tool_indexing_in_progress."""
    rd = MagicMock()
    rd.tool_index_ready = False
    rd.tool_indexing_in_progress = True
    rd.tool_index_failed = False
    rd.tool_content_hashes = {}

    hass = MagicMock()

    with patch(
        "custom_components.home_generative_agent.conversation.gather_store_puts_in_chunks",
        new=AsyncMock(side_effect=RuntimeError("embedding provider down")),
    ):
        await _run_tool_index_background(
            index_tasks=[AsyncMock()],
            tool_hashes={"key": "hash"},
            rd=rd,
            hass=hass,
        )

    assert rd.tool_index_failed is True
    assert rd.tool_index_ready is False
    assert rd.tool_indexing_in_progress is False


@pytest.mark.asyncio
async def test_run_tool_index_background_success_clears_flags() -> None:
    """Successful indexing sets tool_index_ready=True and resets in_progress."""
    rd = MagicMock()
    rd.tool_index_ready = False
    rd.tool_indexing_in_progress = True
    rd.tool_index_failed = False
    rd.tool_content_hashes = {}

    hass = MagicMock()

    with patch(
        "custom_components.home_generative_agent.conversation.gather_store_puts_in_chunks",
        new=AsyncMock(return_value=None),
    ):
        await _run_tool_index_background(
            index_tasks=[AsyncMock()],
            tool_hashes={"key": "hash"},
            rd=rd,
            hass=hass,
        )

    assert rd.tool_index_ready is True
    assert rd.tool_index_failed is False
    assert rd.tool_indexing_in_progress is False
    assert rd.tool_content_hashes == {"key": "hash"}


# ---------------------------------------------------------------------------
# STT hallucination filter helpers
# ---------------------------------------------------------------------------


def test_get_stt_hallucination_patterns_empty() -> None:
    """Empty option returns an empty tuple."""
    assert _get_stt_hallucination_patterns({}) == ()
    assert _get_stt_hallucination_patterns({"stt_hallucination_patterns": []}) == ()
    assert _get_stt_hallucination_patterns({"stt_hallucination_patterns": ""}) == ()


def test_get_stt_hallucination_patterns_list() -> None:
    """List input is normalised to lower-case tuple."""
    patterns = _get_stt_hallucination_patterns(
        {"stt_hallucination_patterns": ["Foo", " BAR ", "baz"]}
    )
    assert patterns == ("foo", "bar", "baz")


def test_get_stt_hallucination_patterns_legacy_string() -> None:
    """Legacy comma-separated string still works."""
    patterns = _get_stt_hallucination_patterns(
        {"stt_hallucination_patterns": "Foo, BAR,  baz "}
    )
    assert patterns == ("foo", "bar", "baz")


def test_get_stt_hallucination_patterns_multiline_string() -> None:
    """Legacy newline-separated string also works."""
    patterns = _get_stt_hallucination_patterns(
        {"stt_hallucination_patterns": "Foo\nBAR\nbaz"}
    )
    assert patterns == ("foo", "bar", "baz")


def test_get_stt_hallucination_patterns_extra_whitespace() -> None:
    """Extra whitespace around commas and empty segments are ignored."""
    patterns = _get_stt_hallucination_patterns(
        {"stt_hallucination_patterns": " a , , b ,c"}
    )
    assert patterns == ("a", "b", "c")


def test_get_stt_hallucination_exact_patterns_empty() -> None:
    """Empty exact option returns an empty tuple."""
    assert _get_stt_hallucination_exact_patterns({}) == ()
    assert (
        _get_stt_hallucination_exact_patterns({"stt_hallucination_exact_patterns": []})
        == ()
    )


def test_get_stt_hallucination_exact_patterns_list() -> None:
    """List input is normalised to lower-case tuple."""
    patterns = _get_stt_hallucination_exact_patterns(
        {"stt_hallucination_exact_patterns": ["Foo", " BAR ", "baz"]}
    )
    assert patterns == ("foo", "bar", "baz")


def test_is_stt_hallucination_empty() -> None:
    """None/empty text never matches, even with non-empty patterns."""
    assert _is_stt_hallucination(None, ("foo",), ()) is False
    assert _is_stt_hallucination("", ("foo",), ()) is False


def test_is_stt_hallucination_no_patterns() -> None:
    """With empty patterns nothing ever matches."""
    assert _is_stt_hallucination("foo", (), ()) is False
    assert _is_stt_hallucination("subtitles", (), ()) is False


def test_is_stt_hallucination_substring_match() -> None:
    """Matching substring returns True (case-insensitive)."""
    sub_patterns = ("subtitles", "dimatorzok")
    assert _is_stt_hallucination("Subtitles by", sub_patterns, ()) is True
    assert _is_stt_hallucination("some dimatorzok noise", sub_patterns, ()) is True


def test_is_stt_hallucination_exact_match() -> None:
    """Exact match returns True only for full text equality (case-insensitive)."""
    exact_patterns = ("to be continued", "the end")
    assert _is_stt_hallucination("To Be Continued", (), exact_patterns) is True
    assert _is_stt_hallucination("The End", (), exact_patterns) is True
    assert _is_stt_hallucination("the end.", (), exact_patterns) is False  # not exact
    assert (
        _is_stt_hallucination("To be continued now", (), exact_patterns) is False
    )  # not exact


def test_is_stt_hallucination_combined() -> None:
    """Both substring and exact patterns work together."""
    sub_patterns = ("sub",)
    exact_patterns = ("the end",)
    assert _is_stt_hallucination("subtitles", sub_patterns, exact_patterns) is True
    assert _is_stt_hallucination("The End", sub_patterns, exact_patterns) is True
    assert _is_stt_hallucination("nothing", sub_patterns, exact_patterns) is False


def test_is_stt_hallucination_no_match() -> None:
    """Non-matching text returns False."""
    assert _is_stt_hallucination("turn on the light", ("subtitles",), ()) is False
    assert (
        _is_stt_hallucination("subtitle", ("subtitles",), ()) is False
    )  # partial, not full


# ---------------------------------------------------------------------------
# _streaming_failure_content: user-visible failure message (issue #502)
# ---------------------------------------------------------------------------


def test_streaming_failure_content_without_error() -> None:
    """No captured stream error yields the generic try-again message."""
    assert (
        _streaming_failure_content(None)
        == "I'm sorry, I was unable to respond in time. Please try again."
    )


def test_streaming_failure_content_includes_error_reason() -> None:
    """The captured stream error's message is surfaced in the chat reply."""
    err = HomeAssistantError("temperature does not support 0.2 with this model")
    result = _streaming_failure_content(err)
    assert result == (
        "I'm sorry, I was unable to respond: "
        "temperature does not support 0.2 with this model"
    )


def test_streaming_failure_content_uses_type_name_for_empty_message() -> None:
    """A HomeAssistantError with an empty str() falls back to the class name."""
    result = _streaming_failure_content(HomeAssistantError())
    assert result == "I'm sorry, I was unable to respond: HomeAssistantError"


def test_streaming_failure_content_hides_non_ha_error_details() -> None:
    """
    Arbitrary exceptions surface only their class name, never their message.

    Non-HomeAssistantError text can carry internals (DSNs, paths, request
    IDs) and is rendered in chat, spoken by voice pipelines, and persisted
    into the model's future context — so the message must stay in the log.
    """
    result = _streaming_failure_content(
        RuntimeError("internal-detail-that-must-not-leak /var/lib/private/path")
    )
    assert "internal-detail" not in result
    assert "/var/lib" not in result
    assert (
        result == "I'm sorry, I was unable to respond (RuntimeError). Please try again."
    )


def test_streaming_failure_content_truncates_long_reason() -> None:
    """Reasons longer than the cap are truncated with an ellipsis marker."""
    long_reason = "x" * (_STREAM_ERROR_REASON_MAX_CHARS + 100)
    result = _streaming_failure_content(HomeAssistantError(long_reason))
    reason = result.removeprefix("I'm sorry, I was unable to respond: ")
    assert len(reason) == _STREAM_ERROR_REASON_MAX_CHARS + 1
    assert reason.endswith("…")
    assert reason[:-1] == "x" * _STREAM_ERROR_REASON_MAX_CHARS


# ---------------------------------------------------------------------------
# _recommit_final_assistant_content: CONTENT_ADDED re-fire after stream failure
# ---------------------------------------------------------------------------


class _FakeChatLog:
    """Minimal ChatLog stand-in tracking recommitted assistant content."""

    def __init__(self, content: list[Any]) -> None:
        self.content = content
        self.recommitted: list[Any] = []

    def async_add_assistant_content_without_tools(self, item: Any) -> None:
        self.recommitted.append(item)
        self.content.append(item)


def _assistant_content(text: str, tool_calls: Any = None) -> Any:
    """
    Build an AssistantContent instance with the given payload.

    Works against both the import stub (no-arg class) and the real HA
    dataclass (required kwargs), since suite ordering decides which one
    is loaded when this module runs.
    """
    try:
        item: Any = ha_conversation.AssistantContent(
            agent_id="conversation.test", content=text
        )
    except TypeError:
        stub_cls: Any = ha_conversation.AssistantContent
        item = stub_cls()
        item.content = text
    item.tool_calls = tool_calls
    return item


def _user_content() -> Any:
    """Build a UserContent instance against either the stub or the real class."""
    try:
        return ha_conversation.UserContent(content="hi")
    except TypeError:
        stub_cls: Any = ha_conversation.UserContent
        return stub_cls()


def test_recommit_refires_final_assistant_content() -> None:
    """
    The final tool-free AssistantContent is popped and re-added.

    Re-adding fires CONTENT_ADDED so the frontend streaming UI shows the
    final text in the main chat area.
    """
    final = _assistant_content("Here is your answer.")
    chat_log = _FakeChatLog([_user_content(), final])

    _recommit_final_assistant_content(chat_log)  # type: ignore[arg-type]

    assert chat_log.recommitted == [final]
    assert chat_log.content[-1] is final
    assert len(chat_log.content) == 2


def test_recommit_skips_empty_chat_log() -> None:
    """An empty chat log is left untouched."""
    chat_log = _FakeChatLog([])
    _recommit_final_assistant_content(chat_log)  # type: ignore[arg-type]
    assert chat_log.recommitted == []


def test_recommit_skips_content_with_tool_calls() -> None:
    """AssistantContent carrying tool calls must not be recommitted."""
    final = _assistant_content("calling tool", tool_calls=[MagicMock()])
    chat_log = _FakeChatLog([final])

    _recommit_final_assistant_content(chat_log)  # type: ignore[arg-type]

    assert chat_log.recommitted == []
    assert chat_log.content == [final]


def test_recommit_skips_empty_assistant_text() -> None:
    """AssistantContent with empty text must not be recommitted."""
    final = _assistant_content("")
    chat_log = _FakeChatLog([final])

    _recommit_final_assistant_content(chat_log)  # type: ignore[arg-type]

    assert chat_log.recommitted == []
    assert chat_log.content == [final]


def test_recommit_skips_non_assistant_final_content() -> None:
    """A trailing non-assistant entry (e.g. UserContent) is left in place."""
    user_item = _user_content()
    chat_log = _FakeChatLog([user_item])

    _recommit_final_assistant_content(chat_log)  # type: ignore[arg-type]

    assert chat_log.recommitted == []
    assert chat_log.content == [user_item]


# ---------------------------------------------------------------------------
# _async_index_tools: per-turn top-up guard (issue #554)
# ---------------------------------------------------------------------------

_CONV = "custom_components.home_generative_agent.conversation"


def _index_entity() -> Any:
    """Build a bare entity: _async_index_tools only touches self.hass."""
    entity = HGAConversationEntity.__new__(HGAConversationEntity)
    entity.hass = MagicMock()
    # Close coroutines handed to async_create_task so un-run background
    # indexing never triggers "coroutine was never awaited" warnings.
    entity.hass.async_create_task = MagicMock(side_effect=lambda coro: coro.close())
    return entity


def _index_runtime_data(**overrides: Any) -> Any:
    rd = types.SimpleNamespace(
        tool_index_ready=True,
        tool_indexing_in_progress=False,
        tool_index_failed=False,
        tool_content_hashes={},
        store=MagicMock(),
    )
    for key, value in overrides.items():
        setattr(rd, key, value)
    return rd


def _loaded_llm_api(*tool_names: str) -> MultiLLMAPI:
    """
    MultiLLMAPI whose assist API exposes the given tools live.

    Tools carry a description and schema because the inline delta path hashes
    them for real via _queue_api_instance_tools.
    """
    api: Any = types.SimpleNamespace(
        tools=[
            types.SimpleNamespace(
                name=name,
                description=f"{name} live",
                parameters={"type": "object", "properties": {}},
            )
            for name in tool_names
        ],
        custom_serializer=None,
    )
    return MultiLLMAPI({"assist": api}, routing_map={})


@pytest.mark.asyncio
async def test_index_tools_fast_path_all_live_keys_hashed() -> None:
    """All live tool keys already hashed: zero discovery, zero store writes."""
    entity = _index_entity()
    rd = _index_runtime_data(tool_content_hashes={"assist::HassCancelAllTimers": "h1"})

    with (
        patch.object(
            entity, "_async_discover_provider_tools", new=AsyncMock()
        ) as provider_mock,
        patch.object(
            entity, "_async_discover_local_tools", new=AsyncMock()
        ) as local_mock,
    ):
        await entity._async_index_tools(
            MagicMock(), rd, _loaded_llm_api("HassCancelAllTimers")
        )

    provider_mock.assert_not_called()
    local_mock.assert_not_called()
    entity.hass.async_create_task.assert_not_called()
    assert rd.tool_indexing_in_progress is False


@pytest.mark.asyncio
async def test_index_tools_delta_missing_key_indexes_inline() -> None:
    """A live tool missing from the index is indexed inline from loaded APIs."""
    entity = _index_entity()
    rd = _index_runtime_data(tool_content_hashes={"assist::HassCancelAllTimers": "h1"})

    with (
        patch.object(
            entity, "_async_discover_provider_tools", new=AsyncMock()
        ) as provider_mock,
        patch(f"{_CONV}.async_dispatcher_send") as dispatch_mock,
        patch(f"{_CONV}.gather_store_puts_in_chunks", new=AsyncMock()) as gather_mock,
    ):
        await entity._async_index_tools(
            MagicMock(),
            rd,
            _loaded_llm_api("HassCancelAllTimers", "HassStartTimer"),
        )

    # Inline await: the write ran before returning, sourced from the loaded
    # API instances — no rediscovery via llm.async_get_api (whose failure
    # would leave the key unindexed and re-fire discovery every turn).
    provider_mock.assert_not_called()
    gather_mock.assert_awaited_once()
    entity.hass.async_create_task.assert_not_called()
    assert "assist::HassStartTimer" in rd.tool_content_hashes
    assert rd.tool_index_ready is True
    assert rd.tool_indexing_in_progress is False
    # The sensor gets the cumulative indexed count, not the delta size.
    state, count = dispatch_mock.call_args_list[-1].args[2:4]
    assert state == "ready"
    assert count == len(rd.tool_content_hashes)


@pytest.mark.asyncio
async def test_index_tools_delta_write_failure_does_not_latch_failed() -> None:
    """
    An inline delta-write failure must not latch tool_index_failed.

    The pre-delta index is still valid and keeps serving retrieval; the next
    turn recomputes the same missing keys and retries. Latching the failed
    flag here would permanently disable top-ups after one transient store or
    embedding-provider blip.
    """
    entity = _index_entity()
    rd = _index_runtime_data(tool_content_hashes={})

    with (
        patch(f"{_CONV}.async_dispatcher_send"),
        patch(
            f"{_CONV}.gather_store_puts_in_chunks",
            new=AsyncMock(side_effect=RuntimeError("embedding provider down")),
        ),
    ):
        await entity._async_index_tools(
            MagicMock(), rd, _loaded_llm_api("HassStartTimer")
        )

    assert rd.tool_index_failed is False
    assert rd.tool_index_ready is True
    assert rd.tool_indexing_in_progress is False
    assert "assist::HassStartTimer" not in rd.tool_content_hashes


@pytest.mark.asyncio
async def test_index_tools_delta_cancelled_resets_in_progress() -> None:
    """
    A cancelled turn mid-write cannot latch the in-progress guard.

    Assist pipeline runs are routinely cancelled (client disconnect, pipeline
    timeout); a latched flag would silently block all future indexing.
    """
    entity = _index_entity()
    rd = _index_runtime_data(tool_content_hashes={})

    with (
        patch(f"{_CONV}.async_dispatcher_send"),
        patch(
            f"{_CONV}.gather_store_puts_in_chunks",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await entity._async_index_tools(
            MagicMock(), rd, _loaded_llm_api("HassStartTimer")
        )

    assert rd.tool_indexing_in_progress is False
    assert rd.tool_index_failed is False


@pytest.mark.asyncio
async def test_index_tools_ready_without_llm_api_is_noop() -> None:
    """The startup-style call (no llm_api) stays a no-op once ready."""
    entity = _index_entity()
    rd = _index_runtime_data(tool_content_hashes={})

    with (
        patch.object(
            entity, "_async_discover_provider_tools", new=AsyncMock()
        ) as provider_mock,
        patch.object(entity, "_async_discover_local_tools", new=AsyncMock()),
    ):
        await entity._async_index_tools(MagicMock(), rd)

    provider_mock.assert_not_called()
    entity.hass.async_create_task.assert_not_called()


@pytest.mark.asyncio
async def test_index_tools_in_progress_short_circuits_delta() -> None:
    """tool_indexing_in_progress blocks the top-up even with missing keys."""
    entity = _index_entity()
    rd = _index_runtime_data(tool_indexing_in_progress=True, tool_content_hashes={})

    with patch.object(
        entity, "_async_discover_provider_tools", new=AsyncMock()
    ) as provider_mock:
        await entity._async_index_tools(
            MagicMock(), rd, _loaded_llm_api("HassStartTimer")
        )

    provider_mock.assert_not_called()
    assert rd.tool_indexing_in_progress is True


@pytest.mark.asyncio
async def test_index_tools_failed_short_circuits_delta() -> None:
    """tool_index_failed blocks the top-up even with missing keys."""
    entity = _index_entity()
    rd = _index_runtime_data(tool_index_failed=True, tool_content_hashes={})

    with patch.object(
        entity, "_async_discover_provider_tools", new=AsyncMock()
    ) as provider_mock:
        await entity._async_index_tools(
            MagicMock(), rd, _loaded_llm_api("HassStartTimer")
        )

    provider_mock.assert_not_called()


@pytest.mark.asyncio
async def test_index_tools_startup_path_still_backgrounds() -> None:
    """The initial (not-ready) indexing pass still runs as a background task."""
    entity = _index_entity()
    rd = _index_runtime_data(tool_index_ready=False, tool_content_hashes={})

    async def fake_provider_discovery(
        _llm_context: Any,
        _runtime_data: Any,
        _api_ids: Any,
        index_tasks: list[Any],
        new_hashes: dict[str, str],
    ) -> None:
        index_tasks.append(MagicMock())
        new_hashes["assist::HassTurnOn"] = "h1"

    with (
        patch.object(
            entity,
            "_async_discover_provider_tools",
            new=AsyncMock(side_effect=fake_provider_discovery),
        ),
        patch.object(entity, "_async_discover_local_tools", new=AsyncMock()),
        patch(f"{_CONV}.llm.async_get_apis", return_value=[]),
        patch(f"{_CONV}.async_dispatcher_send"),
    ):
        await entity._async_index_tools(MagicMock(), rd)

    entity.hass.async_create_task.assert_called_once()
    # The background task was not executed, so hashes are still pending.
    assert rd.tool_content_hashes == {}


@pytest.mark.asyncio
async def test_index_tools_no_changes_resets_in_progress() -> None:
    """A pass that queues no writes marks ready and clears in_progress."""
    entity = _index_entity()
    rd = _index_runtime_data(tool_index_ready=False, tool_content_hashes={})

    with (
        patch.object(entity, "_async_discover_provider_tools", new=AsyncMock()),
        patch.object(entity, "_async_discover_local_tools", new=AsyncMock()),
        patch(f"{_CONV}.llm.async_get_apis", return_value=[]),
        patch(f"{_CONV}.async_dispatcher_send"),
    ):
        await entity._async_index_tools(MagicMock(), rd)

    assert rd.tool_index_ready is True
    assert rd.tool_indexing_in_progress is False
    entity.hass.async_create_task.assert_not_called()


@pytest.mark.asyncio
async def test_index_tools_delta_without_new_writes_resets_in_progress() -> None:
    """
    A delta pass that queues no writes must clear in_progress and warn.

    Regression guard for the top-up path: a live key can be missing from the
    hash cache while queueing still produces nothing (e.g. a tool schema that
    fails to serialize). Without the reset, the stuck tool_indexing_in_progress
    flag would short-circuit every future turn.
    """
    entity = _index_entity()
    rd = _index_runtime_data(tool_content_hashes={})

    with (
        patch.object(entity, "_queue_api_instance_tools") as queue_mock,
        patch(f"{_CONV}.async_dispatcher_send"),
    ):
        await entity._async_index_tools(
            MagicMock(), rd, _loaded_llm_api("HassStartTimer")
        )

    queue_mock.assert_called_once()
    assert rd.tool_index_ready is True
    assert rd.tool_indexing_in_progress is False
    entity.hass.async_create_task.assert_not_called()


def test_per_turn_call_site_passes_llm_api() -> None:
    """
    Pin the call-site wiring: llm_api must reach _async_index_tools.

    A regression dropping the third argument would silently disable the
    issue-#554 top-up (llm_api=None makes the ready guard a no-op) with no
    test failure and no visible error. The full turn cannot be driven in this
    venv (the HA conversation stack is stubbed), so pin the source instead.
    """
    src = inspect.getsource(HGAConversationEntity._async_handle_message_active)
    assert re.search(
        r"_async_index_tools\(\s*llm_context,\s*runtime_data,\s*llm_api\s*\)", src
    )


@pytest.mark.asyncio
async def test_gather_store_puts_closes_remaining_on_failure() -> None:
    """A failing chunk closes later never-scheduled coroutines and re-raises."""
    ran: list[str] = []

    async def ok(tag: str) -> None:
        ran.append(tag)

    async def boom() -> None:
        msg = "store down"
        raise RuntimeError(msg)

    later = ok("later")
    with pytest.raises(RuntimeError, match="store down"):
        await gather_store_puts_in_chunks([ok("first"), boom(), later], chunk_size=2)

    # The failing chunk's sibling still completed (no detached tasks)...
    assert ran == ["first"]
    # ...and the never-scheduled trailing coroutine was closed, not leaked.
    assert later.cr_frame is None
