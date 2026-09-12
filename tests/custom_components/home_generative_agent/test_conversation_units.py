# ruff: noqa: S101
"""
Unit tests for conversation.py helpers (MultiLLMAPI, _run_tool_index_background).

hassil is not installed in the test venv, so this module stubs the entire
homeassistant.components.conversation import chain before importing conversation.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import re
import sys
import types
from enum import IntFlag
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import llm as ha_llm
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_TOOL_EXCLUSIONS,
)
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
    if not hasattr(conv, "ToolResultContent"):

        class _StubToolResultContent:
            pass

        conv.ToolResultContent = _StubToolResultContent


def _ensure_conversation_entity_feature() -> None:
    """
    Guarantee ConversationEntityFeature exists on the loaded module.

    Same suite-ordering concern as _ensure_content_classes: whichever stub won
    the import race may omit the feature enum. HGAConversationEntity.__init__
    reads CONTROL off it to decide whether Home Assistant is allowed to handle
    control commands in its own intent handler before the agent sees them.
    """
    conv: Any = sys.modules["homeassistant.components.conversation"]
    if not hasattr(conv, "ConversationEntityFeature"):

        class _StubConversationEntityFeature(IntFlag):
            CONTROL = 1

        conv.ConversationEntityFeature = _StubConversationEntityFeature


def _ensure_trace_symbols() -> None:
    """
    Guarantee the trace symbols APIInstance.async_call_tool imports exist.

    Home Assistant resolves them lazily *inside* async_call_tool, so any test
    that dispatches a tool through a real APIInstance needs them on whichever
    stub won the import race.
    """
    conv: Any = sys.modules["homeassistant.components.conversation"]
    if not hasattr(conv, "ConversationTraceEventType"):
        conv.ConversationTraceEventType = MagicMock()
    if not hasattr(conv, "async_conversation_trace_append"):
        conv.async_conversation_trace_append = MagicMock()


_stub_ha_conversation()
_ensure_content_classes()
_ensure_conversation_entity_feature()
_ensure_trace_symbols()

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
# _async_init_llm_apis: per-tool exclusions (issue #570)
# ---------------------------------------------------------------------------


def _excl_api_instance(names: list[str]) -> Any:
    """Build a real APIInstance so filter_excluded_tools' replace() applies."""
    tools: list[Any] = []
    for name in names:
        tool = MagicMock()
        tool.name = name
        tools.append(tool)
    return ha_llm.APIInstance(
        api=MagicMock(),
        api_prompt="prompt",
        llm_context=MagicMock(),
        tools=tools,
    )


def _excl_entity(options: dict[str, Any]) -> Any:
    """Build a bare entity: _async_init_llm_apis only reads hass and options."""
    entity = HGAConversationEntity.__new__(HGAConversationEntity)
    entity.hass = MagicMock()
    entity.entry = cast(
        "Any",
        types.SimpleNamespace(runtime_data=types.SimpleNamespace(options=options)),
    )
    return entity


@pytest.mark.asyncio
async def test_init_llm_apis_drops_excluded_tools() -> None:
    """Excluded tools never reach the loaded API instance."""
    entity = _excl_entity(
        {
            CONF_LLM_HASS_API: ["mcp-abc"],
            CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["web_search_images"]},
        }
    )
    instance = _excl_api_instance(["web_search", "web_search_images"])

    with patch(f"{_CONV}.llm.async_get_api", new=AsyncMock(return_value=instance)):
        multi = await entity._async_init_llm_apis(MagicMock())

    assert [tool.name for tool in multi.apis["mcp-abc"].tools] == ["web_search"]


@pytest.mark.asyncio
async def test_init_llm_apis_excluded_tool_cannot_be_dispatched() -> None:
    """
    A hallucinated call to an excluded tool is rejected, not executed.

    APIInstance.async_call_tool resolves the name against `.tools`, which is
    what makes the exclusion deterministic rather than merely advisory.
    """
    entity = _excl_entity(
        {
            CONF_LLM_HASS_API: ["mcp-abc"],
            CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["web_search_images"]},
        }
    )
    instance = _excl_api_instance(["web_search", "web_search_images"])

    with patch(f"{_CONV}.llm.async_get_api", new=AsyncMock(return_value=instance)):
        multi = await entity._async_init_llm_apis(MagicMock())

    tool_input = MagicMock()
    tool_input.tool_name = "web_search_images"

    with pytest.raises(HomeAssistantError, match="No routing target"):
        await multi.async_call_tool(tool_input)


@pytest.mark.asyncio
async def test_init_llm_apis_without_exclusions_is_unchanged() -> None:
    """The absent-key default exposes every tool, as before the feature."""
    entity = _excl_entity({CONF_LLM_HASS_API: ["mcp-abc"]})
    instance = _excl_api_instance(["web_search", "web_search_images"])

    with patch(f"{_CONV}.llm.async_get_api", new=AsyncMock(return_value=instance)):
        multi = await entity._async_init_llm_apis(MagicMock())

    assert multi.apis["mcp-abc"] is instance


@pytest.mark.asyncio
async def test_init_llm_apis_excluding_every_tool_keeps_the_api_loaded() -> None:
    """
    An emptied API still counts as loaded.

    Dropping it would trip the "No LLM APIs could be loaded" hard failure when
    it is the only configured API, turning a tool preference into a broken
    conversation agent.
    """
    entity = _excl_entity(
        {
            CONF_LLM_HASS_API: ["mcp-abc"],
            CONF_TOOL_EXCLUSIONS: {"mcp-abc": ["web_search"]},
        }
    )
    instance = _excl_api_instance(["web_search"])

    with patch(f"{_CONV}.llm.async_get_api", new=AsyncMock(return_value=instance)):
        multi = await entity._async_init_llm_apis(MagicMock())

    assert multi.apis["mcp-abc"].tools == []


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


class _PoisonedTool:
    """A live tool whose schema cannot be read, whatever converter core uses."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"{name} live"

    @property
    def parameters(self) -> Any:
        msg = "unserializable schema"
        raise TypeError(msg)


@pytest.mark.asyncio
async def test_index_tools_poisoned_tool_does_not_starve_neighbors() -> None:
    """
    One unserializable tool must not block indexing of its API's other tools.

    Without per-tool isolation, a poisoned schema aborts the whole API's
    queue pass — the healthy neighbor's key stays missing forever and the
    top-up re-fires every turn (the very bug the top-up exists to fix).
    """
    entity = _index_entity()
    rd = _index_runtime_data(tool_content_hashes={})

    llm_api = _loaded_llm_api("HassStartTimer", "HassPauseTimer")
    # Poison the first tool: its schema blows up on access. A converter-level
    # poison is core-specific (voluptuous-openapi raises on an object() leaf,
    # probatio renders it as a string), so the failure is raised upstream of
    # the converter to hold on both cores.
    tools: Any = llm_api.apis["assist"].tools
    tools[0] = _PoisonedTool("HassStartTimer")

    with (
        patch(f"{_CONV}.async_dispatcher_send"),
        patch(f"{_CONV}.gather_store_puts_in_chunks", new=AsyncMock()),
    ):
        await entity._async_index_tools(MagicMock(), rd, llm_api)

    assert "assist::HassStartTimer" not in rd.tool_content_hashes
    assert "assist::HassPauseTimer" in rd.tool_content_hashes
    assert rd.tool_indexing_in_progress is False


@pytest.mark.asyncio
async def test_delta_write_failure_sends_terminal_ready_signal() -> None:
    """A failed delta write must not leave the sensor stuck on 'indexing'."""
    entity = _index_entity()
    rd = _index_runtime_data(tool_content_hashes={"assist::HassCancelAllTimers": "h1"})

    with (
        patch(f"{_CONV}.async_dispatcher_send") as dispatch_mock,
        patch(
            f"{_CONV}.gather_store_puts_in_chunks",
            new=AsyncMock(side_effect=RuntimeError("embedding provider down")),
        ),
    ):
        await entity._async_index_tools(
            MagicMock(), rd, _loaded_llm_api("HassStartTimer")
        )

    # Retrieval still serves the pre-delta index, so the terminal state is
    # "ready" with the cumulative count — never a stuck "indexing".
    state, count = dispatch_mock.call_args_list[-1].args[2:4]
    assert state == "ready"
    assert count == len(rd.tool_content_hashes)


def _entity_with_options(options: dict[str, Any]) -> Any:
    """Construct the entity through its real __init__ with the given options."""
    entry: Any = types.SimpleNamespace(
        entry_id="test-entry-id",
        title="Home Generative Agent",
        runtime_data=types.SimpleNamespace(options=options),
    )
    return HGAConversationEntity(entry)


def _supported_features(entity: Any) -> int:
    """Read the feature flag without assuming the attribute was ever set."""
    return int(getattr(entity, "_attr_supported_features", 0) or 0)


def test_control_feature_set_when_llm_api_key_absent() -> None:
    """
    An absent CONF_LLM_HASS_API must still advertise CONTROL.

    The options flow deletes the key when no API is selected, and the rest of
    the integration reads that absence as "default to Assist"; __init__ read it
    as falsy, so the entity's capability disagreed with the tools it actually
    had.

    CONTROL is NOT a security control -- its only consumer routes state
    questions and media search to the agent. This test pins capability/tool
    agreement, nothing about the PIN.
    """
    control = ha_conversation.ConversationEntityFeature.CONTROL
    assert _supported_features(_entity_with_options({})) & control


def test_control_feature_set_when_llm_api_configured() -> None:
    """An explicitly configured API advertises CONTROL, as it always did."""
    control = ha_conversation.ConversationEntityFeature.CONTROL
    entity = _entity_with_options({CONF_LLM_HASS_API: ["assist"]})
    assert _supported_features(entity) & control


def test_control_feature_absent_when_stored_api_list_is_empty() -> None:
    """
    A stored empty list must NOT advertise CONTROL.

    With no APIs the agent has no entity-control tools and no GetLiveContext,
    so it has nothing better to offer than Home Assistant's own sentence
    matcher and should not claim the capability. The v5 -> v6 migration writes
    [] for an absent key, so this state is reachable on upgraded installs.

    A PIN does not change this. An earlier revision of this branch also set
    CONTROL whenever a PIN was configured, on the belief that it would force
    lock commands through the agent. It does not: the filter CONTROL installs
    is a reject list matching only HassGetState and media search, so control
    commands stay local either way. The PIN/pipeline conflict is surfaced as a
    repair issue instead -- see test_pipeline_guard.py.
    """
    control = ha_conversation.ConversationEntityFeature.CONTROL
    entity = _entity_with_options(
        {
            CONF_LLM_HASS_API: [],
            CONF_CRITICAL_ACTION_PIN_ENABLED: True,
        }
    )
    assert not _supported_features(entity) & control


# ---------------------------------------------------------------------------
# _async_get_message_history: only foreign turns reach the thread (#588, #621, #590)
# ---------------------------------------------------------------------------

HGA_AGENT_ID = "conversation.home_generative_agent"
FOREIGN_AGENT_ID = "conversation.home_assistant"


def _mk_content(cls: Any, **kwargs: Any) -> Any:
    """
    Build a chat_log content object for the real class or the module stub.

    The real HA classes are frozen dataclasses (constructor works, setattr does
    not); the suite's lean stubs are bare classes (constructor takes nothing).
    """
    try:
        return cls(**kwargs)
    except TypeError:
        obj = cls()
        for key, value in kwargs.items():
            object.__setattr__(obj, key, value)
        return obj


def _history(content: list, entity: Any | None = None) -> list:
    """Run _async_get_message_history for this entity over a chat_log."""
    fake_self = entity if entity is not None else _entity()
    chat_log = cast(
        "Any", types.SimpleNamespace(content=content, conversation_id="conv-1")
    )
    return HGAConversationEntity._async_get_message_history(fake_self, chat_log)


def _entity() -> Any:
    """Build a bare entity: the filter needs nothing beyond its own entity_id."""
    return cast("Any", types.SimpleNamespace(entity_id=HGA_AGENT_ID))


def _own_tool_turn(question: str, answer: str) -> list:
    """Build what HGA itself writes to chat_log for a turn that called a tool."""
    return [
        _mk_content(ha_conversation.UserContent, content=question),
        _mk_content(
            ha_conversation.AssistantContent,
            agent_id=HGA_AGENT_ID,
            content=None,
            tool_calls=[object()],
        ),
        _mk_content(
            ha_conversation.ToolResultContent,
            agent_id=HGA_AGENT_ID,
            tool_call_id="call_1",
            tool_name="plex_library",
            tool_result={"movies": 1389},
        ),
        _mk_content(
            ha_conversation.AssistantContent,
            agent_id=HGA_AGENT_ID,
            content=answer,
            tool_calls=None,
        ),
    ]


def _plain_turn(agent_id: str, question: str, answer: str) -> list:
    """Build a user question answered in prose, with no tool call, by `agent_id`."""
    return [
        _mk_content(ha_conversation.UserContent, content=question),
        _mk_content(
            ha_conversation.AssistantContent,
            agent_id=agent_id,
            content=answer,
            tool_calls=None,
        ),
    ]


def test_message_history_drops_spoken_text_of_a_tool_using_turn() -> None:
    """
    A turn that used tools is not ingested as a bare assistant reply.

    Regression for issue #588. When another agent sharing the conversation
    (Home Assistant's built-in agent) handles a device command, chat_log gets
    three entries: the tool_calls, the ToolResultContent, then the spoken
    "Turned on the light". Ingesting only the third teaches the model that
    "turn on the light" is answered with prose and no tool call — after which
    the model repeats that shape and every device command silently becomes a
    lie. The whole turn must be dropped, not just its tool_calls entry.
    """
    content = [
        _mk_content(ha_conversation.UserContent, content="Turn on the garage light."),
        _mk_content(
            ha_conversation.AssistantContent,
            agent_id=FOREIGN_AGENT_ID,
            content=None,
            tool_calls=[object()],
        ),
        _mk_content(
            ha_conversation.ToolResultContent,
            agent_id=FOREIGN_AGENT_ID,
            tool_call_id="01M16N286Y2A5T0BVZ163SMKT7",
            tool_name="HassTurnOn",
            tool_result={"speech": {"plain": {"speech": "Turned on the light"}}},
        ),
        _mk_content(
            ha_conversation.AssistantContent,
            agent_id=FOREIGN_AGENT_ID,
            content="Turned on the light",
            tool_calls=None,
        ),
        _mk_content(ha_conversation.UserContent, content="Turn off the garage light."),
    ]

    history = _history(content)

    assert [type(m).__name__ for m in history] == ["HumanMessage"], (
        "the spoken tail of a tool-using turn must not reach the model"
    )
    assert history[0].content == "Turn on the garage light."
    assert not any("Turned on the light" in str(m.content) for m in history), (
        "an assistant reply with the tool call erased is the poison itself"
    )


def test_message_history_keeps_a_genuine_toolless_reply() -> None:
    """A foreign turn that really answered without tools is still ingested."""
    content = [
        _mk_content(ha_conversation.UserContent, content="what can you do?"),
        _mk_content(
            ha_conversation.AssistantContent,
            agent_id=FOREIGN_AGENT_ID,
            content="I can control your home.",
            tool_calls=None,
        ),
        _mk_content(ha_conversation.UserContent, content="Turn off the garage light."),
    ]

    history = _history(content)

    assert [type(m).__name__ for m in history] == ["HumanMessage", "AIMessage"], (
        "over-filtering would strip ordinary conversational context"
    )
    assert history[1].content == "I can control your home."


def test_message_history_tool_flag_resets_on_the_next_user_turn() -> None:
    """One tool-using turn must not suppress every later reply."""
    content = [
        _mk_content(ha_conversation.UserContent, content="Turn on the garage light."),
        _mk_content(
            ha_conversation.AssistantContent,
            agent_id=FOREIGN_AGENT_ID,
            content=None,
            tool_calls=[object()],
        ),
        _mk_content(
            ha_conversation.ToolResultContent,
            agent_id=FOREIGN_AGENT_ID,
            tool_call_id="call_1",
            tool_name="HassTurnOn",
            tool_result={},
        ),
        _mk_content(
            ha_conversation.AssistantContent,
            agent_id=FOREIGN_AGENT_ID,
            content="Turned on the light",
            tool_calls=None,
        ),
        _mk_content(ha_conversation.UserContent, content="thanks"),
        _mk_content(
            ha_conversation.AssistantContent,
            agent_id=FOREIGN_AGENT_ID,
            content="You're welcome.",
            tool_calls=None,
        ),
        _mk_content(ha_conversation.UserContent, content="Turn off the garage light."),
    ]

    history = _history(content)

    assert [type(m).__name__ for m in history] == [
        "HumanMessage",
        "HumanMessage",
        "AIMessage",
    ], "the tool flag must clear at the next user message"
    assert history[2].content == "You're welcome."


def test_message_history_non_none_tool_calls_still_excluded() -> None:
    """
    An empty-but-present tool_calls list is excluded, as it always was.

    The inclusion predicate stays `tool_calls is None`, so this fix changes
    only which *other* entries a tool-using turn suppresses. Guards against
    quietly widening the filter while fixing #588.
    """
    content = [
        _mk_content(ha_conversation.UserContent, content="hello"),
        _mk_content(
            ha_conversation.AssistantContent,
            agent_id=FOREIGN_AGENT_ID,
            content="Hi there.",
            tool_calls=[],
        ),
        _mk_content(ha_conversation.UserContent, content="Turn off the garage light."),
    ]

    history = _history(content)

    assert [type(m).__name__ for m in history] == ["HumanMessage"]
    assert history[0].content == "hello"


def test_message_history_skips_the_entitys_own_turns() -> None:
    """
    Nothing HGA itself said is fed back into its own thread.

    Regression for issue #621. The LangGraph checkpointer already holds every
    turn this entity handled, so re-ingesting the chat_log echo of one appends a
    second copy under a fresh message id. The stale copy then sits directly
    before the new request and the model answers it instead — a question about
    free disk space got the previous turn's movie count. Both a tool-using turn
    and a plain prose turn must contribute nothing.
    """
    content = [
        *_own_tool_turn("How many movies are in my library?", "1,389 movies."),
        *_plain_turn(HGA_AGENT_ID, "Thanks!", "Any time."),
        _mk_content(
            ha_conversation.UserContent, content="How much free space is left?"
        ),
    ]

    assert _history(content) == [], "our own turns are already in the thread"


def test_message_history_ingests_only_foreign_turns_after_our_latest_turn() -> None:
    """
    A foreign turn older than our latest turn was ingested when that turn ran.

    Replaces the per-entity high-water mark with a stateless rule: everything
    before HGA's most recent own turn is already in the thread; only foreign
    turns after it are new.
    """
    content = [
        *_plain_turn(FOREIGN_AGENT_ID, "what time is it?", "It is 4:42 AM."),
        *_own_tool_turn("Turn on the garage light.", "Done."),
        *_plain_turn(FOREIGN_AGENT_ID, "and the date?", "September 10th."),
        _mk_content(ha_conversation.UserContent, content="Turn it off again."),
    ]

    history = _history(content)

    assert [(type(m).__name__, m.content) for m in history] == [
        ("HumanMessage", "and the date?"),
        ("AIMessage", "September 10th."),
    ], "only the foreign turn after our latest own turn is new"


def test_message_history_is_per_conversation() -> None:
    """
    Two conversations served by one entity do not starve each other.

    Regression for issue #590: the old counter lived on the entity while
    chat_log is per conversation, so a second conversation got no history until
    it outgrew the first. The walk now needs no entity state at all.
    """
    entity = _entity()
    conv_a = [
        *_plain_turn(FOREIGN_AGENT_ID, "A question", "A answer"),
        _mk_content(ha_conversation.UserContent, content="A follow-up"),
    ]
    conv_b = [
        *_plain_turn(FOREIGN_AGENT_ID, "B question", "B answer"),
        _mk_content(ha_conversation.UserContent, content="B follow-up"),
    ]

    assert [m.content for m in _history(conv_a, entity)] == ["A question", "A answer"]
    assert [m.content for m in _history(conv_b, entity)] == ["B question", "B answer"]
    assert [m.content for m in _history(conv_a, entity)] == ["A question", "A answer"]


def test_message_history_own_failed_turn_still_marks_the_boundary() -> None:
    """
    A turn we failed still counts as ours, so foreign history is not replayed.

    The non-streaming path records its failure as an own AssistantContent;
    the foreign turn before it was already prepended to the failed request
    and checkpointed, so it must not be ingested a second time.
    """
    content = [
        *_plain_turn(FOREIGN_AGENT_ID, "what time is it?", "It is 4:42 AM."),
        _mk_content(ha_conversation.UserContent, content="Turn on the garage light."),
        _mk_content(
            ha_conversation.AssistantContent,
            agent_id=HGA_AGENT_ID,
            content="I'm sorry, I was unable to respond: Something went wrong",
            tool_calls=None,
        ),
        _mk_content(ha_conversation.UserContent, content="Try again."),
    ]

    assert _history(content) == []


def test_message_history_locally_handled_intent_is_a_foreign_turn() -> None:
    """
    A turn HA's built-in agent handled on our pipeline is not ours.

    With prefer_local_intents the default agent stamps its tool_calls and
    ToolResultContent with the pipeline's agent id — this entity's — but the
    pipeline speaks the result under conversation.home_assistant. Judging the
    turn by any entry would swallow it and every foreign turn before it; the
    user's message must survive (the #588 rule drops the rest).
    """
    content = [
        *_plain_turn(FOREIGN_AGENT_ID, "what time is it?", "It is 4:42 AM."),
        _mk_content(ha_conversation.UserContent, content="Turn on the garage light."),
        _mk_content(
            ha_conversation.AssistantContent,
            agent_id=HGA_AGENT_ID,
            content=None,
            tool_calls=[object()],
        ),
        _mk_content(
            ha_conversation.ToolResultContent,
            agent_id=HGA_AGENT_ID,
            tool_call_id="01M16N286Y2A5T0BVZ163SMKT7",
            tool_name="HassTurnOn",
            tool_result={"speech": {"plain": {"speech": "Turned on the light"}}},
        ),
        _mk_content(
            ha_conversation.AssistantContent,
            agent_id=FOREIGN_AGENT_ID,
            content="Turned on the light",
            tool_calls=None,
        ),
        _mk_content(ha_conversation.UserContent, content="Turn it off again."),
    ]

    assert [m.content for m in _history(content)] == [
        "what time is it?",
        "It is 4:42 AM.",
        "Turn on the garage light.",
    ]


def test_message_history_ids_make_re_ingestion_an_upsert() -> None:
    """
    Ingesting the same foreign turn twice leaves one copy in the thread.

    A turn of ours that fails or is cancelled before it leaves its mark means
    the foreign history prepended to it is offered again next turn. The ids
    derive from chat_log position, so add_messages replaces in place.
    """
    content = [
        *_plain_turn(FOREIGN_AGENT_ID, "what time is it?", "It is 4:42 AM."),
        _mk_content(ha_conversation.UserContent, content="Turn on the garage light."),
    ]
    first = _history(content)
    second = _history(content)
    assert [m.id for m in first] == [m.id for m in second]
    assert first[0].id == "chat_log:conv-1:0"

    graph = StateGraph(MessagesState)
    graph.add_node("agent", _no_op_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    app = graph.compile(checkpointer=MemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "conv-1"}}
    app.invoke({"messages": first}, config)
    app.invoke({"messages": second}, config)

    assert [m.content for m in app.get_state(config).values["messages"]] == [
        "what time is it?",
        "It is 4:42 AM.",
    ]


def _no_op_node(state: MessagesState) -> dict[str, Any]:  # noqa: ARG001
    return {"messages": []}


def _runner_entity() -> Any:
    """Build an entity with just what _async_run_ainvoke touches."""
    entity = HGAConversationEntity.__new__(HGAConversationEntity)
    entity.hass = MagicMock()
    entity.entity_id = HGA_AGENT_ID
    entity.entry = cast(
        "Any",
        types.SimpleNamespace(runtime_data=types.SimpleNamespace(options={})),
    )
    return entity


class _RecordedContent:
    """Kwargs-accepting stand-in: the suite's HA import stub takes no arguments."""

    def __init__(self, **fields: Any) -> None:
        self.tool_calls = None
        self.__dict__.update(fields)


class _FakeAssistantContent(_RecordedContent):
    pass


class _FakeToolResultContent(_RecordedContent):
    pass


def _patch_runner_deps() -> list[Any]:
    """Patch what the runner touches that the stubbed venv cannot provide."""
    module = "custom_components.home_generative_agent.conversation"
    return [
        patch(
            f"{module}._fix_entity_ids_in_text", side_effect=lambda text, _hass: text
        ),
        patch(f"{module}.trace.async_conversation_trace_append"),
        patch(f"{module}.conversation.AssistantContent", _FakeAssistantContent),
        patch(f"{module}.conversation.ToolResultContent", _FakeToolResultContent),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_text"),
    [
        (
            HomeAssistantError("Model invocation failed: 429"),
            "I'm sorry, I was unable to respond: Model invocation failed: 429",
        ),
        (
            RuntimeError("psycopg lost the connection to db://secret"),
            "I'm sorry, I was unable to respond (RuntimeError). Please try again.",
        ),
    ],
)
async def test_non_streaming_failure_leaves_an_own_chat_log_entry(
    error: Exception, expected_text: str
) -> None:
    """
    A failed non-streaming turn still writes our AssistantContent.

    Without it HA discards the whole turn from chat_log (no assistant message
    was added), the boundary rule in _async_get_message_history sees no own
    turn, and the foreign history already checkpointed with the failed request
    is offered again next turn (issue #621 review finding). Any exception
    counts, not only HomeAssistantError, and arbitrary ones surface only their
    class name.
    """
    request = HumanMessage(content="Turn on the garage light.", id="req-1")
    app = types.SimpleNamespace(ainvoke=AsyncMock(side_effect=error))
    written: list[Any] = []
    chat_log = types.SimpleNamespace(
        content=written,
        async_add_assistant_content_without_tools=written.append,
    )
    entity = _runner_entity()

    with contextlib.ExitStack() as stack:
        for patcher in _patch_runner_deps():
            stack.enter_context(patcher)
        with pytest.raises(HomeAssistantError):
            await entity._async_run_ainvoke(
                app,
                cast("Any", {"messages": [request]}),
                cast("Any", {}),
                cast("Any", chat_log),
                None,
            )

    assert len(written) == 1
    assert written[0].agent_id == HGA_AGENT_ID
    assert written[0].content == expected_text
    assert "db://secret" not in written[0].content


@pytest.mark.asyncio
async def test_non_streaming_backfill_only_writes_this_turns_messages() -> None:
    """
    The runner gets the whole thread back; only this turn reaches chat_log.

    The old slice `response["messages"][len(input):]` assumed the response was
    input plus output. With a checkpointed thread it is the full history, so
    from the second turn on every earlier reply was written to chat_log again.
    """
    request = HumanMessage(content="How much free space is left?", id="req-2")
    thread = [
        HumanMessage(content="How many movies are in my library?", id="req-1"),
        AIMessage(content="", tool_calls=[{"name": "plex", "args": {}, "id": "c1"}]),
        ToolMessage(content="1389", tool_call_id="c1", name="plex"),
        AIMessage(content="You have 1,389 movies."),
        request,
        AIMessage(content="", tool_calls=[{"name": "disk", "args": {}, "id": "c2"}]),
        ToolMessage(content="2 TB", tool_call_id="c2", name="disk"),
        AIMessage(content="About 2 TB free."),
    ]
    app = types.SimpleNamespace(ainvoke=AsyncMock(return_value={"messages": thread}))

    written: list[Any] = []
    chat_log = types.SimpleNamespace(
        content=written,
        async_add_assistant_content_without_tools=written.append,
    )
    entity = _runner_entity()

    with contextlib.ExitStack() as stack:
        for patcher in _patch_runner_deps():
            stack.enter_context(patcher)
        await entity._async_run_ainvoke(
            app,
            cast("Any", {"messages": [request]}),
            cast("Any", {}),
            cast("Any", chat_log),
            None,
        )

    contents = [getattr(entry, "content", None) for entry in written]
    assert "You have 1,389 movies." not in contents, (
        "the previous turn's reply must not be written to chat_log again"
    )
    assert [type(entry).__name__.removeprefix("_Fake") for entry in written] == [
        "AssistantContent",
        "ToolResultContent",
        "AssistantContent",
    ]
    assert written[-1].content == "About 2 TB free."


def test_message_history_thread_holds_each_user_message_once() -> None:
    """
    End to end through the real reducer: four turns, no duplicate messages.

    Mirrors the trace in issue #621. The chat_log grows the way HGA writes it,
    the filter's output is prepended to each request exactly as
    _async_handle_message_active does, and the thread is checkpointed by
    LangGraph's add_messages reducer. Every user message must appear once.
    """
    turns = [
        ("How many movies are in my library?", "You have 1,389 movies."),
        ("How much free space is left?", "About 2 TB free."),
        ("Did the backup run?", "Yes, last night."),
        ("What's for dinner?", "Pasta?"),
    ]
    answers = iter(answer for _, answer in turns)

    def agent(state: MessagesState) -> dict[str, Any]:  # noqa: ARG001
        return {
            "messages": [
                AIMessage(
                    content="", tool_calls=[{"name": "t", "args": {}, "id": "c"}]
                ),
                AIMessage(content=next(answers)),
            ]
        }

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    app = graph.compile(checkpointer=MemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "conv-1"}}

    entity = _entity()
    chat_log: list = []
    for question, answer in turns:
        chat_log.append(_mk_content(ha_conversation.UserContent, content=question))
        history = _history(chat_log, entity)
        app.invoke({"messages": [*history, HumanMessage(content=question)]}, config)
        chat_log.extend(_own_tool_turn(question, answer)[1:])

    thread = app.get_state(config).values["messages"]
    human = [m.content for m in thread if isinstance(m, HumanMessage)]
    assert human == [question for question, _ in turns], (
        "each user message must be in the thread exactly once, in order"
    )
    assert len(thread) == 3 * len(turns)


# ---------------------------------------------------------------------------
# Issue #617 — _async_render_system_prompt splits stable and volatile parts
# ---------------------------------------------------------------------------


class _EchoTemplate:
    """Stand-in for template.Template that returns its source unrendered."""

    def __init__(self, source: str, _hass: Any) -> None:
        self.source = source

    def async_render(self, _variables: Any, *, parse_result: bool) -> str:
        assert parse_result is False
        return self.source


def _render_entity(options: dict[str, Any], *, sentinel: Any = None) -> Any:
    entity = HGAConversationEntity.__new__(HGAConversationEntity)
    entity.hass = MagicMock()
    entity.hass.config.location_name = "Home"
    entity.tz = ZoneInfo("America/Los_Angeles")
    entity.entry = cast(
        "Any",
        types.SimpleNamespace(
            runtime_data=types.SimpleNamespace(options=options, sentinel=sentinel)
        ),
    )
    return entity


def test_render_system_prompt_audit_instruction_follows_sentinel_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit_home_security instruction appears only when the tool can exist."""
    monkeypatch.setattr(f"{_CONV}.template.Template", _EchoTemplate)
    llm_api = cast("Any", types.SimpleNamespace(api_prompt="EXPOSED"))

    stable, _ = _render_entity({})._async_render_system_prompt(
        MagicMock(), None, llm_api, has_tools=True
    )
    assert "audit_home_security" not in stable

    stable, _ = _render_entity({}, sentinel=object())._async_render_system_prompt(
        MagicMock(), None, llm_api, has_tools=True
    )
    assert "audit_home_security" in stable
    # Still ahead of the tool-error rule that closes the stable prefix.
    assert stable.index("audit_home_security") < stable.index("Always call tools again")

    stable, _ = _render_entity(
        {"sentinel_network_enabled": False}, sentinel=object()
    )._async_render_system_prompt(MagicMock(), None, llm_api, has_tools=True)
    assert "audit_home_security" not in stable


def test_render_system_prompt_moves_date_time_out_of_the_stable_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The time line was the first bytes of the prompt; it must now be the tail."""
    monkeypatch.setattr(f"{_CONV}.template.Template", _EchoTemplate)
    entity = _render_entity({})
    llm_api = cast("Any", types.SimpleNamespace(api_prompt="EXPOSED ENTITIES"))

    stable, volatile = entity._async_render_system_prompt(
        MagicMock(), "lindo", llm_api, has_tools=True
    )

    assert volatile == ha_llm.DATE_TIME_PROMPT.strip()
    assert "Current time is" not in stable
    assert stable.startswith(ha_llm.DEFAULT_INSTRUCTIONS_PROMPT)
    assert "You are in the America/Los_Angeles timezone." in stable
    assert "Always call tools again with your mistakes corrected." in stable
    assert stable.endswith("\nEXPOSED ENTITIES")


def test_render_system_prompt_without_tools_keeps_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the tool-error rule is conditional on tools, not the whole prompt."""
    monkeypatch.setattr(f"{_CONV}.template.Template", _EchoTemplate)
    entity = _render_entity({})
    llm_api = cast("Any", types.SimpleNamespace(api_prompt="EXPOSED ENTITIES"))

    stable, volatile = entity._async_render_system_prompt(
        MagicMock(), None, llm_api, has_tools=False
    )

    assert stable.startswith(ha_llm.DEFAULT_INSTRUCTIONS_PROMPT)
    assert "Always call tools again" not in stable
    assert volatile == ha_llm.DATE_TIME_PROMPT.strip()


def test_render_system_prompt_without_llm_api_has_no_trailing_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(f"{_CONV}.template.Template", _EchoTemplate)
    entity = _render_entity({})

    stable, _volatile = entity._async_render_system_prompt(
        MagicMock(), None, cast("Any", None), has_tools=True
    )

    assert "None" not in stable
    assert stable.rstrip("\n").endswith("Do not repeat mistakes.")


def test_render_system_prompt_volatile_template_error_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing date/time render must surface the same HomeAssistantError."""
    calls: list[str] = []

    class _Boom(_EchoTemplate):
        def async_render(self, _variables: Any, *, parse_result: bool) -> str:
            calls.append(self.source)
            if len(calls) == 2:
                msg = "bad"
                raise TemplateError(msg)
            return self.source

    monkeypatch.setattr(f"{_CONV}.template.Template", _Boom)
    entity = _render_entity({})
    with pytest.raises(HomeAssistantError, match="Error rendering prompt"):
        entity._async_render_system_prompt(
            MagicMock(), None, cast("Any", None), has_tools=False
        )
    assert calls[1] == ha_llm.DATE_TIME_PROMPT


def test_handle_message_passes_volatile_prompt_to_the_graph() -> None:
    """Pin the call-site wiring: the volatile part must reach the graph config."""
    src = inspect.getsource(HGAConversationEntity._async_handle_message_active)
    assert "prompt, volatile_prompt = self._async_render_system_prompt(" in src
    assert '"prompt_volatile": volatile_prompt,' in src
