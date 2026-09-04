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
from enum import IntFlag
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm as ha_llm

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
# _async_get_message_history: a tool-using turn must not be ingested as prose
# ---------------------------------------------------------------------------


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


def _history(content: list) -> list:
    """Run _async_get_message_history against a fresh entity counter."""
    fake_self = cast("Any", types.SimpleNamespace(message_history_len=0))
    chat_log = cast("Any", types.SimpleNamespace(content=content))
    return HGAConversationEntity._async_get_message_history(fake_self, chat_log)


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
            agent_id="conversation.home_assistant",
            content=None,
            tool_calls=[object()],
        ),
        _mk_content(
            ha_conversation.ToolResultContent,
            agent_id="conversation.home_assistant",
            tool_call_id="01M16N286Y2A5T0BVZ163SMKT7",
            tool_name="HassTurnOn",
            tool_result={"speech": {"plain": {"speech": "Turned on the light"}}},
        ),
        _mk_content(
            ha_conversation.AssistantContent,
            agent_id="conversation.home_assistant",
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
    """A turn that really answered without tools is still ingested."""
    content = [
        _mk_content(ha_conversation.UserContent, content="what can you do?"),
        _mk_content(
            ha_conversation.AssistantContent,
            agent_id="conversation.home_generative_agent",
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
            agent_id="conversation.home_assistant",
            content=None,
            tool_calls=[object()],
        ),
        _mk_content(
            ha_conversation.ToolResultContent,
            agent_id="conversation.home_assistant",
            tool_call_id="call_1",
            tool_name="HassTurnOn",
            tool_result={},
        ),
        _mk_content(
            ha_conversation.AssistantContent,
            agent_id="conversation.home_assistant",
            content="Turned on the light",
            tool_calls=None,
        ),
        _mk_content(ha_conversation.UserContent, content="thanks"),
        _mk_content(
            ha_conversation.AssistantContent,
            agent_id="conversation.home_generative_agent",
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
            agent_id="conversation.home_generative_agent",
            content="Hi there.",
            tool_calls=[],
        ),
        _mk_content(ha_conversation.UserContent, content="Turn off the garage light."),
    ]

    history = _history(content)

    assert [type(m).__name__ for m in history] == ["HumanMessage"]
    assert history[0].content == "hello"
