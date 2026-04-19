"""Conversation support for Home Generative Agent using langgraph."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import string
from collections.abc import AsyncGenerator, AsyncIterable
from typing import TYPE_CHECKING, Any, Literal, cast

import homeassistant.util.dt as dt_util
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.components.conversation import (
    AssistantContentDeltaDict,
    ToolResultContentDeltaDict,
    trace,
)
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import intent, llm, template
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.util import ulid
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_debug, set_llm_cache, set_verbose
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from pydantic import PydanticInvalidForJsonSchema

from .agent.graph import workflow
from .agent.helpers import format_tool, is_actuation_tool, safe_convert
from .agent.rag_embedding_text import (
    strip_for_embedding,
    truncate_for_embedding_index,
)
from .agent.tools import (
    add_automation,
    alarm_control,
    confirm_sensitive_action,
    get_and_analyze_camera_image,
    get_camera_last_events,
    get_entity_history,
    resolve_entity_ids,
    upsert_memory,
    write_yaml_file,
)
from .const import (
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_PROMPT,
    CONF_SCHEMA_FIRST_YAML,
    CRITICAL_ACTION_PROMPT,
    DOMAIN,
    LANGCHAIN_LOGGING_LEVEL,
    SCHEMA_FIRST_YAML_PROMPT,
    SIGNAL_TOOL_INDEX_UPDATED,
    SUBENTRY_TYPE_MODEL_PROVIDER,
    TOOL_CALL_ERROR_SYSTEM_MESSAGE,
)
from .core.conversation_helpers import (
    _convert_schema_json_to_yaml,
    _fix_entity_ids_in_text,
    _is_dashboard_request,
    _maybe_fix_dashboard_entities,
)
from .core.utils import gather_store_puts_in_chunks

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable, Callable, Mapping

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddEntitiesCallback
    from homeassistant.util.json import JsonObjectType
    from langchain_core.runnables import RunnableConfig

    from .agent.graph import State
    from .core.runtime import HGAConfigEntry, HGAData

_LOGGER = logging.getLogger(__name__)


if LANGCHAIN_LOGGING_LEVEL == "verbose":
    set_verbose(True)
    set_debug(False)
elif LANGCHAIN_LOGGING_LEVEL == "debug":
    set_verbose(False)
    set_debug(True)
else:
    set_verbose(False)
    set_debug(False)


def _convert_content(
    content: conversation.UserContent | conversation.AssistantContent,
) -> HumanMessage | AIMessage:
    """
    Convert HA native chat messages to LangChain messages.

    Only called with UserContent or tool-call-free AssistantContent (filtered
    upstream — see message_history comprehension in _async_handle_message).
    """
    if isinstance(content, conversation.UserContent):
        return HumanMessage(content=content.content or "")
    return AIMessage(content=content.content or "")


def _normalize_ai_content(content: str | list) -> str | None:
    """
    Normalize AIMessage.content to str for HA AssistantContent.

    AIMessage.content is str for most providers, or list[dict] (content blocks) for
    providers that return structured output (e.g. Anthropic). Extracts text blocks
    only; returns None if the result is empty.
    """
    if isinstance(content, str):
        return content or None
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(parts) or None


def _normalize_tool_result(content: Any) -> JsonObjectType:
    """
    Normalize ToolMessage.content to JsonObjectType for HA ToolResultContent.

    ToolMessage.content is str for plain string results (HA tools serialize their
    response via json.dumps, so the content is a JSON-encoded string), or list[dict]
    (content blocks) from some providers (e.g. Anthropic).

    HA intent tools (e.g. HassTurnOn) return a JSON string whose decoded form
    contains a ``response_type`` key (e.g. ``"action_done"``).  If the raw dict
    is stored in ToolResultContent the HA frontend mistakes it for the final
    conversation response and renders an empty speech bubble.  We detect that
    shape and re-encode it so only the meaningful payload is kept.
    """
    if isinstance(content, str):
        # HA tools produce json.dumps(response) as content, so try to deserialize
        # back to a dict to avoid double-escaping in Show Details.
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return _sanitize_tool_result_dict(parsed)
        except (json.JSONDecodeError, ValueError):
            pass
        return {"result": content}
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        return {"result": "\n".join(parts)}
    return {"result": str(content)}


def _sanitize_tool_result_dict(d: dict[str, Any]) -> JsonObjectType:
    """
    Reformat HA intent-response dicts to avoid frontend misinterpretation.

    HA intent tools (e.g. HassTurnOn) return a dict with ``response_type`` and
    ``speech`` keys that mirror the top-level ConversationResult shape.  When
    stored verbatim in ToolResultContent the HA frontend mistakes it for the
    final conversation response and renders an empty speech bubble.  We flatten
    the meaningful payload instead.
    """
    if "response_type" not in d:
        return d

    out: dict[str, Any] = {"result": d["response_type"]}

    # Lift the plain-speech text when present (non-empty string).
    speech = d.get("speech")
    if isinstance(speech, dict):
        plain = speech.get("plain")
        if isinstance(plain, dict):
            text = plain.get("speech")
            if text:
                out["result"] = text

    # Preserve the data payload (success/failed entity lists, etc.).
    data = d.get("data")
    if isinstance(data, dict):
        out.update(data)

    return out


def _populate_chat_log_from_response(
    chat_log: conversation.ChatLog,
    agent_id: str,
    new_messages: list[AnyMessage],
) -> None:
    """
    Backfill chat_log with LangGraph response messages for HA Show Details.

    Walks new_messages (messages produced this turn, sliced from
    response["messages"][len(app_input["messages"]):]) and appends AssistantContent
    and ToolResultContent entries so HA's "Show Details" panel renders the full tool
    call / result chain.

    All ToolInput entries use external=True because LangGraph, not HA, executed the
    tools. async_add_assistant_content_without_tools() raises ValueError if any
    ToolInput has external=False.

    chat_log state is not persisted across HA restarts. The LangGraph PostgreSQL
    checkpointer holds the full conversation history; chat_log is repopulated each
    turn from the ainvoke() response slice.
    """
    for msg in new_messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                tool_inputs = [
                    llm.ToolInput(
                        tool_name=tc["name"],
                        tool_args=tc["args"],
                        id=tc["id"] or ulid.ulid_now(),
                        external=True,
                    )
                    for tc in msg.tool_calls
                ]
                chat_log.async_add_assistant_content_without_tools(
                    conversation.AssistantContent(
                        agent_id=agent_id,
                        content=_normalize_ai_content(msg.content),
                        tool_calls=tool_inputs,
                    )
                )
            else:
                chat_log.async_add_assistant_content_without_tools(
                    conversation.AssistantContent(
                        agent_id=agent_id,
                        content=_normalize_ai_content(msg.content),
                    )
                )
        elif isinstance(msg, ToolMessage):
            chat_log.async_add_assistant_content_without_tools(
                conversation.ToolResultContent(
                    agent_id=agent_id,
                    tool_call_id=msg.tool_call_id,
                    tool_name=msg.name or "",
                    tool_result=_normalize_tool_result(msg.content),
                )
            )


# ruff: noqa: PLR0912
async def _stream_langgraph_to_ha(
    event_stream: AsyncIterable[Mapping[str, Any]],
    _agent_id: str,
) -> AsyncGenerator[AssistantContentDeltaDict | ToolResultContentDeltaDict]:
    """
    Transform LangGraph astream_events into HA ChatLog deltas.

    Filters for events from the 'agent' node for tokens/tool-calls and the
    'action' node for tool results.
    """
    pending_tool_map: dict[str, ToolCall] = {}

    try:
        async for event in event_stream:
            ev = cast("dict[str, Any]", event)
            metadata = cast("dict[str, Any]", ev.get("metadata", {}))
            node = metadata.get("langgraph_node")
            event_type = ev.get("event")
            data = cast("dict[str, Any]", ev.get("data", {}))

            # 1. Start of a model turn: open an AssistantContent block.
            # HA flushes any previous AssistantContent when it receives a
            # role="assistant" delta. We yield this at every agent start to ensure
            # HA state sync in recursive loops (Agent -> Action -> Agent).
            if node == "agent" and event_type == "on_chat_model_start":
                yield AssistantContentDeltaDict(role="assistant")

            # 2. Incremental text tokens.
            elif node == "agent" and event_type == "on_chat_model_stream":
                chunk = data.get("chunk")
                if not isinstance(chunk, AIMessageChunk) or chunk.tool_call_chunks:
                    continue

                chunk_text = _normalize_ai_content(chunk.content)
                if chunk_text:
                    yield AssistantContentDeltaDict(content=chunk_text)

            # 3. End of a model turn: record pending tool calls.
            elif node == "agent" and event_type == "on_chat_model_end":
                msg = data.get("output")
                if not isinstance(msg, AIMessage):
                    continue

                if msg.tool_calls:
                    # Update mapping for result correlation.
                    call_id_map = []
                    for tc in msg.tool_calls:
                        call_id = tc.get("id") or ulid.ulid_now()
                        pending_tool_map[call_id] = tc
                        call_id_map.append(call_id)

                    yield AssistantContentDeltaDict(
                        tool_calls=[
                            llm.ToolInput(
                                tool_name=str(tc["name"]),
                                tool_args=cast("dict[str, Any]", tc["args"]),
                                id=call_id_map[i],
                                external=True,
                            )
                            for i, tc in enumerate(msg.tool_calls)
                        ]
                    )

            # 4. Tool results from the action node.
            elif node == "action" and event_type == "on_chain_end":
                output = data.get("output")
                # LangGraph emits two on_chain_end events for "action":
                # (a) node function event: output = {"messages": [ToolMessage, ...]}
                # (b) graph-level state event: output = full updated messages list
                # We only want (a). Skip anything that is not a dict.
                if not isinstance(output, dict):
                    continue
                messages = cast("list[Any]", output.get("messages", []))
                tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

                # 1. Yield successful results and remove from pending map.
                for msg in tool_messages:
                    pending_tool_map.pop(msg.tool_call_id, None)

                    yield ToolResultContentDeltaDict(
                        role="tool_result",
                        tool_call_id=str(msg.tool_call_id),
                        tool_name=str(msg.name or ""),
                        tool_result=_normalize_tool_result(msg.content),
                    )

                # 2. Yield synthetic rejections for any calls that didn't return a
                # result. This ensures HA doesn't hang if a tool fails silently or
                # is rejected.
                for call_id, tc in pending_tool_map.items():
                    yield ToolResultContentDeltaDict(
                        role="tool_result",
                        tool_call_id=call_id,
                        tool_name=tc.get("name") or "tool",
                        tool_result={
                            "error": "Tool execution rejected by routing policy."
                        },
                    )
                pending_tool_map.clear()
    except (GeneratorExit, asyncio.CancelledError):
        # Stop iteration on cancellation or closure.
        raise
    except Exception:
        _LOGGER.exception("Error in LangGraph streaming transformation generator.")
        raise
    finally:
        pending_tool_map.clear()


class MultiLLMAPI:
    """Wrapper to route tool calls across multiple LLM APIs."""

    def __init__(
        self, apis: dict[str, llm.APIInstance], routing_map: dict[str, str]
    ) -> None:
        """Initialize the wrapper."""
        self.apis = apis
        self.routing_map = routing_map

    @property
    def api_prompt(self) -> str:
        """Return the concatenated API prompts."""
        return "\n".join(api.api_prompt for api in self.apis.values() if api.api_prompt)

    @property
    def custom_serializer(self) -> Callable[[Any], Any] | None:
        """Return the custom serializer from the first available API."""
        for api in self.apis.values():
            if api.custom_serializer:
                return api.custom_serializer
        return None

    async def async_call_tool(self, tool_input: llm.ToolInput) -> Any:
        """Route the tool call to the specific provider using the routing map."""
        api_id = self.routing_map.get(tool_input.tool_name)
        if not api_id:
            # Fallback for unexpected calls
            for api in self.apis.values():
                try:
                    return await api.async_call_tool(tool_input)
                except (HomeAssistantError, vol.Invalid):
                    continue
            msg = f"No routing target for {tool_input.tool_name}"
            raise HomeAssistantError(msg)

        api = self.apis.get(api_id)
        if not api:
            msg = f"API {api_id} not available for {tool_input.tool_name}"
            raise HomeAssistantError(msg)

        return await api.async_call_tool(tool_input)


async def _run_tool_index_background(
    *,
    index_tasks: list[Any],
    tool_hashes: dict[str, str],
    rd: HGAData,
    hass: HomeAssistant,
) -> None:
    """Batch tool indexing into the store and update hashes on success."""
    try:
        if index_tasks:
            await gather_store_puts_in_chunks(index_tasks)
        rd.tool_content_hashes.update(tool_hashes)
        rd.tool_index_ready = True
        _LOGGER.info(
            "Tool index ready: %d tool(s) indexed/updated.",
            len(tool_hashes),
        )
        async_dispatcher_send(
            hass, SIGNAL_TOOL_INDEX_UPDATED, "ready", len(tool_hashes)
        )
    except Exception:
        _LOGGER.exception("Global tool index background task failed")
        rd.tool_index_failed = True
        async_dispatcher_send(hass, SIGNAL_TOOL_INDEX_UPDATED, "failed", 0)
    finally:
        rd.tool_indexing_in_progress = False


async def async_setup_entry(
    hass: HomeAssistant,  # noqa: ARG001
    config_entry: HGAConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = HGAConversationEntity(config_entry)
    async_add_entities([agent])


class HGAConversationEntity(conversation.ConversationEntity, AbstractConversationAgent):
    """Home Generative Assistant conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Lindo St. Angel",
            model="Home Generative Agent",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        self.message_history_len = 0

        if self.entry.runtime_data.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

        # Use in-memory caching for langgraph calls to LLMs.
        set_llm_cache(InMemoryCache())

        self.tz = dt_util.get_default_time_zone()

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)
        # Kick off tool indexing at startup so the index is ready before the first
        # user query. Construct a minimal LLMContext — no live ConversationInput needed
        # because async_get_apis / tool discovery only uses platform/assistant.
        startup_llm_context = llm.LLMContext(
            platform=DOMAIN,
            context=None,
            language=self.hass.config.language,
            assistant=conversation.DOMAIN,
            device_id=None,
        )
        await self._async_index_tools(startup_llm_context, self.entry.runtime_data)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    def _async_get_message_history(
        self, chat_log: conversation.ChatLog
    ) -> list[HumanMessage | AIMessage]:
        """Extract and slice the relevant chat log history."""
        # Include only HA User/Assistant messages not already seen by this entity.
        # Exclude AssistantContent with tool_calls: those are intermediate LangGraph
        # turns already persisted in the PostgreSQL checkpointer. Including them would
        # duplicate messages that LangGraph already manages internally.
        message_history = [
            _convert_content(m)
            for m in chat_log.content
            if isinstance(m, conversation.UserContent)
            or (isinstance(m, conversation.AssistantContent) and m.tool_calls is None)
        ]
        # The last chat log entry will be the current user request—add it later.
        message_history = message_history[:-1]

        mhlen = len(message_history)
        if mhlen <= self.message_history_len:
            message_history = []
        else:
            diff = mhlen - self.message_history_len
            message_history = message_history[-diff:]
            self.message_history_len = mhlen

        return message_history

    async def _async_init_llm_apis(self, llm_context: llm.LLMContext) -> MultiLLMAPI:
        """Load and validate configured LLM APIs, returning a MultiLLMAPI instance."""
        hass = self.hass
        options = self.entry.runtime_data.options

        active_api_ids = options.get(CONF_LLM_HASS_API, [llm.LLM_API_ASSIST])
        if isinstance(active_api_ids, str):
            active_api_ids = [active_api_ids]

        active_apis: dict[str, llm.APIInstance] = {}
        failed_apis: list[str] = []
        for api_id in active_api_ids:
            try:
                api = await llm.async_get_api(hass, api_id, llm_context)
                active_apis[api_id] = api
            except HomeAssistantError:
                _LOGGER.warning("Could not load LLM API: %s", api_id)
                failed_apis.append(api_id)

        if not active_apis:
            msg = (
                "No LLM APIs could be loaded. "
                f"Failed: {', '.join(failed_apis)}. "
                f"Configured: {', '.join(active_api_ids)}"
            )
            raise HomeAssistantError(msg)

        return MultiLLMAPI(active_apis, {})

    def _async_get_all_tools(
        self, active_apis: dict[str, llm.APIInstance]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Aggregate HA tools and LangChain-native tools."""
        options = self.entry.runtime_data.options

        tools = (
            [
                format_tool(tool, api.custom_serializer)
                for api in active_apis.values()
                for tool in api.tools
            ]
            if active_apis
            else []
        )

        # Add LangChain-native tools (wired in graph via config).
        langchain_tools: dict[str, Any] = {
            "get_and_analyze_camera_image": get_and_analyze_camera_image,
            "get_camera_last_events": get_camera_last_events,
            "upsert_memory": upsert_memory,
            "get_entity_history": get_entity_history,
            "confirm_sensitive_action": confirm_sensitive_action,
            "alarm_control": alarm_control,
            "resolve_entity_ids": resolve_entity_ids,
            "write_yaml_file": write_yaml_file,
        }
        if not options.get(CONF_SCHEMA_FIRST_YAML, False):
            langchain_tools["add_automation"] = add_automation
        tools.extend(langchain_tools.values())

        return tools, langchain_tools

    def _async_render_system_prompt(
        self,
        llm_context: llm.LLMContext,
        user_name: str | None,
        llm_api: MultiLLMAPI,
        *,
        has_tools: bool,
    ) -> str:
        """Render the full system instructions."""
        options = self.entry.runtime_data.options

        pin_enabled = options.get(CONF_CRITICAL_ACTION_PIN_ENABLED, False)
        critical_prompt = CRITICAL_ACTION_PROMPT if pin_enabled else ""
        schema_prompt = (
            SCHEMA_FIRST_YAML_PROMPT
            if options.get(CONF_SCHEMA_FIRST_YAML, False)
            else ""
        )
        prompt_parts = [
            template.Template(
                (
                    llm.DATE_TIME_PROMPT
                    + options.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT)
                    + f"\nYou are in the {self.tz} timezone."
                    + critical_prompt
                    + schema_prompt
                    + TOOL_CALL_ERROR_SYSTEM_MESSAGE
                    if has_tools
                    else ""
                ),
                self.hass,
            ).async_render(
                {
                    "ha_name": self.hass.config.location_name,
                    "user_name": user_name,
                    "llm_context": llm_context,
                },
                parse_result=False,
            )
        ]

        if llm_api:
            prompt_parts.append(llm_api.api_prompt)

        return "\n".join(prompt_parts)

    async def _async_run_ainvoke(
        self,
        app: Any,
        input_data: State,
        config: RunnableConfig,
        chat_log: conversation.ChatLog,
        tools: list[dict[str, Any]] | None,
    ) -> None:
        """Handle the non-streaming ainvoke path."""
        options = self.entry.runtime_data.options
        hass = self.hass

        try:
            response = await app.ainvoke(input=input_data, config=config)
        except HomeAssistantError as err:
            _LOGGER.exception("LangGraph error during conversation processing.")
            msg = f"Something went wrong: {err}"
            raise HomeAssistantError(msg) from err

        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL,
            {"messages": response["messages"], "tools": tools or None},
        )

        _LOGGER.debug("====== End of run (ainvoke) ======")

        # Post-process the final LLM text (entity ID resolution, YAML conversion).
        final_content = response["messages"][-1].content
        if isinstance(final_content, str):
            if options.get(CONF_SCHEMA_FIRST_YAML, False):
                final_content = _maybe_fix_dashboard_entities(final_content, hass)
                final_content = _convert_schema_json_to_yaml(
                    final_content, enabled=True
                )
            else:
                final_content = _fix_entity_ids_in_text(final_content, hass)
            _LOGGER.debug("Final response content: %s", final_content)

        # Backfill chat_log with the messages produced this turn so HA's
        # "Show Details" panel renders the full tool call / result chain.
        # Slice off history + current HumanMessage — only net-new messages.
        new_messages = response["messages"][len(input_data["messages"]) :]

        # Guard: async_get_result_from_chat_log requires last entry to be
        # AssistantContent, which means the last new message must be an
        # AIMessage.
        if not new_messages or not isinstance(new_messages[-1], AIMessage):
            _LOGGER.error(
                "LangGraph response did not end with AIMessage (got %s). "
                "Falling back to manual ConversationResult.",
                type(new_messages[-1]) if new_messages else "empty",
            )
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(
                    agent_id=self.entity_id,
                    content=final_content if isinstance(final_content, str) else "",
                )
            )
            return

        # Replace the final AIMessage content with the post-processed text so
        # Show Details reflects the same text the user hears.
        if isinstance(final_content, str):
            processed_messages: list[AnyMessage] = [
                *new_messages[:-1],
                AIMessage(
                    content=final_content,
                    tool_calls=new_messages[-1].tool_calls,
                ),
            ]
        else:
            processed_messages = list(new_messages)

        _populate_chat_log_from_response(chat_log, self.entity_id, processed_messages)

    async def _async_run_astream(
        self,
        app: Any,
        input_data: State,
        config: RunnableConfig,
        chat_log: conversation.ChatLog,
        tools: list[dict[str, Any]] | None,
    ) -> None:
        """Handle the streaming astream_events path."""
        hass = self.hass

        async def _run_streaming_task() -> None:
            """Coroutine to drive the delta stream and ensure cancellation."""
            event_stream = app.astream_events(
                input=input_data, config=config, version="v2"
            )
            async for _ in chat_log.async_add_delta_content_stream(
                self.entity_id,
                _stream_langgraph_to_ha(event_stream, self.entity_id),
            ):
                pass

        # Create a task for the streaming process to enable explicit cancellation
        # if the client disconnects or the conversation turn is aborted.
        stream_task = hass.async_create_task(_run_streaming_task())
        try:
            await stream_task
        except HomeAssistantError:
            # Partial content already committed — cannot roll back.
            # Log and return whatever was committed.
            _LOGGER.exception(
                "HomeAssistantError mid-stream; partial content committed."
            )
        except asyncio.CancelledError:
            # Hard kill: cancel the background graph run if the consumer has left.
            stream_task.cancel()
            raise
        except Exception:
            # Any other exception (e.g. GraphRecursionError, tool failure) leaves
            # the chat_log in a partial state. Log and fall through to recovery.
            _LOGGER.exception(
                "Unexpected error in streaming task; attempting state recovery."
            )
        finally:
            if not stream_task.done():
                stream_task.cancel()
                # Non-blocking wait to allow generators to close.
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await stream_task

        # Fire trace after stream completes using final graph state.
        # astream_events handles exhaustion only after graph termination.
        _LOGGER.debug("====== End of run (streaming) ======")

        # Re-fire CONTENT_ADDED for the final AssistantContent so the HA
        # frontend's streaming UI shows it in the main chat area.
        # async_add_delta_content_stream flushes via async_add_assistant_content,
        # which does NOT fire CONTENT_ADDED for AssistantContent. Re-committing
        # via async_add_assistant_content_without_tools does fire it.
        if (
            chat_log.content
            and isinstance(chat_log.content[-1], conversation.AssistantContent)
            and not chat_log.content[-1].tool_calls
            and chat_log.content[-1].content
        ):
            final_content = chat_log.content.pop()
            chat_log.async_add_assistant_content_without_tools(final_content)

        try:
            final_state = await app.aget_state(config)
            trace.async_conversation_trace_append(
                trace.ConversationTraceEventType.AGENT_DETAIL,
                {
                    "messages": final_state.values.get("messages", []),
                    "tools": tools or None,
                },
            )

            # If streaming ended without committing a final AssistantContent
            # (e.g. the generator raised before the last LLM turn), recover the
            # final AIMessage from the graph state so the caller can return it.
            if not isinstance(
                chat_log.content[-1] if chat_log.content else None,
                conversation.AssistantContent,
            ):
                messages = final_state.values.get("messages", [])
                if messages and isinstance(messages[-1], AIMessage):
                    final_msg = messages[-1]
                    chat_log.async_add_assistant_content_without_tools(
                        conversation.AssistantContent(
                            agent_id=self.entity_id,
                            content=_normalize_ai_content(final_msg.content),
                        )
                    )
                    _LOGGER.debug(
                        "Recovered final AssistantContent from graph state after "
                        "streaming failure."
                    )
        except ValueError:
            # Expected when no checkpointer is configured (e.g. tests)
            _LOGGER.debug("aget_state unavailable; skipping trace for streaming turn.")
        except Exception:
            _LOGGER.exception("Failed to retrieve final state for tracing.")

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process the user input."""
        hass = self.hass
        options = self.entry.runtime_data.options
        runtime_data = self.entry.runtime_data
        intent_response = intent.IntentResponse(language=user_input.language)

        llm_context = llm.LLMContext(
            platform=DOMAIN,
            context=user_input.context,
            language=user_input.language,
            assistant=conversation.DOMAIN,
            device_id=user_input.device_id,
        )

        message_history = self._async_get_message_history(chat_log)

        conversation_id = (
            ulid.ulid_now()
            if chat_log.conversation_id is None
            else chat_log.conversation_id
        )

        # Multi-API initialization
        try:
            llm_api = await self._async_init_llm_apis(llm_context)
        except HomeAssistantError as err:
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                str(err),
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        # --- Global Tool Indexing (Background) ---
        await self._async_index_tools(llm_context, runtime_data)

        if not options.get(CONF_SCHEMA_FIRST_YAML, False) and _is_dashboard_request(
            user_input.text
        ):
            intent_response.async_set_speech(
                "Please enable 'Schema-first JSON for YAML requests' in "
                "HGA's configuration and try again"
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        tools, langchain_tools = self._async_get_all_tools(llm_api.apis)

        # Resolve user name (None means automation)
        user_name = None
        if (
            user_input.context
            and user_input.context.user_id
            and (user := await hass.auth.async_get_user(user_input.context.user_id))
        ):
            user_name = user.name

        prompt = self._async_render_system_prompt(
            llm_context, user_name, llm_api, has_tools=bool(tools)
        )

        # Use the already-configured chat model from __init__.py.
        base_llm = runtime_data.chat_model
        if not hasattr(base_llm, "bind_tools"):
            _LOGGER.exception("Error during conversation processing.")
            intent_response = intent.IntentResponse(language=user_input.language)
            has_provider = any(
                subentry.subentry_type == SUBENTRY_TYPE_MODEL_PROVIDER
                for subentry in self.entry.subentries.values()
            )
            if not has_provider:
                msg = (
                    "This integration isn't configured with a model provider. "
                    "Go to Settings -> Devices & Services -> Home Generative Agent -> "
                    "Add Model Provider."
                )
            else:
                msg = (
                    "This model doesn't support tool calling. "
                    "Choose a compatible model or provider."
                )
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                msg,
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        # A user name of None indicates an automation is being run.
        clean_user_name = "robot" if user_name is None else user_name
        # Remove special characters since memory namespace labels cannot contain them.
        clean_user_name = clean_user_name.translate(
            str.maketrans("", "", string.punctuation)
        )

        app_config: RunnableConfig = {
            "configurable": {
                "thread_id": conversation_id,
                "user_id": clean_user_name,
                "chat_model": base_llm,
                "chat_model_options": runtime_data.chat_model_options,
                "prompt": prompt,
                "options": options,
                "vlm_model": runtime_data.vision_model,
                "summarization_model": runtime_data.summarization_model,
                "langchain_tools": langchain_tools,
                "ha_llm_api": llm_api or None,
                "hass": hass,
                "pending_actions": runtime_data.pending_actions,
                "tool_index_ready": runtime_data.tool_index_ready,
            },
            "recursion_limit": 10,
        }

        # Compile graph into a LangChain Runnable.
        app = workflow.compile(
            store=self.entry.runtime_data.store,
            checkpointer=self.entry.runtime_data.checkpointer,
            debug=LANGCHAIN_LOGGING_LEVEL == "debug",
        )

        # Agent input: message history + current user request.
        messages = [*message_history, HumanMessage(content=user_input.text)]
        app_input: State = {
            "messages": messages,
            "summary": "",
            "chat_model_usage_metadata": {},
            "messages_to_remove": [],
            "selected_tools": [],
            "tool_routing_map": {},
        }

        # Interact with agent app.
        if options.get(CONF_SCHEMA_FIRST_YAML, False):
            try:
                await self._async_run_ainvoke(
                    app, app_input, app_config, chat_log, tools
                )
            except HomeAssistantError as err:
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    str(err),
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )
        else:
            await self._async_run_astream(app, app_input, app_config, chat_log, tools)

        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def _async_discover_provider_tools(
        self,
        llm_context: llm.LLMContext,
        runtime_data: HGAData,
        all_available_api_ids: list[str],
        index_tasks: list[Any],
        new_hashes: dict[str, str],
    ) -> None:
        """Discover and prepare provider tools for indexing."""
        for api_id in all_available_api_ids:
            try:
                # Isolate each provider discovery
                api_instance = await llm.async_get_api(self.hass, api_id, llm_context)
                for tool in api_instance.tools:
                    # Composite hash: api_id + name + description + schema
                    schema_json = json.dumps(
                        safe_convert(
                            tool.parameters,
                            custom_serializer=api_instance.custom_serializer,
                        ),
                        sort_keys=True,
                    )
                    is_actuation = is_actuation_tool(tool.name)
                    raw_content = (
                        f"provider:{api_id}\nname:{tool.name}\n"
                        f"description:{tool.description}\nparameters:{schema_json}\n"
                        f"actuation:{is_actuation}"
                    )
                    content_hash = hashlib.sha256(raw_content.encode()).hexdigest()

                    tool_key = f"{api_id}::{tool.name}"
                    if runtime_data.tool_content_hashes.get(tool_key) != content_hash:
                        # Prep for embedding
                        embedding_text = strip_for_embedding(
                            f"{tool.name}: {tool.description}"
                        )
                        index_tasks.append(
                            runtime_data.store.aput(
                                ("system", "tools"),
                                key=tool_key,
                                value={
                                    "content": truncate_for_embedding_index(
                                        embedding_text
                                    ),
                                    "name": tool.name,
                                    "api_id": api_id,
                                    "description": tool.description,
                                    "parameters": schema_json,
                                    "is_actuation": is_actuation,
                                },
                            )
                        )
                        new_hashes[tool_key] = content_hash
            except Exception as err:  # noqa: BLE001
                _LOGGER.warning("Failed to index tool provider %s: %s", api_id, err)

    async def _async_discover_local_tools(
        self,
        runtime_data: HGAData,
        index_tasks: list[Any],
        new_hashes: dict[str, str],
    ) -> None:
        """Discover and prepare local tools for indexing."""
        local_tools = {
            "get_and_analyze_camera_image": get_and_analyze_camera_image,
            "get_camera_last_events": get_camera_last_events,
            "upsert_memory": upsert_memory,
            "get_entity_history": get_entity_history,
            "confirm_sensitive_action": confirm_sensitive_action,
            "alarm_control": alarm_control,
            "resolve_entity_ids": resolve_entity_ids,
            "write_yaml_file": write_yaml_file,
        }
        # Mirror the dispatch-time guard: add_automation is excluded when
        # schema_first_yaml=True so the index and langchain_tools stay in sync.
        if not self.entry.options.get(CONF_SCHEMA_FIRST_YAML, False):
            local_tools["add_automation"] = add_automation
        for t_name, t_func in local_tools.items():
            try:
                # Extract the JSON schema for local tools
                params = "{}"
                args_schema = getattr(t_func, "args_schema", None)
                if args_schema is not None:
                    try:
                        # Use getattr to avoid pyright errors
                        # on potential dict types
                        schema_func = getattr(args_schema, "schema", None)
                        if callable(schema_func):
                            params = json.dumps(schema_func(), sort_keys=True)
                    except (
                        AttributeError,
                        TypeError,
                        ValueError,
                        PydanticInvalidForJsonSchema,
                    ):
                        params = "{}"

                is_actuation = is_actuation_tool(t_name)
                # LangChain tools have .name, .description,
                # and .args_schema (or similar)
                # We'll use a simplified schema extraction for indexing
                raw_content = (
                    f"provider:hga_local\nname:{t_name}\n"
                    f"description:{t_func.description}\nparameters:{params}\n"
                    f"actuation:{is_actuation}"
                )
                content_hash = hashlib.sha256(raw_content.encode()).hexdigest()
                tool_key = f"hga_local::{t_name}"
                if runtime_data.tool_content_hashes.get(tool_key) != content_hash:
                    embedding_text = strip_for_embedding(
                        f"{t_name}: {t_func.description}"
                    )

                    index_tasks.append(
                        runtime_data.store.aput(
                            ("system", "tools"),
                            key=tool_key,
                            value={
                                "content": truncate_for_embedding_index(embedding_text),
                                "name": t_name,
                                "api_id": "hga_local",
                                "description": t_func.description,
                                "parameters": params,
                                "is_actuation": is_actuation,
                            },
                        )
                    )
                    new_hashes[tool_key] = content_hash
            except Exception as err:  # noqa: BLE001
                _LOGGER.warning("Failed to index local tool %s: %s", t_name, err)

    async def _async_index_tools(
        self, llm_context: llm.LLMContext, runtime_data: HGAData
    ) -> None:
        """
        Discover and index tools in the background vector store.

        Called at startup (async_added_to_hass) so the index is ready before the
        first user query. Also called per-turn as a no-op guard once ready.
        """
        if (
            runtime_data.tool_index_ready
            or runtime_data.tool_indexing_in_progress
            or runtime_data.tool_index_failed
        ):
            return
        runtime_data.tool_indexing_in_progress = True
        async_dispatcher_send(self.hass, SIGNAL_TOOL_INDEX_UPDATED, "indexing", 0)

        all_available_api_ids = [api.id for api in llm.async_get_apis(self.hass)]
        index_tasks: list[Any] = []
        new_hashes: dict[str, str] = {}

        # 1. Index Providers
        await self._async_discover_provider_tools(
            llm_context, runtime_data, all_available_api_ids, index_tasks, new_hashes
        )

        # 2. Index Local LangChain Tools (provider: hga_local)
        await self._async_discover_local_tools(runtime_data, index_tasks, new_hashes)

        if index_tasks:
            self.hass.async_create_task(
                _run_tool_index_background(
                    index_tasks=index_tasks,
                    tool_hashes=new_hashes,
                    rd=runtime_data,
                    hass=self.hass,
                )
            )
        else:
            runtime_data.tool_index_ready = True
            async_dispatcher_send(
                self.hass, SIGNAL_TOOL_INDEX_UPDATED, "ready", len(new_hashes)
            )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features.
        await hass.config_entries.async_reload(entry.entry_id)
