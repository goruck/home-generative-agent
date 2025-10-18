"""Langgraph graphs for Home Generative Agent."""

from __future__ import annotations

import copy
import json
import logging
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import voluptuous as vol
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import trim_messages
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import ValidationError

from ..const import (  # noqa: TID252
    CONF_CHAT_MODEL_PROVIDER,
    CONF_GEMINI_CHAT_MODEL,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OPENAI_CHAT_MODEL,
    CONTEXT_MANAGE_USE_TOKENS,
    CONTEXT_MAX_MESSAGES,
    CONTEXT_MAX_TOKENS,
    EMBEDDING_MODEL_PROMPT_TEMPLATE,
    REASONING_DELIMITERS,
    SUMMARIZATION_INITIAL_PROMPT,
    SUMMARIZATION_PROMPT_TEMPLATE,
    SUMMARIZATION_SYSTEM_PROMPT,
    TOOL_CALL_ERROR_TEMPLATE,
)
from .token_counter import count_tokens_cross_provider

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from langchain_core.runnables import RunnableConfig
    from langgraph.store.base import BaseStore

LOGGER = logging.getLogger(__name__)


class State(MessagesState):
    """Extend MessagesState."""

    summary: str
    chat_model_usage_metadata: dict[str, Any]
    messages_to_remove: list[AnyMessage]


# ----- Utilities -----


async def _retrieve_camera_activity(
    hass: HomeAssistant, store: BaseStore
) -> list[dict[str, dict[str, str]]]:
    """Retrieve most recent camera activity from video analysis by the VLM."""
    camera_activity: list[dict[str, dict[str, str]]] = []
    for entity_id in hass.states.async_entity_ids():
        if not entity_id.startswith("camera."):
            continue
        camera = entity_id.split(".")[-1]
        results = await store.asearch(("video_analysis", camera), limit=1)
        if results and (la := results[0].value.get("content")):
            camera_activity.append(
                {
                    camera: {
                        "last activity": la,
                        "date_time": results[0].updated_at.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }
                }
            )
    if camera_activity:
        LOGGER.debug("Recent camera activity: %s", camera_activity)
        return camera_activity
    LOGGER.debug("No recent camera activity found.")
    return []


def _determine_model_name(provider: str, opts: dict[str, Any]) -> str:
    """Determine model name based on provider and options."""
    if provider == "openai":
        return opts.get(CONF_OPENAI_CHAT_MODEL, "")
    if provider == "gemini":
        return opts.get(CONF_GEMINI_CHAT_MODEL, "")
    return opts.get(CONF_OLLAMA_CHAT_MODEL, "")


# ----- Graph nodes and edges -----


async def _call_model(
    state: State, config: RunnableConfig, *, store: BaseStore
) -> dict[str, Any]:
    """Coroutine to call the chat model."""
    if "configurable" not in config:
        msg = "Configuration for the model is missing."
        raise HomeAssistantError(msg)

    model = config["configurable"]["chat_model"]
    user_id = config["configurable"]["user_id"]
    hass = config["configurable"]["hass"]
    opts = config["configurable"]["options"]
    chat_model_options = config["configurable"].get("chat_model_options", {})

    # Retrieve memories (semantic if last message is from user).
    last_message = state["messages"][-1]
    last_message_from_user = isinstance(last_message, HumanMessage)
    query_prompt = (
        EMBEDDING_MODEL_PROMPT_TEMPLATE.format(query=last_message.content)
        if last_message_from_user
        else None
    )
    mems = await store.asearch((user_id, "memories"), query=query_prompt, limit=10)

    # Recent camera activity.
    camera_activity = await _retrieve_camera_activity(hass, store)

    # Build system message.
    system_message = config["configurable"]["prompt"]
    if mems:
        formatted_mems = "\n".join(f"[{mem.key}]: {mem.value}" for mem in mems)
        system_message += f"\n<memories>\n{formatted_mems}\n</memories>"
    if camera_activity:
        ca = "\n".join(str(a) for a in camera_activity)
        system_message += f"\n<recent_camera_activity>\n{ca}\n</recent_camera_activity>"
    if summary := state.get("summary", ""):
        system_message += (
            f"\n<past_conversation_summary>\n{summary}\n</past_conversation_summary>"
        )

    # Model input = System + current messages.
    messages = [SystemMessage(content=system_message)] + state["messages"]

    # Trim messages to manage context window length.
    # TODO(goruck): Fix token counting.  # noqa: FIX002
    # If using the token counter from the chat model API, the method
    # 'get_num_tokens_from_messages()' will be called which currently ignores
    # tool schemas and under counts message tokens for the qwen models.
    # Until this is fixed, 'max_tokens' should be set to a value less than
    # the maximum size of the model's context window. See const.py.
    # https://github.com/goruck/home-generative-agent/issues/109

    provider = opts.get(CONF_CHAT_MODEL_PROVIDER)
    model_name = _determine_model_name(provider, opts)

    if CONTEXT_MANAGE_USE_TOKENS:
        max_tokens = CONTEXT_MAX_TOKENS
        token_counter = partial(
            count_tokens_cross_provider,
            model=model_name,
            provider=provider,
            options=opts,
            chat_model_options=chat_model_options,
        )
    else:
        max_tokens = CONTEXT_MAX_MESSAGES
        token_counter = len

    trimmed_messages = await hass.async_add_executor_job(
        partial(
            trim_messages,
            messages=messages,
            token_counter=token_counter,
            max_tokens=max_tokens,
            strategy="last",
            start_on="human",
            include_system=True,
        )
    )

    LOGGER.debug("Model call messages: %s", trimmed_messages)
    LOGGER.debug("Model call messages length: %s", len(trimmed_messages))

    raw_response = await model.ainvoke(trimmed_messages)
    LOGGER.debug("Raw chat model response: %s", raw_response)
    # Clean up raw response.
    response: str = raw_response.content
    # If model used reasoning, just use the final result.
    first, sep, last = response.partition(REASONING_DELIMITERS.get("end", ""))
    response = last.strip("\n") if sep else first.strip("\n")
    # Create AI message, no need to include tool call metadata if there's none.
    if hasattr(raw_response, "tool_calls"):
        ai_response = AIMessage(content=response, tool_calls=raw_response.tool_calls)
    else:
        ai_response = AIMessage(content=response)
    LOGGER.debug("AI response: %s", ai_response)

    metadata: dict[str, str] = (
        raw_response.usage_metadata if hasattr(raw_response, "usage_metadata") else {}
    )
    LOGGER.debug("Token counts from metadata: %s", metadata)

    messages_to_remove = [m for m in state["messages"] if m not in trimmed_messages]
    LOGGER.debug("Messages to remove: %s", messages_to_remove)

    return {
        "messages": ai_response,
        "chat_model_usage_metadata": metadata,
        "messages_to_remove": messages_to_remove,
    }


async def _summarize_and_remove_messages(
    state: State, config: RunnableConfig
) -> dict[str, Any]:
    """Summarize trimmed messages and remove them from state."""
    if "configurable" not in config:
        msg = "Configuration is missing."
        raise HomeAssistantError(msg)

    summary = state.get("summary", "")
    msgs_to_remove = state.get("messages_to_remove", [])
    if not msgs_to_remove:
        return {"summary": summary}

    summary_message = (
        SUMMARIZATION_PROMPT_TEMPLATE.format(summary=summary)
        if summary
        else SUMMARIZATION_INITIAL_PROMPT
    )

    # Build messages for the already-configured summarization model.
    messages = (
        [SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT)]
        + [m for m in msgs_to_remove if isinstance(m, (HumanMessage, AIMessage))]
        + [HumanMessage(content=summary_message)]
    )

    model = config["configurable"]["summarization_model"]
    LOGGER.debug("Summary messages: %s", messages)
    response = await model.ainvoke(messages)
    LOGGER.debug("Summary response: %s", response)

    return {
        "summary": getattr(response, "content", response),
        "messages": [
            RemoveMessage(id=m.id) for m in msgs_to_remove if m.id is not None
        ],
    }


async def _call_tools(
    state: State, config: RunnableConfig, *, store: BaseStore
) -> dict[str, list[ToolMessage]]:
    """Call Home Assistant or LangChain tools requested by the model."""
    if "configurable" not in config:
        msg = "Configuration is missing."
        raise HomeAssistantError(msg)

    langchain_tools = config["configurable"]["langchain_tools"]
    ha_llm_api = config["configurable"]["ha_llm_api"]

    # Expect tool calls in the last AIMessage.
    if not state["messages"] or not isinstance(state["messages"][-1], AIMessage):
        msg = "No tool calls found in the last message."
        raise HomeAssistantError(msg)

    tool_calls = state["messages"][-1].tool_calls or []
    tool_responses: list[ToolMessage] = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        LOGGER.debug("Tool call: %s(%s)", tool_name, tool_args)

        def _handle_tool_error(err: str, name: str, tid: str) -> ToolMessage:
            return ToolMessage(
                content=TOOL_CALL_ERROR_TEMPLATE.format(error=err),
                name=name,
                tool_call_id=tid,
                status="error",
            )

        # LangChain tool
        if tool_name in langchain_tools:
            lc_tool = langchain_tools[tool_name.lower()]
            tool_call_copy = copy.deepcopy(tool_call)
            tool_call_copy["args"].update({"store": store, "config": config})
            try:
                tool_response = await lc_tool.ainvoke(tool_call_copy)
            except (HomeAssistantError, ValidationError) as err:
                tool_response = _handle_tool_error(
                    repr(err), tool_name, tool_call.get("id") or ""
                )
        # Home Assistant tool
        else:
            tool_input = llm.ToolInput(tool_name=tool_name, tool_args=tool_args)
            try:
                response = await ha_llm_api.async_call_tool(tool_input)
                tool_response = ToolMessage(
                    content=json.dumps(response),
                    tool_call_id=tool_call.get("id"),
                    name=tool_name,
                )
            except (HomeAssistantError, vol.Invalid) as err:
                tool_response = _handle_tool_error(
                    repr(err), tool_name, tool_call.get("id") or ""
                )

        LOGGER.debug("Tool response: %s", tool_response)
        tool_responses.append(tool_response)

    return {"messages": tool_responses}


def _should_continue(
    state: State,
) -> Literal["action", "summarize_and_remove_messages"]:
    """Return the next node in graph to execute."""
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "action"
    return "summarize_and_remove_messages"


# Define a new graph
workflow = StateGraph(State)

# Define nodes.
workflow.add_node("agent", _call_model)
workflow.add_node("action", _call_tools)
workflow.add_node("summarize_and_remove_messages", _summarize_and_remove_messages)

# Define edges.
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", _should_continue)
workflow.add_edge("action", "agent")
workflow.add_edge("summarize_and_remove_messages", END)
