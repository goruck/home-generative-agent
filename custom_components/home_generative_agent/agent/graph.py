"""Langgraph graphs for Home Generative Agent."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)

import requests
import tiktoken
import voluptuous as vol
from homeassistant.const import CONF_API_KEY
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_openai import ChatOpenAI as LCChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore  # noqa: TC002
from pydantic import ValidationError

from ..const import (  # noqa: TID252
    CONF_CHAT_MODEL_PROVIDER,
    CONF_GEMINI_API_KEY,
    CONF_GEMINI_CHAT_MODEL,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OLLAMA_URL,
    CONF_OPENAI_CHAT_MODEL,
    CONTEXT_MANAGE_USE_TOKENS,
    CONTEXT_MAX_MESSAGES,
    CONTEXT_MAX_TOKENS,
    EMBEDDING_MODEL_PROMPT_TEMPLATE,
    PROVIDERS,
    REASONING_DELIMITERS,
    SUMMARIZATION_INITIAL_PROMPT,
    SUMMARIZATION_PROMPT_TEMPLATE,
    SUMMARIZATION_SYSTEM_PROMPT,
    TOOL_CALL_ERROR_TEMPLATE,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)


class State(MessagesState):
    """Extend MessagesState."""

    summary: str
    chat_model_usage_metadata: dict[str, Any]
    messages_to_remove: list[AnyMessage]


# ----- Token counting -----

MessageLike = Mapping[str, Any] | Any


# Message normalization
def _normalize_message(msg: MessageLike) -> Mapping[str, Any]:
    """Normalize message to dict with 'role' and 'content'."""
    if hasattr(msg, "type") and hasattr(msg, "content"):  # LangChain
        role = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
            "function": "tool",
        }.get(getattr(msg, "type"), getattr(msg, "type"))  # noqa: B009
        return {"role": role, "content": getattr(msg, "content")}  # noqa: B009
    if isinstance(msg, Mapping):  # OpenAI-style
        return {"role": msg.get("role", "user"), "content": msg.get("content", "")}
    msg = f"Unsupported message type: {type(msg)}"
    raise TypeError(msg)


def _flatten_text_content(content: Any) -> str:
    """Flatten message content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                out.append(part["text"])
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out)
    return str(content)


def _concat_messages_for_count(messages: Sequence[MessageLike]) -> str:
    """Concatenate messages to a single string for token counting."""
    out: list[str] = []
    for m in messages:
        mm = _normalize_message(m)
        out.append(
            f"{mm.get('role', 'user')}:\n{_flatten_text_content(mm.get('content', ''))}"
        )
    return "\n\n".join(out)


# OpenAI, uses tiktoken
def _pick_encoding_for_model(model: str) -> tiktoken.Encoding:
    """Return tiktoken encoding for model, with fallback."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def _count_tokens_tiktoken(
    messages: Iterable[MessageLike],
    model: str,
    tools: Sequence[Any] | None = None,  # noqa: ARG001
) -> int:
    """Count tokens in messages for OpenAI models using tiktoken."""
    enc = _pick_encoding_for_model(model)
    total = 0
    for m in messages:
        mm = _normalize_message(m)
        total += len(enc.encode(str(mm.get("role", "user"))))
        total += len(enc.encode("\n"))
        content_text = _flatten_text_content(mm.get("content"))
        if content_text:
            total += len(enc.encode(content_text))
    return total


# Gemini, uses REST :countTokens
def _count_gemini_tokens(
    messages: Sequence[MessageLike],
    model: str,
    gemini_api_key: Any | None = None,
    *,
    timeout: float = 10.0,
    endpoint_base: str = "https://generativelanguage.googleapis.com",
) -> int:
    """Count tokens in messages for Gemini models using REST API."""
    key = None
    if gemini_api_key is not None:
        key = (
            gemini_api_key.get_secret_value()
            if hasattr(gemini_api_key, "get_secret_value")
            else str(gemini_api_key)
        )
    if not key:
        msg = "Gemini API key required for token counting."
        raise RuntimeError(msg)

    combined = _concat_messages_for_count(messages)
    url = f"{endpoint_base.rstrip('/')}/v1beta/models/{model}:countTokens"
    params = {"key": key}
    payload = {"contents": [{"role": "user", "parts": [{"text": combined}]}]}
    r = requests.post(url, params=params, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "totalTokens" in data:
        return int(data["totalTokens"])
    if "total_tokens" in data:
        return int(data["total_tokens"])
    msg = f"Unexpected Gemini countTokens response: {data}"
    raise RuntimeError(msg)


# Ollama, uses /api/tokenize with fallback to /api/generate
def _count_ollama_via_generate(
    messages: Sequence[MessageLike],
    model: str,
    base_url: str,
    timeout: float = 60.0,
) -> int:
    """Count tokens in messages for Ollama models using /api/generate."""
    combined = _concat_messages_for_count(messages)
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": combined,
        "stream": False,  # single JSON with metrics
        "options": {"num_predict": 0},  # evaluate prompt only
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    n = data.get("prompt_eval_count")
    if isinstance(n, int):
        return n
    msg = f"Unexpected /api/generate response (no prompt_eval_count): {data}"
    raise RuntimeError(msg)


def _count_ollama_tokens(
    messages: Sequence[MessageLike],
    model: str,
    base_url: str,
    timeout: float = 60.0,
    *,
    options: dict[str, Any] | None = None,
) -> int:
    """Use /api/generate with num_predict=0 and the SAME options as chat."""
    combined = _concat_messages_for_count(messages)
    url = f"{base_url.rstrip('/')}/api/generate"
    opts = dict(options or {})
    opts["num_predict"] = 0  # evaluate prompt only, but keep other knobs
    payload = {
        "model": model,
        "prompt": combined,
        "stream": False,
        "options": opts,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    n = data.get("prompt_eval_count")
    if isinstance(n, int):
        return n
    msg = f"Unexpected /api/generate response: {data}"
    raise RuntimeError(msg)


# Entrypoint
def count_tokens_cross_provider(  # noqa: PLR0913
    messages: Sequence[MessageLike],
    model: str,
    provider: PROVIDERS,
    *,
    prefer_langchain_when_available: bool = True,
    options: dict[str, Any],
    chat_model_options: dict[str, Any],
) -> int:
    """
    Count tokens in messages for different providers.

    OpenAI:
      - If prefer_langchain_when_available=True and LCChatOpenAI is imported,
        try its get_num_tokens_from_messages(); on NotImplementedError,
        fall back to tiktoken.
    Gemini:
      - Uses REST models/{model}:countTokens (API key required).
    Ollama:
      - Uses /api/generate (reads prompt_eval_count).
    """
    ollama_base_url = options.get(CONF_OLLAMA_URL)
    if ollama_base_url is None:
        msg = "Ollama base URL must be set in options."
        raise ValueError(msg)
    openai_api_key = options.get(CONF_API_KEY)
    gemini_api_key = options.get(CONF_GEMINI_API_KEY)
    if provider == "openai":
        if prefer_langchain_when_available:
            tmp = LCChatOpenAI(model=model, temperature=0, api_key=openai_api_key)
            try:
                return tmp.get_num_tokens_from_messages(
                    cast("list[Any]", list(messages))
                )
            except NotImplementedError:
                # LC doesn't support this model's token counting yet → fallback
                pass
            except HomeAssistantError:
                # Any LC-specific issue → fallback rather than crashing
                pass
        # Fallback path for OpenAI (or when LC is unavailable/unsupported)
        return _count_tokens_tiktoken(messages, model=model)

    if provider == "gemini":
        return _count_gemini_tokens(
            messages, model=model, gemini_api_key=gemini_api_key
        )

    # Ollama
    return _count_ollama_tokens(
        messages, model=model, base_url=ollama_base_url, options=chat_model_options
    )


# ----- Other utilities -----


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
