"""Langgraph graphs for Home Generative Agent."""
from __future__ import annotations  # noqa: I001

import copy
import json
import logging
from functools import partial
from typing import Any, Literal

import voluptuous as vol
import homeassistant.util.dt as dt_util
from homeassistant.exceptions import (
    HomeAssistantError,
)
from homeassistant.helpers import llm
from langchain_core.messages import (
    AnyMessage,
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig  # noqa: TCH002
from langgraph.store.base import BaseStore  # noqa: TCH002
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import ValidationError

from .const import (
    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_SUMMARIZATION_MODEL_TOP_P,
    CONTEXT_MANAGE_USE_TOKENS,
    CONTEXT_MAX_MESSAGES,
    CONTEXT_MAX_TOKENS,
    CONTEXT_SUMMARIZE_THRESHOLD,
    EMBEDDING_MODEL_PROMPT_TEMPLATE,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
    SUMMARY_INITIAL_PROMPT,
    SUMMARY_PROMPT_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
    TOOL_CALL_ERROR_TEMPLATE,
    CONF_SUMMARIZATION_MODEL,
    RECOMMENDED_SUMMARIZATION_MODEL,
    SUMMARIZATION_MODEL_CTX,
    SUMMARIZATION_MODEL_PREDICT,
)

LOGGER = logging.getLogger(__name__)

class State(MessagesState):
    """Extend the MessagesState to include a summary key and model response metadata."""

    summary: str
    chat_model_usage_metadata: dict[str, Any]

async def _call_model(
        state: State, config: RunnableConfig, *, store: BaseStore
    ) -> dict[str, list[BaseMessage]]:
    """Coroutine to call the model."""
    model = config["configurable"]["chat_model"]
    prompt = config["configurable"]["prompt"]
    user_id = config["configurable"]["user_id"]
    llm_api = config["configurable"]["ha_llm_api"]

    last_message = state["messages"][-1]
    last_message_from_user = isinstance(last_message, HumanMessage)

    # Retrieve most recent or search for most relevant memories for context.
    # Use semantic search if the last message was from the user.
    query_prompt = EMBEDDING_MODEL_PROMPT_TEMPLATE.format(
        query=last_message.content
    ) if last_message_from_user else None
    mems = await store.asearch(
        (user_id, "memories"),
        query=query_prompt,
        limit=10
    )
    formatted_mems = "\n".join(f"[{mem.key}]: {mem.value}" for mem in mems)
    mems_message = f"\n<memories>\n{formatted_mems}\n</memories>" \
        if formatted_mems else ""

    # Base message list is the default system message plus any memories.
    messages = [SystemMessage(content=(prompt + mems_message))]

     # Try to retrieve the latest conversation summary. If it exists, add to messages.
    summary = state.get("summary", "")
    if summary:
        summary_message = f"{summary}"
        messages += [HumanMessage(content=summary_message)]

    # Add the HA LLM API prompt. There are two cases to consider.
    # If the last message was from the user, add the prompt before user message.
    # Else, add the prompt at the end of the messages.
    # This logic is designed to keep the most current status of the smart home near
    # the end of the context window to mitigate data drift when the context gets long.
    # This approach works much better than keeping the prompt in the system message.
    #if last_message_from_user:
        #messages += (
            #state["messages"][:-1] +
            #[HumanMessage(content=llm_api.api_prompt)] +
            #[last_message]
        #)
    #else:
        #messages += (state["messages"] + [HumanMessage(content=llm_api.api_prompt)])

    messages += state["messages"]

    LOGGER.debug("Model call messages: %s", messages)
    LOGGER.debug("Model call messages length: %s", len(messages))

    response = await model.ainvoke(messages)
    metadata = response.usage_metadata if hasattr(response, "usage_metadata") else {}
    # Clean up response, there is no need to include tool calls if there are none.
    if hasattr(response, "tool_calls"):
        response = AIMessage(content=response.content, tool_calls=response.tool_calls)
    else:
        response = AIMessage(content=response.content)
    LOGGER.debug("Model response: %s", response)
    LOGGER.debug("Token counts from metadata: %s", metadata)
    return {
        "messages": response,
        "chat_model_usage_metadata": metadata
    }

async def _summarize_and_trim(
        state: State, config: RunnableConfig, *, store: BaseStore
    ) -> dict[str, list[AnyMessage]]:
    """Coroutine to summarize and trim message history."""
    summary = state.get("summary", "")

    now = dt_util.now()
    dattim = now.strftime("%Y-%m-%d %H:%M:%S")

    if summary:
        summary_message = SUMMARY_PROMPT_TEMPLATE.format(summary=summary)
    else:
        summary_message = SUMMARY_INITIAL_PROMPT

    messages = (
        [SystemMessage(content=SUMMARY_SYSTEM_PROMPT)] +
        [HumanMessage(content=f"These are the smart home messages as of {dattim}:")] +
        [m.content for m in state["messages"] if isinstance(m,HumanMessage|AIMessage)] +
        [HumanMessage(content=summary_message)]
    )

    model = config["configurable"]["summarization_model"]
    options = config["configurable"]["options"]
    model_with_config = model.with_config(
        config={
            "model": options.get(
                CONF_SUMMARIZATION_MODEL,
                RECOMMENDED_SUMMARIZATION_MODEL,
            ),
            "temperature": options.get(
                CONF_SUMMARIZATION_MODEL_TEMPERATURE,
                RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
            ),
            "top_p": options.get(
                CONF_SUMMARIZATION_MODEL_TOP_P,
                RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
            ),
            "num_predict": SUMMARIZATION_MODEL_PREDICT,
            "num_ctx": SUMMARIZATION_MODEL_CTX,
        }
    )

    LOGGER.debug("Summary messages: %s", messages)
    response = await model_with_config.ainvoke(messages)

    if CONTEXT_MANAGE_USE_TOKENS:
        max_tokens = CONTEXT_MAX_TOKENS
        token_counter = config["configurable"]["chat_model"]
    else:
        max_tokens = CONTEXT_MAX_MESSAGES
        token_counter = len

    # Trim message history to manage context window length.
    hass = config["configurable"]["hass"]
    trimmed_messages = await hass.async_add_executor_job(
        partial(
            trim_messages,
            messages=state["messages"],
            token_counter=token_counter,
            max_tokens=max_tokens,
            strategy="last",
            start_on="human",
            include_system=True,
        )
    )
    messages_to_remove = [m for m in state["messages"] if m not in trimmed_messages]
    LOGGER.debug("Messages to remove: %s", messages_to_remove)
    remove_messages = [RemoveMessage(id=m.id) for m in messages_to_remove]

    return {"summary": response.content, "messages": remove_messages}

async def _call_tools(
        state: State, config: RunnableConfig, *, store: BaseStore
    ) -> dict[str, list[ToolMessage]]:
    """Coroutine to call Home Assistant or langchain LLM tools."""
    # Tool calls will be the last message in state.
    tool_calls = state["messages"][-1].tool_calls

    langchain_tools = config["configurable"]["langchain_tools"]
    ha_llm_api = config["configurable"]["ha_llm_api"]

    tool_responses: list[ToolMessage] = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        LOGGER.debug(
            "Tool call: %s(%s)", tool_name, tool_args
        )

        def _handle_tool_error(err:str, name:str, tid:str) -> ToolMessage:
            return ToolMessage(
                content=TOOL_CALL_ERROR_TEMPLATE.format(error=err),
                name=name,
                tool_call_id=tid,
                status="error",
            )

        # A langchain tool was called.
        if tool_name in langchain_tools:
            lc_tool = langchain_tools[tool_name.lower()]

            # Provide hidden args to tool at runtime.
            tool_call_copy = copy.deepcopy(tool_call)
            tool_call_copy["args"].update(
                {
                    "store": store,
                    "config": config,
                }
            )

            try:
                tool_response = await lc_tool.ainvoke(tool_call_copy)
            except (HomeAssistantError, ValidationError) as e:
                tool_response = _handle_tool_error(repr(e), tool_name, tool_call["id"])
        # A Home Assistant tool was called.
        else:
            tool_input = llm.ToolInput(
                tool_name=tool_name,
                tool_args=tool_args,
            )

            try:
                response = await ha_llm_api.async_call_tool(tool_input)

                tool_response = ToolMessage(
                    content=json.dumps(response),
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            except (HomeAssistantError, vol.Invalid) as e:
                tool_response = _handle_tool_error(repr(e), tool_name, tool_call["id"])

        LOGGER.debug("Tool response: %s", tool_response)
        tool_responses.append(tool_response)
    return {"messages": tool_responses}

def _should_continue(
        state: State
    ) -> Literal["action", "summarize_and_trim", "__end__"]:
    """Return the next node in graph to execute."""
    messages = state["messages"]

    if messages[-1].tool_calls:
        return "action"

    if len(messages) > CONTEXT_SUMMARIZE_THRESHOLD:
        LOGGER.debug("Summarizing conversation")
        return "summarize_and_trim"

    return "__end__"

# Define a new graph
workflow = StateGraph(State)

# Define nodes.
workflow.add_node("agent", _call_model)
workflow.add_node("action", _call_tools)
workflow.add_node("summarize_and_trim", _summarize_and_trim)

# Define edges.
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", _should_continue)
workflow.add_edge("action", "agent")
workflow.add_edge("summarize_and_trim", END)
