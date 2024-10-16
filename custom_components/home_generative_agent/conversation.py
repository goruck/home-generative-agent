"""Conversation support for Home Generative Agent using langgraph."""
from __future__ import annotations

import json
import logging
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import voluptuous as vol
from homeassistant.components import assist_pipeline, conversation
from homeassistant.components.conversation import trace
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.exceptions import (
    HomeAssistantError,
    TemplateError,
)
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import intent, llm, template
from homeassistant.util import ulid
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from voluptuous_openapi import convert

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_MESSAGES,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_MESSAGES,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddEntitiesCallback
    from langchain_openai import ChatOpenAI

    from . import HGAConfigEntry

LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: HGAConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = HGAConversationEntity(config_entry)
    async_add_entities([agent])

def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format tool specification."""
    tool_spec = {
        "name": tool.name,
        "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
    }
    if tool.description:
        tool_spec["description"] = tool.description
    return {"type": "function", "function": tool_spec}

def _state_modifier(
        state: MessagesState, max_messages: Literal["int"]
    ) -> list[AnyMessage]:
    """Given the agent state, return a list of trimmed messages for the chat model."""
    return trim_messages(
        state,
        token_counter=len,
        # When token_counter=len, each message will be counted as a single token.
        max_tokens=max_messages,
        # Keep the last <= n_count tokens of the messages.
        strategy="last",
        # Most chat models expect that chat history starts with either:
        # (1) a HumanMessage or
        # (2) a SystemMessage followed by a HumanMessage
        # start_on="human" makes sure we produce a valid chat history
        start_on="human",
        # Most chat models expect that chat history ends with either:
        # (1) a HumanMessage or
        # (2) a ToolMessage
        end_on=("human", "tool"),
        # Usually, we want to keep the SystemMessage
        # if it's present in the original history.
        # The SystemMessage has special instructions for the model.
        include_system=True,
        allow_partial=False,
    )

async def _call_model(
        state: MessagesState,
        model: ChatOpenAI,
        prompt: Literal["str"],
        api: llm.API,
        max_messages: Literal["int"]
    ) -> dict[str, list[BaseMessage]]:
    """Define function that calls the model."""
    messages = [SystemMessage(content=prompt)] + state["messages"]
    trace.async_conversation_trace_append(
        trace.ConversationTraceEventType.AGENT_DETAIL,
        {"messages": messages, "tools": api.tools if api else None},
    )
    # Trim message history to manage context window length.
    trimmed_messages= _state_modifier(
        state=messages,
        max_messages=max_messages
    )
    LOGGER.debug("Trimmed messages: %s", trimmed_messages)
    response = await model.ainvoke(trimmed_messages)
    # Return a list, because it will get added to the existing list.
    return {"messages": response}

async def _call_tools(
        state: MessagesState, api: llm.API
    ) -> dict[str, list[ToolMessage]]:
    """Coroutine to call Home Assistant LLM tools."""
    # Tool calls will be the last message in state.
    tool_calls = state["messages"][-1].tool_calls

    tool_responses: list[ToolMessage] = []
    for tool_call in tool_calls:
        tool_input = llm.ToolInput(
            tool_name=tool_call["name"],
            tool_args= tool_call["args"],
        )
        LOGGER.debug(
            "Tool call: %s(%s)", tool_input.tool_name, tool_input.tool_args
        )

        try:
            tool_response = await api.async_call_tool(tool_input)
        except (HomeAssistantError, vol.Invalid) as e:
            tool_response = {"error": type(e).__name__}
            if str(e):
                tool_response["error_text"] = str(e)

        LOGGER.debug("Tool response: %s", tool_response)
        tool_responses.append(
            ToolMessage(
                content=json.dumps(tool_response),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
        )
    return {"messages": tool_responses}

def _should_continue(state: MessagesState) -> Literal["action", "__end__"]:
    """Return the next node in graph to execute."""
    last_message = state["messages"][-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise if there is, we continue
    return "action"

class HGAConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Home Generative Assistant conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self.app_config: dict[str, dict[str, str]] = {}
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="LinTek",
            model="HGA",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )
        # TODO:Change memory to langgraph-checkpoint-postgres to make robust
        self.memory = MemorySaver()

        set_llm_cache(InMemoryCache())

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process the user input."""
        options = self.entry.options
        intent_response = intent.IntentResponse(language=user_input.language)
        llm_api: llm.API | None = None
        tools: list[dict[str, Any]] | None = None
        user_name: str | None = None
        llm_context = llm.LLMContext(
            platform=DOMAIN,
            context=user_input.context,
            user_prompt=user_input.text,
            language=user_input.language,
            assistant=conversation.DOMAIN,
            device_id=user_input.device_id,
        )

        if self.entry.options.get(CONF_LLM_HASS_API):
            try:
                llm_api = await llm.async_get_api(
                    self.hass,
                    self.entry.options[CONF_LLM_HASS_API],
                    llm_context,
                )
            except HomeAssistantError as err:
                LOGGER.error("Error getting LLM API: %s", err)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Error preparing LLM API: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=user_input.conversation_id
                )
            tools = [
               _format_tool(tool, llm_api.custom_serializer) for tool in llm_api.tools
            ]

        # Conversation IDs are ULIDs. Generate a new one if not provided.
        # If an old ULID is passed in, generate a new one to indicate
        # a new conversation was started. If the user picks their own, they
        # want to track a conversation, so respect it.
        if user_input.conversation_id is None:
            conversation_id = ulid.ulid_now()
        elif user_input.conversation_id in self.app_config["configurable"].values():
            conversation_id = user_input.conversation_id
        else:
            try:
                ulid.ulid_to_bytes(user_input.conversation_id)
                conversation_id = ulid.ulid_now()
            except ValueError:
                conversation_id = user_input.conversation_id

        if (
            user_input.context
            and user_input.context.user_id
            and (
                user := await self.hass.auth.async_get_user(user_input.context.user_id)
            )
        ):
            user_name = user.name

        try:
            prompt_parts = [
                template.Template(
                    llm.BASE_PROMPT
                    + options.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT),
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

        except TemplateError as err:
            LOGGER.error("Error rendering prompt: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem with my template: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        if llm_api:
            prompt_parts.append(llm_api.api_prompt)

        prompt: Literal["str"] = "\n".join(prompt_parts)

        LOGGER.debug("Tools: %s", tools)

        model = self.entry.runtime_data
        model_with_config = model.with_config(
            {"configurable":
                {
                    "model_name": self.entry.options.get(
                        CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL
                    ),
                    "temperature": self.entry.options.get(
                        CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
                    ),
                    "max_tokens": self.entry.options.get(
                        CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
                    ),
                    "top_p": self.entry.options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                }
            }
        )
        model_with_tools = model_with_config.bind_tools(tools)

        # TODO: make graph creation a function and call here

        # Define a new graph
        workflow = StateGraph(MessagesState)

        max_messages = self.entry.options.get(
            CONF_MAX_MESSAGES, RECOMMENDED_MAX_MESSAGES
        )

        # Define the two nodes we will cycle between
        workflow.add_node("agent", partial(
            _call_model,
            model=model_with_tools,
            prompt=prompt,
            api=llm_api,
            max_messages=max_messages)
        )
        workflow.add_node("action", partial(_call_tools, api=llm_api))

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        workflow.add_edge(START, "agent")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            # First, define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, pass in the function that will determine which node is called next.
            _should_continue,
        )

        # Now add a normal edge from `tools` to `agent`.
        # This means that after `tools` is called, `agent` node is called next.
        workflow.add_edge("action", "agent")

        # Complile graph into a LangChain Runnable.
        app = workflow.compile(checkpointer=self.memory, debug=True)

        self.app_config = {"configurable": {"thread_id": conversation_id}}

        # Interact with app.
        try:
            response = await app.ainvoke(
                {"messages": [HumanMessage(content=user_input.text)]},
                config=self.app_config
            )
        except HomeAssistantError as err:
            LOGGER.error(err, exc_info=err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Something went wrong: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )
        except Exception as err:
            LOGGER.error(err, exc_info=err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Something went wrong: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        LOGGER.debug("Response: %s", response)

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response["messages"][-1].content)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)
