"""Conversation support for Home Generative Agent using langgraph."""

from __future__ import annotations

import logging
import string
from typing import TYPE_CHECKING, Any, Literal

import homeassistant.util.dt as dt_util
from homeassistant.components import conversation
from homeassistant.components.conversation import trace
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import intent, llm, template
from homeassistant.util import ulid
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_debug, set_llm_cache, set_verbose
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from voluptuous_openapi import convert

from .agent.graph import workflow
from .agent.tools import (
    add_automation,
    alarm_control,
    confirm_sensitive_action,
    get_and_analyze_camera_image,
    get_entity_history,
    upsert_memory,
)
from .const import (
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_PROMPT,
    CRITICAL_ACTION_PROMPT,
    DOMAIN,
    LANGCHAIN_LOGGING_LEVEL,
    TOOL_CALL_ERROR_SYSTEM_MESSAGE,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from agent.graph import State
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddEntitiesCallback
    from langchain_core.runnables import RunnableConfig

    from .core.runtime import HGAConfigEntry

LOGGER = logging.getLogger(__name__)

if LANGCHAIN_LOGGING_LEVEL == "verbose":
    set_verbose(True)
    set_debug(False)
elif LANGCHAIN_LOGGING_LEVEL == "debug":
    set_verbose(False)
    set_debug(True)
else:
    set_verbose(False)
    set_debug(False)


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format Home Assistant LLM tools to be compatible with OpenAI format."""
    tool_spec = {
        "name": tool.name,
        "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
    }
    if tool.description:
        tool_spec["description"] = tool.description
    return {"type": "function", "function": tool_spec}


def _convert_content(
    content: conversation.UserContent | conversation.AssistantContent,
) -> HumanMessage | AIMessage:
    """Convert HA native chat messages to LangChain messages."""
    if content.content is None:
        LOGGER.warning("Content is None, returning empty message")
        return HumanMessage(content="")
    if isinstance(content, conversation.UserContent):
        return HumanMessage(content=content.content)
    return AIMessage(content=content.content)


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

        if self.entry.options.get(CONF_LLM_HASS_API):
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

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def _async_handle_message(  # noqa: PLR0915
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process the user input."""
        hass = self.hass
        options = self.entry.options
        runtime_data = self.entry.runtime_data
        intent_response = intent.IntentResponse(language=user_input.language)
        tools: list[dict[str, Any]] | None = None
        user_name: str | None = None
        llm_context = llm.LLMContext(
            platform=DOMAIN,
            context=user_input.context,
            language=user_input.language,
            assistant=conversation.DOMAIN,
            device_id=user_input.device_id,
        )

        # Include only HA User/Assistant messages not already seen by this entity.
        message_history = [
            _convert_content(m)
            for m in chat_log.content
            if isinstance(m, conversation.UserContent | conversation.AssistantContent)
        ]
        # The last chat log entry will be the current user request—add it later.
        message_history = message_history[:-1]

        if (mhlen := len(message_history)) <= self.message_history_len:
            message_history = []
        else:
            diff = mhlen - self.message_history_len
            message_history = message_history[-diff:]
            self.message_history_len = mhlen

        # HA tools & schema
        try:
            llm_api = await llm.async_get_api(
                hass,
                options[CONF_LLM_HASS_API],
                llm_context,
            )
        except HomeAssistantError:
            msg = "Error getting LLM API, check your configuration."
            LOGGER.exception(msg)
            intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, msg)
            return conversation.ConversationResult(
                response=intent_response, conversation_id=user_input.conversation_id
            )

        tools = [
            _format_tool(tool, llm_api.custom_serializer) for tool in llm_api.tools
        ]

        # Add LangChain-native tools (wired in graph via config).
        langchain_tools: dict[str, Any] = {
            "get_and_analyze_camera_image": get_and_analyze_camera_image,
            "upsert_memory": upsert_memory,
            "add_automation": add_automation,
            "get_entity_history": get_entity_history,
            "confirm_sensitive_action": confirm_sensitive_action,
            "alarm_control": alarm_control,
        }
        tools.extend(langchain_tools.values())

        # Conversation ID
        conversation_id = (
            ulid.ulid_now()
            if chat_log.conversation_id is None
            else chat_log.conversation_id
        )
        LOGGER.debug("Conversation ID: %s", conversation_id)

        # Resolve user name (None means automation)
        if (
            user_input.context
            and user_input.context.user_id
            and (user := await hass.auth.async_get_user(user_input.context.user_id))
        ):
            user_name = user.name

        # Build system prompt
        try:
            pin_enabled = options.get(CONF_CRITICAL_ACTION_PIN_ENABLED, True)
            critical_prompt = CRITICAL_ACTION_PROMPT if pin_enabled else ""
            prompt_parts = [
                template.Template(
                    (
                        llm.DATE_TIME_PROMPT
                        + options.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT)
                        + f"\nYou are in the {self.tz} timezone."
                        + critical_prompt
                        + TOOL_CALL_ERROR_SYSTEM_MESSAGE
                        if tools
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
        except TemplateError as err:
            LOGGER.exception("Error rendering prompt.")
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

        prompt = "\n".join(prompt_parts)

        # Use the already-configured chat model from __init__.py
        base_llm = runtime_data.chat_model
        try:
            chat_model_with_tools = base_llm.bind_tools(tools)
        except AttributeError:
            LOGGER.exception("Error during conversation processing.")
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Model must support tool calling, model = {type(base_llm).__name__}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        # A user name of None indicates an automation is being run.
        user_name = "robot" if user_name is None else user_name
        # Remove special characters since memory namespace labels cannot contain them.
        user_name = user_name.translate(str.maketrans("", "", string.punctuation))
        LOGGER.debug("User name: %s", user_name)

        app_config: RunnableConfig = {
            "configurable": {
                "thread_id": conversation_id,
                "user_id": user_name,
                "chat_model": chat_model_with_tools,
                "chat_model_options": runtime_data.chat_model_options,
                "prompt": prompt,
                "options": options,
                "vlm_model": runtime_data.vision_model,
                "summarization_model": runtime_data.summarization_model,
                "langchain_tools": langchain_tools,
                "ha_llm_api": llm_api or None,
                "hass": hass,
                "pending_actions": runtime_data.pending_actions,
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
        messages: list[AnyMessage] = []
        messages.extend(message_history)
        messages.append(HumanMessage(content=user_input.text))
        app_input: State = {
            "messages": messages,
            "summary": "",
            "chat_model_usage_metadata": {},
            "messages_to_remove": [],
        }

        # Interact with agent app.
        try:
            response = await app.ainvoke(input=app_input, config=app_config)
        except HomeAssistantError as err:
            LOGGER.exception("LangGraph error during conversation processing.")
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Something went wrong: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL,
            {"messages": response["messages"], "tools": tools if tools else None},
        )

        LOGGER.debug("====== End of run ======")

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response["messages"][-1].content)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features.
        await hass.config_entries.async_reload(entry.entry_id)
