"""Conversation support for Home Generative Agent using langgraph."""
from __future__ import annotations

import logging
import string
from typing import TYPE_CHECKING, Any, Literal

import homeassistant.util.dt as dt_util
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
from langchain.globals import set_debug, set_verbose
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from voluptuous_openapi import convert

from .const import (
    CHAT_MODEL_MAX_TOKENS,
    CHAT_MODEL_NUM_CTX,
    CONF_CHAT_MODEL,
    CONF_CHAT_MODEL_LOCATION,
    CONF_CHAT_MODEL_TEMPERATURE,
    CONF_EDGE_CHAT_MODEL,
    CONF_EDGE_CHAT_MODEL_TEMPERATURE,
    CONF_EDGE_CHAT_MODEL_TOP_P,
    CONF_PROMPT,
    DOMAIN,
    LANGCHAIN_LOGGING_LEVEL,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CHAT_MODEL_LOCATION,
    RECOMMENDED_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_EDGE_CHAT_MODEL,
    RECOMMENDED_EDGE_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_EDGE_CHAT_MODEL_TOP_P,
    TOOL_CALL_ERROR_SYSTEM_MESSAGE,
)
from .graph import workflow
from .tools import (
    add_automation,
    get_and_analyze_camera_image,
    get_entity_history,
    upsert_memory,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddEntitiesCallback

    from . import HGAConfigEntry

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
    # Add dummy arg descriptions if needed for custom token counter.
    for arg in list(tool_spec["parameters"]["properties"]):
        try:
            _ = tool_spec["parameters"]["properties"][arg]["description"]
        except KeyError:
            tool_spec["parameters"]["properties"][arg]["description"] = ""
    if tool.description:
        tool_spec["description"] = tool.description
    return {"type": "function", "function": tool_spec}

def _convert_content(
    content: conversation.UserContent | conversation.AssistantContent,
) -> HumanMessage | AIMessage:
    """Convert HA native chat messages to LangChain messages."""
    if isinstance(content, conversation.UserContent):
        return HumanMessage(content=content.content)

    return AIMessage(content=content.content)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: HGAConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = HGAConversationEntity(config_entry)
    async_add_entities([agent])

class HGAConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Home Generative Assistant conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self.app_config: dict[str, dict[str, str]]
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="LinTek",
            model="HGA",
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

        # Database for thread-based (short-term) memory.
        self.checkpointer = AsyncPostgresSaver(self.entry.pool)
        # NOTE: must call .setup() the first time checkpointer is used.
        #await checkpointer.setup()  # noqa: ERA001

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

    async def _async_handle_message(
            self,
            user_input: conversation.ConversationInput,
            chat_log: conversation.ChatLog,
        ) -> conversation.ConversationResult:
        """Process the user input."""
        hass = self.hass
        options = self.entry.options
        intent_response = intent.IntentResponse(language=user_input.language)
        llm_api: llm.API | None = None
        tools: list[dict[str, Any]] | None = None
        user_name: str | None = None
        llm_context = llm.LLMContext(
            platform=DOMAIN,
            context=user_input.context,
            language=user_input.language,
            assistant=conversation.DOMAIN,
            device_id=user_input.device_id,
        )

        # We are only concerned with User or Assistant content from the chat log since
        # these may be from the local agent when user asks for something that the local
        # agent can quickly process. These messages need to be included in this agent so
        # the entire context is visible. LangChain will handle the rest of the message
        # history so we don't need to stream anything back into the chat log.
        message_history = [
            _convert_content(m) for m in chat_log.content
            if isinstance(m, conversation.UserContent | conversation.AssistantContent)
        ]
        # The last chat log entry will be the current user request, include it later.
        message_history = message_history[:-1]

        # If new HA agent messages are added to the chat history, include them, else
        # ignore the older ones since they were already included in the context.
        if (mhlen := len(message_history)) <= self.message_history_len:
            message_history = []
        else:
            diff = mhlen - self.message_history_len
            message_history = message_history[-diff:]
            self.message_history_len = mhlen

        if options.get(CONF_LLM_HASS_API):
            try:
                llm_api = await llm.async_get_api(
                    hass,
                    options[CONF_LLM_HASS_API],
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

        # Add langchain tools to the list of HA tools.
        langchain_tools = {
            "get_and_analyze_camera_image": get_and_analyze_camera_image,
            "upsert_memory": upsert_memory,
            "add_automation": add_automation,
            "get_entity_history": get_entity_history,
        }
        tools.extend(langchain_tools.values())

        # Conversation IDs are ULIDs. Generate a new one if not provided.
        if chat_log.conversation_id is None:
            conversation_id = ulid.ulid_now()
        else:
            conversation_id = chat_log.conversation_id
        LOGGER.debug("Conversation ID: %s", conversation_id)

        if (
            user_input.context
            and user_input.context.user_id
            and (
                user := await hass.auth.async_get_user(user_input.context.user_id)
            )
        ):
            user_name = user.name

        try:
            prompt_parts = [
                template.Template(
                    (
                        llm.BASE_PROMPT
                        + options.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT)
                        + f"\nYou are in the {self.tz} timezone."
                        + TOOL_CALL_ERROR_SYSTEM_MESSAGE if tools else ""
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

        prompt = "\n".join(prompt_parts)

        chat_model_location = options.get(
            CONF_CHAT_MODEL_LOCATION,
            RECOMMENDED_CHAT_MODEL_LOCATION
        )
        if chat_model_location == "edge":
            chat_model = self.entry.edge_chat_model
            chat_model_with_config = chat_model.with_config(
                {"configurable":
                    {
                        "model": options.get(
                            CONF_EDGE_CHAT_MODEL,
                            RECOMMENDED_EDGE_CHAT_MODEL
                        ),
                        "temperature": options.get(
                            CONF_EDGE_CHAT_MODEL_TEMPERATURE,
                            RECOMMENDED_EDGE_CHAT_MODEL_TEMPERATURE
                        ),
                        "top_p": options.get(
                            CONF_EDGE_CHAT_MODEL_TOP_P,
                            RECOMMENDED_EDGE_CHAT_MODEL_TOP_P,
                        ),
                        "num_predict": CHAT_MODEL_MAX_TOKENS,
                        "num_ctx": CHAT_MODEL_NUM_CTX,

                    }
                }
            )
        else:
            chat_model = self.entry.chat_model
            chat_model_with_config = chat_model.with_config(
                {"configurable":
                    {
                        "model_name": options.get(
                            CONF_CHAT_MODEL,
                            RECOMMENDED_CHAT_MODEL
                        ),
                        "temperature": options.get(
                            CONF_CHAT_MODEL_TEMPERATURE,
                            RECOMMENDED_CHAT_MODEL_TEMPERATURE
                        ),
                        "max_tokens": CHAT_MODEL_MAX_TOKENS,
                    }
                }
            )

        chat_model_with_tools = chat_model_with_config.bind_tools(tools)

        # A user name of None indicates an automation is being run.
        user_name = "robot" if user_name is None else user_name
        # Remove special characters since memory namespace labels cannot contain.
        user_name = user_name.translate(str.maketrans("", "", string.punctuation))
        LOGGER.debug("User name: %s", user_name)

        self.app_config = {
            "configurable": {
                "thread_id": conversation_id,
                "user_id": user_name,
                "chat_model": chat_model_with_tools,
                "prompt": prompt,
                "options": options,
                "vlm_model": self.entry.vision_model,
                "summarization_model": self.entry.summarization_model,
                "langchain_tools": langchain_tools,
                "ha_llm_api": llm_api or None,
                "hass": hass,
            },
            "recursion_limit": 10
        }

        # Compile graph into a LangChain Runnable.
        app = workflow.compile(
            store=self.entry.store,
            checkpointer=self.checkpointer,
            debug=LANGCHAIN_LOGGING_LEVEL=="debug"
        )

        # The input to the agent is the message history combined with
        # the user request.
        messages: list[HumanMessage | AIMessage] = []
        messages.extend(message_history)
        messages.append(HumanMessage(content=user_input.text))

        # Interact with agent app.
        try:
            response = await app.ainvoke(
                {"messages": messages},
                config=self.app_config
            )
        except HomeAssistantError as err:
            LOGGER.error("LangGraph error: %s", err)
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
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)
