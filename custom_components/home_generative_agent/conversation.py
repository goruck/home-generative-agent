"""Conversation support for Home Generative Agent using langgraph."""
from __future__ import annotations

import base64
import copy
import json
import logging
import string
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Literal

import voluptuous as vol
from homeassistant.components import assist_pipeline, camera, conversation
from homeassistant.components.conversation import trace
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
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
    AnyMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field, ValidationError
from ulid import ULID
from voluptuous_openapi import convert

from .const import (
    CONF_CHAT_MODEL,
    CONF_CHAT_MODEL_TEMPERATURE,
    CONF_PROMPT,
    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_VISION_MODEL_TEMPERATURE,
    CONF_VLM,
    CONTEXT_MAX_MESSAGES,
    CONTEXT_SUMMARIZE_THRESHOLD,
    DOMAIN,
    LANGCHAIN_LOGGING_LEVEL,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_VISION_MODEL_TEMPERATURE,
    RECOMMENDED_VLM,
    SUMMARY_INITAL_PROMPT,
    SUMMARY_PROMPT_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
    TOOL_CALL_ERROR_SYSTEM_MESSSAGE,
    TOOL_CALL_ERROR_TEMPLATE,
    VISION_MODEL_IMAGE_HEIGHT,
    VISION_MODEL_IMAGE_WIDTH,
    VISION_MODEL_SYSTEM_PROMPT,
    VISION_MODEL_USER_PROMPT_TEMPLATE,
    VLM_NUM_PREDICT,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from homeassistant.config_entries import ConfigEntry
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

class State(MessagesState):
    """Extend the MessagesState to include a summary key."""

    summary: str

def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format Home Assistant LLM tools to be compatible with OpenAI."""
    tool_spec = {
        "name": tool.name,
        "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
    }
    if tool.description:
        tool_spec["description"] = tool.description
    return {"type": "function", "function": tool_spec}

async def _call_model(
        state: State, config: RunnableConfig, *, store: BaseStore
    ) -> dict[str, list[BaseMessage]]:
    """Coroutine to call the model."""
    model = config["configurable"]["chat_model"]
    prompt = config["configurable"]["prompt"]
    user_id = config["configurable"]["user_id"]

    # Retrieve the most recent memories for context.
    mems = await store.asearch(("memories", user_id), limit=10)
    formatted_mems = "\n".join(f"[{mem.key}]: {mem.value}" for mem in mems)
    mems_message = f"\n<memories>\n{formatted_mems}\n</memories>" if formatted_mems else ""

    # Retrive the latest conversation summary.
    summary = state.get("summary", "")
    summary_message = f"\nSummary of conversation earlier: {summary}" if summary else ""

    messages = [SystemMessage(
        content=(prompt + mems_message + summary_message)
    )] + state["messages"]

    LOGGER.debug("Model call messages: %s", messages)
    LOGGER.debug("Model call messages length: %s", len(messages))
    #LOGGER.debug("Model call messages: ")
    #for m in messages:
        #LOGGER.debug(m.pretty_repr())

    response = await model.ainvoke(messages)
    return {"messages": response}

async def _summarize_and_trim(
        state: State, config: RunnableConfig, *, store: BaseStore
    ) -> dict[str, list[AnyMessage]]:
    """Coroutine to summarize and trim message history."""
    summary = state.get("summary", "")

    if summary:
        summary_message = SUMMARY_PROMPT_TEMPLATE.format(summary=summary)
    else:
        summary_message = SUMMARY_INITAL_PROMPT

    messages = (
        [SystemMessage(content=SUMMARY_SYSTEM_PROMPT)] +
        state["messages"] +
        [HumanMessage(content=summary_message)]
    )

    model = config["configurable"]["vlm_model"]
    options = config["configurable"]["options"]
    model_with_config = model.with_config(
        {"configurable":
            {
                "model": options.get(
                    CONF_VLM,
                    RECOMMENDED_VLM,
                ),
                "format": "",
                "temperature": options.get(
                    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
                    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
                ),
                "num_predict": VLM_NUM_PREDICT,
            }
        }
    )

    LOGGER.debug("Summary messages: %s", messages)
    response = await model_with_config.ainvoke(messages)

    # Trim message history to manage context window length.
    trimmed_messages = trim_messages(
        messages=state["messages"],
        token_counter=len,
        max_tokens=CONTEXT_MAX_MESSAGES,
        strategy="last",
        start_on="human",
        include_system=True,
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

async def _get_camera_image(hass: HomeAssistant, camera_name: str) -> bytes:
    """Get an image from a given camera."""
    camera_entity_id: str = f"camera.{camera_name.lower()}"
    try:
        image = await camera.async_get_image(
            hass=hass,
            entity_id=camera_entity_id,
            width=VISION_MODEL_IMAGE_WIDTH,
            height=VISION_MODEL_IMAGE_HEIGHT
        )
    except HomeAssistantError as err:
        LOGGER.error(
            "Error getting image from camera '%s' with error: %s",
            camera_entity_id, err
        )

    return image.content

async def _analyze_image(
        vlm_model: ChatOllama,
        options: dict[str, Any] | MappingProxyType[str, Any],
        image: bytes
    ) -> str:
    """Analyze an image."""
    encoded_image = base64.b64encode(image).decode("utf-8")

    def prompt_func(data: dict[str, Any]) -> list[AnyMessage]:
        system = data["system"]
        text = data["text"]
        image = data["image"]

        text_part = {"type": "text", "text": text}
        image_part = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        }

        content_parts = []
        content_parts.append(text_part)
        content_parts.append(image_part)

        return [SystemMessage(content=system), HumanMessage(content=content_parts)]

    class ObjectTypeAndLocation(BaseModel):
        """Get type and location of objects in image."""

        object_type: str = Field(
            description="the type of obect in the immage"
        )
        object_location: str = Field(
            description="the location of the object in the image"
        )

    class ImageSceneAnalysis(BaseModel):
        """
        Get image scene analysis.

        Includes a description of the image, type and location of objects present,
        number of people present and number of animals present in the image.
        """

        description: str = Field(
            description="description of the image scene"
        )
        objects: list[ObjectTypeAndLocation] = Field(
            description="object type and location in image"
        )
        people: int = Field(
            description="number of people in the image"
        )
        animals: int = Field(
            description="number of aniamls in the image"
        )

    schema = json.dumps(ImageSceneAnalysis.model_json_schema())

    model = vlm_model
    model_with_config = model.with_config(
        {"configurable":
            {
                "model": options.get(
                    CONF_VLM,
                    RECOMMENDED_VLM,
                ),
                "format": "json",
                "temperature": options.get(
                    CONF_VISION_MODEL_TEMPERATURE,
                    RECOMMENDED_VISION_MODEL_TEMPERATURE,
                ),
                "num_predict": VLM_NUM_PREDICT,
            }
        }
    )

    chain = prompt_func | model_with_config

    try:
        response =  await chain.ainvoke(
            {
                "system": VISION_MODEL_SYSTEM_PROMPT,
                "text": VISION_MODEL_USER_PROMPT_TEMPLATE.format(schema=schema),
                "image": encoded_image
            }
        )
    except HomeAssistantError as err: #TODO: add validation error handling and retry prompt
        LOGGER.error("Error analyzing image %s", err)

    return response

@tool(parse_docstring=False)
async def get_and_analyze_camera_image(
        camera_name: str,
        *,
        # Hide these arguments from the model.
        config: Annotated[RunnableConfig, InjectedToolArg()],
        store: Annotated[BaseStore, InjectedStore()],
    ) -> str:
    """Get an image from a given camera and analyze it."""
    hass = config["configurable"]["hass"]
    vlm_model = config["configurable"]["vlm_model"]
    options = config["configurable"]["options"]
    image = await _get_camera_image(hass, camera_name)
    return await _analyze_image(vlm_model, options, image)

@tool(parse_docstring=False)
async def upsert_memory(
    content: str,
    context: str,
    *,
    memory_id: ULID | None = None,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """
    Upsert a memory in the database.

    If a memory conflicts with an existing one, then just UPDATE the
    existing one by passing in memory_id - don't create two memories
    that are the same. If the user corrects a memory, UPDATE it.

    Args:
        content: The main content of the memory. For example:
            "User expressed interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
        memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
            The memory to overwrite

    Returns:
        A string containing the stored memory id.

    """
    mem_id = memory_id or ulid.ulid_now()
    await store.aput(
        ("memories", config["configurable"]["user_id"]),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"

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
        self.app_config: dict[str, dict[str, str]] = {"configurable": {"thread_id": ""}}
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

        # Create database for thread-based (short-term) memory.
        # TODO:Change memory to langgraph-checkpoint-postgres to make robust
        self.memory = MemorySaver()

        # Use in-memory caching for calls to LLMs.
        set_llm_cache(InMemoryCache())

        self.store = InMemoryStore()

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
        hass = self.hass
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
        }
        tools.extend(langchain_tools.values())

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
                        + TOOL_CALL_ERROR_SYSTEM_MESSSAGE if tools else ""
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

        chat_model = self.entry.chat_model
        chat_model_with_config = chat_model.with_config(
            {"configurable":
                {
                    "model_name": self.entry.options.get(
                        CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL
                    ),
                    "temperature": self.entry.options.get(
                        CONF_CHAT_MODEL_TEMPERATURE, RECOMMENDED_CHAT_MODEL_TEMPERATURE
                    ),
                }
            }
        )
        chat_model_with_tools = chat_model_with_config.bind_tools(tools)

        # TODO: make graph creation a function and call here

        # Define a new graph
        workflow = StateGraph(State)

        # Define nodes.
        workflow.add_node("agent", _call_model)

        workflow.add_node("action", _call_tools)
        workflow.add_node("summarize_and_trim", _summarize_and_trim)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        workflow.add_edge(START, "agent")

        # Add conditional edge for the agent node.
        # This will deterine if a tool call is needed.
        workflow.add_conditional_edges(
            # First, define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, pass in the function that will determine which node is called next.
            _should_continue,
        )

        workflow.add_edge("action", "agent")

        workflow.add_edge("summarize_and_trim", END)

        # Complile graph into a LangChain Runnable.
        app = workflow.compile(store=self.store, checkpointer=self.memory, debug=True)

        # Remove special characters since namespace labels cannot contain.
        user_name_clean = user_name.translate(str.maketrans("", "", string.punctuation))

        self.app_config = {
            "configurable": {
                "thread_id": conversation_id,
                "user_id": user_name_clean,
                "chat_model": chat_model_with_tools,
                "prompt": prompt,
                "options": options,
                "vlm_model": self.entry.vision_model,
                "langchain_tools": langchain_tools,
                "ha_llm_api": llm_api,
                "hass": hass,
            },
            "recursion_limit": 10
        }

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

        #LOGGER.debug("App response: ")
        #for m in response["messages"]:
            #LOGGER.debug(m.pretty_repr())

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
