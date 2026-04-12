"""Conversation support for Home Generative Agent using langgraph."""

from __future__ import annotations

import hashlib
import json
import logging
import string
from typing import TYPE_CHECKING, Any, Literal

import homeassistant.util.dt as dt_util
import voluptuous as vol
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
    from collections.abc import Callable

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddEntitiesCallback
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
    """Convert HA native chat messages to LangChain messages."""
    if content.content is None:
        _LOGGER.warning("Content is None, returning empty message")
        return HumanMessage(content="")
    if isinstance(content, conversation.UserContent):
        return HumanMessage(content=content.content)
    return AIMessage(content=content.content)


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
) -> None:
    """Batch tool indexing into the store and update hashes on success."""
    try:
        if index_tasks:
            await gather_store_puts_in_chunks(index_tasks)
        rd.tool_content_hashes.update(tool_hashes)
        rd.tool_index_ready = True
    except Exception:
        _LOGGER.exception("Global tool index background task failed")
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

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def _async_handle_message(  # noqa: PLR0912, PLR0915
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process the user input."""
        hass = self.hass
        options = self.entry.runtime_data.options
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

        conversation_id = (
            ulid.ulid_now()
            if chat_log.conversation_id is None
            else chat_log.conversation_id
        )

        # HA tools & schema
        # Gather ALL selected APIs
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
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "No LLM APIs could be loaded. "
                f"Failed: {', '.join(failed_apis)}. "
                f"Configured: {', '.join(active_api_ids)}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        # Multi-API initialization
        llm_api = MultiLLMAPI(active_apis, {})

        # --- Global Tool Indexing (Background) ---
        await self._async_index_tools(llm_context, runtime_data)

        if not options.get(CONF_SCHEMA_FIRST_YAML, False) and _is_dashboard_request(
            user_input.text
        ):
            intent_response.async_set_speech(
                """Please enable 'Schema-first JSON for YAML requests' in
                HGA's configuration and try again"""
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

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

        # Conversation ID
        _LOGGER.debug("Conversation ID: %s", conversation_id)

        # Resolve user name (None means automation)
        if (
            user_input.context
            and user_input.context.user_id
            and (user := await hass.auth.async_get_user(user_input.context.user_id))
        ):
            user_name = user.name

        # Build system prompt
        try:
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
            _LOGGER.exception("Error rendering prompt.")
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

        # Use the already-configured chat model from __init__.py.
        # Tool binding happens dynamically per-turn inside _call_model via RAG.
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
        user_name = "robot" if user_name is None else user_name
        # Remove special characters since memory namespace labels cannot contain them.
        user_name = user_name.translate(str.maketrans("", "", string.punctuation))
        _LOGGER.debug("User name: %s", user_name)

        app_config: RunnableConfig = {
            "configurable": {
                "thread_id": conversation_id,
                "user_id": user_name,
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
        messages: list[AnyMessage] = []
        messages.extend(message_history)
        messages.append(HumanMessage(content=user_input.text))
        app_input: State = {
            "messages": messages,
            "summary": "",
            "chat_model_usage_metadata": {},
            "messages_to_remove": [],
            "selected_tools": [],
            "tool_routing_map": {},
        }

        # Interact with agent app.
        try:
            response = await app.ainvoke(input=app_input, config=app_config)
        except HomeAssistantError as err:
            _LOGGER.exception("LangGraph error during conversation processing.")
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
            {"messages": response["messages"], "tools": tools or None},
        )

        _LOGGER.debug("====== End of run ======")

        intent_response = intent.IntentResponse(language=user_input.language)
        final_content = response["messages"][-1].content
        if isinstance(final_content, str):
            if options.get(CONF_SCHEMA_FIRST_YAML, False):
                final_content = _maybe_fix_dashboard_entities(final_content, hass)
            else:
                final_content = _fix_entity_ids_in_text(final_content, hass)
            final_content = _convert_schema_json_to_yaml(
                final_content, options.get(CONF_SCHEMA_FIRST_YAML, False)
            )
            _LOGGER.debug("Final response content: %s", final_content)
        intent_response.async_set_speech(final_content)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

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
        """Discover and index tools in the background vector store."""
        if runtime_data.tool_index_ready or runtime_data.tool_indexing_in_progress:
            return
        runtime_data.tool_indexing_in_progress = True

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
                )
            )
        else:
            runtime_data.tool_index_ready = True

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features.
        await hass.config_entries.async_reload(entry.entry_id)
