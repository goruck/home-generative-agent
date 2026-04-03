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
from voluptuous_openapi import convert

from .agent.graph import workflow
from .agent.rag_embedding_text import strip_for_embedding
from .agent.tools import (
    add_automation,
    alarm_control,
    confirm_sensitive_action,
    get_and_analyze_camera_image,
    get_available_tools,
    get_camera_last_events,
    get_entity_history,
    resolve_entity_ids,
    upsert_memory,
    write_yaml_file,
)
from .const import (
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_DEBUG_ASSIST_TRACE,
    CONF_INSTRUCTIONS_CONFIG,
    CONF_PROMPT,
    CONF_SCHEMA_FIRST_YAML,
    CRITICAL_ACTION_PROMPT,
    DOMAIN,
    GRAPH_CFG_HA_TOOL_INTENT_RESPONSES,
    LANGCHAIN_LOGGING_LEVEL,
    SCHEMA_FIRST_YAML_PROMPT,
    SUBENTRY_TYPE_MODEL_PROVIDER,
    TOOL_CALL_ERROR_SYSTEM_MESSAGE,
)
from .core.assist_chat_log import append_langgraph_turn_to_chat_log
from .core.assist_reasoning_trace import build_assist_reasoning_trace
from .core.conversation_helpers import (
    _convert_schema_json_to_yaml,
    _fix_entity_ids_in_text,
    _is_dashboard_request,
    _maybe_fix_dashboard_entities,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from agent.graph import State
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddEntitiesCallback
    from langchain_core.runnables import RunnableConfig

    from .core.runtime import HGAConfigEntry

_LOGGER = logging.getLogger(__name__)


def _apply_query_answer_when_no_ha_intent(
    result: conversation.ConversationResult,
    *,
    ha_had_intent_response_from_tool: bool,
) -> None:
    """
    Align ``response_type`` with HA defaults for pure Q&A turns.

    When no Home Assistant LLM tool returned ``IntentResponseDict``, core would leave
    the default ``ACTION_DONE``, which mislabels informational answers.

    Note: the Android companion Assist flow (``AssistViewModelBase``) does not branch
    on ``response_type`` today; this still corrects ``intent_output`` for the WebSocket
    API, automations, other clients, and possible future app behavior.
    """
    if ha_had_intent_response_from_tool:
        return
    if result.response.response_type == intent.IntentResponseType.ERROR:
        return
    result.response.response_type = intent.IntentResponseType.QUERY_ANSWER


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
        _LOGGER.warning("Content is None, returning empty message")
        return HumanMessage(content="")
    if isinstance(content, conversation.UserContent):
        return HumanMessage(content=content.content)
    return AIMessage(content=content.content)


class MultiLLMAPI:
    """Wrapper to route tool calls across multiple LLM APIs."""

    def __init__(self, apis: list[llm.APIInstance]) -> None:
        """Initialize the wrapper."""
        self.apis = apis

    @property
    def api_prompt(self) -> str:
        """Return the concatenated API prompts."""
        return "\n".join(api.api_prompt for api in self.apis if api.api_prompt)

    @property
    def custom_serializer(self) -> Callable[[Any], Any] | None:
        """Return the custom serializer from the first API."""
        return self.apis[0].custom_serializer if self.apis else None

    async def async_call_tool(self, tool_input: llm.ToolInput) -> Any:
        """Try calling the tool across all APIs."""
        last_err: Exception | None = None
        for api in self.apis:
            try:
                return await api.async_call_tool(tool_input)
            except (HomeAssistantError, vol.Invalid) as err:
                last_err = err

        if last_err:
            raise last_err

        msg = f"No APIs available to handle tool {tool_input.tool_name}"
        raise HomeAssistantError(msg)


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

    async def _async_handle_message(  # noqa: C901, PLR0912, PLR0915
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

        # Fetch Tool Manager Config
        from .const import SUBENTRY_TYPE_TOOL_MANAGER  # noqa: PLC0415

        tool_mgr_subentry = next(
            (
                s
                for s in self.entry.subentries.values()
                if s.subentry_type == SUBENTRY_TYPE_TOOL_MANAGER
            ),
            None,
        )
        tool_mgr_data = tool_mgr_subentry.data if tool_mgr_subentry else {}
        providers_cfg = tool_mgr_data.get("tool_providers", {})
        tools_cfg = tool_mgr_data.get("tools", {})
        instructions_cfg = tool_mgr_data.get(CONF_INSTRUCTIONS_CONFIG, {})

        # HA tools
        llm_apis: list[llm.APIInstance] = []
        for api in llm.async_get_apis(hass):
            p_cfg = providers_cfg.get(api.id, {})
            if not p_cfg.get("enabled", True):
                continue
            try:
                inst = await llm.async_get_api(hass, api.id, llm_context)
                llm_apis.append(inst)
            except HomeAssistantError:
                _LOGGER.warning("Could not load LLM API %s", api.id)

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

        custom_serializer = llm_apis[0].custom_serializer if llm_apis else None
        tools = []
        for api in llm_apis:
            for tool in api.tools:
                t_cfg = tools_cfg.get(tool.name, {})
                if t_cfg.get("enabled", True):
                    tools.append(_format_tool(tool, custom_serializer))

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
            "get_available_tools": get_available_tools,
        }
        if not options.get(CONF_SCHEMA_FIRST_YAML, False):
            langchain_tools["add_automation"] = add_automation

        if providers_cfg.get("langchain_internal", {}).get("enabled", True):
            for tv in langchain_tools.values():
                t_cfg = tools_cfg.get(tv.name, {})
                if t_cfg.get("enabled", True):
                    tools.append(tv)
        else:
            langchain_tools = {}

        # Version hash and PgVector indexing for RAG tool retrieval
        tools_dict_list = []
        for t in tools:
            if isinstance(t, dict):
                tools_dict_list.append(t)
            else:
                tools_dict_list.append({"name": t.name, "description": t.description})

        tools_json = json.dumps(tools_dict_list, sort_keys=True, default=str)
        instructions_json = json.dumps(instructions_cfg, sort_keys=True, default=str)
        # Force re-index by prefixing with version (content field fix)
        # Include instructions in the hash so they trigger re-index too.
        tools_hash = hashlib.sha256(
            f"v4:{tools_json}:{instructions_json}".encode()
        ).hexdigest()

        if runtime_data.tools_version_hash != tools_hash:
            _LOGGER.debug(
                "Tool definitions changed (or first run). Indexing into vector store."
            )
            store = runtime_data.store
            for i, td in enumerate(tools_dict_list):
                name = (
                    td.get("name") or td.get("function", {}).get("name") or td.get("id")
                )
                description = td.get("description") or td.get("function", {}).get(
                    "description"
                )

                if not name:
                    name = f"unnamed_tool_{i}"
                    _LOGGER.warning(
                        (
                            "Tool at index %d is missing a name (using fallback '%s'). "
                            "Full tool dict: %s"
                        ),
                        i,
                        name,
                        td,
                    )

                # Build the value to be indexed. We MUST use the "content" field because
                # the store is configured to only embed that field in __init__.py.
                content = description or ""
                if not name.startswith("unnamed_tool_"):
                    # For real tools, include the name in the searchable content too.
                    content = f"{name}: {content}"

                t_cfg = tools_cfg.get(name, {})
                custom_tags = t_cfg.get("tags", "")
                if custom_tags:
                    content = f"{content}\nKeywords/Tags: {custom_tags}"

                index_value = {
                    "content": content,
                    "name": name,
                    "description": description or "",
                }

                # We do this asynchronously but concurrently to speed it up.
                hass.async_create_task(
                    store.aput(("system", "tools"), key=name, value=index_value)
                )

            # Index active instructions
            for i_name, i_cfg in instructions_cfg.items():
                if not i_cfg.get("enabled", True):
                    continue

                i_prompt = i_cfg.get("prompt", "")
                i_tags = i_cfg.get("tags", "")

                i_content = f"Instruction: {i_name}"
                if i_tags:
                    i_content = f"{i_content} (Tags: {i_tags})"

                hass.async_create_task(
                    store.aput(
                        ("system", "instructions"),
                        key=i_name,
                        value={
                            "content": i_content,
                            "name": i_name,
                            "prompt": i_prompt,
                        },
                    )
                )

                stripped = strip_for_embedding(i_prompt)
                body_content = (
                    f"Instruction: {i_name}\n{stripped}"
                    if stripped
                    else f"Instruction: {i_name}"
                )
                hass.async_create_task(
                    store.aput(
                        ("system", "instructions_body"),
                        key=i_name,
                        value={
                            "content": body_content,
                            "name": i_name,
                            "prompt": i_prompt,
                        },
                    )
                )

            runtime_data.tools_version_hash = tools_hash

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
            pin_enabled = options.get(CONF_CRITICAL_ACTION_PIN_ENABLED, True)
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

        api_prompt = "\n".join(api.api_prompt for api in llm_apis if api.api_prompt)
        if api_prompt:
            prompt_parts.append(api_prompt)

        prompt = "\n".join(prompt_parts)

        # Use the already-configured chat model from __init__.py
        base_llm = runtime_data.chat_model

        # Tools are no longer bound here. We pass available_tools to the config and bind
        # dynamically via RAG. We still verify the model supports tools; see __init__
        # checks, or bind an empty list to test if needed.
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

        ha_tool_intent_responses: dict[str, intent.IntentResponse] = {}

        app_config: RunnableConfig = {
            "configurable": {
                "thread_id": conversation_id,
                "user_id": user_name,
                "chat_model": base_llm,
                "available_tools": tools,
                "chat_model_options": runtime_data.chat_model_options,
                "prompt": prompt,
                "options": options,
                "vlm_model": runtime_data.vision_model,
                "summarization_model": runtime_data.summarization_model,
                "langchain_tools": langchain_tools,
                "ha_llm_api": MultiLLMAPI(llm_apis) if llm_apis else None,
                "hass": hass,
                "pending_actions": runtime_data.pending_actions,
                "tool_mgr_data": tool_mgr_data,
                GRAPH_CFG_HA_TOOL_INTENT_RESPONSES: ha_tool_intent_responses,
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
            "selected_instructions": [],
            "redacted_thinking_chunks": [],
        }
        input_message_count = len(messages)

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
        reasoning_plain = build_assist_reasoning_trace(response)
        final_str = (
            final_content if isinstance(final_content, str) else str(final_content)
        )
        has_native_thinking = bool(response.get("redacted_thinking_chunks"))
        await append_langgraph_turn_to_chat_log(
            chat_log,
            user_input.agent_id,
            response["messages"],
            input_message_count=input_message_count,
            reasoning_plain=reasoning_plain,
            has_native_thinking=has_native_thinking,
            debug_assist_trace=bool(options.get(CONF_DEBUG_ASSIST_TRACE, False)),
            final_spoken_text=final_str,
            ha_tool_intent_responses=ha_tool_intent_responses,
        )
        conv_result = conversation.async_get_result_from_chat_log(user_input, chat_log)
        _apply_query_answer_when_no_ha_intent(
            conv_result,
            ha_had_intent_response_from_tool=bool(ha_tool_intent_responses),
        )
        return conv_result

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features.
        await hass.config_entries.async_reload(entry.entry_id)
