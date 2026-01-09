"""Conversation support for Home Generative Agent using langgraph."""

from __future__ import annotations

import difflib
import json
import logging
import re
import string
from typing import TYPE_CHECKING, Any, Literal, cast

import homeassistant.util.dt as dt_util
import yaml
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
CODE_FENCE_MIN_LINES = 3
YAML_LIST_INDENT = 2
YAML_NESTED_INDENT = 4
_ENTITY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_ENTITY_ID_TOKEN_PATTERN = re.compile(r"\b[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*\b")
_ENTITY_ID_MATCH_SCORE_MIN = 0.6
_ENTITY_ID_TOKEN_OVERLAP_WEIGHT = 0.2

## json - yaml conversion helpers ##
# This is needed because LLMs often output JSON inside Markdown code fences,
# and we want to convert that to properly indented YAML for Home Assistant automations.
# Additionally, we normalize common automation fields to be nested under trigger/action.
# The YAML indentation fixup ensures that lists under trigger/action/condition
# are indented properly for Home Assistant to parse them correctly.
# This code should be made into a tool at the expense of latency if we find
# that LLMs frequently output invalid YAML automations.


class _IndentDumper(yaml.SafeDumper):
    """Force YAML lists to indent under their parent key."""

    def increase_indent(
        self,
        flow: bool = False,  # noqa: FBT001, FBT002
        indentless: bool = False,  # noqa: ARG002, FBT001, FBT002
    ) -> Any:
        return super().increase_indent(flow, False)  # noqa: FBT003


def _strip_code_fence(content: str) -> str:
    """Remove a surrounding Markdown code fence if present."""
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < CODE_FENCE_MIN_LINES or not lines[0].startswith("```"):
        return stripped
    for idx in range(len(lines) - 1, 0, -1):
        if lines[idx].startswith("```"):
            return "\n".join(lines[1:idx]).strip()
    return stripped


def _extract_json_block(content: str) -> str | None:
    """Extract the first complete JSON object/array from content."""
    decoder = json.JSONDecoder()
    for idx, char in enumerate(content):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(content[idx:])
        except json.JSONDecodeError:
            continue
        return content[idx : idx + end]
    return None


def _load_json_payload(content: str) -> tuple[Any | None, str | None]:
    """Load JSON from content, optionally extracting a JSON block."""
    candidate = _strip_code_fence(content)
    if not candidate:
        return None, None

    try:
        return json.loads(candidate), candidate
    except json.JSONDecodeError as err:
        extracted = _extract_json_block(candidate)
        if extracted and extracted != candidate:
            try:
                return json.loads(extracted), extracted
            except json.JSONDecodeError:
                pass
        LOGGER.warning(
            "Schema-first JSON parsing failed: %s; content=%r", err, candidate[:500]
        )
        return None, candidate


def _resolve_entity_id(entity_id: str, hass: HomeAssistant) -> str:
    """Try to resolve a suggested entity_id to an existing entity_id."""
    if not _ENTITY_ID_PATTERN.match(entity_id):
        return entity_id
    if hass.states.get(entity_id):
        return entity_id

    domain, object_id = entity_id.split(".", 1)
    prefix = f"{domain}."
    candidates = [
        state.entity_id
        for state in hass.states.async_all()
        if state.entity_id.startswith(prefix)
    ]
    if not candidates:
        return entity_id

    def score_match(candidate: str) -> float:
        candidate_obj = candidate.split(".", 1)[1]
        ratio = difflib.SequenceMatcher(None, object_id, candidate_obj).ratio()
        target_tokens = {token for token in object_id.split("_") if token}
        candidate_tokens = {token for token in candidate_obj.split("_") if token}
        overlap = 0.0
        if target_tokens:
            overlap = len(target_tokens & candidate_tokens) / len(target_tokens)
        return ratio + (overlap * _ENTITY_ID_TOKEN_OVERLAP_WEIGHT)

    tokens = [token for token in object_id.split("_") if token]
    if tokens:
        token_matches = [
            candidate
            for candidate in candidates
            if all(token in candidate.split(".", 1)[1] for token in tokens)
        ]
        if token_matches:
            best_match = max(token_matches, key=score_match)
            if score_match(best_match) >= _ENTITY_ID_MATCH_SCORE_MIN:
                return best_match

    scored = max(candidates, key=score_match)
    if score_match(scored) >= _ENTITY_ID_MATCH_SCORE_MIN:
        return scored

    close = difflib.get_close_matches(
        entity_id, candidates, n=1, cutoff=_ENTITY_ID_MATCH_SCORE_MIN
    )
    return close[0] if close else entity_id


def _fix_dashboard_entities(payload: dict[str, Any], hass: HomeAssistant) -> bool:
    """Update DashboardSpec entity_ids when a close existing match is found."""
    if not isinstance(payload.get("views"), list):
        return False

    changed = False

    def update_entity(value: str) -> str:
        nonlocal changed
        resolved = _resolve_entity_id(value, hass)
        if resolved != value:
            LOGGER.debug("Resolved dashboard entity_id %s -> %s", value, resolved)
            changed = True
        return resolved

    def update_entity_rows(entities: list[Any]) -> None:
        for idx, entity in enumerate(entities):
            if isinstance(entity, str):
                entities[idx] = update_entity(entity)
                continue
            if isinstance(entity, dict):
                entity_id = entity.get("entity")
                if isinstance(entity_id, str):
                    entity["entity"] = update_entity(entity_id)

    def update_cards(cards: list[Any]) -> None:
        for card in cards:
            if not isinstance(card, dict):
                continue
            entity_id = card.get("entity")
            if isinstance(entity_id, str):
                card["entity"] = update_entity(entity_id)

            entities = card.get("entities")
            if isinstance(entities, list):
                update_entity_rows(entities)

            nested_cards = card.get("cards")
            if isinstance(nested_cards, list):
                update_cards(nested_cards)

    for view in payload.get("views", []):
        if not isinstance(view, dict):
            continue
        cards = view.get("cards")
        if isinstance(cards, list):
            update_cards(cards)

    return changed


def _maybe_fix_dashboard_entities(content: str, hass: HomeAssistant) -> str:
    payload, candidate = _load_json_payload(content)
    if payload is None or candidate is None:
        return content

    if not isinstance(payload, dict):
        return content
    if not _fix_dashboard_entities(payload, hass):
        return content

    return json.dumps(payload, ensure_ascii=True, separators=(",", ": "))


def _fix_entity_ids_in_text(content: str, hass: HomeAssistant) -> str:
    """Replace entity_id-like tokens in text with existing entity_ids when possible."""

    def replace(match: re.Match[str]) -> str:
        entity_id = match.group(0)
        resolved = _resolve_entity_id(entity_id, hass)
        if resolved != entity_id:
            LOGGER.debug("Resolved text entity_id %s -> %s", entity_id, resolved)
        return resolved

    return _ENTITY_ID_TOKEN_PATTERN.sub(replace, content)


def _is_dashboard_request(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return "dashboard" in lowered or "lovelace" in lowered


def _convert_schema_json_to_yaml(  # noqa: PLR0911, PLR0912
    content: str,
    enabled: bool,  # noqa: FBT001
) -> str:
    """Convert schema-first JSON to YAML when enabled; otherwise passthrough."""
    if not enabled:
        return content
    if "```yaml" in content:
        inner = _strip_code_fence(content)
        fixed = _fix_automation_yaml_indentation(inner)
        return f"```yaml\n{fixed}\n```"
    candidate = _strip_code_fence(content)
    if not candidate:
        return content
    payload: Any | None = None
    if candidate[0] in "{[":
        payload, _ = _load_json_payload(candidate)
        if payload is None:
            LOGGER.warning("Schema-first JSON parsing failed; trying YAML fallback.")
            try:
                payload = yaml.safe_load(candidate)
            except yaml.YAMLError as err:
                LOGGER.warning("Schema-first YAML fallback failed: %s", err)
                payload = None
            if payload is None or isinstance(payload, str):
                return (
                    "Schema-first JSON parsing failed. "
                    "Please respond with valid JSON only."
                )
    else:
        try:
            payload = yaml.safe_load(candidate)
        except yaml.YAMLError:
            payload = None
        if payload is None or isinstance(payload, str):
            return content
    if isinstance(payload, dict) and "yaml" in payload:
        raw_yaml = payload.get("yaml")
        if isinstance(raw_yaml, str):
            raw_yaml = raw_yaml.replace("\\n", "\n").strip()
            try:
                yaml_payload = yaml.safe_load(raw_yaml)
            except yaml.YAMLError:
                yaml_payload = None
            if yaml_payload is not None and not isinstance(yaml_payload, str):
                yaml_payload = _normalize_automation_payload(yaml_payload)
                yaml_text = cast("Any", yaml.dump)(
                    yaml_payload,
                    sort_keys=False,
                    default_flow_style=False,
                    indent=2,
                    Dumper=_IndentDumper,
                ).rstrip()
                yaml_text = _fix_automation_yaml_indentation(yaml_text)
                return f"```yaml\n{yaml_text}\n```"
            return f"```yaml\n{raw_yaml}\n```"
    payload = _normalize_automation_payload(payload)
    yaml_text = cast("Any", yaml.dump)(
        payload,
        sort_keys=False,
        default_flow_style=False,
        indent=2,
        Dumper=_IndentDumper,
    ).rstrip()
    yaml_text = _fix_automation_yaml_indentation(yaml_text)
    return f"```yaml\n{yaml_text}\n```"


def _normalize_automation_payload(payload: Any) -> Any:  # noqa: PLR0912
    """Heuristically nest common automation fields under trigger/action."""
    if isinstance(payload, list) and payload:
        payload[0] = _normalize_automation_payload(payload[0])
        return payload
    if not isinstance(payload, dict):
        return payload
    if "trigger" in payload:
        if isinstance(payload["trigger"], dict):
            payload["trigger"] = [payload["trigger"]]
        if isinstance(payload.get("trigger"), list):
            trigger_items = payload["trigger"]
            if trigger_items:
                first = trigger_items[0]
                if isinstance(first, dict):
                    for key in ("entity_id", "to", "from", "for", "attribute"):
                        if key in payload and key not in first:
                            first[key] = payload.pop(key)
    if "action" in payload:
        if isinstance(payload["action"], dict):
            payload["action"] = [payload["action"]]
        if isinstance(payload.get("action"), list):
            action_items = payload["action"]
            if action_items:
                first = action_items[0]
                if isinstance(first, dict):
                    for key in ("target", "data", "data_template"):
                        if key in payload and key not in first:
                            first[key] = payload.pop(key)
                    if "alias" in first and "alias" not in payload:
                        payload["alias"] = first.pop("alias")
    if "condition" in payload and isinstance(payload["condition"], dict):
        payload["condition"] = [payload["condition"]]
    return _reorder_automation_payload(payload)


def _reorder_automation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Put common automation keys in a stable, HA-friendly order."""
    preferred = [
        "alias",
        "description",
        "trigger",
        "condition",
        "action",
        "mode",
        "max",
        "id",
    ]
    reordered: dict[str, Any] = {}
    for key in preferred:
        if key in payload:
            reordered[key] = payload[key]
    for key, value in payload.items():
        if key not in reordered:
            reordered[key] = value
    return reordered


def _fix_automation_yaml_indentation(yaml_text: str) -> str:
    """Ensure trigger/action/condition list items are indented under their keys."""
    if "trigger:" not in yaml_text and "action:" not in yaml_text:
        return yaml_text
    lines = yaml_text.splitlines()
    out: list[str] = []
    in_block: str | None = None
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent == 0 and stripped.endswith(":"):
            key = stripped[:-1]
            in_block = key if key in {"trigger", "action", "condition"} else None
            out.append(line)
            continue
        if in_block:
            if stripped.startswith("- ") and indent < YAML_LIST_INDENT:
                out.append((" " * YAML_LIST_INDENT) + stripped)
                continue
            if (
                stripped
                and indent < YAML_NESTED_INDENT
                and not stripped.startswith("- ")
            ):
                out.append((" " * YAML_NESTED_INDENT) + stripped)
                continue
        if indent == 0 and stripped and not stripped.startswith("- "):
            in_block = None
        out.append(line)
    return "\n".join(out)


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
                response=intent_response, conversation_id=conversation_id
            )

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

        tools = [
            _format_tool(tool, llm_api.custom_serializer) for tool in llm_api.tools
        ]

        # Add LangChain-native tools (wired in graph via config).
        langchain_tools: dict[str, Any] = {
            "get_and_analyze_camera_image": get_and_analyze_camera_image,
            "upsert_memory": upsert_memory,
            "get_entity_history": get_entity_history,
            "confirm_sensitive_action": confirm_sensitive_action,
            "alarm_control": alarm_control,
            "write_yaml_file": write_yaml_file,
        }
        if not options.get(CONF_SCHEMA_FIRST_YAML, False):
            langchain_tools["add_automation"] = add_automation
        tools.extend(langchain_tools.values())

        # Conversation ID
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
        final_content = response["messages"][-1].content
        if isinstance(final_content, str):
            if options.get(CONF_SCHEMA_FIRST_YAML, False):
                final_content = _maybe_fix_dashboard_entities(final_content, hass)
            else:
                final_content = _fix_entity_ids_in_text(final_content, hass)
            final_content = _convert_schema_json_to_yaml(
                final_content, options.get(CONF_SCHEMA_FIRST_YAML, False)
            )
            LOGGER.debug("Final response content: %s", final_content)
        intent_response.async_set_speech(final_content)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features.
        await hass.config_entries.async_reload(entry.entry_id)
