"""Langgraph graphs for Home Generative Agent."""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import voluptuous as vol
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.util import dt as dt_util
from homeassistant.util import ulid
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
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
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
    CONF_CRITICAL_ACTIONS,
    CONF_GEMINI_CHAT_MODEL,
    CONF_MANAGE_CONTEXT_WITH_TOKENS,
    CONF_MAX_MESSAGES_IN_CONTEXT,
    CONF_MAX_TOKENS_IN_CONTEXT,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OPENAI_CHAT_MODEL,
    EMBEDDING_MODEL_PROMPT_TEMPLATE,
    RECOMMENDED_CRITICAL_ACTIONS,
    SUMMARIZATION_INITIAL_PROMPT,
    SUMMARIZATION_PROMPT_TEMPLATE,
    SUMMARIZATION_SYSTEM_PROMPT,
    TOOL_CALL_ERROR_TEMPLATE,
)
from ..core.utils import extract_final  # noqa: TID252
from .helpers import (
    maybe_fill_lock_entity,
    normalize_intent_for_alarm,
    normalize_intent_for_lock,
    sanitize_tool_args,
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


# ----- Tool helpers -----


@dataclass
class ToolExecutionContext:
    """Reusable execution context for tool handlers."""

    hass: HomeAssistant
    store: BaseStore
    config: RunnableConfig
    ha_llm_api: Any
    critical_actions: list[dict[str, str]]
    pin_enabled: bool
    pin_hash: str | None
    pin_salt: str | None
    pending_actions: dict[str, Any]
    state: State


def _make_tool_error(err: str, name: str, tid: str) -> ToolMessage:
    """Build a standardized error ToolMessage."""
    return ToolMessage(
        content=TOOL_CALL_ERROR_TEMPLATE.format(error=err),
        name=name,
        tool_call_id=tid,
        status="error",
    )


async def _precheck_alarm_open_entries(
    tool_name: str,
    tool_args: dict[str, Any],
    ctx: ToolExecutionContext,
    tool_call: dict[str, Any],
) -> ToolMessage | None:
    """Gate alarm arming when entries are open."""
    if tool_name != "alarm_control":
        return None

    desired_state = str(tool_args.get("state") or "arm_home").lower()
    is_disarm = desired_state.startswith("disarm") or desired_state in {
        "off",
        "disarmed",
    }
    if is_disarm:
        return None

    open_entries = await _find_open_entries(ctx.hass, ctx.state["messages"])
    if not open_entries:
        return None

    consent_ok = False
    consent_records = await ctx.store.asearch(
        ("alarm_control", "open_consent"), limit=1
    )
    if consent_records:
        rec = consent_records[0].value
        if rec.get("open_entries") == open_entries:
            consent_ok = True
    if not consent_ok:
        # If the last user message mentions a bypass, honor it.
        for msg in reversed(ctx.state["messages"]):
            if isinstance(msg, HumanMessage):
                text = str(msg.content).lower()
                if "bypass" in text:
                    consent_ok = True
                break
    if not consent_ok:
        await ctx.store.aput(
            namespace=("alarm_control",),
            key="open_consent",
            value={"open_entries": open_entries, "ts": dt_util.utcnow().isoformat()},
        )
        reason = (
            "These entry sensors are open: "
            f"{', '.join(open_entries)}. "
            "If they are already bypassed at the panel you can proceed; "
            "otherwise close or bypass them before arming. "
            "No critical-action PIN is required—just confirm."
        )
        return ToolMessage(
            content=json.dumps(
                {
                    "status": "error",
                    "reason": reason,
                    "open_entries": open_entries,
                    "action": "confirm_proceed_with_open_entries",
                    "pin_required": False,
                }
            ),
            name=tool_name,
            tool_call_id=tool_call.get("id") or "",
            status="error",
        )

    # Remember consent for this open set for the session.
    await ctx.store.aput(
        namespace=("alarm_control",),
        key="open_consent",
        value={"open_entries": open_entries, "ts": dt_util.utcnow().isoformat()},
    )
    return None


async def _run_langchain_tool(
    tool_name: str,
    tool_call: dict[str, Any],
    langchain_tools: dict[str, Any],
    store: BaseStore,
    config: RunnableConfig,
) -> ToolMessage:
    """Invoke a LangChain tool and normalize its response."""
    lc_tool = langchain_tools[tool_name.lower()]
    tool_call_copy = copy.deepcopy(tool_call)
    tool_call_copy["args"].update({"store": store, "config": config})

    def _stringify(val: Any) -> str:
        if isinstance(val, str):
            return val
        try:
            return json.dumps(val)
        except TypeError:
            return str(val)

    try:
        lc_result = await lc_tool.ainvoke(tool_call_copy)
        if isinstance(lc_result, ToolMessage):
            status = lc_result.status
            result_str = _stringify(lc_result.content)
        elif isinstance(lc_result, BaseMessage):
            raw_content = lc_result.content
            status = None
            result_str = _stringify(raw_content)
        elif isinstance(lc_result, str):
            status = None
            result_str = lc_result
        else:
            status = None
            result_str = _stringify(lc_result)

        if tool_name == "alarm_control":
            try:
                parsed = json.loads(result_str)
                if isinstance(parsed, dict):
                    warning = parsed.get("warning")
                    success_val = bool(parsed.get("success", True))
                    if warning or not success_val:
                        status = "error"
                        if parsed.get("result_text"):
                            result_str = parsed["result_text"]
                        elif warning:
                            result_str = str(warning)
            except json.JSONDecodeError:
                status = None

        return ToolMessage(
            content=result_str,
            name=tool_name,
            tool_call_id=tool_call.get("id") or "",
            status=status,
        )
    except (HomeAssistantError, ValidationError) as err:
        status_msg = (
            f"Alarm control failed: {err}"
            if tool_name == "alarm_control"
            else repr(err)
        )
        return _make_tool_error(status_msg, tool_name, tool_call.get("id") or "")


def _route_alarm_intent_to_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    hass: HomeAssistant,
    tool_call: dict[str, Any],
) -> ToolMessage | None:
    """Redirect alarm-like HA intents to the dedicated alarm tool."""
    domains = tool_args.get("domain") or []
    domains = domains if isinstance(domains, list) else [domains]
    is_alarm = any(str(d).lower() == "alarm_control_panel" for d in domains)
    entity_ids = _coerce_entity_ids(tool_args)
    has_alarm_entities = bool(hass.states.async_entity_ids("alarm_control_panel"))
    name_hint = str(tool_args.get("name", "")).lower()
    looks_like_alarm = is_alarm or any(
        "alarm_control_panel" in str(e).lower() for e in entity_ids
    )
    if not looks_like_alarm and has_alarm_entities:
        if any(k in name_hint for k in ("alarm", "security")):
            looks_like_alarm = True
        if str(tool_args.get("code", "")).strip().isdigit():
            looks_like_alarm = True
    if not looks_like_alarm:
        return None
    return ToolMessage(
        content=(
            "Alarm panels must be controlled with the `alarm_control` tool. "
            "Call `alarm_control` with the alarm code and desired state "
            "(for example, state='arm_home' or 'disarm')."
        ),
        name=tool_name,
        tool_call_id=tool_call.get("id") or "",
        status="error",
    )


def _critical_action_guard(
    tool_name: str,
    tool_args: dict[str, Any],
    ctx: ToolExecutionContext,
    tool_call: dict[str, Any],
) -> ToolMessage | None:
    """Enforce critical-action PIN flow when needed."""
    if not ctx.pin_enabled:
        return None
    if not _is_critical_action(tool_args, ctx.critical_actions, tool_name):
        return None
    if not ctx.pin_hash or not ctx.pin_salt:
        return _make_tool_error(
            (
                "Critical action requires a configured PIN. "
                "No action was queued. Set a PIN in the integration options and retry."
            ),
            tool_name,
            tool_call.get("id") or "",
        )

    action_id = ulid.ulid_now()
    ctx.pending_actions[action_id] = {
        "tool_name": tool_name,
        "tool_args": tool_args,
        "created_at": dt_util.utcnow().isoformat(),
        "user": ctx.config.get("configurable", {}).get("user_id"),
        "attempts": 0,
    }
    return ToolMessage(
        content=json.dumps(
            {
                "status": "requires_pin",
                "action_id": action_id,
                "reason": "Critical action requires PIN confirmation.",
            }
        ),
        tool_call_id=tool_call.get("id"),
        name=tool_name,
    )


async def _run_ha_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    ctx: ToolExecutionContext,
    tool_call: dict[str, Any],
) -> ToolMessage:
    """Normalize arguments and execute an HA tool call."""
    redirect, prepared_args = _prepare_tool_args(tool_name, tool_args, ctx, tool_call)
    if redirect:
        return redirect

    tool_input = llm.ToolInput(tool_name=tool_name, tool_args=prepared_args)
    try:
        response = await ctx.ha_llm_api.async_call_tool(tool_input)
    except (HomeAssistantError, vol.Invalid) as err:
        return _make_tool_error(repr(err), tool_name, tool_call.get("id") or "")

    content_str, status = _parse_tool_response(response, prepared_args)
    return ToolMessage(
        content=content_str,
        tool_call_id=tool_call.get("id"),
        name=tool_name,
        status=status or "success",
    )


def _prepare_tool_args(
    tool_name: str,
    tool_args: dict[str, Any],
    ctx: ToolExecutionContext,
    tool_call: dict[str, Any],
) -> tuple[ToolMessage | None, dict[str, Any]]:
    """Normalize, guard, and sanitize arguments before calling HA."""
    tool_args = normalize_intent_for_alarm(tool_name, tool_args)
    tool_args = normalize_intent_for_lock(tool_name, tool_args)
    tool_args = maybe_fill_lock_entity(tool_args, ctx.hass)
    tool_args = sanitize_tool_args(tool_args)

    # Normalize domains to a clean list for downstream consumers.
    domains = tool_args.get("domain") or []
    if isinstance(domains, str):
        domains = [domains]
    tool_args["domain"] = [str(d).strip() for d in domains if str(d).strip()]

    redirect = _route_alarm_intent_to_tool(tool_name, tool_args, ctx.hass, tool_call)
    if redirect:
        return redirect, tool_args

    critical_guard = _critical_action_guard(
        tool_name=tool_name,
        tool_args=tool_args,
        ctx=ctx,
        tool_call=tool_call,
    )
    if critical_guard:
        return critical_guard, tool_args

    # Reduce payload to the minimal arguments HA expects (avoids extra-key errors).
    tool_args = _minimal_payload_for_domain(tool_args)
    tool_args = sanitize_tool_args(tool_args)
    return None, tool_args


def _parse_tool_response(
    response: Any, tool_args: dict[str, Any]
) -> tuple[str, str | None]:
    """Return (content, status) from a tool response."""
    parsed: dict[str, Any] | None = None
    if isinstance(response, dict):
        parsed = response
    elif isinstance(response, str):
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            parsed = None

    status: str | None = None
    content_str = json.dumps(response)
    if parsed is None:
        return content_str, status

    warning = parsed.get("warning")
    success_val = parsed.get("success", True)
    result_text = (
        parsed.get("result_text") or parsed.get("response") or parsed.get("result")
    )
    inferred_state = parsed.get("inferred_state")

    domains = tool_args.get("domain") or []
    domains = domains if isinstance(domains, list) else [domains]
    is_alarm = any(str(d).lower() == "alarm_control_panel" for d in domains)
    service = str(tool_args.get("service") or "").lower()

    if warning:
        status = "error"
        content_str = str(result_text or warning)
    elif success_val is False:
        status = "error"
        content_str = str(result_text or "Action failed.")
    elif is_alarm:
        if "arm" in service and inferred_state not in {
            "armed",
            "armed_home",
            "armed_away",
            "armed_night",
            "arming",
        }:
            status = "error"
            content_str = str(
                result_text
                or f"Alarm did not change state (panel_state={inferred_state})."
            )
        if "disarm" in service and inferred_state not in {
            "disarmed",
            "disarming",
        }:
            status = "error"
            content_str = str(
                result_text
                or f"Alarm did not change state (panel_state={inferred_state})."
            )

    return content_str, status


# ----- Alarm helpers -----


def _extract_last_live_context(messages: list[AnyMessage]) -> str | None:
    """Return the most recent GetLiveContext payload content, if any."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "GetLiveContext":
            raw_content = msg.content
            if not isinstance(raw_content, str):
                continue
            try:
                data = json.loads(raw_content)
                if isinstance(data, dict) and "result" in data:
                    return str(data["result"])
            except json.JSONDecodeError:
                pass
            return raw_content
    return None


def _parse_open_entries_from_live_context(raw: str) -> list[str]:
    """Best-effort parse of open entry sensors from a Live Context string."""
    open_entries: list[str] = []
    current_name: str | None = None
    is_on = False
    is_opening = False

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # New entry block.
        if stripped.startswith("- names:"):
            if current_name and is_on and is_opening:
                open_entries.append(current_name)
            current_name = stripped.split(":", maxsplit=1)[1].strip().strip("'\"")
            is_on = False
            is_opening = False
            continue
        if stripped.startswith("state:"):
            if re.search(r"['\"]?on['\"]?", stripped, re.IGNORECASE):
                is_on = True
            continue
        if "device_class" in stripped and "opening" in stripped:
            is_opening = True
            continue

    if current_name and is_on and is_opening:
        open_entries.append(current_name)
    return open_entries


async def _find_open_entries(
    hass: HomeAssistant, messages: list[AnyMessage]
) -> list[str]:
    """Find open doors/windows from recent context, falling back to live state."""
    # First, try the most recent live context already in memory.
    if (raw_context := _extract_last_live_context(messages)) is not None:
        entries = _parse_open_entries_from_live_context(raw_context)
        if entries:
            return entries

    # Fallback to current HA state for entry sensors.
    open_entries: list[str] = []
    for entity_id in hass.states.async_entity_ids("binary_sensor"):
        state_obj = hass.states.get(entity_id)
        if not state_obj:
            continue
        if state_obj.attributes.get("device_class") != "opening":
            continue
        if str(state_obj.state).lower() not in {"on", "open", "opening"}:
            continue
        friendly = state_obj.attributes.get("friendly_name") or entity_id
        open_entries.append(str(friendly))
    return open_entries


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
    provider = opts.get(CONF_CHAT_MODEL_PROVIDER)
    model_name = _determine_model_name(provider, opts)
    manage_context_with_tokens: bool = (
        opts.get(CONF_MANAGE_CONTEXT_WITH_TOKENS) == "true"
    )
    context_max_messages: int = opts.get(CONF_MAX_MESSAGES_IN_CONTEXT)
    context_max_tokens: int = opts.get(CONF_MAX_TOKENS_IN_CONTEXT)

    if manage_context_with_tokens:
        max_tokens = context_max_tokens
        token_counter = partial(
            count_tokens_cross_provider,
            model=model_name,
            provider=provider,
            options=opts,
            chat_model_options=chat_model_options,
        )
    else:
        max_tokens = context_max_messages
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

    response = extract_final(getattr(raw_response, "content", "") or "")

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
    raw_response = await model.ainvoke(messages)
    LOGGER.debug("Raw summary response: %s", raw_response)

    response = extract_final(getattr(raw_response, "content", "") or "")

    return {
        "summary": response,
        "messages": [
            RemoveMessage(id=m.id) for m in msgs_to_remove if m.id is not None
        ],
    }


def _coerce_entity_ids(args: dict[str, Any]) -> list[str]:
    """Return a list of entity ids from common arg fields."""
    entity_ids: list[str] = []
    for key in ("entity_id", "entity_ids"):
        if key not in args:
            continue
        val = args[key]
        if isinstance(val, str):
            entity_ids.append(val)
        elif isinstance(val, list):
            entity_ids.extend(str(v) for v in val)
    return entity_ids


def _is_critical_action(
    tool_args: dict[str, Any],
    critical_actions: list[dict[str, str]],
    tool_name: str,
) -> bool:
    """Return True if the call matches a critical domain/service/entity rule."""
    domain_val = tool_args.get("domain", "")
    if isinstance(domain_val, list):
        domain_val = domain_val[0] if domain_val else ""
    domain = str(domain_val).lower()

    service = str(tool_args.get("service", "") or "").lower()
    entities = [e.lower() for e in _coerce_entity_ids(tool_args)]
    if not domain and entities:
        try:
            domain = entities[0].split(".", maxsplit=1)[0]
        except IndexError:
            domain = ""

    # Treat HA intent tools on locks as critical even without a service arg.
    if tool_name in {"HassTurnOn", "HassTurnOff"}:
        domains = tool_args.get("domain") or []
        domains = domains if isinstance(domains, list) else [domains]
        if any(str(d).lower() == "lock" for d in domains):
            return True
        if any(str(d).lower() == "alarm_control_panel" for d in domains):
            return False  # alarm panels use their own code, not the critical PIN

    # Alarm panels already enforce their own code; never gate them with the generic PIN.
    if domain == "alarm_control_panel":
        return False

    # Require both domain and service to be present before treating as critical.
    if not domain or not service:
        return False

    for rule in critical_actions:
        rule_domain = rule.get("domain", "").lower()
        rule_service = rule.get("service", "").lower()
        entity_match = rule.get("entity_match", "").lower()

        if rule_domain and rule_domain != domain:
            continue
        if rule_service and rule_service != service:
            continue
        if entity_match and not any(entity_match in e for e in entities):
            continue
        return True
    return False


def _minimal_payload_for_domain(tool_args: dict[str, Any]) -> dict[str, Any]:
    """Return a reduced payload for HA services to avoid extra keys."""
    domains = tool_args.get("domain") or []
    domains = domains if isinstance(domains, list) else [domains]
    domain = str(domains[0]).lower() if domains else ""

    def slugify(name: str) -> str:
        return str(name).strip().lower().replace(" ", "_")

    if domain == "lock":
        entity_id = tool_args.get("entity_id")
        if not entity_id and (name := tool_args.get("name")):
            entity_id = f"lock.{slugify(str(name))}"
        payload: dict[str, Any] = {
            "domain": ["lock"],
            "service": tool_args.get("service"),
        }
        if entity_id:
            payload["entity_id"] = entity_id
        if code := tool_args.get("code"):
            payload["code"] = code
        return payload

    if domain == "alarm_control_panel":
        payload = {
            "domain": ["alarm_control_panel"],
            "service": tool_args.get("service"),
        }
        if entity_id := tool_args.get("entity_id"):
            payload["entity_id"] = entity_id
        if code := tool_args.get("code"):
            payload["code"] = code
        if state := tool_args.get("state"):
            payload["state"] = state
        return payload

    # Default: return sanitized args.
    return tool_args


async def _call_tools(
    state: State, config: RunnableConfig, *, store: BaseStore
) -> dict[str, list[ToolMessage]]:
    """Call Home Assistant or LangChain tools requested by the model."""
    if "configurable" not in config:
        msg = "Configuration is missing."
        raise HomeAssistantError(msg)

    langchain_tools = config["configurable"]["langchain_tools"]
    ha_llm_api = config["configurable"]["ha_llm_api"]
    options = config["configurable"]["options"]
    hass = config["configurable"]["hass"]
    critical_actions: list[dict[str, str]] = (
        options.get(CONF_CRITICAL_ACTIONS) or RECOMMENDED_CRITICAL_ACTIONS
    )
    pin_hash = options.get(CONF_CRITICAL_ACTION_PIN_HASH)
    pin_salt = options.get(CONF_CRITICAL_ACTION_PIN_SALT)
    pin_configured = bool(pin_hash and pin_salt)
    # Always respect a configured PIN, even if the toggle somehow reads False.
    pin_enabled = bool(
        options.get(CONF_CRITICAL_ACTION_PIN_ENABLED, True) or pin_configured
    )
    pending_actions = config["configurable"].get("pending_actions", {})

    # Expect tool calls in the last AIMessage.
    if not state["messages"] or not isinstance(state["messages"][-1], AIMessage):
        msg = "No tool calls found in the last message."
        raise HomeAssistantError(msg)

    raw_tool_calls = getattr(state["messages"][-1], "tool_calls", None)
    tool_calls: list[dict[str, Any]] = (
        raw_tool_calls if isinstance(raw_tool_calls, list) else []
    )
    tool_responses: list[ToolMessage] = []

    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {}) or {}
        LOGGER.debug("Tool call: %s(%s)", tool_name, tool_args)

        ctx = ToolExecutionContext(
            hass=hass,
            store=store,
            config=config,
            ha_llm_api=ha_llm_api,
            critical_actions=critical_actions,
            pin_enabled=pin_enabled,
            pin_hash=pin_hash,
            pin_salt=pin_salt,
            pending_actions=pending_actions,
            state=state,
        )

        precheck = await _precheck_alarm_open_entries(
            tool_name=tool_name, tool_args=tool_args, ctx=ctx, tool_call=tool_call
        )
        if precheck:
            tool_responses.append(precheck)
            continue

        lc_key = tool_name.lower()
        if lc_key in langchain_tools:
            tool_response = await _run_langchain_tool(
                tool_name, tool_call, langchain_tools, store, config
            )
        else:
            tool_response = await _run_ha_tool(
                tool_name=tool_name,
                tool_args=tool_args,
                ctx=ctx,
                tool_call=tool_call,
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
