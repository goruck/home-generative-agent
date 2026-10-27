"""Langgraph graphs for Home Generative Agent."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NotRequired,
    TypedDict,
    cast,
)

import psycopg
import voluptuous as vol
from homeassistant.core import State as HaState
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.util import dt as dt_util
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
from langgraph.store.base import InvalidNamespaceError
from pydantic import ValidationError

from custom_components.home_generative_agent.const import (
    ACTUATION_KEYWORDS_REGEX,
    AFFIRMATIVE_FOLLOW_UP_REGEX,
    AUTOMATION_ACTION_KEYWORDS_REGEX,
    AUTOMATION_INTENT_MARKERS_REGEX,
    AUTOMATION_TRIGGER_CLAUSE_REGEX,
    CONF_ANTHROPIC_CHAT_MODEL,
    CONF_CHAT_MODEL_PROVIDER,
    CONF_GEMINI_CHAT_MODEL,
    CONF_MANAGE_CONTEXT_WITH_TOKENS,
    CONF_MAX_MESSAGES_IN_CONTEXT,
    CONF_MAX_TOKENS_IN_CONTEXT,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OPENAI_CHAT_MODEL,
    CONF_OPENAI_COMPATIBLE_CHAT_MODEL,
    CONF_SCHEMA_FIRST_YAML,
    CONF_TOOL_RELEVANCE_THRESHOLD,
    CONF_TOOL_RETRIEVAL_LIMIT,
    EMBEDDING_MODEL_PROMPT_TEMPLATE,
    HISTORY_INTENT_REGEX,
    MAX_AFFIRMATIVE_FOLLOW_UP_WORDS,
    MAX_REFERENTIAL_FOLLOW_UP_WORDS,
    NON_OPEN_ACTUATION_KEYWORDS_REGEX,
    OPEN_AS_STATE_REGEX,
    OPEN_COMMAND_CLAUSE_REGEX,
    READ_ONLY_STATE_QUERY_REGEX,
    REFERENTIAL_FOLLOW_UP_REGEX,
    SECURITY_AUDIT_INTENT_REGEX,
    SUMMARIZATION_INITIAL_PROMPT,
    SUMMARIZATION_PROMPT_TEMPLATE,
    SUMMARIZATION_SYSTEM_PROMPT,
    TOOL_CALL_ERROR_TEMPLATE,
    TOOL_CALL_TRANSIENT_ERROR_TEMPLATE,
    TOOL_INCLUSIONS_MAX_PER_TURN,
)

from ..core.fallback import (  # noqa: TID252
    DROPPABLE_SAMPLING_PARAMS,
    EmbeddingsUnavailableError,
    FallbackChatModel,
    ainvoke_dropping_unsupported_params,
    unsupported_sampling_param_in_chain,
)
from ..core.prompt_cache import (  # noqa: TID252
    ANTHROPIC_LLM_TYPE,
    adapt_system_message_for_model,
    build_system_message,
    concrete_chat_model,
)
from ..core.provider_stall import model_call_budget  # noqa: TID252
from ..core.utils import extract_final  # noqa: TID252
from .automation_pin import STEP_TARGET_KEYS
from .automation_targets import AUTOMATION_REFUSAL_PREFIX
from .helpers import (
    ALARM_DOMAIN,
    CONFIRM_TOOL_NAME,
    INTENT_NAME_ALL,
    PIN_REFUSAL_ENDS_TURN,
    REMOTE_TOOL,
    active_llm_api_ids,
    base_tool_name,
    declared_domains,
    declared_intent_service_calls,
    drop_empty_args,
    entity_domain,
    format_tool,
    intent_service_calls,
    intent_target_states,
    is_actuating_intent,
    is_actuation_tool,
    is_device_intent,
    is_intent_tool,
    is_on_off_intent,
    is_tool_excluded,
    langchain_tool_parameters_json,
    matches_critical_rule,
    maybe_fill_lock_entity,
    minimal_payload_for_domain,
    narrow_intent_args,
    normalize_intent_for_alarm,
    normalize_intent_for_lock,
    parse_tool_response,
    pending_action_expired,
    pin_lines_follow,
    pin_refusal,
    register_pending_action,
    remote_house_words,
    remote_tool_label,
    remote_tool_name,
    resolve_critical_action_policy,
    sanitize_tool_args,
    sanitize_tool_text,
    thread_has_pending_action,
    tool_call_data,
    tool_exclusions,
    tool_inclusions,
    tool_index_key,
)
from .indirect_pin import (
    BUILTIN_INTENT,
    OTHER_INTENT,
    RUN_SERVICES,
    SCENE_DOMAIN,
    SCRIPT_DOMAIN,
    StoredActionCheck,
    ToolOrigin,
    bare_script_tool_entity_id,
    call_service_names,
    call_target_domains,
    calls_a_service,
    collect_started_automations,
    has_service,
    intent_scripts_are_critical,
    is_script_like_tool,
    named_services,
    remote_aliases,
    remote_api_names,
    rule_domains,
    runs_critical_action,
    script_rule_matches,
    script_tool_entity_id,
    service_runs_critical_action,
    stored_actions_are_critical,
    tool_args_kind,
    tool_origin,
)
from .on_off_direction import (
    is_bare_yes,
    target_entity_ids,
    with_action_performed,
    wrong_direction_refusal,
)
from .pin_messages import describe_pending_action, pin_msg
from .rule_entities import expand_critical_rules
from .template_actions import entities_of
from .token_counter import count_tokens_cross_provider

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from homeassistant.core import HomeAssistant
    from langchain_core.runnables import RunnableConfig
    from langgraph.store.base import BaseStore

LOGGER = logging.getLogger(__name__)

# Maximum seconds to wait for a single LangChain tool (e.g. camera VLM).
# With asyncio.gather a hung tool blocks all parallel tools in the batch,
# so this converts an infinite hang into a bounded error. Keep below the
# outer HA conversation timeout.
_LC_TOOL_TIMEOUT_S: float = 30.0

# Maximum seconds to wait for the chat LLM to respond.
# The chat model may need exclusive GPU access and can be delayed while the
# chat-priority gate waits for in-flight background jobs (camera analysis,
# embedding, Sentinel LLM) to finish.  With a large prompt, prefill alone
# can take tens of seconds.  Set high enough to cover gate wait + prefill +
# generation for typical conversation history lengths.
_LLM_INVOKE_TIMEOUT_S: float = 180.0

_PIN_LOOKBACK = 20  # messages to scan for an unresolved requires_pin
_ROUTING_REJECTION_MARKER = "is not available for this request"
_OPEN_STATE_RETRY_PHRASES: frozenset[str] = frozenset(
    {"yes", "y", "try again", "again"}
)


class State(MessagesState):
    """Extend MessagesState."""

    summary: str
    chat_model_usage_metadata: dict[str, Any]
    messages_to_remove: list[AnyMessage]
    selected_tools: list[dict[str, Any]]
    tool_routing_map: dict[str, str]
    action_rounds: NotRequired[int]
    # Tool rounds in which every call was an add_automation refusal: nothing
    # ran, the model was told what to correct. Counted apart from
    # action_rounds so a correction never costs the round the corrected call
    # needs, capped so a model that keeps guessing still stops.
    correction_rounds: NotRequired[int]
    # Memories retrieved for the user message this turn serves, memoized so
    # every model call of the turn (tool call → answer) sees the same block
    # and pays for one embedding, not one per call. Reset per turn by the
    # conversation entity; keyed so a stale entry can never be reused.
    turn_memories: NotRequired[dict[str, str]]


def _determine_model_name(provider: str, opts: dict[str, Any]) -> str:
    """Determine model name based on provider and options."""
    if provider == "openai":
        return opts.get(CONF_OPENAI_CHAT_MODEL, "")
    if provider == "openai_compatible":
        return opts.get(CONF_OPENAI_COMPATIBLE_CHAT_MODEL, "")
    if provider == "gemini":
        return opts.get(CONF_GEMINI_CHAT_MODEL, "")
    if provider == "anthropic":
        return opts.get(CONF_ANTHROPIC_CHAT_MODEL, "")
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


def _make_transient_tool_error(err: str, name: str, tid: str) -> ToolMessage:
    """Build a ToolMessage for transient/resource errors that must not be retried."""
    return ToolMessage(
        content=TOOL_CALL_TRANSIENT_ERROR_TEMPLATE.format(error=err),
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

    open_entries = await _find_open_entries(ctx.hass, [])
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
        lc_result = await asyncio.wait_for(
            lc_tool.ainvoke(tool_call_copy), timeout=_LC_TOOL_TIMEOUT_S
        )
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
    except TimeoutError:
        return _make_transient_tool_error(
            f"Tool timed out after {_LC_TOOL_TIMEOUT_S}s.",
            tool_name,
            tool_call.get("id") or "",
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
    *,
    read_code: bool = True,
) -> ToolMessage | None:
    """
    Redirect alarm-like HA intents to the dedicated alarm tool.

    An alarm call is one that declares the alarm panel domain (the alarm
    normalizer adds it when HA's matcher resolves the name to alarm panels
    only) or gives the entity id of an alarm panel. A numeric code suggests
    one only when the call declares no domain and either names no target at
    all or everything HA matches to it, hidden devices included, is an alarm
    panel. A code given for anything else, or for a target HA cannot find
    (a lock in the wrong area), is not a sign of the alarm. Never from the
    letters of the name: "Security Light" and "Security Door Lock" were
    refused as alarms (#782).
    """
    declared = declared_domains(tool_args)
    entity_ids = _coerce_entity_ids(tool_args)
    looks_like_alarm = ALARM_DOMAIN in declared or any(
        entity_domain(e) == ALARM_DOMAIN for e in entity_ids
    )
    # A numeric code hints at the alarm only for HA's own intents: a script
    # may take a code of its own (#836 review).
    if (
        read_code
        and not looks_like_alarm
        and not declared
        and not entity_ids
        and hass.states.async_entity_ids(ALARM_DOMAIN)
        and str(tool_args.get("code", "")).strip().isdigit()
    ):
        looks_like_alarm = _code_hints_alarm(hass, tool_name, tool_args)
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


def _code_hints_alarm(
    hass: HomeAssistant, tool_name: str, tool_args: dict[str, Any]
) -> bool:
    """Return True when a call with a code names no target or only alarm panels."""
    name = tool_args.get("name")
    named = isinstance(name, str) and name.strip() and name != INTENT_NAME_ALL
    scoped = any(str(tool_args.get(key) or "").strip() for key in ("area", "floor"))
    if not named and not scoped:
        return True
    calls = intent_service_calls(hass, tool_name, tool_args)
    return bool(calls) and all(call.domain == ALARM_DOMAIN for call in calls)


def _critical_action_guard(  # noqa: PLR0913
    tool_name: str,
    tool_args: dict[str, Any],
    ctx: ToolExecutionContext,
    tool_call: dict[str, Any],
    args_kind: str | None = None,
    origin: ToolOrigin | None = None,
) -> ToolMessage | None:
    """Enforce critical-action PIN flow when needed."""
    if not ctx.pin_enabled:
        return None
    # Extend entity_match rules to device classes, names and groups (#773,
    # see rule_entities).
    rules = expand_critical_rules(ctx.hass, ctx.critical_actions)
    with collect_started_automations() as started:
        critical = _is_critical_action(
            tool_args,
            rules,
            tool_name,
            ctx.hass,
            origin=origin,
        )
    if not critical:
        return None
    if not ctx.pin_hash or not ctx.pin_salt:
        LOGGER.warning(
            "Critical action PIN is enabled but no PIN is configured. "
            "Allowing '%s' without PIN verification. "
            "Set a PIN in the integration options to enforce confirmation.",
            tool_name,
        )
        return None

    configurable = ctx.config.get("configurable", {})
    pin_reply = configurable.get("pin_reply")
    action = {
        "tool_name": tool_name,
        "tool_args": tool_args,
        "user": configurable.get("user_id"),
        "thread_id": configurable.get("thread_id"),
        "turn_id": pin_reply.turn_id if pin_reply is not None else None,
        # The automations the change would start, for the PIN line (#814).
        "starts": list(started),
        # How the arguments are prepared, fixed now: the confirmation and a
        # repeated request read the call as it was held, even if an intent
        # script is added or removed meanwhile (#836 review).
        "args_kind": args_kind or tool_args_kind(ctx.hass, tool_name),
        # The domains the rules name, so the PIN line reads a tool's name as
        # the gate did (#837 review).
        "rule_domains": sorted(rule_domains(rules)),
        # The API the call goes to: the confirmation sends it there, whatever
        # tools the reply's turn offers, and a remote one's line names it
        # (#840).
        "api_id": origin.api_id if origin is not None else None,
        "api_name": origin.api_name if origin is not None else None,
        # The tool's own name there, which a remote one's differs from (#843).
        "api_tool_name": origin.tool_name if origin is not None else None,
        "remote": origin is not None and origin.remote,
    }
    action_id = register_pending_action(ctx.pending_actions, action)
    if action_id is None:
        # Another protected request is already waiting for this turn's PIN
        # prompt; the user asks for this one after answering it.
        return ToolMessage(
            content=pin_msg(ctx.hass, "pin_one_action_at_a_time"),
            tool_call_id=tool_call.get("id"),
            name=tool_name,
            status="error",
        )
    return ToolMessage(
        content=json.dumps(
            {
                "status": "requires_pin",
                "action_id": action_id,
                "action": describe_pending_action(ctx.hass, action),
                "reason": pin_msg(ctx.hass, "direct_requires_pin_reason"),
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
        response = tool_call_data(await ctx.ha_llm_api.async_call_tool(tool_input))
    except (HomeAssistantError, vol.Invalid) as err:
        return _make_tool_error(repr(err), tool_name, tool_call.get("id") or "")

    content_str, status = parse_tool_response(
        with_action_performed(tool_name, response.data), prepared_args
    )
    if response.error:
        status = "error"
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
    # Only HA's own intents are reshaped: a script, intent script or service
    # tool reads its own arguments, and runs with them as given (#836). So
    # does one only named like HA's own, such as a remote HA's (#840).
    origin = tool_origin(getattr(ctx, "ha_llm_api", None), tool_name)
    kind = tool_args_kind(ctx.hass, tool_name, remote=origin.remote)
    builtin = kind == BUILTIN_INTENT
    if builtin:
        tool_args = normalize_intent_for_alarm(tool_name, tool_args, ctx.hass)
        tool_args = normalize_intent_for_lock(tool_name, tool_args, ctx.hass)
    # A remote Home Assistant's tool is never taken for this one's
    # GetLiveContext, whatever its name (#840 review).
    if (
        not origin.remote
        and _is_live_context_tool(tool_name)
        and not intent_scripts_are_critical(
            ctx.hass, tool_name, expand_critical_rules(ctx.hass, ctx.critical_actions)
        )
    ):
        # Read-only, and HA's schema is name/domain/area only. The lock
        # rewriting below exists for lock/unlock commands; applied to a state
        # query it added service/entity_id keys HA rejects (ExtraKeysInvalid),
        # so every "is the lock locked?" failed and looped since v2.8.0.
        return None, _live_context_args(sanitize_tool_args(tool_args))
    if builtin:
        tool_args = maybe_fill_lock_entity(tool_args, ctx.hass)
        tool_args = sanitize_tool_args(tool_args)
        # Normalize domains to a clean list for downstream consumers.
        domains = tool_args.get("domain") or []
        if isinstance(domains, str):
            domains = [domains]
        tool_args["domain"] = [str(d).strip() for d in domains if str(d).strip()]
    elif kind == OTHER_INTENT:
        # An intent script matches its targets as the built-ins do.
        tool_args = narrow_intent_args(drop_empty_args(tool_args))
    else:
        tool_args = drop_empty_args(tool_args)

    # The alarm tool acts on this Home Assistant's panels, not a remote one's.
    redirect = kind != REMOTE_TOOL and _route_alarm_intent_to_tool(
        tool_name, tool_args, ctx.hass, tool_call, read_code=builtin
    )
    if redirect:
        return redirect, tool_args

    # The other half of HassTurnOn/HassTurnOff from what was asked (#797):
    # refused before the PIN check, so it is never queued for a PIN.
    state = getattr(ctx, "state", None)
    messages = state.get("messages", []) if isinstance(state, dict) else []
    # An intent script that took over HassTurnOn may do anything (#836 review).
    refusal = builtin and wrong_direction_refusal(
        tool_name,
        tool_args,
        _on_off_request_text(messages),
        known=_known_target(ctx.hass, tool_args),
    )
    if refusal:
        LOGGER.info("Refused %s: %s", tool_name, refusal)
        return _make_tool_error(
            refusal, tool_name, tool_call.get("id") or ""
        ), tool_args

    critical_guard = _critical_action_guard(
        tool_name=tool_name,
        tool_args=tool_args,
        ctx=ctx,
        tool_call=tool_call,
        args_kind=kind,
        origin=origin,
    )
    if critical_guard:
        return critical_guard, tool_args

    if builtin:
        # Reduce an intent's payload to the slots HA reads (avoids extra-key
        # errors).
        tool_args = sanitize_tool_args(minimal_payload_for_domain(tool_args, tool_name))
    return None, tool_args


LIVE_CONTEXT_TOOL = "GetLiveContext"


def _is_live_context_tool(name: str) -> bool:
    """Return True when `name` is HA's live-context tool, namespaced or not."""
    return base_tool_name(name) == LIVE_CONTEXT_TOOL


# The intents that only read, on a remote Home Assistant: a call to one that
# sends no read-only mark is not held. Narrower than READ_ONLY_INTENTS, which
# lets this Home Assistant's lists, timers and broadcasts through: on a remote
# one, every change is held (#840 review).
_REMOTE_READ_ONLY_INTENTS = frozenset(
    {
        "HassGetState",
        "HassGetWeather",
        "HassGetCurrentDate",
        "HassGetCurrentTime",
        "HassClimateGetTemperature",
        "HassTimerStatus",
        "HassShoppingListLastItems",
        "HassNevermind",
        "HassRespond",
    }
)
# A timer's slot holding a command the timer runs when it finishes.
_START_TIMER_INTENT = "HassStartTimer"
_TIMER_COMMAND_SLOT = "conversation_command"

# ----- Alarm helpers -----


def _extract_last_live_context(messages: list[AnyMessage]) -> str | None:
    """Return the most recent GetLiveContext payload content, if any."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and _is_live_context_tool(msg.name or ""):
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
        if "device_class" in stripped:
            dc = stripped.split(":", maxsplit=1)[-1].strip().strip("'\"")
            if dc in _OPEN_STATE_DEVICE_CLASSES:
                is_opening = True
            continue

    if current_name and is_on and is_opening:
        open_entries.append(current_name)
    return open_entries


_OPEN_STATE_DEVICE_CLASSES: frozenset[str] = frozenset(
    {"door", "garage_door", "opening", "window"}
)

_OPEN_ENTITY_KEYWORDS: tuple[str, ...] = ("window", "door", "gate")


def _open_state_name_matches_query(name: str, query: str) -> bool:
    """Return whether an open entry name is in scope for the user's query."""
    name_lc = name.lower()
    query_lc = query.lower()
    present = [kw for kw in _OPEN_ENTITY_KEYWORDS if kw in query_lc]
    if not present:
        return True
    return any(kw in name_lc for kw in present)


def _open_state_empty_result(query: str) -> str:
    """Return a scoped empty live-context result for the user's query."""
    query_lc = query.lower()
    present = [kw for kw in _OPEN_ENTITY_KEYWORDS if kw in query_lc]
    if not present:
        return "Live Context: No open entries were found."
    if len(present) == 1:
        return f"Live Context: No open {present[0]}s were found."
    entity_list = ", ".join(f"{kw}s" for kw in present[:-1]) + f" or {present[-1]}s"
    return f"Live Context: No open {entity_list} were found."


def _filter_open_state_live_context_content(content: str, query: str) -> str:
    """Filter broad live context down to the open entries requested by the user."""
    if not _query_is_read_only_open_state(query):
        return content

    parsed: dict[str, Any] | None = None
    try:
        maybe_parsed = json.loads(content)
    except json.JSONDecodeError:
        maybe_parsed = None
    if isinstance(maybe_parsed, dict):
        parsed = maybe_parsed
        if parsed.get("success") is False or parsed.get("error"):
            return content
        raw_result = str(parsed.get("result") or parsed.get("response") or "")
    else:
        raw_result = content

    open_entries = [
        name.replace("\n", " ").replace("\r", " ")[:128]
        for name in _parse_open_entries_from_live_context(raw_result)
        if _open_state_name_matches_query(name, query)
    ]
    if open_entries:
        result = "Live Context: Open entries matching the request:\n" + "\n".join(
            f"- names: {name}\n  state: 'on'\n  attributes:\n    device_class: opening"
            for name in open_entries
        )
    else:
        result = _open_state_empty_result(query)

    if parsed is None:
        return result

    filtered = dict(parsed)
    filtered["result"] = result
    return json.dumps(filtered)


def _maybe_filter_open_state_tool_response(
    tool_response: ToolMessage,
    tool_call: dict[str, Any],
    first_open_state_live_context_call: dict[str, Any] | None,
    tool_name: str,
    query: str,
) -> ToolMessage:
    """Filter broad live context before returning it to the model."""
    if (
        tool_call is not first_open_state_live_context_call
        or not _is_live_context_tool(tool_name)
        or not isinstance(tool_response.content, str)
        # A failed lookup has no states to filter; rewriting it would turn
        # "unavailable" into "no open windows".
        or tool_response.status == "error"
    ):
        return tool_response

    tool_response.content = _filter_open_state_live_context_content(
        tool_response.content, query
    )
    return tool_response


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
        if state_obj.attributes.get("device_class") not in _OPEN_STATE_DEVICE_CLASSES:
            continue
        if str(state_obj.state).lower() not in {"on", "open", "opening"}:
            continue
        friendly = state_obj.attributes.get("friendly_name") or entity_id
        open_entries.append(str(friendly))
    return open_entries


class RawTool(TypedDict):
    """Represent a tool before formatting for the LLM."""

    name: str
    api_id: str
    description: str
    parameters: str
    is_actuation: bool
    # The words of a remote Home Assistant's server name that say which house
    # it is (`remote_house_words`), which the keyword fallback ranks on: the
    # bind adds the server's name to the description only after ranking (#848).
    house: NotRequired[str]


def _get_allowed_api_ids(config: RunnableConfig) -> set[str]:
    """Determine which API IDs are allowed based on configuration."""
    opts = config.get("configurable", {}).get("options", {})
    return set(active_llm_api_ids(opts)) | {"hga_local"}


def _get_live_tool_ids(config: RunnableConfig) -> set[tuple[str, str]] | None:
    """
    Return the (api_id, tool_name) pairs live for this conversation turn.

    The retrieval index is global and cumulative: it can hold tools that do not
    exist for the current request — device-gated Assist tools indexed from a
    satellite turn (timer intents), or tools of a configured API that failed to
    load this turn. Only tools present in the loaded ``ha_llm_api`` (plus local
    LangChain tools) are callable, so candidates are filtered against this set.

    Fail-open contract (None = filtering disabled): ``ha_llm_api`` absent or
    lacking ``.apis``, or the ``langchain_tools`` key absent entirely — a caller
    that wired neither surface gets pre-filter behavior (tests, robot runs). A
    present ``ha_llm_api`` with an *empty* ``langchain_tools`` dict is a real
    state (no local tools this turn) and filters normally; treating it as
    fail-open would let stale local candidates bind, and treating an absent key
    as live-empty would silently drop every local tool for such a caller.
    """
    configurable = config.get("configurable", {})
    ha_llm_api = configurable.get("ha_llm_api")
    if ha_llm_api is None or not hasattr(ha_llm_api, "apis"):
        LOGGER.debug("Live-tool filter disabled: no ha_llm_api in config (fail open)")
        return None
    if "langchain_tools" not in configurable:
        LOGGER.debug(
            "Live-tool filter disabled: no langchain_tools in config (fail open)"
        )
        return None
    live: set[tuple[str, str]] = {
        (api_id, tool.name)
        for api_id, api in ha_llm_api.apis.items()
        for tool in api.tools
    }
    langchain_tools = configurable.get("langchain_tools") or {}
    live |= {("hga_local", name) for name in langchain_tools}
    return live


class _Unset:
    """Sentinel type: `None` is a meaningful live-tool value (fail open)."""


_UNSET = _Unset()


def _filter_live_candidates(
    config: RunnableConfig,
    rag_tools: list[RawTool],
    safety_tools: list[RawTool],
    live_tool_ids: set[tuple[str, str]] | _Unset | None = _UNSET,
) -> tuple[list[RawTool], list[RawTool], set[tuple[str, str]] | None, int]:
    """
    Drop index candidates that are not live this turn.

    Dead candidates (device-gated tools indexed from another device's turn,
    tools of a configured API that failed to load) are removed before the merge
    so they never consume merge slots. Returns the filtered lists, the live set
    (None = filtering disabled, fail open), and the dropped count.

    Callers that already resolved the live set pass it in so it is computed once
    per turn; omitting it resolves it here. ``None`` cannot serve as the "not
    supplied" marker — it is the fail-open signal — hence the sentinel.
    """
    if isinstance(live_tool_ids, _Unset):
        live_tool_ids = _get_live_tool_ids(config)
    if live_tool_ids is None:
        return rag_tools, safety_tools, None, 0
    kept_rag = [t for t in rag_tools if (t["api_id"], t["name"]) in live_tool_ids]
    kept_safety = [t for t in safety_tools if (t["api_id"], t["name"]) in live_tool_ids]
    dropped = (len(rag_tools) - len(kept_rag)) + (len(safety_tools) - len(kept_safety))
    if dropped:
        LOGGER.debug(
            "Live-tool filter dropped %d candidate(s): %s",
            dropped,
            sorted(
                {(t["api_id"], t["name"]) for t in rag_tools + safety_tools}
                - live_tool_ids
            ),
        )
    return kept_rag, kept_safety, live_tool_ids, dropped


_INTENT_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+|,\s*|\band\b\s+|\bthen\b\s+|\balso\b\s+",
    re.IGNORECASE,
)
_MIN_SUBQUERY_LEN = 8  # ignore trivially short fragments


def _split_query_intents(query: str) -> list[str]:
    """
    Split a multi-intent query into sub-queries for per-intent RAG search.

    Always includes the original full query so the caller never loses the
    whole-sentence embedding signal.
    """
    parts = [p.strip() for p in _INTENT_SPLIT_RE.split(query)]
    sub_queries = [p for p in parts if len(p) >= _MIN_SUBQUERY_LEN]
    if len(sub_queries) <= 1:
        return [query]
    return [query, *sub_queries]


async def _get_rag_retrieved_tools(  # noqa: PLR0912, PLR0915
    store: BaseStore | None,
    config: RunnableConfig,
    query: str,
    allowed_api_ids: set[str],
    live_tool_ids: set[tuple[str, str]] | None = None,
) -> tuple[list[RawTool], int]:
    """
    Search for tools in the vector store and filter by score/API.

    For multi-intent queries, splits the query into per-intent sub-queries and
    runs a separate vector search for each.  Each tool's score is the maximum
    across all sub-query passes so that a tool highly relevant to *one* intent
    is not diluted by the blended embedding of the full query.

    Returns the selected tools and how many hits were discarded as not live —
    the caller reports that count, because "the vector search found nothing
    usable" and "everything it found had been renamed away" are very different
    problems and the operator only ever sees the log line.
    """
    if store is None:
        LOGGER.warning("Store is None; skipping RAG tool retrieval")
        return [], 0

    tool_index_ready, _tool_indexing_in_progress, _tool_index_failed = (
        _tool_index_state(config)
    )
    if not tool_index_ready:
        LOGGER.debug("Skipping RAG tool retrieval: tool index is not ready")
        return [], 0

    opts = config.get("configurable", {}).get("options", {})
    limit = int(opts.get(CONF_TOOL_RETRIEVAL_LIMIT, 5))
    threshold = float(opts.get(CONF_TOOL_RELEVANCE_THRESHOLD, 0.15))
    excluded_names = tool_exclusions(opts)
    excluded_hits = 0
    dead_hits = 0

    sub_queries = _split_query_intents(query)

    # best_score[tool_key] -> (score, SearchItem)
    best: dict[str, tuple[float, Any]] = {}
    facing = _facing_name(config)
    total_results = 0
    allowed_results = 0
    below_threshold = 0
    top_seen: list[tuple[str, float]] = []

    for sub_query in sub_queries:
        try:
            results = await store.asearch(
                ("system", "tools"), query=sub_query, limit=limit * 4
            )
        except (
            InvalidNamespaceError,
            psycopg.OperationalError,
            psycopg.ProgrammingError,
            psycopg.DataError,
            ValueError,
        ) as err:
            LOGGER.warning("RAG tool retrieval search failed (known error): %s", err)
            continue
        except Exception:
            LOGGER.exception("Unexpected RAG tool retrieval search failure")
            continue

        tool_index_ready, _tool_indexing_in_progress, _tool_index_failed = (
            _tool_index_state(config)
        )
        if not tool_index_ready:
            LOGGER.debug(
                "Discarding RAG tool retrieval results because the tool index "
                "became stale during vector search"
            )
            return [], dead_hits

        total_results += len(results)
        for item in results:
            name = item.value.get("name", "")
            api_id = item.value.get("api_id")
            if api_id not in allowed_api_ids:
                continue
            # Excluded tools are dropped HERE, beside the api-level check, and
            # not left to the live-tool filter downstream. The index is never
            # pruned when a user excludes a tool, so an excluded tool can still
            # be the top-scoring hit -- and `sorted_items[:limit]` below cuts to
            # the user's retrieval limit *before* `_filter_live_candidates` ever
            # sees the list. Filtering late therefore lets an excluded tool eat
            # a retrieval slot and then vanish, so switching one tool off would
            # silently reduce (or, once the list empties into the keyword
            # fallback, entirely reshape) which OTHER tools reach the model on
            # unrelated queries.
            if is_tool_excluded(name, excluded_names.get(api_id, ())):
                excluded_hits += 1
                continue
            # Dead candidates are dropped HERE for the same reason excluded
            # ones are: `sorted_items[:limit]` below cuts to the user's
            # retrieval limit *before* `_filter_live_candidates` downstream ever
            # sees the list, so a row that is in the index but not live this
            # turn eats a retrieval slot and then vanishes. That is normally a
            # rare nuisance, but the index is never pruned, so a mass rename --
            # HA 2026.9 renamed every built-in tool to `<domain>__<Name>` --
            # leaves a full shadow copy of the old names behind, and those rows
            # *outrank* their live twins (their embedding text is the same
            # minus the prefix). Field report: "turn it off" retrieved 5
            # candidates, 4 were renamed-away rows, and the model was left with
            # one media-player tool and no way to turn the light off.
            if live_tool_ids is not None and (api_id, name) not in live_tool_ids:
                dead_hits += 1
                continue
            allowed_results += 1
            # score is (1 - distance) for pgvector cosine; None when no embedding
            score = getattr(item, "score", None) or 0.0
            top_seen.append((name, score))
            # By the name the agent sees: a remote Home Assistant offers this
            # one's tool names, and both houses' tools must stay eligible,
            # while another API's namesake still takes one slot (#843 review).
            key = facing(api_id, name)
            if score >= threshold and (key not in best or score > best[key][0]):
                best[key] = (score, item)
            else:
                below_threshold += 1

    if not best:
        top_seen.sort(key=lambda item: item[1], reverse=True)
        LOGGER.debug(
            "RAG tool retrieval produced no candidates: sub_queries=%d "
            "raw_results=%d allowed_results=%d below_threshold=%d "
            "excluded=%d not_live=%d threshold=%.3f allowed_api_ids=%s "
            "top_scores=%s",
            len(sub_queries),
            total_results,
            allowed_results,
            below_threshold,
            excluded_hits,
            dead_hits,
            threshold,
            sorted(allowed_api_ids),
            top_seen[:5],
        )

    # Sort by best score descending and return top `limit`
    sorted_items = sorted(best.values(), key=lambda t: t[0], reverse=True)
    raw_tools: list[RawTool] = []
    for score, item in sorted_items[:limit]:
        LOGGER.debug("RAG tool candidate: %s score=%.3f", item.value["name"], score)
        raw_tools.append(
            RawTool(
                name=item.value["name"],
                api_id=item.value["api_id"],
                description=item.value["description"],
                parameters=item.value["parameters"],
                is_actuation=item.value.get("is_actuation", False),
            )
        )

    return raw_tools, dead_hits


def _facing_name(config: RunnableConfig) -> Callable[[str, str], str]:
    """Return a function naming a tool as the agent will see it (#843)."""
    ha_llm_api = config.get("configurable", {}).get("ha_llm_api")
    aliases = remote_aliases(ha_llm_api)
    remote = remote_api_names(ha_llm_api)

    def facing(api_id: str, name: str) -> str:
        if api_id not in remote:
            return name
        return aliases.get((api_id, name)) or remote_tool_name(remote[api_id], name)

    return facing


def _remote_ids(config: RunnableConfig) -> set[str]:
    """Return the ids of this turn's remote Home Assistant APIs (#843)."""
    return set(remote_api_names(config.get("configurable", {}).get("ha_llm_api")))


async def _get_local_first(
    store: BaseStore,
    config: RunnableConfig,
    name: str,
    candidates: Sequence[RawTool],
    live_tool_ids: set[tuple[str, str]] | None,
) -> RawTool | None:
    """
    Fetch this house's tool by name; a remote one's only if none is bound.

    A remote Home Assistant's tool does not stand in for this house's own,
    but when this house offers none it may use (none, or it is excluded) and
    no tool of that name is bound, a remote one's is bound as before (#843
    review), renamed, described as another house's, and held for the PIN.
    """
    allowed = _get_allowed_api_ids(config)
    remote_ids = _remote_ids(config)
    fetched = await _get_tool_by_name(
        store, config, name, allowed - remote_ids, live_tool_ids
    )
    if fetched is not None or any(
        base_tool_name(t["name"]) == base_tool_name(name) for t in candidates
    ):
        return fetched
    return await _get_tool_by_name(
        store, config, name, allowed & remote_ids, live_tool_ids
    )


def _has_local_tool(
    candidates: Sequence[RawTool],
    remote_ids: set[str],
    matches: Callable[[str], bool],
) -> bool:
    """Return True if a candidate of this house's own APIs matches by name."""
    return any(matches(t["name"]) and t["api_id"] not in remote_ids for t in candidates)


def _is_referential_follow_up(query: str) -> bool:
    """
    Return True when a turn points back at the previous one, not at a target.

    Two signals, both required: the query is short, and it leans on a
    referential word ("it", "them", "that", "again"). Requiring both keeps an
    ordinary query that merely contains one of those words -- "set the
    temperature in this room" -- ranking on its own text.
    """
    query = query[:_MAX_INTENT_SCAN_CHARS].strip()
    if not query or len(query.split()) > MAX_REFERENTIAL_FOLLOW_UP_WORDS:
        return False
    return bool(re.search(REFERENTIAL_FOLLOW_UP_REGEX, query))


def _is_affirmative_follow_up(query: str) -> bool:
    """Return True for a short reply accepting an offer ("Yes", "Sure, go ahead")."""
    query = query[:_MAX_INTENT_SCAN_CHARS].strip()
    if not query or len(query.split()) > MAX_AFFIRMATIVE_FOLLOW_UP_WORDS:
        return False
    return bool(re.search(AFFIRMATIVE_FOLLOW_UP_REGEX, query))


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def _offer_text(messages: Sequence[BaseMessage]) -> str:
    """
    Return the end of the agent's last spoken reply, where an offer sits.

    Only the last two sentences: "The light is on because you asked. ...
    Want me to turn it off?" carries its action target at the end, and a long
    reply's earlier sentences would only add noise to the ranking.
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            continue
        if isinstance(msg, AIMessage) and (text := _message_text(msg).strip()):
            sentences = [s for s in _SENTENCE_END_RE.split(text) if s.strip()]
            return " ".join(sentences[-2:])
    return ""


def _known_target(
    hass: HomeAssistant | None, tool_args: Mapping[str, Any]
) -> tuple[set[str], set[str]]:
    """
    Return the domains and friendly names HA knows for a call's target (#797).

    An entity id gives its friendly name ("light.ceiling" is "Open Plan
    Lights", whose "open" is then a name), and a friendly name with no
    domain gives the domains of the entities called that.
    """
    domains: set[str] = set()
    names: set[str] = set()
    states = getattr(hass, "states", None)
    if states is None:
        return domains, names
    ids, others = target_entity_ids(tool_args)
    for entity_id in ids:
        state = states.get(entity_id)
        if isinstance(state, HaState):
            names.add(str(state.attributes.get("friendly_name") or ""))
    wanted = {name.casefold() for name in others}
    if wanted and not tool_args.get("domain"):
        for state in states.async_all():
            if not isinstance(state, HaState):
                continue
            friendly = str(state.attributes.get("friendly_name") or "").casefold()
            if friendly in wanted:
                domains.add(state.domain)
    return domains, names


def _on_off_request_text(messages: Sequence[BaseMessage]) -> str:
    """
    Return the words a HassTurnOn/HassTurnOff call is checked against (#797).

    The user's latest message; for a short "Yes", the end of the agent's
    offer ("I can open the shade. Would that help?"). Only a command reads
    there, so a state it gives ("The shade is closed.") does not.
    """
    for index in range(len(messages) - 1, -1, -1):
        if not isinstance(messages[index], HumanMessage):
            continue
        text = _message_text(messages[index])[:_MAX_INTENT_SCAN_CHARS]
        # "Please shut it" gives its own command; only a bare "Yes" takes
        # the offer's.
        if not is_bare_yes(text):
            return text
        for msg in reversed(messages[:index]):
            if isinstance(msg, AIMessage) and (offer := _message_text(msg).strip()):
                sentences = [s for s in _SENTENCE_END_RE.split(offer) if s.strip()]
                # The last two: "I can open the shade. Would that help?"
                return " ".join(sentences[-2:])
        return ""
    return ""


def _retrieval_query(messages: Sequence[BaseMessage]) -> str:
    """
    Build the query used to RANK tools, which is not always the user's words.

    A referential follow-up ("turn it off") carries no entity, domain, or
    action-target signal, so ranking tools on it alone is close to ranking them
    on noise: after the v3.41.1 index cleanup, "Turn them off." still put five
    media-player tools between 0.600 and 0.512 and never surfaced
    intent__HassTurnOff, so the model's correct
    ``intent__HassTurnOff(name="Garage Light", domain=["light"])`` was rejected
    as a tool it had not been given.

    Prepending the previous user turn restores the missing target. The two are
    joined with ". " so ``_split_query_intents`` searches the combined text AND
    each turn separately, and ``_get_rag_retrieved_tools`` keeps the best score
    per tool across those passes -- a tool that only the current phrasing finds
    is therefore never displaced by the added context.

    This widens RETRIEVAL only. The caller keeps the raw last message for every
    intent detector (automation, read-only open-state, security audit), because
    those gate behaviour and must never fire on a previous turn's words.
    """
    if not messages:
        return ""
    current = _message_text(messages[-1])
    if _is_affirmative_follow_up(current):
        # "Yes" to "Want me to turn it off?": the target and the action are
        # both in the agent's offer, so rank on it, the user's previous turn,
        # and the reply itself (each searched separately; best score wins).
        offer = _offer_text(messages[:-1])
        previous = next(
            (
                _message_text(m).strip()
                for m in reversed(messages[:-1])
                if isinstance(m, HumanMessage)
            ),
            "",
        )
        parts = [p.rstrip(".!? ") for p in (previous, offer) if p]
        if parts:
            LOGGER.debug(
                "Affirmative follow-up: widening the retrieval query with the "
                "agent's offer (model input unchanged)"
            )
            return ". ".join([*parts, current])
        return current
    if not _is_referential_follow_up(current):
        return current
    for msg in reversed(messages[:-1]):
        if not isinstance(msg, HumanMessage):
            continue
        previous = _message_text(msg).strip()
        if not previous:
            break
        LOGGER.debug(
            "Referential follow-up: widening the retrieval query with the "
            "previous user turn (model input unchanged)"
        )
        return f"{previous.rstrip('.!?')}. {current}"
    return current


_MAX_ACTION_ROUNDS = 3

# Extra rounds granted for correcting a refused automation. A refusal is not
# an action: the tool ran nothing and handed back the names to fix, so the
# retry it asks for must not be the round the guard cuts off. Two covers a
# wrong entity fixed in one pass and a wrong service in the next.
_MAX_CORRECTION_ROUNDS = 2


def _is_automation_refusal(msg: BaseMessage) -> bool:
    """Return True for the tool message of a refused add_automation call."""
    return (
        isinstance(msg, ToolMessage)
        and base_tool_name(msg.name or "") == "add_automation"
        and isinstance(msg.content, str)
        and msg.content.startswith(AUTOMATION_REFUSAL_PREFIX)
    )


def _next_round_counters(
    state: State, tool_responses: Sequence[BaseMessage]
) -> dict[str, int]:
    """
    Return the round counters after a tool round.

    A round in which every tool call was a refused add_automation is a
    correction round while the correction budget lasts; any other round,
    including a refusal once that budget is spent, is an action round.
    """
    action_rounds = state.get("action_rounds", 0)
    correction_rounds = state.get("correction_rounds", 0)
    if (
        tool_responses
        and all(_is_automation_refusal(msg) for msg in tool_responses)
        and correction_rounds < _MAX_CORRECTION_ROUNDS
    ):
        return {
            "action_rounds": action_rounds,
            "correction_rounds": correction_rounds + 1,
        }
    return {
        "action_rounds": action_rounds + 1,
        "correction_rounds": correction_rounds,
    }


def _message_text(msg: BaseMessage) -> str:
    """Return only the text portions of a message's content."""
    content = msg.content
    if isinstance(content, str):
        return content
    return " ".join(
        p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text"
    )


# Cap intent scans so a pathological multi-MB chat message cannot stall the
# event loop (the regexes are linear, ~1s/MB, but scans run synchronously).
_MAX_INTENT_SCAN_CHARS = 20_000


def _query_wants_automation(query: str) -> bool:
    """
    Return True when the query asks to create an automation.

    Matches either standalone automation vocabulary ("automate", "every
    morning", "remind me") or a conditional-actuation phrasing — an action
    verb plus a trigger clause ("turn on the porch light when motion is
    detected"). Read-only open-state queries ("check if the garage door is
    open") suppress the weaker conditional-actuation signal so step 3d does
    not re-add an actuation-adjacent tool that step 3b deliberately strips;
    explicit markers still win. Used to force-bind add_automation, whose
    instruction-heavy description cannot win embedding ranking against
    entity-control tools for these phrasings.
    """
    query = query[:_MAX_INTENT_SCAN_CHARS]
    if re.search(AUTOMATION_INTENT_MARKERS_REGEX, query):
        return True
    if _query_is_read_only_open_state(query):
        return False
    return bool(
        re.search(AUTOMATION_ACTION_KEYWORDS_REGEX, query)
        and re.search(AUTOMATION_TRIGGER_CLAUSE_REGEX, query)
    )


def _query_wants_security_audit(query: str) -> bool:
    """Return True when the query asks about the home's security posture."""
    return bool(re.search(SECURITY_AUDIT_INTENT_REGEX, query[:_MAX_INTENT_SCAN_CHARS]))


async def _append_security_audit_tool(  # noqa: PLR0913
    candidates: list[RawTool],
    store: BaseStore,
    config: RunnableConfig,
    query: str,
    allowed_api_ids: set[str],
    live_tool_ids: set[tuple[str, str]] | None,
) -> list[RawTool]:
    """
    Force-bind ``audit_home_security`` when the query asks about security.

    The system prompt tells the model to call it for "is my home secure?",
    but the tool reaches the bound set only through ranking, and the keyword
    fallback used while the index is not ready lists local tools in
    registration order and cuts at the limit. A mandated-but-unbound tool
    costs the model up to ``_MAX_ACTION_ROUNDS`` of "not available" retries.
    Appended outside the limit like ``add_automation``, so it never evicts a
    RAG/safety selection. The tool is registered only when Sentinel's network
    audit is on, so a miss (absent from index and config) is the normal off
    state.
    """
    if not _query_wants_security_audit(query) or _has_local_tool(
        candidates, _remote_ids(config), lambda name: name == "audit_home_security"
    ):
        return candidates
    fetched = await _get_tool_by_name(
        store,
        config,
        "audit_home_security",
        {"hga_local"} & allowed_api_ids,
        live_tool_ids,
    )
    return candidates if fetched is None else [*candidates, fetched]


def _query_wants_history(query: str) -> bool:
    """Return True when the query asks about past entity states or activity."""
    return bool(re.search(HISTORY_INTENT_REGEX, query[:_MAX_INTENT_SCAN_CHARS]))


# Trailing HUMAN turns that keep get_entity_history bound: the current message
# plus the previous exchange. A follow-up ("and yesterday?", "the mudroom
# lights") carries no history signal of its own and ranks on two words.
_HISTORY_CONTEXT_HUMAN_TURNS = 2


def _conversation_has_history_context(messages: Sequence[BaseMessage]) -> bool:
    """
    Return True when the previous turn was about past states.

    Covers the second question of a history conversation, which issue #715
    would otherwise reach again: the previous turn asked a history question or
    called ``get_entity_history``, and the follow-up does not say so itself.
    The tool call keeps a chain going ("and yesterday?", then "and the day
    before?"). The current message is the caller's ``_query_wants_history``
    check, so it is not scanned again here.
    """
    human_turns = 0
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            human_turns += 1
            if human_turns >= _HISTORY_CONTEXT_HUMAN_TURNS and _query_wants_history(
                _message_text(msg)
            ):
                return True
            if human_turns >= _HISTORY_CONTEXT_HUMAN_TURNS:
                return False
        elif (isinstance(msg, ToolMessage) and msg.name == "get_entity_history") or (
            isinstance(msg, AIMessage)
            and any(
                tc.get("name") == "get_entity_history"
                for tc in [*msg.tool_calls, *msg.invalid_tool_calls]
            )
        ):
            return True
    return False


async def _append_history_tool(  # noqa: PLR0913
    candidates: list[RawTool],
    store: BaseStore,
    config: RunnableConfig,
    query: str,
    messages: Sequence[BaseMessage],
    live_tool_ids: set[tuple[str, str]] | None,
) -> list[RawTool]:
    """
    Force-bind ``get_entity_history`` when the conversation asks about the past.

    A history question phrased around the device ("when did the mudroom
    lights turn off today?") ranks entity-control and date/time tools above
    the history tool, and a model that is not handed the tool tells the user
    it has no access to history (issue #715). Appended outside the limit like
    ``audit_home_security``, so it never evicts a RAG/safety selection. The
    lookup is pinned to ``hga_local``: dispatch routes this name to the local
    implementation, so the force-bind must not pull in another API's
    same-named tool. A same-named tool that ranking already bound is left as
    is, as before this step existed.
    """
    if _has_local_tool(
        candidates, _remote_ids(config), lambda name: name == "get_entity_history"
    ) or not (
        _query_wants_history(query) or _conversation_has_history_context(messages)
    ):
        return candidates
    fetched = await _get_tool_by_name(
        store, config, "get_entity_history", {"hga_local"}, live_tool_ids
    )
    return candidates if fetched is None else [*candidates, fetched]


_TURN_ON_TOOL = "HassTurnOn"
_ON_OFF_PARTNER = {_TURN_ON_TOOL: "HassTurnOff", "HassTurnOff": _TURN_ON_TOOL}


async def _append_on_off_partners(
    candidates: list[RawTool],
    store: BaseStore,
    config: RunnableConfig,
    live_tool_ids: set[tuple[str, str]] | None,
) -> list[RawTool]:
    """
    Bind the other half of HassTurnOn/HassTurnOff when ranking bound one.

    For a lock, HassTurnOn locks and HassTurnOff unlocks. Field test
    2026-10-15: "Block the playroom, block" ranked only HassTurnOff, the
    model's lock call was rejected as not retrieved, and it fell back to the
    one it had, which unlocked the door. Looked up in the same API, and an
    excluded tool is never bound (the raw index keeps excluded rows when the
    live filter is off; see `_append_included_tools`).
    """
    excluded_names = tool_exclusions(config.get("configurable", {}).get("options", {}))
    # Per API: another server's same-named tool is not this one's partner.
    bound = {(t["api_id"], base_tool_name(t["name"])) for t in candidates}
    for tool in candidates:
        base = base_tool_name(tool["name"])
        partner = _ON_OFF_PARTNER.get(base)
        if partner is None or (tool["api_id"], partner) in bound:
            continue
        # Spelled like the bound one ("intent__HassTurnOff" → "intent__HassTurnOn").
        partner_name = tool["name"].removesuffix(base) + partner
        excluded = excluded_names.get(tool["api_id"], ())
        if is_tool_excluded(partner_name, excluded):
            # Before the store round-trip: with the live filter on, an excluded
            # tool misses the index and falls through to the fallback scan.
            continue
        fetched = await _get_tool_by_name(
            store, config, partner_name, {tool["api_id"]}, live_tool_ids
        )
        if fetched is None or is_tool_excluded(fetched["name"], excluded):
            continue
        candidates = [*candidates, fetched]
        bound.add((tool["api_id"], partner))
    return candidates


async def _append_turn_on_tool(  # noqa: PLR0913
    candidates: list[RawTool],
    store: BaseStore,
    config: RunnableConfig,
    messages: Sequence[BaseMessage],
    allowed_api_ids: set[str],
    live_tool_ids: set[tuple[str, str]] | None,
) -> list[RawTool]:
    """
    Bind HassTurnOn on an actuation request when ranking left it out (#795).

    HassTurnOn is what opens and closes a cover, turns on a switch and presses
    a button: Home Assistant offers no cover-specific open tool. The actuation
    safety leg ranks by similarity and keeps three, so for "open the study
    shade" device tools (vacuum, media) that scored higher crowded it out, and
    the model guessed a tool that does not exist. Appended outside the limit
    like GetLiveContext; `_append_on_off_partners` then binds HassTurnOff.

    A "Yes" to "Want me to open the Study Shade?" has no action word, so the
    agent's offer is checked too. That widens the tool list only and gates
    nothing, which is why the raw-query rule for intent detectors does not
    apply here. The offer alone, not the ranking query: that one also carries
    the user's earlier "is it open?", which reads as a read-only question.
    """
    query = _message_text(messages[-1]) if messages else ""
    wants_action = (
        _query_needs_actuation_safety(query)
        or (
            _is_affirmative_follow_up(query)
            # Action word only: an offer often opens with the state ("It is
            # closed."), which the read-only check would take as a question.
            and bool(re.search(ACTUATION_KEYWORDS_REGEX, _offer_text(messages[:-1])))
        )
    )
    # This house's own: a remote Home Assistant's turn on or off does not
    # answer for it (#843 review).
    remote_ids = _remote_ids(config)
    if not wants_action or _has_local_tool(
        candidates, remote_ids, lambda name: base_tool_name(name) in _ON_OFF_PARTNER
    ):
        return candidates
    excluded_names = tool_exclusions(config.get("configurable", {}).get("options", {}))
    # Before the fetch, and by base name: with the live filter off the bare
    # lookup can return the pre-2026.9 "HassTurnOn" row, which a namespaced
    # exclusion ("intent__HassTurnOn") would not match by name.
    excluded_apis = {
        api_id
        for api_id, names in excluded_names.items()
        if any(base_tool_name(n) == _TURN_ON_TOOL for n in names)
    }
    fetched = await _get_tool_by_name(
        store,
        config,
        _TURN_ON_TOOL,
        allowed_api_ids - excluded_apis - remote_ids,
        live_tool_ids,
    )
    if fetched is None and not any(
        base_tool_name(t["name"]) in _ON_OFF_PARTNER for t in candidates
    ):
        # This house offers none it may use (none, or it is excluded): bind
        # a remote Home Assistant's, as before (#843 review). It reaches the
        # agent renamed, described as another house's, and held for the PIN.
        fetched = await _get_tool_by_name(
            store,
            config,
            _TURN_ON_TOOL,
            (allowed_api_ids - excluded_apis) & remote_ids,
            live_tool_ids,
        )
    if fetched is None:
        return candidates
    return [*candidates, fetched]


# How many trailing HUMAN turns establish automation-creation context for
# follow-up turns: the current message plus two prior user exchanges. Counting
# human turns (not raw messages) keeps the window stable when a single turn
# emits many AI/tool messages (up to _MAX_ACTION_ROUNDS tool rounds), which
# would otherwise push the intent message out of a fixed-size slice. The
# footprint is real: while context is active, add_automation stays bound on
# every turn (one extra store lookup plus its description's prompt tokens).
_AUTOMATION_CONTEXT_HUMAN_TURNS = 3


def _conversation_has_automation_context(
    messages: Sequence[BaseMessage], summary: str = ""
) -> bool:
    """
    Return True when recent turns establish automation-creation context.

    Short continuations ("yes", "do that") carry the prior turn's intent but
    produce useless retrieval queries, so add_automation must stay bound when
    a recent human message asked for an automation or a recent turn already
    called add_automation — including malformed attempts — (field report:
    "yes" after an add_automation offer was rejected because the tool was no
    longer retrieved). The conversation summary is also scanned because
    end-of-turn trimming can delete the original intent message while the
    model still sees the automation-in-progress in the summary.
    """
    if summary and _query_wants_automation(summary):
        return True
    human_turns = 0
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            human_turns += 1
            if _query_wants_automation(_message_text(msg)):
                return True
            if human_turns >= _AUTOMATION_CONTEXT_HUMAN_TURNS:
                return False
        elif (isinstance(msg, ToolMessage) and msg.name == "add_automation") or (
            isinstance(msg, AIMessage)
            and any(
                tc.get("name") == "add_automation"
                for tc in [*msg.tool_calls, *msg.invalid_tool_calls]
            )
        ):
            return True
    return False


def _query_is_read_only_open_state(query: str) -> bool:
    """Return True when open/opened describes state instead of an action."""
    return bool(
        re.search(ACTUATION_KEYWORDS_REGEX, query)
        and re.search(READ_ONLY_STATE_QUERY_REGEX, query)
        and re.search(OPEN_AS_STATE_REGEX, query)
        and not re.search(NON_OPEN_ACTUATION_KEYWORDS_REGEX, query)
        and not re.search(OPEN_COMMAND_CLAUSE_REGEX, query)
    )


def _latest_open_state_query(messages: Sequence[BaseMessage]) -> str:
    """Return the active read-only open-state query, including short retries."""
    latest_human = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_human = str(msg.content)
            break

    if _query_is_read_only_open_state(latest_human):
        return latest_human

    if latest_human.strip().lower() not in _OPEN_STATE_RETRY_PHRASES:
        return ""

    for msg in reversed(messages):
        if not isinstance(msg, HumanMessage):
            continue
        query = str(msg.content)
        if query.strip().lower() in _OPEN_STATE_RETRY_PHRASES:
            continue
        return query if _query_is_read_only_open_state(query) else ""
    return ""


def _normalize_live_context_args_for_open_state(
    tool_name: str,
    tool_args: dict[str, Any],
    query: str,
    *,
    force_broad: bool = True,
) -> dict[str, Any]:
    """Normalize brittle GetLiveContext calls for read-only open-state queries."""
    if (
        not force_broad
        or not _is_live_context_tool(tool_name)
        or not _query_is_read_only_open_state(query)
    ):
        return tool_args
    return {"domain": "binary_sensor"}


def _open_state_live_context_normalization_context(
    messages: Sequence[BaseMessage],
    tool_calls: list[dict[str, Any]],
    ha_llm_api: Any = None,
) -> tuple[str, dict[str, Any] | None]:
    """Return the open-state query and first live-context call to widen."""
    query = _latest_open_state_query(messages)
    if not query:
        return "", None
    # This house's own: a remote Home Assistant's call is not widened (#843
    # review), and must not take this house's call's turn.
    first_call = next(
        (
            tc
            for tc in tool_calls
            if _is_live_context_tool(str(tc.get("name", "")))
            and not (
                tool_origin(ha_llm_api, str(tc.get("name", ""))).remote
                and _has_local_ha_api(ha_llm_api)
            )
        ),
        None,
    )
    if first_call is not None and _names_a_specific_device(
        (first_call.get("args") or {}).get("name")
    ):
        # "Is the front door open?" asks about one device. Widening the call
        # and keeping only open entries removed a closed front door from the
        # answer, so the model reported it could not find it. Widening is for
        # list questions whose name filter is a category ("Window").
        return query, None
    return query, first_call


# Words that make a GetLiveContext name filter a category rather than a device:
# "Window", "doors", "all gates" -- the brittle filters the widening exists for.
_GENERIC_OPEN_NAME_WORDS: frozenset[str] = frozenset(
    {
        "all",
        "any",
        "the",
        "open",
        "opened",
        "opening",
        "openings",
        "sensor",
        "sensors",
        "binary",
        "door",
        "doors",
        "window",
        "windows",
        "gate",
        "gates",
    }
)


def _names_a_specific_device(name: object) -> bool:
    """Return True when a name filter names a device, not a category of them."""
    words = re.findall(r"[a-z0-9]+", str(name or "").lower())
    return bool(words) and not set(words) <= _GENERIC_OPEN_NAME_WORDS


def _query_needs_actuation_safety(query: str) -> bool:
    """Return True when the query warrants force-injection of actuation tools."""
    if not re.search(ACTUATION_KEYWORDS_REGEX, query):
        return False
    return not _query_is_read_only_open_state(query)


def _tool_index_state(config: RunnableConfig) -> tuple[bool, bool, bool]:
    """Return current tool-index state, preferring live runtime data if present."""
    configurable = config.get("configurable", {})
    runtime_data = configurable.get("hga_runtime_data")
    if runtime_data is not None:
        return (
            bool(getattr(runtime_data, "tool_index_ready", True)),
            bool(getattr(runtime_data, "tool_indexing_in_progress", False)),
            bool(getattr(runtime_data, "tool_index_failed", False)),
        )
    return (
        bool(configurable.get("tool_index_ready", True)),
        bool(configurable.get("tool_indexing_in_progress", False)),
        bool(configurable.get("tool_index_failed", False)),
    )


async def _get_actuation_safety_tools(
    store: BaseStore | None,
    config: RunnableConfig,
    query: str,
    allowed_api_ids: set[str],
) -> list[RawTool]:
    """Find essential actuation tools if the query suggests actuation."""
    if store is None:
        LOGGER.warning("Store is None; skipping actuation safety tools")
        return []

    tool_index_ready, _tool_indexing_in_progress, _tool_index_failed = (
        _tool_index_state(config)
    )
    if not tool_index_ready or not _query_needs_actuation_safety(query):
        if not tool_index_ready and _query_needs_actuation_safety(query):
            LOGGER.debug("Skipping actuation safety tools: tool index is not ready")
        return []

    try:
        # Score actuation tools against the query so the most relevant ones are
        # ranked first.  This lets the merge drop low-relevance actuation tools
        # (e.g. alarm_control for a "turn on light" query) and free those slots
        # for non-actuation RAG tools (e.g. camera analysis).
        results = await store.asearch(
            ("system", "tools"),
            query=query,
            filter={"is_actuation": True},
        )
    except (
        InvalidNamespaceError,
        psycopg.OperationalError,
        psycopg.ProgrammingError,
        psycopg.DataError,
        ValueError,
    ) as err:
        LOGGER.warning("Deterministic safety tool filter failed (known error): %s", err)
        try:
            results = await store.asearch(("system", "tools"))
        except (
            InvalidNamespaceError,
            psycopg.OperationalError,
            psycopg.ProgrammingError,
            psycopg.DataError,
            ValueError,
        ) as err2:
            LOGGER.warning(
                "Actuation safety fallback list failed (known error): %s", err2
            )
            return []
        except Exception:
            LOGGER.exception("Unexpected actuation safety fallback failure")
            return []
    except Exception:
        LOGGER.exception("Unexpected deterministic safety tool filter failure")
        return []

    # Sort by relevance score descending so the merge can cap to the top-N.
    sorted_results = sorted(
        results, key=lambda i: getattr(i, "score", None) or 0.0, reverse=True
    )
    safety_tools: list[RawTool] = []
    for item in sorted_results:
        val = item.value
        name = val["name"]
        is_actuation = val.get("is_actuation", False)

        # Fallback check if the store's filter was ignored or is_actuation is missing.
        if not is_actuation:
            is_actuation = is_actuation_tool(name)

        if is_actuation and val.get("api_id") in allowed_api_ids:
            safety_tools.append(
                RawTool(
                    name=name,
                    api_id=val["api_id"],
                    description=val["description"],
                    parameters=val["parameters"],
                    is_actuation=True,
                )
            )
    return safety_tools


# Keyword ranking for the no-index fallback (issue #757). Without it the
# fallback bound the first `limit` tools in API declaration order whatever the
# user asked, so a tool past that slice — every exposed script, which HA
# declares after its intent tools — was unreachable for as long as embeddings
# were down. Deliberately crude: it only has to lift the tool the user named
# above the ones they did not, in any language whose words are space-separated.
_FALLBACK_WORD_RE = re.compile(r"[^\W_]+")
_FALLBACK_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_FALLBACK_STOPWORDS = frozenset(
    {
        "and", "are", "can", "could", "does", "for", "from", "have", "how",
        "into", "its", "me", "please", "that", "the", "this", "what", "when",
        "which", "who", "why", "will", "with", "would", "you", "your",
    }
)  # fmt: skip
_FALLBACK_MIN_WORD_LEN = 3
# A query word found in more than half the tools ("device", "entity") says
# nothing about which one the user wants; below this many tools that cutoff
# would discard nearly every word, so it is not applied.
_FALLBACK_COMMON_WORD_MIN_TOOLS = 4
_FALLBACK_NAME_WEIGHT = 2
# Query and tool text are cut before tokenizing, so a pasted wall of text or
# an MCP server's oversized tool description cannot stall the event loop.
_FALLBACK_MAX_TEXT_CHARS = 2000


def _fallback_words(text: str | None) -> set[str]:
    """Return the normalized words of a query, tool name, or description."""
    words: set[str] = set()
    # A third-party tool can declare an explicit null description.
    for raw in _FALLBACK_WORD_RE.findall(_FALLBACK_CAMEL_RE.sub(" ", text or "")):
        word = raw.casefold()
        if len(word) < _FALLBACK_MIN_WORD_LEN or word in _FALLBACK_STOPWORDS:
            continue
        # Plural-insensitive on both sides: "doors" finds script__door_history.
        # A double "s" is not a plural ("hass", "glass", "access").
        if (
            len(word) > _FALLBACK_MIN_WORD_LEN
            and word.endswith("s")
            and not word.endswith("ss")
        ):
            word = word[:-1]
        words.add(word)
    return words


def _rank_fallback_by_query(
    fallback: list[RawTool], query: str
) -> tuple[list[RawTool], int]:
    """
    Stable-sort fallback tools by keyword overlap with the query.

    A query word in a tool's name scores double one in its description only.
    A remote Home Assistant's house words count as description, so naming
    that house ranks its tools above this one's namesakes (#848); they are
    never too common to count, since one house can offer most of the tools.
    Tools that tie (including every tool when nothing matches) keep their
    incoming order, so the caller's actuation / GetLiveContext ordering is the
    tiebreak. Returns the ranked list and how many tools matched at all.
    """
    if not fallback:
        return fallback, 0
    house_words = [_fallback_words(t.get("house")) for t in fallback]
    tool_words = [
        (
            _fallback_words(t["name"][:_FALLBACK_MAX_TEXT_CHARS]),
            _fallback_words((t["description"] or "")[:_FALLBACK_MAX_TEXT_CHARS])
            | house,
        )
        for t, house in zip(fallback, house_words, strict=True)
    ]
    # How many tools each word appears in. Every pass below is linear in the
    # tools' words, so a long pasted query costs no more than its own
    # tokenizing: words no tool uses are dropped before any scoring.
    doc_freq = Counter(word for name, desc in tool_words for word in name | desc)
    cutoff = (
        len(fallback) // 2
        if len(fallback) >= _FALLBACK_COMMON_WORD_MIN_TOOLS
        else len(fallback)
    )
    # The newest text is at the end of the ranking query, so keep the tail.
    any_house = set().union(*house_words)
    informative = {
        word
        for word in _fallback_words(query[-_FALLBACK_MAX_TEXT_CHARS:])
        if 0 < doc_freq[word] <= cutoff or word in any_house
    }
    if not informative:
        return fallback, 0
    scores = [
        _FALLBACK_NAME_WEIGHT * len(name & informative)
        + len((desc - name) & informative)
        for name, desc in tool_words
    ]
    order = sorted(range(len(fallback)), key=lambda i: -scores[i])
    return [fallback[i] for i in order], sum(1 for s in scores if s)


def _fallback_slice(
    ranked: list[RawTool], limit: int, *, actuation_quota: int, matched: int
) -> list[RawTool]:
    """
    Cut the ranked fallback to ``limit``, reserving slots for actuation tools.

    On an actuation request the best-ranked ``actuation_quota`` actuation
    tools are kept even when keyword matches outrank them, so tools that share
    words with the request (three ``garage_*`` scripts for "open the garage
    door") cannot crowd HassTurnOn out. The remaining slots go to the best
    matches of any kind, which is how a script named in "run the door history
    script" (an actuation request; scripts are not actuation tools) still
    binds. ``matched`` is how many tools lead ``ranked`` on a keyword match.
    Ranked order is kept. With no matches the actuation tools lead the
    ranking anyway, so the result is the first ``limit`` tools.
    """
    reserved = {
        id(t) for t in [t for t in ranked if t["is_actuation"]][:actuation_quota]
    }
    others = [t for t in ranked if id(t) not in reserved]
    rest = others[: max(0, limit - len(reserved))]
    if (
        not rest
        and matched
        and others
        and id(others[0]) in {id(t) for t in ranked[:matched]}
    ):
        # The quota filled the whole limit (a limit of 1): the best keyword
        # match still gets one slot past it, as the RAG path's safety slots
        # sit on top of its limit, or a named script could never bind.
        rest = others[:1]
    keep = reserved | {id(t) for t in rest}
    return [t for t in ranked if id(t) in keep]


def _get_fallback_tools(
    config: RunnableConfig,
    allowed_api_ids: set[str],
) -> list[RawTool]:
    """Gather all available tools as a fallback when none are selected."""
    fallback_tools: list[RawTool] = []

    ha_llm_api = config.get("configurable", {}).get("ha_llm_api")
    if ha_llm_api and hasattr(ha_llm_api, "apis"):
        houses = {
            api_id: words
            for api_id, api_name in remote_api_names(ha_llm_api).items()
            if (words := remote_house_words(api_name))
        }
        for api_id, api in ha_llm_api.apis.items():
            if api_id in allowed_api_ids:
                for tool in api.tools:
                    formatted = format_tool(tool, api.custom_serializer)
                    f_func = formatted["function"]
                    name = tool.name
                    raw = RawTool(
                        name=name,
                        api_id=api_id,
                        description=f_func.get("description", ""),
                        parameters=json.dumps(f_func["parameters"]),
                        is_actuation=is_actuation_tool(name),
                    )
                    if api_id in houses:
                        raw["house"] = houses[api_id]
                    fallback_tools.append(raw)

    # Gated like the HA APIs above: every caller filters results by api_id
    # membership anyway, so formatting hga_local tools for a lookup restricted
    # to one MCP api (_get_tool_by_name's miss path, run per always-included
    # tool) would json.dumps every local tool's pydantic schema just to have
    # the caller discard it.
    if "hga_local" not in allowed_api_ids:
        return fallback_tools
    langchain_tools = config.get("configurable", {}).get("langchain_tools", {})
    for name, lc_tool in langchain_tools.items():
        fallback_tools.append(
            RawTool(
                name=name,
                api_id="hga_local",
                description=lc_tool.description,
                parameters=langchain_tool_parameters_json(lc_tool),
                is_actuation=is_actuation_tool(name),
            )
        )
    return fallback_tools


def _tool_retrieval_fallback_reason(
    *,
    store: BaseStore | None,
    config: RunnableConfig,
) -> tuple[str, bool, bool, bool]:
    """Return a concise reason for falling back to keyword-ranked tools."""
    tool_index_ready, tool_indexing_in_progress, tool_index_failed = _tool_index_state(
        config
    )
    if store is None:
        reason = "store unavailable"
    elif tool_index_failed:
        reason = "tool index failed"
    elif not tool_index_ready:
        reason = (
            "tool index still building"
            if tool_indexing_in_progress
            else "tool index not ready"
        )
    else:
        reason = "vector search returned no usable tool candidates"
    return reason, tool_index_ready, tool_indexing_in_progress, tool_index_failed


def _ensure_array_items(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively ensure every array-type node has an ``items`` field.

    Gemini rejects function declarations where an array property lacks ``items``.
    OpenAI tolerates the omission; this normalisation is a no-op for other providers.

    Also handles anyOf schemas such as ``vol.Any(cv.string, [cv.string])`` which
    safe_convert emits as ``{"anyOf": [{}, {"type": "array", "items": {...}}]}``.
    langchain_google_genai resolves these to type_:ARRAY but then checks
    ``v.get("items")`` on the outer anyOf dict (where items is absent), so it
    never populates properties_item["items"].  Hoisting items to the top level
    of the anyOf schema makes it visible to the library's check.
    """
    # anyOf with an array variant: hoist items upward so langchain_google_genai
    # can find them when it resolves the anyOf to type_:ARRAY.
    if "anyOf" in schema and "items" not in schema:
        array_variants = [
            v
            for v in schema["anyOf"]
            if isinstance(v, dict) and v.get("type") == "array"
        ]
        if array_variants:
            schema = {
                **schema,
                "items": array_variants[-1].get("items", {"type": "string"}),
            }
    if schema.get("type") == "array" and "items" not in schema:
        schema = {**schema, "items": {"type": "string"}}
    if "properties" in schema:
        schema = {
            **schema,
            "properties": {
                k: _ensure_array_items(v) for k, v in schema["properties"].items()
            },
        }
    if isinstance(schema.get("items"), dict):
        schema = {**schema, "items": _ensure_array_items(schema["items"])}
    return schema


def _filter_variant_required(
    variant: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """
    Drop the ``required`` entries Gemini rejects, returning what was lost.

    A variant carrying ``properties`` but no explicit ``type`` is treated as an
    object: both current producers emit the type, but the filter leg is the
    correct one for an implicit object.
    """
    required = [r for r in variant["required"] if isinstance(r, str)]
    is_object = variant.get("type") == "object" or "properties" in variant
    kept = (
        [r for r in required if r in variant.get("properties", {})] if is_object else []
    )
    cleaned = {k: v for k, v in variant.items() if k != "required"}
    if kept:
        cleaned["required"] = kept
    return cleaned, [r for r in required if r not in kept]


def _append_requirement_hint(
    schema: dict[str, Any], dropped: list[list[str]]
) -> dict[str, Any]:
    """
    Restate a dropped union (anyOf/oneOf) constraint in the description.

    Used by both the Gemini sanitizer and the OpenAI top-level-union
    flattener: the provider cannot validate the constraint, but the model
    does read descriptions, so surfacing it here keeps the model from
    emitting a call that the stripped schema permits and Home Assistant's
    own validation then rejects.
    """
    if all(len(group) == 1 for group in dropped):
        names = list(dict.fromkeys(group[0] for group in dropped))
        hint = f"At least one of: {', '.join(names)} is required."
    else:
        groups = "; ".join(f"({', '.join(group)})" for group in dropped)
        hint = f"At least one of these property groups is required: {groups}."
    existing = schema.get("description")
    return {
        **schema,
        "description": f"{existing} {hint}" if existing else hint,
    }


def _strip_additional_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Remove ``additionalProperties`` keys from a tool schema, recursively.

    Probatio's ``to_openapi`` (the converter on HA 2026.9 cores) emits
    ``additionalProperties: false`` on every closed object schema —
    voluptuous-openapi never did. langchain-google-genai marshals tool
    schemas into the gapic ``Schema`` proto, which has no such field, and
    logs a WARNING per unrecognised key, so on a 2026.9 core every Gemini
    turn floods the HA log with one warning per bound tool. The key is
    purely advisory for Gemini (the proto cannot express it either way), so
    stripping loses nothing. Gated to Gemini in
    ``_format_and_dedupe_tools()`` because other providers accept the key —
    OpenAI's strict mode even requires it.
    """

    def _strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                key: _strip(value)
                for key, value in node.items()
                if key != "additionalProperties"
            }
        if isinstance(node, list):
            return [_strip(item) for item in node]
        return node

    return _strip(schema)


def _sanitize_any_of_required(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively strip ``required`` entries that Gemini's validation rejects.

    Gemini rejects ``required`` on a non-OBJECT ``anyOf`` variant, and rejects
    entries naming a property that variant does not itself define::

        parameters.any_of[0].required: only allowed for OBJECT type
        parameters.any_of[0].required[0]: property is not defined

    voluptuous_openapi emits exactly that shape for Home Assistant's
    "at least one target" pattern — ``vol.Required(vol.Any("name", "area",
    "floor"))`` becomes parent-level ``properties`` plus bare-``required``
    variants::

        {"anyOf": [{"required": ["name"]}, {"required": ["area"]}]}

    Those variants are valid JSON Schema honoured by Anthropic and Ollama
    (OpenAI rejects them only at the top level of ``parameters`` — see
    ``_flatten_top_level_union()``), so unlike the purely additive
    ``_ensure_array_items()`` this subtractive pass is scoped to Gemini by
    ``_format_and_dedupe_tools()``.
    Whatever it has to drop is restated in the description so the constraint
    survives as guidance even though it cannot survive in the proto schema.
    """
    if not isinstance(schema, dict):
        # A boolean schema (``"freeform": true``) is valid JSON Schema.
        return schema
    if isinstance(schema.get("anyOf"), list):
        variants: list[Any] = []
        dropped: list[list[str]] = []
        for raw_variant in schema["anyOf"]:
            if not isinstance(raw_variant, dict):
                variants.append(raw_variant)
                continue
            variant = _sanitize_any_of_required(raw_variant)
            if isinstance(variant.get("required"), list):
                variant, lost = _filter_variant_required(variant)
                if lost:
                    dropped.append(lost)
            # A variant emptied by the strip carries no constraint at all;
            # langchain_google_genai passes anyOf straight to the gapic proto,
            # so leaving {} behind would reach Gemini as TYPE_UNSPECIFIED.
            if variant:
                variants.append(variant)
        schema = (
            {**schema, "anyOf": variants}
            if variants
            else {k: v for k, v in schema.items() if k != "anyOf"}
        )
        if dropped:
            schema = _append_requirement_hint(schema, dropped)
    if "properties" in schema:
        schema = {
            **schema,
            "properties": {
                k: _sanitize_any_of_required(v) for k, v in schema["properties"].items()
            },
        }
    if isinstance(schema.get("items"), dict):
        schema = {**schema, "items": _sanitize_any_of_required(schema["items"])}
    return schema


_TOP_LEVEL_UNION_KEYS = ("oneOf", "anyOf", "allOf")

# Providers whose function-calling validator rejects oneOf/anyOf/allOf at the
# top level of `parameters`. openai_compatible is included because those
# endpoints proxy or emulate OpenAI's validation. Anthropic's Messages API
# rejects the same shape on its own account ("tools.N.custom.input_schema:
# input_schema does not support oneOf, allOf, or anyOf at the top level",
# issue #585) — Home Assistant's built-in Anthropic integration strips those
# keys outright before dispatch. The schema loss is recovered via the
# description hint either way.
_FLATTEN_UNION_PROVIDERS = ("anthropic", "openai", "openai_compatible")


def _merge_conjunctive_required(
    schema: dict[str, Any], conjunctive_required: list[str]
) -> dict[str, Any]:
    """
    Merge allOf-derived ``required`` names into the top-level ``required``.

    Entries naming a property the merged schema does not define are dropped;
    non-list or non-string pre-existing entries are ignored rather than
    raised.
    """
    raw_existing = schema.get("required")
    existing = (
        [r for r in raw_existing if isinstance(r, str)]
        if isinstance(raw_existing, list)
        else []
    )
    merged = dict.fromkeys(existing + conjunctive_required)
    filtered = [r for r in merged if r in schema["properties"]]
    if filtered:
        schema["required"] = filtered
    return schema


def _flatten_top_level_union(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten a top-level oneOf/anyOf/allOf into a single object schema.

    OpenAI's function-calling validator rejects oneOf/anyOf/allOf/enum/
    const/not at the top level of ``parameters`` ("schema must have type
    'object' and not have 'oneOf'/'anyOf'/'allOf'/'enum'/'const'/'not' at
    the top level"), and Anthropic's Messages API rejects the union keys
    the same way ("tools.N.custom.input_schema: input_schema does not
    support oneOf, allOf, or anyOf at the top level"). On HA 2026.8.2
    exactly two assist tools carry that shape — HassStartTimer and
    HassDecreaseTimer, whose
    ``vol.Required(vol.Any("hours", "minutes", "seconds"))`` slot converts
    to a top-level ``anyOf`` of bare ``{"required": [...]}`` variants — and
    both are exposed only for timer-capable satellite devices, so the 400
    surfaces only on turns whose tool set includes one of them.

    Every variant's ``properties`` are unioned into one object schema
    (pre-existing top-level properties win).  ``anyOf``/``oneOf`` variant
    ``required`` entries cannot survive flattening — the arms differ — so
    they are restated in the description via ``_append_requirement_hint()``,
    the same recovery the Gemini sanitizer uses.  ``allOf`` variants are
    conjunctive, so their ``required`` entries merge into the top-level
    ``required`` instead (entries naming a property no variant defines are
    dropped).  Malformed variant fields (non-dict ``properties``, non-list
    ``required``) are ignored rather than raised — one bad tool schema must
    not fail the whole tool-retrieval pass.

    This pass is subtractive (it deletes union structure other providers
    honour), so ``_format_and_dedupe_tools()`` gates it to
    ``_FLATTEN_UNION_PROVIDERS``.

    Scope: only the union keywords are handled, and only variant
    ``properties``/``required`` survive the merge. Both converters wrap tool
    schemas in a top-level object, so OpenAI's other forbidden top-level
    keywords (enum/const/not) and richer variant keywords (dependentRequired,
    discriminator const, ...) never occur in this pipeline's input and are
    deliberately not handled. ``additionalProperties`` is the exception:
    probatio (HA 2026.9) emits it on every closed object, and
    ``_strip_additional_properties()`` removes it for Gemini in
    ``_format_and_dedupe_tools()``.
    """
    merged_keys = [k for k in _TOP_LEVEL_UNION_KEYS if isinstance(schema.get(k), list)]
    if not merged_keys:
        return schema
    merged_properties: dict[str, Any] = {}
    conjunctive_required: list[str] = []
    dropped: list[list[str]] = []
    for key in merged_keys:
        for variant in schema[key]:
            if not isinstance(variant, dict):
                continue
            variant_properties = variant.get("properties")
            if isinstance(variant_properties, dict):
                merged_properties.update(variant_properties)
            raw_required = variant.get("required")
            if not isinstance(raw_required, list):
                continue
            required = [r for r in raw_required if isinstance(r, str)]
            if not required:
                continue
            if key == "allOf":
                conjunctive_required.extend(required)
            else:
                dropped.append(required)
    schema = {k: v for k, v in schema.items() if k not in merged_keys}
    schema["type"] = "object"
    existing_properties = schema.get("properties")
    if not isinstance(existing_properties, dict):
        existing_properties = {}
    schema["properties"] = {**merged_properties, **existing_properties}
    if conjunctive_required:
        schema = _merge_conjunctive_required(schema, conjunctive_required)
    if dropped:
        schema = _append_requirement_hint(schema, dropped)
    return schema


def _flatten_union_for_provider(
    parameters: dict[str, Any], provider: str | None, name: str
) -> dict[str, Any]:
    """Flatten a top-level union for the providers that reject one."""
    if provider not in _FLATTEN_UNION_PROVIDERS:
        return parameters
    flattened = _flatten_top_level_union(parameters)
    if flattened is not parameters:
        LOGGER.debug(
            "Flattened top-level union in %s tool schema for %s", name, provider
        )
    return flattened


def _sanitize_for_gemini(
    parameters: dict[str, Any], provider: str | None
) -> dict[str, Any]:
    """Apply the Gemini-only subtractive schema passes."""
    if provider != "gemini":
        return parameters
    return _strip_additional_properties(_sanitize_any_of_required(parameters))


# ``_llm_type`` of each chat model class the integration builds, mapped to the
# provider name the schema passes are gated on. ``openai_compatible`` shares
# ChatOpenAI and the same passes, so it needs no entry of its own.
_LLM_TYPE_PROVIDERS = {
    "chat-google-generative-ai": "gemini",
    "openai-chat": "openai",
    ANTHROPIC_LLM_TYPE: "anthropic",
}


def _tools_for_chain_member(model: Any, tools: list[Any]) -> list[Any]:
    """
    Re-normalise formatted tools for the model they are about to be bound to.

    ``_format_and_dedupe_tools()`` shapes schemas for the configured primary
    provider, but a mixed-provider chain binds the list to every member: an
    Ollama primary failing over to Gemini handed Gemini the un-sanitized
    ``anyOf`` and the turn died on a 400. Both passes are idempotent, so
    running them on the primary's own list changes nothing.
    """
    concrete = concrete_chat_model(model)
    provider = _LLM_TYPE_PROVIDERS.get(getattr(concrete, "_llm_type", ""))
    if provider is None:
        return tools
    adapted: list[Any] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        parameters = function.get("parameters") if isinstance(function, dict) else None
        if not isinstance(function, dict) or not isinstance(parameters, dict):
            adapted.append(tool)
            continue
        name = str(function.get("name"))
        try:
            parameters = _flatten_union_for_provider(parameters, provider, name)
            parameters = _sanitize_for_gemini(parameters, provider)
        except Exception:
            # Binding runs for every chain member on every turn, so a schema
            # one member's pass cannot digest must not cost a healthy primary
            # the turn. Bind it as-is; the provider's own rejection is handled
            # by the drop-and-retry in _invoke_chat_model_with_schema_recovery.
            LOGGER.exception(
                "Could not shape the schema of tool %s for %s; binding it unchanged",
                sanitize_tool_text(name, limit=200),
                provider,
            )
            adapted.append(tool)
            continue
        adapted.append({**tool, "function": {**function, "parameters": parameters}})
    return adapted


def _format_and_dedupe_tools(
    raw_tools: list[RawTool],
    provider: str | None = None,
    remote_apis: Mapping[str, str] | None = None,
    aliases: Mapping[tuple[str, str], str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """
    Convert raw tools to output format and deduplicate (first-seen wins).

    A remote Home Assistant's tools (`remote_apis`: id to name) are named
    after their server (`aliases`, by API id and own name), and their
    description says so: it offers its tools under this Home Assistant's
    names, and the agent can be given one tool per name, so a command could
    reach the wrong house (#843).
    """
    selected_tools: list[dict[str, Any]] = []
    routing_map: dict[str, str] = {}

    for tool in raw_tools:
        name = tool["name"]
        description = tool["description"]
        api_name = (remote_apis or {}).get(tool["api_id"])
        if api_name is not None:
            name = (aliases or {}).get((tool["api_id"], name)) or remote_tool_name(
                api_name, name
            )
            description = f"{remote_tool_label(api_name)} {description}"
        if name in routing_map:
            continue

        # Parse BEFORE claiming the route: a corrupt stored row (partial
        # write, migration bug) must cost one skipped tool, not the whole
        # turn. Unguarded, a bad row for a force-bound tool (GetLiveContext,
        # an always-included tool) crashed _retrieve_tools on EVERY turn
        # until the row was repaired. Skipping also must not leave a phantom
        # routing entry, or dispatch would route calls to a tool that was
        # never offered to the model.
        try:
            parameters = json.loads(tool["parameters"])
        except (TypeError, ValueError):
            LOGGER.warning(
                "Skipping tool %s (api %s): stored parameters are not valid JSON",
                sanitize_tool_text(name, limit=200),
                tool["api_id"],
            )
            continue
        routing_map[name] = tool["api_id"]
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        else:
            # OpenAI and Anthropic both reject oneOf/anyOf/allOf at the top
            # level of `parameters` (HassStartTimer / HassDecreaseTimer).
            # Subtractive, so gated — Ollama honours the union and keeps it,
            # and the Gemini pass below needs the anyOf intact to restate it.
            parameters = _flatten_union_for_provider(parameters, provider, name)
            if not parameters.get("type"):
                parameters["type"] = "object"
        # OpenAI requires 'properties' on all type:object schemas.
        if parameters.get("type") == "object" and "properties" not in parameters:
            parameters["properties"] = {}
        # Gemini requires 'items' on all type:array properties.
        parameters = _ensure_array_items(parameters)
        # Gemini alone rejects 'required' inside anyOf branches. This pass is
        # subtractive — it would delete HA's "at least one target" constraint
        # for providers that can honour it — so it is gated on the provider.
        parameters = _sanitize_for_gemini(parameters, provider)
        selected_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )

    # Bind in name order. Retrieval ranks candidates by per-query relevance,
    # so the same set of tools arrives in a different order on different
    # turns; providers cache by exact prefix and the tool array is the head
    # of that prefix, so an order change alone forfeits the whole cache
    # (issue #617). Dedupe above already fixed which api_id owns each name,
    # and dispatch routes by name, so the order carries no meaning here.
    selected_tools.sort(key=lambda t: str(t["function"]["name"]))

    return selected_tools, routing_map


# ----- Graph nodes and edges -----


def _last_requires_pin_idx(recent: Sequence[AnyMessage]) -> int | None:
    """Return the index of the most recent unresolved requires_pin ToolMessage."""
    idx: int | None = None
    for i, msg in enumerate(recent):
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content
        if not isinstance(content, str):
            continue
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict) and data.get("status") == "requires_pin":
            idx = i
    return idx


def _last_routing_rejection_idx(recent: Sequence[AnyMessage]) -> int | None:
    """Return index of the last unresolved routing-rejected actuation ToolMessage."""
    idx: int | None = None
    for i, msg in enumerate(recent):
        if not isinstance(msg, ToolMessage):
            continue
        if not is_actuation_tool(msg.name or ""):
            continue
        content = msg.content if isinstance(msg.content, str) else ""
        if msg.status == "error" and _ROUTING_REJECTION_MARKER in content:
            idx = i
        else:
            idx = None  # tool ran (success or other error) — rejection resolved
    return idx


def _has_pending_pin_confirmation(messages: Sequence[AnyMessage]) -> bool:
    """
    Return True if a PIN confirmation is pending in recent message history.

    Detects two cases:

    Case 1 (normal flow): A ``requires_pin`` ToolMessage exists with no later
    ``confirm_sensitive_action`` reply that resolved it. A wrong PIN, an
    ``awaiting_user_pin`` refusal or a routing error leave it pending.

    Case 2 (degraded flow): An actuation tool was routing-rejected and the model
    subsequently asked for a PIN in an AI message.  Injecting
    ``confirm_sensitive_action`` lets the model recover on the next turn.
    """
    recent = messages[-_PIN_LOOKBACK:] if len(messages) > _PIN_LOOKBACK else messages

    pin_idx = _last_requires_pin_idx(recent)
    if pin_idx is not None:
        for msg in recent[pin_idx + 1 :]:
            if isinstance(msg, ToolMessage) and msg.name == CONFIRM_TOOL_NAME:
                content = msg.content if isinstance(msg.content, str) else ""
                if "Incorrect PIN" in content:
                    # Wrong PIN entered — action is still pending, keep scanning.
                    continue
                if pin_refusal(content) is not None:
                    # Refused before the PIN was checked (not in the user's
                    # message, or the message already used) — still pending.
                    continue
                if getattr(msg, "status", None) == "error":
                    # Routing or tool error on confirm_sensitive_action itself —
                    # the PIN was never checked. Keep the action pending.
                    continue
                return False
        return True

    rejection_idx = _last_routing_rejection_idx(recent)
    if rejection_idx is None:
        return False

    # AI asked for PIN after routing rejection but before a successful actuation call.
    for msg in recent[rejection_idx + 1 :]:
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            content = msg.content if isinstance(msg.content, str) else ""
            if "pin" in content.lower():
                return True

    return False


async def _get_pending_pin_tools(
    messages: list[AnyMessage],
    store: BaseStore | None,
    config: RunnableConfig | None = None,
) -> list[RawTool]:
    """Return [confirm_sensitive_action] when a PIN confirmation is pending."""
    cfg = (config or {}).get("configurable", {})
    # The store, not only the history: the history is trimmed to a lookback
    # window and its detector reads English text, while the store holds the
    # conversation's action until its reply turn ends.
    if not (
        _has_pending_pin_confirmation(messages)
        or thread_has_pending_action(
            cfg.get("pending_actions", {}), cfg.get("thread_id")
        )
    ):
        return []

    if store is None:
        return []

    try:
        item = await store.aget(
            ("system", "tools"), key="hga_local::confirm_sensitive_action"
        )
    except Exception:  # noqa: BLE001
        LOGGER.debug(
            "Could not fetch confirm_sensitive_action from store for PIN injection"
        )
        return []

    if item is None:
        return []

    val = item.value
    # Validate the stored row before binding it into the security-critical
    # PIN flow: a colliding composite key (or an API registering as
    # hga_local) must not swap the confirmation tool's schema/description.
    if (
        val.get("name") != "confirm_sensitive_action"
        or val.get("api_id") != "hga_local"
    ):
        LOGGER.warning(
            "PIN injection skipped: stored confirm_sensitive_action row "
            "failed validation (name=%r api_id=%r)",
            val.get("name"),
            val.get("api_id"),
        )
        return []
    LOGGER.debug("PIN flow active: force-injecting confirm_sensitive_action")
    tools = [
        RawTool(
            name=val["name"],
            api_id=val["api_id"],
            description=val["description"],
            parameters=val["parameters"],
            is_actuation=val.get("is_actuation", False),
        )
    ]
    if config is not None:
        tools.extend(await _get_held_action_tools(store, config, cfg))
    return tools


async def _get_held_action_tools(
    store: BaseStore, config: RunnableConfig, cfg: Mapping[str, Any]
) -> list[RawTool]:
    """
    Return the tools of this conversation's pending actions.

    The design lets the model re-issue the held request in the PIN reply turn
    (after a wrong PIN, or when it believes the prompt expired), but the reply
    itself ("1234") ranks no actuation tool, so the re-issue was rejected as
    not retrieved (field test 2026-10-15). The pending action names its tool.
    """
    thread_id = cfg.get("thread_id")
    if not thread_id:
        return []
    now = dt_util.utcnow()
    # Each held tool by its own name at the API it was held for: a remote
    # Home Assistant's tool reaches the agent under its server's name, and
    # looking that name up anywhere else found this house's namesake (#843
    # review).
    held = sorted(
        {
            (
                str(action.get("api_tool_name") or action["tool_name"]),
                str(action["api_id"]) if action.get("api_id") else "",
            )
            for action in cfg.get("pending_actions", {}).values()
            if action.get("thread_id") == thread_id
            and action.get("tool_name")
            and action.get("tool_name") != "add_automation"
            and not pending_action_expired(action, now)
        }
    )
    if not held:
        return []
    allowed_api_ids = _get_allowed_api_ids(config)
    live_tool_ids = _get_live_tool_ids(config)
    excluded_names = tool_exclusions(config.get("configurable", {}).get("options", {}))
    tools: list[RawTool] = []
    for name, api_id in held:
        apis = {api_id} & allowed_api_ids if api_id else allowed_api_ids
        if not apis:
            continue
        fetched = await _get_tool_by_name(store, config, name, apis, live_tool_ids)
        if fetched is not None and not is_tool_excluded(
            fetched["name"], excluded_names.get(fetched["api_id"], ())
        ):
            tools.append(fetched)
    return tools


def _tool_lookup_targets(
    name: str,
    allowed_api_ids: set[str],
    live_tool_ids: set[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    """
    Return the (api_id, tool_name) index keys to try for a caller-supplied name.

    Force-injection sites name their tool the way it is written in the code
    ("GetLiveContext"), but HA publishes it namespaced
    ("homeassistant__GetLiveContext"). The bare key still resolves -- to the
    pre-2026.9 index row, which the live check below then rejects -- so the
    injection silently did nothing on an upgraded install. Appending every live
    tool whose base name matches makes the lookup survive the rename in both
    directions: the bare key still wins on an install that has not upgraded,
    and the namespaced one is found on an install that has.
    """
    targets = [(api_id, name) for api_id in sorted(allowed_api_ids)]
    if live_tool_ids is None:
        return targets
    return [
        *targets,
        *sorted(
            (api_id, live_name)
            for api_id, live_name in live_tool_ids
            if api_id in allowed_api_ids
            and live_name != name
            and base_tool_name(live_name) == name
        ),
    ]


async def _get_tool_by_name(
    store: BaseStore,
    config: RunnableConfig,
    name: str,
    allowed_api_ids: set[str],
    live_tool_ids: set[tuple[str, str]] | None = None,
) -> RawTool | None:
    """Fetch a specific tool by name from the index, falling back to config tools."""
    for api_id, lookup_name in _tool_lookup_targets(
        name, allowed_api_ids, live_tool_ids
    ):
        try:
            item = await store.aget(
                ("system", "tools"), key=tool_index_key(api_id, lookup_name)
            )
        except (
            InvalidNamespaceError,
            psycopg.OperationalError,
            psycopg.ProgrammingError,
            ValueError,
        ) as err:
            LOGGER.debug(
                "Could not fetch %s from tool index: %s",
                sanitize_tool_text(name, limit=200),
                err,
            )
            break
        except Exception:
            LOGGER.exception(
                "Unexpected failure fetching %s from tool index",
                sanitize_tool_text(name, limit=200),
            )
            break

        if item is None:
            continue

        val = item.value
        if val.get("name") != lookup_name or val.get("api_id") not in allowed_api_ids:
            continue

        if (
            live_tool_ids is not None
            and (val["api_id"], val["name"]) not in live_tool_ids
        ):
            continue

        return RawTool(
            name=val["name"],
            api_id=val["api_id"],
            description=val.get("description", ""),
            parameters=val.get("parameters", "{}"),
            is_actuation=val.get("is_actuation", False),
        )

    # Base-name match: the fallback set is built from the tools live this turn,
    # so it carries HA's namespaced spelling while callers pass the bare one.
    # The exact name first: `script__morning` and `intent_script__morning`
    # share a base name (#843 review).
    fallback = [
        tool
        for tool in _get_fallback_tools(config, allowed_api_ids)
        if tool["api_id"] in allowed_api_ids
    ]
    if name != base_tool_name(name):
        for tool in fallback:
            if tool["name"] == name:
                return tool
    for tool in fallback:
        if base_tool_name(tool["name"]) == base_tool_name(name):
            return tool

    return None


async def _append_included_tools(
    all_candidates: list[RawTool],
    store: BaseStore,
    config: RunnableConfig,
    live_tool_ids: set[tuple[str, str]] | None,
) -> list[RawTool]:
    """
    Append the user's always-included tools to the candidate list.

    Three filters still apply to an inclusion — the API must be active this
    turn, the live-tool filter (via ``_get_tool_by_name``) must know the tool,
    and an exclusion beats an inclusion. The exclusion check looks redundant
    (an excluded tool is already stripped from the ``APIInstance``, so
    ``live_tool_ids`` cannot carry it) but is not: when the live filter is
    fail-open (``live_tool_ids`` is None) ``_get_tool_by_name`` reads the raw
    index, which is never pruned on exclusion — and a security control must
    not depend on an optional filter being wired.
    """
    opts = config.get("configurable", {}).get("options", {})
    allowed_api_ids = _get_allowed_api_ids(config)
    excluded_names = tool_exclusions(opts)
    bound_api_by_name = {t["name"]: t["api_id"] for t in all_candidates}
    bound = {(t["api_id"], t["name"]) for t in all_candidates}
    # A remote Home Assistant's tools reach the agent under their server's
    # name, so one never shadows a same-named tool of another API (#843).
    remote_ids = set(remote_api_names(config.get("configurable", {}).get("ha_llm_api")))
    inclusions = tool_inclusions(opts)
    if len(inclusions) > TOOL_INCLUSIONS_MAX_PER_TURN:
        # Only a programmatic options write can get here — the picker offers
        # real tools and the docs say to pin one or two. Without the cap each
        # entry is a sequential store round-trip per turn plus prompt tokens,
        # so a degenerate map would stall every turn and overflow the
        # provider's tool limit.
        LOGGER.warning(
            "Honouring only the first %d of %d configured always-included "
            "tools this turn",
            TOOL_INCLUSIONS_MAX_PER_TURN,
            len(inclusions),
        )
        inclusions = inclusions[:TOOL_INCLUSIONS_MAX_PER_TURN]
    for api_id, name in inclusions:
        if api_id not in allowed_api_ids or is_tool_excluded(
            name, excluded_names.get(api_id, ())
        ):
            continue
        # Dedupe by bare name between this house's APIs: _format_and_dedupe_tools
        # routes by name (first seen wins), so a same-named tool from another
        # of them would be dropped there anyway — skipping saves the store
        # fetch. A remote Home Assistant's tools are renamed there (#843), so
        # one never shadows a same-named tool of another API.
        # A SAME-api duplicate is the normal case (RAG already selected the
        # pinned tool) and stays quiet; a DIFFERENT api holding the name means
        # that other server's tool will answer calls meant for the pinned one,
        # which the operator cannot otherwise see — say so at warning level.
        if (api_id, name) in bound:
            continue
        bound_api = bound_api_by_name.get(name)
        if bound_api is not None and not remote_ids & {api_id, bound_api}:
            if bound_api != api_id:
                LOGGER.warning(
                    "Always-included tool %s (api %s) is shadowed by a "
                    "same-named tool from api %s, which will receive its "
                    "calls this turn",
                    sanitize_tool_text(name, limit=200),
                    api_id,
                    bound_api,
                )
            continue
        fetched = await _get_tool_by_name(store, config, name, {api_id}, live_tool_ids)
        if fetched is None:
            LOGGER.debug(
                "Always-included tool %s (api %s) is not in the tool index, "
                "fallback set, or live tool list; skipping",
                sanitize_tool_text(name, limit=200),
                api_id,
            )
            continue
        all_candidates.append(fetched)
        bound_api_by_name[name] = api_id
        bound.add((api_id, name))
    return all_candidates


async def _retrieve_tools(  # noqa: PLR0915
    state: State,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> dict[str, Any]:
    """Retrieve relevant tools from the vector store and merge with essentials."""
    query = _message_text(state["messages"][-1]) if state["messages"] else ""
    # Tools are RANKED on this, which for a referential follow-up ("turn it
    # off") is the previous user turn plus this one. `query` stays the raw last
    # message everywhere below, because every intent detector past this point
    # gates behaviour and must not fire on a previous turn's words.
    ranking_query = _retrieval_query(state["messages"])

    allowed_api_ids = _get_allowed_api_ids(config)
    # A remote Home Assistant's tools never stand in for this house's own
    # when a tool is force-bound (#843 review).
    remote_ids = _remote_ids(config)
    opts = config.get("configurable", {}).get("options", {})
    limit = int(opts.get(CONF_TOOL_RETRIEVAL_LIMIT, 5))

    # 1. Gather candidates. The live set is resolved first so the RAG pass can
    # drop dead rows *before* it cuts to the retrieval limit — step 1b below
    # runs after that cut and cannot give a stolen slot back.
    live_tool_ids = _get_live_tool_ids(config)
    rag_result, safety_tools, pin_tools = await asyncio.gather(
        _get_rag_retrieved_tools(
            store, config, ranking_query, allowed_api_ids, live_tool_ids
        ),
        _get_actuation_safety_tools(store, config, ranking_query, allowed_api_ids),
        _get_pending_pin_tools(state["messages"], store, config),
    )
    rag_tools, rag_dropped_not_live = rag_result

    # 1b. Drop index candidates that are not live this turn (device-gated tools
    # indexed from another device's turn, tools of a failed API) before the
    # merge, so dead candidates never consume merge slots. The RAG leg already
    # dropped its own; this catches the safety leg and any caller that resolved
    # the live set differently.
    rag_tools, safety_tools, live_tool_ids, live_filter_dropped = (
        _filter_live_candidates(config, rag_tools, safety_tools, live_tool_ids)
    )
    live_filter_dropped += rag_dropped_not_live

    # 2. Merge: safety tools take priority; RAG fills remaining slots.
    #
    # Safety tools are now score-sorted by relevance (see _get_actuation_safety_tools).
    # Cap them at ¾ of the configured limit so low-relevance actuation tools
    # (e.g. alarm_control for "turn on light") don't crowd out non-actuation RAG
    # tools needed for other intents (e.g. camera analysis).
    #
    # effective_limit expands just enough to fit all rag_only tools that survived
    # the per-sub-query RAG budget, but is capped at safety_cap + limit to avoid
    # unbounded growth.
    #
    # pin_tools are force-injected outside the limit — they must always be present
    # when a PIN confirmation is pending, regardless of RAG scores.
    # By the name the agent sees: a remote Home Assistant's namesake is
    # another tool, another API's is not (#843).
    facing = _facing_name(config)
    pin_keys = {facing(t["api_id"], t["name"]) for t in pin_tools}
    safety_cap = min(len(safety_tools), max(1, limit * 3 // 4))
    safety_capped = [
        t
        for t in safety_tools[:safety_cap]
        if facing(t["api_id"], t["name"]) not in pin_keys
    ]
    safety_keys = {facing(t["api_id"], t["name"]) for t in safety_capped}
    rag_only = [
        t
        for t in rag_tools
        if facing(t["api_id"], t["name"]) not in safety_keys | pin_keys
    ]
    effective_limit = min(safety_cap + len(rag_only), safety_cap + limit)
    all_candidates = pin_tools + (safety_capped + rag_only)[:effective_limit]

    # 3. Fallback: vector store unavailable or index not yet ready.
    # Still apply the limit so the user-configured cap is always respected.
    # Tools are ranked by keyword overlap with the query (issue #757); ties keep
    # this order: actuation tools first for actuation queries, otherwise
    # GetLiveContext, then other read-only tools, then actuation tools.
    if not all_candidates:
        fallback = _get_fallback_tools(config, allowed_api_ids)
        (
            fallback_reason,
            tool_index_ready,
            tool_indexing_in_progress,
            tool_index_failed,
        ) = _tool_retrieval_fallback_reason(store=store, config=config)
        if live_filter_dropped:
            # Without this, the reason claims the vector search returned
            # nothing usable when it did — the live-tool filter removed it.
            fallback_reason += (
                f"; live-tool filter dropped {live_filter_dropped} candidate(s)"
            )
        if _query_needs_actuation_safety(query):
            actuation = [t for t in fallback if t["is_actuation"]]
            rest = [t for t in fallback if not t["is_actuation"]]
            fallback = actuation + rest
        else:
            live_context = [t for t in fallback if _is_live_context_tool(t["name"])]
            non_actuation = [
                t
                for t in fallback
                if not t["is_actuation"] and not _is_live_context_tool(t["name"])
            ]
            actuation = [t for t in fallback if t["is_actuation"]]
            fallback = live_context + non_actuation + actuation
        fallback, matched = _rank_fallback_by_query(fallback, ranking_query)
        LOGGER.warning(
            "Tool retrieval using keyword-ranked fallback: %s "
            "(limit=%d, total=%d, matched=%d, ready=%s, indexing=%s, failed=%s).",
            fallback_reason,
            limit,
            len(fallback),
            matched,
            tool_index_ready,
            tool_indexing_in_progress,
            tool_index_failed,
        )
        # Same share the RAG path's safety_cap gives actuation tools.
        actuation_quota = (
            max(1, limit * 3 // 4) if _query_needs_actuation_safety(query) else 0
        )
        all_candidates = _fallback_slice(
            fallback, limit, actuation_quota=actuation_quota, matched=matched
        )

    # 3b. For read-only open-state queries, force-bind GetLiveContext, promote
    # it to the front, and drop actuation tools (they are not needed).
    if _query_is_read_only_open_state(query):
        all_candidates = [
            t
            for t in all_candidates
            if not t["is_actuation"] or _is_live_context_tool(t["name"])
        ]
        if not _has_local_tool(all_candidates, remote_ids, _is_live_context_tool):
            fetched_live_ctx = await _get_local_first(
                store, config, LIVE_CONTEXT_TOOL, all_candidates, live_tool_ids
            )
            if fetched_live_ctx is not None:
                all_candidates = [fetched_live_ctx, *all_candidates]
        live_ctx = [t for t in all_candidates if _is_live_context_tool(t["name"])]
        rest = [t for t in all_candidates if not _is_live_context_tool(t["name"])]
        all_candidates = (live_ctx + rest)[: limit + 1]

    # 3c. Always inject GetLiveContext if not already present. This ensures the
    # model can evaluate any conditional clause, verify state before acting, and
    # respond accurately regardless of phrasing. Step 3b already injects it for
    # read-only open-state queries, so the deduplication check prevents doubling.
    # This house's own: a remote Home Assistant's does not answer for it
    # (#843 review).
    if not _has_local_tool(all_candidates, remote_ids, _is_live_context_tool):
        fetched_live_ctx = await _get_local_first(
            store, config, LIVE_CONTEXT_TOOL, all_candidates, live_tool_ids
        )
        if fetched_live_ctx is not None:
            all_candidates = [*all_candidates, fetched_live_ctx]

    # 3d. Force-bind add_automation for automation-creation intents. Natural
    # phrasings ("always turn on X when Y") embed closer to entity-control
    # tools than to add_automation's instruction-heavy description, so RAG
    # ranking alone cannot guarantee selection. Recent-context check keeps the
    # tool bound across short follow-ups ("yes" to an offered modification),
    # whose retrieval queries carry no intent signal. Appended outside the
    # limit, like GetLiveContext above, so it never evicts a RAG/safety
    # selection. Skipped in schema-first YAML mode, which deliberately
    # excludes add_automation from both the index and dispatch.
    if (
        not opts.get(CONF_SCHEMA_FIRST_YAML, False)
        and (
            _query_wants_automation(query)
            or _conversation_has_automation_context(
                state["messages"], state.get("summary", "")
            )
        )
        and not _has_local_tool(
            all_candidates, remote_ids, lambda name: name == "add_automation"
        )
    ):
        fetched_add_automation = await _get_tool_by_name(
            store,
            config,
            "add_automation",
            {"hga_local"} & allowed_api_ids,
            live_tool_ids,
        )
        if fetched_add_automation is not None:
            all_candidates = [*all_candidates, fetched_add_automation]
        else:
            LOGGER.debug(
                "Automation intent detected but add_automation is not in the "
                "tool index or fallback; skipping force-injection"
            )

    # 3f. Force-bind audit_home_security for security-posture questions.
    all_candidates = await _append_security_audit_tool(
        all_candidates, store, config, query, allowed_api_ids, live_tool_ids
    )

    # 3g. Force-bind get_entity_history for questions about past states,
    # including follow-ups to one. Before 3e so inclusions dedupe against it.
    all_candidates = await _append_history_tool(
        all_candidates, store, config, query, state["messages"], live_tool_ids
    )

    # 3e. Force-bind the user's always-included tools (issue #579). A
    # general-purpose tool — a web-search MCP tool on "who won the World
    # Cup?" — can score below the relevance threshold on exactly the queries
    # it exists for, and no system-prompt instruction can reach a tool that
    # was filtered out before the model saw the list. Appended outside the
    # limit, like GetLiveContext and add_automation above, so an inclusion
    # never evicts a RAG/safety selection. Deliberately after step 3b's
    # actuation strip: "always included" means always.
    all_candidates = await _append_included_tools(
        all_candidates, store, config, live_tool_ids
    )

    # 3h. Bind HassTurnOn on an actuation request ranking left it out (#795),
    # including a "Yes" to the agent's offer. Before 3i, which then binds
    # HassTurnOff with it.
    all_candidates = await _append_turn_on_tool(
        all_candidates,
        store,
        config,
        state["messages"],
        allowed_api_ids,
        live_tool_ids,
    )

    # 3i. Bind HassTurnOn and HassTurnOff together. After 3e, so an
    # always-included one gets its partner too; it dedupes by (api, name).
    all_candidates = await _append_on_off_partners(
        all_candidates, store, config, live_tool_ids
    )

    # 4. Format and deduplicate. The provider drives the subtractive schema
    # normalisation passes: the OpenAI/Anthropic top-level-union flatten
    # (_flatten_top_level_union) and the Gemini anyOf-required sanitizer
    # (_sanitize_any_of_required).
    provider_raw = opts.get(CONF_CHAT_MODEL_PROVIDER)
    ha_llm_api = config.get("configurable", {}).get("ha_llm_api")
    selected_tools, routing_map = _format_and_dedupe_tools(
        all_candidates,
        str(provider_raw) if provider_raw else None,
        remote_api_names(ha_llm_api),
        remote_aliases(ha_llm_api),
    )

    LOGGER.debug(
        "Tool retrieval: limit=%d eff=%d rag=%d safety=%d/%d pin=%d merged=%d tools=%s",
        limit,
        effective_limit,
        len(rag_tools),
        safety_cap,
        len(safety_tools),
        len(pin_tools),
        len(all_candidates),
        [t["function"]["name"] for t in selected_tools],
    )

    return {
        "selected_tools": selected_tools,
        "tool_routing_map": routing_map,
        "action_rounds": 0,
        "correction_rounds": 0,
    }


async def _trim_messages_for_model(
    messages: list[AnyMessage],
    opts: dict[str, Any],
    chat_model_options: dict[str, Any],
    hass: HomeAssistant,
) -> list[AnyMessage]:
    """Trim messages to manage context window length."""
    _known_providers = {"openai", "openai_compatible", "gemini", "ollama", "anthropic"}
    provider_raw = opts.get(CONF_CHAT_MODEL_PROVIDER)
    provider = str(provider_raw) if provider_raw else "openai"
    if provider not in _known_providers:
        LOGGER.warning(
            "Unknown model provider %r, defaulting token counter to 'openai'", provider
        )
        provider = "openai"
    model_name = _determine_model_name(provider, opts)
    manage_context_with_tokens: bool = (
        str(opts.get(CONF_MANAGE_CONTEXT_WITH_TOKENS)).lower() == "true"
    )
    context_max_messages: int = int(opts.get(CONF_MAX_MESSAGES_IN_CONTEXT) or 100)
    context_max_tokens: int = int(opts.get(CONF_MAX_TOKENS_IN_CONTEXT) or 4096)

    if manage_context_with_tokens:
        max_tokens = context_max_tokens
        token_counter = partial(
            count_tokens_cross_provider,
            model=model_name,
            provider=cast("Any", provider),
            options=opts,
            chat_model_options=chat_model_options,
        )
    else:
        max_tokens = context_max_messages
        token_counter = len

    return await hass.async_add_executor_job(
        cast(
            "Callable[..., list[AnyMessage]]",
            partial(
                trim_messages,
                messages=messages,
                token_counter=token_counter,
                max_tokens=max_tokens,
                strategy="last",
                start_on="human",
                include_system=True,
            ),
        )
    )


def _format_memories(mems: list[Any]) -> str:
    """Render retrieved memories as the <memories> block, or "" when none."""
    if not mems:
        return ""
    formatted_mems = "\n".join(f"[{mem.key}]: {mem.value}" for mem in mems)
    return f"<memories>\n{formatted_mems}\n</memories>"


def _build_system_message(
    base_prompt: str,
    volatile_prompt: str,
    memories: str,
    summary: str,
) -> SystemMessage:
    """
    Compose the system message: stable prefix first, per-turn context last.

    ``base_prompt`` is the cacheable part (instructions, tool guidance, the
    exposed-entity context). Everything that changes from turn to turn — the
    wall-clock ``volatile_prompt``, the memories retrieved for this request,
    the running summary — goes after it, so a cloud provider's exact-prefix
    cache can match the stable part (issue #617). The two parts travel as
    separate text blocks; ``adapt_system_message_for_model`` shapes them per
    provider at call time.
    """
    volatile = volatile_prompt
    if memories:
        volatile += f"\n{memories}"
    if summary:
        volatile += (
            f"\n<past_conversation_summary>\n{summary}\n</past_conversation_summary>"
        )
    return build_system_message(base_prompt, volatile.strip("\n"))


async def _invoke_model(
    model: Any,
    messages: list[AnyMessage],
    config: RunnableConfig,
    *,
    drop_unsupported: bool = True,
) -> Any:
    """Invoke a chat model, wrapping non-HA exceptions as HomeAssistantError."""
    # A fallback chain resolves to no concrete model here and shapes the
    # system message per member itself (FallbackChatModel.ainvoke).
    messages = adapt_system_message_for_model(model, messages)
    try:
        # The budget lets a fallback chain split the time between providers,
        # so a hung primary still leaves the fallback time to answer (#828).
        with model_call_budget(_LLM_INVOKE_TIMEOUT_S):
            async with asyncio.timeout(_LLM_INVOKE_TIMEOUT_S):
                if drop_unsupported:
                    return await ainvoke_dropping_unsupported_params(
                        model, messages, config
                    )
                return await model.ainvoke(messages, config)
    except TimeoutError:
        msg = f"Model invocation timed out after {_LLM_INVOKE_TIMEOUT_S}s."
        raise HomeAssistantError(msg) from None
    except HomeAssistantError:
        raise
    except Exception as err:
        msg = f"Model invocation failed: {err}"
        raise HomeAssistantError(msg) from err


async def _invoke_chat_model_with_sampling_rebind(  # noqa: PLR0913
    hass: Any,
    base_model: Any,
    model: Any,
    messages: list[AnyMessage],
    config: RunnableConfig,
    selected_tools: list[Any],
    *,
    disable_reasoning: bool,
) -> Any:
    """
    Invoke the (possibly tool-bound) chat model, rebinding on sampling 400s.

    bind_tools() collapses a configurable-fields model into a concrete binding
    with sampling params baked in, so the invoke-time config drop in
    ainvoke_dropping_unsupported_params cannot help a tool-bound chat model.
    When the provider rejects temperature/top_p (issue #502), rebuild from the
    pre-bind model with both sampling params nulled, re-bind the tools, and
    retry once. Configurable-field models omit None params from the payload.
    """
    try:
        return await _invoke_model(
            model, messages, config, drop_unsupported=not selected_tools
        )
    except HomeAssistantError as err:
        param = unsupported_sampling_param_in_chain(err)
        if param is None:
            raise
        LOGGER.warning(
            "Chat model rejected unsupported sampling parameter %r; retrying "
            "once with provider-default sampling parameters.",
            param,
        )
        stripped = base_model.with_config(
            config={"configurable": dict.fromkeys(DROPPABLE_SAMPLING_PARAMS)}
        )
        if selected_tools:
            stripped = await hass.async_add_executor_job(
                partial(
                    _bind_model_tools,
                    stripped,
                    selected_tools,
                    disable_reasoning=disable_reasoning,
                )
            )
        return await _invoke_model(stripped, messages, config, drop_unsupported=False)


# A provider that refuses a tool's JSON Schema names the offender by its
# position in the request's tool array, never by name — Anthropic reports
# "tools.3.custom.input_schema: input_schema does not support oneOf, allOf,
# or anyOf at the top level" (issue #585), Gemini reports
# "...function_declarations[1].parameters...". Both indices address the tool
# list in the order it was bound, so they map back onto selected_tools.
_TOOL_SCHEMA_REJECTION_RE = re.compile(
    r"\b(?:tools\.(\d+)\.|function_declarations\[(\d+)\])"
)

# Tools dropped per turn before giving up. Only two HA assist schemas have
# ever carried a rejected shape, so a small budget covers a novel offender
# without letting a mis-parsed index strip the whole tool set.
_MAX_TOOL_SCHEMA_DROPS = 3

# Longest run of digits accepted as a tool position, so a mis-parsed message
# cannot turn into an unbounded int().
_MAX_TOOL_INDEX_DIGITS = 4


def _rejected_tool_index(err: BaseException, max_depth: int = 10) -> int | None:
    """Return the tool position a provider rejected for its schema, if any."""
    current: BaseException | None = err
    for _ in range(max_depth):
        if current is None:
            return None
        message = str(current)
        match = _TOOL_SCHEMA_REJECTION_RE.search(message)
        # "tools.N." is loose enough to need the word "schema" beside it;
        # Gemini's "function_declarations[N]" only ever addresses a tool
        # declaration, and its real 400 never says "schema".
        if match and (match.group(2) or "schema" in message):
            index = match.group(1) or match.group(2)
            if len(index) <= _MAX_TOOL_INDEX_DIGITS:
                return int(index)
        current = current.__cause__
    return None


def _rejected_tool_name(selected_tools: list[Any], index: int | None) -> str | None:
    """Return the name of the tool at ``index``, or None if unresolvable."""
    if index is None or not 0 <= index < len(selected_tools):
        return None
    entry = selected_tools[index]
    if not isinstance(entry, dict):
        return None
    function = entry.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) else None


async def _invoke_chat_model_with_schema_recovery(  # noqa: PLR0913
    hass: Any,
    base_model: Any,
    model: Any,
    messages: list[AnyMessage],
    config: RunnableConfig,
    selected_tools: list[Any],
    *,
    disable_reasoning: bool,
) -> Any:
    """
    Invoke the chat model, dropping tools whose schema the provider rejects.

    A provider validates every tool schema before it reads the conversation,
    so one unsupported tool fails the whole turn — the user hears an error
    even though nothing about their request was wrong (issue #585: Anthropic
    rejects the top-level ``anyOf`` HA's timer intents emit). The schema
    normalisation in ``_format_and_dedupe_tools()`` is the real fix for the
    shapes we know about; this is the net under it for the ones we do not.

    The rejection names the offender only by position, so resolve that to a
    tool name, log it, drop it and retry: losing one tool beats losing the
    turn. When the position cannot be resolved the original error is raised
    untouched, and when retries run out the tool name is added to it so the
    log names what to exclude.
    """
    tools = list(selected_tools)
    dropped = 0
    while True:
        try:
            return await _invoke_chat_model_with_sampling_rebind(
                hass,
                base_model,
                model,
                messages,
                config,
                tools,
                disable_reasoning=disable_reasoning,
            )
        except HomeAssistantError as err:
            index = _rejected_tool_index(err)
            name = _rejected_tool_name(tools, index)
            if index is None or name is None:
                raise
            if dropped == _MAX_TOOL_SCHEMA_DROPS:
                msg = (
                    f"Model invocation failed: the provider rejected the schema "
                    f"of tool '{name}' and dropping it did not clear the error. "
                    f"{err}"
                )
                raise HomeAssistantError(msg) from err
            LOGGER.warning(
                "Provider rejected the schema of tool '%s' (position %d); "
                "dropping it and retrying the turn without it. Exclude the tool "
                "to avoid the retry. %s",
                name,
                index,
                err,
            )
            del tools[index]
            dropped += 1
            model = (
                await hass.async_add_executor_job(
                    partial(
                        _bind_model_tools,
                        base_model,
                        tools,
                        disable_reasoning=disable_reasoning,
                    )
                )
                if tools
                else base_model
            )


def _bind_model_tools(
    model: Any, selected_tools: list[Any], *, disable_reasoning: bool
) -> Any:
    """Bind tools to a model in a worker thread."""
    if disable_reasoning:
        model = model.with_config(config={"configurable": {"reasoning": False}})
    if isinstance(model, FallbackChatModel):
        return model.bind_tools(
            selected_tools, tools_for_member=_tools_for_chain_member
        )
    # Not only chains: when the primary is down at setup the chat model IS the
    # fallback provider, unwrapped, while tools are still shaped for the
    # configured primary.
    return model.bind_tools(_tools_for_chain_member(model, selected_tools))


async def _recent_memories(store: BaseStore, user_id: str) -> list:
    """Return the most recent memories, without semantic search."""
    try:
        return await store.asearch((user_id, "memories"), limit=10)
    except Exception:
        LOGGER.exception("Memory recency search also failed; no memories injected")
        return []


async def _search_memories(store: BaseStore, user_id: str, query: str | None) -> list:
    """Search the memory store, falling back gracefully on embedding errors."""
    try:
        return await store.asearch((user_id, "memories"), query=query, limit=10)
    except AttributeError:
        # Some OpenAI-compatible servers (e.g. llama-server) return a raw JSON
        # list from /v1/embeddings instead of {"data": [...]}, which causes the
        # OpenAI SDK parser to raise AttributeError.  Fall back to recency search.
        LOGGER.warning(
            "Memory semantic search failed — embedding endpoint returned an "
            "incompatible response (not OpenAI-standard). Falling back to "
            "recency-based memory retrieval. Check the /v1/embeddings response "
            "format of your OpenAI-compatible server."
        )
    except EmbeddingsUnavailableError:
        # No embedding provider can answer: none was up at setup (#751), or
        # the fallback is held until its build is checked (#857). Each is
        # logged where it happens (setup, the chain's provider failures, the
        # build check), so recent memories without a warning per turn.
        pass
    except psycopg.DataError as err:
        LOGGER.warning(
            "Memory semantic search failed due to vector store data mismatch: %s. "
            "Falling back to recency-based memory retrieval.",
            err,
        )
    except Exception:
        LOGGER.exception("Unexpected memory store search failure; no memories injected")
        return []
    return await _recent_memories(store, user_id)


async def _call_model(
    state: State, config: RunnableConfig, *, store: BaseStore
) -> dict[str, Any]:
    """Coroutine to call the chat model."""
    if "configurable" not in config:
        msg = "Configuration for the model is missing."
        raise HomeAssistantError(msg)

    conf = config["configurable"]
    model = conf["chat_model"]
    user_id = conf["user_id"]
    hass = conf["hass"]
    opts = conf["options"]
    chat_model_options = conf.get("chat_model_options", {})

    # Retrieve memories once per turn, for the user message being served.
    # Inside a tool loop the last message is a ToolMessage; the block must
    # nonetheless stay byte-identical across the turn's model calls, or the
    # system prompt — and with it the provider's cached prefix — changes
    # between the call that requested a tool and the call that answers.
    # So the first call searches and memoizes; later calls reuse. A memory
    # written by a tool mid-turn is therefore seen next turn, by design.
    last_user_message = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    user_text = _message_text(last_user_message) if last_user_message else ""
    memo_key = (
        (last_user_message.id or user_text) if last_user_message is not None else ""
    )
    memo = state.get("turn_memories") or {}
    if memo.get("key") == memo_key and "text" in memo:
        memories = memo["text"]
    else:
        # An attachment-only turn has no text: embedding the bare template
        # would inject the memories nearest to nothing, so list by recency.
        query_prompt = (
            EMBEDDING_MODEL_PROMPT_TEMPLATE.format(query=user_text)
            if user_text.strip()
            else None
        )
        memories = _format_memories(
            await _search_memories(store, user_id, query_prompt)
        )
        memo = {"key": memo_key, "text": memories}

    # Build system message: stable prefix, then this turn's volatile context.
    system_message = _build_system_message(
        conf["prompt"],
        conf.get("prompt_volatile", ""),
        memories,
        state.get("summary", ""),
    )

    # Model input = System + current messages.
    messages = [system_message, *state["messages"]]

    trimmed_messages = await _trim_messages_for_model(
        messages, opts, chat_model_options, hass
    )

    LOGGER.debug("Model call messages: %s", trimmed_messages)

    # Bind retrieved tools
    base_model = model
    selected_tools = state.get("selected_tools", [])
    disable_reasoning = bool(chat_model_options.get("reasoning"))
    if selected_tools:
        # Disable reasoning/thinking before binding tools: Qwen3's <think> tokens
        # interleave with tool call JSON output and break Ollama's qwen3.go parser
        # (ResponseError: "invalid character 'g' looking for beginning of value").
        model = await hass.async_add_executor_job(
            partial(
                _bind_model_tools,
                model,
                selected_tools,
                disable_reasoning=disable_reasoning,
            )
        )

    # Pass routing map to MultiLLMAPI
    routing_map = state.get("tool_routing_map", {})
    llm_api = conf.get("ha_llm_api")
    if llm_api and hasattr(llm_api, "routing_map"):
        llm_api.routing_map = routing_map

    raw_response = await _invoke_chat_model_with_schema_recovery(
        hass,
        base_model,
        model,
        trimmed_messages,
        config,
        selected_tools,
        disable_reasoning=disable_reasoning,
    )
    LOGGER.debug("Raw chat model response: %s", raw_response)

    # collapse_whitespace=False: the reply is rendered as markdown and is
    # compared byte-for-byte with the streamed copy to detect a mid-stream
    # model fallback, so newlines must survive (issue #628).
    response = extract_final(
        getattr(raw_response, "content", "") or "", collapse_whitespace=False
    )

    tool_calls = getattr(raw_response, "tool_calls", []) or []

    # Qwen3 extended-thinking: Ollama strips <think>…</think> tokens from content,
    # leaving content='' when all output was reasoning.  On a post-tool turn with no
    # follow-up tool call the user would receive no reply at all.  Inject a minimal
    # acknowledgement so the conversation doesn't go silent.
    if (
        not response
        and not tool_calls
        and isinstance(state["messages"][-1], ToolMessage)
    ):
        LOGGER.debug("Empty model response after tool execution; injecting fallback.")
        response = "Done."

    # Create AI message, no need to include tool call metadata if there's none.
    if tool_calls:
        ai_response = AIMessage(content=response, tool_calls=tool_calls)
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
        "turn_memories": memo,
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
    messages = cast(
        "list[AnyMessage]",
        [SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT)]
        + [m for m in msgs_to_remove if isinstance(m, (HumanMessage, AIMessage))]
        + [HumanMessage(content=summary_message)],
    )

    model = config["configurable"]["summarization_model"]
    LOGGER.debug("Summary messages: %s", messages)
    raw_response = await _invoke_model(model, messages, {})
    LOGGER.debug("Raw summary response: %s", raw_response)

    # Keep the summary's own line breaks; it is fed back into the prompt.
    response = extract_final(
        getattr(raw_response, "content", "") or "", collapse_whitespace=False
    )

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


def _is_critical_action(  # noqa: PLR0911, PLR0912
    tool_args: dict[str, Any],
    critical_actions: list[dict[str, str]],
    tool_name: str,
    hass: HomeAssistant | None = None,
    *,
    origin: ToolOrigin | None = None,
) -> bool:
    """
    Return True if the call could do a critical action.

    A call matching a domain/service/entity rule is critical. Beyond the
    rules: an actuating intent is checked per entity HA will act on, and
    HassTurnOn on a script or scene opens it; a script's own tool is screened
    by the steps the script runs; and a service tool's call is screened like
    an automation step, with any script or scene it runs opened.
    """
    # A tool runs with its arguments as given (#836), so a blank entry in a
    # list of domains is skipped here, never read as "no domain".
    if isinstance(tool_args.get("domain"), list):
        tool_args = {
            **tool_args,
            "domain": [
                d for d in tool_args["domain"] if d is not None and str(d).strip()
            ],
        }
    domain_val = tool_args.get("domain", "")
    if isinstance(domain_val, list):
        domain_val = domain_val[0] if domain_val else ""
    # A null or blank domain names none: the entities' own domain is read.
    domain_given = domain_val is not None and bool(str(domain_val).strip())
    entities = [e.lower() for e in _coerce_entity_ids(tool_args)]

    # A remote Home Assistant's tool (#840): its devices, scripts and
    # automations cannot be seen from here, so any call that can change
    # something is held.
    if origin is not None and origin.ambiguous:
        return True
    if origin is not None and origin.remote:
        # Read by the tool's own name: the name the agent sees may be cut
        # short (#843 review).
        return not _remote_call_is_read_only(
            origin.tool_name or tool_name, tool_args, read_only_hint=origin.read_only
        )

    # An intent script's tool names no domain or service, and one may take
    # over a built-in intent's tool; what it runs is checked first, and only
    # ever adds a hold (#779).
    if intent_scripts_are_critical(hass, tool_name, critical_actions):
        return True

    # An intent carries no service: HA picks it per matched entity (on/off
    # opens or closes a cover, #770), so check each call HA will make.
    if is_actuating_intent(tool_name):
        return _intent_is_critical(tool_args, critical_actions, tool_name, hass)
    if is_device_intent(tool_name) and hass is not None:
        # Setting a template light's level runs its own action scripts (#780).
        states = intent_target_states(hass, tool_name, tool_args)
        check = StoredActionCheck(hass, critical_actions)
        if states is None or any(
            stored_actions_are_critical(
                hass, state.entity_id, None, critical_actions, check
            )
            for state in states
        ):
            return True

    # A script's own tool names no domain or service; what it runs decides.
    if (script_id := script_tool_entity_id(tool_name)) is not None:
        return _script_tool_is_critical(script_id, critical_actions, hass)
    # A tool named like a script (HA before 2026.9 named script tools so) may
    # be that script's tool or another tool sharing its name: screen both.
    if (
        bare_script := bare_script_tool_entity_id(tool_name, hass)
    ) is not None and _script_tool_is_critical(bare_script, critical_actions, hass):
        return True

    # Alarm panels already enforce their own code; never gate them with the
    # generic PIN. Only when the call names that domain alone: with none, the
    # first target being a panel says nothing of the rest.
    raw_domain = tool_args.get("domain")
    declared = raw_domain if isinstance(raw_domain, list) else [raw_domain]
    alarm_only = domain_given and {str(d).lower() for d in declared} == {
        "alarm_control_panel"
    }
    if (alarm_only or base_tool_name(tool_name) == _ALARM_TOOL) and (
        _template_alarm_is_critical(tool_args, entities, critical_actions, hass)
    ):
        # A template panel runs its own arm and disarm actions, with no code of
        # its own to enforce (#780).
        return True
    # A tool may name its service another way than `domain` and `service`
    # (#837); each name it gives is screened, whichever one the tool runs.
    names = call_service_names(tool_args)
    if names.unreadable and calls_a_service(tool_args, entities):
        return True  # which service it runs cannot be read
    if any(
        _named_service_is_critical(action, tool_args, entities, critical_actions, hass)
        for action in named_services(
            tool_args, tool_name, hass, rule_domains(critical_actions)
        )
    ):
        return True
    if alarm_only:
        return False
    given = declared if domain_given else []
    return any(
        _bare_service_is_critical(service, tool_args, given, critical_actions, hass)
        for service in names.bare_services
    ) or any(
        _bare_service_is_critical(
            service, tool_args, given, critical_actions, hass, known_only=True
        )
        for service in names.bare_actions
    )


def _remote_call_is_read_only(
    tool_name: str, tool_args: dict[str, Any], *, read_only_hint: bool
) -> bool:
    """
    Return True if a remote Home Assistant's tool call only reads (#840).

    A script or intent script may do anything, whatever it is called. A timer
    given a command runs it later through the remote side's conversation
    agent. Otherwise one of Home Assistant's read-only intents, or a tool the
    remote server marks read-only, only reads.
    """
    if is_script_like_tool(tool_name):
        return False
    base = base_tool_name(tool_name)
    if (
        base == _START_TIMER_INTENT
        and str(tool_args.get(_TIMER_COMMAND_SLOT) or "").strip()
    ):
        return False
    return read_only_hint or (
        is_intent_tool(tool_name) and base in _REMOTE_READ_ONLY_INTENTS
    )


def _bare_service_is_critical(  # noqa: PLR0913
    service: str,
    tool_args: dict[str, Any],
    declared: list[Any],
    critical_actions: list[dict[str, str]],
    hass: HomeAssistant | None,
    *,
    known_only: bool = False,
) -> bool:
    """
    Return True if a service named without its domain could be critical.

    It runs in each domain the call declares; with none, in the domain of the
    first entity it targets. With `known_only` (a bare `action`, which HA
    never writes without its domain), a domain read from the targets counts
    only for a service HA has there: a tool's own `action: get` on a script is
    not `script.get`.
    """
    entities = [e.lower() for e in _coerce_entity_ids(tool_args)]
    domain_given = bool(declared)
    domain = ""
    if not domain_given and entities:
        domain = entities[0].split(".", maxsplit=1)[0]
    # A list of domains is screened in each one it names. With none, each
    # domain the call targets, wherever its targets sit (#837 review).
    domains: list[str] = (
        [d for d in (str(d).strip().lower() for d in declared) if d]
        if domain_given
        else list(
            dict.fromkeys(
                [
                    *([domain] if domain else []),
                    *sorted(call_target_domains(tool_args)[0]),
                ]
            )
        )
    )
    # A rule names its service outright, wherever the call may run it, so it
    # is matched in every domain. Otherwise a bare `action` is read in a
    # domain the call targets only if HA has the service there; a domain the
    # call declares is read as given (#837 review).
    if any(
        matches_critical_rule(
            domain=d,
            service=service,
            entity_ids=entities,
            critical_actions=critical_actions,
        )
        for d in domains
    ):
        return True
    if known_only and not domain_given:
        domains = [d for d in domains if has_service(hass, d, service)]
    actions = [f"{d}.{service}" for d in domains]
    # A service tool's call is screened like an automation step, opening any
    # script or scene it runs. With no domain given, the one read from the
    # first entity is a guess, and the targets may sit under `target` or
    # `data`: a tool may send such a call as `homeassistant.<service>`, which
    # runs each target's own service, so it is screened that way too when it
    # could run a script or scene, reaches several domains or targets one
    # the gate cannot name.
    if not domain_given and (
        entities or any(key in tool_args for key in STEP_TARGET_KEYS)
    ):
        target_domains, unresolved = call_target_domains(tool_args)
        if (
            service in RUN_SERVICES
            or len(target_domains) > 1
            or not domain
            or unresolved
        ) and (not known_only or has_service(hass, "homeassistant", service)):
            actions.append(f"homeassistant.{service}")
    return any(
        service_runs_critical_action(hass, action, tool_args, critical_actions)
        for action in actions
    )


def _named_service_is_critical(
    action: str,
    tool_args: dict[str, Any],
    entities: list[str],
    critical_actions: list[dict[str, str]],
    hass: HomeAssistant | None,
) -> bool:
    """Return True if a rule names the service, or what it runs is critical."""
    domain, _, service = action.partition(".")
    return matches_critical_rule(
        domain=domain,
        service=service,
        entity_ids=entities,
        critical_actions=critical_actions,
    ) or service_runs_critical_action(hass, action, tool_args, critical_actions)


_ALARM_TOOL = "alarm_control"
_ALARM_DOMAIN = "alarm_control_panel"


def _template_alarm_is_critical(
    tool_args: dict[str, Any],
    entities: list[str],
    critical_actions: list[dict[str, str]],
    hass: HomeAssistant | None,
) -> bool:
    """
    Return True if a template alarm panel the call reaches runs a critical action.

    The panel named by entity id, else by name; with neither, any template
    panel, since the tool picks one.
    """
    if hass is None or getattr(hass, "states", None) is None:
        return False
    panels = [e for e in entities if e.startswith(f"{_ALARM_DOMAIN}.")]
    name = str(tool_args.get("name") or "").strip().lower()
    if not panels and name:
        panels = [
            state.entity_id
            for state in hass.states.async_all(_ALARM_DOMAIN)
            if str(state.attributes.get("friendly_name", "")).lower() == name
        ]
    if not panels:
        panels = entities_of(hass, _ALARM_DOMAIN)
    check = StoredActionCheck(hass, critical_actions)
    return any(
        stored_actions_are_critical(hass, panel, None, critical_actions, check)
        for panel in panels
    )


def _script_tool_is_critical(
    entity_id: str,
    critical_actions: list[dict[str, str]],
    hass: HomeAssistant | None,
) -> bool:
    """Return True if a rule names the script or a step it runs is critical."""
    return script_rule_matches(
        hass, entity_id, critical_actions
    ) or runs_critical_action(hass, entity_id, critical_actions)


def _intent_is_critical(  # noqa: PLR0911
    tool_args: dict[str, Any],
    critical_actions: list[dict[str, str]],
    tool_name: str,
    hass: HomeAssistant | None,
) -> bool:
    """
    Return True if any service call an actuating intent makes is critical.

    The calls HA's matcher resolves on the payload HA is sent (an entity id
    is in it as the name when HA will read it), plus a declared lock domain,
    plus the entity ids the model named when nothing matched. Without
    `hass`, the domain the model declared, with an unnamed target matching
    every `entity_match` rule. An on/off
    intent on any lock is critical whatever the rules say (the domain of an
    entity id the model gave counts: with no domain and no service the call
    would otherwise run with no PIN at all); alarm panels keep their own
    code. HassTurnOn on a script or a scene runs it, so it is opened and
    held when what it does is critical (#775). A template entity runs its own
    action script for the call, and pressing an input button starts the
    automations that trigger on it; those are opened the same way (#780).
    When the matcher cannot tell what HA will act on, the call is held.
    """
    declared = declared_intent_service_calls(tool_name, tool_args)
    if declared is None:
        return True  # a slot the gate cannot read: fail closed
    if hass is None:
        calls = declared
    else:
        resolved = intent_service_calls(hass, tool_name, tool_args)
        if resolved is None:
            return True  # the matcher could not tell: fail closed
        # What HA matched decides: an entity id HA ignores, or a domain with
        # no entity, would otherwise gate every blind as a possible door. A
        # declared lock is held whatever matched, and an entity id the model
        # named is checked when nothing matched (HA refuses such a call, but
        # the gate does not count on it).
        calls = resolved + [
            call
            for call in declared
            if call.domain == "lock" or (call.entity_id and not resolved)
        ]
    turns_on = base_tool_name(tool_name) == "HassTurnOn"
    check = StoredActionCheck(hass, critical_actions) if hass is not None else None
    for call in calls:
        if call.domain == "alarm_control_panel":
            # A real panel enforces its own code; a template panel runs its
            # own actions (#780).
            if call.entity_id and stored_actions_are_critical(
                hass, call.entity_id, None, critical_actions, check
            ):
                return True
            continue
        if call.domain == "lock" and is_on_off_intent(tool_name):
            return True
        if matches_critical_rule(
            domain=call.domain,
            service=call.service,
            entity_ids=[call.entity_id] if call.entity_id else [],
            critical_actions=critical_actions,
            unresolved_target=call.entity_id is None,
        ):
            return True
        # Turning on a script runs it; turning on a scene sets its entities.
        if (
            turns_on
            and call.domain in (SCRIPT_DOMAIN, SCENE_DOMAIN)
            and (
                call.entity_id is None
                or (
                    call.domain == SCRIPT_DOMAIN
                    and script_rule_matches(hass, call.entity_id, critical_actions)
                )
                or runs_critical_action(hass, call.entity_id, critical_actions, check)
            )
        ):
            return True
        if (
            call.entity_id
            and call.domain not in (SCRIPT_DOMAIN, SCENE_DOMAIN)
            and stored_actions_are_critical(
                hass, call.entity_id, call.service, critical_actions, check
            )
        ):
            return True
    return False


_LIVE_CONTEXT_KEYS = ("name", "domain", "area")


def _live_context_args(tool_args: dict[str, Any]) -> dict[str, Any]:
    """Keep only the filters GetLiveContext accepts; models add entity_id too."""
    return {key: tool_args[key] for key in _LIVE_CONTEXT_KEYS if tool_args.get(key)}


def _has_local_ha_api(ha_llm_api: Any) -> bool:
    """
    Return True if an API of this house (not a remote Home Assistant) is on.

    With only remote Home Assistants, the user's house is the remote one, and
    its GetLiveContext gets the open-state rewrite as before (#843 review).
    """
    apis = getattr(ha_llm_api, "apis", None)
    if not isinstance(apis, Mapping):
        return True
    remote = remote_api_names(ha_llm_api)
    return any(api_id not in remote for api_id in apis)


def _open_state_args(
    ha_llm_api: Any,
    tool_name: str,
    tool_args: dict[str, Any],
    open_state_query: Any,
    *,
    force_broad: bool,
) -> tuple[bool, dict[str, Any]]:
    """
    Return whether a call goes to a remote Home Assistant, and its arguments.

    The open-state rewrite and reply filter are for this house's
    GetLiveContext, not a remote Home Assistant's, whose call keeps its
    arguments (#843 review).
    """
    if tool_origin(ha_llm_api, tool_name).remote and _has_local_ha_api(ha_llm_api):
        return True, tool_args
    return False, _normalize_live_context_args_for_open_state(
        tool_name, tool_args, open_state_query, force_broad=force_broad
    )


async def _call_tools(
    state: State, config: RunnableConfig, *, store: BaseStore
) -> dict[str, Any]:
    """Call Home Assistant or LangChain tools requested by the model."""
    if "configurable" not in config:
        msg = "Configuration is missing."
        raise HomeAssistantError(msg)

    langchain_tools = config["configurable"]["langchain_tools"]
    ha_llm_api = config["configurable"]["ha_llm_api"]
    options = config["configurable"]["options"]
    hass = config["configurable"]["hass"]
    policy = resolve_critical_action_policy(options)
    critical_actions: list[dict[str, str]] = policy.critical_actions
    pin_hash = policy.pin_hash
    pin_salt = policy.pin_salt
    pin_enabled = policy.enabled
    pending_actions = config["configurable"].get("pending_actions", {})

    # Expect tool calls in the last AIMessage.
    if not state["messages"] or not isinstance(state["messages"][-1], AIMessage):
        msg = "No tool calls found in the last message."
        raise HomeAssistantError(msg)

    raw_tool_calls = getattr(state["messages"][-1], "tool_calls", None)
    tool_calls: list[dict[str, Any]] = (
        raw_tool_calls if isinstance(raw_tool_calls, list) else []
    )

    tool_routing_map: dict[str, str] = state.get("tool_routing_map", {})
    open_state_query, first_open_state_live_context_call = (
        _open_state_live_context_normalization_context(
            state["messages"], tool_calls, ha_llm_api
        )
    )

    # Execute all tool calls concurrently.
    # asyncio.gather preserves submission order: results[i] matches tool_calls[i].
    # Concurrent HA tool calls may interleave (e.g. turn_on + get_state in same
    # batch). If strict sequencing is needed, issue separate model turns.
    async def _invoke_one(tool_call: dict[str, Any]) -> ToolMessage:
        """Execute a single tool call and return its ToolMessage."""
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {}) or {}
        remote_call, normalized_args = _open_state_args(
            ha_llm_api,
            tool_name,
            tool_args,
            open_state_query,
            force_broad=tool_call is first_open_state_live_context_call,
        )
        if normalized_args != tool_args:
            LOGGER.debug(
                "Normalized GetLiveContext args for read-only open-state query: "
                "%s -> %s",
                tool_args,
                normalized_args,
            )
            tool_args = normalized_args
        LOGGER.debug("Tool call: %s(%s)", tool_name, tool_args)

        # Enforce the retrieved tool set: reject calls for tools that were not
        # selected by _retrieve_tools this turn.
        if tool_routing_map and tool_name not in tool_routing_map:
            LOGGER.warning(
                "Model called tool '%s' which was not retrieved for this turn "
                "(routing_map=%s). Rejecting.",
                tool_name,
                list(tool_routing_map),
            )
            return _make_tool_error(
                f"Tool '{tool_name}' is not available for this request. "
                "Only use the tools listed in your schema.",
                tool_name,
                tool_call.get("id") or "",
            )

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
            return precheck

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
        tool_response = _maybe_filter_open_state_tool_response(
            tool_response,
            tool_call,
            # A remote Home Assistant's reply is not filtered (#843 review).
            None if remote_call else first_open_state_live_context_call,
            tool_name,
            open_state_query,
        )
        LOGGER.debug("Tool response: %s", tool_response)
        return tool_response

    results = await asyncio.gather(
        *[_invoke_one(tc) for tc in tool_calls],
        return_exceptions=True,
    )
    tool_responses: list[ToolMessage] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Log the full exception; surface a sanitized message to avoid
            # leaking tokens, URLs, or internal details to the model.
            tc = tool_calls[i]
            LOGGER.error("Tool %s raised during gather: %r", tc.get("name", ""), result)
            tool_responses.append(
                _make_tool_error(
                    f"Tool execution failed: {type(result).__name__}",
                    tc.get("name", ""),
                    tc.get("id") or "",
                )
            )
        else:
            tool_responses.append(result)  # type: ignore[arg-type]

    return {
        "messages": tool_responses,
        **_next_round_counters(state, tool_responses),
    }


def _should_continue(
    state: State,
) -> Literal["action", "tool_loop_guard", "summarize_and_remove_messages"]:
    """Return the next node in graph to execute."""
    messages = state["messages"]
    has_tool_calls = isinstance(messages[-1], AIMessage) and messages[-1].tool_calls
    if has_tool_calls and state.get("action_rounds", 0) < _MAX_ACTION_ROUNDS:
        return "action"
    if has_tool_calls:
        return "tool_loop_guard"
    return "summarize_and_remove_messages"


def _last_round_responses(messages: Sequence[AnyMessage]) -> list[ToolMessage]:
    """Return the last tool round's messages: those after the last AIMessage."""
    responses: list[ToolMessage] = []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            break
        if isinstance(msg, ToolMessage):
            responses.append(msg)
    responses.reverse()
    return responses


def _pin_refusal_of(msg: BaseMessage) -> dict[str, Any] | None:
    """Return the refusal payload of a `confirm_sensitive_action` tool message."""
    if not isinstance(msg, ToolMessage) or not isinstance(msg.content, str):
        return None
    return pin_refusal(msg.content, msg.name or "")


def _refusal_ends_turn(msg: BaseMessage) -> bool:
    """Return True for a confirmation refused because the user was not asked yet."""
    refusal = _pin_refusal_of(msg)
    return refusal is not None and refusal.get(PIN_REFUSAL_ENDS_TURN) is True


def _after_tools(state: State) -> Literal["agent", "summarize_and_remove_messages"]:
    """
    Return the next node after a tool round.

    The model gets the results back, except when every call in the round was
    a confirmation refused because the user has not been asked for the PIN
    yet (the message is not the prompt's reply). The reply asks for the PIN in
    fixed words (conversation.py), and the model's next move would be to ask
    again, or to try another PIN of its own: in a field test (2026-10-16) a
    local model guessed a PIN three times, spoke its prompt three times, and
    the turn ended in the round-limit apology. The turn ends here instead, and
    the user's next message is the prompt's reply.
    """
    responses = _last_round_responses(state["messages"])
    if responses and all(_refusal_ends_turn(msg) for msg in responses):
        return "summarize_and_remove_messages"
    return "agent"


def _pin_line_is_the_reply(config: RunnableConfig) -> bool:
    """
    Return True when a fixed PIN line will be the turn's reply.

    The reply then says, in fixed words (conversation.py), what the PIN will
    confirm, what it ran, or that it was wrong, so the guard has nothing to
    add. Whatever the last round held: the field case is a model that keeps
    confirming with PINs of its own after the prompt was created, and the
    prompt itself may be the last result before the cut-off.
    """
    cfg = config.get("configurable", {})
    pin_reply = cfg.get("pin_reply")
    return pin_reply is not None and pin_lines_follow(
        cfg.get("pending_actions", {}), pin_reply
    )


# The result given to a call the round-limit guard cuts off. Not the error
# template: that one asks the model to call the tool again, which is the one
# thing this turn must not do, and the next model call sees this text right
# before the user's next message.
_ROUND_LIMIT_RESULT = (
    "Not run: this turn's tool-round limit was reached. Do not call it again "
    "unless the user asks."
)


def _unanswered_calls(messages: Sequence[AnyMessage]) -> list[ToolMessage]:
    """
    Answer the tool calls the guard cuts off, so none is left dangling.

    The model's last message asked for tools that never ran. A provider
    rejects a history whose assistant tool call has no result, so the next
    turn (the user's PIN, for one) would fail on it. Each gets a fixed
    "not run" result.
    """
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return []
    return [
        ToolMessage(
            content=_ROUND_LIMIT_RESULT,
            name=tool_call.get("name", ""),
            tool_call_id=tool_call.get("id") or "",
            status="error",
        )
        for tool_call in last.tool_calls
    ]


async def _tool_loop_guard(state: State, config: RunnableConfig) -> dict[str, Any]:
    """
    Emit a friendly message when the action-round limit is reached.

    The calls being cut off get a "not run" result first, so the history stays
    valid for the next model call. When the round before the cut-off was a
    refused automation, the refusal text is the one thing the user can act
    on, so it is repeated rather than lost behind the generic apology. When
    a PIN line will be the reply (the prompt is waiting, an action ran, or
    the PIN was wrong), nothing more is added: an apology next to "What is
    your PIN?" reads as a failure, and the user is never asked.
    """
    messages = state["messages"]
    answers: list[BaseMessage] = list(_unanswered_calls(messages))
    if _pin_line_is_the_reply(config):
        return {"messages": answers}
    content = (
        "I wasn't able to complete this request after several tool-use "
        "attempts. Please try rephrasing your query or breaking it into "
        "smaller steps."
    )
    last_tool = next(
        (m for m in reversed(messages) if isinstance(m, ToolMessage)), None
    )
    if last_tool is not None and _is_automation_refusal(last_tool):
        content += f"\n\nThe last problem was:\n{last_tool.content}"
    return {"messages": [*answers, AIMessage(content=content)]}


# Define a new graph
workflow = StateGraph(State)

# Define nodes.
workflow.add_node("retrieve_tools", _retrieve_tools)
workflow.add_node("agent", _call_model)
workflow.add_node("action", _call_tools)
workflow.add_node("tool_loop_guard", _tool_loop_guard)
workflow.add_node("summarize_and_remove_messages", _summarize_and_remove_messages)

# Define edges.
workflow.add_edge(START, "retrieve_tools")
workflow.add_edge("retrieve_tools", "agent")
workflow.add_conditional_edges("agent", _should_continue)
workflow.add_conditional_edges("action", _after_tools)
workflow.add_edge("tool_loop_guard", "summarize_and_remove_messages")
workflow.add_edge("summarize_and_remove_messages", END)
