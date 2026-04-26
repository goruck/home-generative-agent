"""Langgraph graphs for Home Generative Agent."""

from __future__ import annotations

import asyncio
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
    TypedDict,
    cast,
)

import psycopg
import voluptuous as vol
from homeassistant.const import (
    CONF_LLM_HASS_API,
)
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
from langgraph.store.base import InvalidNamespaceError
from pydantic import PydanticInvalidForJsonSchema, ValidationError

from custom_components.home_generative_agent.const import (
    ACTUATION_KEYWORDS_REGEX,
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
    CONF_OPENAI_COMPATIBLE_CHAT_MODEL,
    CONF_TOOL_RELEVANCE_THRESHOLD,
    CONF_TOOL_RETRIEVAL_LIMIT,
    EMBEDDING_MODEL_PROMPT_TEMPLATE,
    RECOMMENDED_CRITICAL_ACTIONS,
    SUMMARIZATION_INITIAL_PROMPT,
    SUMMARIZATION_PROMPT_TEMPLATE,
    SUMMARIZATION_SYSTEM_PROMPT,
    TOOL_CALL_ERROR_TEMPLATE,
    TOOL_CALL_TRANSIENT_ERROR_TEMPLATE,
)

from ..core.utils import chat_priority_context, extract_final  # noqa: TID252
from .camera_activity import get_recent_camera_activity
from .helpers import (
    format_tool,
    is_actuation_tool,
    maybe_fill_lock_entity,
    normalize_intent_for_alarm,
    normalize_intent_for_lock,
    sanitize_tool_args,
)
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


class State(MessagesState):
    """Extend MessagesState."""

    summary: str
    chat_model_usage_metadata: dict[str, Any]
    messages_to_remove: list[AnyMessage]
    selected_tools: list[dict[str, Any]]
    tool_routing_map: dict[str, str]


def _determine_model_name(provider: str, opts: dict[str, Any]) -> str:
    """Determine model name based on provider and options."""
    if provider == "openai":
        return opts.get(CONF_OPENAI_CHAT_MODEL, "")
    if provider == "openai_compatible":
        return opts.get(CONF_OPENAI_COMPATIBLE_CHAT_MODEL, "")
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
        LOGGER.warning(
            "Critical action PIN is enabled but no PIN is configured. "
            "Allowing '%s' without PIN verification. "
            "Set a PIN in the integration options to enforce confirmation.",
            tool_name,
        )
        return None

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


class RawTool(TypedDict):
    """Represent a tool before formatting for the LLM."""

    name: str
    api_id: str
    description: str
    parameters: str
    is_actuation: bool


def _get_allowed_api_ids(config: RunnableConfig) -> set[str]:
    """Determine which API IDs are allowed based on configuration."""
    opts = config.get("configurable", {}).get("options", {})
    active_api_ids = opts.get(CONF_LLM_HASS_API, [llm.LLM_API_ASSIST])
    if isinstance(active_api_ids, str):
        active_api_ids = [active_api_ids]
    return set(active_api_ids) | {"hga_local"}


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


async def _get_rag_retrieved_tools(
    store: BaseStore | None,
    config: RunnableConfig,
    query: str,
    allowed_api_ids: set[str],
) -> list[RawTool]:
    """
    Search for tools in the vector store and filter by score/API.

    For multi-intent queries, splits the query into per-intent sub-queries and
    runs a separate vector search for each.  Each tool's score is the maximum
    across all sub-query passes so that a tool highly relevant to *one* intent
    is not diluted by the blended embedding of the full query.
    """
    if store is None:
        LOGGER.warning("Store is None; skipping RAG tool retrieval")
        return []

    tool_index_ready = config.get("configurable", {}).get("tool_index_ready", True)
    if not tool_index_ready:
        return []

    opts = config.get("configurable", {}).get("options", {})
    limit = int(opts.get(CONF_TOOL_RETRIEVAL_LIMIT, 5))
    threshold = float(opts.get(CONF_TOOL_RELEVANCE_THRESHOLD, 0.15))

    sub_queries = _split_query_intents(query)

    # best_score[tool_key] -> (score, SearchItem)
    best: dict[str, tuple[float, Any]] = {}

    for sub_query in sub_queries:
        try:
            results = await store.asearch(
                ("system", "tools"), query=sub_query, limit=limit * 4
            )
        except (
            InvalidNamespaceError,
            psycopg.OperationalError,
            psycopg.ProgrammingError,
            ValueError,
        ) as err:
            LOGGER.warning("RAG tool retrieval search failed (known error): %s", err)
            continue
        except Exception:
            LOGGER.exception("Unexpected RAG tool retrieval search failure")
            continue

        for item in results:
            name = item.value.get("name", "")
            api_id = item.value.get("api_id")
            if api_id not in allowed_api_ids:
                continue
            # score is (1 - distance) for pgvector cosine
            score = getattr(item, "score", 0.0)
            if score >= threshold and (name not in best or score > best[name][0]):
                best[name] = (score, item)

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

    return raw_tools


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

    tool_index_ready = config.get("configurable", {}).get("tool_index_ready", True)
    if not tool_index_ready or not re.search(ACTUATION_KEYWORDS_REGEX, query):
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
        ValueError,
    ) as err:
        LOGGER.warning("Deterministic safety tool filter failed (known error): %s", err)
        try:
            results = await store.asearch(("system", "tools"))
        except (
            InvalidNamespaceError,
            psycopg.OperationalError,
            psycopg.ProgrammingError,
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
        results, key=lambda i: getattr(i, "score", 0.0), reverse=True
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


def _get_fallback_tools(
    config: RunnableConfig,
    allowed_api_ids: set[str],
) -> list[RawTool]:
    """Gather all available tools as a fallback when none are selected."""
    fallback_tools: list[RawTool] = []

    ha_llm_api = config.get("configurable", {}).get("ha_llm_api")
    if ha_llm_api and hasattr(ha_llm_api, "apis"):
        for api_id, api in ha_llm_api.apis.items():
            if api_id in allowed_api_ids:
                for tool in api.tools:
                    formatted = format_tool(tool, api.custom_serializer)
                    f_func = formatted["function"]
                    name = tool.name
                    fallback_tools.append(
                        RawTool(
                            name=name,
                            api_id=api_id,
                            description=f_func.get("description", ""),
                            parameters=json.dumps(f_func["parameters"]),
                            is_actuation=is_actuation_tool(name),
                        )
                    )

    langchain_tools = config.get("configurable", {}).get("langchain_tools", {})
    for name, lc_tool in langchain_tools.items():
        params = "{}"
        if hasattr(lc_tool, "args_schema") and lc_tool.args_schema:
            try:
                params = json.dumps(lc_tool.args_schema.schema())
            except (
                AttributeError,
                TypeError,
                ValueError,
                PydanticInvalidForJsonSchema,
            ):
                params = "{}"
        fallback_tools.append(
            RawTool(
                name=name,
                api_id="hga_local",
                description=lc_tool.description,
                parameters=params,
                is_actuation=is_actuation_tool(name),
            )
        )
    return fallback_tools


def _format_and_dedupe_tools(
    raw_tools: list[RawTool],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Convert raw tools to output format and deduplicate (first-seen wins)."""
    selected_tools: list[dict[str, Any]] = []
    routing_map: dict[str, str] = {}

    for tool in raw_tools:
        name = tool["name"]
        if name in routing_map:
            continue

        routing_map[name] = tool["api_id"]
        selected_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": json.loads(tool["parameters"]),
                },
            }
        )

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

    Case 1 (normal flow): A ``requires_pin`` ToolMessage exists without a
    subsequent ``confirm_sensitive_action`` call.

    Case 2 (degraded flow): An actuation tool was routing-rejected and the model
    subsequently asked for a PIN in an AI message.  Injecting
    ``confirm_sensitive_action`` lets the model recover on the next turn.
    """
    recent = messages[-_PIN_LOOKBACK:] if len(messages) > _PIN_LOOKBACK else messages

    pin_idx = _last_requires_pin_idx(recent)
    if pin_idx is not None:
        for msg in recent[pin_idx + 1 :]:
            if isinstance(msg, ToolMessage) and msg.name == "confirm_sensitive_action":
                content = msg.content if isinstance(msg.content, str) else ""
                if "Incorrect PIN" in content:
                    # Wrong PIN entered — action is still pending, keep scanning.
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
) -> list[RawTool]:
    """Return [confirm_sensitive_action] when a PIN confirmation is pending."""
    if not _has_pending_pin_confirmation(messages):
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
    LOGGER.debug("PIN flow active: force-injecting confirm_sensitive_action")
    return [
        RawTool(
            name=val["name"],
            api_id=val["api_id"],
            description=val["description"],
            parameters=val["parameters"],
            is_actuation=val.get("is_actuation", False),
        )
    ]


async def _retrieve_tools(
    state: State,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> dict[str, Any]:
    """Retrieve relevant tools from the vector store and merge with essentials."""
    query = state["messages"][-1].content if state["messages"] else ""
    if not isinstance(query, str):
        # Multimodal content is a list of parts; extract text segments.
        query = " ".join(
            p["text"] for p in query if isinstance(p, dict) and p.get("type") == "text"
        )

    allowed_api_ids = _get_allowed_api_ids(config)
    opts = config.get("configurable", {}).get("options", {})
    limit = int(opts.get(CONF_TOOL_RETRIEVAL_LIMIT, 5))

    # 1. Gather candidates
    rag_tools, safety_tools, pin_tools = await asyncio.gather(
        _get_rag_retrieved_tools(store, config, query, allowed_api_ids),
        _get_actuation_safety_tools(store, config, query, allowed_api_ids),
        _get_pending_pin_tools(state["messages"], store),
    )

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
    pin_names = {t["name"] for t in pin_tools}
    safety_cap = min(len(safety_tools), max(1, limit * 3 // 4))
    safety_capped = [t for t in safety_tools[:safety_cap] if t["name"] not in pin_names]
    safety_names = {t["name"] for t in safety_capped}
    rag_only = [
        t
        for t in rag_tools
        if t["name"] not in safety_names and t["name"] not in pin_names
    ]
    effective_limit = min(safety_cap + len(rag_only), safety_cap + limit)
    all_candidates = pin_tools + (safety_capped + rag_only)[:effective_limit]

    # 3. Fallback: vector store unavailable or index not yet ready.
    # Still apply the limit so the user-configured cap is always respected.
    # Prioritize actuation tools for actuation queries; otherwise use
    # declaration order (HA tools first, then LangChain tools).
    if not all_candidates:
        fallback = _get_fallback_tools(config, allowed_api_ids)
        LOGGER.warning(
            "Tool index not ready or returned no results; "
            "applying keyword-filtered fallback (limit=%d, total=%d).",
            limit,
            len(fallback),
        )
        if re.search(ACTUATION_KEYWORDS_REGEX, query):
            actuation = [t for t in fallback if t["is_actuation"]]
            rest = [t for t in fallback if not t["is_actuation"]]
            fallback = actuation + rest
        all_candidates = fallback[:limit]

    # 4. Format and deduplicate
    selected_tools, routing_map = _format_and_dedupe_tools(all_candidates)

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
    }


async def _trim_messages_for_model(
    messages: list[AnyMessage],
    opts: dict[str, Any],
    chat_model_options: dict[str, Any],
    hass: HomeAssistant,
) -> list[AnyMessage]:
    """Trim messages to manage context window length."""
    _known_providers = {"openai", "openai_compatible", "gemini", "ollama"}
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


async def _invoke_model(
    model: Any, messages: list[AnyMessage], config: RunnableConfig
) -> Any:
    """Invoke a chat model, wrapping non-HA exceptions as HomeAssistantError."""
    # asyncio.timeout wraps the full block — including lock acquisition inside
    # chat_priority_context — so the total wait (queue time + inference time)
    # is bounded.  Previously wait_for started only after both locks were held,
    # leaving lock-wait time unbounded if a background job was slow to finish.
    try:
        async with asyncio.timeout(_LLM_INVOKE_TIMEOUT_S):
            async with chat_priority_context():
                return await model.ainvoke(messages, config)
    except TimeoutError:
        msg = f"Model invocation timed out after {_LLM_INVOKE_TIMEOUT_S}s."
        raise HomeAssistantError(msg) from None
    except HomeAssistantError:
        raise
    except Exception as err:
        msg = f"Model invocation failed: {err}"
        raise HomeAssistantError(msg) from err


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

    # Retrieve memories (semantic if last message is from user).
    last_message = state["messages"][-1]
    last_message_from_user = isinstance(last_message, HumanMessage)
    query_prompt = None
    if last_message_from_user:
        query_prompt = EMBEDDING_MODEL_PROMPT_TEMPLATE.format(
            query=last_message.content
        )

    mems = await store.asearch((user_id, "memories"), query=query_prompt, limit=10)

    # Recent camera activity.
    camera_activity = await get_recent_camera_activity(hass, store)

    # Build system message.
    system_message = conf["prompt"]
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

    trimmed_messages = await _trim_messages_for_model(
        messages, opts, chat_model_options, hass
    )

    LOGGER.debug("Model call messages: %s", trimmed_messages)

    # Bind retrieved tools
    selected_tools = state.get("selected_tools", [])
    if selected_tools:
        # Disable reasoning/thinking before binding tools: Qwen3's <think> tokens
        # interleave with tool call JSON output and break Ollama's qwen3.go parser
        # (ResponseError: "invalid character 'g' looking for beginning of value").
        if chat_model_options.get("reasoning"):
            model = model.with_config(config={"configurable": {"reasoning": False}})
        model = model.bind_tools(selected_tools)

    # Pass routing map to MultiLLMAPI
    routing_map = state.get("tool_routing_map", {})
    llm_api = conf.get("ha_llm_api")
    if llm_api and hasattr(llm_api, "routing_map"):
        llm_api.routing_map = routing_map

    raw_response = await _invoke_model(model, trimmed_messages, config)
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
        options.get(CONF_CRITICAL_ACTION_PIN_ENABLED, False) or pin_configured
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

    tool_routing_map: dict[str, str] = state.get("tool_routing_map", {})

    # Execute all tool calls concurrently.
    # asyncio.gather preserves submission order: results[i] matches tool_calls[i].
    # Concurrent HA tool calls may interleave (e.g. turn_on + get_state in same
    # batch). If strict sequencing is needed, issue separate model turns.
    async def _invoke_one(tool_call: dict[str, Any]) -> ToolMessage:
        """Execute a single tool call and return its ToolMessage."""
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {}) or {}
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
workflow.add_node("retrieve_tools", _retrieve_tools)
workflow.add_node("agent", _call_model)
workflow.add_node("action", _call_tools)
workflow.add_node("summarize_and_remove_messages", _summarize_and_remove_messages)

# Define edges.
workflow.add_edge(START, "retrieve_tools")
workflow.add_edge("retrieve_tools", "agent")
workflow.add_conditional_edges("agent", _should_continue)
workflow.add_edge("action", "agent")
workflow.add_edge("summarize_and_remove_messages", END)
