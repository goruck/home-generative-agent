"""Helper utilities for Home Generative Agent tool conversion and normalization."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, TypedDict

import voluptuous as vol
from voluptuous_openapi import UNSUPPORTED, convert

from custom_components.home_generative_agent.const import (
    ACTUATION_LANGCHAIN_TOOLS,
    ACTUATION_TOOL_PREFIXES,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from homeassistant.core import HomeAssistant
    from homeassistant.helpers import llm


def safe_convert(
    schema: Any, custom_serializer: Callable[[Any], Any] | None = None
) -> Any:
    """Safely convert a voluptuous schema to OpenAPI, handling unhashable Selectors."""

    def robust_serializer(obj: Any) -> Any:
        """Robustly handle HA types that might be unhashable or need mapping."""
        # 0. Skip basic types that voluptuous_openapi handles natively.
        # vol.Schema, dict, and list are unhashable but handled by the library.
        if isinstance(obj, (vol.Schema, dict, list)):
            return UNSUPPORTED

        # 1. First, call the original custom serializer if it exists
        if custom_serializer:
            try:
                res = custom_serializer(obj)
                # If external serializer returns non-None/UNSUPPORTED, use it.
                # Some HA serializers might return None for unhandled types.
                if res is not None and res is not UNSUPPORTED:
                    return res
            except Exception:  # noqa: BLE001, S110
                # Fallback to the robust serializer if custom_serializer fails.
                pass

        # 2. Check for Home Assistant Selectors by looking for 'config'
        # These are often unhashable and require specific extraction.
        config = getattr(obj, "config", obj if isinstance(obj, dict) else {})
        if isinstance(config, dict) and "options" in config:
            # Extract options into an Enum for SelectSelector
            raw_options = config.get("options")
            if isinstance(raw_options, list):
                options = [
                    opt.get("value", opt) if isinstance(opt, dict) else opt
                    for opt in raw_options
                ]
                return {"type": "string", "enum": options}

        # 3. Handle other selectors that have a config but aren't SelectSelector
        if hasattr(obj, "config") or (isinstance(obj, dict) and "selector" in obj):
            return {"type": "string"}

        # 4. General unhashable safety net to prevent voluptuous_openapi crash
        try:
            hash(obj)
        except TypeError:
            return {"type": "string"}

        return UNSUPPORTED

    return convert(schema, custom_serializer=robust_serializer)


class ConfigurableData(TypedDict, total=False):
    """Typed view of the configurable payload passed through tools."""

    options: Mapping[str, Any]
    pending_actions: dict[str, dict[str, Any]]
    hass: HomeAssistant
    user_id: str
    ha_llm_api: Any


def sanitize_tool_args(tool_args: dict[str, Any]) -> dict[str, Any]:
    """Remove empty/None slot values for HA intent tools to avoid validation errors."""
    cleaned: dict[str, Any] = {}
    for key, val in tool_args.items():
        if val is None:
            continue
        if isinstance(val, str) and not val.strip():
            continue
        if isinstance(val, list) and all((not v and v != 0) for v in val):
            continue
        cleaned[key] = val
    if "domain" in cleaned and isinstance(cleaned["domain"], str):
        cleaned["domain"] = [cleaned["domain"]]
    return cleaned


def maybe_fill_lock_entity(
    tool_args: dict[str, Any], hass: HomeAssistant | None
) -> dict[str, Any]:
    """Best-effort map a friendly lock name to an entity_id without heavy fuzzing."""
    domains = tool_args.get("domain") or []
    domains = domains if isinstance(domains, list) else [domains]
    if "lock" not in {str(d).lower() for d in domains}:
        return tool_args
    if not hass or tool_args.get("entity_id"):
        return tool_args

    name_hint = str(tool_args.get("name") or "").strip()
    if not name_hint:
        return tool_args

    def _slugify(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

    target_slug = _slugify(name_hint)
    target_compact = target_slug.replace("_", "")
    target_with_lock = f"{target_compact}lock"
    best_match: tuple[int, int, str, str] | None = None

    for entity_id in hass.states.async_entity_ids("lock"):
        state_obj = hass.states.get(entity_id)
        friendly = (
            state_obj.attributes.get("friendly_name") if state_obj else ""
        ) or entity_id
        cand_slug = _slugify(friendly)
        cand_compact = cand_slug.replace("_", "")
        if cand_slug == target_slug:
            tool_args["entity_id"] = entity_id
            tool_args["name"] = friendly
            return tool_args
        if cand_compact in (target_compact, target_with_lock):
            tool_args["entity_id"] = entity_id
            tool_args["name"] = friendly
            return tool_args

        rank = (
            1 if target_compact in cand_compact or cand_compact in target_compact else 2
        )
        diff = abs(len(cand_compact) - len(target_compact))
        if best_match is None or (rank, diff) < best_match[:2]:
            best_match = (rank, diff, entity_id, friendly)

    if not tool_args.get("entity_id") and best_match:
        _, _, entity_id, friendly = best_match
        tool_args["entity_id"] = entity_id
        tool_args["name"] = friendly

    return tool_args


def normalize_intent_for_alarm(
    tool_name: str, tool_args: dict[str, Any]
) -> dict[str, Any]:
    """Heuristic to route alarm control panel intents to the proper service."""
    if tool_name not in {"HassTurnOn", "HassTurnOff"}:
        return tool_args

    domains = tool_args.get("domain") or []
    domains = domains if isinstance(domains, list) else [domains]
    name_hint = str(tool_args.get("name", "")).lower()
    if not domains and any(k in name_hint for k in ("alarm", "security")):
        tool_args = {**tool_args, "domain": ["alarm_control_panel"]}
        domains = tool_args["domain"]

    if not any(str(d).lower() == "alarm_control_panel" for d in domains):
        return tool_args

    is_arm = tool_name == "HassTurnOn"
    desired_service = "alarm_arm_home" if is_arm else "alarm_disarm"
    tool_args = {
        **tool_args,
        "domain": ["alarm_control_panel"],
        "service": desired_service,
    }
    if "entity_id" not in tool_args and (name := tool_args.get("name")):
        slug = str(name).strip().lower().replace(" ", "_")
        tool_args["entity_id"] = f"alarm_control_panel.{slug}"
    return tool_args


def normalize_intent_for_lock(
    tool_name: str, tool_args: dict[str, Any]
) -> dict[str, Any]:
    """Normalize lock intents: set domain and service for lock/unlock."""
    if tool_name not in {"HassTurnOn", "HassTurnOff"}:
        return tool_args

    name = str(tool_args.get("name", "")).lower()
    domains = tool_args.get("domain") or []
    domains = domains if isinstance(domains, list) else [domains]
    is_lock = any(str(d).lower() == "lock" for d in domains) or "lock" in name
    if not is_lock:
        return tool_args

    normalized = {**tool_args, "domain": ["lock"]}
    if tool_name == "HassTurnOff":
        normalized.setdefault("service", "unlock")
    else:
        normalized.setdefault("service", "lock")
    return normalized


def format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format Home Assistant LLM tools to be compatible with OpenAI format."""
    tool_spec = {
        "name": tool.name,
        "parameters": safe_convert(
            tool.parameters, custom_serializer=custom_serializer
        ),
    }
    if tool.description:
        tool_spec["description"] = tool.description
    return {"type": "function", "function": tool_spec}


def is_actuation_tool(name: str) -> bool:
    """Check if a tool name indicates an actuation tool."""
    name_lower = name.lower()
    # Check exact matches first (greedy check for specific tools)
    if name_lower in {t.lower() for t in ACTUATION_LANGCHAIN_TOOLS}:
        return True
    # Check prefix matches for provider tools
    return any(name_lower.startswith(p.lower()) for p in ACTUATION_TOOL_PREFIXES)
