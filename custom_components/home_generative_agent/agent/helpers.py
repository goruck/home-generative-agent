"""Shared agent helpers for argument normalization and resolution."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from homeassistant.core import HomeAssistant


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
