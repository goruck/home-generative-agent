"""Helper utilities for Home Generative Agent tool conversion and normalization."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict

import homeassistant.util.dt as dt_util
import voluptuous as vol
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import llm
from homeassistant.util import ulid
from voluptuous_openapi import UNSUPPORTED, convert

from custom_components.home_generative_agent.const import (
    ACTUATION_LANGCHAIN_TOOLS,
    ACTUATION_TOOL_PREFIXES,
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
    CONF_CRITICAL_ACTIONS,
    RECOMMENDED_CRITICAL_ACTIONS,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from homeassistant.core import HomeAssistant


def active_llm_api_ids(options: Mapping[str, Any]) -> list[str]:
    """
    Return the LLM API ids to expose, defaulting to Assist when unset.

    Single definition on purpose.  The options flow *deletes* the key when no
    API is selected (``_cleanup_none_llm_api`` in ``config_flow.py``), so an
    absent key is a normal state meaning "the recommended default", not "no
    APIs at all".  Every reader must apply that same default, because they are
    not independent: ``conversation.py`` derives
    ``ConversationEntityFeature.CONTROL`` from this, and Home Assistant reads
    that feature flag to decide whether its built-in intent handler may take
    control commands *before* the agent ever sees them
    (``assist_pipeline/pipeline.py`` installs a local-intent allowlist only for
    agents that advertise CONTROL).  A reader that treats the absent key as
    "no APIs" therefore drops the flag and silently routes lock/unlock around
    the critical-action PIN while the rest of the integration still runs the
    Assist API.
    """
    if CONF_LLM_HASS_API not in options:
        return [llm.LLM_API_ASSIST]
    return normalize_llm_api_value(options[CONF_LLM_HASS_API])


def normalize_llm_api_value(raw: Any) -> list[str]:
    """
    Normalize a stored ``CONF_LLM_HASS_API`` value to a list of id strings.

    Storage has carried several shapes over the integration's life: a list of
    ids, a pre-v6 bare string, and — via programmatic options updates that
    bypass the form schema — ``None``, ``""``, or lists holding non-string
    elements.  Iterating or set-testing those degenerate shapes crashes both
    the options-form build and ``active_llm_api_ids`` at runtime (issue #568's
    failure class), so every reader funnels through this one normalizer:
    strings wrap to a single-element list, everything non-list collapses to
    ``[]``, and non-string or empty elements are dropped.
    """
    if isinstance(raw, str):
        return [raw] if raw else []
    if not isinstance(raw, list):
        return []
    return [api_id for api_id in raw if isinstance(api_id, str) and api_id]


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


# How long a pending PIN confirmation stays valid. Mirrors the expiry check in
# `_load_pending_action`, which is lazy and only ever inspects the one action
# being confirmed.
PENDING_ACTION_TTL = timedelta(minutes=10)

# Hard cap on concurrently pending confirmations, so an unconfirmed challenge
# loop cannot grow the store without bound.
MAX_PENDING_ACTIONS = 16


class CriticalActionPolicy(NamedTuple):
    """Resolved critical-action PIN policy for a config entry."""

    enabled: bool
    pin_hash: str
    pin_salt: str
    critical_actions: list[dict[str, str]]

    @property
    def enforceable(self) -> bool:
        """Return True when the PIN is enabled *and* actually configured."""
        return self.enabled and bool(self.pin_hash and self.pin_salt)


def resolve_critical_action_policy(options: Mapping[str, Any]) -> CriticalActionPolicy:
    """Resolve the critical-action PIN policy from integration options."""
    pin_hash = options.get(CONF_CRITICAL_ACTION_PIN_HASH) or ""
    pin_salt = options.get(CONF_CRITICAL_ACTION_PIN_SALT) or ""
    return CriticalActionPolicy(
        # Always respect a configured PIN, even if the toggle somehow reads False.
        enabled=bool(
            options.get(CONF_CRITICAL_ACTION_PIN_ENABLED, False)
            or (pin_hash and pin_salt)
        ),
        pin_hash=pin_hash,
        pin_salt=pin_salt,
        # Copy: the default is a module-level list handed to every config
        # entry, so an in-place edit anywhere would corrupt it process-wide.
        critical_actions=list(
            options.get(CONF_CRITICAL_ACTIONS) or RECOMMENDED_CRITICAL_ACTIONS
        ),
    )


def register_pending_action(
    pending_actions: dict[str, dict[str, Any]],
    action: dict[str, Any],
) -> str:
    """
    Store a pending PIN-confirmation action and return its action ID.

    Sweeps expired entries and caps the store on every registration. Without
    this the dict only ever shrinks when a specific action ID is confirmed, so
    every challenge the model abandons is retained for the life of the config
    entry — and once more than one entry accumulates, the single-pending-action
    convenience path in ``_resolve_action_id`` stops resolving.
    """
    now = dt_util.utcnow()
    for key, existing in list(pending_actions.items()):
        created_at = existing.get("created_at")
        if not isinstance(created_at, str):
            # A record with no usable timestamp can never expire, so skipping
            # it would make it immortal and let the cap evict live entries
            # around it. `_load_pending_action` rejects it anyway.
            pending_actions.pop(key, None)
            continue
        try:
            created = datetime.fromisoformat(created_at)
            expired = now - created > PENDING_ACTION_TTL
        except (ValueError, TypeError):
            # TypeError covers a timezone-naive stored timestamp, which cannot
            # be compared against the aware `now`. One malformed record must
            # not break every later registration.
            pending_actions.pop(key, None)
            continue
        if expired:
            pending_actions.pop(key, None)

    while len(pending_actions) >= MAX_PENDING_ACTIONS:
        pending_actions.pop(next(iter(pending_actions)), None)

    action_id = ulid.ulid_now()
    pending_actions[action_id] = {
        **action,
        "created_at": now.isoformat(),
        "attempts": 0,
    }
    return action_id


def matches_critical_rule(
    *,
    domain: str,
    service: str,
    entity_ids: Sequence[str],
    critical_actions: Iterable[Mapping[str, str]],
    unresolved_target: bool = False,
) -> bool:
    """
    Return True when a domain/service call matches a critical-action rule.

    Shared by the conversation-tool guard (``_is_critical_action``) and the
    automation-content guard (``automation_pin``) so both enforce identical
    rule semantics.

    ``unresolved_target`` fails closed for ``entity_match`` rules: when a call
    targets an area, device, label, or template, the entity IDs it will hit are
    unknown at check time, so an otherwise-matching rule is treated as a match.
    """
    domain = domain.lower()
    service = service.lower()
    entities = [e.lower() for e in entity_ids]

    for rule in critical_actions:
        rule_domain = (rule.get("domain") or "").lower()
        rule_service = (rule.get("service") or "").lower()
        entity_match = (rule.get("entity_match") or "").lower()

        if rule_domain and rule_domain != domain:
            continue
        if rule_service and rule_service != service:
            continue
        if (
            entity_match
            and not any(entity_match in e for e in entities)
            and not unresolved_target
        ):
            continue
        return True
    return False


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


def tool_index_key(api_id: str, name: str) -> str:
    """
    Composite tool-index store key.

    Single definition on purpose: the per-turn top-up compares live keys
    against cached hashes, so any drift between the key spelling used for
    discovery writes and the one used for the live-set comparison would make
    a key permanently "missing" and re-fire the top-up every turn.
    """
    return f"{api_id}::{name}"


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
