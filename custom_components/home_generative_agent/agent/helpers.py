"""Helper utilities for Home Generative Agent tool conversion and normalization."""

from __future__ import annotations

import logging
import re
from dataclasses import replace
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
    CONF_TOOL_EXCLUSIONS,
    RECOMMENDED_CRITICAL_ACTIONS,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)


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


def normalize_tool_exclusions(raw: Any) -> dict[str, list[str]]:
    """
    Normalize a stored ``CONF_TOOL_EXCLUSIONS`` value to ``{api_id: [name]}``.

    Mirrors ``normalize_llm_api_value``'s contract: every reader funnels through
    one normalizer so a degenerate shape written by a programmatic options
    update (``None``, a bare string, non-string elements) can never crash the
    options-form build or the per-turn tool filter.  API ids mapping to no
    surviving names are dropped, because "present with an empty list" and
    "absent" mean the same thing — expose every tool of that API — and keeping
    both spellings would invite readers to disagree about which is which.
    """
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for api_id, raw_names in raw.items():
        if not isinstance(api_id, str) or not api_id:
            continue
        names = [raw_names] if isinstance(raw_names, str) else raw_names
        if not isinstance(names, list):
            continue
        cleaned = sorted({n for n in names if isinstance(n, str) and n})
        if cleaned:
            normalized[api_id] = cleaned
    return normalized


def api_id_is_form_representable(api_id: str) -> bool:
    """
    Return whether an API id can round-trip through the options form.

    The picker flattens exclusions to ``api_id::tool_name`` values and regroups
    them by splitting on the FIRST separator, so an id containing the separator
    does not survive: ``{"a::b": ["t"]}`` comes back as ``{"a": ["b::t"]}``, an
    exclusion matching nothing.

    This is deliberately **not** enforced inside ``normalize_tool_exclusions``.
    That normalizer is shared by the options form and the per-turn runtime
    filter, and the two want opposite things: the form must refuse to *offer*
    an exclusion it cannot store faithfully, while the runtime must keep
    *honouring* one that is already stored. Refusing in the shared path means a
    stored exclusion for such an id stops being enforced — the tools silently
    go live again — and is then erased from storage by the next unrelated save.
    Fail-open on a security control, so: refuse at the form boundary only.
    """
    return TOOL_KEY_SEP not in api_id


def tool_exclusions(options: Mapping[str, Any]) -> dict[str, set[str]]:
    """
    Return the configured per-API excluded tool names as lookup sets.

    Sets, not the sorted lists ``normalize_tool_exclusions`` returns, because
    this is the per-turn runtime path: ``filter_excluded_tools`` membership-
    tests every tool of every loaded API against them.  Storage and the options
    form want the list form instead — it is what round-trips through JSON and
    what gives the picker a stable order — so the two shapes are deliberate,
    not redundant.  Take this one when you are filtering, that one when you are
    reading or writing the stored value.
    """
    raw = options.get(CONF_TOOL_EXCLUSIONS)
    return {
        api_id: set(names) for api_id, names in normalize_tool_exclusions(raw).items()
    }


def filter_excluded_tools(
    api_id: str,
    api: llm.APIInstance,
    excluded: Mapping[str, set[str]],
) -> tuple[llm.APIInstance, list[str]]:
    """
    Return an APIInstance exposing only the tools the user has not excluded.

    Returns the (possibly unchanged) instance and the names actually dropped.

    A *copy* is returned rather than an in-place edit of ``api.tools``: Home
    Assistant's MCP integration hands every ``APIInstance`` the very same
    coordinator-owned list object (``tools=self.coordinator.data``), so mutating
    it would strip the excluded tools from every other consumer of that server
    in the whole instance — and keep them stripped until the coordinator's next
    30-minute refresh.

    Filtering here, on the instance the conversation entity loads, is the single
    enforcement point: tool binding, the RAG fallback set, the live-tool filter
    and ``APIInstance.async_call_tool`` all read ``.tools``, so an excluded tool
    is not merely hidden from the model — a hallucinated call to it is rejected
    deterministically with ``Tool "x" not found``.
    """
    names = excluded.get(api_id)
    if not names:
        return api, []
    # `.tools` is None for an MCP API whose coordinator has not completed its
    # first refresh (HA builds the instance with `tools=self.coordinator.data`),
    # and a third-party API may pass None freely. The options form already
    # guards this with `instance.tools or []`; unguarded here it raises
    # TypeError on the per-turn path, outside the caller's
    # `except HomeAssistantError`, and kills every conversation turn until the
    # coordinator recovers.
    tools = list(api.tools or [])
    # `tool.name` is remote data and is not guaranteed to be a string, let alone
    # a hashable one. An unguarded `tool.name not in names` raises TypeError on
    # a list/dict name, and this runs on the per-turn path *outside* the
    # `except HomeAssistantError` in `_async_init_llm_apis` -- so an unguarded
    # comprehension turns one malformed descriptor into every conversation turn
    # failing, for as long as that server advertises it. A non-str name can
    # never equal an excluded name, so treating it as "keep" is also correct,
    # not merely safe. Guarding here rather than wrapping the call in the caller
    # is deliberate: a try/except around a security control fails *open*.
    live_names = [
        tool.name for tool in tools if isinstance(getattr(tool, "name", None), str)
    ]
    kept = [
        tool
        for tool in tools
        if not (isinstance(getattr(tool, "name", None), str) and tool.name in names)
    ]
    if len(kept) == len(tools):
        return api, []
    dropped = sorted(set(live_names) & names)
    # replace() rather than the constructor so a field added to APIInstance by a
    # future Home Assistant release is carried over instead of silently reset.
    return replace(api, tools=kept), dropped


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


# Separator between the API id and the tool name in a composite tool key.
# Shared by `tool_index_key` and `split_tool_index_key` so the join and the
# split can never drift — the options flow round-trips these keys through a
# flat picker value, and a one-sided change would void every stored exclusion.
TOOL_KEY_SEP = "::"

# Cap for any single rendered tool string. Long enough for real MCP tool names,
# short enough that one hostile name cannot flood a log line or a form label.
TOOL_TEXT_MAX_LEN = 120


def sanitize_tool_text(text: str, limit: int = TOOL_TEXT_MAX_LEN) -> str:
    """
    Make a remote-controlled tool string safe to render, and bound its length.

    Tool names and API titles arrive from remote MCP servers, so they are
    attacker-controlled text that this integration renders in two places: log
    lines and the options-form exclusion picker.  ``str.isprintable()`` is
    False for exactly the classes that make rendered text lie — C0/C1 controls,
    newlines, zero-width characters, and the bidi overrides (U+202E and
    friends) that reverse the text after them — so replacing every
    non-printable with ``?`` closes log-line forgery and the reordering
    attacks.  ASCII space stays printable, so ordinary names are untouched.

    What this deliberately does **not** do is stop a name whose characters are
    all perfectly printable from *reading* like something else.
    ``list_files (not currently available)`` passes through here unchanged, so
    this function alone cannot keep remote text from imitating a trust marker
    it is concatenated with; the picker's own ``_label_text`` in
    ``config_flow.py`` is what does that.  Treat this as making remote text
    safe to *render*, not safe to *concatenate with trusted text*.

    Single definition on purpose: a second copy would drift from this one, and
    a renderer that forgot to sanitize is exactly the hole this closes.

    The cap is applied *before* the per-character pass, not after. The
    transform is one character in, one character out, so the two orders return
    byte-identical strings — but capping afterwards would run the Python-level
    generator across the whole input first, which is unbounded in the length of
    text a remote server chose. A multi-megabyte tool name costs ~132ms of
    event-loop block that way against ~0.05ms this way, which is the difference
    between a cap that bounds rendering and a cap that only bounds the result.
    """
    # Fast path: one C-level scan, and it is what virtually every real tool
    # name hits, so the common case never enters the generator at all.
    if len(text) <= limit and text.isprintable():
        return text
    return "".join(ch if ch.isprintable() else "?" for ch in text[:limit])


def tool_index_key(api_id: str, name: str) -> str:
    """
    Composite tool-index store key.

    Single definition on purpose: the per-turn top-up compares live keys
    against cached hashes, so any drift between the key spelling used for
    discovery writes and the one used for the live-set comparison would make
    a key permanently "missing" and re-fire the top-up every turn.
    """
    return f"{api_id}{TOOL_KEY_SEP}{name}"


def split_tool_index_key(key: str) -> tuple[str, str] | None:
    """
    Split a composite tool key back into ``(api_id, tool_name)``.

    Returns ``None`` for anything that is not a well-formed key.  Splits on the
    *first* separator: registered API ids never contain ``::`` (Home Assistant
    builds them from a domain or a config-entry id), while a remote MCP server
    is free to name a tool anything at all.
    """
    api_id, sep, name = key.partition(TOOL_KEY_SEP)
    if not sep or not api_id or not name:
        return None
    return api_id, name


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
