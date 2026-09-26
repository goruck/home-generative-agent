"""
Existence checks for the entities and services an LLM-authored automation names.

The Assist API shows the model friendly names, domains and areas, never entity
IDs, so an automation it writes carries entity IDs it *derived* from names
(``binary_sensor.sink_moisture_sensor`` for "Sink Moisture Sensor") and service
names it remembered from training (``notify.mobile_app``). Home Assistant's
automation validator checks only the schema: a trigger on an entity that does
not exist and an action calling a service that does not exist are both valid
config, so the automation installs, the tool reports success, and the
automation never does anything. Nothing is logged until the trigger fires,
and a trigger on a missing entity never fires.

This module walks the validated config for the entity references Home
Assistant resolves against the state machine (``entity_id`` anywhere;
``zone``, ``scene``, ``at``, ``before``, ``after``, ``above`` and ``below``
in triggers, conditions and steps; ``snapshot_entities``, ``entities``,
``add_entities`` and ``remove_entities`` in a service's data) and for every
service call, and reports the ones Home Assistant does not know, each with
the closest real thing so the model's retry is deterministic:

* an Assist-exposed entity whose friendly name slugifies to the guessed
  object id (all of them, when several share the name; in any domain, since
  ``sensor.`` for a ``binary_sensor.`` is the other common slip), or failing
  that the closest exposed entity id in the same domain;
* for a ``notify.X`` written as an entity, that ``notify.X`` is the service
  to call;
* for a missing ``notify.*`` service, the mobile push service configured for
  this integration, then every ``notify.mobile_app_*`` service;
* for any other missing service, the services its domain does offer.

Suggestions draw only from entities exposed to Assist: the model can only
guess a name it was shown, and naming a hidden entity's real id would turn a
typo into a lookup of what the user chose not to expose. The existence check
itself accepts any real entity, exposed or not, so an id the user dictated
still installs.

Left to Home Assistant: templates, entity registry IDs, ``entity_id: all``,
areas, devices and labels, a device action's ``device_id`` and ``type``,
opaque payloads (``event_data``, ``variables``), any other key inside a
service's data (a script's arguments are whatever the script says they are),
and anything under a step, trigger or condition with ``enabled: false``.
Entities the automation creates for itself (``scene.create``, ``group.set``,
its own ``automation.<alias>``) count as existing in its actions, but not in
its triggers or conditions, which must hold before anything is created. This
is a check for the invented name, not a second validator.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from homeassistant.const import ENTITY_MATCH_ALL, ENTITY_MATCH_NONE
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import entity_registry as er
from homeassistant.util import slugify

from ..core.utils import list_mobile_notify_services  # noqa: TID252
from .automation_pin import (
    _ENTITY_ID_RE,
    _INDIRECTION_DOMAINS,
    _MAX_STEP_DEPTH,
    _is_template,
    _service_name,
    _TooDeepError,
    _walk_steps,
)
from .helpers import sanitize_tool_text

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from homeassistant.core import HomeAssistant, State

# The assistant whose exposure settings bound what the model may be told.
_ASSISTANT = "conversation"

# Values of an entity-reference key that are keywords rather than entities.
_ENTITY_ID_KEYWORDS = frozenset({ENTITY_MATCH_ALL, ENTITY_MATCH_NONE})

# The one key that names entities wherever it appears, service data included.
_ENTITY_ID_KEY = "entity_id"

# Keys that name entities in a trigger, condition or step, but that inside a
# service's data are just arguments (`zone: us.east` handed to a script).
# A time trigger's `at`, a time condition's `before`/`after` and a
# numeric_state threshold accept a clock time or number as well as an entity
# id; anything that is not an entity id is skipped by the regex.
_STEP_REF_KEYS = frozenset({"zone", "scene", "at", "before", "after", "above", "below"})

# Keys inside a service's data whose values (or dict keys) are entity ids:
# scene.create's snapshot, scene.apply's states, group.set's members. Only
# for those services: a script called with `entities:` receives a variable.
_DATA_REF_KEYS = frozenset(
    {"snapshot_entities", "entities", "add_entities", "remove_entities"}
)
_DATA_REF_DOMAINS = frozenset({"scene", "group"})

# Keys that open a service call's data.
_DATA_KEYS = frozenset({"data", "data_template", "service_data"})

# Subtrees that carry opaque payloads: an event's data goes to whoever listens
# for the event and a variable is whatever the author says it is, so an
# `entity_id` key inside them is not a promise about this home.
_OPAQUE_KEYS = frozenset(
    {"event_data", "event_data_template", "variables", "trigger_variables", "for_each"}
)

# How many missing entities get a fuzzy lookup per call. Each lookup scans a
# domain's exposed states once (the index is shared across the call); the
# rest are still reported as missing, without a suggestion.
_MAX_SUGGESTIONS = 5

# How many same-name entities to list when a friendly name is not unique.
_MAX_SAME_NAME = 5

# How many of a domain's real services to offer for a missing one.
_MAX_SERVICE_SUGGESTIONS = 8

# How many findings the tool response lists before summarizing the rest.
_MAX_REPORTED = 20

# Friendly names are device- and user-supplied text rendered into a reply the
# model is told to act on: bound them and strip anything non-printable.
_MAX_NAME_CHARS = 60

# Fuzzy-match floor for the closest-id fallback (difflib ratio).
_FUZZY_CUTOFF = 0.6

# Services Home Assistant itself provides on the script/automation domains,
# as opposed to the user's own `script.<name>` entries.
_GENERIC_SERVICES = frozenset({"turn_on", "turn_off", "toggle", "reload", "trigger"})
_GENERIC_SERVICE_DOMAINS = frozenset({"automation", "script"})

# Every refusal this module (or the blueprint guard) returns starts with
# this, so the graph can tell a refused automation from an executed action.
AUTOMATION_REFUSAL_PREFIX = "Automation not added"

# Report kinds. "automation" is a whole-config finding with no name.
KIND_ENTITY = "entity"
KIND_SERVICE = "service"
KIND_AUTOMATION = "automation"


@dataclass(frozen=True)
class MissingAutomationTarget:
    """An entity or service the automation names that Home Assistant lacks."""

    kind: str  # KIND_ENTITY, KIND_SERVICE or KIND_AUTOMATION
    name: str
    detail: str

    def describe(self) -> str:
        """Return one line the model can act on."""
        if self.kind == KIND_AUTOMATION:
            return f"The automation {self.detail}"
        return f"{self.kind.capitalize()} '{self.name}' {self.detail}"


def _label(state: State) -> str:
    """Return a state's display name, bounded and safe to render."""
    return sanitize_tool_text(state.name, limit=_MAX_NAME_CHARS)


def _entity_candidates(value: Any) -> Iterator[str]:
    """Yield the concrete entity ids one entity-reference value names."""
    if isinstance(value, dict):
        if _ENTITY_ID_KEY in value:
            # A time trigger's `at: {entity_id: sensor.x, offset: ...}`.
            yield from _entity_candidates(value[_ENTITY_ID_KEY])
            return
        # scene.apply's `entities: {light.x: on}` keys entities by id.
        for key in value:
            yield from _entity_candidates(key)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _entity_candidates(item)
        return
    if not isinstance(value, str) or _is_template(value):
        return
    # Service data is free-form, so HA splits a comma-separated entity_id
    # string into targets at call time; validated targets are already lists.
    for part in value.split(","):
        lowered = part.strip().lower()
        if lowered in _ENTITY_ID_KEYWORDS or not _ENTITY_ID_RE.fullmatch(lowered):
            continue
        if _is_number(lowered):
            # "25.5" satisfies the entity-id shape but is a threshold.
            continue
        yield lowered


def _is_number(value: str) -> bool:
    """Return True for a string that is a number, however it is spelled."""
    try:
        float(value)
    except ValueError:
        return False
    return True


def _is_disabled(node: Mapping[str, Any]) -> bool:
    """Return True for a trigger, condition or step HA will skip outright."""
    return node.get("enabled") is False


def _walk_entity_refs(
    node: Any, depth: int = 0, *, data_of: str | None = None
) -> Iterator[str]:
    """
    Yield every concrete entity id the config resolves against the home.

    ``data_of`` is the service domain when below a service call's data, where
    only ``entity_id`` (and, for scene and group services, their entity
    containers) is a reference and every other key is an argument. Raises
    ``_TooDeepError`` past the step walker's depth cap so a hand-built config
    fails closed rather than passing half-checked.
    """
    if depth > _MAX_STEP_DEPTH:
        raise _TooDeepError
    if isinstance(node, (list, tuple)):
        # Lists are free: a nesting level is a dict, so the cap counts
        # containers the way an author sees them, not YAML punctuation.
        for item in node:
            yield from _walk_entity_refs(item, depth, data_of=data_of)
        return
    if not isinstance(node, dict) or _is_disabled(node):
        return
    service = _step_service(node) if data_of is None else None
    for key, value in node.items():
        if key == _ENTITY_ID_KEY:
            yield from _entity_candidates(value)
        elif data_of is not None:
            if key in _DATA_REF_KEYS and data_of in _DATA_REF_DOMAINS:
                yield from _entity_candidates(value)
            else:
                yield from _walk_entity_refs(value, depth + 1, data_of=data_of)
        elif key in _STEP_REF_KEYS:
            yield from _entity_candidates(value)
        elif key in _DATA_KEYS:
            domain = service.partition(".")[0] if service else ""
            yield from _walk_entity_refs(value, depth + 1, data_of=domain)
        elif key not in _OPAQUE_KEYS:
            yield from _walk_entity_refs(value, depth + 1)


def _own_automation_entity(config: Mapping[str, Any]) -> set[str]:
    """Return the automation's own entity id, which exists once it installs."""
    alias = config.get("alias")
    if isinstance(alias, str) and alias.strip():
        return {f"automation.{slugify(alias)}"}
    return set()


def _created_by(actions: Any) -> set[str]:
    """
    Return the entity ids the given steps bring into existence.

    A ``scene.create`` or ``group.set`` step creates an entity that later
    steps may target; it does not exist at write time.
    """
    created: set[str] = set()
    steps = list(_walk_steps(_without_disabled(actions)))
    for step, action_type in steps:
        if action_type != cv.SCRIPT_ACTION_CALL_SERVICE:
            continue
        service = _step_service(step)
        # HA merges `data` and the legacy `data_template` at call time.
        data: dict[str, Any] = {}
        for key in _DATA_KEYS:
            if isinstance(step.get(key), dict):
                data.update(step[key])
        if service == "scene.create":
            object_id = data.get("scene_id")
        elif service == "group.set":
            object_id = data.get("object_id")
        else:
            continue
        if isinstance(object_id, str) and not _is_template(object_id):
            created.add(f"{service.partition('.')[0]}.{slugify(object_id)}")
    return created


def _without_disabled(node: Any, depth: int = 0) -> Any:
    """Return a copy of a config tree with every ``enabled: false`` node removed."""
    if depth > _MAX_STEP_DEPTH:
        raise _TooDeepError
    if isinstance(node, (list, tuple)):
        return [
            _without_disabled(item, depth)
            for item in node
            if not (isinstance(item, dict) and _is_disabled(item))
        ]
    if isinstance(node, dict):
        return {key: _without_disabled(value, depth + 1) for key, value in node.items()}
    return node


def _exposure_available(hass: HomeAssistant) -> bool:
    """Return True if Home Assistant's exposure registry is up."""
    from homeassistant.components.homeassistant.const import (  # noqa: PLC0415
        DATA_EXPOSED_ENTITIES,
    )

    # Created by the homeassistant component at startup; without it nothing
    # can be shown to be exposed, so no suggestion is made rather than a
    # wrong one.
    return DATA_EXPOSED_ENTITIES in hass.data


def _exposed(hass: HomeAssistant, entity_id: str) -> bool:
    """
    Return True if the conversation assistant may see this entity.

    Home Assistant records the answer as an entity option the first time it
    is asked, which is what the Assist API does for every entity on every
    turn; callers memoise per call so no entity is asked twice here.
    """
    from homeassistant.components.homeassistant.exposed_entities import (  # noqa: PLC0415
        async_should_expose,
    )

    return async_should_expose(hass, _ASSISTANT, entity_id)


@dataclass
class _SuggestionIndex:
    """Exposed entities per domain, built once per call and only when needed."""

    hass: HomeAssistant
    lookups_left: int = _MAX_SUGGESTIONS
    _domains: dict[str, list[tuple[str, str, str]]] = field(default_factory=dict)
    _all: list[tuple[str, str]] | None = None
    _exposed: dict[str, bool] = field(default_factory=dict)

    def _is_exposed(self, entity_id: str) -> bool:
        """Memoised exposure lookup: HA writes an option on a first ask."""
        if entity_id not in self._exposed:
            self._exposed[entity_id] = _exposed(self.hass, entity_id)
        return self._exposed[entity_id]

    def _domain(self, domain: str) -> list[tuple[str, str, str]]:
        """Return ``(entity_id, object_id, slug-of-name)`` per exposed entity."""
        if domain not in self._domains:
            self._domains[domain] = [
                (state.entity_id, state.object_id, slugify(state.name))
                for state in self.hass.states.async_all(domain)
                if self._is_exposed(state.entity_id)
            ]
        return self._domains[domain]

    def _everywhere(self) -> list[tuple[str, str]]:
        """Return ``(entity_id, slug-of-name)`` for every exposed entity."""
        if self._all is None:
            self._all = [
                (state.entity_id, slugify(state.name))
                for state in self.hass.states.async_all()
                if self._is_exposed(state.entity_id)
            ]
        return self._all

    def suggest(self, entity_id: str) -> list[str] | None:
        """
        Return the exposed entities the model most likely meant, best first.

        The model builds an id by slugifying the friendly name it was shown,
        so every entity whose name slugifies to the guessed object id is an
        exact answer (several, when the name is not unique), first in the
        guessed domain and then anywhere, since ``sensor.`` for a
        ``binary_sensor.`` is the other common slip; otherwise the closest
        ids among same-domain entities that share a word with the guess.
        Returns None once the per-call budget is spent, which is not the
        same as finding nothing.
        """
        if self.lookups_left <= 0 or not _exposure_available(self.hass):
            return None
        self.lookups_left -= 1

        domain, _, object_id = entity_id.partition(".")
        entries = self._domain(domain)
        exact = [eid for eid, _oid, slug in entries if slug == object_id]
        if not exact:
            exact = [eid for eid, slug in self._everywhere() if slug == object_id]
        if exact:
            return exact[:_MAX_SAME_NAME]

        tokens = {token for token in object_id.split("_") if token}
        if not tokens:
            return []
        by_key = {
            key: eid
            for eid, oid, slug in entries
            for key in (oid, slug)
            if tokens & set(key.split("_"))
        }
        close = difflib.get_close_matches(
            object_id, list(by_key), n=1, cutoff=_FUZZY_CUTOFF
        )
        return [by_key[close[0]]] if close else []


def _describe_suggestions(hass: HomeAssistant, suggestions: Sequence[str]) -> str:
    """Render suggested entities as ``'id' (Name)`` pairs."""
    parts: list[str] = []
    for entity_id in suggestions:
        state = hass.states.get(entity_id)
        parts.append(f"'{entity_id}' ({_label(state)})" if state else f"'{entity_id}'")
    return ", ".join(parts)


def _check_entity(
    hass: HomeAssistant,
    entity_id: str,
    index: _SuggestionIndex,
    notify_service: str | None,
) -> MissingAutomationTarget | None:
    """Return a finding if ``entity_id`` is unknown or disabled."""
    if hass.states.get(entity_id) is not None:
        return None
    entry = er.async_get(hass).async_get(entity_id)
    if entry is not None:
        if entry.disabled_by is None:
            # Registered but not loaded right now (integration starting or
            # unavailable): the automation will work once it is.
            return None
        return MissingAutomationTarget(
            kind=KIND_ENTITY,
            name=entity_id,
            detail=(
                "exists but is disabled in Home Assistant, so the automation "
                "would never see it. Enable the entity first or use another one."
            ),
        )
    domain, _, object_id = entity_id.partition(".")
    if hass.services.has_service(domain, object_id):
        # `notify.send_message` aimed at `notify.mobile_app_x`: the string the
        # model wrote is a legacy notify service, not a notify entity.
        return MissingAutomationTarget(
            kind=KIND_ENTITY,
            name=entity_id,
            detail=(
                f"is a service, not an entity. Call 'action: {entity_id}' "
                "directly instead of targeting it as an entity."
            ),
        )
    suggestions = index.suggest(entity_id)
    if domain == "notify" and not suggestions:
        # `notify.mobile_app` written as an entity: the guess this check
        # exists for, in its other spelling. Unless an exposed notify entity
        # matches, the push hint is the answer.
        return MissingAutomationTarget(
            kind=KIND_ENTITY,
            name=entity_id,
            detail=(
                "does not exist, and mobile push targets are services, not "
                f"entities. {_suggest_notify_service(hass, notify_service)}"
            ),
        )
    if suggestions is None:
        detail = (
            "does not exist. A lookup for a similar name was not made for this "
            "one; correct any entities listed above first, then call the tool "
            "again."
        )
    elif not suggestions:
        detail = (
            "does not exist, and nothing exposed to Assist has that name. "
            "Entity IDs are not the names shown in the home overview; tell "
            "the user which device you could not resolve and ask for its "
            "exact entity ID."
        )
    elif len(suggestions) == 1:
        detail = (
            f"does not exist. Did you mean {_describe_suggestions(hass, suggestions)}?"
        )
    else:
        detail = (
            "does not exist, and several entities have that name. Did you mean "
            f"one of {_describe_suggestions(hass, suggestions)}? Ask the user "
            "which one if the area does not settle it."
        )
    if suggestions and any(
        suggestion.partition(".")[0] != domain for suggestion in suggestions
    ):
        detail += (
            " That is a different domain, so the trigger's to/from states or "
            "threshold must match what that entity reports."
        )
    return MissingAutomationTarget(kind=KIND_ENTITY, name=entity_id, detail=detail)


def normalize_notify_service(service: str) -> str:
    """
    Return a configured push service as ``notify.<name>``.

    The option is stored either way (the flow falls back to free text when no
    mobile app is set up), and the rest of the integration tolerates the bare
    form, so this check must too.
    """
    lowered = service.strip().lower()
    return lowered if "." in lowered else f"notify.{lowered}"


def service_exists(hass: HomeAssistant, service: str) -> bool:
    """Return True if a ``domain.service`` string names a registered service."""
    domain, _, name = service.strip().lower().partition(".")
    return bool(domain and name and hass.services.has_service(domain, name))


def _suggest_service(
    hass: HomeAssistant, domain: str, notify_service: str | None
) -> str:
    """Return the sentence that tells the model which service to use instead."""
    if domain == "notify":
        return _suggest_notify_service(hass, notify_service)
    offered = sorted(hass.services.async_services_for_domain(domain))
    if domain in _INDIRECTION_DOMAINS:
        # These domains run user-defined commands and scripts; their names are
        # not for a model to browse, and the PIN screen gates them anyway. HA's
        # own generic services on script and automation are not user names, so
        # those may be; a `shell_command.turn_on` is still the user's.
        if domain not in _GENERIC_SERVICE_DOMAINS:
            return ""
        offered = [name for name in offered if name in _GENERIC_SERVICES]
        if not offered:
            return ""
    if not offered:
        return f"There is no '{domain}' service domain in Home Assistant."
    shown = ", ".join(f"{domain}.{name}" for name in offered[:_MAX_SERVICE_SUGGESTIONS])
    more = "" if len(offered) <= _MAX_SERVICE_SUGGESTIONS else ", …"
    return f"Services available in that domain: {shown}{more}."


def _suggest_notify_service(hass: HomeAssistant, notify_service: str | None) -> str:
    """
    Return the hint for a missing ``notify.*`` service.

    The configured push service comes first, but never alone: a guess at a
    second household member's phone must not be steered to the first one.
    """
    configured = (
        sanitize_tool_text(normalize_notify_service(notify_service))
        if notify_service
        else ""
    )
    mobile = list_mobile_notify_services(hass)
    others = [service for service in mobile if service != configured]
    if configured and service_exists(hass, configured):
        hint = (
            f"Use '{configured}', the mobile push service configured for this "
            "integration."
        )
        if others:
            hint += f" Other mobile push services: {', '.join(others)}."
        return hint
    if configured and others:
        return (
            f"The configured mobile push service '{configured}' no longer exists. "
            f"Mobile push services available: {', '.join(others)}."
        )
    if others:
        return f"Mobile push services available: {', '.join(others)}."
    offered = sorted(hass.services.async_services_for_domain("notify"))
    hint = "No mobile push service is set up in Home Assistant."
    if configured:
        hint = (
            f"The configured mobile push service '{configured}' no longer exists "
            "and no other mobile push service is set up in Home Assistant."
        )
    if offered:
        shown = ", ".join(
            f"notify.{name}" for name in offered[:_MAX_SERVICE_SUGGESTIONS]
        )
        more = "" if len(offered) <= _MAX_SERVICE_SUGGESTIONS else ", …"
        hint += f" Services available in that domain: {shown}{more}."
    return hint


def _step_service(step: Mapping[str, Any]) -> str | None:
    """Return the concrete ``domain.service`` a call_service step names."""
    # HA keeps a literal `service_template:` as written, so a plain service
    # name under that key is still a name that has to exist.
    literal = step.get("service_template")
    if isinstance(literal, str) and not _is_template(literal):
        lowered = literal.strip().lower()
        return lowered if _ENTITY_ID_RE.fullmatch(lowered) else None
    domain, service, templated = _service_name(step)
    if templated or not domain or not service:
        return None
    return f"{domain}.{service}"


def _walk_services(actions: Any) -> Iterator[str]:
    """Yield every concrete ``domain.service`` a call_service step names."""
    for step, action_type in _walk_steps(actions):
        if action_type != cv.SCRIPT_ACTION_CALL_SERVICE:
            continue
        service = _step_service(step)
        if service is not None:
            yield service


# Automation sections in the order the model wrote them, so findings read
# trigger first. Both the validated (plural) and raw (singular) keys count.
_SECTION_KEYS = (
    "triggers",
    "trigger",
    "conditions",
    "condition",
    "actions",
    "action",
)
_ACTION_KEYS = frozenset({"actions", "action"})


def _ordered_sections(config: Mapping[str, Any], *, actions: bool = True) -> list[Any]:
    """Return the config's sections, triggers first, then everything else."""
    ordered = [
        config[key]
        for key in _SECTION_KEYS
        if key in config and (actions or key not in _ACTION_KEYS)
    ]
    ordered.extend(
        value
        for key, value in config.items()
        if key not in _SECTION_KEYS and key not in _OPAQUE_KEYS
    )
    return ordered


def _too_deep() -> list[MissingAutomationTarget]:
    """Return the single finding for a config nested past the depth cap."""
    return [
        MissingAutomationTarget(
            kind=KIND_AUTOMATION,
            name="",
            detail="nests steps too deeply to check. Flatten it and try again.",
        )
    ]


def find_missing_automation_targets(
    hass: HomeAssistant,
    validated_config: Mapping[str, Any],
    *,
    notify_service: str | None = None,
) -> list[MissingAutomationTarget]:
    """
    Return the entities and services in an automation that do not exist.

    ``validated_config`` is the config returned by Home Assistant's
    ``_async_validate_config_item`` (blueprints substituted, ``service:``
    normalized to ``action:``, ``entity_id`` values normalized to lists);
    pre-validation spellings are accepted too. ``notify_service`` is the
    mobile push service configured for this integration, offered first when
    a ``notify.*`` service is missing.
    """
    found: list[MissingAutomationTarget] = []
    index = _SuggestionIndex(hass)
    actions = validated_config.get("actions")
    if actions is None:
        actions = validated_config.get("action")

    # An entity a step creates is a fine target for a later step, but a
    # trigger, a condition, or an earlier step has to find it already there.
    seen_entities: set[str] = set()

    def check(section: Any, exempt: set[str]) -> None:
        for entity_id in _walk_entity_refs(section):
            if entity_id in seen_entities or entity_id in exempt:
                continue
            seen_entities.add(entity_id)
            finding = _check_entity(hass, entity_id, index, notify_service)
            if finding is not None:
                found.append(finding)

    try:
        check(_ordered_sections(validated_config, actions=False), set())
        created = _own_automation_entity(validated_config)
        steps = actions if isinstance(actions, (list, tuple)) else [actions]
        for step in steps:
            check(step, created)
            created |= _created_by(step)
    except _TooDeepError:
        return _too_deep()

    seen_services: set[str] = set()
    try:
        services = list(_walk_services(_without_disabled(actions)))
    except _TooDeepError:
        return _too_deep()
    for service in services:
        if service in seen_services or service_exists(hass, service):
            continue
        domain, _, name = service.partition(".")
        if domain in _INDIRECTION_DOMAINS and not (
            domain in _GENERIC_SERVICE_DOMAINS and name in _GENERIC_SERVICES
        ):
            # A user's own command or script name is not this check's to
            # confirm or deny: reporting only the missing ones would reveal
            # the real ones by omission, before the PIN gate that guards
            # every call on these domains. Home Assistant reports the failed
            # call itself when it runs.
            continue
        seen_services.add(service)
        hint = _suggest_service(hass, domain, notify_service)
        found.append(
            MissingAutomationTarget(
                kind=KIND_SERVICE,
                name=service,
                detail="is not a Home Assistant service."
                + (f" {hint}" if hint else ""),
            )
        )
    return found


def describe_missing_targets(missing: Sequence[MissingAutomationTarget]) -> str:
    """Return the tool response for an automation that names missing targets."""
    shown = missing[:_MAX_REPORTED]
    lines = [f"- {item.describe()}" for item in shown]
    if len(missing) > len(shown):
        lines.append(f"- … and {len(missing) - len(shown)} more.")
    return (
        f"{AUTOMATION_REFUSAL_PREFIX}: it refers to entities or services that do "
        "not exist in Home Assistant, so it would never run. Correct these and "
        "call the tool again with the full automation:\n" + "\n".join(lines)
    )
