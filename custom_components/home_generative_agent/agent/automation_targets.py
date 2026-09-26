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

This module walks the validated config for every concrete ``entity_id`` and
every ``call_service`` step and reports the ones Home Assistant does not know,
each with the closest real thing so the model's retry is deterministic:

* an entity whose friendly name slugifies to the guessed object id, or
  failing that the closest entity id in the same domain;
* for a missing ``notify.*`` service, the mobile push service configured for
  this integration, then every ``notify.mobile_app_*`` service;
* for any other missing service, the services its domain does offer.

Templates, entity registry IDs, ``entity_id: all`` and device actions are not
resolvable here and are left alone; this is a check for the invented name,
not a second validator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from homeassistant.const import ATTR_FRIENDLY_NAME
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import entity_registry as er
from homeassistant.util import slugify

from ..core.conversation_helpers import _resolve_entity_id  # noqa: TID252
from ..core.utils import list_mobile_notify_services  # noqa: TID252
from .automation_pin import _service_name, _walk_steps

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from homeassistant.core import HomeAssistant

# A concrete entity_id. Anything else (registry UUID, template, `all`) names a
# target Home Assistant resolves at run time, which this check cannot judge.
_ENTITY_ID_RE = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")

# Values of `entity_id` that are keywords rather than entities.
_ENTITY_ID_KEYWORDS = frozenset({"all", "none"})

# Depth cap for the config walk; mirrors the step walker's.
_MAX_DEPTH = 50

# How many of a domain's real services to offer for a missing one.
_MAX_SERVICE_SUGGESTIONS = 8


@dataclass(frozen=True)
class MissingAutomationTarget:
    """An entity or service the automation names that Home Assistant lacks."""

    kind: str  # "entity" or "service"
    name: str
    detail: str

    def describe(self) -> str:
        """Return one line the model can act on."""
        return f"{self.kind.capitalize()} '{self.name}' {self.detail}"


def _walk_entity_ids(node: Any, depth: int = 0) -> Iterator[str]:
    """Yield every concrete entity_id string under any ``entity_id`` key."""
    if depth > _MAX_DEPTH:
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk_entity_ids(item, depth + 1)
        return
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if key == "entity_id":
            candidates = value if isinstance(value, (list, tuple)) else [value]
            for candidate in candidates:
                if not isinstance(candidate, str):
                    continue
                lowered = candidate.strip().lower()
                if lowered in _ENTITY_ID_KEYWORDS or not _ENTITY_ID_RE.fullmatch(
                    lowered
                ):
                    continue
                yield lowered
        else:
            yield from _walk_entity_ids(value, depth + 1)


def _friendly_name(hass: HomeAssistant, entity_id: str) -> str:
    """Return the friendly name of an entity, or its id when it has none."""
    state = hass.states.get(entity_id)
    if state is None:
        return entity_id
    name = state.attributes.get(ATTR_FRIENDLY_NAME)
    return name if isinstance(name, str) and name.strip() else entity_id


def _suggest_entity(hass: HomeAssistant, entity_id: str) -> str | None:
    """
    Return the real entity the model most likely meant, or None.

    The model builds an id by slugifying the friendly name it was shown, so an
    entity whose friendly name slugifies to the guessed object id is the exact
    answer; the token-overlap resolver is the fallback for partial guesses.
    """
    domain, _, object_id = entity_id.partition(".")
    for state in hass.states.async_all(domain):
        name = state.attributes.get(ATTR_FRIENDLY_NAME)
        if isinstance(name, str) and slugify(name) == object_id:
            return state.entity_id
    resolved = _resolve_entity_id(entity_id, hass)
    return resolved if resolved != entity_id else None


def _check_entity(
    hass: HomeAssistant, entity_id: str
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
            kind="entity",
            name=entity_id,
            detail=(
                "exists but is disabled in Home Assistant, so the automation "
                "would never see it. Enable the entity first or use another one."
            ),
        )
    suggestion = _suggest_entity(hass, entity_id)
    if suggestion is None:
        detail = (
            "does not exist. Entity IDs are not the names shown in the home "
            "overview; use an entity ID you have seen in a tool result."
        )
    else:
        detail = (
            f"does not exist. Did you mean '{suggestion}' "
            f"({_friendly_name(hass, suggestion)})?"
        )
    return MissingAutomationTarget(kind="entity", name=entity_id, detail=detail)


def _suggest_service(
    hass: HomeAssistant, domain: str, notify_service: str | None
) -> str:
    """Return the sentence that tells the model which service to use instead."""
    if domain == "notify":
        configured = (notify_service or "").strip()
        if configured and _service_exists(hass, configured):
            return (
                f"Use '{configured}', the mobile push service configured for "
                "this integration."
            )
        mobile = list_mobile_notify_services(hass)
        if mobile:
            return f"Mobile push services available: {', '.join(mobile)}."
        return "No mobile push service is set up in Home Assistant."
    offered = sorted(hass.services.async_services_for_domain(domain))
    if not offered:
        return f"There is no '{domain}' service domain in Home Assistant."
    shown = ", ".join(f"{domain}.{name}" for name in offered[:_MAX_SERVICE_SUGGESTIONS])
    more = "" if len(offered) <= _MAX_SERVICE_SUGGESTIONS else ", …"
    return f"Services available in that domain: {shown}{more}."


def _service_exists(hass: HomeAssistant, service: str) -> bool:
    """Return True if a ``domain.service`` string names a registered service."""
    domain, _, name = service.strip().lower().partition(".")
    return bool(domain and name and hass.services.has_service(domain, name))


def _walk_services(actions: Any) -> Iterator[str]:
    """Yield every concrete ``domain.service`` a call_service step names."""
    for step, action_type in _walk_steps(actions):
        if action_type != cv.SCRIPT_ACTION_CALL_SERVICE:
            continue
        domain, service, templated = _service_name(step)
        if templated or not domain or not service:
            continue
        yield f"{domain}.{service}"


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


def _ordered_sections(config: Mapping[str, Any]) -> list[Any]:
    """Return the config's sections, triggers first, then everything else."""
    ordered = [config[key] for key in _SECTION_KEYS if key in config]
    ordered.extend(value for key, value in config.items() if key not in _SECTION_KEYS)
    return ordered


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
    seen: set[str] = set()

    for entity_id in _walk_entity_ids(_ordered_sections(validated_config)):
        if entity_id in seen:
            continue
        seen.add(entity_id)
        finding = _check_entity(hass, entity_id)
        if finding is not None:
            found.append(finding)

    actions = validated_config.get("actions")
    if actions is None:
        actions = validated_config.get("action")
    for service in _walk_services(actions):
        if service in seen or _service_exists(hass, service):
            continue
        seen.add(service)
        domain = service.partition(".")[0]
        found.append(
            MissingAutomationTarget(
                kind="service",
                name=service,
                detail=(
                    "is not a Home Assistant service. "
                    + _suggest_service(hass, domain, notify_service)
                ),
            )
        )
    return found


def describe_missing_targets(missing: Sequence[MissingAutomationTarget]) -> str:
    """Return the tool response for an automation that names missing targets."""
    lines = "\n".join(f"- {item.describe()}" for item in missing)
    return (
        "Automation not added: it refers to entities or services that do not "
        "exist in Home Assistant, so it would never run. Correct these and call "
        f"the tool again with the full automation:\n{lines}"
    )
