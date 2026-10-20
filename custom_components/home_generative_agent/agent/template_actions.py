"""
Read the actions an entity runs when the agent acts on it (#780).

Some entities run stored actions of their own, which no rule names:

* **A template entity** (from the Template integration) runs action scripts.
  For a template switch, button, lock and cover, Home Assistant keys each
  script by the service it answers, so a call runs the script with its
  service's name: a template switch's ``turn_on``, a template button's
  ``press``, a template lock's ``unlock``. A call with no script of its own
  (``toggle``, or a cover opened through its position script) may run any of
  them. Any other template entity picks its script from the call's data (a
  template light's ``turn_on`` with a brightness runs ``set_level``; a fan's
  runs ``set_percentage`` too), so every script it has counts.
* **A button or an input button** starts the automations that react to its
  press. An automation counts when a trigger names the button, by entity id
  or registry id, or when a trigger cannot be tied to entities ahead (an
  event or template trigger, a templated value, an area, device, label or
  floor target): a ``state_changed`` event filtered in its conditions can
  react to the press. Its actions are what the press runs.

``indirect_pin`` screens these steps like a script's; ``rule_entities`` reads
the entities a guarded template cover or lock drives. What cannot be read is
returned as None, and held by the callers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.helpers import entity_registry as er

from .automation_pin import iter_service_calls, step_targets

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

TEMPLATE_PLATFORM = "template"
INPUT_BUTTON_DOMAIN = "input_button"
BUTTON_DOMAIN = "button"
PRESS_DOMAINS = frozenset({BUTTON_DOMAIN, INPUT_BUTTON_DOMAIN})
SCRIPT_DOMAIN = "script"
_AUTOMATION_DOMAIN = "automation"
_TRIGGER_KEYS = ("triggers", "trigger")
_TRIGGER_PLATFORM_KEYS = ("trigger", "platform")
# Triggers that react to events or templates rather than to named entities.
_UNATTRIBUTABLE_TRIGGERS = frozenset({"event", "template"})
_TARGET_KEYS_NOT_ENTITIES = ("area_id", "device_id", "label_id", "floor_id")
# Template domains whose action scripts are keyed by the service they answer.
_SCRIPT_PER_SERVICE_DOMAINS = frozenset({"switch", "button", "lock", "cover"})
# Scripts and template entities a template entity's actions reach, and the
# members of a group, are followed this deep.
MAX_FOLLOW_DEPTH = 5
# A group lists its members in `entity_id`; MQTT and ZHA groups in
# `group_entities`.
_MEMBER_ATTRIBUTES = ("entity_id", "group_entities")


def _loaded_entity(hass: HomeAssistant, entity_id: str) -> Any:
    """Return the loaded entity object, or None."""
    component = hass.data.get(entity_id.split(".", 1)[0])
    get_entity = getattr(component, "get_entity", None)
    return get_entity(entity_id) if callable(get_entity) else None


def _platform(hass: HomeAssistant, entity_id: str, entity: Any) -> str | None:
    """Return the integration an entity comes from, loaded or only registered."""
    platform = getattr(getattr(entity, "platform", None), "platform_name", None)
    if isinstance(platform, str):
        return platform
    entry = er.async_get(hass).async_get(entity_id)
    return entry.platform if entry is not None else None


def is_template_entity(hass: HomeAssistant, entity_id: str) -> bool:
    """Return True for an entity of the Template integration."""
    entity_id = entity_id.lower()
    return (
        _platform(hass, entity_id, _loaded_entity(hass, entity_id)) == TEMPLATE_PLATFORM
    )


def runs_stored_actions(hass: HomeAssistant, entity_id: str) -> bool:
    """Return True for a template entity, a button or an input button."""
    entity_id = entity_id.lower()
    if entity_id.split(".", 1)[0] in PRESS_DOMAINS:
        return True
    return is_template_entity(hass, entity_id)


def action_scripts(entity: Any, entity_id: str) -> dict[str, Any] | None:
    """
    Return a loaded template entity's action scripts by key, or None.

    The only reader of the Template integration's private ``_action_scripts``:
    if Home Assistant renames it, every caller holds rather than passes, and
    this logs once per call.
    """
    scripts = getattr(entity, "_action_scripts", None)
    if not isinstance(scripts, dict):
        LOGGER.warning(
            "Critical Action PIN: cannot read the actions of %s; holding it",
            entity_id,
        )
        return None
    return scripts


def _sequences(scripts: Mapping[str, Any], keys: Iterable[str]) -> list[Any] | None:
    """Return the step lists of the scripts with these keys, or None."""
    sequences = [getattr(scripts[key], "sequence", None) for key in keys]
    if any(not isinstance(sequence, list) for sequence in sequences):
        return None
    return sequences


def stored_action_sequences(
    hass: HomeAssistant, entity_id: str, service: str | None
) -> list[Any] | None:
    """
    Return the step lists acting on the entity with ``service`` runs.

    See the module docstring. None when they cannot be read: a template
    entity that is not loaded, or an automation whose config or actions
    cannot be read.
    """
    entity_id = entity_id.lower()
    domain = entity_id.split(".", 1)[0]
    sequences: list[Any] = []
    if domain in PRESS_DOMAINS:
        started = _automations_started_by(hass, entity_id)
        if started is None:
            return None
        sequences.extend(started)
        if not is_template_entity(hass, entity_id):
            return sequences
    entity = _loaded_entity(hass, entity_id)
    scripts = action_scripts(entity, entity_id) if entity is not None else None
    if scripts is None:
        LOGGER.debug("Cannot read the actions of %s; holding it", entity_id)
        return None
    keys: list[str] = (
        [service]
        if domain in _SCRIPT_PER_SERVICE_DOMAINS and service in scripts
        else list(scripts)
    )
    own = _sequences(scripts, keys)
    if own is None:
        LOGGER.debug("Cannot read the actions of %s; holding it", entity_id)
        return None
    return sequences + own


def _texts(value: Any) -> Iterable[str]:
    """Yield every text in a config value, keys included, lowercased."""
    if isinstance(value, str):
        yield value.lower()
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _texts(key)
            yield from _texts(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _texts(item)
    else:
        # A validated template keeps its text here.
        text = getattr(value, "template", None)
        if isinstance(text, str):
            yield text.lower()


def _trigger_may_react(trigger: Any, names: Iterable[str]) -> bool:
    """Return True if the trigger names the entity or cannot be tied down."""
    if not isinstance(trigger, dict):
        return True
    texts = list(_texts(trigger))
    if any(name in text for text in texts for name in names):
        return True
    if any("{{" in text or "{%" in text for text in texts):
        return True  # a templated entity, event filter or variable
    platform = next(
        (trigger[key] for key in _TRIGGER_PLATFORM_KEYS if key in trigger), None
    )
    if not isinstance(platform, str) or platform.lower() in _UNATTRIBUTABLE_TRIGGERS:
        return True
    target = trigger.get("target")
    if target is not None and (
        not isinstance(target, dict)
        or any(target.get(key) for key in _TARGET_KEYS_NOT_ENTITIES)
    ):
        return True
    # A device trigger names its device, not the entity, when it has no
    # entity_id: it may be the button's.
    return platform == "device" and not trigger.get("entity_id")


def _automations_started_by(hass: HomeAssistant, entity_id: str) -> list[Any] | None:
    """Return the actions of each automation that may react to the entity."""
    component = hass.data.get(_AUTOMATION_DOMAIN)
    if component is None:
        return []
    names = [entity_id]
    entry = er.async_get(hass).async_get(entity_id)
    if entry is not None:
        names.append(entry.id.lower())
    sequences: list[Any] = []
    for automation in getattr(component, "entities", ()):
        script = getattr(automation, "action_script", None)
        if script is None:
            continue  # an automation that failed to set up does not run
        config = getattr(automation, "raw_config", None)
        if not isinstance(config, dict):
            LOGGER.debug("Cannot read %s; holding %s", automation, entity_id)
            return None
        triggers = next((config[key] for key in _TRIGGER_KEYS if key in config), [])
        if not isinstance(triggers, list):
            triggers = [triggers]
        if not any(_trigger_may_react(trigger, names) for trigger in triggers):
            continue
        sequence = getattr(script, "sequence", None)
        if not isinstance(sequence, list):
            LOGGER.debug("Cannot read %s; holding %s", automation, entity_id)
            return None
        sequences.append(sequence)
    return sequences


def entities_of(hass: HomeAssistant, domain: str) -> list[str]:
    """Return the entity ids of a domain that run stored actions."""
    if domain in PRESS_DOMAINS:
        return [state.entity_id for state in hass.states.async_all(domain)]
    return [
        str(entity.entity_id)
        for entity in template_entities(hass, domain)
        if getattr(entity, "entity_id", None)
    ]


def template_entities(hass: HomeAssistant, domain: str) -> list[Any]:
    """Return the loaded template entities of a domain."""
    component = hass.data.get(domain)
    return [
        entity
        for entity in getattr(component, "entities", ())
        if getattr(getattr(entity, "platform", None), "platform_name", None)
        == TEMPLATE_PLATFORM
    ]


def call_targets(
    hass: HomeAssistant, domain: str, service: str, step: Mapping[str, Any]
) -> tuple[set[str], bool]:
    """
    Return the entities a call acts on, and whether some could not be named.

    Registry ids are resolved, ``script.<key>`` names its script, and a group
    target adds its members. An area, device, label, floor, ``all`` or a
    template stays unnamed.
    """
    registry = er.async_get(hass)
    if domain == SCRIPT_DOMAIN and service not in {"turn_on", "toggle", "turn_off"}:
        script_id = registry.async_get_entity_id(SCRIPT_DOMAIN, SCRIPT_DOMAIN, service)
        return {script_id or f"{SCRIPT_DOMAIN}.{service}"}, False
    entities, unresolved = step_targets(step, frozenset({SCRIPT_DOMAIN, "scene"}))
    named = {entity.lower() for entity in entities}
    if unresolved:
        resolved, unresolved = _resolve_registry_ids(registry, step)
        named |= resolved
    return _with_members(hass, named), unresolved


def _resolve_registry_ids(
    registry: er.EntityRegistry, step: Mapping[str, Any]
) -> tuple[set[str], bool]:
    """Resolve the registry ids among a step's targets; flag anything else."""
    containers = [step, *(step.get(key) for key in ("target", "data"))]
    named: set[str] = set()
    for container in containers:
        if container is None:
            continue
        if not isinstance(container, dict):
            return named, True
        if any(container.get(key) for key in _TARGET_KEYS_NOT_ENTITIES):
            return named, True
        raw = container.get("entity_id")
        values = raw if isinstance(raw, list) else [raw] if raw is not None else []
        for value in values:
            if not isinstance(value, str) or "{" in value:
                return named, True
            resolved = er.async_resolve_entity_id(registry, value)
            if resolved is None:
                return named, True  # `all`, `none`, or an unknown id
            named.add(resolved.lower())
    return named, not named


def _with_members(hass: HomeAssistant, entities: set[str], depth: int = 0) -> set[str]:
    """Return the entities and, for each group among them, its members."""
    found = set(entities)
    if depth >= MAX_FOLLOW_DEPTH:
        return found
    for entity_id in entities:
        state = hass.states.get(entity_id)
        if state is None or entity_id.split(".", 1)[0] in {SCRIPT_DOMAIN, "scene"}:
            continue
        members: set[str] = set()
        for attribute in _MEMBER_ATTRIBUTES:
            value = state.attributes.get(attribute)
            if isinstance(value, (list, tuple)):
                members |= {str(member).lower() for member in value}
        members -= found
        if members:
            found |= _with_members(hass, members, depth + 1)
    return found


def driven_entities(
    hass: HomeAssistant, sequences: Iterable[Any], *, depth: int = 0
) -> set[str]:
    """
    Return the entities the steps act on, by entity id.

    A script or template entity among them is followed into its own steps.
    Targets that cannot be named (an area, a template) are left out: the
    caller holds what it can.
    """
    found: set[str] = set()
    for sequence in sequences:
        for domain, service, step in iter_service_calls(sequence):
            entities, _unresolved = call_targets(hass, domain, service, step)
            found |= entities
    if depth < MAX_FOLLOW_DEPTH:
        for entity_id in sorted(found):
            nested = _own_sequences(hass, entity_id)
            if nested:
                found |= driven_entities(hass, nested, depth=depth + 1)
    return found


def _own_sequences(hass: HomeAssistant, entity_id: str) -> list[Any]:
    """Return the steps a script or template entity runs, when readable."""
    entity = _loaded_entity(hass, entity_id)
    if entity_id.startswith(f"{SCRIPT_DOMAIN}."):
        sequence = getattr(getattr(entity, "script", None), "sequence", None)
        return [sequence] if isinstance(sequence, list) else []
    if entity is None or not is_template_entity(hass, entity_id):
        return []
    scripts = action_scripts(entity, entity_id)
    return (_sequences(scripts, list(scripts)) or []) if scripts else []
