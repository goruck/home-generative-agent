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
* **A button or an input button** starts what reacts to its press: an
  automation, a trigger-based template block with ``actions:``, and a script
  or automation waiting for it (``wait_for_trigger`` or ``wait_template``),
  which the press resumes. One counts when a trigger or wait names the
  button, by entity id or registry id (or through the automation's
  variables), or cannot be tied to entities ahead: a ``state_changed`` or
  ``call_service`` event trigger (one filtered in its conditions can react
  to the press), an event trigger whose type is templated or missing, a
  template reading states generically (``states.input_button``,
  ``expand(...)``, ``area_entities(...)``), a templated entity, target or
  event filter, an area, device, label or floor target, or a device trigger
  for buttons or for the button's own device. Its actions are what the
  press runs. A template naming other entities (``states('sensor.x')``),
  another event type (a ZHA or mobile notification event) or another
  device's trigger (a remote) cannot react to a press and does not count.

A call's targets are read the way Home Assistant reads them: the step, its
``target`` and its data blocks are merged, registry ids are resolved, an
area, device, floor or label is expanded with Home Assistant's own target
resolver, and a group adds its members. Only a templated target or ``all``
stays unnamed.

``indirect_pin`` screens these steps like a script's; ``rule_entities`` reads
the entities a guarded template cover or lock drives. What cannot be read is
returned as None, and held by the callers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.target import (
    TargetSelection,
    async_extract_referenced_entity_ids,
)

from .automation_pin import DATA_CONTAINER_KEYS, TooDeepError, iter_service_calls

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

TEMPLATE_PLATFORM = "template"
INPUT_BUTTON_DOMAIN = "input_button"
BUTTON_DOMAIN = "button"
PRESS_DOMAINS = frozenset({BUTTON_DOMAIN, INPUT_BUTTON_DOMAIN})
SCRIPT_DOMAIN = "script"
SCENE_DOMAIN = "scene"
_GENERIC_DOMAIN = "homeassistant"
_AUTOMATION_DOMAIN = "automation"
_TRIGGER_KEYS = ("triggers", "trigger")
_ACTION_KEYS = ("actions", "action")
_TRIGGER_PLATFORM_KEYS = ("trigger", "platform")
_WAIT_KEY = "wait_for_trigger"
_WAIT_TEMPLATE_KEY = "wait_template"
_VARIABLE_KEYS = ("variables", "trigger_variables")
_TEMPLATE_DOMAIN = TEMPLATE_PLATFORM
# A template reading states without naming one entity: `states.input_button`,
# `states | ...`, `expand(...)`, `area_entities(...)` and the like.
_GENERIC_STATES = re.compile(
    r"\bstates\b(?!\s*\()(?!\.[a-z0-9_]+\.[a-z0-9_]+)"
    r"|\b(?:expand|area_entities|label_entities|device_entities"
    r"|floor_entities|integration_entities)\s*\("
)
# The events a button press can fire: its state change, and the service call.
_PRESS_EVENTS = frozenset({"state_changed", "call_service"})
# Trigger fields that name what the trigger watches.
_WATCHED_KEYS = ("entity_id", "target", "device_id", "event_data", "event_type")
_TARGET_KEYS_NOT_ENTITIES = ("area_id", "device_id", "label_id", "floor_id")
_TARGET_KEYS = ("entity_id", *_TARGET_KEYS_NOT_ENTITIES)
# An entity id, as opposed to a registry id, `all` or a template.
_ENTITY_ID = re.compile(r"[a-z0-9_]+\.[a-z0-9_]+")
# Template domains whose action scripts are keyed by the service they answer.
_SCRIPT_PER_SERVICE_DOMAINS = frozenset({"switch", "button", "lock", "cover"})
# Script and inline-scene services that run something other than their target.
_SCENE_INLINE_SERVICES = frozenset({"apply", "create"})
_INLINE_STATE_KEYS = ("entities", "snapshot_entities")
# Scripts and template entities a template entity's actions reach, and the
# members of a group, are followed this deep.
MAX_FOLLOW_DEPTH = 5
# A group lists its members in `entity_id`; MQTT and ZHA groups in
# `group_entities`.
_MEMBER_ATTRIBUTES = ("entity_id", "group_entities")
# Beyond the first step, what a template cover or lock drives is followed
# only into these domains: a relay, a lock, a cover, a button, another
# script. A script that also lights the house does not hold the lights.
_RELAY_DOMAINS = frozenset(
    {"switch", "cover", "valve", "lock", "button", "input_button", SCRIPT_DOMAIN}
)


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
    this logs.
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


@dataclass
class _Automation:
    """One automation, read once per check: what it watches and runs."""

    triggers: list[Any]
    waits: list[Any]
    sequence: Any
    variables: list[str] = field(default_factory=list)


@dataclass
class StoredActionReader:
    """
    Reads stored actions for one check, reading the automations only once.

    A button press scans every automation; an area of buttons would scan
    them once per button without this.
    """

    hass: HomeAssistant
    _automations: list[_Automation] | None = None
    _read: bool = False
    _unreadable: bool = False
    _started: dict[str, list[Any] | None] = field(default_factory=dict)

    def sequences(self, entity_id: str, service: str | None) -> list[Any] | None:
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
            started = self.started_by(entity_id)
            if started is None:
                return None
            sequences.extend(started)
            if not is_template_entity(self.hass, entity_id):
                return sequences
        entity = _loaded_entity(self.hass, entity_id)
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

    def started_by(self, entity_id: str) -> list[Any] | None:
        """Return the actions of each automation that may react to a press."""
        if entity_id not in self._started:
            self._started[entity_id] = self._find_started_by(entity_id)
        return self._started[entity_id]

    def _find_started_by(self, entity_id: str) -> list[Any] | None:
        automations = self._all_automations()
        if automations is None:
            return None
        names = [entity_id]
        entry = er.async_get(self.hass).async_get(entity_id)
        device_id = None
        if entry is not None:
            names.append(entry.id.lower())
            device_id = entry.device_id
        found: list[Any] = []
        for automation in automations:
            # A template reading one of the automation's variables that names
            # the button reads the button.
            named_by_variable = _names_in(automation.variables, names)
            if any(
                _trigger_may_react(
                    trigger,
                    names,
                    device_id=device_id,
                    named_by_variable=named_by_variable,
                )
                for trigger in (*automation.triggers, *automation.waits)
            ):
                found.append(automation.sequence)
        return found

    def _all_automations(self) -> list[_Automation] | None:
        if not self._read:
            self._read = True
            self._automations = _read_automations(self.hass)
        return self._automations


def _read_automations(hass: HomeAssistant) -> list[_Automation] | None:
    """
    Read what can react to a press: automations, scripts and template blocks.

    None if one cannot be read: a press is then held.
    """
    found: list[_Automation] = []
    for automation in getattr(hass.data.get(_AUTOMATION_DOMAIN), "entities", ()):
        script = getattr(automation, "action_script", None)
        if script is None:
            continue  # an automation that failed to set up does not run
        read = _reactor(
            getattr(automation, "raw_config", None),
            getattr(script, "sequence", None),
        )
        if read is None:
            LOGGER.debug("Cannot read %s; holding presses", automation)
            return None
        found.append(read)
    # A script runs only when started; one waiting for the press resumes.
    for script_entity in getattr(hass.data.get(SCRIPT_DOMAIN), "entities", ()):
        config = getattr(script_entity, "raw_config", None)
        sequence = getattr(getattr(script_entity, "script", None), "sequence", None)
        if not isinstance(config, dict) or not isinstance(sequence, list):
            LOGGER.debug("Cannot read %s; holding presses", script_entity)
            return None
        waits = list(_wait_triggers(config.get("sequence")))
        if waits:
            found.append(
                _Automation(
                    triggers=[],
                    waits=waits,
                    sequence=sequence,
                    variables=list(_texts(config.get("variables"))),
                )
            )
    # A trigger-based template block with `actions:` runs them when triggered.
    coordinators = hass.data.get(_TEMPLATE_DOMAIN)
    if coordinators is not None and not isinstance(coordinators, list):
        LOGGER.debug("Cannot read the template blocks; holding presses")
        return None
    for coordinator in coordinators or ():
        config = getattr(coordinator, "config", None)
        if not isinstance(config, dict):
            LOGGER.debug("Cannot read a template block; holding presses")
            return None
        actions = next((config[key] for key in _ACTION_KEYS if key in config), None)
        if actions is None:
            continue
        read = _reactor(config, actions)
        if read is None:
            return None
        found.append(read)
    return found


def _reactor(config: Any, sequence: Any) -> _Automation | None:
    """Read an automation or template block's triggers, waits and actions."""
    if not isinstance(config, dict) or not isinstance(sequence, list):
        return None
    triggers = next((config[key] for key in _TRIGGER_KEYS if key in config), [])
    if not isinstance(triggers, list):
        triggers = [triggers]
    actions = next((config[key] for key in _ACTION_KEYS if key in config), [])
    return _Automation(
        triggers=triggers,
        waits=list(_wait_triggers(actions)),
        sequence=sequence,
        variables=[text for key in _VARIABLE_KEYS for text in _texts(config.get(key))],
    )


def _wait_triggers(node: Any) -> Iterable[Any]:
    """Yield the triggers of every ``wait_for_trigger`` step, however nested."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == _WAIT_KEY:
                yield from value if isinstance(value, list) else [value]
            elif key == _WAIT_TEMPLATE_KEY:
                # Waits like a template trigger on the same text.
                yield {"trigger": "template", "value_template": value}
            else:
                yield from _wait_triggers(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _wait_triggers(item)


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


def _templated(texts: Iterable[str]) -> bool:
    return any("{{" in text or "{%" in text for text in texts)


def _names_in(texts: Iterable[str], names: Iterable[str]) -> bool:
    """Return True if a text names one of the entities (not a longer id)."""
    patterns = [
        re.compile(rf"(?<![a-z0-9_]){re.escape(name)}(?![a-z0-9_])") for name in names
    ]
    return any(pattern.search(text) for text in texts for pattern in patterns)


def _trigger_may_react(  # noqa: PLR0911
    trigger: Any,
    names: Iterable[str],
    *,
    device_id: str | None = None,
    named_by_variable: bool = False,
) -> bool:
    """Return True if the trigger names the entity or cannot be tied down."""
    if not isinstance(trigger, dict):
        return True
    names = list(names)
    texts = list(_texts(trigger))
    if _names_in(texts, names):
        return True
    if named_by_variable and _templated(texts):
        return True  # a template reading a variable that names the entity
    platform = next(
        (trigger[key] for key in _TRIGGER_PLATFORM_KEYS if key in trigger), None
    )
    if not isinstance(platform, str):
        return True
    platform = platform.lower()
    if platform == "event":
        types = list(_texts(trigger.get("event_type")))
        return not types or _templated(types) or bool(_PRESS_EVENTS & set(types))
    if platform == "template":
        return any(_GENERIC_STATES.search(text) for text in texts)
    if _templated(_texts({key: trigger.get(key) for key in _WATCHED_KEYS})):
        return True  # a templated entity, target or event filter
    target = trigger.get("target")
    if target is not None and (
        not isinstance(target, dict)
        or any(target.get(key) for key in _TARGET_KEYS_NOT_ENTITIES)
    ):
        return True
    if platform == "device" and not trigger.get("entity_id"):
        # A device trigger names its device, not the entity: it may be the
        # button's, or any button's. A remote's trigger is not a press.
        domain = str(trigger.get("domain") or "").lower()
        return domain in PRESS_DOMAINS or (
            device_id is not None and trigger.get("device_id") == device_id
        )
    return False


def entities_of(hass: HomeAssistant, domain: str) -> list[str]:
    """
    Return the entity ids of a domain that run stored actions.

    Template entities registered but not loaded count while the Template
    integration has not loaded: one that loads later runs its actions, so the
    caller holds them as unreadable.
    """
    if domain in PRESS_DOMAINS:
        return [state.entity_id for state in hass.states.async_all(domain)]
    found = {
        str(entity.entity_id).lower()
        for entity in template_entities(hass, domain)
        if getattr(entity, "entity_id", None)
    }
    if TEMPLATE_PLATFORM not in hass.config.components:
        # Before the Template integration has loaded, each registered entity
        # may still load and run its actions. Once it has, one still not
        # loaded is left over from removed config and cannot run.
        found |= {
            entry.entity_id
            for entry in er.async_get(hass).entities.values()
            if entry.domain == domain
            and entry.platform == TEMPLATE_PLATFORM
            and entry.disabled_by is None
        }
    return sorted(found)


def template_entities(hass: HomeAssistant, domain: str) -> list[Any]:
    """Return the loaded template entities of a domain."""
    component = hass.data.get(domain)
    return [
        entity
        for entity in getattr(component, "entities", ())
        if getattr(getattr(entity, "platform", None), "platform_name", None)
        == TEMPLATE_PLATFORM
    ]


def inline_states(step: Mapping[str, Any]) -> tuple[dict[str, str | None], bool]:
    """Return the states `scene.apply`/`scene.create` sets, and if readable."""
    fields: dict[str, Any] = {}
    for key in DATA_CONTAINER_KEYS:
        block = step.get(key)
        if block is None:
            continue
        if not isinstance(block, dict):
            return {}, False
        fields.update(block)
    states: dict[str, str | None] = {}
    for key in _INLINE_STATE_KEYS:
        block = fields.get(key)
        if isinstance(block, dict):
            for entity_id, value in block.items():
                state = value.get("state") if isinstance(value, dict) else value
                states[str(entity_id).lower()] = (
                    str(state).lower() if state is not None else None
                )
        elif isinstance(block, (list, tuple)):
            states.update({str(entity_id).lower(): None for entity_id in block})
        elif block is not None:
            return {}, False
    return states, True


def call_targets(
    hass: HomeAssistant, domain: str, service: str, step: Mapping[str, Any]
) -> tuple[set[str], bool]:
    """
    Return the entities a call acts on, and whether some could not be named.

    See the module docstring. ``script.<key>`` names its script, and an
    inline scene its states' entities.
    """
    registry = er.async_get(hass)
    if domain == SCRIPT_DOMAIN and service not in {"turn_on", "toggle", "turn_off"}:
        script_id = registry.async_get_entity_id(SCRIPT_DOMAIN, SCRIPT_DOMAIN, service)
        return {script_id or f"{SCRIPT_DOMAIN}.{service}"}, False
    if domain == SCENE_DOMAIN and service in _SCENE_INLINE_SERVICES:
        states, readable = inline_states(step)
        return set(states), not readable
    selection, unresolved = _merged_targets(registry, step)
    named = set(selection.get("entity_id", []))
    indirect = {
        key: selection[key] for key in _TARGET_KEYS_NOT_ENTITIES if key in selection
    }
    if indirect:
        try:
            selected = async_extract_referenced_entity_ids(
                hass,
                TargetSelection(indirect),
                expand_group=False,
                primary_entities_only=False,
            )
        except Exception:
            LOGGER.exception("Could not resolve the targets of %s.%s", domain, service)
            return named, True
        named |= {
            entity_id.lower()
            for entity_id in selected.referenced | selected.indirectly_referenced
            if domain == _GENERIC_DOMAIN or entity_id.split(".", 1)[0] == domain
        }
    return with_members(hass, named), unresolved


def _merged_targets(
    registry: er.EntityRegistry, step: Mapping[str, Any]
) -> tuple[dict[str, list[str]], bool]:
    """
    Merge a step's targets from the step, ``target`` and its data blocks.

    Home Assistant merges ``data_template`` and the other data blocks with
    ``target`` when the call runs, so a target given in any of them counts.
    Registry ids are resolved; ``all``, an unknown id or a template is
    unresolved.
    """
    merged: dict[str, list[str]] = {}
    unresolved = False
    for container in (
        step,
        *(step.get(key) for key in ("target", *DATA_CONTAINER_KEYS)),
    ):
        if container is None:
            continue
        if not isinstance(container, dict) or _templated(_texts(list(container))):
            unresolved = True
            continue
        for key in _TARGET_KEYS:
            raw = container.get(key)
            values = raw if isinstance(raw, (list, tuple)) else [raw]
            for value in values:
                if value is None:
                    continue
                if not isinstance(value, str) or _templated([value]):
                    unresolved = True
                    continue
                text = value.strip()
                if key != "entity_id":
                    merged.setdefault(key, []).append(text)
                    continue
                lowered = text.lower()
                if lowered == "none":
                    continue
                if _ENTITY_ID.fullmatch(lowered):
                    merged.setdefault(key, []).append(lowered)
                    continue
                try:
                    resolved = er.async_resolve_entity_id(registry, text)
                except Exception:
                    LOGGER.exception("Could not resolve the target %s", text)
                    resolved = None
                if resolved is None:
                    unresolved = True  # `all`, an unknown id
                    continue
                merged.setdefault(key, []).append(resolved.lower())
    return merged, unresolved


def with_members(hass: HomeAssistant, entities: set[str], depth: int = 0) -> set[str]:
    """Return the entities and, for each group among them, its members."""
    found = set(entities)
    if depth >= MAX_FOLLOW_DEPTH:
        return found
    for entity_id in entities:
        state = hass.states.get(entity_id)
        if state is None or entity_id.split(".", 1)[0] in {SCRIPT_DOMAIN, SCENE_DOMAIN}:
            continue
        members: set[str] = set()
        for attribute in _MEMBER_ATTRIBUTES:
            value = state.attributes.get(attribute)
            if isinstance(value, (list, tuple)):
                members |= {str(member).lower() for member in value}
        members -= found
        if members:
            found |= with_members(hass, members, depth + 1)
    return found


def driven_entities(hass: HomeAssistant, sequences: Iterable[Any]) -> set[str]:
    """
    Return the entities the steps act on.

    Every target the steps name directly counts. A script or template entity
    among them is followed into its own steps, there keeping only relay-like
    entities (see ``_RELAY_DOMAINS``). Targets that cannot be named (a
    template) are left out: the caller holds what it can.
    """
    found: set[str] = set()
    followed: set[str] = set()
    reader = StoredActionReader(hass)
    pending: list[tuple[Any, int]] = [(sequence, 0) for sequence in sequences]
    while pending:
        sequence, depth = pending.pop()
        try:
            calls = list(iter_service_calls(sequence, strict=True))
        except TooDeepError:
            LOGGER.warning(
                "Critical Action PIN: a template entity's actions nest too deeply "
                "to follow; the devices inside are not held"
            )
            continue
        for domain, service, step in calls:
            entities, _unresolved = call_targets(hass, domain, service, step)
            if depth:
                entities = {e for e in entities if e.split(".", 1)[0] in _RELAY_DOMAINS}
            found |= entities
            if depth >= MAX_FOLLOW_DEPTH:
                continue
            for entity_id in sorted(entities - followed):
                followed.add(entity_id)
                if entity_id.startswith(f"{SCENE_DOMAIN}."):
                    found |= {
                        e
                        for e in _scene_entities(hass, entity_id)
                        if e.split(".", 1)[0] in _RELAY_DOMAINS
                    }
                    continue
                if entity_id.split(".", 1)[0] in PRESS_DOMAINS:
                    # A press starts automations; what they switch is driven.
                    pending.extend(
                        (nested, depth + 1)
                        for nested in reader.started_by(entity_id) or []
                    )
                pending.extend(
                    (nested, depth + 1) for nested in _own_sequences(hass, entity_id)
                )
    return found


def _scene_entities(hass: HomeAssistant, entity_id: str) -> set[str]:
    """Return the entities a scene from Home Assistant's own config sets."""
    platform = hass.data.get("homeassistant_scene")
    entity = getattr(platform, "entities", {}).get(entity_id) if platform else None
    states = getattr(getattr(entity, "scene_config", None), "states", None)
    if not isinstance(states, dict):
        return set()
    return with_members(hass, {str(key).lower() for key in states})


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
