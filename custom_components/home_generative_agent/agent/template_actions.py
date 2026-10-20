"""
Read the actions an entity runs when the agent acts on it (#780).

Some entities run stored actions of their own, which no rule names:

* **A template entity** (from the Template integration) runs an action
  script for each of its services: a template switch's ``turn_on``, a
  template button's ``press``, a template cover's ``open_cover``, a template
  lock's ``unlock``. Home Assistant keys each script by the service it
  answers, so a call runs the script with its service's name. A call with no
  script of its own (``toggle``, or a cover opened through its position
  script) may run any of them.
* **An input button** runs the automations that trigger on it.

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
    from collections.abc import Iterable

    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

TEMPLATE_PLATFORM = "template"
INPUT_BUTTON_DOMAIN = "input_button"
_AUTOMATION_DOMAIN = "automation"
_SCRIPT_DOMAIN = "script"
_TRIGGER_KEYS = ("triggers", "trigger")
# A script's steps can drive other scripts; followed this deep.
_MAX_DEPTH = 5


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


def runs_stored_actions(hass: HomeAssistant, entity_id: str) -> bool:
    """Return True for a template entity or an input button."""
    entity_id = entity_id.lower()
    if entity_id.split(".", 1)[0] == INPUT_BUTTON_DOMAIN:
        return True
    return _platform(hass, entity_id, _loaded_entity(hass, entity_id)) == (
        TEMPLATE_PLATFORM
    )


def stored_action_sequences(
    hass: HomeAssistant, entity_id: str, service: str | None
) -> list[Any] | None:
    """
    Return the step lists acting on the entity with ``service`` runs.

    For a template entity, the script for ``service``, or all of them when it
    has none of that name or ``service`` is None; for an input button, the
    actions of each automation that triggers on it. None when they cannot be
    read: a template entity that is not loaded, or an automation whose config
    or actions cannot be read.
    """
    entity_id = entity_id.lower()
    if entity_id.split(".", 1)[0] == INPUT_BUTTON_DOMAIN:
        return _automations_started_by(hass, entity_id)
    scripts = getattr(_loaded_entity(hass, entity_id), "_action_scripts", None)
    if not isinstance(scripts, dict):
        LOGGER.debug("Cannot read the actions of %s; holding it", entity_id)
        return None
    chosen = [scripts[service]] if service in scripts else list(scripts.values())
    sequences = [getattr(script, "sequence", None) for script in chosen]
    if any(not isinstance(sequence, list) for sequence in sequences):
        LOGGER.debug("Cannot read the actions of %s; holding it", entity_id)
        return None
    return sequences


def _mentions(value: Any, entity_id: str) -> bool:
    """Return True if any text in a config value names the entity."""
    if isinstance(value, str):
        return entity_id in value.lower()
    if isinstance(value, dict):
        return any(
            _mentions(key, entity_id) or _mentions(item, entity_id)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(_mentions(item, entity_id) for item in value)
    # A validated template keeps its text here.
    text = getattr(value, "template", None)
    return isinstance(text, str) and entity_id in text.lower()


def _automations_started_by(hass: HomeAssistant, entity_id: str) -> list[Any] | None:
    """Return the actions of each automation whose triggers name the entity."""
    component = hass.data.get(_AUTOMATION_DOMAIN)
    if component is None:
        return []
    sequences: list[Any] = []
    for automation in getattr(component, "entities", ()):
        config = getattr(automation, "raw_config", None)
        if not isinstance(config, dict):
            LOGGER.debug("Cannot read %s; holding %s", automation, entity_id)
            return None
        triggers = next((config[key] for key in _TRIGGER_KEYS if key in config), None)
        if not _mentions(triggers, entity_id):
            continue
        script = getattr(automation, "action_script", None)
        if script is None:
            continue  # an automation that failed to set up does not run
        sequence = getattr(script, "sequence", None)
        if not isinstance(sequence, list):
            LOGGER.debug("Cannot read %s; holding %s", automation, entity_id)
            return None
        sequences.append(sequence)
    return sequences


def template_entities(hass: HomeAssistant, domain: str) -> list[Any]:
    """Return the loaded template entities of a domain."""
    component = hass.data.get(domain)
    return [
        entity
        for entity in getattr(component, "entities", ())
        if getattr(getattr(entity, "platform", None), "platform_name", None)
        == TEMPLATE_PLATFORM
    ]


def driven_entities(
    hass: HomeAssistant, sequences: Iterable[Any], *, depth: int = 0
) -> set[str]:
    """
    Return the entities the steps call a service on, by entity id.

    A script among them is followed into its own steps. Targets that cannot be
    named (an area, a template) are left out: the caller holds what it can.
    """
    found: set[str] = set()
    for sequence in sequences:
        for _domain, _service, step in iter_service_calls(sequence):
            entities, _unresolved = step_targets(step)
            found.update(entity.lower() for entity in entities)
    if depth < _MAX_DEPTH:
        for script_id in sorted(e for e in found if e.startswith(f"{_SCRIPT_DOMAIN}.")):
            script = getattr(_loaded_entity(hass, script_id), "script", None)
            sequence = getattr(script, "sequence", None)
            if isinstance(sequence, list):
                found |= driven_entities(hass, [sequence], depth=depth + 1)
    return found
