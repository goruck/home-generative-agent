"""
Read the actions an entity runs when the agent acts on it (#780, #814).

Acting on an entity can run stored actions no rule names:

* **A template entity** (from the Template integration) runs action scripts.
  For a template switch, button, lock and cover, Home Assistant keys each
  script by the service it answers, so a call runs the script with its
  service's name: a template switch's ``turn_on``, a template button's
  ``press``, a template lock's ``unlock``. A call with no script of its own
  (``toggle``, or a cover opened through its position script) may run any of
  them. Any other template entity picks its script from the call's data (a
  template light's ``turn_on`` with a brightness runs ``set_level``; a fan's
  runs ``set_percentage`` too), so every script it has counts.
* **Any entity whose state the call changes** (an input boolean turned on, a
  switch, a button pressed) starts what reacts to that change: an
  automation, a trigger-based template block with ``actions:``, and a script
  or automation waiting for it (``wait_for_trigger`` or ``wait_template``),
  which the change resumes. One counts when a trigger or wait names the
  entity, by entity id or registry id (or through the automation's
  variables), or cannot be tied to entities ahead: a ``state_changed`` or
  ``call_service`` event trigger (one filtered in its conditions can react),
  an event trigger whose type is templated or missing, a template reading
  states generically (``states.<its domain>``, ``states | ...``,
  ``expand(...)``, ``area_entities(...)``), a templated entity, target or
  event filter, an area, device, label or floor target, or a device trigger
  for the entity's domain or its own device. Its actions are what the
  change runs. A disabled automation does not count, and a wait counts
  only while its script or automation may be running. A group holding the
  entity changes with it, as a person does with a tracker and a zone with
  the people in it. An entity computed from it changes with it too:
  a template entity reading it, a helper subscribed to it (#820, see
  ``DerivedEntities``). A trigger's ``to:``/``from:`` is not read: turning a switch
  off counts if turning it on would start the automation. A template naming
  other entities (``states('sensor.x')``), another event type (a ZHA or
  mobile notification event) or another device's trigger (a remote) cannot
  react and does not count.

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
from functools import partial
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from homeassistant.helpers import entity_platform
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import event as ha_event
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import async_get_platforms
from homeassistant.helpers.target import (
    TargetSelection,
    async_extract_referenced_entity_ids,
)

from .automation_pin import DATA_CONTAINER_KEYS, TooDeepError, iter_service_calls

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

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
# A template reading states without naming one entity (see
# `_template_domains`): `states` alone or `states.<domain>`, not
# `states(...)` or `states.<domain>.<object>`; and the entity-list helpers.
_STATES_ACCESS = re.compile(
    r"\bstates\b(?:\s*\.\s*([a-z0-9_]+)(?:\.([a-z0-9_]+))?)?(?!\s*[(.a-z0-9_])"
)
# An entity id built from parts reads that domain: `'input_boolean.' ~ name`,
# `'input_boolean.let_' ~ 'in'`, `'input_boolean.{}'.format(name)`.
_BUILT_ID = re.compile(r"""['"]([a-z0-9_]+)\.(?:['"]|[a-z0-9_]*(?:['"]\s*[~+]|[{%]))""")
# A run of words joined by dots, for the names a text holds (see `_tokens`).
_DOTTED_RUN = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*")
_ENTITY_LISTS = re.compile(
    r"\b(?:expand|area_entities|label_entities|device_entities"
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
# Entities following one another (a template of a helper of a template...)
# are followed this many steps; one still followed past it is held.
_MAX_FOLLOWING_DEPTH = 10
# A group lists its members in `entity_id`; MQTT and ZHA groups in
# `group_entities`.
_MEMBER_ATTRIBUTES = ("entity_id", "group_entities")
# Entities whose state follows others': a group's members, a person's
# trackers, the people a zone counts (#814).
_PARENT_ATTRIBUTES = (*_MEMBER_ATTRIBUTES, "device_trackers", "persons")
_MQTT_DOMAIN = "mqtt"
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


@dataclass(frozen=True)
class _Trigger:
    """
    One trigger or wait, read once per check: when it may react to a change.

    ``always`` reacts to any change; otherwise a change reacts when its
    entity is named in ``text``, its domain is in ``domains`` (or
    ``all_domains``), or it is the device trigger's domain or device.
    """

    text: str
    always: bool = False
    templated: bool = False
    all_domains: bool = False
    domains: frozenset[str] = frozenset()
    device_domain: str | None = None
    device_id: str | None = None
    platform: str = ""


@dataclass
class Reactor:
    """
    An automation, waiting script or template block that may react to a change.

    ``name`` is what the PIN line calls it; None for a template block.
    """

    name: str | None
    triggers: list[_Trigger]
    sequence: Any
    variables: str = ""


@dataclass
class _ReactorIndex:
    """The reactors by what a change must name or be for each to react."""

    always: set[int] = field(default_factory=set)
    by_domain: dict[str, set[int]] = field(default_factory=dict)
    by_device: dict[str, set[int]] = field(default_factory=dict)
    by_name: dict[str, set[int]] = field(default_factory=dict)
    by_variable: dict[str, set[int]] = field(default_factory=dict)

    @classmethod
    def build(cls, reactors: Sequence[Reactor]) -> _ReactorIndex:
        """Index each reactor once, so a change looks up only its own."""
        index = cls()
        for number, reactor in enumerate(reactors):
            for trigger in reactor.triggers:
                index.add(number, trigger)
            if any(trigger.templated for trigger in reactor.triggers):
                # A template reading a variable that names the entity.
                for name in _tokens(reactor.variables):
                    index.by_variable.setdefault(name, set()).add(number)
        return index

    def add(self, number: int, trigger: _Trigger) -> None:
        """Index one trigger: see the module docstring for what reacts."""
        if trigger.always or trigger.all_domains:
            self.always.add(number)
            return
        for name in _tokens(trigger.text):
            self.by_name.setdefault(name, set()).add(number)
        for domain in trigger.domains:
            self.by_domain.setdefault(domain, set()).add(number)
        if trigger.device_domain:
            self.by_domain.setdefault(trigger.device_domain, set()).add(number)
        if trigger.device_id:
            self.by_device.setdefault(trigger.device_id, set()).add(number)

    def find(
        self, names: Iterable[str], domain: str, device_id: str | None
    ) -> set[int]:
        """Return the reactors a change to this entity may set off."""
        found = set(self.always) | self.by_domain.get(domain, set())
        if device_id is not None:
            found |= self.by_device.get(device_id, set())
        for name in names:
            found |= self.by_name.get(name, set())
            found |= self.by_variable.get(name, set())
        return found


def _tokens(text: str) -> set[str]:
    """
    Return every name a text may hold, as `_name_in` reads it.

    Each word, and each pair of words joined by a dot: `states.light.porch`
    holds `light.porch`, and a registry id is one word.
    """
    found: set[str] = set()
    for run in _DOTTED_RUN.findall(text):
        parts = run.split(".")
        found.update(parts)
        found.update(f"{a}.{b}" for a, b in pairwise(parts))
    return found


def _compile_trigger(trigger: Any) -> _Trigger:  # noqa: PLR0911
    """Read one trigger once (see the module docstring for what counts)."""
    if not isinstance(trigger, dict):
        return _Trigger(text="", always=True)
    texts = list(_texts(trigger))
    text = "\n".join(texts)
    templated = _templated(texts)
    platform = next(
        (trigger[key] for key in _TRIGGER_PLATFORM_KEYS if key in trigger), None
    )
    if not isinstance(platform, str):
        return _Trigger(text=text, always=True, templated=templated)
    platform = platform.lower()
    base = {"text": text, "templated": templated, "platform": platform}
    if platform == "event":
        return _compile_event(trigger, base)
    if platform == "template":
        all_domains, domains = _template_domains(texts)
        return _Trigger(**base, all_domains=all_domains, domains=domains)
    if _templated(_texts({key: trigger.get(key) for key in _WATCHED_KEYS})):
        return _Trigger(**base, always=True)  # a templated entity or target
    target = trigger.get("target")
    if target is not None and (
        not isinstance(target, dict)
        or any(target.get(key) for key in _TARGET_KEYS_NOT_ENTITIES)
    ):
        return _Trigger(**base, always=True)
    if platform == "device" and not trigger.get("entity_id"):
        # A device trigger names its device, not the entity: it may be the
        # entity's own, or one of its domain. A remote's trigger is not a
        # change the agent makes to a light.
        return _Trigger(
            **base,
            device_domain=str(trigger.get("domain") or "").lower(),
            device_id=str(trigger.get("device_id") or "") or None,
        )
    return _Trigger(**base)


def _compile_event(trigger: Mapping[str, Any], base: dict[str, Any]) -> _Trigger:  # noqa: PLR0911
    """
    Read an event trigger: only a change's own events can start it.

    A ``state_changed`` filtered to a named entity reacts to that entity only;
    a ``call_service`` filtered to a domain, to that domain's entities.
    """
    types = set(_texts(trigger.get("event_type")))
    if not types or _templated(types):
        return _Trigger(**base, always=True)
    if not types & _PRESS_EVENTS:
        return _Trigger(**base)  # another event type: only if it names it
    data = trigger.get("event_data")
    if not isinstance(data, dict) or _templated(_texts(data)):
        return _Trigger(**base, always=True)
    if "state_changed" in types and data.get("entity_id"):
        return _Trigger(**base)  # named entities only
    if types == {"call_service"} and data.get("domain"):
        domains = frozenset(_texts(data.get("domain")))
        if _GENERIC_DOMAIN in domains:
            return _Trigger(**base, all_domains=True)
        return _Trigger(**base, domains=domains)
    return _Trigger(**base, always=True)


def _template_domains(texts: Iterable[str]) -> tuple[bool, frozenset[str]]:
    """
    Return whether a template reads every entity, and the domains it reads.

    ``states.<domain>`` reads that domain; ``states | ...``, ``states[...]``,
    ``expand(...)`` and the ``*_entities(...)`` helpers read every entity; an
    id built from parts (``'input_boolean.' ~ name``) reads its domain.
    ``states('x.y')`` and ``states.x.y`` name one entity.
    """
    domains: set[str] = set()
    for text in texts:
        if _ENTITY_LISTS.search(text):
            return True, frozenset()
        for match in _STATES_ACCESS.finditer(text):
            accessed_domain, object_id = match.group(1), match.group(2)
            if accessed_domain is None:
                return True, frozenset()
            if object_id is None:
                domains.add(accessed_domain)
        domains.update(_BUILT_ID.findall(text))
    return False, frozenset(domains)


def _name_in(text: str, name: str) -> bool:
    """Return True if a text names the entity (not a longer id)."""
    if name not in text:
        return False
    return bool(re.search(rf"(?<![a-z0-9_]){re.escape(name)}(?![a-z0-9_])", text))


@dataclass
class StoredActionReader:
    """
    Reads stored actions for one check, reading the automations only once.

    Every entity a call changes looks up the automations that may react to
    it, so they are read, and their triggers compiled, once per check.
    """

    hass: HomeAssistant
    _reactors: list[Reactor] | None = None
    _index: _ReactorIndex | None = None
    _read: bool = False
    _parents: dict[str, set[str]] | None = None
    _derived: DerivedEntities | None = None
    _started: dict[str, list[Reactor] | None] = field(default_factory=dict)

    def own_sequences(self, entity_id: str, service: str | None) -> list[Any] | None:
        """
        Return the action scripts a template entity runs for ``service``.

        See the module docstring. None when they cannot be read (a template
        entity that is not loaded); empty for any other entity.
        """
        entity_id = entity_id.lower()
        if not is_template_entity(self.hass, entity_id):
            return []
        domain = entity_id.split(".", 1)[0]
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
        return own

    def started_by(self, entity_id: str, depth: int = 0) -> list[Reactor] | None:
        """
        Return what may react to a change to the entity, or to one following it.

        A group's state follows its members', so turning on a light changes
        the light group it is in too (#814); a template sensor or a helper
        computed from it changes too (#820). None when an automation cannot
        be read, when what follows the entity cannot be read, or when it goes
        deeper than ``_MAX_FOLLOWING_DEPTH``: the change is then held.
        """
        entity_id = entity_id.lower()
        if entity_id in self._started:
            return self._started[entity_id]
        self._started[entity_id] = []  # a group holding itself adds nothing
        found = self._find_started_by(entity_id)
        following = self._following(entity_id) if found is not None else set()
        if following is None:
            found = None
        elif following and depth >= _MAX_FOLLOWING_DEPTH:
            LOGGER.debug("%s is followed too deep to read; holding it", entity_id)
            found = None
        if found is not None and following:
            for parent in sorted(following):
                parents = self.started_by(parent, depth + 1)
                if parents is None:
                    found = None
                    break
                found += [r for r in parents if r not in found]
        self._started[entity_id] = found
        return found

    def _following(self, entity_id: str) -> set[str] | None:
        """Return the entities whose state follows this one's, or None."""
        try:
            if self._derived is None:
                self._derived = DerivedEntities(self.hass)
            derived = self._derived.of(entity_id)
        except Exception:
            # Hold rather than fail the turn or pass: an unexpected shape in
            # what Home Assistant tracks.
            LOGGER.exception(
                "Critical Action PIN: cannot read what follows %s; holding it",
                entity_id,
            )
            return None
        return (self._groups_holding(entity_id) | derived) - {entity_id}

    def mqtt_reactors(self) -> list[Reactor] | None:
        """
        Return what an `mqtt.publish` call may set off.

        An MQTT trigger, and an MQTT device trigger (a button on an MQTT
        device), react to a topic, which the gate does not match.
        """
        reactors = self._all_reactors()
        if reactors is None:
            return None
        return [
            reactor
            for reactor in reactors
            if any(
                _MQTT_DOMAIN in (trigger.platform, trigger.device_domain)
                # The `call_service` event the publish fires.
                or trigger.always
                or trigger.all_domains
                or _MQTT_DOMAIN in trigger.domains
                for trigger in reactor.triggers
            )
        ]

    def mqtt_entities(self) -> list[str]:
        """
        Return the MQTT entities a publish may set or command.

        Registered ones, and loaded ones with no registry entry (YAML
        without a unique id).
        """
        found = {
            entry.entity_id
            for entry in er.async_get(self.hass).entities.values()
            if entry.platform == _MQTT_DOMAIN and entry.disabled_by is None
        }
        for platform in async_get_platforms(self.hass, _MQTT_DOMAIN):
            found |= {str(entity_id).lower() for entity_id in platform.entities}
        return sorted(found)

    def automation_sequence(self, entity_id: str) -> list[Any] | None:
        """Return the steps `automation.trigger` runs, or None if unreadable."""
        for automation in getattr(
            self.hass.data.get(_AUTOMATION_DOMAIN), "entities", ()
        ):
            if str(getattr(automation, "entity_id", "")).lower() == entity_id:
                sequence = getattr(
                    getattr(automation, "action_script", None), "sequence", None
                )
                return sequence if isinstance(sequence, list) else None
        return None

    def _find_started_by(self, entity_id: str) -> list[Reactor] | None:
        reactors = self._all_reactors()
        if reactors is None:
            return None
        if self._index is None:
            self._index = _ReactorIndex.build(reactors)
        names, domain, device_id = _change_keys(self.hass, entity_id)
        return [
            reactors[number]
            for number in sorted(self._index.find(names, domain, device_id))
        ]

    def _groups_holding(self, entity_id: str) -> set[str]:
        if self._parents is None:
            parents: dict[str, set[str]] = {}
            for state in self.hass.states.async_all():
                if state.domain in {SCRIPT_DOMAIN, SCENE_DOMAIN}:
                    continue
                for attribute in _PARENT_ATTRIBUTES:
                    members = state.attributes.get(attribute)
                    if isinstance(members, (list, tuple)):
                        for member in members:
                            parents.setdefault(str(member).lower(), set()).add(
                                state.entity_id
                            )
            # A zone lists only the people in it: one arriving joins it.
            zones = {state.entity_id for state in self.hass.states.async_all("zone")}
            for person in self.hass.states.async_entity_ids("person"):
                parents.setdefault(person, set()).update(zones)
            self._parents = parents
        return self._parents.get(entity_id, set())

    def _all_reactors(self) -> list[Reactor] | None:
        if not self._read:
            self._read = True
            self._reactors = _read_reactors(self.hass)
        return self._reactors


def _change_keys(
    hass: HomeAssistant, entity_id: str
) -> tuple[list[str], str, str | None]:
    """Return what a trigger may name a change by: ids, domain and device."""
    names = [entity_id]
    entry = er.async_get(hass).async_get(entity_id)
    device_id = None
    if entry is not None:
        names.append(entry.id.lower())
        device_id = entry.device_id
    return names, entity_id.split(".", 1)[0], device_id


# Where Home Assistant keeps its per-entity state-change subscriptions
# (`helpers/event.py`); private, so read by name and warned about if gone.
_TRACK_STATE_CHANGE_KEY = "_TRACK_STATE_CHANGE_DATA"
_ALL_LISTENER, _ENTITIES_LISTENER, _DOMAINS_LISTENER = "all", "entities", "domains"
_warned_unreadable: set[str] = set()


def _warn_once(what: str) -> None:
    if what not in _warned_unreadable:
        _warned_unreadable.add(what)
        LOGGER.warning(
            "Critical Action PIN: cannot read %s; entities computed from a "
            "change the agent makes may not be followed",
            what,
        )


class DerivedEntities:
    """
    The entities whose state is computed from another's, as HA tracks them (#820).

    Read once per check from what Home Assistant itself listens to:

    * a template entity re-renders on the entities, domains or all states its
      templates read at their last render (``TrackTemplateResultInfo``), so a
      template that reads the helper only while another entity is on is
      followed only then; one whose listeners cannot be read follows every
      change;
    * a trigger-based template entity changes when its block's triggers fire;
    * any other entity listening to a state change through
      ``async_track_state_change_event`` (min/max, threshold, bayesian,
      statistics, filter, trend, derivative, integration, utility meter,
      mold indicator, switch_as_x; YAML or UI) follows that entity.
    """

    def __init__(self, hass: HomeAssistant) -> None:
        """Read what Home Assistant tracks, once."""
        self.hass = hass
        self.by_entity: dict[str, set[str]] = {}
        self.by_domain: dict[str, set[str]] = {}
        self.every: set[str] = set()
        self._triggered: list[tuple[Reactor, set[str]]] = []
        self._trigger_index: _ReactorIndex | None = None
        self._read_templates()
        self._read_subscriptions()
        self._read_entity_trackers()

    def of(self, entity_id: str) -> set[str]:
        """Return the entities a change to this one changes too."""
        found = (
            self.by_entity.get(entity_id, set())
            | self.by_domain.get(entity_id.split(".", 1)[0], set())
            | self.every
        )
        if self._triggered:
            if self._trigger_index is None:
                self._trigger_index = _ReactorIndex.build(
                    [reactor for reactor, _ in self._triggered]
                )
            names, domain, device_id = _change_keys(self.hass, entity_id)
            for number in self._trigger_index.find(names, domain, device_id):
                found = found | self._triggered[number][1]
        return found - {entity_id}

    def _read_templates(self) -> None:
        blocks: dict[int, tuple[Any, set[str]]] = {}
        for platform in async_get_platforms(self.hass, TEMPLATE_PLATFORM):
            for entity_id, entity in list(getattr(platform, "entities", {}).items()):
                entity_id = str(entity_id).lower()  # noqa: PLW2901
                coordinator = getattr(entity, "coordinator", None)
                if coordinator is not None and isinstance(
                    getattr(coordinator, "config", None), dict
                ):
                    # Trigger-based: changes when its block is triggered.
                    blocks.setdefault(id(coordinator), (coordinator, set()))[1].add(
                        entity_id
                    )
                    continue
                self._read_template(entity_id, entity)
        for coordinator, entity_ids in blocks.values():
            config = coordinator.config
            raw = next((config[key] for key in _TRIGGER_KEYS if key in config), [])
            triggers = raw if isinstance(raw, list) else [raw]
            self._add_triggered(
                [_compile_trigger(t) for t in triggers],
                entity_ids,
                "\n".join(
                    text for key in _VARIABLE_KEYS for text in _texts(config.get(key))
                ),
            )

    def _read_template(self, entity_id: str, entity: Any) -> None:
        """
        Follow a template entity from what it reads.

        HA's live listeners, plus what its text reads generically: while a
        rate-limited render is pending, HA drops the domain and all-states
        listeners from its summary. Before it first renders, or when the
        listeners cannot be read, its text alone: every entity or domain it
        names.
        """
        texts = _template_texts(entity)
        listeners = self._listeners(entity)
        if listeners is None or (not listeners and texts):
            if not texts:
                _warn_once(f"what template entity {entity_id} reads")
                self.every.add(entity_id)
                return
            self._add_triggered(
                [_compile_trigger({"trigger": "template", "value_template": texts})],
                {entity_id},
            )
            return
        reads_all, domains = _template_domains([texts.lower()])
        if listeners.get(_ALL_LISTENER) or reads_all or _IMPORTS_TEMPLATE.search(texts):
            self.every.add(entity_id)
        for source in listeners.get(_ENTITIES_LISTENER) or ():
            self.by_entity.setdefault(str(source).lower(), set()).add(entity_id)
        for domain in {*(listeners.get(_DOMAINS_LISTENER) or ()), *domains}:
            self.by_domain.setdefault(str(domain).lower(), set()).add(entity_id)

    def _add_triggered(
        self, triggers: list[_Trigger], entity_ids: set[str], variables: str = ""
    ) -> None:
        self._triggered.append(
            (
                Reactor(name=None, triggers=triggers, sequence=[], variables=variables),
                entity_ids,
            )
        )

    @staticmethod
    def _listeners(entity: Any) -> dict[str, Any] | None:
        """Return what a template entity re-renders on, {} before it renders."""
        if not hasattr(entity, "_template_result_info"):
            return None
        info = entity._template_result_info  # noqa: SLF001
        if info is None:
            return {}  # not rendered yet
        listeners = info.listeners
        return listeners if isinstance(listeners, dict) else None

    def _read_subscriptions(self) -> None:
        """Follow each entity subscribed to a state change, by entity or to all."""
        key = getattr(ha_event, _TRACK_STATE_CHANGE_KEY, None)
        if key is None:
            _warn_once("Home Assistant's state-change subscriptions")
            return
        data = self.hass.data.get(key)
        callbacks = getattr(data, "callbacks", None) if data is not None else {}
        if not isinstance(callbacks, dict):
            _warn_once("Home Assistant's state-change subscriptions")
            return
        search = _OwnerSearch()
        for source, jobs in list(callbacks.items()):
            for job in list(jobs):
                owners = search.owners(job)
                if owners:
                    self.by_entity.setdefault(str(source).lower(), set()).update(owners)
        # A template reading every state listens on the bus itself.
        for job in _bus_state_listeners(self.hass):
            self.every |= search.owners(job)
        search.finish()

    def _read_entity_trackers(self) -> None:
        """
        Follow each entity's template trackers (bayesian, universal...).

        An entity keeps its tracker's ``async_remove`` to call on removal,
        so its trackers are found whatever they subscribe to right now: a
        pending rate-limited render has no domain or all-states subscription.
        """
        platforms = self.hass.data.get(entity_platform.DATA_ENTITY_PLATFORM) or {}
        for domain_platforms in list(platforms.values()):
            for platform in list(domain_platforms):
                for entity_id, entity in list(
                    getattr(platform, "entities", {}).items()
                ):
                    try:
                        trackers = [
                            tracker
                            for remove in list(
                                getattr(entity, "_on_remove", None) or ()
                            )
                            if hasattr(
                                tracker := getattr(remove, "__self__", None),
                                "_track_templates",
                            )
                        ]
                    except Exception:  # noqa: BLE001
                        _warn_once(f"the template trackers of {entity_id}")
                        continue
                    for tracker in trackers:
                        self._read_tracker(str(entity_id).lower(), tracker)

    def _read_tracker(self, owner: str, tracker: Any) -> None:
        """Follow a template tracker's owner by what it and its text read."""
        try:
            listeners = tracker.listeners
        except Exception:  # noqa: BLE001
            listeners = None  # not set up yet: the text alone
        if isinstance(listeners, dict):
            if listeners.get(_ALL_LISTENER):
                self.every.add(owner)
            for source in listeners.get(_ENTITIES_LISTENER) or ():
                self.by_entity.setdefault(str(source).lower(), set()).add(owner)
            for domain in listeners.get(_DOMAINS_LISTENER) or ():
                self.by_domain.setdefault(str(domain).lower(), set()).add(owner)
        self._read_tracker_text(tracker, {owner}, names=not isinstance(listeners, dict))

    def _read_tracker_text(
        self, tracker: Any, owners: set[str], *, names: bool = False
    ) -> None:
        """
        Follow a template tracker's owners by what its text reads.

        Generic reads always: a pending rate-limited render drops its domain
        and all-states subscriptions. Named entities too (``names``) when its
        live listeners cannot be read.
        """
        texts = "\n".join(
            text
            for track in getattr(tracker, "_track_templates", None) or ()
            if isinstance(
                text := getattr(getattr(track, "template", None), "template", None), str
            )
        )
        reads_all, domains = _template_domains([texts.lower()])
        if reads_all or _IMPORTS_TEMPLATE.search(texts):
            self.every |= owners
        for domain in domains:
            self.by_domain.setdefault(domain, set()).update(owners)
        if names and texts:
            self._add_triggered(
                [_compile_trigger({"trigger": "template", "value_template": texts})],
                set(owners),
            )


def _bus_state_listeners(hass: HomeAssistant) -> list[Any]:
    """Return the jobs listening to every state change on the bus."""
    listeners = getattr(getattr(hass, "bus", None), "_listeners", None)
    if not isinstance(listeners, dict):
        _warn_once("Home Assistant's event bus listeners")
        return []
    jobs = listeners.get("state_changed") or []
    return [
        entry[0] if isinstance(entry, tuple) and entry else entry
        for entry in list(jobs)
    ]


def _template_texts(entity: Any) -> str:
    """Return a template entity's templates as text, or "" if unreadable."""
    attrs = getattr(entity, "_template_attrs", None)
    if not isinstance(attrs, dict):
        return ""
    return "\n".join(
        text
        for template in attrs
        if isinstance(text := getattr(template, "template", None), str)
    )


# A template importing another template (a macro) may read anything.
_IMPORTS_TEMPLATE = re.compile(r"\{%-?\s*(?:from|import|include)\b")
# How many links one owner search follows, and how many objects one check
# visits in all before it stops reading (and warns).
_OWNER_SEARCH_DEPTH = 8
_OWNER_SEARCH_BUDGET = 50_000
_CONTAINERS = (dict, list, tuple, set, frozenset, str, bytes)
# The owners of each state-change job, kept across checks: a job's owner does
# not change. Pruned to the jobs still subscribed at each read.
_JOB_OWNERS: dict[int, tuple[Any, frozenset[str]]] = {}


class _OwnerSearch:
    """
    Find the entities a state-change callback updates (#820).

    Only the links Home Assistant uses are followed, not every object a
    callback can reach (a shared dict of an integration's entities would
    follow them all):

    * a job to its target, a bound method to its object, a partial to its
      function and arguments, a function to the values it closes over (an
      entity's method defines its listener inside, closing over the entity);
    * a filtered state tracker to its action, a template tracker
      (``TrackTemplateResultInfo``) to its job, a coordinator (history
      stats, proximity) to the callbacks of its listeners.

    Every entity reached counts. An object that cannot be read is skipped
    with a warning, rather than holding every call.
    """

    def __init__(self) -> None:
        self._memo: dict[int, frozenset[str]] = {}
        self._visited = 0
        self._jobs: dict[int, tuple[Any, frozenset[str]]] = {}

    def owners(self, job: Any) -> set[str]:
        cached = _JOB_OWNERS.get(id(job))
        if cached is not None and cached[0] is job:
            found = cached[1]
        else:
            found = self._search(job, 0, set())
        self._jobs[id(job)] = (job, found)
        return set(found)

    def finish(self) -> None:
        """Keep the owners of the jobs seen in this read, for the next."""
        _JOB_OWNERS.clear()
        _JOB_OWNERS.update(self._jobs)

    def _search(self, node: Any, depth: int, path: set[int]) -> frozenset[str]:
        if node is None or isinstance(node, _CONTAINERS) or id(node) in path:
            return frozenset()
        if id(node) in self._memo:
            return self._memo[id(node)]
        self._visited += 1
        if self._visited > _OWNER_SEARCH_BUDGET:
            _warn_once("all of Home Assistant's state-change subscriptions in time")
            return frozenset()
        if isinstance(node, Entity):
            entity_id = getattr(node, "entity_id", None)
            owner = frozenset(
                [entity_id.lower()] if isinstance(entity_id, str) and entity_id else []
            )
            self._memo[id(node)] = owner
            return owner
        if depth >= _OWNER_SEARCH_DEPTH or type(node).__name__ == "HomeAssistant":
            return frozenset()
        try:
            links = _owner_links(node)
        except Exception:  # noqa: BLE001
            _warn_once(f"a state-change callback ({type(node).__name__})")
            links = []
        found: set[str] = set()
        path.add(id(node))
        for link in links:
            found |= self._search(link, depth + 1, path)
        path.discard(id(node))
        result = frozenset(found)
        self._memo[id(node)] = result
        return result


def _owner_links(node: Any) -> list[Any]:
    """Return the objects a callback, job or tracker hands state changes to."""
    if isinstance(node, partial):
        return [node.func, *node.args, *node.keywords.values()]
    links: list[Any] = [getattr(node, "__self__", None)]
    links.extend(
        cell.cell_contents
        for cell in getattr(node, "__closure__", None) or ()
        if _cell_filled(cell)
    )
    links.extend(getattr(node, name, None) for name in ("target", "_job", "_action"))
    listeners = getattr(node, "_listeners", None)
    if isinstance(listeners, dict) and hasattr(node, "async_update_listeners"):
        # A coordinator: each listener is (update callback, context).
        links.extend(
            entry[0] if isinstance(entry, tuple) and entry else entry
            for entry in listeners.values()
        )
    return links


def _cell_filled(cell: Any) -> bool:
    try:
        cell.cell_contents  # noqa: B018
    except ValueError:
        return False
    return True


def _read_reactors(hass: HomeAssistant) -> list[Reactor] | None:
    """
    Read what can react to a change: automations, scripts and template blocks.

    None if one that can run cannot be read: changes are then held.
    """
    found: list[Reactor] = []
    for automation in getattr(hass.data.get(_AUTOMATION_DOMAIN), "entities", ()):
        script = getattr(automation, "action_script", None)
        if script is None:
            continue  # an automation that failed to set up does not run
        entity_id = str(getattr(automation, "entity_id", "") or "")
        state = hass.states.get(entity_id) if entity_id else None
        disabled = state is not None and state.state == "off"
        waiting = _may_be_running(state, "current")
        if disabled and not waiting:
            continue  # neither triggers nor waits
        config = getattr(automation, "raw_config", None)
        # The validated triggers, after blueprint substitution; the raw
        # config only where they are not kept. A disabled automation does
        # not trigger, but one turned off without stopping it still waits.
        triggers = getattr(automation, "_trigger_config", None)
        read = _reactor(
            _friendly_name(state) or entity_id or "an automation",
            config,
            getattr(script, "sequence", None),
            [] if disabled else triggers if isinstance(triggers, list) else None,
            waiting=waiting,
        )
        if read is None:
            LOGGER.warning(
                "Critical Action PIN: cannot read %s; holding changes it may react to",
                getattr(automation, "entity_id", automation),
            )
            return None
        found.append(read)
    # A script runs only when started; one waiting for the change resumes.
    waiting = _waiting_scripts(hass)
    if waiting is None:
        return None
    found += waiting
    # A trigger-based template block with `actions:` runs them when triggered.
    coordinators = hass.data.get(_TEMPLATE_DOMAIN)
    if coordinators is not None and not isinstance(coordinators, list):
        LOGGER.warning("Critical Action PIN: cannot read the template blocks")
        return None
    for coordinator in coordinators or ():
        config = getattr(coordinator, "config", None)
        if not isinstance(config, dict):
            LOGGER.warning("Critical Action PIN: cannot read a template block")
            return None
        actions = next((config[key] for key in _ACTION_KEYS if key in config), None)
        if actions is None:
            continue
        read = _reactor(None, config, actions, None)
        if read is None:
            return None
        found.append(read)
    return found


def _waiting_scripts(hass: HomeAssistant) -> list[Reactor] | None:
    """Read the running scripts' waits; None if a running one cannot be read."""
    found: list[Reactor] = []
    for script_entity in getattr(hass.data.get(SCRIPT_DOMAIN), "entities", ()):
        script = getattr(script_entity, "script", None)
        if script is None:
            continue  # a script that failed to set up cannot run or wait
        entity_id = str(getattr(script_entity, "entity_id", "") or "")
        state = hass.states.get(entity_id) if entity_id else None
        if state is not None and state.state == "off":
            continue  # not running: nothing waits
        config = getattr(script_entity, "raw_config", None)
        sequence = getattr(script, "sequence", None)
        if not isinstance(config, dict) or not isinstance(sequence, list):
            LOGGER.warning(
                "Critical Action PIN: cannot read %s; holding changes it may wait on",
                entity_id or script_entity,
            )
            return None
        waits = [
            _compile_trigger(t)
            for t in _wait_triggers([config.get("sequence"), sequence])
        ]
        if waits:
            found.append(
                Reactor(
                    name=_friendly_name(state) or entity_id or "a script",
                    triggers=waits,
                    sequence=sequence,
                    variables="\n".join(_texts(config.get("variables"))),
                )
            )
    return found


def _friendly_name(state: Any) -> str | None:
    name = state.attributes.get("friendly_name") if state is not None else None
    return name if isinstance(name, str) and name else None


def _may_be_running(state: Any, runs_attribute: str) -> bool:
    """Return False only for an automation known to have no run waiting."""
    runs = state.attributes.get(runs_attribute) if state is not None else None
    return not (isinstance(runs, int) and runs == 0)


def _reactor(
    name: str | None,
    config: Any,
    sequence: Any,
    triggers: list[Any] | None,
    *,
    waiting: bool = True,
) -> Reactor | None:
    """
    Read an automation or template block's triggers, waits and actions.

    Its waits count while a run may be waiting; they are read from the
    validated steps (a blueprint's raw config holds only its inputs) and the
    raw ones.
    """
    if not isinstance(config, dict) or not isinstance(sequence, list):
        return None
    if triggers is None:
        raw = next((config[key] for key in _TRIGGER_KEYS if key in config), [])
        triggers = raw if isinstance(raw, list) else [raw]
    actions = next((config[key] for key in _ACTION_KEYS if key in config), [])
    alias = config.get("alias")
    waits = _wait_triggers([actions, sequence]) if waiting else ()
    return Reactor(
        name=str(alias) if isinstance(alias, str) and alias else name,
        triggers=[
            *(_compile_trigger(t) for t in triggers),
            *(_compile_trigger(t) for t in waits),
        ],
        sequence=sequence,
        variables="\n".join(
            text for key in _VARIABLE_KEYS for text in _texts(config.get(key))
        ),
    )


def _wait_triggers(node: Any) -> Iterable[Any]:
    """Yield the triggers of every wait step, however nested."""
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


def entities_of(hass: HomeAssistant, domain: str) -> list[str]:
    """
    Return the entity ids of a domain a call aimed at `all` may act on.

    Every entity of the domain may start automations (#814). Template
    entities registered but not loaded count while the Template integration
    has not loaded: one that loads later runs its actions, so the caller
    holds them as unreadable.
    """
    found = {state.entity_id.lower() for state in hass.states.async_all(domain)}
    found |= {
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
                        (reactor.sequence, depth + 1)
                        for reactor in reader.started_by(entity_id) or []
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
