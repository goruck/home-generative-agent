"""
Extend critical-action rules to the entities their ids miss (#773, #794).

A rule's ``entity_match`` is matched against the entity ids a call targets.
Three kinds of entity act like a matching one without carrying the word in
their own id:

* **An entity whose device class is the word.** A cover whose device class is
  ``garage`` but whose id is ``cover.msg100_abc`` opens a garage door. The
  device class must equal the word, so a rule for ``door`` does not reach
  device class ``garage``. The class the user set ("Show as") and the
  integration's own both count: a garage opener shown as a blind is held.
* **An entity named with the word.** A cover with no device class, id
  ``cover.shelly1_abc`` and name "Garage Door", which "open the garage door"
  finds by its name. The word, or its plural, must have no letter next to it
  in the friendly name, the registry or integration name, an alias, or the
  device's name: "Driveway Gates" and "Gate1" match ``gate``, "Indoor Blinds"
  does not match ``door``. A cover its integration classes as a window
  covering is not matched by name ("French Door Blinds"); a user's "Show as"
  does not turn the name match off. A word written like an id
  (``garage_door``) matches "Garage Door". Compound words ("GarageDoor")
  and names in other languages are not read; device class or a rule naming
  the id covers those.
* **A group holding a matching entity.** Opening ``cover.all_covers``, or a
  call on an old-style ``group.*``, acts on every member, the garage door
  included. Groups of groups are followed to any depth. A member matches by
  the ways above, or by the word anywhere in its id, as a direct call on it
  is matched (``cover.backdoor``, and so ``cover.outdoor_awning`` too). A member
  Home Assistant has not set up or registered yet could be anything, so its
  groups are held. A copy made for an old-style ``group.*`` keeps the rule's
  domain, so it holds ``cover.open_cover`` aimed at the group. A call in the
  ``group`` or ``homeassistant`` domain on it is not matched here: in an
  automation it is held as an unresolved target (``automation_pin``), and
  Home Assistant refuses HassTurnOn on it (the group domain has no turn_on).

A fourth way to the same door is the switch under it (#794). A ``switch_as_x``
cover, valve or lock is a switch wrapped as one: opening it or unlocking it
switches the relay. So a rule that holds such a cover or valve, or unlocking
such a lock, also holds calls on the relay, on any group holding it, on the
members of a switch group it wraps, and on any other wrapper of any of those
(a second cover, a light). An entity in the relay's domain that is not set up
or whose members cannot be read is held too, as ``reach`` holds one. For a
cover or valve every call on the relay is held: a momentary relay moves the
door whichever way it is switched, and the wrapper's invert option says
nothing of the wiring. For a lock only the unlocking direction (on, or off
when inverted) and ``toggle`` are held.
The relay is read from the wrapper's config entry, whether or not the wrapper
is enabled: disabling it does not disconnect the relay. A template cover that
drives another device is not followed here (#780).

``expand_critical_rules`` adds, for each such entity, a copy of the rule that
matches that entity id exactly (``CRITICAL_RULE_EXACT_ENTITY``). Every screen
downstream (the tool-call gate, intents, scripts, scenes, ``add_automation``)
matches on entity ids through ``rule_entity_matches``, so each one holds the
call without knowing why the entity counts, and ``cover.a`` being a garage
door does not hold ``cover.awning``.

Every check counts entities that are not loaded or not available, because an
automation the agent writes runs later. An entity only in the entity registry
is read from it. One not set up yet (no state, or the placeholder Home
Assistant restores at startup) in the rule's domain is held unless its
integration classes it as a window covering, since it could be a group whose
members cannot be seen yet; a disabled one is not, since it cannot run. An
entity that cannot be read is held the same way and logged, so one bad entity
does not fail every call.

A group's members are read from its state (``entity_id``, or
``group_entities`` for MQTT and ZHA groups, which Home Assistant writes even
while the group is unavailable), else from the group helper's config entry. A
group helper whose members cannot be read is held. Membership and device
class are read when the check runs: a member added to a group after an
automation is written is not checked again.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, cast

from homeassistant.const import (
    ATTR_DEVICE_CLASS,
    ATTR_ENTITY_ID,
    ATTR_FRIENDLY_NAME,
    ATTR_RESTORED,
    CONF_ENTITIES,
)
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er

from .helpers import CRITICAL_RULE_EXACT_ENTITY

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from homeassistant.core import HomeAssistant, State

# MQTT and ZHA group entities list their members here. It is a capability
# attribute, which Home Assistant writes even while the entity is unavailable.
_ATTR_GROUP_ENTITIES = "group_entities"
# The integration whose entities are group helpers (cover, light, ... groups).
_GROUP_PLATFORM = "group"
# Scenes and scripts list the entities they set in `entity_id` too, but they
# are opened and screened by `indirect_pin`, not followed as groups here.
_NOT_GROUP_DOMAINS = ("script.", "scene.")
# Characters a name or a rule's word may use between words.
_SEPARATORS = re.compile(r"[_\-]+")
# The integration that wraps a switch as a cover, lock and so on, and the
# option keys of its config entry. Read as strings: importing the component
# would make it a dependency.
_SWITCH_AS_X = "switch_as_x"
_WRAPPED_ENTITY = "entity_id"
_WRAPPED_INVERT = "invert"
# The services that open a cover or valve or unlock a lock, per domain: a rule
# holding one of these holds the relay under a wrapper of that domain.
_OPENING_SERVICES: dict[str, frozenset[str]] = {
    "cover": frozenset({"open_cover", "open", "toggle", "set_cover_position"}),
    "valve": frozenset({"open_valve", "open", "toggle", "set_valve_position"}),
    "lock": frozenset({"unlock", "open", "toggle"}),
}
# Wrapper domains whose relay is held both ways (see the module docstring).
_EVERY_RELAY_CALL = frozenset({"cover", "valve"})


def read_switch_wrappers(hass: HomeAssistant) -> dict[str, tuple[str, bool]]:
    """
    Return every switch_as_x wrapper -> (the switch it wraps, inverted).

    Disabled wrappers count: the relay still works. An entry that cannot be
    read is logged and skipped.
    """
    config_entries = getattr(hass, "config_entries", None)
    entries = config_entries.async_entries(_SWITCH_AS_X) if config_entries else []
    if not entries:
        return {}
    registry = er.async_get(hass)
    wrapped: dict[str, tuple[str, bool]] = {}
    for config_entry in entries:
        try:
            switch_id = er.async_resolve_entity_id(
                registry, str(config_entry.options.get(_WRAPPED_ENTITY) or "")
            )
            invert = bool(config_entry.options.get(_WRAPPED_INVERT))
            wrappers = [
                entry.entity_id
                for entry in er.async_entries_for_config_entry(
                    registry, config_entry.entry_id
                )
            ]
        except Exception:
            # One bad entry must not fail every guarded call in the house.
            # Nothing here raises for a well-formed entry (an unknown id
            # resolves to None), so this is not a way past the gate.
            LOGGER.exception(
                "Critical Action PIN: could not read switch_as_x entry %s",
                config_entry.entry_id,
            )
            continue
        if not switch_id:
            continue
        for wrapper in wrappers:
            wrapped[wrapper.lower()] = (switch_id.lower(), invert)
    return wrapped


# Cover device classes that say the cover is a window covering. Its name is
# not read for the word: "French Door Blinds" is not a door. `shutter` is not
# here, since roller garage doors are often classed so.
_WINDOW_COVERINGS = frozenset({"awning", "blind", "curtain", "shade", "window"})


def _text_list(value: object) -> list[str]:
    """Return the strings in a list attribute, lowercased, or none."""
    if not isinstance(value, (list, tuple, set)):
        return []
    items = cast("Iterable[object]", value)
    return [item.lower() for item in items if isinstance(item, str)]


def _plain(text: str) -> str:
    """Return text lowercased with `_` and `-` read as spaces."""
    return _SEPARATORS.sub(" ", text.lower())


class _EntityIndex:
    """Every entity's names, device classes and group membership, read once."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass
        registry = er.async_get(hass)
        self._devices = dr.async_get(hass)
        # Every class an entity has been given: the one HA reports, the one
        # the user set ("Show as") and the integration's own. A garage opener
        # shown as a blind is still a garage opener. Keyed by every entity the
        # index knows, so its keys are also the "known" set.
        self.classes: dict[str, set[str]] = {}
        self.names: dict[str, list[str]] = {}
        self.members: dict[str, list[str]] = {}
        self.unreadable_groups: set[str] = set()
        self.unknown: set[str] = set()
        for state in hass.states.async_all():
            entity_id = state.entity_id.lower()
            self._read(entity_id, registry.async_get(entity_id), state)
        for entry in registry.entities.values():
            if entry.entity_id not in self.classes:
                self._read(entry.entity_id, entry, None)
        # Member -> the groups holding it, so a match climbs to every group
        # holding it in one pass, however deep the nesting.
        self.holders: dict[str, set[str]] = {}
        for group_id, group_members in self.members.items():
            for member in group_members:
                self.holders.setdefault(member, set()).add(group_id)
        # A switch_as_x cover, valve or lock -> (the switch it wraps, inverted),
        # and each wrapped switch -> its wrappers.
        self.wrapped = read_switch_wrappers(hass)
        self.wrappers_of: dict[str, set[str]] = {}
        for wrapper, (switch_id, _invert) in self.wrapped.items():
            self.wrappers_of.setdefault(switch_id, set()).add(wrapper)

    def climb(self, entity_ids: Iterable[str]) -> set[str]:
        """Return ``entity_ids`` and every group holding one, however deep."""
        found = set(entity_ids)
        pending = list(found)
        while pending:
            for group_id in self.holders.get(pending.pop(), ()):
                if group_id not in found:
                    found.add(group_id)
                    pending.append(group_id)
        return found

    def relay_reach(self, switch_id: str) -> tuple[set[str], set[str]]:
        """
        Return what reaches a relay: switches and groups, and other wrappers.

        The relay and the members of a switch group it is; every group holding
        any of them; and, failing closed as ``reach`` does, any entity in the
        relay's domain that is not set up or whose members cannot be read.
        Then every switch_as_x wrapper of any of those, and the groups holding
        a wrapper.
        """
        domain = switch_id.split(".", 1)[0]
        switches = self.climb(self.descend(switch_id))
        switches |= {
            entity_id
            for entity_id in self.unknown | self.unreadable_groups
            if entity_id.split(".", 1)[0] == domain
        }
        wrappers = {
            wrapper for held in switches for wrapper in self.wrappers_of.get(held, ())
        }
        return switches, self.climb(wrappers)

    def descend(self, entity_id: str) -> set[str]:
        """Return ``entity_id`` and every member of it, if it is a group."""
        found = {entity_id}
        pending = [entity_id]
        while pending:
            for member in self.members.get(pending.pop(), ()):
                if member not in found:
                    found.add(member)
                    pending.append(member)
        return found

    def _read(
        self, entity_id: str, entry: er.RegistryEntry | None, state: State | None
    ) -> None:
        """Add one entity; one that cannot be read is held, not the whole call."""
        try:
            self._add(entity_id, entry, state)
        except Exception:
            # Without this, one bad entity fails every guarded call in the
            # house. It is held instead: it could be anything.
            LOGGER.exception(
                "Critical Action PIN: could not read %s; it is held", entity_id
            )
            self.classes.setdefault(entity_id, set())
            self.names.setdefault(entity_id, [])
            self.unknown.add(entity_id)

    def _add(
        self, entity_id: str, entry: er.RegistryEntry | None, state: State | None
    ) -> None:
        attributes: Mapping[str, Any] = state.attributes if state is not None else {}
        reported = attributes.get(ATTR_DEVICE_CLASS)
        given: list[object] = [reported]
        if entry is not None:
            given += [entry.device_class, entry.original_device_class]
        self.classes[entity_id] = {c.lower() for c in given if isinstance(c, str)}
        # The integration's own class, not the user's "Show as" or YAML
        # `customize`: only it can say the entity is a window covering, so a
        # garage door relay shown as a blind keeps its name match. With no
        # registry entry it is read from the loaded entity; unknown, the name
        # is read.
        own = (
            entry.original_device_class
            if entry is not None
            else getattr(self._loaded_entity(entity_id), "device_class", None)
        )
        window_covering = isinstance(own, str) and own.lower() in _WINDOW_COVERINGS
        names: list[object] = [attributes.get(ATTR_FRIENDLY_NAME)]
        if entry is not None:
            names += [entry.name, entry.original_name, *entry.aliases]
            if entry.device_id and (device := self._devices.async_get(entry.device_id)):
                # An entity named by its device ("Garage Door") has no name
                # of its own in the registry.
                names += [device.name_by_user, device.name]
        self.names[entity_id] = (
            []
            if window_covering
            else [_plain(name) for name in names if isinstance(name, str)]
        )
        not_loaded = state is None or bool(attributes.get(ATTR_RESTORED))
        if (
            not_loaded
            and not window_covering
            and not (entry is not None and entry.disabled)
        ):
            # Not loaded (no state, or the placeholder Home Assistant restores
            # at startup), so its members cannot be seen: unless its
            # integration says it is a window covering, it could be a group
            # (an MQTT group classed shutter, say). A disabled entity cannot
            # run, so it is not held.
            self.unknown.add(entity_id)
        if entity_id.startswith(_NOT_GROUP_DOMAINS):
            return
        members = _text_list(attributes.get(ATTR_ENTITY_ID)) or _text_list(
            attributes.get(_ATTR_GROUP_ENTITIES)
        )
        if not members and self._platform(entity_id, entry) == _GROUP_PLATFORM:
            # An unavailable group helper writes no member list.
            members = self._helper_members(entry)
            if not members:
                self.unreadable_groups.add(entity_id)
        if members:
            self.members[entity_id] = members

    def _platform(self, entity_id: str, entry: er.RegistryEntry | None) -> str | None:
        """Return the integration of an entity, registered or only loaded."""
        if entry is not None:
            return entry.platform
        entity = self._loaded_entity(entity_id)
        platform = getattr(getattr(entity, "platform", None), "platform_name", None)
        return platform if isinstance(platform, str) else None

    def _loaded_entity(self, entity_id: str) -> object | None:
        """Return the entity object of a loaded entity with no registry entry."""
        # A YAML entity with no unique id has no registry entry. Its domain's
        # EntityComponent is stored in `hass.data` under the domain (the
        # component's DATA_COMPONENT key equals it) and holds the entity.
        component = self._hass.data.get(entity_id.split(".", 1)[0])
        get_entity = getattr(component, "get_entity", None)
        return get_entity(entity_id) if callable(get_entity) else None

    def _helper_members(self, entry: er.RegistryEntry | None) -> list[str]:
        """Return the members a group helper's config entry lists."""
        if entry is None or entry.config_entry_id is None:
            return []
        config_entry = self._hass.config_entries.async_get_entry(entry.config_entry_id)
        if config_entry is None:
            return []
        return _text_list(config_entry.options.get(CONF_ENTITIES))

    def reach(self, domain: str, match: str) -> set[str]:
        """Return every entity a rule on ``domain`` with this match reaches."""
        # A word with no letter next to it: "Gate1" and "Barn Doors" match,
        # "Gatekeeper" and "Indoor" do not.
        word = re.compile(rf"(?<![a-z]){re.escape(_plain(match))}(?:e?s)?(?![a-z])")
        # An id is read as the rule reads a direct call's target: the word
        # anywhere in it. `cover.backdoor` is held when opened directly, so a
        # group holding it is held too.

        def in_domain(entity_id: str) -> bool:
            return not domain or entity_id.split(".", 1)[0] == domain

        matched = {
            entity_id
            for entity_id in self.classes
            if in_domain(entity_id)
            and (
                match in entity_id
                or match in self.classes[entity_id]
                or any(word.search(name) for name in self.names[entity_id])
            )
        }
        # A member no state or registry entry describes (its integration not
        # set up yet) could be anything a group in the domain could hold.
        matched |= {
            member
            for member in self.holders
            if member not in self.classes and in_domain(member)
        }
        # A group helper whose members cannot be read could hold anything; an
        # entity not loaded (not disabled, and not a window covering by its
        # integration's class) could be such a group; and one that could not
        # be read could be anything.
        matched |= {
            entity_id
            for entity_id in self.unreadable_groups | self.unknown
            if in_domain(entity_id)
        }
        return self.climb(matched)


def relay_wrappers_reached(hass: HomeAssistant, targets: Iterable[str]) -> list[str]:
    """
    Return the cover, valve and lock wrappers a call on ``targets`` can move.

    Through the relay under one: the relay itself, a group holding it, a
    member of a switch group it wraps, or another wrapper of any of those.
    For the PIN prompt, which names what the PIN allows (#794).
    """
    wanted = {target.lower() for target in targets}
    if not wanted or not read_switch_wrappers(hass):
        return []
    index = _EntityIndex(hass)
    moved: list[str] = []
    for wrapper, (switch_id, _invert) in sorted(index.wrapped.items()):
        if wrapper.split(".", 1)[0] not in _OPENING_SERVICES:
            continue
        switches, wrappers = index.relay_reach(switch_id)
        if wanted & ((switches | wrappers) - {wrapper}):
            moved.append(wrapper)
    return moved


def _switch_rules(
    index: _EntityIndex, rule: Mapping[str, str], reached: set[str] | None
) -> list[dict[str, str]]:
    """
    Return copies of an opening ``rule`` for the relays under its entities.

    ``reached`` is what the rule's ``entity_match`` reaches; None for a rule
    with none, which covers every entity of its domain. A rule with no domain
    applies to each wrapper in the wrapper's own domain, as it does to calls.
    """
    rule_domain = (rule.get("domain") or "").lower()
    service = (rule.get("service") or "").lower()
    copies: list[dict[str, str]] = []
    for wrapper, (switch_id, invert) in sorted(index.wrapped.items()):
        domain = wrapper.split(".", 1)[0]
        opening = _OPENING_SERVICES.get(domain)
        if (
            opening is None
            or (rule_domain and rule_domain != domain)
            # A cover or valve relay moves either way, so any rule on its
            # wrapper holds it (one holding closing, too); a lock relay, only
            # an unlocking rule.
            or (service and domain not in _EVERY_RELAY_CALL and service not in opening)
            or (reached is not None and wrapper not in reached)
        ):
            continue
        if domain in _EVERY_RELAY_CALL or not service:
            relay_services = [""]
        else:
            relay_services = ["turn_off" if invert else "turn_on", "toggle"]
        switches, wrappers = index.relay_reach(switch_id)
        wrappers.discard(wrapper)
        # Another wrapper of the relay (a second cover, a light, one over a
        # group holding it) drives it too: every call on it is held.
        held = [
            (target, relay_service)
            for target in sorted(switches)
            for relay_service in relay_services
        ] + [(target, "") for target in sorted(wrappers)]
        copies.extend(
            {
                **rule,
                "domain": target.split(".", 1)[0],
                "service": relay_service,
                CRITICAL_RULE_EXACT_ENTITY: target,
            }
            for target, relay_service in held
        )
    return copies


def expand_critical_rules(
    hass: HomeAssistant | None, rules: Iterable[Mapping[str, str]]
) -> list[dict[str, str]]:
    """
    Return the rules plus a copy per entity they reach that their ids miss.

    One per entity an ``entity_match`` misses, and one per relay under a
    switch_as_x wrapper a rule holds (see the module docstring).
    """
    expanded = [dict(rule) for rule in rules]
    if hass is None or getattr(hass, "states", None) is None:
        if any(rule.get("entity_match") for rule in expanded):
            LOGGER.warning(
                "Critical Action PIN: no Home Assistant instance; covers, "
                "locks and the relays under them are matched by entity id only"
            )
        return expanded
    config_entries = getattr(hass, "config_entries", None)
    if not any(rule.get("entity_match") for rule in expanded) and not (
        config_entries and config_entries.async_entries(_SWITCH_AS_X)
    ):
        # Nothing for a rule with no entity_match to reach but relays.
        return expanded
    index = _EntityIndex(hass)
    reached: dict[tuple[str, str], set[str]] = {}
    extra: list[dict[str, str]] = []
    for rule in expanded:
        match = (rule.get("entity_match") or "").lower()
        if not match:
            extra.extend(_switch_rules(index, rule, None))
            continue
        key = ((rule.get("domain") or "").lower(), match)
        if key not in reached:
            reached[key] = index.reach(*key)
        extra.extend(
            {**rule, CRITICAL_RULE_EXACT_ENTITY: entity_id}
            for entity_id in sorted(reached[key])
            if match not in entity_id
        )
        extra.extend(_switch_rules(index, rule, reached[key]))
    # One copy per entity, domain and service: copies made from several rules
    # (`door` and `garage` both reaching "Garage Door") hold the same calls.
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for rule in extra:
        key = (
            rule[CRITICAL_RULE_EXACT_ENTITY],
            (rule.get("domain") or "").lower(),
            (rule.get("service") or "").lower(),
        )
        unique.setdefault(key, rule)
    return expanded + list(unique.values())
