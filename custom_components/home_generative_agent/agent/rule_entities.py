"""
Extend ``entity_match`` rules to the entities a name match misses (#773).

A rule's ``entity_match`` is matched against the entity ids a call targets.
Three kinds of entity act like a matching one without carrying the word in
their own id:

* **An entity whose device class is the word.** A cover whose device class is
  ``garage`` but whose id is ``cover.msg100_abc`` opens a garage door. The
  device class must equal the word, so a rule for ``door`` does not reach
  device class ``garage``.
* **An entity named with the word.** A cover with no device class, id
  ``cover.shelly1_abc`` and name "Garage Door", which "open the garage door"
  finds by its name. The word, or its plural, must have no letter next to it
  in the friendly name, the registry or integration name, an alias, or the
  device's name: "Driveway Gates" and "Gate1" match ``gate``, "Indoor Blinds"
  does not match ``door``. A cover whose device class is a window covering is
  not matched by name ("French Door Blinds"). A word written like an id
  (``garage_door``) matches "Garage Door". Compound words ("GarageDoor")
  and names in other languages are not read; device class or a rule naming
  the id covers those.
* **A group holding a matching entity.** Opening ``cover.all_covers``, or a
  call on an old-style ``group.*``, acts on every member, the garage door
  included. Groups of groups are followed to any depth. A member matches by
  the ways above, or by the word starting its id or following ``_`` or
  ``.`` in it (``cover.garage_door``, not ``cover.outdoor_awning``). A member
  Home Assistant has not set up or registered yet could be anything, so its
  groups are held.

``expand_critical_rules`` adds, for each such entity, a copy of the rule that
matches that entity id exactly (``CRITICAL_RULE_EXACT_ENTITY``). Every screen
downstream (the tool-call gate, intents, scripts, scenes, ``add_automation``)
matches on entity ids through ``rule_entity_matches``, so each one holds the
call without knowing why the entity counts, and ``cover.a`` being a garage
door does not hold ``cover.awning``.

Every check counts entities that are not loaded or not available, because an
automation the agent writes runs later. An entity only in the entity registry
is read from it. One not set up yet (no state, or the placeholder Home
Assistant restores at startup) in the rule's domain with no device class is
held, since it could be a group whose members cannot be seen yet; a disabled
one is not, since it cannot run. A group's members
are read from its state (``entity_id``, or ``group_entities`` for MQTT and
ZHA groups, which Home Assistant writes even while the group is unavailable),
else from the group helper's config entry. A group helper whose members
cannot be read is held. Membership and device class are read when the check
runs: a member added to a group after an automation is written is not
checked again.
"""

from __future__ import annotations

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
    """Every entity's names, device class and group membership, read once."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass
        registry = er.async_get(hass)
        self._devices = dr.async_get(hass)
        self.device_class: dict[str, str] = {}
        self.names: dict[str, list[str]] = {}
        self.members: dict[str, list[str]] = {}
        self.unreadable_groups: set[str] = set()
        self.unknown: set[str] = set()
        for state in hass.states.async_all():
            entity_id = state.entity_id.lower()
            self._add(entity_id, registry.async_get(entity_id), state)
        for entry in registry.entities.values():
            if entry.entity_id not in self.device_class:
                self._add(entry.entity_id, entry, None)
        # Member -> the groups holding it, so a match climbs to every group
        # holding it in one pass, however deep the nesting.
        self.holders: dict[str, set[str]] = {}
        for group_id, group_members in self.members.items():
            for member in group_members:
                self.holders.setdefault(member, set()).add(group_id)

    def _add(
        self, entity_id: str, entry: er.RegistryEntry | None, state: State | None
    ) -> None:
        attributes: Mapping[str, Any] = state.attributes if state is not None else {}
        device_class = attributes.get(ATTR_DEVICE_CLASS)
        if not isinstance(device_class, str) and entry is not None:
            device_class = entry.device_class or entry.original_device_class
        self.device_class[entity_id] = (
            device_class.lower() if isinstance(device_class, str) else ""
        )
        names: list[object] = [attributes.get(ATTR_FRIENDLY_NAME)]
        if entry is not None:
            names += [entry.name, entry.original_name, *entry.aliases]
            if entry.device_id and (device := self._devices.async_get(entry.device_id)):
                # An entity named by its device ("Garage Door") has no name
                # of its own in the registry.
                names += [device.name_by_user, device.name]
        self.names[entity_id] = (
            []
            if self.device_class[entity_id] in _WINDOW_COVERINGS
            else [_plain(name) for name in names if isinstance(name, str)]
        )
        not_loaded = state is None or bool(attributes.get(ATTR_RESTORED))
        if (
            not_loaded
            and not self.device_class[entity_id]
            and not (entry is not None and entry.disabled)
        ):
            # Not loaded (no state, or the placeholder Home Assistant restores
            # at startup) and nothing says what it is: it could be a group. A
            # disabled entity cannot run, so it is not held.
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
        # A YAML entity with no unique id has no registry entry. Its domain's
        # EntityComponent is stored in `hass.data` under the domain (the
        # component's DATA_COMPONENT key equals it) and knows its platform.
        component = self._hass.data.get(entity_id.split(".", 1)[0])
        get_entity = getattr(component, "get_entity", None)
        entity = get_entity(entity_id) if callable(get_entity) else None
        platform = getattr(getattr(entity, "platform", None), "platform_name", None)
        return platform if isinstance(platform, str) else None

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
        # The word starting an id or a part of it, not inside a longer word.
        in_id = re.compile(rf"(?<![a-z0-9]){re.escape(match)}")

        def in_domain(entity_id: str) -> bool:
            return not domain or entity_id.split(".", 1)[0] == domain

        matched = {
            entity_id
            for entity_id, device_class in self.device_class.items()
            if in_domain(entity_id)
            and (
                in_id.search(entity_id)
                or match == device_class
                or any(word.search(name) for name in self.names[entity_id])
            )
        }
        # A member no state or registry entry describes (its integration not
        # set up yet) could be anything a group in the domain could hold.
        matched |= {
            member
            for member in self.holders
            if member not in self.device_class and in_domain(member)
        }
        # A group helper whose members cannot be read could hold anything, and
        # an entity not loaded with no device class could be such a group.
        matched |= {
            entity_id
            for entity_id in self.unreadable_groups | self.unknown
            if in_domain(entity_id)
        }
        pending = list(matched)
        while pending:
            for group_id in self.holders.get(pending.pop(), ()):
                if group_id not in matched:
                    matched.add(group_id)
                    pending.append(group_id)
        return matched


def expand_critical_rules(
    hass: HomeAssistant | None, rules: Iterable[Mapping[str, str]]
) -> list[dict[str, str]]:
    """Return the rules plus one per entity an ``entity_match`` misses."""
    expanded = [dict(rule) for rule in rules]
    if hass is None or getattr(hass, "states", None) is None:
        return expanded
    if not any(rule.get("entity_match") for rule in expanded):
        return expanded
    index = _EntityIndex(hass)
    reached: dict[tuple[str, str], set[str]] = {}
    extra: list[dict[str, str]] = []
    for rule in expanded:
        match = (rule.get("entity_match") or "").lower()
        if not match:
            continue
        key = ((rule.get("domain") or "").lower(), match)
        if key not in reached:
            reached[key] = index.reach(*key)
        extra.extend(
            {**rule, CRITICAL_RULE_EXACT_ENTITY: entity_id}
            for entity_id in sorted(reached[key])
            if match not in entity_id
        )
    return expanded + extra
