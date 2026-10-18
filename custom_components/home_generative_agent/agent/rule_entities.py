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
  finds by its name. The word must stand alone in the friendly name, the
  registry name or an alias, as ``is_sensitive_entity`` in the network audit
  reads it, so "Garage Door" matches ``garage`` and "Indoor Blinds" does not
  match ``door``.
* **A group holding a matching entity.** Opening ``cover.all_covers``, or a
  call on an old-style ``group.*``, acts on every member, the garage door
  included. Groups of groups are followed to any depth.

``expand_critical_rules`` adds, for each such entity, a copy of the rule whose
``entity_match`` is that entity's id. Every screen downstream (the tool-call
gate, intents, scripts, scenes, ``add_automation``) matches on entity ids, so
each one holds the call without knowing why the entity counts. An id is
matched as a substring, as the user's own rules are, so ``cover.gate`` also
holds ``cover.gate_2``: more prompts, never fewer.

``add_automation`` writes an automation that runs later, so an entity that is
not loaded or not available now still counts. An entity only in the entity
registry is read from it. A group's members are read from its state
(``entity_id``, or ``group_entities`` for MQTT and ZHA groups), else from the
group helper's config entry. A group helper whose members cannot be read is
held, whatever it holds. Membership and device class are read when the check
runs: a member added to a group after an automation is written is not checked
again.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast

from homeassistant.const import ATTR_DEVICE_CLASS, ATTR_ENTITY_ID, ATTR_FRIENDLY_NAME
from homeassistant.helpers import entity_registry as er

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from homeassistant.core import HomeAssistant, State

# MQTT and ZHA group entities list their members here, even when unavailable.
_ATTR_GROUP_ENTITIES = "group_entities"
# The integration whose entities are group helpers (cover, light, ... groups).
_GROUP_PLATFORM = "group"
# Scenes and scripts list the entities they set in `entity_id` too, but they
# are opened and screened by `indirect_pin`, not followed as groups here.
_NOT_GROUP_DOMAINS = ("script.", "scene.")


def _text_list(value: object) -> list[str]:
    """Return the strings in a list attribute, lowercased, or none."""
    if not isinstance(value, (list, tuple, set)):
        return []
    items = cast("Iterable[object]", value)
    return [item.lower() for item in items if isinstance(item, str)]


class _EntityIndex:
    """Every entity's names, device class and group membership, read once."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass
        registry = er.async_get(hass)
        self.device_class: dict[str, str] = {}
        self.names: dict[str, list[str]] = {}
        self.members: dict[str, list[str]] = {}
        self.unreadable_groups: set[str] = set()
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
        self.names[entity_id] = [
            name.lower().replace("_", " ") for name in names if isinstance(name, str)
        ]
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
        return _text_list(config_entry.options.get("entities"))

    def reach(self, domain: str, match: str) -> set[str]:
        """Return every entity a rule on ``domain`` with this match reaches."""
        word = re.compile(rf"\b{re.escape(match)}\b")

        def in_domain(entity_id: str) -> bool:
            return not domain or entity_id.split(".", 1)[0] == domain

        matched = {
            entity_id
            for entity_id, device_class in self.device_class.items()
            if in_domain(entity_id)
            and (
                match in entity_id
                or match == device_class
                or any(word.search(name) for name in self.names[entity_id])
            )
        }
        # A group helper whose members cannot be read could hold anything.
        matched |= {g for g in self.unreadable_groups if in_domain(g)}
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
            {**rule, "entity_match": entity_id}
            for entity_id in sorted(reached[key])
            if match not in entity_id
        )
    return expanded + extra
