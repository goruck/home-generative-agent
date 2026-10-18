"""
Extend ``entity_match`` rules to the entities a name match misses (#773).

A rule's ``entity_match`` is matched against the entity ids a call targets.
Two kinds of entity act on a matching one without carrying the word in their
own id:

* **An entity whose device class is the word.** A cover whose device class is
  ``garage`` but whose id is ``cover.msg100_abc`` opens a garage door.
* **A group holding a matching entity.** Opening ``cover.all_covers``, or a
  call on an old-style ``group.*``, acts on every member, the garage door
  included. Groups of groups are followed to any depth.

``expand_critical_rules`` adds, for each such entity, a copy of the rule whose
``entity_match`` is that entity's id. Every screen downstream (the tool-call
gate, intents, scripts, scenes, ``add_automation``) matches on entity ids, so
each one holds the call without knowing why the entity counts. An id is
matched as a substring, as the user's own rules are, so ``cover.gate`` also
holds ``cover.gate_2``: more prompts, never fewer.

A group's members are read from its state's ``entity_id`` attribute, which
cover, light and old-style groups all set. A group Home Assistant has not
loaded has no members to read, but then it cannot open anything either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from homeassistant.const import ATTR_DEVICE_CLASS, ATTR_ENTITY_ID
from homeassistant.helpers import entity_registry as er

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from homeassistant.core import HomeAssistant, State


def _device_class(state: State, registry: er.EntityRegistry) -> str:
    """Return the entity's device class, from its state or else its registry."""
    value = state.attributes.get(ATTR_DEVICE_CLASS)
    if not isinstance(value, str) and (entry := registry.async_get(state.entity_id)):
        value = entry.device_class or entry.original_device_class
    return value.lower() if isinstance(value, str) else ""


def _members(state: State) -> list[str]:
    """Return the entity ids a group's state lists, or none."""
    members: object = state.attributes.get(ATTR_ENTITY_ID)
    if not isinstance(members, (list, tuple)):
        return []
    items = cast("Sequence[object]", members)
    return [m.lower() for m in items if isinstance(m, str)]


def _matching_entities(
    states: Iterable[State],
    registry: er.EntityRegistry,
    domain: str,
    match: str,
) -> set[str]:
    """Return every entity a rule on ``domain`` with this ``entity_match`` reaches."""
    matched: set[str] = set()
    groups: dict[str, list[str]] = {}
    for state in states:
        entity_id = state.entity_id.lower()
        if (members := _members(state)) and not entity_id.startswith(
            ("script.", "scene.")
        ):
            groups[entity_id] = members
        in_domain = not domain or entity_id.split(".", 1)[0] == domain
        if in_domain and (
            match in entity_id or match in _device_class(state, registry)
        ):
            matched.add(entity_id)
    # A group reaches what its members reach; repeat until no group is added.
    added = True
    while added:
        added = False
        for group_id, members in groups.items():
            if group_id not in matched and any(m in matched for m in members):
                matched.add(group_id)
                added = True
    return matched


def expand_critical_rules(
    hass: HomeAssistant | None, rules: Iterable[Mapping[str, str]]
) -> list[dict[str, str]]:
    """Return the rules plus one per entity an ``entity_match`` misses."""
    expanded = [dict(rule) for rule in rules]
    if hass is None or getattr(hass, "states", None) is None:
        return expanded
    states = hass.states.async_all()
    registry = er.async_get(hass)
    reached: dict[tuple[str, str], set[str]] = {}
    extra: list[dict[str, str]] = []
    for rule in expanded:
        match = (rule.get("entity_match") or "").lower()
        if not match:
            continue
        domain = (rule.get("domain") or "").lower()
        key = (domain, match)
        if key not in reached:
            reached[key] = _matching_entities(states, registry, domain, match)
        extra.extend(
            {**rule, "entity_match": entity_id}
            for entity_id in sorted(reached[key])
            if match not in entity_id
        )
    return expanded + extra
