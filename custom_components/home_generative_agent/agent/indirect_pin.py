"""
Critical-action screening for the scripts and scenes the agent runs (#775).

The PIN gate in ``graph.py`` matches the call the agent makes. Running a
script or activating a scene makes a call (``script.turn_on``,
``scene.turn_on``, or a script's own tool) that no rule names, while what
the script or scene then does can unlock a door. This module looks inside:

* **A script** is screened with ``automation_pin``, the same allowlist that
  screens an automation's actions before ``add_automation`` installs it, so
  both paths hold the same steps. A step that runs another script or scene,
  fires an event, or builds its service from a template is held there.
* **A scene** from Home Assistant's own scene config is screened per entity
  it sets. Where the state names one service (a lock set to ``locked`` is
  ``lock.lock``), that call is matched against the rules; otherwise any rule
  for the entity counts, as for ``scene.apply`` in an automation. A script it
  turns on and the members of an old-style group it sets are screened too.
  A scene from another integration (a Hue scene) is stored on the device and
  can only set that integration's lights, so it is not held.

Anything that cannot be read (a script or scene the gate cannot find, nesting
deeper than ``_MAX_DEPTH``) is held.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.core import State
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.group import get_entity_ids

from .automation_pin import find_critical_automation_calls
from .helpers import TOOL_NAMESPACE_SEP, matches_critical_rule

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

SCRIPT_DOMAIN = "script"
SCENE_DOMAIN = "scene"
_GROUP_DOMAIN = "group"

# Where Home Assistant keeps the script entities (`script/__init__.py`) and the
# platform of the scenes in its own scene config (`homeassistant/scene.py`).
# Read by key rather than imported, so the integration does not depend on
# either component.
_SCRIPT_COMPONENT_KEY = SCRIPT_DOMAIN
_HA_SCENE_PLATFORM_KEY = "homeassistant_scene"
_HA_SCENE_PLATFORM = "homeassistant"

# A scene can turn on a script that activates a scene, and so on.
_MAX_DEPTH = 5

# Target states that name one service when a scene reproduces them. Any other
# state, or one with a position, leaves the service open.
_REPRODUCE_SERVICES: dict[tuple[str, str], str] = {
    ("lock", "locked"): "lock",
    ("lock", "locking"): "lock",
    ("cover", "closed"): "close_cover",
    ("cover", "closing"): "close_cover",
    ("valve", "closed"): "close_valve",
    ("valve", "closing"): "close_valve",
}
_POSITION_ATTRIBUTES = ("current_position", "position", "current_tilt_position")


def script_tool_entity_id(tool_name: str) -> str | None:
    """
    Return the script a tool runs, for Home Assistant's per-script tools.

    HA names each exposed script's tool ``script__<object id>``, and a merged
    API prefixes another namespace. Entity ids cannot hold ``__``, so the last
    two parts are the domain and the object id.
    """
    parts = tool_name.split(TOOL_NAMESPACE_SEP)
    if len(parts) < 2 or parts[-2] != SCRIPT_DOMAIN or not parts[-1]:  # noqa: PLR2004
        return None
    return f"{SCRIPT_DOMAIN}.{parts[-1]}"


def runs_critical_action(
    hass: HomeAssistant | None,
    entity_id: str,
    rules: Sequence[Mapping[str, str]],
) -> bool:
    """Return True if running this script or scene could do a critical action."""
    return _is_critical(hass, entity_id.lower(), rules, depth=0, seen=set())


def _is_critical(
    hass: HomeAssistant | None,
    entity_id: str,
    rules: Sequence[Mapping[str, str]],
    *,
    depth: int,
    seen: set[str],
) -> bool:
    if hass is None or depth > _MAX_DEPTH:
        return True
    if entity_id in seen:
        # Already being screened higher up; its own result decides.
        return False
    seen.add(entity_id)
    domain = entity_id.split(".", 1)[0]
    if domain == SCRIPT_DOMAIN:
        return _script_is_critical(hass, entity_id, rules)
    if domain == SCENE_DOMAIN:
        return _scene_is_critical(hass, entity_id, rules, depth=depth, seen=seen)
    return False


def _script_is_critical(
    hass: HomeAssistant, entity_id: str, rules: Sequence[Mapping[str, str]]
) -> bool:
    component = hass.data.get(_SCRIPT_COMPONENT_KEY)
    entity = component.get_entity(entity_id) if component is not None else None
    sequence = getattr(getattr(entity, "script", None), "sequence", None)
    if not isinstance(sequence, list):
        LOGGER.debug("Cannot read the steps of %s; holding it", entity_id)
        return True
    found = find_critical_automation_calls({"actions": sequence}, rules)
    if found:
        LOGGER.debug(
            "%s runs %s", entity_id, "; ".join(call.describe() for call in found)
        )
    return bool(found)


def _scene_is_critical(
    hass: HomeAssistant,
    entity_id: str,
    rules: Sequence[Mapping[str, str]],
    *,
    depth: int,
    seen: set[str],
) -> bool:
    platform = hass.data.get(_HA_SCENE_PLATFORM_KEY)
    entity = platform.entities.get(entity_id) if platform is not None else None
    states = getattr(getattr(entity, "scene_config", None), "states", None)
    if not isinstance(states, dict):
        entry = er.async_get(hass).async_get(entity_id)
        if entry is not None and entry.platform != _HA_SCENE_PLATFORM:
            return False
        LOGGER.debug("Cannot read the states %s sets; holding it", entity_id)
        return True
    return any(
        _state_is_critical(hass, state, rules, depth=depth, seen=seen)
        for state in states.values()
    )


def _state_is_critical(
    hass: HomeAssistant,
    state: State,
    rules: Sequence[Mapping[str, str]],
    *,
    depth: int,
    seen: set[str],
) -> bool:
    """Return True if reproducing one scene state could do a critical action."""
    entity_id = state.entity_id.lower()
    domain = entity_id.split(".", 1)[0]
    if domain == SCRIPT_DOMAIN:
        # Reproducing "on" runs the script.
        return state.state == "on" and _is_critical(
            hass, entity_id, rules, depth=depth + 1, seen=seen
        )
    if domain == _GROUP_DOMAIN:
        # A group's state is reproduced on each member.
        if depth + 1 > _MAX_DEPTH:
            return True
        return any(
            _state_is_critical(
                hass,
                State(member, state.state, state.attributes),
                rules,
                depth=depth + 1,
                seen=seen,
            )
            for member in get_entity_ids(hass, entity_id)
        )
    service = _reproduce_service(domain, state)
    if service is not None:
        return matches_critical_rule(
            domain=domain,
            service=service,
            entity_ids=[entity_id],
            critical_actions=rules,
        )
    return _entity_is_guarded(domain, entity_id, rules)


def _reproduce_service(domain: str, state: State) -> str | None:
    """Return the one service reproducing this state calls, when there is one."""
    service = _REPRODUCE_SERVICES.get((domain, state.state))
    if service is None:
        return None
    attributes: Mapping[str, Any] = state.attributes
    if any(attributes.get(key) not in (None, 0) for key in _POSITION_ATTRIBUTES):
        return None
    return service


def _entity_is_guarded(
    domain: str, entity_id: str, rules: Sequence[Mapping[str, str]]
) -> bool:
    """Return True if any rule applies to this entity, whatever the service."""
    for rule in rules:
        rule_domain = (rule.get("domain") or "").lower()
        entity_match = (rule.get("entity_match") or "").lower()
        if rule_domain and rule_domain != domain:
            continue
        if entity_match and entity_match not in entity_id:
            continue
        return True
    return False
