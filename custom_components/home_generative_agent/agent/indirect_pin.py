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
  that applies to the entity counts, whatever its service. Unlike
  ``scene.apply`` in an automation, which can carry templated or unreadable
  entities and so holds on any rule for the domain, a stored scene names its
  entities, so a rule's ``entity_match`` is applied to each one. The members
  of an old-style group it sets are screened the same way. A script state in
  a scene does nothing (Home Assistant cannot reproduce one), so it is
  screened like any other entity.
* **A scene stored by another integration** cannot be read. It is held,
  except for the integrations in ``_LIGHT_SCENE_PLATFORMS``, whose scenes
  only set lights or shades. A hub's scene (Fibaro, Vera, Overkiz, MQTT, KNX)
  can open a gate or unlock a door.

Anything that cannot be read (a script or scene the gate cannot find, a
group nested deeper than ``_MAX_DEPTH``) is held.
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
_GENERIC_DOMAIN = "homeassistant"

# Where Home Assistant keeps the script entities (`script/__init__.py`) and the
# platform of the scenes in its own scene config (`homeassistant/scene.py`).
# Read by key rather than imported, so the integration does not depend on
# either component.
_SCRIPT_COMPONENT_KEY = SCRIPT_DOMAIN
_HA_SCENE_PLATFORM_KEY = "homeassistant_scene"

# Integrations whose scenes are stored on a lighting or shade system and can
# only set its lights or shades. Any other integration's scene is held.
_LIGHT_SCENE_PLATFORMS = frozenset(
    {
        "deconz",
        "hue",
        "hunterdouglas_powerview",
        "lifx_cloud",
        "litejet",
        "sunricher_dali",
    }
)

# Services that run a script or activate a scene, by domain. `script.<key>`
# runs the script with that key too; `scene.apply`/`scene.create` carry their
# states inline.
_RUN_SERVICES = frozenset({"turn_on", "toggle"})
_SCRIPT_SERVICES_THAT_DO_NOT_RUN = frozenset({"turn_off", "reload"})
_SCENE_INLINE_SERVICES = frozenset({"apply", "create"})
_INLINE_STATE_KEYS = ("entities", "snapshot_entities")

# A group can contain a group.
_MAX_DEPTH = 5

# The tool HA gives each script is `script__<object id>`: two parts.
_SCRIPT_TOOL_PARTS = 2

# Target states that name one service when a scene reproduces them. Any other
# state, or one with a position, leaves the service open: Home Assistant moves
# a cover to a position with `set_cover_position`, even to 0.
_REPRODUCE_SERVICES: dict[tuple[str, str], str] = {
    ("lock", "locked"): "lock",
    ("lock", "locking"): "lock",
    ("cover", "closed"): "close_cover",
    ("cover", "closing"): "close_cover",
    ("valve", "closed"): "close_valve",
    ("valve", "closing"): "close_valve",
}
_POSITION_ATTRIBUTES = (
    "current_position",
    "position",
    "current_tilt_position",
    "tilt_position",
)


def script_tool_entity_id(tool_name: str) -> str | None:
    """
    Return the script a tool runs, for Home Assistant's per-script tools.

    HA names each exposed script's tool ``script__<object id>``, and a merged
    API prefixes another namespace. Entity ids cannot hold ``__``, so the last
    two parts are the domain and the object id.
    """
    parts = tool_name.split(TOOL_NAMESPACE_SEP)
    if len(parts) < _SCRIPT_TOOL_PARTS or parts[-2] != SCRIPT_DOMAIN or not parts[-1]:
        return None
    return f"{SCRIPT_DOMAIN}.{parts[-1]}"


def script_rule_matches(
    hass: HomeAssistant | None,
    entity_id: str,
    rules: Sequence[Mapping[str, str]],
) -> bool:
    """
    Return True if a rule names running this script.

    A rule can name `script.turn_on` or the script's own service. That service
    is named by the script's key (its registry unique id), which stays the
    same when the entity is renamed, so both the key and the object id count.
    """
    services = {"turn_on", entity_id.split(".", 1)[1]}
    if hass is not None and (entry := er.async_get(hass).async_get(entity_id)):
        services.add(str(entry.unique_id).lower())
    return any(
        matches_critical_rule(
            domain=SCRIPT_DOMAIN,
            service=service,
            entity_ids=[entity_id],
            critical_actions=rules,
        )
        for service in services
    )


def runs_critical_action(
    hass: HomeAssistant | None,
    entity_id: str,
    rules: Sequence[Mapping[str, str]],
) -> bool:
    """Return True if running this script or scene could do a critical action."""
    if hass is None:
        return True
    entity_id = entity_id.lower()
    domain = entity_id.split(".", 1)[0]
    if domain == SCRIPT_DOMAIN:
        return _script_steps_are_critical(hass, entity_id, rules)
    if domain == SCENE_DOMAIN:
        return _scene_is_critical(hass, entity_id, rules)
    return False


def service_runs_critical_action(  # noqa: PLR0911
    hass: HomeAssistant | None,
    action: str,
    entity_ids: Sequence[str],
    data: Mapping[str, Any],
    rules: Sequence[Mapping[str, str]],
) -> bool:
    """
    Return True if a direct service call runs a script or scene that could.

    For tools that call services by name (an MCP service tool): `script.*`,
    `scene.*`, and the generic `homeassistant.*` aimed at a script or scene.
    A call that runs one with no entity id named is held. `action` is
    `<domain>.<service>`.
    """
    domain, _, service = action.partition(".")
    if domain == SCRIPT_DOMAIN:
        if service in _SCRIPT_SERVICES_THAT_DO_NOT_RUN:
            return False
        if service not in _RUN_SERVICES:
            # `script.<key>` runs the script with that key.
            return runs_critical_action(hass, _script_for_key(hass, service), rules)
        return _any_runs_critical(hass, entity_ids, rules)
    if domain == SCENE_DOMAIN:
        if service in _SCENE_INLINE_SERVICES:
            return _inline_states_are_critical(data, rules)
        if service == "turn_on":
            return _any_runs_critical(hass, entity_ids, rules)
        return False
    if domain == _GENERIC_DOMAIN and service in _RUN_SERVICES:
        return any(
            runs_critical_action(hass, entity_id, rules)
            for entity_id in entity_ids
            if entity_id.split(".", 1)[0] in (SCRIPT_DOMAIN, SCENE_DOMAIN)
        )
    return False


def _inline_states_are_critical(
    data: Mapping[str, Any], rules: Sequence[Mapping[str, str]]
) -> bool:
    """Screen `scene.apply`/`scene.create` the way an automation step is."""
    nested = data.get("data")
    fields: dict[str, Any] = (
        {**data, **nested} if isinstance(nested, dict) else {**data}
    )
    if not any(key in fields for key in _INLINE_STATE_KEYS):
        return True  # the states it sets are not where they can be read
    step: dict[str, Any] = {"action": f"{SCENE_DOMAIN}.apply", "data": fields}
    return bool(find_critical_automation_calls({"actions": [step]}, rules))


def _any_runs_critical(
    hass: HomeAssistant | None,
    entity_ids: Sequence[str],
    rules: Sequence[Mapping[str, str]],
) -> bool:
    if not entity_ids:
        return True  # an area, a device or nothing: which one is unknown
    return any(runs_critical_action(hass, entity_id, rules) for entity_id in entity_ids)


def _script_for_key(hass: HomeAssistant | None, key: str) -> str:
    """Return the entity id of the script with this key, renamed or not."""
    if hass is not None and (
        entity_id := er.async_get(hass).async_get_entity_id(
            SCRIPT_DOMAIN, SCRIPT_DOMAIN, key
        )
    ):
        return entity_id
    return f"{SCRIPT_DOMAIN}.{key}"


def _script_steps_are_critical(
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
    hass: HomeAssistant, entity_id: str, rules: Sequence[Mapping[str, str]]
) -> bool:
    platform = hass.data.get(_HA_SCENE_PLATFORM_KEY)
    entity = platform.entities.get(entity_id) if platform is not None else None
    states = getattr(getattr(entity, "scene_config", None), "states", None)
    if not isinstance(states, dict):
        entry = er.async_get(hass).async_get(entity_id)
        if entry is not None and entry.platform in _LIGHT_SCENE_PLATFORMS:
            return False
        LOGGER.debug("Cannot read the states %s sets; holding it", entity_id)
        return True
    seen: set[str] = set()
    return any(
        _state_is_critical(hass, state, rules, depth=0, seen=seen)
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
    if domain == _GROUP_DOMAIN:
        # A group's state is reproduced on each member.
        if depth >= _MAX_DEPTH:
            return True
        # A group reached twice with the same state needs no second look; with
        # another state (locked here, unlocked there) it does.
        key = f"{entity_id}|{state.state}|{sorted(state.attributes.items())!r}"
        if key in seen:
            return False
        seen.add(key)
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
    if any(attributes.get(key) is not None for key in _POSITION_ATTRIBUTES):
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
