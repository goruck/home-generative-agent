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
  only set lights. A hub's scene (Fibaro, Vera, Overkiz, MQTT, KNX,
  deCONZ, which can hold a lock) can open a gate or unlock a door.

Anything that cannot be read (a script or scene the gate cannot find, a
group nested deeper than ``_MAX_DEPTH``) is held.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from homeassistant.core import State
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.group import get_entity_ids

from .automation_pin import (
    DATA_CONTAINER_KEYS,
    STEP_TARGET_KEYS,
    find_critical_automation_calls,
    step_targets,
)
from .helpers import TOOL_NAMESPACE_SEP, matches_critical_rule, rule_entity_matches

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

# Integrations whose scenes are stored on a lighting system and can only set
# its lights. Any other integration's scene is held: a shade system's scene
# (Hunter Douglas PowerView) moves covers, which a rule can guard.
_LIGHT_SCENE_PLATFORMS = frozenset(
    {
        "hue",
        "lifx_cloud",
        "litejet",
        "sunricher_dali",
    }
)

# Services that run a script or activate a scene, by domain. `script.<key>`
# runs the script with that key too; `scene.apply`/`scene.create` carry their
# states inline.
RUN_SERVICES = frozenset({"turn_on", "toggle"})
_SCRIPT_SERVICES_THAT_DO_NOT_RUN = frozenset({"turn_off", "reload"})
_SCENE_INLINE_SERVICES = frozenset({"apply", "create"})
_INLINE_STATE_KEYS = ("entities", "snapshot_entities")
# Tool arguments that name the call rather than carry its fields.
_CALL_KEYS = frozenset({"domain", "service", "action", "entity_ids"})

# A group can contain a group.
_MAX_DEPTH = 5

# The tool HA gives each script is `script__<object id>`: two parts.
_SCRIPT_TOOL_PARTS = 2
_OBJECT_ID = re.compile(r"[a-z0-9_]+")

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
    # A relay under a garage lock: off is the locking way unless inverted, so
    # the rules decide by the service (#794).
    ("switch", "on"): "turn_on",
    ("switch", "off"): "turn_off",
}
# `CoverEntityFeature.CLOSE` and `ValveEntityFeature.CLOSE` are both 2.
_CLOSE_FEATURE = 2
_CLOSE_FEATURE_DOMAINS = frozenset({"cover", "valve"})
_POSITION_ATTRIBUTES = (
    "current_position",
    "position",
    "current_tilt_position",
    "tilt_position",
)


def script_tool_entity_id(tool_name: str) -> str | None:
    """
    Return the script a tool runs, for Home Assistant's per-script tools.

    HA 2026.9 and later name each exposed script's tool ``script__<object
    id>``, and a merged API prefixes another namespace. Entity ids cannot hold
    ``__``, so the last two parts are the domain and the object id.
    """
    parts = tool_name.split(TOOL_NAMESPACE_SEP)
    if len(parts) < _SCRIPT_TOOL_PARTS or parts[-2] != SCRIPT_DOMAIN or not parts[-1]:
        return None
    return f"{SCRIPT_DOMAIN}.{parts[-1]}"


def bare_script_tool_entity_id(
    tool_name: str, hass: HomeAssistant | None
) -> str | None:
    """
    Return the script a tool named by the script alone may run.

    Before 2026.9, HA named a script's tool by its object id (with ``_`` before
    one that starts with a digit). A tool named like an existing script may be
    that script's tool, or another tool that only shares its name, so the
    caller screens it both ways rather than taking it for the script.
    """
    states = getattr(hass, "states", None)
    if states is None:
        return None
    object_id = tool_name.rsplit(TOOL_NAMESPACE_SEP, maxsplit=1)[-1]
    # An object id is a lowercase slug, unlike an intent's tool (HassTurnOn).
    if not _OBJECT_ID.fullmatch(object_id):
        return None
    if object_id.startswith("_") and object_id[1:2].isdigit():
        object_id = object_id[1:]
    entity_id = f"{SCRIPT_DOMAIN}.{object_id}"
    return entity_id if states.get(entity_id) is not None else None


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
    tool_args: Mapping[str, Any],
    rules: Sequence[Mapping[str, str]],
) -> bool:
    """
    Return True if a direct service call could do a critical action.

    For tools that call services by name (an MCP service tool). The call is
    screened like one automation step (`automation_pin`), so its targets are
    read the way Home Assistant reads them: `all`, a registry id, a group, an
    area, a device or a label is unresolved and held, and a generic
    `homeassistant.*` call is checked against each target's own domain. A
    script or scene it runs is opened rather than held wholesale. `action` is
    `<domain>.<service>`.
    """
    if _fields_disagree(tool_args):
        return True  # which copy HA reads is up to the tool
    domain, _, service = action.partition(".")
    step = _service_step(action, tool_args)
    if domain == SCRIPT_DOMAIN:
        if service in _SCRIPT_SERVICES_THAT_DO_NOT_RUN:
            return False
        if service not in RUN_SERVICES:
            # `script.<key>` runs the script with that key.
            return _opened_is_critical(hass, _script_for_key(hass, service), rules)
        return _targets_run_critical(hass, step, frozenset({SCRIPT_DOMAIN}), rules)
    if domain == SCENE_DOMAIN:
        if service in _SCENE_INLINE_SERVICES:
            return _inline_states_are_critical(tool_args, rules)
        if service == "turn_on":
            return _targets_run_critical(hass, step, frozenset({SCENE_DOMAIN}), rules)
        return False
    if domain == _GENERIC_DOMAIN and service in RUN_SERVICES:
        return _targets_run_critical(
            hass, step, frozenset({SCRIPT_DOMAIN, SCENE_DOMAIN}), rules
        )
    return bool(find_critical_automation_calls({"actions": [step]}, rules))


def call_target_domains(tool_args: Mapping[str, Any]) -> tuple[set[str], bool]:
    """
    Return the domains a service tool's call targets, and if any is unknown.

    Read wherever the targets sit (`entity_id`, `entity_ids`, `target`,
    `data`), the way the step screen reads them. A target of `all`, an area,
    a device, a label or a registry id is unknown.
    """
    entities, unresolved = step_targets(
        _service_step(f"{_GENERIC_DOMAIN}.turn_on", tool_args),
        frozenset({SCRIPT_DOMAIN, SCENE_DOMAIN}),
    )
    return {entity_id.split(".", 1)[0] for entity_id in entities}, unresolved


def _service_step(action: str, tool_args: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return a service tool's call as an automation step.

    Only target keys stay at the top: any other argument goes under `data`,
    where it is a service field. Left at the top, a field named like a script
    action (`variables`, `delay`) would make the step read as that action.
    """
    step: dict[str, Any] = {"action": action}
    fields: dict[str, Any] = {}
    for key, value in tool_args.items():
        if key in _CALL_KEYS:
            continue
        if key in STEP_TARGET_KEYS:
            step[key] = value
        else:
            fields[key] = value
    if fields:
        nested = step.get("data")
        if nested is None:
            step["data"] = fields
        elif isinstance(nested, dict):
            step["data"] = {**fields, **nested}
        # Any other `data` stays as it is, for the screen to call unreadable.
    if (plural := tool_args.get("entity_ids")) is not None:
        named = step.get("entity_id")
        step["entity_id"] = [
            *(named if isinstance(named, list) else [named] if named else []),
            *(plural if isinstance(plural, list) else [plural]),
        ]
    return step


def _fields_disagree(tool_args: Mapping[str, Any]) -> bool:
    """
    Return True if a field is given twice with two values.

    At the top of the call and inside `data` (or `service_data`,
    `data_template`), or in two of those: the screen reads one copy, and the
    tool may send the other.
    """
    seen: dict[str, Any] = {}
    top = {
        key: value
        for key, value in tool_args.items()
        if key not in _CALL_KEYS and key not in DATA_CONTAINER_KEYS
    }
    nested = [
        value
        for key in DATA_CONTAINER_KEYS
        if isinstance(value := tool_args.get(key), dict)
    ]
    for fields in (top, *nested):
        for key, raw in fields.items():
            # HA reads one value and a list of that one value alike.
            value = raw[0] if isinstance(raw, list) and len(raw) == 1 else raw
            if key in seen and seen[key] != value:
                return True
            seen[key] = value
    return False


def _targets_run_critical(
    hass: HomeAssistant | None,
    step: Mapping[str, Any],
    opened: frozenset[str],
    rules: Sequence[Mapping[str, str]],
) -> bool:
    """Open each script or scene a call targets; screen any other target."""
    entities, unresolved = step_targets(step, opened)
    if unresolved:
        return True  # `all`, a group, an area or an id: which ones is unknown
    others = [e for e in entities if e.split(".", 1)[0] not in opened]
    if others and find_critical_automation_calls(
        {"actions": [{"action": step["action"], "entity_id": others}]}, rules
    ):
        return True
    return any(
        _opened_is_critical(hass, entity_id, rules)
        for entity_id in entities
        if entity_id.split(".", 1)[0] in opened
    )


def _opened_is_critical(
    hass: HomeAssistant | None,
    entity_id: str,
    rules: Sequence[Mapping[str, str]],
) -> bool:
    """Return True if a rule names running it, or what it does is critical."""
    if entity_id.split(".", 1)[0] == SCRIPT_DOMAIN:
        named = script_rule_matches(hass, entity_id, rules)
    else:
        named = matches_critical_rule(
            domain=SCENE_DOMAIN,
            service="turn_on",
            entity_ids=[entity_id],
            critical_actions=rules,
        )
    return named or runs_critical_action(hass, entity_id, rules)


def _inline_states_are_critical(
    data: Mapping[str, Any], rules: Sequence[Mapping[str, str]]
) -> bool:
    """Screen `scene.apply`/`scene.create` the way an automation step is."""
    fields: dict[str, Any] = {
        key: value
        for key, value in data.items()
        if key not in _CALL_KEYS and key not in DATA_CONTAINER_KEYS
    }
    for key in DATA_CONTAINER_KEYS:
        if key not in data:
            continue
        nested = data[key]
        if not isinstance(nested, dict):
            return True  # fields the screen cannot read, such as JSON text
        fields.update(nested)
    if not any(key in fields for key in _INLINE_STATE_KEYS):
        return True  # the states it sets are not where they can be read
    step: dict[str, Any] = {"action": f"{SCENE_DOMAIN}.apply", "data": fields}
    return bool(find_critical_automation_calls({"actions": [step]}, rules))


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
        if _scene_platform(hass, entity_id) in _LIGHT_SCENE_PLATFORMS:
            return False
        LOGGER.debug("Cannot read the states %s sets; holding it", entity_id)
        return True
    seen: set[str] = set()
    return any(
        _state_is_critical(hass, state, rules, depth=0, seen=seen)
        for state in states.values()
    )


def _scene_platform(hass: HomeAssistant, entity_id: str) -> str | None:
    """
    Return the integration a scene comes from.

    From the loaded entity, since a scene with no unique id (LIFX Cloud's)
    has no registry entry; else from the registry.
    """
    component = hass.data.get(SCENE_DOMAIN)
    entity = component.get_entity(entity_id) if component is not None else None
    platform = getattr(getattr(entity, "platform", None), "platform_name", None)
    if isinstance(platform, str):
        return platform
    entry = er.async_get(hass).async_get(entity_id)
    return entry.platform if entry is not None else None


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
    service = _reproduce_service(hass, domain, state)
    if service is not None:
        return matches_critical_rule(
            domain=domain,
            service=service,
            entity_ids=[entity_id],
            critical_actions=rules,
        )
    return _entity_is_guarded(domain, entity_id, rules)


def _reproduce_service(hass: HomeAssistant, domain: str, state: State) -> str | None:
    """Return the one service reproducing this state calls, when there is one."""
    service = _REPRODUCE_SERVICES.get((domain, state.state))
    if service is None:
        return None
    attributes: Mapping[str, Any] = state.attributes
    if any(attributes.get(key) is not None for key in _POSITION_ATTRIBUTES):
        return None
    if domain in _CLOSE_FEATURE_DOMAINS:
        # A cover or valve that cannot close is moved to position 0 instead
        # (`cover/reproduce_state.py`), so only one that can close is known.
        current = hass.states.get(state.entity_id)
        features = current.attributes.get("supported_features") if current else None
        if not isinstance(features, int) or not features & _CLOSE_FEATURE:
            return None
    return service


def _entity_is_guarded(
    domain: str, entity_id: str, rules: Sequence[Mapping[str, str]]
) -> bool:
    """Return True if any rule applies to this entity, whatever the service."""
    for rule in rules:
        rule_domain = (rule.get("domain") or "").lower()
        if rule_domain and rule_domain != domain:
            continue
        if not rule_entity_matches(rule, [entity_id]):
            continue
        return True
    return False
