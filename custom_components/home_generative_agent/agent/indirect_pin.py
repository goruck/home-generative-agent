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

* **A template entity or an input button** runs stored actions of its own
  (``template_actions``): a template switch's ``turn_on``, a template
  button's press, the automations an input button starts (#780). Wherever a
  call, a script step or a scene state reaches one by entity id, those
  actions are screened like a script's steps.

Anything that cannot be read (a script or scene the gate cannot find, a
group nested deeper than ``_MAX_DEPTH``, the actions of a template entity
that is not loaded) is held.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from homeassistant.core import State
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import intent, llm
from homeassistant.helpers.group import get_entity_ids
from slugify import slugify

from .automation_pin import (
    DATA_CONTAINER_KEYS,
    STEP_TARGET_KEYS,
    CriticalAutomationCall,
    find_critical_automation_calls,
    iter_service_calls,
    step_targets,
)
from .helpers import (
    BUILTIN_INTENT,
    NOT_AN_INTENT,
    OTHER_INTENT,
    REMOTE_TOOL,
    TOOL_NAMESPACE_SEP,
    base_tool_name,
    is_intent_tool,
    matches_critical_rule,
    rule_entity_matches,
)
from .rule_entities import read_switch_wrappers
from .template_actions import (
    INLINE_STATE_KEYS,
    PRESS_DOMAINS,
    Reactor,
    StoredActionReader,
    call_targets,
    entities_of,
    inline_states,
    with_members,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

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
# Tool arguments that name the call rather than carry its fields.
_CALL_KEYS = frozenset({"domain", "service", "action", "entity_ids"})

# The tool HA gives each intent script is `intent_script__<name>` (#779).
_INTENT_SCRIPT = "intent_script"
_SCRIPT_INTENT_HANDLER = "ScriptIntentHandler"
# An intent script tool whose intent script is gone.
_MISSING = object()
# A group can contain a group.
_MAX_DEPTH = 5
# Entities one check follows through stored actions and group members before
# it holds rather than read on (#780).
_MAX_FOLLOWED_ENTITIES = 2000
# Changes that name no entity: they mark a tracker home, or set off MQTT
# triggers (#814).
_ENTITYLESS_CHANGES = frozenset({("device_tracker", "see"), ("mqtt", "publish")})
_MQTT_DOMAIN = "mqtt"
_AUTOMATION_DOMAIN = "automation"
# Entities followed one inside another (a change starting an automation whose
# step changes the next entity) before a check holds rather than read deeper.
_MAX_CHAIN = 25
# The automations a check finds would perform a critical action, for the PIN
# line (see `collect_started_automations`).
_STARTED: ContextVar[list[str] | None] = ContextVar("hga_pin_started", default=None)
# What a template entity's actions run is opened rather than held when it is
# one of these (#780); anything else that runs stored config is held.
_OPENED_DOMAINS = frozenset({SCRIPT_DOMAIN, SCENE_DOMAIN, "button", "input_button"})
# The service a stored state reproduces on a template switch, for choosing
# the action script it runs; any other state may run any of them.
_INLINE_STATE_SERVICES = {"on": "turn_on", "off": "turn_off"}

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
    if parts[-1] in RUN_SERVICES | _SCRIPT_SERVICES_THAT_DO_NOT_RUN:
        # HA allows no script named after its own services: this is a tool
        # for the service, such as `script__turn_off` (#837 review).
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


@dataclass(frozen=True)
class ToolOrigin:
    """The API a tool call goes to, and whether its name can be read here."""

    api_id: str | None = None
    api_name: str | None = None
    # A tool of a remote Home Assistant reached over MCP (#840): this Home
    # Assistant's devices, scripts and automations say nothing of it.
    remote: bool = False
    # The remote server marks the tool read-only (MCP `readOnlyHint`).
    read_only: bool = False


def tool_origin(
    ha_llm_api: Any, tool_name: str, api_id: str | None = None
) -> ToolOrigin:
    """
    Return the API that runs a tool, and whether it is a remote Home Assistant.

    The routing map names the API each offered tool is sent to (or `api_id`,
    the one a held call was sent to). A name the routing map lacks goes to
    the first API offering it, as the wrapper's fallback sends it. A remote
    Home Assistant reached over MCP offers its intents and scripts under the
    same names as this one's (`intent__HassTurnOn`, `script__open_door`), and
    other tools under names that say nothing (a script named by its id alone
    before 2026.9, `media_player__play_media`), so the whole server is judged:
    an API offering any tool named like Home Assistant's own intent or script
    tools that is not one is a remote Home Assistant, and every tool on it is
    remote (#840 review). This Home Assistant's own are intent and action
    tools. With no API to read, nothing is known and the name is read as
    before.
    """
    apis = getattr(ha_llm_api, "apis", None)
    if not isinstance(apis, Mapping):
        return ToolOrigin()
    if api_id is None:
        routing = getattr(ha_llm_api, "routing_map", None)
        api_id = routing.get(tool_name) if isinstance(routing, Mapping) else None
    if api_id is None:
        # The wrapper tries each API offering the name in turn, so the call
        # is read as the first one's, and as remote if any of them is.
        holders = [
            key
            for key, instance in apis.items()
            if _find_tool(instance, tool_name) is not None
        ]
        if not holders:
            return ToolOrigin()
        origins = [tool_origin(ha_llm_api, tool_name, key) for key in holders]
        remote = next((o for o in origins if o.remote), None)
        return remote or origins[0]
    instance = apis.get(api_id)
    if instance is None:
        return ToolOrigin(api_id=api_id)
    api_name = str(getattr(getattr(instance, "api", None), "name", "") or api_id)
    tool = _find_tool(instance, tool_name)
    remote = (
        tool is not None
        and not _is_ha_tool(tool)
        and _is_remote_home_assistant(instance)
    )
    read_only = bool(getattr(getattr(tool, "annotations", None), "read_only", False))
    return ToolOrigin(
        api_id=api_id, api_name=api_name, remote=remote, read_only=read_only
    )


def _find_tool(instance: Any, tool_name: str) -> Any | None:
    """Return the tool an API instance offers under a name, unwrapped."""
    tool = next(
        (t for t in getattr(instance, "tools", None) or [] if t.name == tool_name),
        None,
    )
    while isinstance(tool, llm.NamespacedTool):
        tool = tool.tool
    return tool


def _is_ha_tool(tool: Any) -> bool:
    """Return True for one of this Home Assistant's intent or action tools."""
    return isinstance(tool, llm.IntentTool | llm.ActionTool)


def _is_remote_home_assistant(instance: Any) -> bool:
    """Return True for an API offering a Home Assistant tool it does not run."""
    for tool in getattr(instance, "tools", None) or []:
        inner = tool
        while isinstance(inner, llm.NamespacedTool):
            inner = inner.tool
        if not _is_ha_tool(inner) and _named_like_ha_tool(str(tool.name)):
            return True
    return False


def is_script_like_tool(tool_name: str) -> bool:
    """Return True for a script's or an intent script's tool name."""
    parts = tool_name.split(TOOL_NAMESPACE_SEP)
    return script_tool_entity_id(tool_name) is not None or (
        len(parts) >= _SCRIPT_TOOL_PARTS and parts[-2] == _INTENT_SCRIPT
    )


def _named_like_ha_tool(tool_name: str) -> bool:
    """Return True for a name Home Assistant gives its intent or script tools."""
    return is_intent_tool(tool_name) or is_script_like_tool(tool_name)


def tool_args_kind(  # noqa: PLR0911
    hass: HomeAssistant | None, tool_name: str, *, remote: bool = False
) -> str:
    """
    Return how a tool's arguments are prepared before it runs (#836).

    `BUILTIN_INTENT`: one of Home Assistant's own intents, reduced to the
    slots it reads. `OTHER_INTENT`: an intent script (including one that took
    over a built-in intent's name) or another integration's intent, which
    keeps its own slots but matches its targets the way the built-ins do.
    `NOT_AN_INTENT`: a script, service or other tool, run as given.
    `REMOTE_TOOL`: a tool named like this Home Assistant's own that another
    API runs (`tool_origin`), run as given (#840).

    A built-in intent's name counts as an intent script's only when the
    intent script has that exact name, the one HA registers it under: one
    whose name only slugifies to it ("HassTurnOn!") leaves the built-in in
    place (#836 review).
    """
    if remote:
        return REMOTE_TOOL
    if script_tool_entity_id(tool_name) is not None:
        return NOT_AN_INTENT
    parts = tool_name.split(TOOL_NAMESPACE_SEP)
    if len(parts) >= _SCRIPT_TOOL_PARTS and parts[-2] == _INTENT_SCRIPT:
        return OTHER_INTENT  # an intent script's own tool, whatever its name
    base = base_tool_name(tool_name)
    handlers = intent_script_handlers(hass, tool_name)
    if is_intent_tool(tool_name):
        if any(str(getattr(h, "intent_type", "")) == base for h in handlers):
            return OTHER_INTENT
        return BUILTIN_INTENT
    if handlers:
        return OTHER_INTENT
    if _has_entities(hass):
        try:
            handlers = list(intent.async_get(cast("HomeAssistant", hass)))
        except Exception:
            LOGGER.exception("Could not read the intents")
            handlers = []
        if any(str(getattr(h, "intent_type", "")) == base for h in handlers):
            return OTHER_INTENT
    return NOT_AN_INTENT


def intent_script_handlers(hass: HomeAssistant | None, tool_name: str) -> list[Any]:
    """
    Return every intent script a tool may run (#779).

    Home Assistant 2026.9 and later name an intent script's tool
    ``intent_script__<slug>``; 2026.8 named it by the slug alone. A merged
    API prefixes its own namespace, and an intent script may take over a
    built-in intent (``HassTurnOn``), whose tool then runs it. So the tool
    name and each of its ``__``-suffixes are compared with every intent
    script's name and its slug, and all that match are returned: HA picks
    one by the exposed platforms, which the gate cannot see, so each one
    counts. A ``intent_script__`` tool matching none returns ``[_MISSING]``.
    """
    parts = tool_name.split(TOOL_NAMESPACE_SEP)
    namespaced = len(parts) >= _SCRIPT_TOOL_PARTS and parts[-2] == _INTENT_SCRIPT
    if not _has_entities(hass):
        return [_MISSING] if namespaced else []
    try:
        handlers = [
            handler
            for handler in intent.async_get(cast("HomeAssistant", hass))
            if type(handler).__name__ == _SCRIPT_INTENT_HANDLER
        ]
    except Exception:
        LOGGER.exception("Could not read the intent scripts")
        return [_MISSING] if namespaced else []
    suffixes = {TOOL_NAMESPACE_SEP.join(parts[index:]) for index in range(len(parts))}
    found = [
        handler
        for handler in handlers
        if {
            str(getattr(handler, "intent_type", "")),
            slugify(
                str(getattr(handler, "intent_type", "")), separator="_", lowercase=False
            ),
        }
        & suffixes
    ]
    if not found and namespaced:
        return [_MISSING]
    return found


def intent_scripts_are_critical(
    hass: HomeAssistant | None,
    tool_name: str,
    rules: Sequence[Mapping[str, str]],
) -> bool:
    """Return True if an intent script the tool may run does a critical action."""
    return any(
        _intent_script_is_critical(hass, handler, rules)
        for handler in intent_script_handlers(hass, tool_name)
    )


def _intent_script_is_critical(
    hass: HomeAssistant | None,
    handler: Any,
    rules: Sequence[Mapping[str, str]],
) -> bool:
    """Return True if an intent script's action could do a critical action."""
    if handler is _MISSING or hass is None:
        return True  # an intent script the gate cannot find: hold it
    config = getattr(handler, "config", None)
    if not isinstance(config, dict):
        return True
    action = config.get("action")
    if action is None:
        return False  # speech only
    sequence = getattr(action, "sequence", None)
    if not isinstance(sequence, list):
        LOGGER.debug("Cannot read the action of intent %s; holding it", handler)
        return True
    if find_critical_automation_calls({"actions": sequence}, rules):
        return True
    return _has_entities(hass) and any(StoredActionCheck(hass, rules).calls(sequence))


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
    check: StoredActionCheck | None = None,
) -> bool:
    """
    Return True if running this script or scene could do a critical action.

    Pass one ``check`` for several scripts or scenes, so the automations are
    read once.
    """
    if hass is None:
        return True
    entity_id = entity_id.lower()
    domain = entity_id.split(".", 1)[0]
    if check is None and _has_entities(hass):
        check = StoredActionCheck(hass, rules)
    # Running it changes its state, which an automation may watch (#814).
    if check is not None and check.reactors_critical(entity_id):
        return True
    if domain == SCRIPT_DOMAIN:
        return _script_steps_are_critical(hass, entity_id, rules, check)
    if domain == SCENE_DOMAIN:
        return _scene_is_critical(hass, entity_id, rules, check=check)
    return False


def stored_actions_are_critical(
    hass: HomeAssistant | None,
    entity_id: str,
    service: str | None,
    rules: Sequence[Mapping[str, str]],
    check: StoredActionCheck | None = None,
) -> bool:
    """
    Return True if acting on the entity runs stored actions that are critical.

    See ``template_actions``: a template entity's action scripts, or the
    automations a button press may start (#780). A group is checked through
    its members, and a switch_as_x wrapper through the switch it wraps. False
    for any other entity: its call is matched against the rules as it is.
    Pass one ``check`` to every entity of one call, so each is read once.
    """
    if not _has_entities(hass):
        return False
    walk = check or StoredActionCheck(cast("HomeAssistant", hass), rules)
    return walk.entity(entity_id.lower(), service)


def _has_entities(hass: HomeAssistant | None) -> bool:
    """Return True for an instance whose entities can be read."""
    return getattr(hass, "data", None) is not None


class StoredActionCheck:
    """
    One check through stored actions: each entity and step list read once.

    An entity reached again is answered from its first result; one reached
    while it is still being read (a cycle) adds nothing, since the reading
    higher up decides.
    """

    def __init__(self, hass: HomeAssistant, rules: Sequence[Mapping[str, str]]) -> None:
        """Start a check of stored actions against ``rules``."""
        self.hass = hass
        self.rules = rules
        self.reader = StoredActionReader(hass)
        self._entities: dict[tuple[str, str | None, bool], bool | None] = {}
        self._steps: dict[tuple[int, bool], bool] = {}
        self._wrapped: dict[str, tuple[str, bool]] | None = None
        self._chain = 0

    def entity(
        self, entity_id: str, service: str | None, *, lenient: bool = False
    ) -> bool:
        """
        Return True if acting on the entity runs critical stored actions.

        ``lenient`` reads what it runs the way a reacting automation is read
        (see ``reactors_critical``): it is reached from one.
        """
        key = (entity_id, service, lenient)
        if key in self._entities:
            return bool(self._entities[key])
        if len(self._entities) >= _MAX_FOLLOWED_ENTITIES or self._chain >= _MAX_CHAIN:
            return True  # a safety bound: past it, hold rather than read on
        self._entities[key] = None
        self._chain += 1
        try:
            result = self._read_entity(entity_id, service, lenient=lenient)
        finally:
            self._chain -= 1
        self._entities[key] = result
        if result:
            LOGGER.debug("%s runs a critical action", entity_id)
        return result

    def _read_entity(  # noqa: PLR0911
        self, entity_id: str, service: str | None, *, lenient: bool
    ) -> bool:
        members = with_members(self.hass, {entity_id}) - {entity_id}
        if any(
            self.entity(member, service, lenient=lenient) for member in sorted(members)
        ):
            return True
        # A switch_as_x wrapper's switch gets its own service, not the
        # wrapper's (closing a cover turns its relay on or off), so it is
        # followed with none: every action a template switch has counts, and
        # the relay rules are left to the call on the wrapper.
        if (wrapped := self._wrapped_switch(entity_id)) is not None and self.entity(
            wrapped, None, lenient=lenient
        ):
            return True
        domain = entity_id.split(".", 1)[0]
        # The call reaching the entity is matched against the rules first: a
        # rule naming a button a template switch presses still holds it.
        if service is not None and matches_critical_rule(
            domain=domain,
            service=service,
            entity_ids=[entity_id],
            critical_actions=self.rules,
        ):
            return True
        # An automation the change may start (#814).
        if self.reactors_critical(entity_id):
            return True
        if domain == _AUTOMATION_DOMAIN and service == "trigger":
            # `automation.trigger` runs its actions, as a reacting
            # automation may (#814).
            sequence = self.reader.automation_sequence(entity_id)
            return sequence is None or self.steps(sequence, lenient=lenient)
        if domain == SCRIPT_DOMAIN:
            if service in _SCRIPT_SERVICES_THAT_DO_NOT_RUN:
                return False
            return script_rule_matches(self.hass, entity_id, self.rules) or (
                self._script_steps(entity_id, lenient=lenient)
            )
        if domain == SCENE_DOMAIN:
            return matches_critical_rule(
                domain=SCENE_DOMAIN,
                service="turn_on",
                entity_ids=[entity_id],
                critical_actions=self.rules,
            ) or _scene_is_critical(
                self.hass, entity_id, self.rules, check=self, lenient=lenient
            )
        sequences = self.reader.own_sequences(entity_id, service)
        if sequences is None:
            return True
        # A template entity's own actions are read strictly, as since #780,
        # however it is reached: they are not an automation the user wrote.
        return any(self.steps(sequence) for sequence in sequences)

    def reactors_critical(self, entity_id: str) -> bool:
        """
        Return True if what a change to the entity may start does a critical action.

        Read leniently: the automations are the user's own, so a step that
        cannot be read ahead (an event it fires, a templated service, a REST
        call) does not hold the change; what can be read as protected does,
        down through the scripts and scenes they run. A button press is read
        strictly, as since #780: a press exists to start them.
        """
        reactors = self.reader.started_by(entity_id)
        if reactors is None:
            return True
        return self._any_critical(
            reactors, lenient=entity_id.split(".", 1)[0] not in PRESS_DOMAINS
        )

    def _any_critical(self, reactors: Sequence[Reactor], *, lenient: bool) -> bool:
        for reactor in reactors:
            if self.steps(reactor.sequence, lenient=lenient):
                started = _STARTED.get()
                if (
                    started is not None
                    and reactor.name is not None
                    and reactor.name not in started
                ):
                    started.append(reactor.name)
                return True
        return False

    def _entityless_change_critical(
        self, domain: str, step: Mapping[str, Any], *, lenient: bool
    ) -> bool:
        """
        Return True if a change naming no entity may start a critical automation.

        ``device_tracker.see`` marks a tracker (and the people it tracks) home or
        away. ``mqtt.publish`` sets off MQTT triggers, and can set the state of
        an MQTT entity or send it a command: the topic is not matched, so any
        MQTT entity a rule guards or an automation watches counts.
        """
        if domain == _MQTT_DOMAIN:
            reactors = self.reader.mqtt_reactors()
            if reactors is None:
                return True
            mqtt_entities = self.reader.mqtt_entities()
            if any(
                _entity_is_guarded(e.split(".", 1)[0], e, self.rules)
                for e in mqtt_entities
            ):
                return True
            # What watches an MQTT entity, gathered first so a home with
            # thousands of them is read once, not held by the entity bound.
            for entity_id in mqtt_entities:
                watching = self.reader.started_by(entity_id)
                if watching is None:
                    return True
                reactors += [r for r in watching if r not in reactors]
            return self._any_critical(reactors, lenient=True)
        fields: dict[str, Any] = {}
        for key in ("data", "data_template", "service_data"):
            block = step.get(key)
            if isinstance(block, dict):
                fields.update(block)
        dev_id = fields.get("dev_id")
        if isinstance(dev_id, str) and dev_id and "{" not in dev_id:
            trackers = {f"device_tracker.{dev_id.lower()}"}
        else:
            trackers = {
                state.entity_id
                for state in self.hass.states.async_all("device_tracker")
            }
        people = {
            state.entity_id
            for state in self.hass.states.async_all("person")
            if trackers
            & {str(t).lower() for t in state.attributes.get("device_trackers") or []}
        }
        if not isinstance(dev_id, str) or not dev_id:
            people |= {
                state.entity_id for state in self.hass.states.async_all("person")
            }
        # A tracker it creates is in no zone's list yet, but joins the zone
        # it is placed in (#820): every zone may change.
        zones = set(self.hass.states.async_entity_ids("zone"))
        return any(
            self.entity(entity_id, None, lenient=lenient)
            for entity_id in sorted(trackers | people | zones)
        )

    def _script_steps(self, entity_id: str, *, lenient: bool) -> bool:
        component = self.hass.data.get(_SCRIPT_COMPONENT_KEY)
        entity = component.get_entity(entity_id) if component is not None else None
        sequence = getattr(getattr(entity, "script", None), "sequence", None)
        if not isinstance(sequence, list):
            LOGGER.debug("Cannot read the steps of %s; holding it", entity_id)
            return True
        return self.steps(sequence, lenient=lenient)

    def _wrappers_in(self, domain: str) -> list[str]:
        """Return the switch_as_x wrappers in a domain."""
        if self._wrapped is None:
            self._wrapped = read_switch_wrappers(self.hass)
        return sorted(w for w in self._wrapped if w.split(".", 1)[0] == domain)

    def _wrapped_switch(self, entity_id: str) -> str | None:
        """Return the switch a switch_as_x wrapper drives, if it is one."""
        if self._wrapped is None:
            self._wrapped = read_switch_wrappers(self.hass)
        found = self._wrapped.get(entity_id)
        return found[0] if found else None

    def steps(self, sequence: Any, *, lenient: bool = False) -> bool:
        """
        Return True if the steps do or reach a critical action.

        A script, scene or button the steps run is opened rather than held,
        when its target can be named: a template TV switch whose `turn_on`
        runs `script.tv_on` asks for the PIN only if that script does
        something that would be held.
        """
        key = (id(sequence), lenient)
        if key not in self._steps:
            self._steps[key] = False  # a cycle back to these steps adds nothing
            self._steps[key] = self._read_steps(sequence, lenient=lenient)
        return self._steps[key]

    def _read_steps(self, sequence: Any, *, lenient: bool) -> bool:
        found = find_critical_automation_calls({"actions": sequence}, self.rules)
        if lenient:
            found = [call for call in found if not call.unverifiable]
        if any(
            "." not in call.service
            or call.service.split(".", 1)[0] not in _OPENED_DOMAINS
            for call in found
        ):
            return True
        if found and not lenient:
            for domain, service, step in iter_service_calls(sequence):
                if (
                    domain in _OPENED_DOMAINS
                    and (call_targets(self.hass, domain, service, step)[1])
                ):
                    return True  # which script, scene or button is unknown
        return any(self.calls(sequence, lenient=lenient))

    def calls(
        self, sequence: Any, *, lenient: bool = False
    ) -> Iterator[CriticalAutomationCall]:
        """
        Yield the calls in the steps that run critical stored actions.

        A call whose targets cannot all be named (``all``, a template) is
        held when any entity of its domain runs critical stored actions,
        since it may be among them, as a relay is (#794).
        """
        for domain, service, step in iter_service_calls(sequence):
            call = f"{domain}.{service}"
            if (domain, service) in _ENTITYLESS_CHANGES:
                # A change that names no entity (#814).
                if self._entityless_change_critical(domain, step, lenient=lenient):
                    yield CriticalAutomationCall(
                        service=call,
                        entity_ids=(),
                        reason="it may start an automation that performs a "
                        "critical action",
                    )
                continue
            if domain == SCENE_DOMAIN and service in _SCENE_INLINE_SERVICES:
                states, readable = inline_states(step)
                if not readable:
                    continue  # `find_critical_automation_calls` holds it
                for entity_id, state in states.items():
                    if entity_id.split(".", 1)[0] in (SCRIPT_DOMAIN, SCENE_DOMAIN):
                        continue  # HA cannot reproduce a script's or scene's state
                    if self.entity(
                        entity_id,
                        _INLINE_STATE_SERVICES.get(state or ""),
                        lenient=lenient,
                    ):
                        yield _stored_call(call, entity_id)
                continue
            entities, unresolved = call_targets(self.hass, domain, service, step)
            for entity_id in sorted(entities):
                if self.entity(entity_id, service, lenient=lenient):
                    yield _stored_call(call, entity_id)
            if unresolved and (
                domain == _GENERIC_DOMAIN
                or any(
                    self.entity(entity_id.lower(), service, lenient=lenient)
                    for entity_id in (
                        *entities_of(self.hass, domain),
                        *self._wrappers_in(domain),
                    )
                )
            ):
                yield CriticalAutomationCall(
                    service=call,
                    entity_ids=(),
                    reason="its targets cannot all be named, and an entity of "
                    "that kind runs stored actions that perform a critical action",
                )


@contextmanager
def collect_started_automations() -> Iterator[list[str]]:
    """Collect the automations the checks in this block find a change would start."""
    names: list[str] = []
    token = _STARTED.set(names)
    try:
        yield names
    finally:
        _STARTED.reset(token)


def stored_action_calls(
    hass: HomeAssistant | None,
    sequence: Any,
    rules: Sequence[Mapping[str, str]],
) -> list[CriticalAutomationCall]:
    """
    Return the calls in the steps whose target runs critical stored actions.

    For an automation the agent writes: a step turning on a template switch
    that opens the garage door (#780).
    """
    if not _has_entities(hass):
        return []
    return list(StoredActionCheck(cast("HomeAssistant", hass), rules).calls(sequence))


def _stored_call(call: str, entity_id: str) -> CriticalAutomationCall:
    return CriticalAutomationCall(
        service=call,
        entity_ids=(entity_id,),
        reason="the entity runs stored actions that perform a critical action",
    )


# `<domain>.<service>` the way HA spells a service.
_SERVICE_NAME_SHAPE = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")
_SERVICE_SLUG = re.compile(r"^[a-z0-9_]+$")
_TEMPLATE_MARKS = ("{{", "{%")
# Spaces and zero-width characters, which a tool may strip from a name.
_NAME_SPACE = re.compile(r"[\s\u200b-\u200d\u2060\ufeff]+")
# Domains whose services a tool may send to another Home Assistant, or that
# register theirs late: a tool named after one is screened even when this
# Home Assistant has no such service (#837 review).
_SCREENED_TOOL_DOMAINS = frozenset(
    {
        "alarm_control_panel",
        "button",
        "cover",
        "homeassistant",
        "input_button",
        "lock",
        SCENE_DOMAIN,
        SCRIPT_DOMAIN,
        "switch",
        "valve",
    }
)


@dataclass(frozen=True)
class ServiceNames:
    """The service names a call gives in `service` and `action` (#837)."""

    dotted: list[str] = field(default_factory=list)
    bare_services: list[str] = field(default_factory=list)
    bare_actions: list[str] = field(default_factory=list)
    # A name that cannot be read: a template, a value that is not text, a
    # dotted name not shaped like a service, or a templated domain.
    unreadable: bool = False


def call_service_names(tool_args: Mapping[str, Any]) -> ServiceNames:
    """
    Return the service names a call gives (#837).

    From `service` and `action`, each one name or a list of them. Both count
    when both are given: which one runs is up to the tool. Spaces inside a
    name are dropped, as a tool may drop them.
    """
    dotted: list[str] = []
    bare: dict[str, list[str]] = {"service": [], "action": []}
    unreadable = False
    for key in ("service", "action"):
        value = tool_args.get(key)
        if value is None or value == "":
            continue
        for item in value if isinstance(value, list) else [value]:
            if not isinstance(item, str):
                unreadable = True
                continue
            if any(mark in item for mark in _TEMPLATE_MARKS):
                unreadable = True
                continue
            name = _NAME_SPACE.sub("", item).lower()
            if "." not in name:
                if name:
                    bare[key].append(name)
            elif _SERVICE_NAME_SHAPE.match(name):
                dotted.append(name)
            else:
                unreadable = True
    domains = tool_args.get("domain")
    if any(
        isinstance(domain, str) and any(mark in domain for mark in _TEMPLATE_MARKS)
        for domain in (domains if isinstance(domains, list) else [domains])
    ):
        unreadable = True
    return ServiceNames(
        dotted=list(dict.fromkeys(dotted)),
        bare_services=list(dict.fromkeys(bare["service"])),
        bare_actions=list(dict.fromkeys(bare["action"])),
        unreadable=unreadable,
    )


# Fields a scene service sets its states from.
_SCENE_STATE_KEYS = frozenset({"entities", "snapshot_entities"})
# A dotted name anywhere in a value: a service, an entity, or one in a
# template.
_DOTTED_NAME = re.compile(r"[a-z0-9_]+\.[a-z0-9_]+", re.IGNORECASE)


def mentions_a_service_call(value: object) -> bool:
    """
    Return True if a value may hold a service call or its target (#837).

    For a `service` or `action` value that cannot be read: a template or a
    nested object naming a service or an entity, or carrying a target or a
    scene's states, could run one. An odd value naming none could not.
    """
    if isinstance(value, str):
        return bool(_DOTTED_NAME.search(value))
    if isinstance(value, Mapping):
        return any(
            key in STEP_TARGET_KEYS
            or key in _SCENE_STATE_KEYS
            or mentions_a_service_call(key)
            or mentions_a_service_call(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(mentions_a_service_call(item) for item in value)
    return False


def calls_a_service(tool_args: Mapping[str, Any], entity_ids: Sequence[str]) -> bool:
    """
    Return True if a call whose service cannot be read may still make one.

    It gives a domain, a `service`, a target or a scene's states, or its
    `service` or `action` holds one. An `action` field alone, in a tool that
    takes one for its own use, does not make it one.
    """
    return bool(
        tool_args.get("domain")
        or "service" in tool_args
        or entity_ids
        or any(key in tool_args for key in STEP_TARGET_KEYS)
        or any(key in tool_args for key in _SCENE_STATE_KEYS)
        or any(
            mentions_a_service_call(tool_args.get(key)) for key in ("service", "action")
        )
    )


def rule_domains(rules: Sequence[Mapping[str, str]]) -> frozenset[str]:
    """Return the domains critical-action rules name."""
    return frozenset(
        domain for rule in rules if (domain := str(rule.get("domain") or "").lower())
    )


def has_service(hass: HomeAssistant | None, domain: str, service: str) -> bool:
    """Return True if HA has the service, or when it cannot be asked."""
    services = getattr(hass, "services", None)
    return services is None or bool(services.has_service(domain, service))


def named_services(
    tool_args: Mapping[str, Any],
    tool_name: str,
    hass: HomeAssistant | None,
    domains: frozenset[str] = frozenset(),
) -> list[str]:
    """
    Return each `<domain>.<service>` a call names in full (#837).

    A dotted `service` or `action`, whether or not this Home Assistant has
    the service (a tool may send it elsewhere), and a tool named after a
    service: `scene.apply`, or `scene__apply` the way HA names an action's
    tool, namespaced or not. A tool name counts when HA has the service, or
    its domain is one critical services live in (``_SCREENED_TOOL_DOMAINS``
    and the given `domains`), so a tool such as `mcp__get_weather` is not
    read as one. A bare `service` or `action` is left to the caller, which
    reads its domain the way the `domain`/`service` form is read.
    """
    named = list(call_service_names(tool_args).dotted)
    from_tool: list[str] = []
    if "." in (base := tool_name.rsplit(TOOL_NAMESPACE_SEP, maxsplit=1)[-1]):
        from_tool.append(base)
    parts = tool_name.split(TOOL_NAMESPACE_SEP)
    if len(parts) > 1:
        from_tool.append(f"{parts[-2]}.{parts[-1]}")
    for candidate in from_tool:
        name = candidate.strip().lower()
        if not _SERVICE_NAME_SHAPE.match(name):
            continue
        domain, _, service = name.partition(".")
        # A name HA does not have counts in a critical domain only when it is
        # spelled the way HA spells a service: an MCP server titled "Lock"
        # gives `lock__HassListAddItem`, which names none (#837 review).
        spelled = _SERVICE_SLUG.match(candidate.strip().partition(".")[2])
        if has_service(hass, domain, service) or (
            spelled and (domain in _SCREENED_TOOL_DOMAINS or domain in domains)
        ):
            named.append(name)
    return list(dict.fromkeys(named))


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
            return _inline_states_are_critical(tool_args, rules) or (
                _stored_calls_reached(hass, step, rules)
            )
        if service == "turn_on":
            return _targets_run_critical(hass, step, frozenset({SCENE_DOMAIN}), rules)
        return False
    if domain == _GENERIC_DOMAIN and service in RUN_SERVICES:
        if _targets_run_critical(
            hass, step, frozenset({SCRIPT_DOMAIN, SCENE_DOMAIN}), rules
        ):
            return True
    elif find_critical_automation_calls({"actions": [step]}, rules):
        return True
    # A template switch or button it acts on, or a button it presses, runs
    # stored actions no rule names (#780).
    return _stored_calls_reached(hass, step, rules)


def _stored_calls_reached(
    hass: HomeAssistant | None,
    step: Mapping[str, Any],
    rules: Sequence[Mapping[str, str]],
) -> bool:
    """Return True if the step runs critical stored actions."""
    return _has_entities(hass) and any(
        StoredActionCheck(cast("HomeAssistant", hass), rules).calls([step])
    )


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
    if "device_id" in step and isinstance(step.get("target", {}), dict):
        # At the top of a step, `device_id` makes it read as a device action
        # (Home Assistant's own classification), so a service call aimed at a
        # device was held as one that cannot be identified. Under `target:`
        # it is the call's target, as Home Assistant reads it (#804).
        target = dict(step.get("target", {}))
        devices = step.pop("device_id")
        if (already := target.get("device_id")) is not None:
            devices = [
                *(already if isinstance(already, list) else [already]),
                *(devices if isinstance(devices, list) else [devices]),
            ]
        target["device_id"] = devices
        step["target"] = target
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
    check = (
        StoredActionCheck(cast("HomeAssistant", hass), rules)
        if _has_entities(hass)
        else None
    )
    return any(
        _opened_is_critical(hass, entity_id, rules, check)
        for entity_id in entities
        if entity_id.split(".", 1)[0] in opened
    )


def _opened_is_critical(
    hass: HomeAssistant | None,
    entity_id: str,
    rules: Sequence[Mapping[str, str]],
    check: StoredActionCheck | None = None,
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
    return named or runs_critical_action(hass, entity_id, rules, check)


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
    if not any(key in fields for key in INLINE_STATE_KEYS):
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
    hass: HomeAssistant,
    entity_id: str,
    rules: Sequence[Mapping[str, str]],
    check: StoredActionCheck | None = None,
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
        return True
    # A step on a template switch or an input button runs its stored actions.
    return any((check or StoredActionCheck(hass, rules)).calls(sequence))


def _scene_is_critical(
    hass: HomeAssistant,
    entity_id: str,
    rules: Sequence[Mapping[str, str]],
    *,
    check: StoredActionCheck | None = None,
    lenient: bool = False,
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
    check = check or StoredActionCheck(hass, rules)
    return any(
        _state_is_critical(
            hass, state, rules, depth=0, seen=seen, check=check, lenient=lenient
        )
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


def _state_is_critical(  # noqa: PLR0911, PLR0913
    hass: HomeAssistant,
    state: State,
    rules: Sequence[Mapping[str, str]],
    *,
    depth: int,
    seen: set[str],
    check: StoredActionCheck,
    lenient: bool = False,
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
                check=check,
                lenient=lenient,
            )
            for member in get_entity_ids(hass, entity_id)
        )
    service = _reproduce_service(hass, domain, state)
    if service is not None:
        if matches_critical_rule(
            domain=domain,
            service=service,
            entity_ids=[entity_id],
            critical_actions=rules,
        ):
            return True
    elif _entity_is_guarded(domain, entity_id, rules):
        return True
    # A template switch set on or off runs that action script; with a state
    # that names no one service, any of them may run (#780).
    if domain in (SCRIPT_DOMAIN, SCENE_DOMAIN):
        return False  # HA cannot reproduce a script's or a scene's state
    return check.entity(entity_id, service, lenient=lenient)


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
