"""
Critical push for the safety automations the agent writes (#869).

An automation the agent writes for a leak alert calls ``notify.mobile_app_*``
with an ordinary payload, and the phone treats it like every other Home
Assistant push: grouped with the day's camera and Sentinel notifications,
summarized, or held back by a Focus mode. A leak, smoke, gas, carbon monoxide
or safety alarm is the one alert that must not wait, so ``add_automation``
rewrites the push itself after validation, whatever the model wrote:

* iOS: ``push.sound`` becomes a critical sound at full volume, which sounds
  even in Do Not Disturb (once the user has allowed Critical Alerts for the
  companion app).
* Android: the notification goes to the ``alarm_stream`` channel at high
  importance and high priority, so a dozing phone gets it now. No ``ttl``:
  FCM drops a zero-ttl message the phone cannot take at once, and a late
  leak alarm beats a lost one.

Both go in the push payload, the ``data`` inside the notify call's own data;
a key beside ``message`` fails Home Assistant's notify schema when the action
runs. Each platform ignores the other's keys, so both are added to every push
and no phone has to be identified.

The automation counts when an enabled trigger fires on the alarm itself: a
state trigger on a binary sensor of a safety device class into ``on`` (not
one that only fires when the alarm clears, nor an attribute trigger), a
device trigger of an alarm type, or ``moisture.detected``. Triggers are read
from the validated config, where Home Assistant has flattened nested trigger
lists, parsed ``enabled`` and renamed ``trigger`` to ``platform``. A
``choose`` option or ``if`` branch gated on trigger ids that name no alarm
trigger (the "leak cleared" half of a wet/dry automation) keeps its pushes
as written.

Only services the mobile_app integration registered for its phones are
rewritten: a notify group or another platform may reject keys it does not
know, whatever it is called. A companion app command (``clear_notification``,
``command_*``...) is not an alert and is left alone, as is a push whose data
is a template.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from homeassistant.core import valid_entity_id
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import entity_registry as er
from homeassistant.util import slugify

from .automation_pin import _MAX_STEP_DEPTH, _is_template, _TooDeepError, _walk_steps
from .automation_targets import _step_service

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from homeassistant.core import HomeAssistant

# The binary sensor device classes whose alarm must reach the user whatever
# the phone's mode. Strings, not BinarySensorDeviceClass, so this module does
# not import the binary_sensor component (Hassfest counts every import).
SAFETY_DEVICE_CLASSES = frozenset(
    {"moisture", "smoke", "gas", "carbon_monoxide", "safety"}
)

# Binary sensor device trigger types that fire when one of those alarms
# starts; their "cleared" twins (not_moist, no_smoke, ...) do not count.
_ALARM_DEVICE_TRIGGER_TYPES = frozenset({"moist", "smoke", "gas", "co", "unsafe"})

# Purpose-specific triggers that fire on an alarm, whatever they target; the
# matching `*_cleared` triggers do not count.
_ALARM_PURPOSE_TRIGGERS = frozenset(
    {
        "moisture.detected",
        "air_quality.smoke_detected",
        "air_quality.gas_detected",
        "air_quality.co_detected",
    }
)

_ALARM_STATE = "on"

_NOTIFY_DOMAIN = "notify"
_MOBILE_APP_DOMAIN = "mobile_app"
# A notify service that writes to Home Assistant itself, not to a phone.
_NOT_A_PUSH = frozenset({"notify.persistent_notification"})

# Messages the companion apps treat as commands, not notifications.
_COMMAND_MESSAGES = frozenset(
    {"clear_notification", "remove_channel", "request_location_update", "tts"}
)
_COMMAND_PREFIX = "command_"
# A command named as a quoted literal inside a message template.
_TEMPLATED_COMMAND_RE = re.compile(
    r"""["'](?:clear_notification|remove_channel|request_location_update|tts|command_\w*)["']"""
)

# iOS: a critical sound at full volume.
_IOS_CRITICAL_SOUND: dict[str, Any] = {"critical": 1, "volume": 1.0}
_IOS_DEFAULT_SOUND = "default"
_IOS_NO_SOUND = "none"
_IOS_INTERRUPTION_LEVEL = "interruption-level"

# Android: the alarm stream channel, delivered now.
_ANDROID_CRITICAL_DATA: dict[str, Any] = {
    "channel": "alarm_stream",
    "importance": "high",
    "priority": "high",
}

# Step keys holding the notify call's data. HA renders `data`, then merges
# the legacy `data_template` over it key by key.
_STEP_DATA = "data"
_STEP_DATA_TEMPLATE = "data_template"
# The push payload inside the notify call's data.
_PUSH_DATA = "data"


@dataclass(frozen=True)
class CriticalPushResult:
    """What the rewrite did to one automation."""

    # Push steps made critical.
    rewritten: int = 0
    # Notifications the alarm sends that could not be made critical: another
    # notify service, or a templated payload.
    left_ordinary: int = 0


def _trigger_entity_ids(
    hass: HomeAssistant, trigger: Mapping[str, Any]
) -> Iterator[str]:
    """Yield the entity ids one trigger names, registry ids resolved."""
    for value in cv.ensure_list(trigger.get("entity_id")):
        if not isinstance(value, str) or _is_template(value):
            continue
        for part in value.split(","):
            lowered = part.strip().lower()
            if valid_entity_id(lowered):
                yield lowered
            # A device trigger names its entity by registry id.
            elif entity_id := er.async_resolve_entity_id(er.async_get(hass), lowered):
                yield entity_id


def _device_class(hass: HomeAssistant, entity_id: str) -> str | None:
    """Return a binary sensor's device class, from its state or its registry entry."""
    if not entity_id.startswith("binary_sensor."):
        return None
    if (state := hass.states.get(entity_id)) is not None:
        device_class = state.attributes.get("device_class")
        if device_class:
            return str(device_class)
    entry = er.async_get(hass).async_get(entity_id)
    if entry is None:
        return None
    return entry.device_class or entry.original_device_class


def _fires_on_alarm(trigger: Mapping[str, Any], kind: Any) -> bool:
    """Return True if a state or device trigger fires when the alarm starts."""
    if kind == "device":
        return trigger.get("type") in _ALARM_DEVICE_TRIGGER_TYPES
    if kind != "state" or "attribute" in trigger:
        # An attribute trigger compares attribute values, not the alarm.
        return False
    if _ALARM_STATE in cv.ensure_list(trigger.get("not_to")):
        return False
    to_states = cv.ensure_list(trigger.get("to"))
    if to_states:
        return _ALARM_STATE in to_states
    from_states = set(cv.ensure_list(trigger.get("from")))
    # Neither `to` nor `from` also fires on every attribute update (battery,
    # signal), which must not sound through Do Not Disturb; a `from` alone is
    # a state change, unless it can only be the alarm ending.
    return bool(from_states) and from_states != {_ALARM_STATE}


def _is_alarm_trigger(hass: HomeAssistant, trigger: Mapping[str, Any]) -> bool:
    """Return True if one validated trigger fires on a safety alarm."""
    # A templated `enabled` is decided when HA attaches the trigger; never
    # let one escalate the automation's other notifications.
    if trigger.get("enabled", True) is not True:
        return False
    # Validation renames `trigger:` to `platform:`.
    kind = trigger.get("platform", trigger.get("trigger"))
    if kind in _ALARM_PURPOSE_TRIGGERS:
        return True
    return _fires_on_alarm(trigger, kind) and any(
        _device_class(hass, entity_id) in SAFETY_DEVICE_CLASSES
        for entity_id in _trigger_entity_ids(hass, trigger)
    )


def _triggers(validated_config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the validated triggers, which HA has already flattened."""
    raw = validated_config.get("triggers", validated_config.get("trigger"))
    return [t for t in cv.ensure_list(raw) if isinstance(t, dict)]


def _alarm_trigger_ids(
    hass: HomeAssistant, validated_config: Mapping[str, Any]
) -> tuple[set[str], set[str]]:
    """Return (alarm trigger ids, all trigger ids), as trigger conditions see them."""
    alarm: set[str] = set()
    every: set[str] = set()
    for index, trigger in enumerate(_triggers(validated_config)):
        # HA gives a trigger without an `id` its index as the id.
        trigger_id = str(trigger.get("id", index))
        every.add(trigger_id)
        if _is_alarm_trigger(hass, trigger):
            alarm.add(trigger_id)
    return alarm, every


def is_safety_automation(
    hass: HomeAssistant, validated_config: Mapping[str, Any]
) -> bool:
    """Return True if an enabled trigger fires on a leak, smoke, gas or CO alarm."""
    return bool(_alarm_trigger_ids(hass, validated_config)[0])


def _trigger_condition_ids(condition: Any) -> set[str] | None:
    """Return the ids an enabled trigger condition requires, else None."""
    if (
        not isinstance(condition, dict)
        or condition.get("condition") != "trigger"
        # HA skips a disabled condition, and decides a templated one later.
        or condition.get("enabled", True) is not True
    ):
        return None
    return {str(value) for value in cv.ensure_list(condition.get("id"))}


def _gating_trigger_ids(conditions: Any, *, only: bool = False) -> set[str] | None:
    """
    Return the trigger ids a branch's conditions require, or None if any.

    Conditions in a list must all hold, so each trigger condition narrows the
    set; anything else (``or``, templates, state checks) is not about which
    trigger fired. With ``only``, every condition must be a trigger condition,
    as when the result decides the branch taken when they do not hold.
    """
    required: set[str] | None = None
    for condition in cv.ensure_list(conditions):
        ids = _trigger_condition_ids(condition)
        if ids is None:
            if only:
                return None
            continue
        required = ids if required is None else required & ids
    return required


def _unreachable_choose(
    node: Mapping[str, Any], alarm: set[str], every: set[str]
) -> Iterator[Any]:
    """Yield the options, and the default, of a choose only non-alarms reach."""
    # Trigger ids that can still reach `default`, while every option is
    # gated on trigger ids alone.
    default_ids: set[str] | None = set(every)
    for option in cv.ensure_list(node.get("choose")):
        if not isinstance(option, dict):
            default_ids = None
            continue
        conditions = option.get("conditions")
        ids = _gating_trigger_ids(conditions)
        if ids is not None and not ids & alarm:
            yield option.get("sequence")
        pure = _gating_trigger_ids(conditions, only=True)
        default_ids = (
            None if pure is None or default_ids is None else default_ids - pure
        )
    if default_ids is not None and not default_ids & alarm:
        yield node.get("default")


def _unreachable_if(
    node: Mapping[str, Any], alarm: set[str], every: set[str]
) -> Iterator[Any]:
    """Yield the then, or the else, of an if that only non-alarms reach."""
    ids = _gating_trigger_ids(node["if"])
    if ids is not None and not ids & alarm:
        yield node.get("then")
    # `else` runs whenever the conditions fail, so only conditions that are
    # all about the trigger say which triggers reach it.
    pure = _gating_trigger_ids(node["if"], only=True)
    if pure is not None and not (every - pure) & alarm:
        yield node.get("else")


def _unreachable_branches(
    node: Any, alarm: set[str], every: set[str], depth: int = 0
) -> Iterator[Any]:
    """Yield every step list only non-alarm triggers can reach."""
    if depth > _MAX_STEP_DEPTH:
        raise _TooDeepError
    if isinstance(node, (list, tuple)):
        for item in cast("list[Any]", node):
            yield from _unreachable_branches(item, alarm, every, depth)
        return
    if not isinstance(node, dict):
        return
    node = cast("dict[str, Any]", node)
    if "choose" in node:
        yield from _unreachable_choose(node, alarm, every)
    if "if" in node:
        yield from _unreachable_if(node, alarm, every)
    for value in node.values():
        if isinstance(value, (list, tuple, dict)):
            yield from _unreachable_branches(value, alarm, every, depth + 1)


def _unaliased(node: Any, depth: int = 0) -> Any:
    """
    Return a copy of a config tree in which no two places share an object.

    ``copy.deepcopy`` keeps shared references shared, which is what a YAML
    alias is.
    """
    if depth > _MAX_STEP_DEPTH:
        raise _TooDeepError
    if isinstance(node, dict):
        return {
            key: _unaliased(value, depth + 1)
            for key, value in cast("dict[Any, Any]", node).items()
        }
    if isinstance(node, (list, tuple)):
        return [_unaliased(item, depth) for item in cast("list[Any]", node)]
    return node


def _mobile_app_services(hass: HomeAssistant) -> set[str]:
    """Return the notify services the mobile_app integration made for its phones."""
    # mobile_app registers `notify.<slug of "mobile_app_<device name>">`; a
    # notify group may take a name like that too.
    return {
        f"{_NOTIFY_DOMAIN}.{slugify(f'{_MOBILE_APP_DOMAIN}_{name}')}"
        for entry in hass.config_entries.async_entries(_MOBILE_APP_DOMAIN)
        if isinstance(name := entry.data.get("device_name"), str) and name
    }


def _is_push(service: str | None) -> bool:
    """Return True for a notify service that may reach a phone."""
    return (
        service is not None
        and service.startswith(f"{_NOTIFY_DOMAIN}.")
        and service not in _NOT_A_PUSH
    )


def _effective_message(data: Mapping[str, Any], template: Mapping[str, Any]) -> Any:
    """Return the message HA sends, `data_template` winning over `data`."""
    return template.get("message", data.get("message"))


def _is_command(message: Any) -> bool:
    """Return True if a push is, or may render to, a companion app command."""
    if not isinstance(message, str):
        return False
    lowered = message.strip().lower()
    if _is_template(lowered):
        # Only a command quoted in the template; a templated alert ("Leak at
        # {{ ... }}", "Massachusetts") is an alert like any other.
        return bool(_TEMPLATED_COMMAND_RE.search(lowered))
    return lowered in _COMMAND_MESSAGES or lowered.startswith(_COMMAND_PREFIX)


def _critical_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return a push payload made critical, or None if part of it is a template."""
    push: Any = payload.get("push")
    if push is None:
        push = {}
    if not isinstance(push, dict):
        return None
    push = cast("dict[str, Any]", push)
    sound: Any = push.get("sound")
    if _is_template(sound):
        return None
    if isinstance(sound, dict):
        sound = cast("dict[str, Any]", sound)
        name = sound.get("name")
    else:
        name, sound = sound, {}
    if not isinstance(name, str) or not name or name.lower() == _IOS_NO_SOUND:
        name = _IOS_DEFAULT_SOUND
    push["sound"] = {**sound, "name": name, **_IOS_CRITICAL_SOUND}
    if _IOS_INTERRUPTION_LEVEL in push:
        # A lower level written by the model would outrank the sound.
        push[_IOS_INTERRUPTION_LEVEL] = "critical"
    return {**payload, "push": push, **_ANDROID_CRITICAL_DATA}


def _make_critical(step: dict[str, Any]) -> bool | None:
    """
    Add the critical payload to one push step.

    Returns True when rewritten, False when its data is a template, and None
    for a companion app command, which is not an alert.
    """
    data: Any = step.get(_STEP_DATA)
    template: Any = step.get(_STEP_DATA_TEMPLATE)
    if data is None:
        data = {}
    if not isinstance(data, dict) or (
        template is not None and not isinstance(template, dict)
    ):
        return False
    data = cast("dict[str, Any]", data)
    template = cast("dict[str, Any]", template or {})
    message = _effective_message(data, template)
    if _is_command(message):
        return None
    # The legacy key wins when both carry a payload, so rewrite that one.
    key, source = _STEP_DATA, data
    if _PUSH_DATA in template:
        key, source = _STEP_DATA_TEMPLATE, template
    # A copy: YAML anchors can share one mapping between several steps.
    container = copy.deepcopy(source)
    payload: Any = container.get(_PUSH_DATA)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return False
    if _is_template(message) and "channel" in payload:
        # The template may render to a command that takes `channel` as its
        # argument (remove_channel from a variable); never retarget it.
        return False
    critical = _critical_payload(cast("dict[str, Any]", payload))
    if critical is None:
        return False
    container[_PUSH_DATA] = critical
    step[key] = container
    return True


def apply_critical_push(
    hass: HomeAssistant,
    config: dict[str, Any],
    validated_config: Mapping[str, Any],
) -> CriticalPushResult:
    """
    Make every mobile push a safety alarm sends critical, in place.

    ``config`` is the raw config that will be written; ``validated_config``
    is Home Assistant's validated copy, which decides whether, and through
    which trigger ids, the automation fires on a safety alarm.
    """
    alarm, every = _alarm_trigger_ids(hass, validated_config)
    if not alarm:
        return CriticalPushResult()
    if "use_blueprint" in config:
        # The pushes live in the blueprint, which this cannot rewrite.
        try:
            pushes = sum(
                1
                for step, action_type in _walk_steps(validated_config.get("actions"))
                if action_type == cv.SCRIPT_ACTION_CALL_SERVICE
                and _is_push(_step_service(step))
            )
        except _TooDeepError:
            pushes = 0
        return CriticalPushResult(left_ordinary=pushes)
    key = "actions" if "actions" in config else "action"
    try:
        # One object per step: a YAML alias can share a step between the
        # alarm's branch and the all-clear's, which are rewritten differently.
        actions = None
        if key in config:
            config[key] = actions = _unaliased(config[key])
        steps = list(_walk_steps(actions))
        unreachable = {
            id(step)
            for branch in _unreachable_branches(actions, alarm, every)
            for step, _ in _walk_steps(branch)
        }
    except _TooDeepError:
        # The existence check refuses such a config before this runs.
        return CriticalPushResult()
    mobile = _mobile_app_services(hass)
    rewritten = left_ordinary = 0
    seen: set[int] = set()
    for step, action_type in steps:
        if (
            action_type != cv.SCRIPT_ACTION_CALL_SERVICE
            or id(step) in seen
            or id(step) in unreachable
        ):
            continue
        seen.add(id(step))
        service = _step_service(step)
        if not _is_push(service):
            continue
        outcome = (
            _make_critical(cast("dict[str, Any]", step)) if service in mobile else False
        )
        if outcome:
            rewritten += 1
        elif outcome is False:
            left_ordinary += 1
    return CriticalPushResult(rewritten=rewritten, left_ordinary=left_ordinary)
