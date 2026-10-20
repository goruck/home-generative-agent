"""
Keep HassTurnOn / HassTurnOff going the way the user asked (#797).

For a cover, HassTurnOn opens and HassTurnOff closes; for a lock, HassTurnOn
locks and HassTurnOff unlocks. Both are bound together, and a small model can
call the wrong one: "open the study shade" ran HassTurnOff and closed it.
Home Assistant's result names the entity but not what was done, so the
model's reply then described what it meant to do ("the shade is now open").

* **Before the call** (``wrong_direction_refusal``): when the request says one
  direction only (``open``, ``turn on``, ``lock`` against ``close``,
  ``turn off``, ``unlock``) and the tool goes the other way, it is not run;
  the model is told which tool does what was asked. A request naming both
  directions, a negation, or none of these words (another language) is left
  alone.
* **After the call** (``with_action_performed``): the result says what was
  done to each entity ("closed Study Shade"), whatever the request, so the
  reply can match it.
"""

from __future__ import annotations

import re
from typing import Any

from .helpers import base_tool_name

ON = "on"
OFF = "off"
_TOOL_DIRECTIONS = {"HassTurnOn": ON, "HassTurnOff": OFF}
_TOOL_FOR = {ON: "HassTurnOn", OFF: "HassTurnOff"}

# Words that ask for one direction. A split verb ("turn the light on") counts
# up to four words apart.
_SPLIT_VERB = re.compile(
    r"\b(?:turn|switch|power)\s+(?:[\w'-]+\s+){0,4}?(on|off)\b", re.IGNORECASE
)
_ON_WORDS = re.compile(r"\b(open|lock|press|activate|enable)\b", re.IGNORECASE)
_OFF_WORDS = re.compile(r"\b(close|shut|unlock|deactivate|disable)\b", re.IGNORECASE)
# A request that says what not to do is not read.
_NEGATION = re.compile(r"\b(?:not|never|don't|dont|do not|n't)\b|n't\b", re.IGNORECASE)
_MAX_SCAN_CHARS = 2_000

# What each tool does, by domain (Home Assistant's OnOffIntentHandler).
_DONE: dict[str, tuple[str, str]] = {
    "cover": ("opened", "closed"),
    "valve": ("opened", "closed"),
    "lock": ("locked", "unlocked"),
    "button": ("pressed", "turned off"),
    "input_button": ("pressed", "turned off"),
}
_DEFAULT_DONE = ("turned on", "turned off")
_REQUEST_WORDS = {ON: "open / turn on / lock", OFF: "close / turn off / unlock"}


def tool_direction(tool_name: str) -> str | None:
    """Return ``on`` for HassTurnOn, ``off`` for HassTurnOff, else None."""
    return _TOOL_DIRECTIONS.get(base_tool_name(tool_name))


def requested_direction(text: str) -> tuple[str, str] | None:
    """
    Return the one direction a request asks for, and the word that says it.

    None when it says none, both, or what not to do.
    """
    text = text[:_MAX_SCAN_CHARS]
    if not text or _NEGATION.search(text):
        return None
    found: dict[str, str] = {}
    for match in _SPLIT_VERB.finditer(text):
        found.setdefault(match.group(1).lower(), match.group(0))
    for direction, pattern in ((ON, _ON_WORDS), (OFF, _OFF_WORDS)):
        for match in pattern.finditer(text):
            found.setdefault(direction, match.group(1))
    if len(found) != 1:
        return None
    direction, word = next(iter(found.items()))
    return direction, word


def wrong_direction_refusal(tool_name: str, request_text: str) -> str | None:
    """
    Return why the call is not run, when it goes against the request.

    None when the tool is not HassTurnOn/HassTurnOff, or the request does not
    say one direction, or the call goes the way it says.
    """
    called = tool_direction(tool_name)
    asked = requested_direction(request_text)
    if called is None or asked is None or asked[0] == called:
        return None
    direction, word = asked
    right = tool_name.removesuffix(base_tool_name(tool_name)) + _TOOL_FOR[direction]
    return (
        f"Not run: the user asked to '{word}', and {tool_name} does the opposite "
        f"({_REQUEST_WORDS[called]}). Call {right} with the same arguments instead."
    )


def with_action_performed(tool_name: str, data: Any) -> Any:
    """
    Return the result with what was done to each entity added.

    Only for a HassTurnOn/HassTurnOff result naming the entities it acted on;
    anything else is returned as it is.
    """
    called = tool_direction(tool_name)
    if called is None or not isinstance(data, dict):
        return data
    inner = data.get("data")
    success = inner.get("success") if isinstance(inner, dict) else None
    if not isinstance(success, list):
        return data
    done: list[str] = []
    for target in success:
        if not isinstance(target, dict):
            continue
        entity_id = str(target.get("id") or "")
        if target.get("type") != "entity" or "." not in entity_id:
            continue
        verbs = _DONE.get(entity_id.split(".", 1)[0], _DEFAULT_DONE)
        verb = verbs[0] if called == ON else verbs[1]
        done.append(f"{verb} {target.get('name') or entity_id}")
    if not done:
        return data
    return {**data, "action_performed": done}
