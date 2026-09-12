# ruff: noqa: S101
"""
Unit tests for Home Assistant's namespaced LLM tool names.

HA 2026.9 moved the built-in intent tools into per-integration `llm.py`
modules that name them `f"{DOMAIN}__{intent_type}"` — `intent__HassTurnOff`,
`light__HassLightSet`, `homeassistant__GetLiveContext` — and `MergedAPI` can
prepend a second namespace on top of that. Every comparison this integration
makes against a known tool name therefore has to run on the BASE name.

These are regression tests for a silent upgrade breakage: nothing raised, no
error was logged, the guards simply stopped matching. The lock PIN gate and
the actuation classifier both went dead, and the tool index filled with a
shadow copy of the old names that outranked the live ones in vector search.
"""

from __future__ import annotations

import pytest

from custom_components.home_generative_agent.agent.graph import (
    _is_critical_action,
    _is_live_context_tool,
    _minimal_payload_for_domain,
    _tool_lookup_targets,
)
from custom_components.home_generative_agent.agent.helpers import (
    base_tool_name,
    is_actuation_tool,
    is_on_off_intent,
    normalize_intent_for_alarm,
    normalize_intent_for_lock,
    sanitize_tool_args,
)

LOCK_RULES = [{"domain": "lock", "service": "unlock"}]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("HassTurnOff", "HassTurnOff"),
        ("intent__HassTurnOff", "HassTurnOff"),
        ("light__HassLightSet", "HassLightSet"),
        ("homeassistant__GetLiveContext", "GetLiveContext"),
        # MergedAPI wraps an already-namespaced tool in a second namespace.
        ("assist__intent__HassTurnOn", "HassTurnOn"),
        ("alarm_control", "alarm_control"),
        ("", ""),
    ],
)
def test_base_tool_name_strips_ha_namespaces(name: str, expected: str) -> None:
    """The base name is what every known-name comparison must match on."""
    assert base_tool_name(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "intent__HassTurnOn",
        "intent__HassTurnOff",
        "light__HassLightSet",
        "media_player__HassMediaPlayerMute",
        "lock__HassTurnOff",
        "HassTurnOn",
        "alarm_control",
    ],
)
def test_namespaced_actuation_tools_are_still_actuation(name: str) -> None:
    """
    The prefix list names HA *intents*, not HA *domains*.

    `ACTUATION_TOOL_PREFIXES` is matched with `startswith`, so under the new
    naming every HA actuation tool started with its domain and classified as
    non-actuation. That silently disabled the actuation-safety retrieval pass,
    the read-only open-state actuation strip, the fallback ordering, and the
    routing-rejection recovery, all at once.
    """
    assert is_actuation_tool(name)


@pytest.mark.parametrize(
    "name",
    ["homeassistant__GetLiveContext", "GetLiveContext", "llm__GetDateTime"],
)
def test_read_only_tools_are_not_actuation(name: str) -> None:
    """Stripping the namespace must not make read-only tools look like actuation."""
    assert not is_actuation_tool(name)


@pytest.mark.parametrize(
    "tool_name",
    ["HassTurnOff", "intent__HassTurnOff", "lock__HassTurnOff"],
)
def test_lock_intent_is_critical_under_any_spelling(tool_name: str) -> None:
    """
    A lock command must hit the PIN gate whatever HA calls the tool.

    `_is_critical_action` short-circuits to True for turn-on/turn-off intents
    aimed at a lock precisely because those calls carry no `service` arg for
    the generic rule match to work on. Keyed on the bare name, that
    short-circuit stopped firing on HA 2026.9 — and the generic path cannot
    cover for it, so unlocking a door through `intent__HassTurnOff` skipped
    PIN verification entirely.
    """
    args = {"name": "Garage Door Lock", "domain": ["lock"]}
    assert _is_critical_action(args, LOCK_RULES, tool_name)


@pytest.mark.parametrize(
    "tool_name",
    ["HassTurnOn", "intent__HassTurnOn", "alarm_control_panel__HassTurnOn"],
)
def test_alarm_intent_keeps_its_carve_out_under_any_spelling(tool_name: str) -> None:
    """Alarm panels enforce their own code and must never take the generic PIN."""
    args = {"name": "Home Alarm", "domain": ["alarm_control_panel"]}
    rules = [{"domain": "alarm_control_panel", "service": "alarm_disarm"}]
    assert not _is_critical_action(args, rules, tool_name)


@pytest.mark.parametrize("tool_name", ["HassTurnOff", "intent__HassTurnOff"])
def test_lock_intent_normalization_survives_the_rename(tool_name: str) -> None:
    """Unlock must resolve to the `unlock` service, namespaced or not."""
    out = normalize_intent_for_lock(tool_name, {"name": "Front Door Lock"})
    assert out["domain"] == ["lock"]
    assert out["service"] == "unlock"


@pytest.mark.parametrize("tool_name", ["HassTurnOn", "intent__HassTurnOn"])
def test_alarm_intent_normalization_survives_the_rename(tool_name: str) -> None:
    """Arming must map to `alarm_arm_home`, not fall through to disarm."""
    out = normalize_intent_for_alarm(tool_name, {"name": "Home Alarm"})
    assert out["domain"] == ["alarm_control_panel"]
    assert out["service"] == "alarm_arm_home"


def test_non_intent_tools_are_left_alone_by_the_normalizers() -> None:
    """Base-name matching must not start capturing unrelated tools."""
    args = {"name": "Front Door Lock"}
    assert not is_on_off_intent("get_entity_history")
    assert normalize_intent_for_lock("get_entity_history", args) == args
    assert normalize_intent_for_alarm("get_entity_history", args) == args


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("GetLiveContext", True),
        ("homeassistant__GetLiveContext", True),
        ("get_entity_history", False),
        ("", False),
    ],
)
def test_is_live_context_tool(name: str, *, expected: bool) -> None:
    """Live-context handling keys on the base name in both directions."""
    assert _is_live_context_tool(name) is expected


def test_tool_lookup_targets_resolves_a_bare_name_to_its_live_spelling() -> None:
    """
    Force-injection sites pass the bare name; the index holds the live one.

    The bare key stays first so an install that has not upgraded is unaffected,
    with the live namespaced spelling appended behind it.
    """
    live = {
        ("assist", "homeassistant__GetLiveContext"),
        ("assist", "intent__HassTurnOn"),
    }
    targets = _tool_lookup_targets("GetLiveContext", {"assist"}, live)
    assert targets == [
        ("assist", "GetLiveContext"),
        ("assist", "homeassistant__GetLiveContext"),
    ]


def test_tool_lookup_targets_is_inert_when_live_filtering_is_off() -> None:
    """Fail open: with no live set there is nothing to resolve against."""
    assert _tool_lookup_targets("GetLiveContext", {"assist", "hga_local"}, None) == [
        ("assist", "GetLiveContext"),
        ("hga_local", "GetLiveContext"),
    ]


@pytest.mark.parametrize(
    ("args", "expected_target"),
    [
        ({"name": "Garage Door Lock", "domain": ["lock"]}, "Garage Door Lock"),
        ({"name": "Garage Door Lock"}, "Garage Door Lock"),
        (
            {"entity_id": "lock.garage_door_lock", "domain": ["lock"]},
            "lock.garage_door_lock",
        ),
    ],
)
def test_lock_payload_always_names_one_entity(
    args: dict[str, object], expected_target: str
) -> None:
    """
    A lock command must target ONE lock, never the whole domain.

    Home Assistant's turn-on/turn-off intents resolve targets from
    name/area/floor and never read an `entity_id` slot, while `domains` alone
    already satisfies `MatchTargetsConstraints.has_constraints` -- so a payload
    of only `{"domain": ["lock"], "entity_id": ...}` slips past HA's "cannot
    target all devices" guard and unlocks every exposed lock. Restoring the
    lock normalizer for namespaced tool names is what routed these calls
    through the reducer, so the reducer has to keep a name.
    """
    out = sanitize_tool_args(
        _minimal_payload_for_domain(
            sanitize_tool_args(normalize_intent_for_lock("intent__HassTurnOff", args))
        )
    )
    assert out["domain"] == ["lock"]
    assert out["name"] == expected_target
