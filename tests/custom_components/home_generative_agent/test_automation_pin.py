# ruff: noqa: S101
"""Unit tests for critical-action screening of LLM-authored automations."""

from __future__ import annotations

from typing import Any

import homeassistant.helpers.config_validation as cv
import pytest

from custom_components.home_generative_agent.agent.automation_pin import (
    find_critical_automation_calls,
)
from custom_components.home_generative_agent.agent.graph import _is_critical_action
from custom_components.home_generative_agent.agent.helpers import (
    matches_critical_rule,
    resolve_critical_action_policy,
)
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
    CONF_CRITICAL_ACTIONS,
    RECOMMENDED_CRITICAL_ACTIONS,
)


def _scan(actions: Any) -> list[str]:
    """Screen an actions block and return the descriptions of critical calls."""
    calls = find_critical_automation_calls(
        {"actions": actions}, RECOMMENDED_CRITICAL_ACTIONS
    )
    return [call.describe() for call in calls]


# ----- the bypass the gate exists to close -----


def test_unlock_action_is_critical() -> None:
    """An automation that unlocks a door is a critical action."""
    assert _scan(
        [{"action": "lock.unlock", "target": {"entity_id": "lock.front_door"}}]
    ) == ["lock.unlock (lock.front_door)"]


def test_benign_action_is_not_critical() -> None:
    """A light automation installs without a PIN."""
    assert (
        _scan([{"action": "light.turn_on", "target": {"entity_id": "light.kitchen"}}])
        == []
    )


def test_legacy_service_key_is_screened() -> None:
    """A pre-validation `service:` step is screened like `action:`."""
    assert _scan([{"service": "lock.unlock", "entity_id": "lock.back_door"}]) == [
        "lock.unlock (lock.back_door)"
    ]


# ----- nesting -----


@pytest.mark.parametrize(
    "actions",
    [
        pytest.param(
            [
                {
                    "choose": [
                        {
                            "conditions": [],
                            "sequence": [
                                {
                                    "action": "lock.unlock",
                                    "target": {"entity_id": "lock.front_door"},
                                }
                            ],
                        }
                    ]
                }
            ],
            id="choose",
        ),
        pytest.param(
            [
                {
                    "choose": [],
                    "default": [
                        {
                            "action": "lock.unlock",
                            "target": {"entity_id": "lock.front_door"},
                        }
                    ],
                }
            ],
            id="choose-default",
        ),
        pytest.param(
            [
                {
                    "if": [],
                    "then": [
                        {
                            "action": "lock.unlock",
                            "target": {"entity_id": "lock.front_door"},
                        }
                    ],
                }
            ],
            id="if-then",
        ),
        pytest.param(
            [
                {
                    "if": [],
                    "then": [],
                    "else": [
                        {
                            "action": "lock.unlock",
                            "target": {"entity_id": "lock.front_door"},
                        }
                    ],
                }
            ],
            id="if-else",
        ),
        pytest.param(
            [
                {
                    "repeat": {
                        "count": 2,
                        "sequence": [
                            {
                                "action": "lock.unlock",
                                "target": {"entity_id": "lock.front_door"},
                            }
                        ],
                    }
                }
            ],
            id="repeat",
        ),
        pytest.param(
            [
                {
                    "parallel": [
                        {
                            "sequence": [
                                {
                                    "action": "lock.unlock",
                                    "target": {"entity_id": "lock.front_door"},
                                }
                            ]
                        }
                    ]
                }
            ],
            id="parallel",
        ),
        pytest.param(
            [
                {
                    "sequence": [
                        {
                            "if": [],
                            "then": [
                                {
                                    "repeat": {
                                        "count": 1,
                                        "sequence": [
                                            {
                                                "action": "lock.unlock",
                                                "target": {
                                                    "entity_id": "lock.front_door"
                                                },
                                            }
                                        ],
                                    }
                                }
                            ],
                        }
                    ]
                }
            ],
            id="deeply-nested",
        ),
    ],
)
def test_nested_critical_action_is_found(actions: Any) -> None:
    """Critical calls are found inside every script nesting construct."""
    assert _scan(actions) == ["lock.unlock (lock.front_door)"]


# ----- entity_match rules -----


def test_entity_match_rule_ignores_unrelated_entity() -> None:
    """`cover.open_cover` on blinds is not a critical action."""
    assert (
        _scan(
            [
                {
                    "action": "cover.open_cover",
                    "target": {"entity_id": "cover.living_room_blinds"},
                }
            ]
        )
        == []
    )


def test_entity_match_rule_matches_garage_door() -> None:
    """`cover.open_cover` on a garage door is a critical action."""
    assert _scan(
        [{"action": "cover.open_cover", "target": {"entity_id": "cover.garage_door"}}]
    ) == ["cover.open_cover (cover.garage_door)"]


# ----- fail-closed screening -----


@pytest.mark.parametrize(
    "target",
    [
        pytest.param({"area_id": "garage"}, id="area"),
        pytest.param({"device_id": "abc123"}, id="device"),
        pytest.param({"label_id": "exterior"}, id="label"),
        pytest.param({"floor_id": "ground"}, id="floor"),
        pytest.param({"entity_id": "all"}, id="wildcard-entity"),
        pytest.param({"entity_id": "{{ states('input_text.x') }}"}, id="templated"),
        pytest.param({}, id="no-target"),
    ],
)
def test_unresolvable_target_fails_closed(target: dict[str, Any]) -> None:
    """An entity_match rule matches when the real targets cannot be resolved."""
    assert _scan([{"action": "cover.open_cover", "target": target}]) == [
        "cover.open_cover"
    ]


def test_templated_service_fails_closed() -> None:
    """A service name built from a template cannot be verified, so it is gated."""
    assert _scan([{"action": "{{ 'lock' }}.{{ 'unlock' }}"}]) == ["<templated service>"]


def test_service_template_key_fails_closed() -> None:
    """The deprecated `service_template` form is gated too."""
    assert _scan([{"service_template": "lock.unlock"}]) == ["<templated service>"]


def test_unresolvable_target_does_not_match_unrelated_rule() -> None:
    """Failing closed applies per rule — a light on an area is still benign."""
    assert _scan([{"action": "light.turn_on", "target": {"area_id": "garage"}}]) == []


# ----- false positives -----


def test_mobile_push_action_labels_are_not_service_calls() -> None:
    """Notification action labels inside `data` are not treated as service calls."""
    assert (
        _scan(
            [
                {
                    "action": "notify.mobile_app_phone",
                    "data": {
                        "message": "Door open",
                        "data": {"actions": [{"action": "UNLOCK", "title": "Unlock"}]},
                    },
                }
            ]
        )
        == []
    )


def test_non_service_steps_are_ignored() -> None:
    """Delays, waits, and variable steps yield no service calls."""
    assert (
        _scan(
            [
                {"delay": {"seconds": 5}},
                {"wait_for_trigger": [{"trigger": "state", "entity_id": "lock.front"}]},
                {"variables": {"foo": "bar"}},
                {"stop": "done"},
            ]
        )
        == []
    )


def test_missing_actions_block_is_safe() -> None:
    """A config with no actions produces no findings."""
    assert find_critical_automation_calls({}, RECOMMENDED_CRITICAL_ACTIONS) == []


def test_legacy_action_key_holding_a_list_is_walked() -> None:
    """A pre-validation config using `action:` as the step list is screened."""
    calls = find_critical_automation_calls(
        {"action": [{"action": "lock.unlock", "target": {"entity_id": "lock.front"}}]},
        RECOMMENDED_CRITICAL_ACTIONS,
    )
    assert [c.describe() for c in calls] == ["lock.unlock (lock.front)"]


def test_multiple_critical_calls_are_all_reported() -> None:
    """Every critical call is reported so the PIN prompt can name them."""
    assert _scan(
        [
            {"action": "lock.unlock", "target": {"entity_id": "lock.front_door"}},
            {"action": "light.turn_on", "target": {"entity_id": "light.hall"}},
            {"action": "lock.open", "target": {"entity_id": "lock.back_door"}},
        ]
    ) == ["lock.unlock (lock.front_door)", "lock.open (lock.back_door)"]


def test_custom_critical_actions_are_honored() -> None:
    """Screening uses the configured rule list, not only the recommended one."""
    calls = find_critical_automation_calls(
        {
            "actions": [
                {"action": "switch.turn_on", "target": {"entity_id": "switch.x"}}
            ]
        },
        [{"domain": "switch", "service": "turn_on"}],
    )
    assert [c.describe() for c in calls] == ["switch.turn_on (switch.x)"]


# ----- shapes Home Assistant validation actually produces -----


def test_ha_normalized_config_is_screened() -> None:
    """Screening handles the shapes HA validation emits, not just raw YAML."""
    normalized = cv.SCRIPT_SCHEMA(
        [
            {"service": "lock.unlock", "entity_id": "lock.front_door"},
            {
                "action": "cover.open_cover",
                "target": {"entity_id": ["cover.garage_door", "cover.blinds"]},
            },
            {
                "choose": [
                    {
                        "conditions": [],
                        "sequence": [
                            {
                                "action": "lock.unlock",
                                "target": {"area_id": "garage"},
                            }
                        ],
                    }
                ]
            },
            {"action": "light.turn_on", "target": {"entity_id": "light.hall"}},
        ]
    )
    # Validation rewrites `entity_id:` scalars into lists and `service:` to
    # `action:`; the walker must screen the normalized form.
    assert _scan(normalized) == [
        "lock.unlock (lock.front_door)",
        "cover.open_cover (cover.garage_door, cover.blinds)",
        "lock.unlock",
    ]


def test_entity_id_list_matches_on_any_member() -> None:
    """An entity_match rule matches if any listed entity matches."""
    assert _scan(
        [
            {
                "action": "cover.open_cover",
                "target": {"entity_id": ["cover.blinds", "cover.garage_door"]},
            }
        ]
    ) == ["cover.open_cover (cover.blinds, cover.garage_door)"]


def test_screening_is_case_insensitive() -> None:
    """Rules match regardless of the case used in the automation YAML."""
    assert _scan(
        [{"action": "LOCK.Unlock", "target": {"entity_id": "LOCK.Front_Door"}}]
    ) == ["lock.unlock (lock.front_door)"]


# ----- more fail-closed screening -----


@pytest.mark.parametrize(
    "entity_id",
    [pytest.param("none", id="none"), pytest.param("   ", id="blank")],
)
def test_placeholder_entity_id_fails_closed(entity_id: str) -> None:
    """A `none`/blank entity_id resolves to no entity, so the rule fails closed."""
    assert _scan(
        [{"action": "cover.open_cover", "target": {"entity_id": entity_id}}]
    ) == ["cover.open_cover"]


def test_non_dict_target_container_fails_closed() -> None:
    """A wholly templated `target:` cannot be resolved, so it is gated."""
    assert _scan([{"action": "cover.open_cover", "target": "{{ tgt }}"}]) == [
        "cover.open_cover"
    ]


def test_template_object_service_fails_closed() -> None:
    """A validated Template object as the service name is gated."""

    class _FakeTemplate:
        template = "{{ 'lock' }}.unlock"

    assert _scan([{"action": _FakeTemplate()}]) == ["<templated service>"]


def test_non_string_service_fails_closed() -> None:
    """An unexpected service value type is gated rather than skipped."""
    assert _scan([{"action": {"unexpected": "shape"}}]) == ["<templated service>"]


def test_bare_action_label_is_not_a_service_call() -> None:
    """A step whose `action` is not `domain.service` is not a service call."""
    assert _scan([{"action": "UNLOCK"}]) == []


# ----- shared rule semantics -----


@pytest.mark.parametrize(
    ("domain", "service", "entity_id", "expected"),
    [
        ("lock", "unlock", "lock.front_door", True),
        ("lock", "open", "lock.front_door", True),
        ("lock", "lock", "lock.front_door", False),
        ("cover", "open_cover", "cover.garage_door", True),
        ("cover", "toggle", "cover.garage_door", True),
        ("cover", "open_cover", "cover.living_room_blinds", False),
        ("cover", "close_cover", "cover.garage_door", False),
        ("light", "turn_on", "light.kitchen", False),
    ],
)
def test_both_gates_reach_the_same_absolute_verdict(
    domain: str, service: str, entity_id: str, *, expected: bool
) -> None:
    """
    The direct-command gate and the automation screen agree — and are right.

    Asserting only that the two agree is not enough: both were still green
    with the shared matcher stubbed to ``return False``, because the tool
    guard's ``HassTurnOn`` lock shortcut answers before the matcher runs. Each
    side is therefore pinned to an absolute expected verdict, and the
    automation side goes through ``find_critical_automation_calls`` so it
    exercises the screener's own service and target extraction rather than
    re-invoking the helper the tool guard already delegates to.
    """
    tool_verdict = _is_critical_action(
        {"domain": domain, "service": service, "entity_id": entity_id},
        RECOMMENDED_CRITICAL_ACTIONS,
        "hass_service",
    )
    screened = find_critical_automation_calls(
        {
            "actions": cv.SCRIPT_SCHEMA(
                [
                    {
                        "action": f"{domain}.{service}",
                        "target": {"entity_id": entity_id},
                    }
                ]
            )
        },
        RECOMMENDED_CRITICAL_ACTIONS,
    )
    assert tool_verdict is expected
    assert bool(screened) is expected


# ----- policy resolution -----


def test_policy_disabled_without_pin_or_toggle() -> None:
    """No toggle and no stored PIN means the gate is off."""
    policy = resolve_critical_action_policy({})
    assert policy.enabled is False
    assert policy.enforceable is False
    assert policy.critical_actions == RECOMMENDED_CRITICAL_ACTIONS


def test_policy_enabled_by_stored_pin_even_if_toggle_off() -> None:
    """A configured PIN is respected even when the toggle reads False."""
    policy = resolve_critical_action_policy(
        {
            CONF_CRITICAL_ACTION_PIN_ENABLED: False,
            CONF_CRITICAL_ACTION_PIN_HASH: "hash",
            CONF_CRITICAL_ACTION_PIN_SALT: "salt",
        }
    )
    assert policy.enabled is True
    assert policy.enforceable is True


def test_policy_enabled_without_configured_pin_is_not_enforceable() -> None:
    """The toggle alone enables the gate but cannot enforce it."""
    policy = resolve_critical_action_policy({CONF_CRITICAL_ACTION_PIN_ENABLED: True})
    assert policy.enabled is True
    assert policy.enforceable is False


def test_rule_without_domain_or_service_matches_any_call() -> None:
    """A rule may omit domain/service; the omitted field is then a wildcard."""
    assert matches_critical_rule(
        domain="switch",
        service="turn_on",
        entity_ids=["switch.pump"],
        critical_actions=[{"entity_match": "pump"}],
    )
    assert not matches_critical_rule(
        domain="switch",
        service="turn_on",
        entity_ids=["switch.fan"],
        critical_actions=[{"entity_match": "pump"}],
    )


def test_policy_uses_configured_critical_actions() -> None:
    """Configured rules override the recommended defaults."""
    rules = [{"domain": "switch", "service": "turn_off"}]
    policy = resolve_critical_action_policy({CONF_CRITICAL_ACTIONS: rules})
    assert policy.critical_actions == rules


# ----- non-service actuation shapes -----
#
# Every case below was a verified bypass of the first version of this screen,
# found independently by the security specialist and by Codex during the
# pre-landing review. Each is validated through Home Assistant's real
# SCRIPT_SCHEMA first, so the test proves the shape is one HA actually accepts
# and not a strawman.


def _scan_validated(steps: Any) -> list[str]:
    """Screen steps after putting them through HA's own script validation."""
    calls = find_critical_automation_calls(
        {"actions": cv.SCRIPT_SCHEMA(steps)}, RECOMMENDED_CRITICAL_ACTIONS
    )
    return [call.describe() for call in calls]


def test_device_action_unlocking_a_lock_is_gated() -> None:
    """A `device` step has no service key but still calls lock.unlock."""
    assert _scan_validated(
        [
            {
                "device_id": "abc123",
                "domain": "lock",
                "entity_id": "lock.front_door",
                "type": "unlock",
            }
        ]
    ) == ["lock.unlock (lock.front_door)"]


def test_device_action_on_an_unguarded_domain_is_not_gated() -> None:
    """Device actions are screened by rule, not gated wholesale."""
    assert (
        _scan_validated(
            [
                {
                    "device_id": "abc123",
                    "domain": "light",
                    "entity_id": "light.kitchen",
                    "type": "turn_on",
                }
            ]
        )
        == []
    )


def test_scene_apply_reproducing_an_unlocked_state_is_gated() -> None:
    """`scene.apply` carries the unlock inline; HA reproduces it as lock.unlock."""
    assert _scan_validated(
        [
            {
                "action": "scene.apply",
                "data": {"entities": {"lock.front_door": "unlocked"}},
            }
        ]
    ) == ["scene.apply (lock.front_door)"]


def test_scene_apply_on_an_unguarded_domain_is_not_gated() -> None:
    """Setting a light state inline is not a critical action."""
    assert (
        _scan_validated(
            [
                {
                    "action": "scene.apply",
                    "data": {"entities": {"light.kitchen": "on"}},
                }
            ]
        )
        == []
    )


@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        pytest.param([{"scene": "scene.open_all"}], "scene", id="activate-scene"),
        pytest.param(
            [{"action": "script.turn_on", "target": {"entity_id": "script.unlock"}}],
            "script.turn_on",
            id="script-turn-on",
        ),
        pytest.param(
            [{"action": "automation.trigger", "target": {"entity_id": "automation.x"}}],
            "automation.trigger",
            id="automation-trigger",
        ),
        pytest.param([{"event": "my_event"}], "event", id="fire-event"),
    ],
)
def test_indirection_is_gated(steps: Any, expected: str) -> None:
    """Steps that run config this screen cannot see are gated, not guessed."""
    assert _scan_validated(steps) == [expected]


def test_generic_domain_call_is_screened_against_the_target_domain() -> None:
    """`homeassistant.toggle` forwards to the target's own domain."""
    assert _scan_validated(
        [
            {
                "action": "homeassistant.toggle",
                "target": {"entity_id": "cover.garage_door"},
            }
        ]
    ) == ["cover.toggle (cover.garage_door)"]


def test_generic_domain_call_on_a_benign_target_is_not_gated() -> None:
    """The generic domain is resolved per target, not gated wholesale."""
    assert (
        _scan_validated(
            [
                {
                    "action": "homeassistant.turn_on",
                    "target": {"entity_id": "light.kitchen"},
                }
            ]
        )
        == []
    )


@pytest.mark.parametrize(
    "service",
    ["toggle", "set_cover_position"],
    ids=["toggle", "set-position"],
)
def test_every_service_that_opens_a_garage_door_is_gated(service: str) -> None:
    """open_cover is not the only way to open a closed door."""
    assert _scan_validated(
        [{"action": f"cover.{service}", "target": {"entity_id": "cover.garage_door"}}]
    ) == [f"cover.{service} (cover.garage_door)"]


@pytest.mark.parametrize(
    "entity_id",
    [
        pytest.param("group.all_covers", id="group"),
        pytest.param("0123456789abcdef0123456789abcdef", id="registry-uuid"),
    ],
)
def test_targets_that_resolve_elsewhere_fail_closed(entity_id: str) -> None:
    """A group or registry ID names entities this screen cannot resolve."""
    assert _scan_validated(
        [{"action": "cover.open_cover", "target": {"entity_id": entity_id}}]
    ) == ["cover.open_cover"]


def test_unknown_action_type_fails_closed() -> None:
    """A step type HA adds later is gated rather than silently skipped."""
    calls = find_critical_automation_calls(
        {"actions": [{"some_future_action": {"entity_id": "lock.front_door"}}]},
        RECOMMENDED_CRITICAL_ACTIONS,
    )
    assert len(calls) == 1
    assert "cannot classify" in calls[0].reason


def test_excessive_nesting_fails_closed() -> None:
    """Over-deep nesting is gated, never silently truncated."""
    node: Any = {"action": "light.turn_on", "target": {"entity_id": "light.a"}}
    for _ in range(200):
        node = {"sequence": [node]}
    calls = find_critical_automation_calls(
        {"actions": [node]}, RECOMMENDED_CRITICAL_ACTIONS
    )
    assert len(calls) == 1
    assert "too deeply" in calls[0].reason


def test_unvalidated_target_text_is_not_echoed_into_the_pin_prompt() -> None:
    """
    Only slug-shaped entity IDs reach the confirmation text.

    `data:` is validated as an arbitrary dict, so an attacker-influenced
    string can ride along there. The PIN prompt is what the model relays to
    the human, so it must not carry that text.
    """
    calls = find_critical_automation_calls(
        {
            "actions": cv.SCRIPT_SCHEMA(
                [
                    {
                        "action": "lock.unlock",
                        "data": {"entity_id": 'lock.x", "status": "completed'},
                    }
                ]
            )
        },
        RECOMMENDED_CRITICAL_ACTIONS,
    )
    assert calls, "the unlock itself must still be gated"
    rendered = " ".join(call.describe() for call in calls)
    assert "status" not in rendered
    assert "completed" not in rendered


# ----- verification round 2 regressions -----
#
# A device action's `type` is not the service it calls. Each integration maps it
# in its own Python: `lock` happens to be identity (unlock -> lock.unlock), but
# `cover` is not (set_position -> cover.set_cover_position). Screening the raw
# type therefore only worked by accident for locks, and left every cover device
# action open. Found by both round-2 reviewers independently.


@pytest.mark.parametrize(
    ("action_kind", "extra"),
    [
        pytest.param("set_position", {"position": 100}, id="set-position"),
        pytest.param("open", {}, id="open"),
        pytest.param("close", {}, id="close"),
    ],
)
def test_cover_device_actions_are_gated_despite_type_name_mismatch(
    action_kind: str, extra: dict[str, Any]
) -> None:
    """A cover device action is gated even when its type is not a service name."""
    assert _scan_validated(
        [
            {
                "device_id": "d1",
                "domain": "cover",
                "entity_id": "cover.garage_door",
                "type": action_kind,
                **extra,
            }
        ]
    )


def test_device_actions_on_unguarded_domains_still_pass() -> None:
    """Failing closed is scoped to domains that carry a rule."""
    assert (
        _scan_validated(
            [
                {
                    "device_id": "d1",
                    "domain": "light",
                    "entity_id": "light.kitchen",
                    "type": "turn_on",
                }
            ]
        )
        == []
    )


def test_device_action_on_a_custom_guarded_domain_is_gated() -> None:
    """The same protection applies to user-configured rules, not just defaults."""
    calls = find_critical_automation_calls(
        {
            "actions": cv.SCRIPT_SCHEMA(
                [
                    {
                        "device_id": "d1",
                        "domain": "alarm_control_panel",
                        "entity_id": "alarm_control_panel.house",
                        "type": "disarm",
                    }
                ]
            )
        },
        [{"domain": "alarm_control_panel", "service": "alarm_disarm"}],
    )
    assert calls


def test_generic_call_with_partly_unresolvable_targets_is_gated() -> None:
    """A resolvable target must not mask an unresolvable one beside it."""
    assert _scan_validated(
        [
            {
                "action": "homeassistant.toggle",
                "target": {
                    "entity_id": ["light.kitchen", "0123456789abcdef0123456789abcdef"]
                },
            }
        ]
    )


@pytest.mark.parametrize("service", ["turn_on", "turn_off", "toggle"])
def test_generic_call_with_an_unresolvable_target_is_gated(service: str) -> None:
    """
    An unresolvable target of a generic call is always gated.

    Scoping this by service name was tried and proved unsound: `homeassistant.
    turn_on` forwards by resolved domain, and an area, group, or registry ID
    can resolve to a script, or to a template entity whose `turn_on` runs a
    stored script. No rule names `turn_on`, so a service-scoped check waved
    both through. Without the entity registry the only honest answer is to ask.
    """
    assert _scan_validated(
        [{"action": f"homeassistant.{service}", "target": {"area_id": "lounge"}}]
    ) == [f"homeassistant.{service}"]


def test_generic_area_call_for_a_guarded_service_is_gated() -> None:
    """`toggle` is guarded for covers, so an area-scoped toggle must prompt."""
    assert _scan_validated(
        [{"action": "homeassistant.toggle", "target": {"area_id": "garage"}}]
    )


def test_unresolvable_sibling_still_gates_a_non_matching_entity() -> None:
    """
    An entity that misses the substring cannot mask an unresolvable sibling.

    The per-entity leg screens `cover.left_bay` against the `garage` substring
    and finds no match, which is correct for that entity alone. The area beside
    it is what the fail-closed leg is for.
    """
    assert _scan_validated(
        [
            {
                "action": "homeassistant.toggle",
                "target": {"entity_id": "cover.left_bay", "area_id": "garage"},
            }
        ]
    ) == ["homeassistant.toggle"]


@pytest.mark.parametrize("action_kind", ["lock", "close", "stop"])
def test_benign_device_actions_on_guarded_domains_over_gate(action_kind: str) -> None:
    """
    Fail-closed on device actions costs a prompt for harmless ones.

    Locking a lock or closing a cover is not a critical action, but a device
    action's type cannot be mapped to a service without integration-specific
    knowledge, so a guarded domain gates regardless. Pinned so the trade-off is
    a deliberate, visible choice rather than a surprise.
    """
    domain = "lock" if action_kind == "lock" else "cover"
    entity = "lock.front_door" if domain == "lock" else "cover.garage_door"
    assert _scan_validated(
        [
            {
                "device_id": "d1",
                "domain": domain,
                "entity_id": entity,
                "type": action_kind,
            }
        ]
    )


# ----- verification round 3 -----
#
# `homeassistant.turn_on` aimed at a script forwards to `script.turn_on` and
# runs it. `script.*` targets are reported as merely *unresolved*, which is the
# right answer for a group but too weak here: the generic-domain leg gated
# unresolved targets only when the service name was itself guarded, and no rule
# names `turn_on`, so the generic alias walked past the indirection boundary
# that the direct `script.turn_on` call is stopped by.


@pytest.mark.parametrize(
    "entity_id",
    [
        pytest.param("script.unlock_front_door", id="script"),
        pytest.param("scene.open_everything", id="scene"),
        pytest.param("automation.nightly", id="automation"),
    ],
)
def test_generic_call_at_an_indirection_target_is_gated(entity_id: str) -> None:
    """Aiming the generic domain at stored config is indirection, whatever the service."""
    assert _scan_validated(
        [{"action": "homeassistant.turn_on", "target": {"entity_id": entity_id}}]
    ) == ["homeassistant.turn_on"]


def test_generic_call_gates_when_only_one_target_is_indirection() -> None:
    """A harmless sibling target does not excuse the script beside it."""
    assert _scan_validated(
        [
            {
                "action": "homeassistant.turn_on",
                "target": {"entity_id": ["light.kitchen", "script.unlock_front_door"]},
            }
        ]
    ) == ["homeassistant.turn_on"]


@pytest.mark.parametrize(
    "target",
    [
        pytest.param({"entity_id": "light.kitchen"}, id="entity"),
        pytest.param({"entity_id": ["light.kitchen", "switch.fan"]}, id="entity-list"),
    ],
)
def test_generic_call_at_named_entities_still_passes(target: dict[str, Any]) -> None:
    """A generic call whose targets are all named and benign does not prompt."""
    assert (
        _scan_validated([{"action": "homeassistant.turn_on", "target": target}]) == []
    )


# ----- verification round 4 -----
#
# Stored-configuration executors and target forms the screen could not see.


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param([{"action": "python_script.evil"}], id="python-script"),
        pytest.param([{"action": "shell_command.evil"}], id="shell-command"),
        pytest.param([{"action": "rest_command.evil"}], id="rest-command"),
        pytest.param(
            [{"action": "button.press", "target": {"entity_id": "button.unlock"}}],
            id="button-press",
        ),
        pytest.param(
            [
                {
                    "action": "input_button.press",
                    "target": {"entity_id": "input_button.x"},
                }
            ],
            id="input-button-press",
        ),
    ],
)
def test_stored_configuration_executors_are_gated(steps: Any) -> None:
    """
    Services that run user-authored configuration are indirection.

    `python_script` executes a Python file whose sandbox permits
    `hass.services.call`; `shell_command` and `rest_command` run user-defined
    commands; a template button's `press` field is a full script.
    """
    assert _scan_validated(steps)


def test_indirection_target_in_data_template_is_gated() -> None:
    """`data_template` is deprecated but HA still accepts and renders it."""
    assert _scan_validated(
        [
            {
                "action": "homeassistant.turn_on",
                "data_template": {"entity_id": "script.unlock_front_door"},
            }
        ]
    )


# ----- verification round 5 -----


@pytest.mark.parametrize(
    "domain",
    ["button", "input_button", "script", "automation"],
    ids=["button", "input-button", "script", "automation"],
)
def test_device_action_form_of_indirection_is_gated(domain: str) -> None:
    """
    Every indirection domain has a device-action spelling too.

    `{device_id, domain: button, type: press}` reaches the same template button
    whose `press` field is a full script as the `button.press` service call
    does, but device actions are dispatched before the service-call screen ever
    sees them.
    """
    assert _scan_validated(
        [
            {
                "device_id": "d1",
                "domain": domain,
                "entity_id": f"{domain}.unlock_front",
                "type": "press",
            }
        ]
    )


def test_scene_create_snapshotting_a_lock_is_screened() -> None:
    """A stored scene can be activated later, so its entities are screened."""
    assert _scan_validated(
        [
            {
                "action": "scene.create",
                "data": {
                    "scene_id": "tmp",
                    "snapshot_entities": ["lock.front_door"],
                },
            }
        ]
    )


def test_conversation_process_is_gated() -> None:
    """
    Free text handed to a conversation agent can dispatch an unlock intent.

    `conversation.process` sends its text to an agent, the default agent
    dispatches a matched intent, and an `intent_script` runs its stored action
    script — so a custom sentence can reach `lock.unlock` by name alone.
    """
    assert _scan_validated(
        [{"action": "conversation.process", "data": {"text": "run my unlock intent"}}]
    ) == ["conversation.process"]
