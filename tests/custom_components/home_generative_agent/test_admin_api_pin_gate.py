# ruff: noqa: S101
"""
The PIN gate reads the tools of Home Assistant's admin-only API (#918).

Home Assistant 2026.11's "Home Assistant" API (`homeassistant`) has a tool
that calls any action, `homeassistant__call_action`, with the action, its
target and its data as arguments. The gate reads that call as the service it
names, the way it reads an automation step, and lets the API's read-only
tools through.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.core import Context
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import llm
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent.automation_pin import (
    _is_template,
    find_critical_automation_calls,
)
from custom_components.home_generative_agent.agent.graph import (
    _get_allowed_api_ids,
    _is_critical_action,
    _run_ha_tool,
)
from custom_components.home_generative_agent.agent.helpers import is_actuation_tool
from custom_components.home_generative_agent.agent.indirect_pin import (
    CALL_ACTION_TOOL,
    tool_origin,
)
from custom_components.home_generative_agent.agent.pin_messages import (
    describe_pending_action,
)
from custom_components.home_generative_agent.agent.rule_entities import (
    expand_critical_rules,
)
from custom_components.home_generative_agent.agent.template_actions import _templated
from custom_components.home_generative_agent.const import (
    RECOMMENDED_CRITICAL_ACTIONS,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from pytest_homeassistant.common import MockUser

ADMIN_API = "homeassistant"

pytestmark = pytest.mark.skipif(
    not hasattr(llm, "LLM_API_HOME_ASSISTANT"),
    reason="Home Assistant before 2026.11 has no admin-only API",
)

FRONT_DOOR = {"entity_id": "lock.front_door"}


@pytest.fixture
async def admin_api(hass: HomeAssistant, hass_admin_user: MockUser) -> SimpleNamespace:
    """Set up a house and load the admin-only API for an admin's turn."""
    hass.states.async_set("lock.front_door", "locked", {"friendly_name": "Front Door"})
    hass.states.async_set("light.porch", "off", {"friendly_name": "Porch"})
    hass.states.async_set(
        "cover.garage_door",
        "closed",
        {"friendly_name": "Garage Door", "device_class": "garage"},
    )
    for domain, config in (
        ("lock", {}),
        ("light", {}),
        ("cover", {}),
        ("input_button", {"unlock": {}}),
        ("input_boolean", {"home": {}}),
    ):
        assert await async_setup_component(hass, domain, {domain: config})
    assert await async_setup_component(hass, "scene", {"scene": []})
    assert await async_setup_component(
        hass,
        "script",
        {
            "script": {
                "unlock_door": {
                    "sequence": [{"action": "lock.unlock", "target": FRONT_DOOR}]
                },
                "porch": {
                    "sequence": [
                        {
                            "action": "light.turn_on",
                            "target": {"entity_id": "light.porch"},
                        }
                    ]
                },
            }
        },
    )
    assert await async_setup_component(
        hass,
        "automation",
        {
            "automation": [
                {
                    "id": "arrive",
                    "alias": "Unlock on arrival",
                    "triggers": [
                        {
                            "trigger": "state",
                            "entity_id": "input_boolean.home",
                            "to": "on",
                        }
                    ],
                    "actions": [{"action": "lock.unlock", "target": FRONT_DOOR}],
                },
                {
                    "id": "button",
                    "alias": "Button unlock",
                    "triggers": [
                        {"trigger": "state", "entity_id": "input_button.unlock"}
                    ],
                    "actions": [{"action": "lock.unlock", "target": FRONT_DOOR}],
                },
            ]
        },
    )
    hass.states.async_set("cover.shade", "closed", {"friendly_name": "Shade"})
    for domain in ("system_log", "system_health", "search", "config", "llm"):
        assert await async_setup_component(hass, domain, {})
    await hass.async_block_till_done()
    instance = await llm.async_get_api(
        hass,
        ADMIN_API,
        llm.LLMContext(
            platform="home_generative_agent",
            context=Context(user_id=hass_admin_user.id),
            language="en",
            assistant="conversation",
            device_id=None,
        ),
    )
    # Stands in for MultiLLMAPI: importing conversation.py needs hassil.
    return SimpleNamespace(
        apis={ADMIN_API: instance},
        routing_map={},
        remote_home_assistants=frozenset(),
    )


def _held(
    hass: HomeAssistant, api: SimpleNamespace, tool_name: str, args: dict[str, Any]
) -> bool:
    return _is_critical_action(
        dict(args),
        expand_critical_rules(hass, RECOMMENDED_CRITICAL_ACTIONS),
        tool_name,
        hass,
        origin=tool_origin(api, tool_name),
    )


async def test_the_api_is_known_as_an_admin_one(admin_api: SimpleNamespace) -> None:
    """The gate's branch is keyed on the API's own admin flag."""
    origin = tool_origin(admin_api, CALL_ACTION_TOOL)

    assert origin.admin_api
    assert not origin.read_only
    assert not origin.remote
    assert CALL_ACTION_TOOL in {tool.name for tool in admin_api.apis[ADMIN_API].tools}


@pytest.mark.parametrize(
    "args",
    [
        pytest.param({"action": "lock.unlock", "target": FRONT_DOOR}, id="target"),
        pytest.param({"action": "lock.unlock", "data": FRONT_DOOR}, id="data"),
        pytest.param({"action": "lock.unlock"}, id="no-target"),
        pytest.param(
            {"action": "lock.unlock", "target": {"entity_id": "all"}}, id="all"
        ),
        pytest.param(
            {"action": "lock.unlock", "target": {"area_id": "hall"}}, id="area"
        ),
        pytest.param(
            {
                "action": "lock.unlock",
                "target": {"entity_id": "{{ 'lock.' ~ 'front_door' }}"},
            },
            id="templated-target",
        ),
        pytest.param(
            {"action": "homeassistant.toggle", "target": FRONT_DOOR},
            id="generic-toggle",
        ),
        pytest.param(
            {
                "action": "homeassistant.turn_on",
                "target": {
                    "entity_id": "{{ states.lock | map(attribute='entity_id') | list }}"
                },
            },
            id="generic-templated",
        ),
        pytest.param(
            {
                "action": "homeassistant.turn_on",
                "data": {"{{ 'entity_id' }}": "lock.front_door"},
            },
            id="templated-field-name",
        ),
        pytest.param(
            {
                "action": "cover.open_cover",
                "target": {"entity_id": "cover.garage_door"},
            },
            id="garage",
        ),
        pytest.param(
            {
                "action": "cover.open_cover",
                "data": {"{{ 'entity_id' }}": "cover.garage_door"},
            },
            id="garage-templated-field-name",
        ),
        pytest.param(
            {"action": "cover.open_cover", "target": {"label_id": "doors"}},
            id="garage-label",
        ),
        pytest.param(
            {
                "action": "script.turn_on",
                "target": {"entity_id": "script.unlock_door"},
            },
            id="script",
        ),
        pytest.param({"action": "script.unlock_door"}, id="script-by-key"),
        pytest.param(
            {
                "action": "script.turn_on",
                "target": {"entity_id": "{{ 'script.unlock_door' }}"},
            },
            id="templated-script",
        ),
        pytest.param(
            {
                "action": "scene.apply",
                "data": {"entities": {"lock.front_door": "unlocked"}},
            },
            id="scene",
        ),
        pytest.param(
            {
                "action": "automation.trigger",
                "target": {"entity_id": "automation.unlock_on_arrival"},
            },
            id="automation",
        ),
        pytest.param(
            {
                "action": "input_button.press",
                "target": {"entity_id": "input_button.unlock"},
            },
            id="button-starts-automation",
        ),
        pytest.param(
            {
                "action": "input_boolean.turn_on",
                "target": {"entity_id": "input_boolean.home"},
            },
            id="helper-starts-automation",
        ),
        # Home Assistant renders a field's name, and the rendered one replaces
        # the same name given plainly: each of these acted on the second
        # (#918 review). Any templated field name is held.
        pytest.param(
            {
                "action": "cover.open_cover",
                "data": {
                    "entity_id": "cover.shade",
                    "entity_id{# #}": "cover.garage_door",
                },
            },
            id="comment-field-name-garage",
        ),
        pytest.param(
            {
                "action": "script.turn_on",
                "data": {
                    "entity_id": "script.porch",
                    "entity_id{# #}": "script.unlock_door",
                },
            },
            id="comment-field-name-script",
        ),
        pytest.param(
            {
                "action": "homeassistant.toggle",
                "data": {
                    "entity_id": "light.porch",
                    "entity_id{##}": "cover.garage_door",
                },
            },
            id="comment-field-name-generic",
        ),
        pytest.param(
            {
                "action": "scene.apply",
                "data": {
                    "entities": {"light.porch": "on"},
                    "entities{##}": {"lock.front_door": "unlocked"},
                },
            },
            id="comment-field-name-scene",
        ),
        pytest.param(
            {
                "action": "scene.apply",
                "data": {
                    "entities": {"light.porch": "on", "lock.front_door{##}": "unlocked"}
                },
            },
            id="comment-field-name-nested",
        ),
        pytest.param(
            {
                "action": "input_boolean.turn_on",
                "data": {
                    "entity_id": "input_boolean.other",
                    "entity_id{##}": "input_boolean.home",
                },
            },
            id="comment-field-name-helper",
        ),
        pytest.param(
            {
                "action": "cover.open_cover",
                "target": {
                    "entity_id": "cover.shade",
                    "{% if true %}entity_id{% endif %}": "cover.garage_door",
                },
            },
            id="statement-field-name-target",
        ),
        pytest.param(
            {
                "action": "light.turn_on",
                "target": {"entity_id": "light.porch"},
                "data": {"brightness{# #}": 100},
            },
            id="comment-field-name-harmless",
        ),
        pytest.param({"action": "shell_command.open_garage"}, id="shell-command"),
        pytest.param({"action": "rest_command.open_garage"}, id="rest-command"),
        pytest.param({"action": "python_script.do_it"}, id="python-script"),
        pytest.param(
            {
                "action": "conversation.process",
                "data": {"text": "unlock the front door"},
            },
            id="another-agent",
        ),
        pytest.param(
            {"action": "{{ 'lock.unlock' }}", "target": FRONT_DOOR},
            id="templated-action",
        ),
        pytest.param({"target": FRONT_DOOR}, id="no-action"),
        pytest.param({"action": ["lock.unlock"], "target": FRONT_DOOR}, id="list"),
        pytest.param(
            {"action": "light.turn_on", "service": "lock.unlock", "target": FRONT_DOOR},
            id="second-name",
        ),
        pytest.param(
            {
                "action": "light.turn_on",
                "target": {"entity_id": "light.porch"},
                "entity_id": "lock.front_door",
                "service": "unlock",
            },
            id="bare-second-name",
        ),
    ],
)
async def test_a_call_that_could_do_a_critical_action_is_held(
    hass: HomeAssistant, admin_api: SimpleNamespace, args: dict[str, Any]
) -> None:
    """Each way the call-action tool could reach a guarded device is held."""
    assert _held(hass, admin_api, CALL_ACTION_TOOL, args)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            {"action": "light.turn_on", "target": {"entity_id": "light.porch"}},
            id="light",
        ),
        pytest.param(
            {"action": "light.turn_on", "target": {"area_id": "porch"}},
            id="light-by-area",
        ),
        pytest.param(
            {
                "action": "light.turn_on",
                "target": {"entity_id": "light.porch"},
                "data": {"brightness": "{{ 100 }}"},
            },
            id="templated-brightness",
        ),
        pytest.param({"action": "lock.lock", "target": FRONT_DOOR}, id="lock"),
        pytest.param(
            {
                "action": "cover.close_cover",
                "target": {"entity_id": "cover.garage_door"},
            },
            id="close-garage",
        ),
        pytest.param(
            {"action": "script.turn_on", "target": {"entity_id": "script.porch"}},
            id="harmless-script",
        ),
        pytest.param(
            {
                "action": "persistent_notification.create",
                "data": {"message": "The washer is done."},
            },
            id="notification",
        ),
    ],
)
async def test_an_ordinary_call_runs_without_a_pin(
    hass: HomeAssistant, admin_api: SimpleNamespace, args: dict[str, Any]
) -> None:
    """The tool's own name is not read as a service aimed at its targets."""
    assert not _held(hass, admin_api, CALL_ACTION_TOOL, args)


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        (
            "homeassistant__render_template",
            {"template": "{{ states('lock.front_door') }}"},
        ),
        ("homeassistant__search_actions", {"query": "lock.unlock"}),
        ("search__search", {"query": "front door lock"}),
        ("system_log__get_entries", {}),
        ("config__get_config_entries", {}),
    ],
)
async def test_a_read_only_tool_is_never_held(
    hass: HomeAssistant,
    admin_api: SimpleNamespace,
    tool_name: str,
    args: dict[str, Any],
) -> None:
    """Naming a lock or its unlock service in a query reads nothing as a call."""
    assert tool_origin(admin_api, tool_name).read_only
    assert not _held(hass, admin_api, tool_name, args)


async def test_the_tools_that_can_write_are_the_two_looked_at(
    admin_api: SimpleNamespace,
) -> None:
    """
    A writing tool added to the integrations set up here has to be looked at.

    Not every integration that adds tools to the API is set up (`history`
    needs the recorder, `hassio` a supervisor), and a custom one may add its
    own; a tool not read here is read as any other tool is.
    """
    writing = {
        tool.name
        for tool in admin_api.apis[ADMIN_API].tools
        if not tool_origin(admin_api, tool.name).read_only
    }

    assert writing == {CALL_ACTION_TOOL, "config__reload_config_entry"}


async def test_reloading_an_integration_needs_the_pin(
    hass: HomeAssistant, admin_api: SimpleNamespace
) -> None:
    """
    A reload names no service, and what it sets off cannot be read.

    Its entities go unavailable for a moment, which an automation may be
    waiting for.
    """
    assert _held(
        hass, admin_api, "config__reload_config_entry", {"entry_id": "hue:8K2M"}
    )


@pytest.mark.parametrize("tool_name", [CALL_ACTION_TOOL, "config__reload_config_entry"])
def test_the_writing_tools_count_as_actuation(tool_name: str) -> None:
    """A question about a state does not keep a tool that can change one."""
    assert is_actuation_tool(tool_name)


def test_a_reading_tool_is_not_actuation() -> None:
    """The admin-only API's other tools stay available to a question."""
    assert not is_actuation_tool("homeassistant__render_template")


async def test_the_same_name_on_another_api_is_still_read_by_its_name(
    hass: HomeAssistant, admin_api: SimpleNamespace
) -> None:
    """Only the admin-only API's own tool is read by its arguments alone."""
    args = {"action": "script.turn_on", "target": {"entity_id": "script.porch"}}

    assert _held(hass, admin_api, CALL_ACTION_TOOL, args) is False
    assert _is_critical_action(
        dict(args),
        expand_critical_rules(hass, RECOMMENDED_CRITICAL_ACTIONS),
        CALL_ACTION_TOOL,
        hass,
    )


@pytest.mark.usefixtures("admin_api")
async def test_the_pin_line_names_the_action(hass: HomeAssistant) -> None:
    """The user is told what the PIN allows, not the tool's name."""
    line = describe_pending_action(
        hass,
        {
            "tool_name": CALL_ACTION_TOOL,
            "tool_args": {"action": "lock.unlock", "target": FRONT_DOOR},
            "api_id": ADMIN_API,
        },
    )

    assert "unlock" in line.lower()
    assert "front door" in line.lower()


async def test_the_pin_line_names_a_tool_that_names_no_action(
    hass: HomeAssistant,
) -> None:
    """A held reload reads as the tool and its arguments, not as a device."""
    line = describe_pending_action(
        hass,
        {
            "tool_name": "config__reload_config_entry",
            "tool_args": {"entry_id": "hue:8K2M"},
            "api_id": ADMIN_API,
            "api_name": "Home Assistant",
            "api_tool_name": "config__reload_config_entry",
            "admin_api": True,
        },
    )

    assert "reload_config_entry" in line
    assert "hue:8K2M" in line
    assert "Home Assistant" in line


def _turn(hass: HomeAssistant, api: SimpleNamespace) -> Any:
    """Return the context a tool call runs in, with a PIN set."""

    async def _call_tool(tool_input: llm.ToolInput) -> Any:
        # The tool itself: APIInstance.async_call_tool imports the voice
        # stack (hassil), which the test venv lacks.
        instance = api.apis[ADMIN_API]
        tool = next(t for t in instance.tools if t.name == tool_input.tool_name)
        return await tool.async_call(hass, tool_input, instance.llm_context)

    return cast(
        "Any",
        SimpleNamespace(
            hass=hass,
            config={"configurable": {"thread_id": "t1"}},
            ha_llm_api=SimpleNamespace(
                apis=api.apis,
                routing_map={CALL_ACTION_TOOL: ADMIN_API},
                remote_home_assistants=frozenset(),
                async_call_tool=_call_tool,
            ),
            critical_actions=RECOMMENDED_CRITICAL_ACTIONS,
            pin_enabled=True,
            pin_hash="hash",
            pin_salt="salt",
            pending_actions={},
            state={"messages": []},
        ),
    )


async def test_an_ordinary_call_reaches_home_assistant_as_given(
    hass: HomeAssistant, admin_api: SimpleNamespace
) -> None:
    """The call's target and data are sent on unchanged, and the action runs."""
    assert await async_setup_component(hass, "input_text", {"input_text": {"note": {}}})
    await hass.async_block_till_done()

    result = await _run_ha_tool(
        CALL_ACTION_TOOL,
        {
            "action": "input_text.set_value",
            "target": {"entity_id": "input_text.note"},
            "data": {"value": "{{ 'washer ' ~ 'done' }}"},
        },
        _turn(hass, admin_api),
        {"id": "call1"},
    )

    assert result.status == "success", result.content
    state = hass.states.get("input_text.note")
    assert state is not None
    assert state.state == "washer done"


async def test_a_guarded_call_is_held_and_does_not_run(
    hass: HomeAssistant, admin_api: SimpleNamespace
) -> None:
    """The lock stays locked, and the reply asks for the PIN."""
    unlocked: list[Any] = []
    hass.services.async_register("lock", "unlock", unlocked.append)
    turn = _turn(hass, admin_api)

    result = await _run_ha_tool(
        CALL_ACTION_TOOL,
        {"action": "lock.unlock", "target": FRONT_DOOR},
        turn,
        {"id": "call1"},
    )

    assert "requires_pin" in str(result.content)
    assert unlocked == []
    (held,) = turn.pending_actions.values()
    assert held["api_id"] == ADMIN_API
    assert held["tool_args"] == {"action": "lock.unlock", "target": FRONT_DOOR}


async def test_a_guarded_call_behind_a_comment_field_name_does_not_run(
    hass: HomeAssistant, admin_api: SimpleNamespace
) -> None:
    """The decoy target is not what Home Assistant would act on (#918 review)."""
    opened: list[Any] = []
    hass.services.async_register("cover", "open_cover", opened.append)
    turn = _turn(hass, admin_api)

    result = await _run_ha_tool(
        CALL_ACTION_TOOL,
        {
            "action": "cover.open_cover",
            "data": {"entity_id": "cover.shade", "entity_id{# #}": "cover.garage_door"},
        },
        turn,
        {"id": "call1"},
    )

    assert "requires_pin" in str(result.content)
    assert opened == []
    assert len(turn.pending_actions) == 1


@pytest.mark.usefixtures("admin_api")
async def test_an_automation_step_with_a_comment_field_name_is_held(
    hass: HomeAssistant,
) -> None:
    """The same step in an automation the agent writes is held too."""
    step = cv.script_action(
        {
            "action": "cover.open_cover",
            "data": {"entity_id": "cover.shade", "entity_id{# #}": "cover.garage_door"},
        }
    )

    assert find_critical_automation_calls(
        {"actions": [step]}, expand_critical_rules(hass, RECOMMENDED_CRITICAL_ACTIONS)
    )


def test_retrieval_skips_an_api_the_turn_did_not_load() -> None:
    """A satellite's turn does not spend retrieval places on the admin API's tools."""
    config: Any = {
        "configurable": {
            "options": {"llm_hass_api": ["assist", ADMIN_API]},
            "ha_llm_api": SimpleNamespace(apis={"assist": object()}),
        }
    }

    assert _get_allowed_api_ids(config) == {"assist", "hga_local"}
    config["configurable"]["ha_llm_api"] = SimpleNamespace(
        apis={"assist": object(), ADMIN_API: object()}
    )
    assert _get_allowed_api_ids(config) == {"assist", ADMIN_API, "hga_local"}
    # With no API to read (a test, a robot run), the configured ones as before.
    del config["configurable"]["ha_llm_api"]
    assert _get_allowed_api_ids(config) == {"assist", ADMIN_API, "hga_local"}


@pytest.mark.parametrize("text", ["{{ x }}", "{% if x %}a{% endif %}", "a{# #}b"])
def test_a_comment_is_a_template_wherever_templates_are_looked_for(text: str) -> None:
    """Home Assistant renders all three forms (`is_template_string`)."""
    assert _is_template(text)
    assert _templated([text])
