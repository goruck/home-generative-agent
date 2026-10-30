# ruff: noqa: S101
"""Tests for the critical-action PIN gate on `add_automation`."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import yaml
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.agent import tools as tools_module
from custom_components.home_generative_agent.agent.critical_push import (
    CriticalPushResult,
)
from custom_components.home_generative_agent.agent.helpers import (
    MAX_PENDING_ACTIONS,
    PIN_AWAITING_USER_REPLY,
    PIN_OUTCOME_DONE,
    PIN_OUTCOME_UNKNOWN,
    PinReply,
    disclose_pin_prompts,
    start_pin_turn,
)
from custom_components.home_generative_agent.agent.tools import (
    add_automation,
    confirm_sensitive_action,
)
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
)
from custom_components.home_generative_agent.core.utils import hash_pin

if TYPE_CHECKING:
    from pathlib import Path

AddAutomationTool = Callable[..., Awaitable[str]]
add_automation_tool = cast("AddAutomationTool", cast("Any", add_automation).coroutine)
_confirm = cast("AddAutomationTool", cast("Any", confirm_sensitive_action).coroutine)


async def confirm_tool(
    action_id: str, pin: str, *, config: dict[str, Any], store: Any
) -> str:
    """Confirm as the user's reply to the PIN prompt: a new message with the PIN."""
    configurable = config["configurable"]
    # The request turn ended by naming the prompt, as conversation.py does.
    disclose_pin_prompts(configurable["pending_actions"], configurable["pin_reply"])
    configurable["pin_reply"] = start_pin_turn(
        configurable["pending_actions"], configurable["thread_id"], pin
    )
    return await _confirm(action_id, pin, config=config, store=store)


PIN = "4321"

UNLOCK_YAML = """
alias: Unlock when I get home
triggers:
  - trigger: state
    entity_id: person.me
    to: home
actions:
  - action: lock.unlock
    target:
      entity_id: lock.front_door
"""

UNLOCK_YAML_LIST = """
- alias: Unlock when I get home
  triggers:
    - trigger: state
      entity_id: person.me
      to: home
  actions:
    - action: lock.unlock
      target:
        entity_id: lock.front_door
"""

LIGHT_YAML = """
alias: Light when I get home
triggers:
  - trigger: state
    entity_id: person.me
    to: home
actions:
  - action: light.turn_on
    target:
      entity_id: light.hall
"""


class FakeServices:
    """Record service calls made by the tool."""

    def __init__(self) -> None:
        """Initialize the recorder."""
        self.calls: list[tuple[str, str]] = []

    async def async_call(self, domain: str, service: str) -> None:
        """Record a service call."""
        self.calls.append((domain, service))


class FakeBus:
    """Record bus events fired by the tool."""

    def __init__(self) -> None:
        """Initialize the recorder."""
        self.events: list[tuple[str, dict[str, Any]]] = []

    def async_fire(self, event_type: str, data: dict[str, Any]) -> None:
        """Record a fired event."""
        self.events.append((event_type, data))


def _fake_hass(tmp_path: Path) -> Any:
    """Build a minimal hass stand-in with an empty automations.yaml."""
    (tmp_path / "automations.yaml").write_text("", encoding="utf-8")

    async def _executor_job(target: Callable[..., Any], *args: Any) -> Any:
        return target(*args)

    return SimpleNamespace(
        config=SimpleNamespace(config_dir=str(tmp_path)),
        services=FakeServices(),
        bus=FakeBus(),
        async_add_executor_job=_executor_job,
    )


def _config(hass: Any, options: dict[str, Any]) -> dict[str, Any]:
    """Build the configurable payload passed to the tools."""
    return {
        "configurable": {
            "hass": hass,
            "options": options,
            "pending_actions": {},
            "user_id": "user1",
            "thread_id": "conv1",
            "pin_reply": PinReply("add the automation"),
            "ha_llm_api": None,
        }
    }


def _pin_options() -> dict[str, Any]:
    """Return options with the critical-action PIN configured."""
    hashed, salt = hash_pin(PIN)
    return {
        CONF_CRITICAL_ACTION_PIN_ENABLED: True,
        CONF_CRITICAL_ACTION_PIN_HASH: hashed,
        CONF_CRITICAL_ACTION_PIN_SALT: salt,
    }


@pytest.fixture(autouse=True)
def _stub_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip HA's trigger/blueprint validation, which needs a real hass."""

    async def _validate(**kwargs: Any) -> dict[str, Any]:
        return dict(kwargs["config"])

    monkeypatch.setattr(tools_module, "_async_validate_config_item", _validate)
    # The existence check needs a real state machine and entity registry;
    # it has its own tests in test_automation_targets.py.
    monkeypatch.setattr(
        tools_module, "find_missing_automation_targets", lambda *_a, **_k: []
    )
    # So does the critical push rewrite (#869), tested in test_critical_push.py.
    monkeypatch.setattr(
        tools_module, "apply_critical_push", lambda *_a, **_k: CriticalPushResult()
    )


def _written(tmp_path: Path) -> list[dict[str, Any]]:
    """Return the automations currently persisted to automations.yaml."""
    text = (tmp_path / "automations.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text) or []


@pytest.mark.asyncio
async def test_critical_automation_requires_pin(tmp_path: Path) -> None:
    """An automation that unlocks a door is held for PIN confirmation."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    result = await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)

    payload = json.loads(result)
    assert payload["status"] == "requires_pin"
    assert "lock.unlock" in payload["reason"]
    assert len(config["configurable"]["pending_actions"]) == 1
    # Nothing was written or reloaded.
    assert _written(tmp_path) == []
    assert hass.services.calls == []
    assert hass.bus.events == []


@pytest.mark.asyncio
async def test_benign_automation_is_written_without_pin(tmp_path: Path) -> None:
    """A non-critical automation installs immediately."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    result = await add_automation_tool(automation_yaml=LIGHT_YAML, config=config)

    assert result.startswith("Added automation ")
    assert config["configurable"]["pending_actions"] == {}
    assert len(_written(tmp_path)) == 1
    assert hass.services.calls == [("automation", "reload")]


@pytest.mark.asyncio
async def test_gate_is_off_when_pin_not_enabled(tmp_path: Path) -> None:
    """With no PIN configured the pre-existing behavior is unchanged."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, {})

    result = await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)

    assert result.startswith("Added automation ")
    assert len(_written(tmp_path)) == 1


@pytest.mark.asyncio
async def test_enabled_without_configured_pin_refuses_critical_automation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """
    The toggle without a stored PIN refuses the install.

    The direct-tool guard lets a call through with a warning in this state, but
    an automation is durable — it keeps firing — so this path fails closed
    instead. The user is told how to fix it.
    """
    hass = _fake_hass(tmp_path)
    config = _config(hass, {CONF_CRITICAL_ACTION_PIN_ENABLED: True})

    result = await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)

    assert "cannot be confirmed" in result
    assert "Set a PIN" in result
    assert _written(tmp_path) == []
    assert hass.services.calls == []
    assert "Refusing to install" in caplog.text


@pytest.mark.asyncio
async def test_enabled_without_configured_pin_still_allows_benign_automation(
    tmp_path: Path,
) -> None:
    """Failing closed applies only to automations that screen as critical."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, {CONF_CRITICAL_ACTION_PIN_ENABLED: True})

    result = await add_automation_tool(automation_yaml=LIGHT_YAML, config=config)

    assert result.startswith("Added automation ")
    assert len(_written(tmp_path)) == 1


@pytest.mark.asyncio
async def test_confirming_pin_writes_the_held_automation(tmp_path: Path) -> None:
    """The correct PIN installs exactly the automation that was screened."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    challenge = json.loads(
        await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)
    )
    result = await confirm_tool(challenge["action_id"], PIN, config=config, store=None)

    payload = json.loads(result)
    assert payload["status"] == "completed"
    written = _written(tmp_path)
    assert len(written) == 1
    assert written[0]["actions"][0]["action"] == "lock.unlock"
    assert config["configurable"]["pending_actions"] == {}
    assert hass.services.calls == [("automation", "reload")]


@pytest.mark.asyncio
async def test_a_pin_the_user_did_not_write_cannot_install_the_automation(
    tmp_path: Path,
) -> None:
    """
    A confirmation whose PIN is not in the user's message is refused.

    The hold records the conversation, and nothing is written or counted.
    """
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    challenge = json.loads(
        await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)
    )
    action_id = challenge["action_id"]
    pending = config["configurable"]["pending_actions"][action_id]
    reply = json.loads(await _confirm(action_id, PIN, config=config, store=None))

    assert reply["status"] == PIN_AWAITING_USER_REPLY
    assert pending["thread_id"] == "conv1"
    assert pending["attempts"] == 0
    assert not (tmp_path / "automations.yaml").exists() or _written(tmp_path) == []


@pytest.mark.asyncio
async def test_wrong_pin_does_not_write_the_automation(tmp_path: Path) -> None:
    """An incorrect PIN leaves the automation pending and uninstalled."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    challenge = json.loads(
        await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)
    )
    result = await confirm_tool(
        challenge["action_id"], "9999", config=config, store=None
    )

    assert result.startswith("Incorrect PIN. Action not executed.")
    assert _written(tmp_path) == []
    assert len(config["configurable"]["pending_actions"]) == 1


@pytest.mark.asyncio
async def test_yaml_list_form_is_screened(tmp_path: Path) -> None:
    """A single-item YAML list — a shape models emit — is screened too."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())
    assert isinstance(yaml.safe_load(UNLOCK_YAML_LIST), list)

    result = await add_automation_tool(automation_yaml=UNLOCK_YAML_LIST, config=config)

    assert json.loads(result)["status"] == "requires_pin"
    assert _written(tmp_path) == []


@pytest.mark.asyncio
async def test_blueprint_automation_is_screened_from_validated_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Screening reads the blueprint-substituted config, not the raw stanza."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    async def _substituting_validate(**kwargs: Any) -> dict[str, Any]:
        # What HA returns once the blueprint is expanded.
        return {
            **kwargs["config"],
            "actions": [
                {"action": "lock.unlock", "target": {"entity_id": ["lock.front_door"]}}
            ],
        }

    monkeypatch.setattr(
        tools_module, "_async_validate_config_item", _substituting_validate
    )

    blueprint_yaml = (
        "alias: Blueprint automation\n"
        "use_blueprint:\n"
        "  path: some/blueprint.yaml\n"
        "  input:\n"
        "    which: front\n"
    )
    challenge = json.loads(
        await add_automation_tool(automation_yaml=blueprint_yaml, config=config)
    )
    assert challenge["status"] == "requires_pin"
    assert "lock.unlock" in challenge["reason"]

    await confirm_tool(challenge["action_id"], PIN, config=config, store=None)

    written = _written(tmp_path)
    # The stanza persisted is the blueprint reference, not the expansion.
    assert written[0]["use_blueprint"]["path"] == "some/blueprint.yaml"
    assert "actions" not in written[0]


@pytest.mark.asyncio
async def test_missing_pending_action_store_blocks_the_write(tmp_path: Path) -> None:
    """With nowhere to park the challenge the automation is refused, not written."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())
    del config["configurable"]["pending_actions"]

    result = await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)

    assert result == "Unable to confirm this automation right now; please try again."
    assert _written(tmp_path) == []
    assert hass.services.calls == []


@pytest.mark.asyncio
async def test_invalid_configuration_is_rejected_before_screening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config HA rejects never reaches the gate or the write path."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    async def _reject(**_kwargs: Any) -> dict[str, Any]:
        msg = "bad trigger"
        raise HomeAssistantError(msg)

    monkeypatch.setattr(tools_module, "_async_validate_config_item", _reject)

    result = await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)

    assert result.startswith("Invalid automation configuration")
    assert _written(tmp_path) == []
    assert config["configurable"]["pending_actions"] == {}


@pytest.mark.asyncio
async def test_missing_configurable_is_rejected() -> None:
    """Without a configurable payload nothing is screened and nothing is written."""
    result = await add_automation_tool(automation_yaml=UNLOCK_YAML, config={})

    assert result == "Configuration not found. Please check your setup."


@pytest.mark.asyncio
async def test_expired_pending_automation_is_not_written(tmp_path: Path) -> None:
    """A challenge older than the expiry window cannot install the automation."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    challenge = json.loads(
        await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)
    )
    pending = config["configurable"]["pending_actions"][challenge["action_id"]]
    pending["created_at"] = (dt_util.utcnow() - timedelta(minutes=30)).isoformat()

    result = await confirm_tool(challenge["action_id"], PIN, config=config, store=None)

    assert "expired" in result
    assert _written(tmp_path) == []
    assert config["configurable"]["pending_actions"] == {}


@pytest.mark.asyncio
async def test_a_wrong_pin_keeps_the_automation_prompt_for_one_more_reply(
    tmp_path: Path,
) -> None:
    """
    A typo is answered in the next message; guessing stays bounded.

    One message is one attempt, and a prompt is kept open for at most three
    replies, so the PIN cannot be guessed against one held automation.
    """
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    challenge = json.loads(
        await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)
    )
    wrong = await confirm_tool(
        challenge["action_id"], "9999", config=config, store=None
    )
    result = await confirm_tool(challenge["action_id"], PIN, config=config, store=None)

    assert wrong.startswith("Incorrect PIN")
    assert json.loads(result)["status"] == "completed"
    assert len(_written(tmp_path)) == 1


@pytest.mark.asyncio
async def test_too_many_wrong_pins_locks_out_the_automation(tmp_path: Path) -> None:
    """
    Attempt exhaustion blocks the write even if the right PIN arrives.

    Failed attempts carry over to a request that replaces the held one in its
    reply turn, so the limit is reachable; set it directly here.
    """
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    challenge = json.loads(
        await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)
    )
    config["configurable"]["pending_actions"][challenge["action_id"]]["attempts"] = 5
    result = await confirm_tool(challenge["action_id"], PIN, config=config, store=None)

    assert "Too many incorrect attempts" in result
    assert _written(tmp_path) == []


@pytest.mark.asyncio
async def test_confirming_without_a_prior_challenge_writes_nothing(
    tmp_path: Path,
) -> None:
    """A speculative confirmation with no pending action is a no-op."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    result = await confirm_tool("01JUNK", PIN, config=config, store=None)

    assert result == "Pending action not found or expired."
    assert _written(tmp_path) == []


@pytest.mark.asyncio
async def test_pending_automation_without_a_config_is_rejected(tmp_path: Path) -> None:
    """A malformed pending automation record cannot install anything."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())
    config["configurable"]["pending_actions"]["abc"] = {
        "tool_name": "add_automation",
        "tool_args": {},
        "created_at": dt_util.utcnow().isoformat(),
        "user": "user1",
        "thread_id": "conv1",
        "turn_id": config["configurable"]["pin_reply"].turn_id,
        "attempts": 0,
    }

    result = await confirm_tool("abc", PIN, config=config, store=None)

    assert result == "Pending automation is invalid; please re-run the request."
    assert _written(tmp_path) == []


@pytest.mark.asyncio
async def test_write_failure_is_reported_and_consumes_the_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A failed write says the automation may exist, and the confirmation is spent.

    The pending action is claimed *before* the write so two confirmations of
    the same action_id in one concurrent tool batch cannot both install it.
    Any step after the claim can fail once the file was appended (the reload
    does), so the reply never asks for the automation again.
    """
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    challenge = json.loads(
        await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)
    )

    async def _boom(*_args: Any, **_kwargs: Any) -> str:
        msg = "disk full at /config/automations.yaml"
        raise OSError(msg)

    monkeypatch.setattr(tools_module, "_async_write_automation", _boom)

    result = await confirm_tool(challenge["action_id"], PIN, config=config, store=None)

    assert "may have been installed" in result
    assert "/config" not in result
    assert _written(tmp_path) == []
    assert config["configurable"]["pending_actions"] == {}
    # Said in fixed words too: "PIN accepted; add the automation … may have run."
    pin_reply = config["configurable"]["pin_reply"]
    assert pin_reply.outcome == PIN_OUTCOME_UNKNOWN
    assert pin_reply.confirmed.startswith('add the automation "Unlock when I get home"')
    assert "lock.unlock" in pin_reply.confirmed


@pytest.mark.asyncio
async def test_an_unexpected_install_error_says_the_automation_may_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error after the claim may follow a partial write: say so, do not raise."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    challenge = json.loads(
        await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)
    )

    async def _boom(*_args: Any, **_kwargs: Any) -> str:
        msg = "integration bug"
        raise RuntimeError(msg)

    monkeypatch.setattr(tools_module, "_async_write_automation", _boom)

    result = await confirm_tool(challenge["action_id"], PIN, config=config, store=None)

    assert "may have been installed" in result
    assert "integration bug" not in result
    assert config["configurable"]["pending_actions"] == {}
    assert config["configurable"]["pin_reply"].outcome == PIN_OUTCOME_UNKNOWN


@pytest.mark.asyncio
async def test_a_confirmed_automation_cannot_be_installed_twice(
    tmp_path: Path,
) -> None:
    """Replaying the same action_id must not append the automation again."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    challenge = json.loads(
        await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)
    )
    first = await confirm_tool(challenge["action_id"], PIN, config=config, store=None)
    first_reply = config["configurable"]["pin_reply"]
    second = await confirm_tool(challenge["action_id"], PIN, config=config, store=None)

    assert json.loads(first)["status"] == "completed"
    assert first_reply.outcome == PIN_OUTCOME_DONE
    assert second == "Pending action not found or expired."
    assert len(_written(tmp_path)) == 1


@pytest.mark.asyncio
async def test_confirmation_from_another_user_is_rejected(tmp_path: Path) -> None:
    """A held automation can only be confirmed by the user who requested it."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    challenge = json.loads(
        await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)
    )
    config["configurable"]["user_id"] = "someone_else"
    result = await confirm_tool(challenge["action_id"], PIN, config=config, store=None)

    assert "different user" in result
    assert _written(tmp_path) == []


@pytest.mark.asyncio
async def test_blueprint_tool_arguments_are_screened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The `time_pattern`/`message` blueprint path is gated like raw YAML.

    This path builds a config containing only `use_blueprint:` and no actions
    of its own, so the gate depends entirely on screening the *substituted*
    config Home Assistant returns from validation.
    """
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    async def _substituting_validate(**kwargs: Any) -> dict[str, Any]:
        assert "use_blueprint" in kwargs["config"]
        return {
            **kwargs["config"],
            "actions": [
                {"action": "lock.unlock", "target": {"entity_id": ["lock.front_door"]}}
            ],
        }

    monkeypatch.setattr(
        tools_module, "_async_validate_config_item", _substituting_validate
    )

    result = await add_automation_tool(
        time_pattern="/30", message="check the front porch", config=config
    )

    assert json.loads(result)["status"] == "requires_pin"
    assert _written(tmp_path) == []
    assert hass.services.calls == []


@pytest.mark.asyncio
async def test_benign_blueprint_tool_arguments_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordinary camera-analysis blueprint still installs without a PIN."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    async def _substituting_validate(**kwargs: Any) -> dict[str, Any]:
        return {
            **kwargs["config"],
            "actions": [
                {"action": "notify.mobile_app_phone", "data": {"message": "x"}}
            ],
        }

    monkeypatch.setattr(
        tools_module, "_async_validate_config_item", _substituting_validate
    )

    result = await add_automation_tool(
        time_pattern="/30", message="check the front porch", config=config
    )

    assert result.startswith("Added automation ")
    assert len(_written(tmp_path)) == 1


@pytest.mark.asyncio
async def test_abandoned_challenges_do_not_grow_the_pending_store(
    tmp_path: Path,
) -> None:
    """A challenge loop cannot grow the in-memory store without bound."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    for _ in range(40):
        await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)

    assert len(config["configurable"]["pending_actions"]) <= MAX_PENDING_ACTIONS
    assert _written(tmp_path) == []


@pytest.mark.asyncio
async def test_expired_challenges_are_swept_when_a_new_one_registers(
    tmp_path: Path,
) -> None:
    """Registering a challenge drops entries past the confirmation window."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())
    pending = config["configurable"]["pending_actions"]
    pending["stale"] = {
        "tool_name": "add_automation",
        "created_at": (dt_util.utcnow() - timedelta(minutes=11)).isoformat(),
        "attempts": 0,
    }

    await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)

    assert "stale" not in pending
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_malformed_pending_record_does_not_break_registration(
    tmp_path: Path,
) -> None:
    """
    A record with an unusable timestamp is dropped, not stepped over.

    A timezone-naive timestamp cannot be compared against an aware `now`, and
    a record with no timestamp can never expire. Either one left in place
    would be immortal and would push live confirmations out of the cap.
    """
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())
    pending = config["configurable"]["pending_actions"]
    pending["naive"] = {
        "tool_name": "add_automation",
        "created_at": "2026-08-05T10:00:00",
    }
    pending["missing"] = {"tool_name": "add_automation"}

    result = await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)

    assert json.loads(result)["status"] == "requires_pin"
    assert "naive" not in pending
    assert "missing" not in pending
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_a_slow_install_after_the_claim_says_it_may_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A timeout after the claim gets the "may have been installed" reply.

    TimeoutError is an OSError, so it was reported as a definite failure
    that told the user to request the automation again (pass 3).
    """
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    challenge = json.loads(
        await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)
    )

    async def _slow(*_args: Any, **_kwargs: Any) -> str:
        await asyncio.sleep(5)
        return "Added automation"

    monkeypatch.setattr(tools_module, "_async_write_automation", _slow)
    monkeypatch.setattr(tools_module, "POST_CLAIM_TIMEOUT_S", 0.01)

    result = await confirm_tool(challenge["action_id"], PIN, config=config, store=None)

    assert "may have been installed" in result
    assert "request the automation again" not in result
    assert config["configurable"]["pending_actions"] == {}


@pytest.mark.asyncio
async def test_a_second_automation_in_one_turn_waits_for_the_first(
    tmp_path: Path,
) -> None:
    """Only one protected action waits for the PIN, so the PIN runs a known one."""
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())

    await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)
    second = await add_automation_tool(
        automation_yaml=UNLOCK_YAML.replace("front_door", "back_door"), config=config
    )

    assert "another protected action is already waiting" in second
    assert len(config["configurable"]["pending_actions"]) == 1


@pytest.mark.asyncio
async def test_reissuing_the_automation_in_the_pin_reply_still_installs(
    tmp_path: Path,
) -> None:
    """
    The re-issued YAML gets a new generated id but is the same request.

    Pass 3: comparing the whole config, generated id included, replaced the
    challenge with one the user's reply could not confirm.
    """
    hass = _fake_hass(tmp_path)
    config = _config(hass, _pin_options())
    configurable = config["configurable"]

    await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)
    disclose_pin_prompts(configurable["pending_actions"], configurable["pin_reply"])
    configurable["pin_reply"] = start_pin_turn(
        configurable["pending_actions"], configurable["thread_id"], PIN
    )
    reissued = json.loads(
        await add_automation_tool(automation_yaml=UNLOCK_YAML, config=config)
    )
    result = await _confirm(reissued["action_id"], PIN, config=config, store=None)

    assert json.loads(result)["status"] == "completed"
    assert len(_written(tmp_path)) == 1
