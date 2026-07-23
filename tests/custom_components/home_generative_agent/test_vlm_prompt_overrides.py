# ruff: noqa: S101
"""Tests for user-configurable VLM prompt overrides (issue #497)."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.helpers.selector import TextSelector

from custom_components.home_generative_agent.agent.tools import (
    VLMPromptOverrides,
    analyze_image,
    get_and_analyze_camera_image,
)
from custom_components.home_generative_agent.config_flow import (
    HomeGenerativeAgentOptionsFlow,
    _schema_for_options,
)
from custom_components.home_generative_agent.const import (
    CONF_PROMPT,
    CONF_VLM_PROMPT_EXTRA,
    CONF_VLM_RESPONSE_LANGUAGE,
    VLM_SYSTEM_PROMPT,
    VLM_USER_PROMPT,
)
from custom_components.home_generative_agent.core.video_helpers import (
    is_no_change_reply,
)

_IMAGE_BYTES = b"fake-jpeg-bytes"


def _capture_model() -> MagicMock:
    """Return a stub VLM model that records the messages it is invoked with."""
    model = MagicMock()
    reply = MagicMock()
    reply.content = "A dog sits by the gate."
    model.ainvoke = AsyncMock(return_value=reply)
    return model


def _system_prompt(model: MagicMock) -> str:
    """Return the system message content passed to the stub model."""
    messages = model.ainvoke.await_args.args[0]
    return cast("str", messages[0].content)


def _user_text(model: MagicMock) -> str:
    """Return the user-turn request text passed to the stub model."""
    messages = model.ainvoke.await_args.args[0]
    return cast("str", messages[1].content[0]["text"])


# ---------------------------
# analyze_image prompt assembly
# ---------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        None,
        VLMPromptOverrides(),
        VLMPromptOverrides(response_language="", prompt_extra=""),
    ],
)
async def test_no_overrides_leaves_system_prompt_unchanged(
    overrides: VLMPromptOverrides | None,
) -> None:
    """Without overrides both prompt sides are exactly the base constants."""
    model = _capture_model()
    await analyze_image(model, _IMAGE_BYTES, None, overrides=overrides)
    assert _system_prompt(model) == VLM_SYSTEM_PROMPT
    assert _user_text(model) == VLM_USER_PROMPT


@pytest.mark.asyncio
async def test_language_override_appends_with_sentinel_carve_out() -> None:
    """A language override appends after the base prompt and keeps the sentinel English."""
    model = _capture_model()
    overrides = VLMPromptOverrides(response_language="Czech")
    await analyze_image(model, _IMAGE_BYTES, None, overrides=overrides)

    system = _system_prompt(model)
    assert system.startswith(VLM_SYSTEM_PROMPT)
    appended = system[len(VLM_SYSTEM_PROMPT) :]
    assert "Respond in Czech" in appended
    assert '"Scene unchanged."' in appended
    assert "must stay in English" in appended


@pytest.mark.asyncio
async def test_prompt_extra_appends_to_both_prompt_sides() -> None:
    """
    Extra instructions are appended on both sides, never replacing the base.

    The user-turn restatement exists because chat-tuned VLMs weight the
    user turn's description request over system-prompt-only instructions.
    """
    model = _capture_model()
    overrides = VLMPromptOverrides(prompt_extra="Ignore cars in the driveway.")
    await analyze_image(model, _IMAGE_BYTES, None, overrides=overrides)

    system = _system_prompt(model)
    assert system.startswith(VLM_SYSTEM_PROMPT)
    assert "Ignore cars in the driveway." in system
    assert "except " in system  # precedence framing keeps the sentinel contract
    assert '"Scene unchanged." sentinel contract' in system

    user = _user_text(model)
    assert user.startswith(VLM_USER_PROMPT)
    assert user.endswith(
        "this instruction takes precedence: Ignore cars in the driveway."
    )


@pytest.mark.asyncio
async def test_prompt_extra_keyword_path_keeps_user_turn_untouched() -> None:
    """
    Detection keywords come from the live chat request and outrank config.

    The user-turn restatement must not fire on the focused-keyword path;
    the system-prompt append still applies there.
    """
    model = _capture_model()
    overrides = VLMPromptOverrides(prompt_extra="Ignore cars in the driveway.")
    await analyze_image(model, _IMAGE_BYTES, ["boxes", "dogs"], overrides=overrides)

    user = _user_text(model)
    assert "boxes or dogs" in user
    assert "takes precedence" not in user
    assert "Ignore cars in the driveway." not in user

    system = _system_prompt(model)
    assert system.startswith(VLM_SYSTEM_PROMPT)
    assert "Ignore cars in the driveway." in system


@pytest.mark.asyncio
async def test_language_override_precedes_prompt_extra() -> None:
    """With both overrides set, the language instruction comes before the extra."""
    model = _capture_model()
    overrides = VLMPromptOverrides(
        response_language="Czech", prompt_extra="Ignore cars in the driveway."
    )
    await analyze_image(model, _IMAGE_BYTES, None, overrides=overrides)

    system = _system_prompt(model)
    assert system.startswith(VLM_SYSTEM_PROMPT)
    assert system.index("Respond in Czech") < system.index(
        "Ignore cars in the driveway."
    )


def test_sentinel_reply_still_detected() -> None:
    """The exact sentinel the carve-out protects still trips detection (#493)."""
    assert is_no_change_reply("Scene unchanged.")


# ---------------------------
# Camera tool plumbing
# ---------------------------


@pytest.mark.asyncio
async def test_camera_tool_threads_options_into_overrides() -> None:
    """The chat tool reads overrides from config['configurable']['options']."""
    model = _capture_model()
    config = {
        "configurable": {
            "hass": MagicMock(),
            "vlm_model": model,
            "options": {
                CONF_VLM_RESPONSE_LANGUAGE: "Czech",
                CONF_VLM_PROMPT_EXTRA: "Ignore cars in the driveway.",
            },
        }
    }

    with patch(
        "custom_components.home_generative_agent.agent.tools._get_camera_image",
        new=AsyncMock(return_value=_IMAGE_BYTES),
    ):
        result = await get_and_analyze_camera_image.coroutine(  # type: ignore[misc]
            camera_name="camera.test", detection_keywords=None, config=config
        )

    assert result == "A dog sits by the gate."
    system = _system_prompt(model)
    assert system.startswith(VLM_SYSTEM_PROMPT)
    assert "Respond in Czech" in system
    assert "Ignore cars in the driveway." in system


@pytest.mark.asyncio
async def test_camera_tool_without_options_keeps_base_prompt() -> None:
    """Missing options in the tool config leaves the system prompt untouched."""
    model = _capture_model()
    config = {"configurable": {"hass": MagicMock(), "vlm_model": model}}

    with patch(
        "custom_components.home_generative_agent.agent.tools._get_camera_image",
        new=AsyncMock(return_value=_IMAGE_BYTES),
    ):
        await get_and_analyze_camera_image.coroutine(  # type: ignore[misc]
            camera_name="camera.test", detection_keywords=None, config=config
        )

    assert _system_prompt(model) == VLM_SYSTEM_PROMPT


# ---------------------------
# Options flow
# ---------------------------


def _schema_keys(schema: dict[Any, Any]) -> list[str]:
    return [str(cast("Any", key).schema) for key in schema]


def _schema_key(schema: dict[Any, Any], key_name: str) -> Any:
    return next(key for key in schema if cast("Any", key).schema == key_name)


@pytest.mark.asyncio
async def test_options_schema_vlm_overrides_render_after_prompt(hass: Any) -> None:
    """The VLM override fields render directly below the instructions prompt."""
    schema = await _schema_for_options(hass, {})
    keys = _schema_keys(schema)

    prompt_idx = keys.index(CONF_PROMPT)
    assert keys[prompt_idx + 1] == CONF_VLM_RESPONSE_LANGUAGE
    assert keys[prompt_idx + 2] == CONF_VLM_PROMPT_EXTRA

    language_selector = schema[_schema_key(schema, CONF_VLM_RESPONSE_LANGUAGE)]
    extra_selector = schema[_schema_key(schema, CONF_VLM_PROMPT_EXTRA)]
    assert isinstance(language_selector, TextSelector)
    assert isinstance(extra_selector, TextSelector)
    assert not language_selector.config.get("multiline")
    assert extra_selector.config.get("multiline") is True


def test_drop_empty_fields_removes_blank_vlm_overrides() -> None:
    """Empty or whitespace-only override values are not stored."""
    flow = HomeGenerativeAgentOptionsFlow.__new__(HomeGenerativeAgentOptionsFlow)
    options: dict[str, Any] = {
        CONF_VLM_RESPONSE_LANGUAGE: "  ",
        CONF_VLM_PROMPT_EXTRA: "",
        CONF_PROMPT: "keep me",
    }
    flow._drop_empty_fields(options)
    assert CONF_VLM_RESPONSE_LANGUAGE not in options
    assert CONF_VLM_PROMPT_EXTRA not in options
    assert options[CONF_PROMPT] == "keep me"


def test_drop_empty_fields_keeps_populated_vlm_overrides() -> None:
    """Populated override values survive the empty-field cleanup."""
    flow = HomeGenerativeAgentOptionsFlow.__new__(HomeGenerativeAgentOptionsFlow)
    options: dict[str, Any] = {
        CONF_VLM_RESPONSE_LANGUAGE: "Czech",
        CONF_VLM_PROMPT_EXTRA: "Ignore cars in the driveway.",
    }
    flow._drop_empty_fields(options)
    assert options[CONF_VLM_RESPONSE_LANGUAGE] == "Czech"
    assert options[CONF_VLM_PROMPT_EXTRA] == "Ignore cars in the driveway."
