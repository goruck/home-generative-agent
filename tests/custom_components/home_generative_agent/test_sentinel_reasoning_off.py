# ruff: noqa: S101
"""
A Sentinel model on Ollama that reasons with reasoning off says so (#1065).

Some Ollama builds ignore ``reasoning: False`` and write a <think> block into
the reply. Sentinel calls then run slow enough to time out, and the log only
said "timed out". The warning names the model that answered and the fix, once
per model and setup.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from custom_components.home_generative_agent.core import utils
from custom_components.home_generative_agent.core.fallback import FallbackChatModel
from custom_components.home_generative_agent.core.utils import (
    reasoning_field,
    reset_reasoning_despite_off_warnings,
    run_sentinel_model_call,
)

_MESSAGES = [HumanMessage(content="Explain this finding.")]
_THINKING_REPLY = (
    "<think>The user wants an explanation. Let me reason...</think>A door."
)


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_reasoning_despite_off_warnings()
    # Tests time out back to back; production counts one per 5 minutes.
    monkeypatch.setattr(utils, "_REASONING_OFF_TIMEOUT_SPACING_S", 0.0)


def _member(
    reply: str | None,
    *,
    model: str = "qwen3:4b",
    reasoning: bool = False,
    delay: float = 0,
) -> Any:
    """One provider as __init__ binds it; ``reply=None`` cannot connect."""

    def _answer(_messages: Any) -> AIMessage:
        if delay:
            time.sleep(delay)
        if reply is None:
            msg = "connection refused"
            raise httpx.ConnectError(msg)
        # ChatOllama reports the model that answered here.
        return AIMessage(content=reply, response_metadata={"model": model})

    return RunnableLambda(_answer).with_config(
        config={
            "configurable": {
                "model": model,
                "temperature": 0.2,
                **reasoning_field(model=model, enabled=reasoning),
            }
        }
    )


def _chain(*members: Any) -> FallbackChatModel:
    return FallbackChatModel(
        [(member, "edge", f"p{index}") for index, member in enumerate(members)]
    )


def _sentinel_model(reply: str | None, **kwargs: Any) -> FallbackChatModel:
    return _chain(_member(reply, **kwargs))


async def _call(model: Any, *, category: str = "explain", budget_s: float = 5) -> Any:
    return await run_sentinel_model_call(
        model,
        _MESSAGES,
        deployment="edge",
        category=category,
        admission_timeout_s=1.0,
        call_timeout_s=budget_s,
    )


async def _time_out(model: Any, times: int) -> None:
    for _ in range(times):
        with pytest.raises(TimeoutError):
            await _call(model, budget_s=0.1)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "reason" in record.getMessage()
    ]


@pytest.mark.asyncio
async def test_a_reply_that_reasons_with_reasoning_off_names_the_fix(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    model = _sentinel_model(_THINKING_REPLY)

    result = await _call(model)
    await _call(model, category="triage")

    assert result.content == _THINKING_REPLY  # the caller still strips it
    [warning] = _warnings(caplog)
    assert "qwen3:4b" in warning
    assert "although reasoning is off" in warning
    assert "Conversation model while Sentinel runs on it" in warning


@pytest.mark.asyncio
async def test_a_reply_with_only_a_closing_think_tag_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Qwen3 templates put <think> in the prompt; the reply opens mid-reasoning."""
    caplog.set_level(logging.WARNING)

    await _call(_sentinel_model("Let me think.</think>A door."))

    assert len(_warnings(caplog)) == 1


@pytest.mark.asyncio
async def test_timeouts_in_a_row_with_reasoning_off_say_what_it_may_be(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One timeout is often a cold load; three in a row are worth a warning."""
    caplog.set_level(logging.WARNING)
    model = _sentinel_model("A door.", delay=0.5)

    await _time_out(model, utils._REASONING_OFF_TIMEOUTS_BEFORE_WARNING - 1)
    assert _warnings(caplog) == []
    await _time_out(model, 2)

    [warning] = _warnings(caplog)
    assert "timed out 3 times since its last reply, with reasoning off" in warning
    assert "qwen3:4b" in warning


@pytest.mark.asyncio
async def test_timeouts_close_together_count_once(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Calls queued behind an abandoned one time out together: one slow moment.

    A timed-out call keeps running at the server, so the next calls of the
    same run wait behind it and time out as well.
    """
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(utils, "_REASONING_OFF_TIMEOUT_SPACING_S", 300.0)

    await _time_out(_sentinel_model("A door.", delay=0.5), 4)

    assert _warnings(caplog) == []


@pytest.mark.asyncio
async def test_empty_think_tags_are_not_reasoning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A model that honors "off" may still write an empty <think></think>."""
    caplog.set_level(logging.WARNING)

    await _call(_sentinel_model("<think>\n\n</think>A door."))
    await _call(_sentinel_model("\n</think>A door."))

    assert _warnings(caplog) == []


@pytest.mark.asyncio
async def test_a_single_model_is_named_whatever_name_the_server_echoes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``qwen3`` may come back as ``qwen3:latest``; the reply is still its own."""
    caplog.set_level(logging.WARNING)
    member = _member(_THINKING_REPLY, model="qwen3")

    def _echo_latest(_messages: Any) -> AIMessage:
        return AIMessage(
            content=_THINKING_REPLY, response_metadata={"model": "qwen3:latest"}
        )

    bound = RunnableLambda(_echo_latest).with_config(config=member.config)

    await _call(_chain(bound))

    [warning] = _warnings(caplog)
    assert "Ollama model qwen3 " in warning


@pytest.mark.asyncio
async def test_a_tag_with_reasoning_on_elsewhere_in_the_chain_is_not_named(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The same tag reasoning on and off: the reply cannot say which wrote it."""
    caplog.set_level(logging.WARNING)
    model = _chain(
        _member(_THINKING_REPLY, model="qwen3:4b", reasoning=True),
        _member("A door.", model="qwen3:4b"),
    )

    await _call(model)

    assert _warnings(caplog) == []


@pytest.mark.asyncio
async def test_a_reply_resets_the_timeout_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    slow = _sentinel_model("A door.", delay=0.5)
    quick = _sentinel_model("A door.")

    await _time_out(slow, 2)
    await _call(quick)
    await _time_out(slow, 2)

    assert _warnings(caplog) == []


@pytest.mark.asyncio
async def test_a_fallback_reply_is_blamed_on_the_model_that_wrote_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A primary that cannot connect is not named for its fallback's reply."""
    caplog.set_level(logging.WARNING)
    model = _chain(
        _member(None, model="qwen3:8b"),
        _member(_THINKING_REPLY, model="qwen3:4b"),
    )

    await _call(model)

    [warning] = _warnings(caplog)
    assert "qwen3:4b" in warning
    assert "qwen3:8b" not in warning


@pytest.mark.asyncio
async def test_a_reasoning_on_reply_does_not_blame_an_unused_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    model = _chain(
        _member(_THINKING_REPLY, model="qwen3:8b", reasoning=True),
        _member("A door.", model="qwen3:4b"),
    )

    await _call(model)

    assert _warnings(caplog) == []


@pytest.mark.asyncio
async def test_a_timeout_in_a_fallback_chain_names_no_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The chain does not say which member ran out of time."""
    caplog.set_level(logging.WARNING)
    model = _chain(
        _member("A door.", model="qwen3:4b", delay=0.5),
        _member("A door.", model="qwen3:8b", delay=0.5),
    )

    await _time_out(model, utils._REASONING_OFF_TIMEOUTS_BEFORE_WARNING)

    assert _warnings(caplog) == []


@pytest.mark.asyncio
async def test_a_reload_lets_the_warning_show_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """After a re-pull and reload, a model that still reasons is named again."""
    caplog.set_level(logging.WARNING)
    model = _sentinel_model(_THINKING_REPLY)

    await _call(model)
    reset_reasoning_despite_off_warnings()  # async_setup_entry does this
    await _call(model)

    assert len(_warnings(caplog)) == 2


@pytest.mark.asyncio
async def test_no_warning_when_reasoning_is_on_or_the_reply_is_plain(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)

    await _call(_sentinel_model(_THINKING_REPLY, reasoning=True))
    await _call(_sentinel_model("A door."))
    # Not a thinking model: reasoning_field binds nothing, so nothing is said.
    await _call(_sentinel_model(_THINKING_REPLY, model="llama3.2:3b"))

    assert _warnings(caplog) == []
