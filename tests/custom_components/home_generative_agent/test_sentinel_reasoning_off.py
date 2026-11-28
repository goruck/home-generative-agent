# ruff: noqa: S101
"""
A Sentinel model on Ollama that reasons with reasoning off says so (#1065).

Some Ollama builds ignore ``reasoning: False`` and write a <think> block into
the reply. Sentinel calls then run slow enough to time out, and the log only
said "timed out". The warning names the model and the fix, once per model.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from custom_components.home_generative_agent.core import utils
from custom_components.home_generative_agent.core.fallback import FallbackChatModel
from custom_components.home_generative_agent.core.utils import (
    reasoning_field,
    run_sentinel_model_call,
)

_MESSAGES = [HumanMessage(content="Explain this finding.")]
_THINKING_REPLY = (
    "<think>The user wants an explanation. Let me reason...</think>A door."
)


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils, "_REASONING_DESPITE_OFF_WARNED", set(), raising=False)


def _sentinel_model(
    reply: str, *, model: str = "qwen3:4b", reasoning: bool = False, delay: float = 0
) -> FallbackChatModel:
    """Build the Sentinel model as __init__ does: bound config in a chain."""

    def _answer(_messages: Any) -> AIMessage:
        if delay:
            time.sleep(delay)
        return AIMessage(content=reply)

    bound = RunnableLambda(_answer).with_config(
        config={
            "configurable": {
                "model": model,
                "temperature": 0.2,
                **reasoning_field(model=model, enabled=reasoning),
            }
        }
    )
    return FallbackChatModel([(bound, "edge", "p0")])


async def _call(model: Any, *, category: str = "explain", budget_s: float = 5) -> Any:
    return await run_sentinel_model_call(
        model,
        _MESSAGES,
        deployment="edge",
        category=category,
        admission_timeout_s=1.0,
        call_timeout_s=budget_s,
    )


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
    assert "Choose another Sentinel model" in warning


@pytest.mark.asyncio
async def test_a_reply_with_only_a_closing_think_tag_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Qwen3 templates put <think> in the prompt; the reply opens mid-reasoning."""
    caplog.set_level(logging.WARNING)

    await _call(_sentinel_model("Let me think.</think>A door."))

    assert len(_warnings(caplog)) == 1


@pytest.mark.asyncio
async def test_a_timeout_with_reasoning_off_says_what_it_may_be(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    model = _sentinel_model("A door.", delay=0.5)

    for _ in range(2):
        with pytest.raises(TimeoutError):
            await _call(model, budget_s=0.1)

    [warning] = _warnings(caplog)
    assert "timed out with reasoning off" in warning
    assert "qwen3:4b" in warning


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
