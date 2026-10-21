# ruff: noqa: S101
"""Tests for stalled model providers: fail over, fail fast, recover (#828)."""

from __future__ import annotations

import asyncio
from time import monotonic
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from homeassistant.exceptions import HomeAssistantError
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from custom_components.home_generative_agent.agent import graph
from custom_components.home_generative_agent.core import provider_stall
from custom_components.home_generative_agent.core.fallback import (
    FallbackChatModel,
    FallbackVLM,
    ModelChainTimeoutError,
    is_timeout_error,
)
from custom_components.home_generative_agent.core.provider_stall import (
    ProviderStallTracker,
    member_timeout,
    model_call_budget,
)
from custom_components.home_generative_agent.core.utils import (
    run_sentinel_model_call,
)

_MESSAGES: list[AnyMessage] = [HumanMessage(content="hi")]


class _Model:
    """A provider that answers, or hangs until released."""

    def __init__(self, answer: str, *, hang: bool = False) -> None:
        self.answer = answer
        self.hang = hang
        self.calls = 0
        self._released = asyncio.Event()

    def release(self) -> None:
        self.hang = False
        self._released.set()

    async def ainvoke(
        self, _input: Any, _config: Any = None, **_kwargs: Any
    ) -> AIMessage:
        self.calls += 1
        if self.hang:
            await self._released.wait()
        return AIMessage(content=self.answer)


_probe_tasks: list[asyncio.Task[None]] = []


@pytest.fixture(autouse=True)
async def _cancel_probe_tasks() -> Any:
    """Stop probes a test left running (the entry unload does this in HA)."""
    yield
    for task in _probe_tasks:
        task.cancel()
    await asyncio.gather(*_probe_tasks, return_exceptions=True)
    _probe_tasks.clear()


def _tracker(changes: list[str] | None = None) -> ProviderStallTracker:
    """Build a tracker whose background tasks run on the test loop."""

    def _create(_hass: Any, coro: Any, _name: str) -> asyncio.Task[None]:
        task = asyncio.get_running_loop().create_task(coro)
        _probe_tasks.append(task)
        return task

    entry = SimpleNamespace(async_create_background_task=_create)
    log = changes if changes is not None else []
    return ProviderStallTracker(
        SimpleNamespace(),  # type: ignore[arg-type]
        entry,  # type: ignore[arg-type]
        provider_name=lambda provider_id: f"Name of {provider_id}",
        on_change=lambda: log.append("changed"),
    )


def _chain(
    *models: _Model, tracker: ProviderStallTracker | None = None
) -> FallbackChatModel:
    return FallbackChatModel(
        [(model, "edge", f"p{i}") for i, model in enumerate(models)],
        stall_tracker=tracker,
    )


@pytest.fixture(autouse=True)
def _short_conversation_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph, "_LLM_INVOKE_TIMEOUT_S", 2.0)


@pytest.mark.asyncio
async def test_a_hung_primary_fails_over_within_the_conversation_budget() -> None:
    """The fallback answers instead of the whole turn timing out."""
    hung, fallback = _Model("primary", hang=True), _Model("fallback")
    started = monotonic()

    result = await graph._invoke_model(_chain(hung, fallback), _MESSAGES, {})

    assert result.content == "fallback"
    assert monotonic() - started < 2.0
    assert hung.calls == 1


@pytest.mark.asyncio
async def test_a_hung_primary_fails_over_for_sentinel_calls() -> None:
    """The budget reaches the Sentinel call task, so triage fails over too."""
    hung, fallback = _Model("primary", hang=True), _Model("fallback")

    result = await run_sentinel_model_call(
        _chain(hung, fallback),
        _MESSAGES,
        deployment="edge",
        category="triage",
        admission_timeout_s=1.0,
        call_timeout_s=2.0,
    )

    assert result.content == "fallback"


@pytest.mark.asyncio
async def test_three_timeouts_stall_the_provider_and_later_calls_skip_it() -> None:
    """After the third timeout in a row the hung provider is not called again."""
    changes: list[str] = []
    tracker = _tracker(changes)
    hung, fallback = _Model("primary", hang=True), _Model("fallback")
    chain = _chain(hung, fallback, tracker=tracker)

    for _ in range(provider_stall.STALL_TIMEOUT_THRESHOLD):
        assert (await graph._invoke_model(chain, _MESSAGES, {})).content == "fallback"
    assert tracker.is_stalled("p0")
    assert changes == ["changed"]
    [status] = tracker.status()
    assert status["provider"] == "Name of p0"
    assert status["consecutive_timeouts"] == provider_stall.STALL_TIMEOUT_THRESHOLD

    started = monotonic()
    assert (await graph._invoke_model(chain, _MESSAGES, {})).content == "fallback"
    assert monotonic() - started < 0.5
    assert hung.calls == provider_stall.STALL_TIMEOUT_THRESHOLD


@pytest.mark.asyncio
async def test_a_stalled_only_provider_fails_fast_with_a_clear_error() -> None:
    """With nothing to fall back to, the reply names the stalled provider."""
    tracker = _tracker()
    hung = _Model("primary", hang=True)
    chain = _chain(hung, tracker=tracker)
    for _ in range(provider_stall.STALL_TIMEOUT_THRESHOLD):
        with pytest.raises(HomeAssistantError, match="timed out"):
            await graph._invoke_model(chain, _MESSAGES, {})

    started = monotonic()
    with pytest.raises(HomeAssistantError) as err:
        await graph._invoke_model(chain, _MESSAGES, {})

    assert monotonic() - started < 0.5
    assert "Name of p0 stopped answering" in str(err.value)
    assert not isinstance(err.value, TimeoutError)


@pytest.mark.asyncio
async def test_an_answer_before_the_threshold_resets_the_count() -> None:
    """Only timeouts in a row count; one answer starts the count over."""
    tracker = _tracker()
    model = _Model("primary", hang=True)
    chain = _chain(model, tracker=tracker)
    for _ in range(provider_stall.STALL_TIMEOUT_THRESHOLD - 1):
        with pytest.raises(HomeAssistantError):
            await graph._invoke_model(chain, _MESSAGES, {})
    model.release()
    await graph._invoke_model(chain, _MESSAGES, {})
    model.hang = True
    model._released = asyncio.Event()

    with pytest.raises(HomeAssistantError):
        await graph._invoke_model(chain, _MESSAGES, {})

    assert not tracker.is_stalled("p0")


@pytest.mark.asyncio
async def test_the_probe_clears_the_stall_once_the_provider_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The background probe brings a recovered provider back."""
    monkeypatch.setattr(provider_stall, "STALL_PROBE_INTERVAL_S", 0.05)
    monkeypatch.setattr(provider_stall, "STALL_PROBE_TIMEOUT_S", 0.1)
    changes: list[str] = []
    tracker = _tracker(changes)
    model = _Model("primary", hang=True)
    for _ in range(provider_stall.STALL_TIMEOUT_THRESHOLD):
        tracker.record_timeout("p0", model)
    assert tracker.is_stalled("p0")

    await asyncio.sleep(0.3)  # probes time out while it still hangs
    assert tracker.is_stalled("p0")
    assert tracker.status()[0]["last_probe"] is not None

    model.release()
    for _ in range(20):
        await asyncio.sleep(0.05)
        if not tracker.is_stalled("p0"):
            break

    assert not tracker.is_stalled("p0")
    assert tracker.status() == []
    assert changes[0] == changes[-1] == "changed"


@pytest.mark.asyncio
async def test_every_provider_timing_out_is_still_a_timeout() -> None:
    """Callers that handle TimeoutError (triage's reason, camera counts) still do."""
    chain = _chain(_Model("a", hang=True), _Model("b", hang=True))

    with model_call_budget(0.5), pytest.raises(ModelChainTimeoutError) as err:
        await chain.ainvoke(_MESSAGES)

    assert isinstance(err.value, TimeoutError)
    assert isinstance(err.value, HomeAssistantError)


@pytest.mark.asyncio
async def test_the_vlm_chain_fails_over_and_records_the_answer() -> None:
    """Camera analysis gets the same split budget and stall tracking."""
    tracker = _tracker()
    hung, fallback = _Model("primary", hang=True), _Model("fallback")
    chain = FallbackVLM(
        [(hung, "edge", "p0"), (fallback, "edge", "p1")], stall_tracker=tracker
    )

    with model_call_budget(1.0):
        result = await chain.ainvoke(_MESSAGES)

    assert result.content == "fallback"
    assert tracker.status() == []
    assert tracker._states["p0"].consecutive_timeouts == 1


def test_member_timeout_splits_the_callers_budget() -> None:
    """Every provider but the last gets half of what is left; no budget, no limit."""
    assert member_timeout(last=False) is None
    with model_call_budget(10.0):
        first = member_timeout(last=False)
        last = member_timeout(last=True)
        with model_call_budget(100.0):
            nested = member_timeout(last=True)
    assert first is not None
    assert 4.9 < first <= 5.0
    assert last is not None
    assert 8.9 < last <= 9.0
    assert nested is not None
    assert nested <= 9.0  # an inner budget never extends the outer one
    with model_call_budget(2.0):
        short_last = member_timeout(last=True)
    assert short_last is not None
    assert 1.7 < short_last <= 1.8  # a short budget keeps most of its time


class APITimeoutError(Exception):
    """Stand-in for openai/anthropic APITimeoutError (not a TimeoutError)."""


def test_sdk_timeouts_count_as_timeouts() -> None:
    """SDK timeout errors are recognised, also when wrapped as a cause."""
    wrapped = HomeAssistantError("model failed")
    wrapped.__cause__ = APITimeoutError("Request timed out.")
    assert is_timeout_error(TimeoutError())
    assert is_timeout_error(httpx.ReadTimeout("slow"))
    assert is_timeout_error(APITimeoutError("Request timed out."))
    assert is_timeout_error(wrapped)
    assert not is_timeout_error(ConnectionRefusedError())
