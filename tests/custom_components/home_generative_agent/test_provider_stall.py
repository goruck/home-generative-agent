# ruff: noqa: S101
"""Tests for stalled model providers: fail over, fail fast, recover (#828)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import httpx2
import openai
import pytest
from homeassistant.exceptions import HomeAssistantError
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from custom_components.home_generative_agent.agent import graph, tools
from custom_components.home_generative_agent.core import fallback, provider_stall
from custom_components.home_generative_agent.core.fallback import (
    FallbackChatModel,
    FallbackVLM,
    ModelChainTimeoutError,
    ProviderStalledError,
    is_timeout_error,
    with_member_configurable,
)
from custom_components.home_generative_agent.core.provider_stall import (
    ProviderStallTracker,
    budget_is_split,
    member_timeout,
    model_call_budget,
)
from custom_components.home_generative_agent.core.utils import (
    run_sentinel_model_call,
)
from custom_components.home_generative_agent.core.video_analyzer import VideoAnalyzer

_MESSAGES: list[AnyMessage] = [HumanMessage(content="hi")]


class _Model:
    """A provider that answers, or hangs until released."""

    def __init__(self, answer: str, *, hang: bool = False) -> None:
        self.answer = answer
        self.hang = hang
        self.calls = 0
        self.cancelled = False
        self._released = asyncio.Event()

    def release(self) -> None:
        self.hang = False
        self._released.set()

    async def ainvoke(
        self, _input: Any, _config: Any = None, **_kwargs: Any
    ) -> AIMessage:
        self.calls += 1
        if self.hang:
            try:
                await self._released.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
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


def _tracker(
    changes: list[str] | None = None,
    recovered: list[str] | None = None,
    run_probe: Any = None,
) -> ProviderStallTracker:
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
        on_recovered=recovered.append if recovered is not None else None,
        run_probe=run_probe,
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
    """Scale the conversation budget and the split/evidence floors down."""
    monkeypatch.setattr(graph, "_LLM_INVOKE_TIMEOUT_S", 2.0)
    monkeypatch.setattr(provider_stall, "SPLIT_MIN_BUDGET_S", 0.5)
    monkeypatch.setattr(provider_stall, "STALL_EVIDENCE_MIN_S", 0.2)


def _real_floors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_stall, "SPLIT_MIN_BUDGET_S", 90.0)
    monkeypatch.setattr(provider_stall, "STALL_EVIDENCE_MIN_S", 40.0)


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
    recovered: list[str] = []
    tracker = _tracker(changes, recovered)
    model = _Model("primary", hang=True)
    for _ in range(provider_stall.STALL_TIMEOUT_THRESHOLD):
        tracker.record_timeout("p0", lambda: model.ainvoke(_MESSAGES))
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
    # Setup resets the circuit breakers its timeouts may have opened.
    assert recovered == ["p0"]


@pytest.mark.asyncio
async def test_a_probe_round_is_skipped_while_chat_holds_the_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate skips a round without probing; the stall clears once it opens."""
    monkeypatch.setattr(provider_stall, "STALL_PROBE_INTERVAL_S", 0.1)
    monkeypatch.setattr(provider_stall, "STALL_PROBE_TIMEOUT_S", 0.1)
    monkeypatch.setattr(provider_stall, "STALL_PROBE_RETRY_S", 0.01)
    gate_open = False
    rounds: list[str] = []

    async def _run_probe(provider_id: str, probe: Any) -> Any:
        rounds.append(provider_id)
        if not gate_open:
            # A skipped round is retried after STALL_PROBE_RETRY_S, not a full
            # interval.
            raise provider_stall.StallProbeDeferredError(
                provider_stall.PROBE_GATE_CLOSED
            )
        return await probe()

    changes: list[str] = []
    tracker = _tracker(changes, run_probe=_run_probe)
    model = _Model("primary")
    for _ in range(provider_stall.STALL_TIMEOUT_THRESHOLD):
        tracker.record_timeout("p0", lambda: model.ainvoke(_MESSAGES))
    assert tracker.is_stalled("p0")

    await asyncio.sleep(0.25)  # rounds come and go, none is sent
    # gstack-timing-ok: lower bound; with the full 0.1 s interval at most 2 rounds fit
    assert len(rounds) >= 4
    assert tracker.is_stalled("p0")
    status = tracker.status()[0]
    assert status["last_probe"] is None
    assert status["probe_rounds_skipped"] == len(rounds)
    assert model.calls == 0
    # The sensor is refreshed on the first skip of the streak only: the stall
    # itself ("changed") and that first skip.
    assert changes == ["changed", "changed"]

    gate_open = True
    for _ in range(20):
        await asyncio.sleep(0.05)
        if not tracker.is_stalled("p0"):
            break

    assert not tracker.is_stalled("p0")
    assert model.calls == 1

    # A second stall starts its skip count from zero.
    for _ in range(provider_stall.STALL_TIMEOUT_THRESHOLD):
        tracker.record_timeout("p0", lambda: model.ainvoke(_MESSAGES))
    assert tracker.status()[0]["probe_rounds_skipped"] == 0


@pytest.mark.asyncio
async def test_a_recovery_recorded_meanwhile_lets_a_sent_probe_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """record_success from another call must not cancel a probe request in flight."""
    monkeypatch.setattr(provider_stall, "STALL_PROBE_INTERVAL_S", 0.02)
    monkeypatch.setattr(provider_stall, "STALL_PROBE_TIMEOUT_S", 1.0)
    recovered: list[str] = []
    tracker = _tracker(recovered=recovered)
    model = _Model("primary", hang=True)
    for _ in range(provider_stall.STALL_TIMEOUT_THRESHOLD):
        tracker.record_timeout("p0", lambda: model.ainvoke(_MESSAGES))
    for _ in range(20):
        await asyncio.sleep(0.01)
        if model.calls:
            break
    assert model.calls == 1  # the probe request is with the provider

    tracker.record_success("p0")  # an older call answered meanwhile
    assert not tracker.is_stalled("p0")
    await asyncio.sleep(0.05)
    assert not model.cancelled  # the probe was left to finish
    model.release()
    await asyncio.sleep(0.05)
    assert recovered == ["p0"]


@pytest.mark.asyncio
async def test_a_restall_while_a_probe_is_out_reuses_its_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second stall must not start a second probe loop beside the one in flight."""
    monkeypatch.setattr(provider_stall, "STALL_PROBE_INTERVAL_S", 0.02)
    monkeypatch.setattr(provider_stall, "STALL_PROBE_TIMEOUT_S", 1.0)
    recovered: list[str] = []
    tracker = _tracker(recovered=recovered)
    model = _Model("primary", hang=True)
    for _ in range(provider_stall.STALL_TIMEOUT_THRESHOLD):
        tracker.record_timeout("p0", lambda: model.ainvoke(_MESSAGES))
    for _ in range(20):
        await asyncio.sleep(0.01)
        if model.calls:
            break
    assert model.calls == 1
    loops_before = len(_probe_tasks)

    tracker.record_success("p0")  # an older call answered; the probe is still out
    for _ in range(provider_stall.STALL_TIMEOUT_THRESHOLD):
        tracker.record_timeout("p0", lambda: model.ainvoke(_MESSAGES))
    assert tracker.is_stalled("p0")
    assert len(_probe_tasks) == loops_before  # no second loop

    model.release()
    for _ in range(20):
        await asyncio.sleep(0.02)
        if not tracker.is_stalled("p0"):
            break
    assert not tracker.is_stalled("p0")
    assert not model.cancelled
    assert model.calls == 1  # the one probe out there settled the second stall too
    assert recovered == ["p0", "p0"]


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
    # openai 3 and anthropic 2 raise httpx2's, which do not subclass httpx's (#878).
    assert is_timeout_error(httpx2.ReadTimeout("slow"))
    assert is_timeout_error(APITimeoutError("Request timed out."))
    assert is_timeout_error(wrapped)
    assert not is_timeout_error(ConnectionRefusedError())


@pytest.mark.asyncio
async def test_a_short_sentinel_budget_is_neither_split_nor_a_stall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow model missing triage's short deadline is not a hung provider."""
    _real_floors(monkeypatch)
    tracker = _tracker()
    slow, fallback = _Model("primary", hang=True), _Model("fallback")
    chain = _chain(slow, fallback, tracker=tracker)

    for _ in range(provider_stall.STALL_TIMEOUT_THRESHOLD + 1):
        with pytest.raises(TimeoutError):
            await run_sentinel_model_call(
                chain,
                _MESSAGES,
                deployment="edge",
                category="triage",
                admission_timeout_s=1.0,
                call_timeout_s=0.3,
            )

    # The primary had the whole budget (no split), as before #828 ...
    assert fallback.calls == 0
    # ... and missing it is not stall evidence.
    assert not tracker.is_stalled("p0")
    assert tracker.status() == []


@pytest.mark.asyncio
async def test_the_alert_names_the_next_provider_actually_tried() -> None:
    """A stalled provider is skipped in the fallback notice too."""
    tracker = _tracker()
    hung, stalled, fallback = (
        _Model("p0", hang=True),
        _Model("p1"),
        _Model("p2"),
    )
    for _ in range(provider_stall.STALL_TIMEOUT_THRESHOLD):
        tracker.record_timeout("p1", lambda: stalled.ainvoke(_MESSAGES))
    alert = MagicMock()
    chain = FallbackChatModel(
        [(hung, "edge", "p0"), (stalled, "edge", "p1"), (fallback, "cloud", "p2")],
        alert_callback=alert,
        stall_tracker=tracker,
    )

    with model_call_budget(1.0):
        assert (await chain.ainvoke(_MESSAGES)).content == "p2"

    assert alert.call_args.kwargs["fallback_to"] == "p2"
    assert stalled.calls == 0


@pytest.mark.asyncio
async def test_a_lone_provider_raises_no_fallback_alert() -> None:
    """With nothing to fall back to there is no 'all providers failed' alert."""
    alert = MagicMock()
    chain = FallbackChatModel(
        [(_Model("p0", hang=True), "edge", "p0")], alert_callback=alert
    )

    with model_call_budget(0.5), pytest.raises(TimeoutError) as err:
        await chain.ainvoke(_MESSAGES)

    alert.assert_not_called()
    # Its own error, unwrapped, as before every model was wrapped.
    assert not isinstance(err.value, HomeAssistantError)


def _rejected_temperature() -> Exception:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {
        "message": "Unsupported value: 'temperature' does not support 0.2.",
        "type": "invalid_request_error",
        "param": "temperature",
        "code": "unsupported_value",
    }
    return openai.BadRequestError(
        "Error code: 400",
        response=httpx.Response(400, request=request),  # pyright: ignore[reportArgumentType]
        body=body,
    )


@pytest.mark.asyncio
async def test_the_sampling_retry_shares_the_providers_deadline() -> None:
    """A rebound provider that hangs still leaves the fallback time to answer."""
    bound_p0 = AsyncMock()
    bound_p0.ainvoke.side_effect = [_rejected_temperature(), _rejected_temperature()]
    stripped_p0 = MagicMock()
    stripped_p0.bind_tools.return_value = _Model("rebound", hang=True)
    src_p0 = MagicMock()
    src_p0.bind_tools.return_value = bound_p0
    src_p0.with_config.return_value = stripped_p0
    fallback_model = _Model("fallback")
    src_p1 = MagicMock()
    src_p1.bind_tools.return_value = fallback_model
    chain = FallbackChatModel([(src_p0, "cloud", "p0"), (src_p1, "cloud", "p1")])
    bound = chain.bind_tools([{"name": "t"}])
    started = monotonic()

    with model_call_budget(2.0):
        result = await bound.ainvoke(_MESSAGES)

    assert result.content == "fallback"
    assert monotonic() - started < 2.0


@pytest.mark.asyncio
async def test_the_probe_drops_parameters_the_model_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe goes through the same parameter drop as live calls."""
    spy = AsyncMock(return_value=AIMessage(content="OK"))
    monkeypatch.setattr(fallback, "ainvoke_dropping_unsupported_params", spy)
    model = object()

    await fallback._probe_call(model)()

    assert spy.await_args is not None
    assert spy.await_args.args[0] is model


class _Configured:
    """A model with a bound configurable section, like a LangChain binding."""

    def __init__(self, configurable: dict[str, Any]) -> None:
        self.config = {"configurable": configurable}

    def with_config(self, *, config: dict[str, Any]) -> _Configured:
        return _Configured({**self.config["configurable"], **config["configurable"]})


def _video_cfg(model: Any, deployment: str) -> dict[str, Any]:
    cfg = dict(model.config["configurable"])
    cfg["num_predict"] = 128
    if deployment == "edge" and "reasoning" in cfg:
        cfg["reasoning"] = False
    return cfg


def test_member_configurable_reads_each_providers_own_settings() -> None:
    """Video limits turn thinking off per member, not from the empty wrapper."""
    edge = _Configured({"model": "qwen3:8b", "reasoning": True})
    cloud = _Configured({"model": "gpt-x", "reasoning": True})
    plain = _Configured({"model": "llava"})
    chain = FallbackChatModel([(edge, "edge", "p0"), (cloud, "cloud", "p1")])

    limited = with_member_configurable(chain, _video_cfg, "edge")

    assert isinstance(limited, FallbackChatModel)
    [(first, _, _), (second, _, _)] = limited.chain
    assert first.config["configurable"] == {
        "model": "qwen3:8b",
        "reasoning": False,
        "num_predict": 128,
    }
    # Each member keeps its own model; a cloud one keeps its reasoning.
    assert second.config["configurable"]["model"] == "gpt-x"
    assert second.config["configurable"]["reasoning"] is True
    bare = with_member_configurable(plain, _video_cfg, "edge")
    assert bare.config["configurable"] == {"model": "llava", "num_predict": 128}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [ModelChainTimeoutError("slow"), ProviderStalledError("stalled")]
)
async def test_analyze_image_leaves_timeouts_and_stalls_to_the_caller(
    error: Exception,
) -> None:
    """Camera analysis counts these itself; they are not an error caption."""
    model = AsyncMock()
    model.ainvoke.side_effect = error

    with pytest.raises(type(error)):
        await tools.analyze_image(model, b"image")


@pytest.mark.asyncio
async def test_analyze_image_still_captions_other_errors() -> None:
    model = AsyncMock()
    model.ainvoke.side_effect = HomeAssistantError("bad image")

    assert await tools.analyze_image(model, b"image") == tools.VLM_ERROR_CAPTION


def test_a_camera_frame_budget_is_split_from_the_budget_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """90 s splits however long setup took, and each share is stall evidence."""
    _real_floors(monkeypatch)
    with model_call_budget(90.0):
        assert budget_is_split()
        share = member_timeout(last=False)
    assert share is not None
    assert 44.0 < share <= 45.0
    assert provider_stall.counts_as_stall_evidence(share - 1.0)


@pytest.mark.asyncio
async def test_every_member_of_a_long_chain_gets_a_real_share() -> None:
    """Two hung providers still leave the third time to answer."""
    hung_a, hung_b, healthy = (
        _Model("a", hang=True),
        _Model("b", hang=True),
        _Model("c"),
    )
    chain = FallbackChatModel(
        [(hung_a, "edge", "p0"), (hung_b, "edge", "p1"), (healthy, "cloud", "p2")]
    )

    with model_call_budget(2.0):
        result = await chain.ainvoke(_MESSAGES)

    assert result.content == "c"


class _FailsWith:
    """A provider that raises *error* after *delay* seconds."""

    def __init__(self, error: Exception, delay: float = 0.0) -> None:
        self.error = error
        self.delay = delay
        self.calls = 0

    async def ainvoke(self, _input: Any, _config: Any = None, **_kw: Any) -> Any:
        self.calls += 1
        await asyncio.sleep(self.delay)
        raise self.error


@pytest.mark.asyncio
async def test_an_early_sdk_timeout_still_moves_on_in_an_unsplit_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a used-up budget stops a short call, not any timeout error."""
    _real_floors(monkeypatch)
    early = _FailsWith(APITimeoutError("Request timed out."))
    fallback_model = _Model("fallback")
    chain = FallbackChatModel([(early, "cloud", "p0"), (fallback_model, "edge", "p1")])

    with model_call_budget(1.0):
        result = await chain.ainvoke(_MESSAGES)

    assert result.content == "fallback"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "delay"),
    [
        (APITimeoutError("Request timed out."), 0.0),  # gave up early
        (httpx.PoolTimeout("no free connection"), 0.3),  # our own pool
        (httpx2.PoolTimeout("no free connection"), 0.3),  # same, on httpx2 (#878)
    ],
)
async def test_quick_or_client_side_timeouts_are_not_a_stall(
    monkeypatch: pytest.MonkeyPatch, error: Exception, delay: float
) -> None:
    """Stall evidence is a long wait for the provider, not any timeout."""
    monkeypatch.setattr(provider_stall, "STALL_EVIDENCE_MIN_S", 0.2)
    tracker = _tracker()
    chain = FallbackChatModel(
        [(_FailsWith(error, delay), "edge", "p0"), (_Model("ok"), "cloud", "p1")],
        stall_tracker=tracker,
    )

    for _ in range(provider_stall.STALL_TIMEOUT_THRESHOLD):
        with model_call_budget(2.0):
            await chain.ainvoke(_MESSAGES)

    assert not tracker.is_stalled("p0")


def test_a_tool_bound_member_is_probed_through_its_unbound_source() -> None:
    """Parameters baked into a tool binding cannot be dropped by the probe."""
    source = MagicMock()
    source.bind_tools.return_value = _Model("bound")
    chain = FallbackChatModel([(source, "cloud", "p0")]).bind_tools([{"name": "t"}])

    assert fallback._probe_model(chain, 0) is source
    assert fallback._probe_model(FallbackVLM([(source, "edge", "p0")]), 0) is source


def test_member_configurable_keeps_the_sampling_rebind_sources() -> None:
    """Cloning a tool-bound chain keeps its rebind sources, configured alike."""
    source = _Configured({"model": "qwen3:8b", "reasoning": True})
    source.bind_tools = lambda _tools, **_kw: _Configured(  # type: ignore[attr-defined]
        dict(source.config["configurable"])
    )
    bound = FallbackChatModel([(source, "edge", "p0")]).bind_tools([{"name": "t"}])

    limited = with_member_configurable(bound, _video_cfg, "edge")

    assert isinstance(limited, FallbackChatModel)
    [rebind] = limited._rebind_sources
    assert rebind is not None
    assert rebind[0].config["configurable"]["reasoning"] is False
    assert rebind[1] == [{"name": "t"}]


def test_a_stalled_provider_is_not_logged_as_an_error_per_frame(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The tracker already warned once; frames skipped while stalled stay quiet."""
    with caplog.at_level(logging.DEBUG):
        VideoAnalyzer._log_snapshot_error(
            cast("VideoAnalyzer", None), "cam", Path("x.jpg"), ProviderStalledError("s")
        )

    assert [r.levelno for r in caplog.records] == [logging.DEBUG]


def test_a_short_budget_keeps_its_whole_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Triage's 10 s goes entirely to the provider, as before #828."""
    _real_floors(monkeypatch)
    with model_call_budget(10.0):
        share = member_timeout(last=True)
    assert share is not None
    assert share > 9.99


@pytest.mark.asyncio
async def test_a_failed_sampling_rebind_reports_its_own_error() -> None:
    """The rebound request's timeout is what the caller sees and the tracker counts."""
    bound = AsyncMock()
    bound.ainvoke.side_effect = [_rejected_temperature(), _rejected_temperature()]
    rebound = AsyncMock()
    rebound.ainvoke.side_effect = httpx.ReadTimeout("no answer")
    stripped = MagicMock()
    stripped.bind_tools.return_value = rebound
    source = MagicMock()
    source.bind_tools.return_value = bound
    source.with_config.return_value = stripped
    chain = FallbackChatModel([(source, "cloud", "p0")]).bind_tools([{"name": "t"}])

    with pytest.raises(httpx.ReadTimeout):
        await chain.ainvoke(_MESSAGES)
