# ruff: noqa: S101
"""
Tests for deployment-aware admission control in core/utils.py.

Covers:
- is_edge_deployment: edge/cloud/None classification
- local_chat_session: edge clears _chat_idle, cloud is a no-op,
  reference counting, exception/cancellation safety
- sentinel_admission: cloud always admitted, edge defers when chat active,
  edge admitted when idle, starvation tracking and WARNING threshold
- generate_embeddings: runs freely during active chat (no gate)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from langchain_ollama import OllamaEmbeddings

import custom_components.home_generative_agent.core.utils as utils_mod
from custom_components.home_generative_agent.core.provider_stall import (
    StallProbeDeferredError,
)

# ---------------------------------------------------------------------------
# Override pytest-homeassistant-custom-component autouse fixtures.
# These tests use pytest-asyncio's function-scoped asyncio.Runner (no `hass`).
# Shadowing the plugin fixtures avoids "Event loop is closed" errors in teardown.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def enable_event_loop_debug() -> None:
    """No-op override: pure-asyncio tests don't need HA's debug-mode hook."""


@pytest.fixture(autouse=True)
def verify_cleanup() -> None:
    """No-op override: all tasks are explicitly awaited; no HA resources to clean up."""


# ---------------------------------------------------------------------------
# Fixture: fresh gate primitives per test.
# Module-level asyncio.Event objects bind to the first event loop that touches
# them.  pytest-asyncio gives each async test its own loop, so we replace the
# module attributes with brand-new objects via monkeypatch.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def _fresh_gate_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace module-level gate primitives with fresh instances for each test."""
    new_idle = asyncio.Event()
    new_idle.set()

    new_video_idle = asyncio.Event()
    new_video_idle.set()

    monkeypatch.setattr(utils_mod, "_chat_idle", new_idle)
    monkeypatch.setattr(utils_mod, "_chat_active_count", 0)
    monkeypatch.setattr(utils_mod, "_chat_last_end", 0.0)
    monkeypatch.setattr(utils_mod, "_video_idle", new_video_idle)
    monkeypatch.setattr(utils_mod, "_video_active_count", 0)
    monkeypatch.setattr(utils_mod, "_sentinel_last_run", 0.0)
    monkeypatch.setattr(utils_mod, "_sentinel_first_defer", 0.0)
    monkeypatch.setattr(utils_mod, "_sentinel_last_defer", 0.0)
    monkeypatch.setattr(utils_mod, "_sentinel_defer_count", 0)
    monkeypatch.setattr(utils_mod, "_sentinel_llm_tasks", set())


def _idle() -> asyncio.Event:
    return utils_mod._chat_idle  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# is_edge_deployment
# ---------------------------------------------------------------------------


def test_is_edge_deployment_edge() -> None:
    assert utils_mod.is_edge_deployment("edge") is True


def test_is_edge_deployment_cloud() -> None:
    assert utils_mod.is_edge_deployment("cloud") is False


def test_is_edge_deployment_none() -> None:
    assert utils_mod.is_edge_deployment(None) is False


def test_is_edge_deployment_unknown_string() -> None:
    assert utils_mod.is_edge_deployment("other") is False


# ---------------------------------------------------------------------------
# local_chat_session: edge deployment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_chat_session_edge_clears_idle() -> None:
    """Edge chat session must clear _chat_idle on entry."""
    assert _idle().is_set()
    async with utils_mod.local_chat_session("edge"):
        assert not _idle().is_set()
    assert _idle().is_set()


@pytest.mark.asyncio
async def test_local_chat_session_edge_restores_on_exception() -> None:
    """_chat_idle must be restored even when the body raises."""
    err_msg = "boom"
    with pytest.raises(RuntimeError, match=err_msg):
        async with utils_mod.local_chat_session("edge"):
            raise RuntimeError(err_msg)
    assert _idle().is_set()
    assert utils_mod._chat_active_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_local_chat_session_edge_restores_on_cancellation() -> None:
    """_chat_idle and counter must reset when the session task is cancelled."""

    async def _run() -> None:
        async with utils_mod.local_chat_session("edge"):
            await asyncio.sleep(9999)

    task = asyncio.create_task(_run())
    await asyncio.sleep(0)
    assert not _idle().is_set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _idle().is_set()
    assert utils_mod._chat_active_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_concurrent_edge_sessions_keep_idle_cleared() -> None:
    """Ref count: _chat_idle stays cleared until the last session exits."""
    results: list[bool] = []

    async def _session() -> None:
        async with utils_mod.local_chat_session("edge"):
            results.append(_idle().is_set())
            await asyncio.sleep(0)
            results.append(_idle().is_set())

    await asyncio.gather(_session(), _session())
    assert all(r is False for r in results)
    assert _idle().is_set()


# ---------------------------------------------------------------------------
# local_chat_session: cloud deployment (no-op)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_chat_session_cloud_is_noop() -> None:
    """Cloud deployment must not touch _chat_idle."""
    assert _idle().is_set()
    async with utils_mod.local_chat_session("cloud"):
        assert _idle().is_set()
    assert _idle().is_set()
    assert utils_mod._chat_active_count == 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# sentinel_admission: cloud always admitted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sentinel_admission_cloud_always_admitted() -> None:
    """Cloud deployment is always admitted regardless of chat state."""
    _idle().clear()  # simulate active edge chat
    result = await utils_mod.sentinel_admission(
        "cloud", category="triage", timeout_s=0.1
    )
    assert result is True


@pytest.mark.asyncio
async def test_sentinel_admission_cloud_resets_defer_count() -> None:
    """Cloud admission resets starvation counters."""
    utils_mod._sentinel_defer_count = 5  # type: ignore[attr-defined]
    await utils_mod.sentinel_admission("cloud", category="triage", timeout_s=1.0)
    assert utils_mod._sentinel_defer_count == 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# sentinel_admission: edge — idle case
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sentinel_admission_edge_admitted_when_idle() -> None:
    """Edge Sentinel is admitted immediately when no chat is active."""
    result = await utils_mod.sentinel_admission(
        "edge", category="triage", timeout_s=1.0
    )
    assert result is True


@pytest.mark.asyncio
async def test_sentinel_admission_edge_updates_last_run_on_success() -> None:
    """Successful admission updates _sentinel_last_run."""
    before = time.monotonic()
    await utils_mod.sentinel_admission("edge", category="triage", timeout_s=1.0)
    assert utils_mod._sentinel_last_run >= before  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_sentinel_admission_edge_resets_defer_count_on_success() -> None:
    """Successful admission resets the deferral counter."""
    utils_mod._sentinel_defer_count = 3  # type: ignore[attr-defined]
    await utils_mod.sentinel_admission("edge", category="triage", timeout_s=1.0)
    assert utils_mod._sentinel_defer_count == 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# sentinel_admission: edge — deferred (chat active, timeout)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sentinel_admission_edge_deferred_when_chat_active() -> None:
    """Edge Sentinel must be denied when chat is active and timeout elapses."""
    _idle().clear()
    result = await utils_mod.sentinel_admission(
        "edge", category="triage", timeout_s=0.05
    )
    assert result is False


@pytest.mark.asyncio
async def test_sentinel_admission_edge_increments_defer_count() -> None:
    """Deferral must increment _sentinel_defer_count."""
    _idle().clear()
    await utils_mod.sentinel_admission("edge", category="discovery", timeout_s=0.05)
    assert utils_mod._sentinel_defer_count == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_sentinel_admission_edge_starvation_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A WARNING is emitted when the starvation threshold is exceeded."""
    _idle().clear()
    # Last run was far in the past.
    utils_mod._sentinel_last_run = time.monotonic() - 400.0  # type: ignore[attr-defined]
    with caplog.at_level(
        logging.WARNING, logger="custom_components.home_generative_agent.core.utils"
    ):
        await utils_mod.sentinel_admission("edge", category="triage", timeout_s=0.05)
    assert any("deferred" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_sentinel_admission_edge_marks_health_degraded_without_prior_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sustained deferrals degrade health even if no Sentinel LLM run succeeded yet."""
    _idle().clear()
    health_stats: dict[str, object] = {}
    # time.monotonic() is seconds since boot on a fresh CI runner and can be
    # below any fixed offset; a negative "first deferral" reads as "none yet"
    # and starvation resets to zero. Lower the threshold instead of reaching
    # hundreds of seconds into the past.
    monkeypatch.setattr(utils_mod, "_SENTINEL_STARVATION_WARN_S", 1.0)
    monkeypatch.setattr(utils_mod, "_sentinel_first_defer", time.monotonic() - 5.0)
    # Sustained: the last deferral was recent, so the streak is still alive.
    monkeypatch.setattr(utils_mod, "_sentinel_last_defer", time.monotonic())

    result = await utils_mod.sentinel_admission(
        "edge", category="triage", timeout_s=0.05, health_stats=health_stats
    )

    assert result is False
    assert health_stats["sentinel_admission_degraded"] is True
    assert health_stats["sentinel_admission_degraded_category"] == "triage"
    assert health_stats["sentinel_admission_consecutive_deferrals"] == 1


@pytest.mark.asyncio
async def test_sentinel_admission_edge_defers_at_once_while_chat_is_active() -> None:
    """A chat turn can only end into the grace, so there is nothing to wait for (#866)."""
    _idle().clear()
    start = time.monotonic()
    result = await utils_mod.sentinel_admission(
        "edge", category="triage", timeout_s=2.0
    )
    # gstack-timing-ok: the alternative is the full 2 s wait; 1 s is a generous bound
    assert time.monotonic() - start < 1.0
    assert result is False


@pytest.mark.asyncio
async def test_local_chat_session_cancels_inflight_sentinel_llm() -> None:
    """Foreground edge chat must interrupt an already-running Sentinel LLM call."""
    sentinel_started = asyncio.Event()
    sentinel_deferred: list[bool] = []

    async def _slow_llm() -> str:
        sentinel_started.set()
        await asyncio.sleep(9999)
        return "done"

    async def _run_sentinel() -> None:
        try:
            await utils_mod.run_sentinel_llm_call(
                _slow_llm,
                deployment="edge",
                category="triage",
                admission_timeout_s=0.1,
                call_timeout_s=60.0,
            )
        except utils_mod.SentinelLLMDeferredError:
            sentinel_deferred.append(True)

    sentinel_task = asyncio.create_task(_run_sentinel())
    await sentinel_started.wait()

    async with utils_mod.local_chat_session("edge"):
        # Wait for the outer task to fully handle SentinelLLMDeferredError before
        # asserting. _cancel_active_sentinel_llm_tasks awaits the inner future but
        # the outer coroutine needs one more event-loop turn to run its except branch.
        await asyncio.wait_for(sentinel_task, timeout=1.0)
        assert sentinel_deferred == [True]


# ---------------------------------------------------------------------------
# generate_embeddings: no gate — runs freely even during active chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_embeddings_runs_during_active_chat() -> None:
    """Embeddings must complete immediately even while edge chat is active."""
    mock_emb = MagicMock()
    mock_emb.aembed_documents = AsyncMock(return_value=[[0.1, 0.2]])
    mock_emb.__class__ = OllamaEmbeddings  # type: ignore[assignment]

    async with utils_mod.local_chat_session("edge"):
        result = await asyncio.wait_for(
            utils_mod.generate_embeddings(mock_emb, ["hello"]),
            timeout=1.0,
        )
    assert result == [[0.1, 0.2]]


@pytest.mark.asyncio
async def test_generate_embeddings_ollama_proceeds_when_idle() -> None:
    """Ollama generate_embeddings completes immediately when no chat is active."""
    mock_emb = MagicMock()
    mock_emb.aembed_documents = AsyncMock(return_value=[[0.1, 0.2]])
    mock_emb.__class__ = OllamaEmbeddings  # type: ignore[assignment]

    result = await utils_mod.generate_embeddings(mock_emb, ["hello"])
    assert result == [[0.1, 0.2]]


@pytest.mark.asyncio
async def test_generate_embeddings_empty_texts_returns_empty() -> None:
    """generate_embeddings with empty input returns [] without hitting the model."""
    mock_emb = MagicMock()
    mock_emb.aembed_documents = AsyncMock(return_value=[])

    result = await utils_mod.generate_embeddings(mock_emb, [])
    mock_emb.aembed_documents.assert_not_called()
    assert result == []


# ---------------------------------------------------------------------------
# run_sentinel_llm_call: call_timeout_s exceeded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_sentinel_llm_call_timeout_raises_timeout_error() -> None:
    """call_factory that exceeds call_timeout_s must raise TimeoutError."""

    async def _slow() -> str:
        await asyncio.sleep(9999)
        return "done"

    with pytest.raises(TimeoutError):
        await utils_mod.run_sentinel_llm_call(
            _slow,
            deployment="cloud",  # always admitted — bypasses the chat gate
            category="triage",
            admission_timeout_s=0.1,
            call_timeout_s=0.05,
        )


@pytest.mark.asyncio
async def test_run_sentinel_model_call_uses_sync_invoke_off_event_loop() -> None:
    """Models with sync invoke should run in an executor, not on the event loop."""
    loop_thread_id = threading.get_ident()

    class SyncModel:
        def __init__(self) -> None:
            self.invoke_thread_id: int | None = None
            self.ainvoke_called = False

        def invoke(self, _messages: list[str]) -> str:
            self.invoke_thread_id = threading.get_ident()
            return "ok"

        async def ainvoke(self, _messages: list[str]) -> str:
            self.ainvoke_called = True
            return "async"

    model = SyncModel()

    result = await utils_mod.run_sentinel_model_call(
        model,
        ["message"],
        deployment="cloud",
        category="triage",
        admission_timeout_s=0.1,
        call_timeout_s=1.0,
    )

    assert result == "ok"
    assert model.invoke_thread_id is not None
    assert model.invoke_thread_id != loop_thread_id
    assert model.ainvoke_called is False


def _unsupported_temperature_error() -> Exception:
    """Build the OpenAI 400 raised when a model rejects the temperature param."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(400, request=request)
    body = {
        "message": "Unsupported value: 'temperature' does not support 0.2.",
        "type": "invalid_request_error",
        "param": "temperature",
        "code": "unsupported_value",
    }
    return openai.BadRequestError("Error code: 400", response=response, body=body)  # pyright: ignore[reportArgumentType]  # httpx vs openai 3's httpx2


@pytest.mark.asyncio
async def test_run_sentinel_model_call_sync_leg_drops_unsupported_temperature() -> None:
    """The executor (sync invoke) leg retries once without temperature (issue #502)."""

    class SyncModel:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def invoke(self, _messages: list[str], config: object = None) -> str:
            self.calls.append(config)
            if len(self.calls) == 1:
                raise _unsupported_temperature_error()
            return "ok"

    model = SyncModel()

    result = await utils_mod.run_sentinel_model_call(
        model,
        ["message"],
        deployment="cloud",
        category="triage",
        admission_timeout_s=0.1,
        call_timeout_s=5.0,
    )

    assert result == "ok"
    assert len(model.calls) == 2
    retry_config = model.calls[1]
    assert retry_config["configurable"]["temperature"] is None  # type: ignore[index]


@pytest.mark.asyncio
async def test_run_sentinel_model_call_async_leg_drops_unsupported_temperature() -> (
    None
):
    """The async (ainvoke-only) leg retries once without temperature (issue #502)."""

    class AsyncOnlyModel:
        def __init__(self) -> None:
            self.calls: list[object] = []

        async def ainvoke(self, _messages: list[str], config: object = None) -> str:
            self.calls.append(config)
            if len(self.calls) == 1:
                raise _unsupported_temperature_error()
            return "ok"

    model = AsyncOnlyModel()

    result = await utils_mod.run_sentinel_model_call(
        model,
        ["message"],
        deployment="cloud",
        category="triage",
        admission_timeout_s=0.1,
        call_timeout_s=5.0,
    )

    assert result == "ok"
    assert len(model.calls) == 2
    retry_config = model.calls[1]
    assert retry_config["configurable"]["temperature"] is None  # type: ignore[index]


# ---------------------------------------------------------------------------
# Grace period after a chat turn, and tracked calls without admission (#866)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sentinel_is_deferred_within_the_grace_after_a_chat_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A call just after a chat turn is deferred at once, then admitted."""
    monkeypatch.setattr(utils_mod, "SENTINEL_CHAT_GRACE_S", 0.3)
    utils_mod.sentinel_admission_counters()  # start the deferral counters at 0
    async with utils_mod.local_chat_session("edge"):
        pass
    health: dict[str, Any] = {}

    assert utils_mod.chat_grace_remaining_s() > 0.0
    start = time.monotonic()
    admitted = await utils_mod.sentinel_admission(
        "edge", category="triage", timeout_s=2.0, health_stats=health
    )
    assert admitted is False
    # gstack-timing-ok: the alternative is the full 2 s wait; 1 s is a generous bound
    assert time.monotonic() - start < 1.0
    assert health["sentinel_admission_consecutive_deferrals"] == 1
    assert utils_mod.sentinel_admission_counters()["deferred_chat"] == 1

    # Move the last chat end past the grace instead of sleeping it out.
    utils_mod._chat_last_end = time.monotonic() - 1.0
    assert utils_mod.chat_grace_remaining_s() == 0.0
    assert (
        await utils_mod.sentinel_admission(
            "edge", category="triage", timeout_s=0.1, health_stats=health
        )
        is True
    )
    assert health["sentinel_admission_consecutive_deferrals"] == 0


@pytest.mark.asyncio
async def test_a_cloud_chat_turn_leaves_no_grace() -> None:
    """Cloud chat never touches the local gates."""
    async with utils_mod.local_chat_session("cloud"):
        pass
    assert utils_mod.chat_grace_remaining_s() == 0.0


@pytest.mark.asyncio
async def test_the_grace_yields_after_a_deferral_streak_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Chat turns every few seconds cannot hold Sentinel off past the degraded threshold."""
    monkeypatch.setattr(utils_mod, "_SENTINEL_STARVATION_WARN_S", 0.2)
    async with utils_mod.local_chat_session("edge"):
        pass
    assert utils_mod.chat_grace_remaining_s() > 0.0
    health: dict[str, Any] = {}

    # A quiet house: the last admitted call is long ago, but nothing was
    # deferred since. That is not starvation; the grace holds.
    monkeypatch.setattr(utils_mod, "_sentinel_last_run", time.monotonic() - 5.0)
    monkeypatch.setattr(utils_mod, "_sentinel_first_defer", 0.0)
    assert (
        await utils_mod.sentinel_admission(
            "edge", category="triage", timeout_s=0.05, health_stats=health
        )
        is False
    )
    # That deferral started the streak; a young streak still holds.
    assert utils_mod._sentinel_first_defer > 0.0
    assert (
        await utils_mod.sentinel_admission(
            "edge", category="triage", timeout_s=0.05, health_stats=health
        )
        is False
    )
    # A streak past the threshold yields, whatever the last admitted call,
    # and says so at WARNING: the next turn can cancel this call.
    monkeypatch.setattr(utils_mod, "_sentinel_first_defer", time.monotonic() - 1.0)
    monkeypatch.setattr(utils_mod, "_sentinel_last_defer", time.monotonic())
    monkeypatch.setattr(utils_mod, "_sentinel_last_run", time.monotonic())
    with caplog.at_level(logging.WARNING, logger=utils_mod.LOGGER.name):
        assert (
            await utils_mod.sentinel_admission(
                "edge", category="triage", timeout_s=0.05, health_stats=health
            )
            is True
        )
    assert [r for r in caplog.records if "admitted inside the chat grace" in r.message]
    # The admission ended the streak: the grace holds again.
    assert utils_mod._sentinel_first_defer == 0.0
    assert (
        await utils_mod.sentinel_admission(
            "edge", category="triage", timeout_s=0.05, health_stats=health
        )
        is False
    )
    # An active chat turn defers, grace or streak.
    monkeypatch.setattr(utils_mod, "_sentinel_first_defer", time.monotonic() - 1.0)
    _idle().clear()
    assert (
        await utils_mod.sentinel_admission("edge", category="triage", timeout_s=0.05)
        is False
    )


@pytest.mark.asyncio
async def test_sentinel_admission_defers_when_chat_starts_during_the_video_wait() -> (
    None
):
    """A chat turn arriving while Sentinel waits for video still wins."""
    utils_mod._video_idle.clear()
    utils_mod.sentinel_admission_counters()
    task = asyncio.create_task(
        utils_mod.sentinel_admission("edge", category="triage", timeout_s=2.0)
    )
    await asyncio.sleep(0.01)
    _idle().clear()
    utils_mod._video_idle.set()
    assert await asyncio.wait_for(task, timeout=1.0) is False
    assert utils_mod.sentinel_admission_counters()["deferred_chat"] == 1


@pytest.mark.asyncio
async def test_a_chat_exit_after_a_reload_reset_does_not_close_the_gate() -> None:
    """A reload zeroes the count mid-turn; that turn's exit must not go below zero."""
    async with utils_mod.local_chat_session("edge"):
        utils_mod._chat_active_count = 0  # what the setup reset block does
    assert utils_mod._chat_active_count == 0
    async with utils_mod.local_chat_session("edge"):
        assert not _idle().is_set()
    assert _idle().is_set()


@pytest.mark.asyncio
async def test_uncontended_bypasses_every_local_gate() -> None:
    """model_provider_uncontended: chat marks nothing, Sentinel is always admitted."""
    utils_mod.set_model_provider_uncontended(True)
    try:
        async with utils_mod.local_chat_session("edge"):
            assert _idle().is_set()
            assert not utils_mod.foreground_is_active()
            assert (
                await utils_mod.sentinel_admission(
                    "edge", category="triage", timeout_s=0.05
                )
                is True
            )
        assert utils_mod.chat_grace_remaining_s() == 0.0
    finally:
        utils_mod.set_model_provider_uncontended(False)


# ---------------------------------------------------------------------------
# The stall-probe gate (#866)
# ---------------------------------------------------------------------------


def _deployments(**by_id: str) -> Any:
    return by_id.get


@pytest.mark.asyncio
async def test_probe_runner_sends_a_cloud_providers_probe_directly() -> None:
    runner = utils_mod.make_gated_probe_runner(_deployments(p0="cloud"))
    _idle().clear()
    probe = AsyncMock(return_value="ok")
    assert await runner("p0", probe) == "ok"
    probe.assert_awaited_once()


@pytest.mark.asyncio
async def test_probe_runner_skips_the_round_while_chat_is_active() -> None:
    runner = utils_mod.make_gated_probe_runner(_deployments(p0="edge"))
    _idle().clear()
    probe = AsyncMock(return_value="ok")
    with pytest.raises(StallProbeDeferredError) as excinfo:
        await runner("p0", probe)
    probe.assert_not_called()
    assert "holds the GPU" in str(excinfo.value)


@pytest.mark.asyncio
async def test_probe_runner_ignores_the_chat_grace() -> None:
    """A stalled chat provider's fast-failing turns must not hold its probe off."""
    runner = utils_mod.make_gated_probe_runner(_deployments(p0="edge"))
    async with utils_mod.local_chat_session("edge"):
        pass
    assert utils_mod.chat_grace_remaining_s() > 0.0  # inside the grace
    probe = AsyncMock(return_value="ok")
    assert await runner("p0", probe) == "ok"


@pytest.mark.asyncio
async def test_probe_runner_bypasses_the_gate_when_uncontended() -> None:
    runner = utils_mod.make_gated_probe_runner(_deployments(p0="edge"))
    utils_mod.set_model_provider_uncontended(True)
    try:
        _idle().clear()
        probe = AsyncMock(return_value="ok")
        assert await runner("p0", probe) == "ok"
    finally:
        utils_mod.set_model_provider_uncontended(False)


@pytest.mark.asyncio
async def test_probe_runner_lets_a_sent_probe_finish() -> None:
    """A probe that started is not cancelled by a chat turn (it could re-wedge the runner)."""
    runner = utils_mod.make_gated_probe_runner(_deployments(p0="edge"))
    started = asyncio.Event()
    release = asyncio.Event()

    async def _probe() -> str:
        started.set()
        await release.wait()
        return "answered"

    task = asyncio.ensure_future(runner("p0", _probe))
    await started.wait()
    async with utils_mod.local_chat_session("edge"):
        pass  # cancels tracked Sentinel tasks; the probe is not one
    release.set()
    assert await task == "answered"


@pytest.mark.asyncio
async def test_a_stale_deferral_does_not_count_as_a_streak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One deferral this morning plus a quiet afternoon is not 300 s of pressure."""
    monkeypatch.setattr(utils_mod, "_SENTINEL_STARVATION_WARN_S", 0.2)
    async with utils_mod.local_chat_session("edge"):
        pass
    # A streak that began long ago but whose last deferral is older than a
    # grace is over: the grace holds, and a new streak starts now.
    monkeypatch.setattr(utils_mod, "_sentinel_first_defer", time.monotonic() - 100.0)
    monkeypatch.setattr(utils_mod, "_sentinel_last_defer", time.monotonic() - 50.0)
    assert (
        await utils_mod.sentinel_admission("edge", category="triage", timeout_s=0.05)
        is False
    )
    assert time.monotonic() - utils_mod._sentinel_first_defer < 1.0


@pytest.mark.asyncio
async def test_a_chat_turn_from_before_a_reload_does_not_open_the_gate() -> None:
    """A turn that entered before the reset must not decrement the count the reset made."""
    async with utils_mod.local_chat_session("edge"):
        # The setup reset block of a reload, while this turn is still inside.
        utils_mod._chat_active_count = 0
        utils_mod._chat_idle.set()
        utils_mod._gate_generation += 1
        async with utils_mod.local_chat_session("edge"):
            assert utils_mod._chat_active_count == 1
            assert not _idle().is_set()
            # Inner turn is the newer one; the outer one exits first here:
        assert utils_mod._chat_active_count == 0
        assert _idle().is_set()
    # The old turn's exit changed nothing.
    assert utils_mod._chat_active_count == 0
    assert _idle().is_set()


@pytest.mark.asyncio
async def test_an_old_turn_exiting_under_a_new_one_keeps_the_gate_closed() -> None:
    old_turn_entered = asyncio.Event()
    release_old = asyncio.Event()

    async def _old_turn() -> None:
        async with utils_mod.local_chat_session("edge"):
            old_turn_entered.set()
            await release_old.wait()

    task = asyncio.create_task(_old_turn())
    await old_turn_entered.wait()
    utils_mod._chat_active_count = 0
    utils_mod._chat_idle.set()
    utils_mod._gate_generation += 1  # the reload
    async with utils_mod.local_chat_session("edge"):  # a turn after the reload
        release_old.set()
        await task
        assert utils_mod._chat_active_count == 1
        assert not _idle().is_set()
    assert _idle().is_set()


def test_sentinel_calls_are_gated_on_shared_local_models_only() -> None:
    assert utils_mod.sentinel_calls_gated("edge")
    assert not utils_mod.sentinel_calls_gated("cloud")
    assert not utils_mod.sentinel_calls_gated(None)
    utils_mod.set_model_provider_uncontended(True)
    try:
        assert not utils_mod.sentinel_calls_gated("edge")
    finally:
        utils_mod.set_model_provider_uncontended(False)


def test_sentinel_call_gap_remaining() -> None:
    utils_mod._sentinel_last_call_end = 0.0
    assert utils_mod.sentinel_call_gap_remaining_s(1.0) == 0.0
    utils_mod._sentinel_last_call_end = time.monotonic()
    assert utils_mod.sentinel_call_gap_remaining_s(0.0) == 0.0
    assert 0.9 < utils_mod.sentinel_call_gap_remaining_s(1.0) <= 1.0
    utils_mod._sentinel_last_call_end = time.monotonic() - 5.0
    assert utils_mod.sentinel_call_gap_remaining_s(1.0) == 0.0


@pytest.mark.asyncio
async def test_wait_sentinel_call_gap_only_on_gated_deployments() -> None:
    utils_mod._sentinel_last_call_end = time.monotonic()
    start = time.monotonic()
    await utils_mod.wait_sentinel_call_gap(0.5, "cloud")
    # gstack-timing-ok: the alternative is a 0.5 s wait; 0.3 s is a generous bound
    assert time.monotonic() - start < 0.3
    start = time.monotonic()
    await utils_mod.wait_sentinel_call_gap(0.2, "edge")
    # gstack-timing-ok: lower bound only
    assert time.monotonic() - start >= 0.19
