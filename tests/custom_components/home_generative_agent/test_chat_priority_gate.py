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
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_ollama import OllamaEmbeddings

import custom_components.home_generative_agent.core.utils as utils_mod

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

    monkeypatch.setattr(utils_mod, "_chat_idle", new_idle)
    monkeypatch.setattr(utils_mod, "_chat_active_count", 0)
    monkeypatch.setattr(utils_mod, "_sentinel_last_run", 0.0)
    monkeypatch.setattr(utils_mod, "_sentinel_first_defer", 0.0)
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
    now = time.monotonic()
    monkeypatch.setattr(utils_mod, "_sentinel_first_defer", now - 400.0)

    result = await utils_mod.sentinel_admission(
        "edge", category="triage", timeout_s=0.05, health_stats=health_stats
    )

    assert result is False
    assert health_stats["sentinel_admission_degraded"] is True
    assert health_stats["sentinel_admission_degraded_category"] == "triage"
    assert health_stats["sentinel_admission_consecutive_deferrals"] == 1


@pytest.mark.asyncio
async def test_sentinel_admission_edge_admitted_after_chat_ends() -> None:
    """Edge Sentinel must be admitted once chat becomes idle."""
    _idle().clear()
    admitted: list[bool] = []

    async def _admit() -> None:
        result = await utils_mod.sentinel_admission(
            "edge", category="triage", timeout_s=2.0
        )
        admitted.append(result)

    task = asyncio.create_task(_admit())
    await asyncio.sleep(0.01)
    assert not admitted

    _idle().set()
    await asyncio.wait_for(task, timeout=2.0)
    assert admitted == [True]


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
