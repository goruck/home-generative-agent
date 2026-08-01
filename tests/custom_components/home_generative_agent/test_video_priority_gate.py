# ruff: noqa: S101
"""
Tests for the video-priority admission extensions in core/utils.py.

Covers (test plan items):
- local_video_session: edge clears _video_idle, cloud is a no-op,
  reference counting, exception/cancellation safety
- sentinel_admission defers while a local video session is active
- chat still outranks video and Sentinel
- video does not cancel chat
- video and chat can be simultaneously active without either canceling the other
- sentinel_admission_counters tracks deferred_video/deferred_chat, resets on read
- video_is_active reflects live session state
"""

from __future__ import annotations

import asyncio

import pytest

import custom_components.home_generative_agent.core.utils as utils_mod

# ---------------------------------------------------------------------------
# Override autouse fixtures from pytest-homeassistant-custom-component
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def enable_event_loop_debug() -> None:
    """No-op override: pure-asyncio tests don't need HA's debug-mode hook."""


@pytest.fixture(autouse=True)
def verify_cleanup() -> None:
    """No-op override: all tasks explicitly awaited; no HA resources to clean up."""


# ---------------------------------------------------------------------------
# Fresh gate primitives for each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def _fresh_gate_primitives(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset all module-level gate state before each test."""
    chat_idle = asyncio.Event()
    chat_idle.set()
    video_idle = asyncio.Event()
    video_idle.set()

    monkeypatch.setattr(utils_mod, "_chat_idle", chat_idle)
    monkeypatch.setattr(utils_mod, "_chat_active_count", 0)
    monkeypatch.setattr(utils_mod, "_video_idle", video_idle)
    monkeypatch.setattr(utils_mod, "_video_active_count", 0)
    monkeypatch.setattr(utils_mod, "_sentinel_last_run", 0.0)
    monkeypatch.setattr(utils_mod, "_sentinel_first_defer", 0.0)
    monkeypatch.setattr(utils_mod, "_sentinel_defer_count", 0)
    monkeypatch.setattr(utils_mod, "_sentinel_admissions_deferred_chat", 0)
    monkeypatch.setattr(utils_mod, "_sentinel_admissions_deferred_video", 0)
    monkeypatch.setattr(utils_mod, "_sentinel_llm_tasks", set())


def _vid() -> asyncio.Event:
    return utils_mod._video_idle  # type: ignore[attr-defined]


def _chat() -> asyncio.Event:
    return utils_mod._chat_idle  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# local_video_session: edge deployment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_video_session_edge_clears_idle() -> None:
    """Edge video session clears _video_idle on entry and restores on exit."""
    assert _vid().is_set()
    async with utils_mod.local_video_session("edge"):
        assert not _vid().is_set()
    assert _vid().is_set()


@pytest.mark.asyncio
async def test_local_video_session_edge_restores_on_exception() -> None:
    """_video_idle is restored even when the body raises."""
    msg = "boom"
    with pytest.raises(RuntimeError, match=msg):
        async with utils_mod.local_video_session("edge"):
            raise RuntimeError(msg)
    assert _vid().is_set()
    assert utils_mod._video_active_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_local_video_session_edge_restores_on_cancellation() -> None:
    """_video_idle and ref-count reset when the session task is cancelled."""

    async def _run() -> None:
        async with utils_mod.local_video_session("edge"):
            await asyncio.sleep(9999)

    task = asyncio.create_task(_run())
    await asyncio.sleep(0)
    assert not _vid().is_set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _vid().is_set()
    assert utils_mod._video_active_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_concurrent_video_sessions_ref_counted() -> None:
    """_video_idle stays cleared until the last concurrent session exits."""
    observations: list[bool] = []

    async def _session() -> None:
        async with utils_mod.local_video_session("edge"):
            observations.append(_vid().is_set())
            await asyncio.sleep(0)

    await asyncio.gather(_session(), _session())
    assert all(obs is False for obs in observations)
    assert _vid().is_set()
    assert utils_mod._video_active_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_local_video_session_cloud_is_noop() -> None:
    """Cloud deployment must not touch _video_idle."""
    assert _vid().is_set()
    async with utils_mod.local_video_session("cloud"):
        assert _vid().is_set()
    assert _vid().is_set()
    assert utils_mod._video_active_count == 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# video_is_active helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_video_is_active_true_during_session() -> None:
    """video_is_active() reflects the live session state."""
    assert not utils_mod.video_is_active()
    async with utils_mod.local_video_session("edge"):
        assert utils_mod.video_is_active()
    assert not utils_mod.video_is_active()


# ---------------------------------------------------------------------------
# sentinel_admission: deferred by video
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sentinel_admission_deferred_when_video_active() -> None:
    """Sentinel is denied when video is active and timeout elapses."""
    _vid().clear()
    result = await utils_mod.sentinel_admission(
        "edge", category="triage", timeout_s=0.05
    )
    assert result is False


@pytest.mark.asyncio
async def test_sentinel_admission_increments_deferred_video_counter() -> None:
    """Video-active deferral increments the video counter, not the chat counter."""
    _vid().clear()
    await utils_mod.sentinel_admission("edge", category="triage", timeout_s=0.05)
    assert utils_mod._sentinel_admissions_deferred_video == 1  # type: ignore[attr-defined]
    assert utils_mod._sentinel_admissions_deferred_chat == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_sentinel_admission_increments_deferred_chat_counter() -> None:
    """Chat-active deferral increments the chat counter, not the video counter."""
    _chat().clear()
    await utils_mod.sentinel_admission("edge", category="triage", timeout_s=0.05)
    assert utils_mod._sentinel_admissions_deferred_chat == 1  # type: ignore[attr-defined]
    assert utils_mod._sentinel_admissions_deferred_video == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_sentinel_admission_deferred_when_both_active() -> None:
    """Sentinel is denied when both chat and video are active."""
    _chat().clear()
    _vid().clear()
    result = await utils_mod.sentinel_admission(
        "edge", category="discovery", timeout_s=0.05
    )
    assert result is False


@pytest.mark.asyncio
async def test_sentinel_admission_admitted_when_both_idle() -> None:
    """Sentinel is admitted immediately when chat and video are both idle."""
    result = await utils_mod.sentinel_admission(
        "edge", category="triage", timeout_s=1.0
    )
    assert result is True


@pytest.mark.asyncio
async def test_sentinel_admission_admitted_after_video_ends() -> None:
    """Sentinel is admitted once video becomes idle."""
    _vid().clear()
    admitted: list[bool] = []

    async def _admit() -> None:
        result = await utils_mod.sentinel_admission(
            "edge", category="triage", timeout_s=2.0
        )
        admitted.append(result)

    task = asyncio.create_task(_admit())
    await asyncio.sleep(0.01)
    assert not admitted

    _vid().set()
    await asyncio.wait_for(task, timeout=2.0)
    assert admitted == [True]


@pytest.mark.asyncio
async def test_sentinel_admission_requires_both_idle() -> None:
    """Sentinel waits for BOTH chat and video before admitting."""
    _chat().clear()
    _vid().clear()
    admitted: list[bool] = []

    async def _admit() -> None:
        result = await utils_mod.sentinel_admission(
            "edge", category="triage", timeout_s=2.0
        )
        admitted.append(result)

    task = asyncio.create_task(_admit())
    await asyncio.sleep(0.01)

    # Release video but keep chat busy — still blocked
    _vid().set()
    await asyncio.sleep(0.01)
    assert not admitted

    # Release chat — now both idle, Sentinel is admitted
    _chat().set()
    await asyncio.wait_for(task, timeout=2.0)
    assert admitted == [True]


# ---------------------------------------------------------------------------
# Chat outranks video: video does not cancel chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_video_does_not_cancel_chat_session() -> None:
    """Starting a video session must not interrupt an active chat session."""
    chat_completed = asyncio.Event()

    async def _chat() -> None:
        async with utils_mod.local_chat_session("edge"):
            await asyncio.sleep(0.05)
        chat_completed.set()

    async def _video() -> None:
        await asyncio.sleep(0.01)
        async with utils_mod.local_video_session("edge"):
            await asyncio.sleep(0)

    await asyncio.gather(_chat(), _video())
    assert chat_completed.is_set()


@pytest.mark.asyncio
async def test_chat_and_video_simultaneously_active() -> None:
    """Chat and video can overlap without canceling each other."""
    video_saw_chat_active: list[bool] = []
    chat_saw_video_active: list[bool] = []

    async def _chat_work() -> None:
        async with utils_mod.local_chat_session("edge"):
            chat_saw_video_active.append(not _vid().is_set())
            await asyncio.sleep(0.05)

    async def _video_work() -> None:
        await asyncio.sleep(0.01)  # let chat start first
        async with utils_mod.local_video_session("edge"):
            video_saw_chat_active.append(not _chat().is_set())
            await asyncio.sleep(0)

    await asyncio.gather(_chat_work(), _video_work())

    # Both ran successfully
    assert len(chat_saw_video_active) == 1
    assert len(video_saw_chat_active) == 1

    # Gates restored after both finish
    assert _chat().is_set()
    assert _vid().is_set()


# ---------------------------------------------------------------------------
# sentinel_admission_counters: read-and-reset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sentinel_admission_counters_read_and_reset() -> None:
    """sentinel_admission_counters() returns current values then resets them."""
    _vid().clear()
    await utils_mod.sentinel_admission("edge", category="triage", timeout_s=0.05)
    _vid().set()
    _chat().clear()
    await utils_mod.sentinel_admission("edge", category="triage", timeout_s=0.05)
    _chat().set()

    counters = utils_mod.sentinel_admission_counters()
    assert counters["deferred_video"] == 1
    assert counters["deferred_chat"] == 1

    # Second read returns zeros
    counters2 = utils_mod.sentinel_admission_counters()
    assert counters2["deferred_video"] == 0
    assert counters2["deferred_chat"] == 0


@pytest.mark.asyncio
async def test_sentinel_admission_counters_not_incremented_on_success() -> None:
    """Successful Sentinel admission must not increment deferral counters."""
    await utils_mod.sentinel_admission("edge", category="triage", timeout_s=1.0)
    counters = utils_mod.sentinel_admission_counters()
    assert counters["deferred_video"] == 0
    assert counters["deferred_chat"] == 0
