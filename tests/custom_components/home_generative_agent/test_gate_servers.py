# ruff: noqa: S101
"""A Sentinel model on servers of its own is not held back by chat (#877)."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

import custom_components.home_generative_agent.core.utils as utils_mod
from custom_components.home_generative_agent.core.provider_stall import (
    StallProbeDeferredError,
)
from custom_components.home_generative_agent.core.utils import (
    SentinelLLMDeferredError,
    gate_server_for_url,
    local_chat_session,
    local_video_session,
    make_gated_probe_runner,
    run_sentinel_llm_call,
    run_sentinel_llm_call_in_executor,
    sentinel_admission,
    sentinel_calls_gated,
    servers_isolated,
    set_sentinel_isolated,
)

# ---------------------------------------------------------------------------
# Pure-asyncio tests: shadow the pytest-homeassistant autouse fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def enable_event_loop_debug() -> None:
    """No-op override: pure-asyncio tests don't need HA's debug-mode hook."""


@pytest.fixture(autouse=True)
def verify_cleanup() -> None:
    """No-op override: all tasks are explicitly awaited."""


@pytest.fixture(autouse=True)
async def _fresh_gate_primitives(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test fresh gate state bound to its own event loop."""
    chat_idle = asyncio.Event()
    chat_idle.set()
    video_idle = asyncio.Event()
    video_idle.set()
    monkeypatch.setattr(utils_mod, "_chat_idle", chat_idle)
    monkeypatch.setattr(utils_mod, "_chat_active_count", 0)
    monkeypatch.setattr(utils_mod, "_chat_last_end", 0.0)
    monkeypatch.setattr(utils_mod, "_video_idle", video_idle)
    monkeypatch.setattr(utils_mod, "_video_active_count", 0)
    monkeypatch.setattr(utils_mod, "_sentinel_llm_tasks", set())
    monkeypatch.setattr(utils_mod, "_sentinel_last_run", 0.0)
    monkeypatch.setattr(utils_mod, "_sentinel_first_defer", 0.0)
    monkeypatch.setattr(utils_mod, "_sentinel_last_defer", 0.0)
    monkeypatch.setattr(utils_mod, "_sentinel_defer_count", 0)
    monkeypatch.setattr(utils_mod, "_sentinel_isolated", False)
    monkeypatch.setattr(utils_mod, "_uncontended", False)


async def _admitted() -> bool:
    return await sentinel_admission("edge", category="triage", timeout_s=0.05)


# ---------------------------------------------------------------------------
# gate_server_for_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://GPU2.lan:11434", "gpu2.lan"),
        ("http://gpu2.lan:11434/", "gpu2.lan"),
        ("https://gpu2.lan/v1", "gpu2.lan"),
        ("gpu2.lan:11434", "gpu2.lan"),
        ("http://gpu2.lan.:11434", "gpu2.lan"),
        ("http://192.168.1.121:11434", "192.168.1.121"),
        # IP literals in canonical form.
        ("http://[FE80:0:0:0:0:0:0:1]:11434", "fe80::1"),
        # Every loopback or unspecified spelling is this machine.
        ("http://localhost:11434", "localhost"),
        ("http://127.0.0.1:11435", "localhost"),
        ("http://127.5.6.7", "localhost"),
        ("http://[::1]:11434", "localhost"),
        ("http://[0:0:0:0:0:0:0:1]:11434", "localhost"),
        ("http://[::ffff:127.0.0.1]:11434", "localhost"),
        ("http://0.0.0.0:11434", "localhost"),
        # A DNS name that merely starts with 127 is a name.
        ("http://127.lan:11434", "127.lan"),
        # No host: a server not known.
        ("", None),
        ("   ", None),
        (None, None),
        (11434, None),
        ("http://", None),
    ],
)
def test_gate_server_for_url(url: object, expected: str | None) -> None:
    assert gate_server_for_url(url) == expected


def test_two_ports_on_one_host_are_one_server() -> None:
    """Two Ollama instances on one machine are taken to share its GPU."""
    assert gate_server_for_url("http://box:11434") == "box"
    assert gate_server_for_url("http://box:11435") == "box"


# ---------------------------------------------------------------------------
# servers_isolated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sentinel", "foreground", "expected"),
    [
        ({"gpu2"}, {"gpu1"}, True),
        ({"gpu2", "gpu3"}, {"gpu1"}, True),
        ({"gpu2"}, set(), True),  # every foreground model is in the cloud
        ({"gpu1"}, {"gpu1"}, False),
        ({"gpu2", "gpu1"}, {"gpu1"}, False),  # its fallback is on chat's server
        ({"gpu2"}, {"gpu1", "gpu2"}, False),  # a camera or chat fallback on gpu2
        ({None}, {"gpu1"}, False),
        ({"gpu2", None}, {"gpu1"}, False),
        ({"gpu2"}, {None}, False),
        ({"gpu2"}, {"gpu1", None}, False),
        (set(), {"gpu1"}, False),  # no local server known for Sentinel
    ],
)
def test_servers_isolated(
    *, sentinel: set[str | None], foreground: set[str | None], expected: bool
) -> None:
    assert servers_isolated(sentinel, foreground) is expected


# ---------------------------------------------------------------------------
# Admission, cancellation, per-run limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_isolated_sentinel_is_admitted_during_chat_and_video() -> None:
    set_sentinel_isolated(True)
    async with local_chat_session("edge"), local_video_session("edge"):
        assert await _admitted()
    # Not inside the chat grace either.
    assert utils_mod.chat_grace_remaining_s() > 0.0
    assert await _admitted()


@pytest.mark.asyncio
async def test_a_shared_sentinel_still_defers_to_chat() -> None:
    set_sentinel_isolated(False)
    async with local_chat_session("edge"):
        assert not await _admitted()


async def _start_sentinel_call(started: asyncio.Event) -> asyncio.Task[Any]:
    async def _call() -> str:
        started.set()
        await asyncio.sleep(5)
        return "done"

    task = asyncio.create_task(
        run_sentinel_llm_call(
            _call,
            deployment="edge",
            category="triage",
            admission_timeout_s=0.05,
            call_timeout_s=10,
        )
    )
    await started.wait()
    return task


@pytest.mark.asyncio
async def test_chat_does_not_cancel_an_isolated_sentinel_call() -> None:
    set_sentinel_isolated(True)
    call = await _start_sentinel_call(asyncio.Event())
    async with local_chat_session("edge"):
        await asyncio.sleep(0)
        assert not call.done()
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call


@pytest.mark.asyncio
async def test_chat_cancels_a_shared_sentinel_call() -> None:
    set_sentinel_isolated(False)
    call = await _start_sentinel_call(asyncio.Event())
    async with local_chat_session("edge"):
        with pytest.raises(SentinelLLMDeferredError):
            await call


def test_an_isolated_sentinel_is_not_limited() -> None:
    set_sentinel_isolated(True)
    assert not sentinel_calls_gated("edge")
    set_sentinel_isolated(False)
    assert sentinel_calls_gated("edge")
    assert not sentinel_calls_gated("cloud")


# ---------------------------------------------------------------------------
# A reload cancelling the caller is a cancellation, not a deferral
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("isolated", [True, False])
async def test_cancelling_the_caller_raises_cancelled(*, isolated: bool) -> None:
    """The engine's stop() must see CancelledError to finish unsubscribing."""
    set_sentinel_isolated(isolated)
    call = await _start_sentinel_call(asyncio.Event())
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call


@pytest.mark.asyncio
async def test_cancelling_the_caller_of_an_executor_call_raises_cancelled() -> None:
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def _blocking() -> str:
        loop.call_soon_threadsafe(started.set)
        release.wait(timeout=5)  # set right after the cancel
        return "done"

    call = asyncio.create_task(
        run_sentinel_llm_call_in_executor(
            _blocking,
            deployment="edge",
            category="triage",
            admission_timeout_s=0.05,
            call_timeout_s=10,
        )
    )
    await started.wait()
    call.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await call


# ---------------------------------------------------------------------------
# Stall probes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_probe_of_an_isolated_provider_is_sent_during_chat() -> None:
    isolated = {"sentinel_box": True, "chat_box": False}
    run_probe = make_gated_probe_runner(lambda _pid: "edge", isolated.__getitem__)

    async def _probe() -> str:
        return "ok"

    async with local_chat_session("edge"):
        assert await run_probe("sentinel_box", _probe) == "ok"
        with pytest.raises(StallProbeDeferredError):
            await run_probe("chat_box", _probe)
