# ruff: noqa: S101
"""
Regression tests for the deferred background-engine start listeners.

The video analyzer, sentinel engine, discovery engine, and baseline updater all
have their start deferred to ``EVENT_HOMEASSISTANT_STARTED`` when the entry sets
up before Home Assistant finishes starting. Two things have to hold at once:

* unloading *before* the event fires must cancel the listener, or the orphaned
  engine starts alongside its replacement (duplicate capture loops and a second
  retention deque with deletion power over live files for the analyzer;
  duplicate evaluation, notifications, and writes for the sentinel engines);
* unloading *after* the event fires must not call the already-consumed remove
  function, or HA logs ``Unable to remove unknown job listener`` with a
  traceback on every post-startup reload.

Both depend on the listener body running on the **event loop**, not an executor
thread: ``_OneTimeListener`` unsubscribes on the loop and then dispatches the
body, so a thread hop would leave a window where the listener is already gone
but the guard flag still says it never fired.

The cancel alone is not enough. ``async_unload_entry`` stops each engine and
then awaits several more times before HA runs the on-unload callbacks, so the
event can land in one of those windows and start an engine the unload has
already stopped. Each engine therefore also carries a one-way ``_stopped``
latch, checked in its own ``start()`` — the one place downstream of every
ordering.

This module also covers the two lifecycle leaks that shared the same code
block: the ``EVENT_HOMEASSISTANT_STOP`` listener that used to be registered
unconditionally on every setup and never removed, and the synchronous OpenAI
httpx client that only that leaked listener ever closed.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import CoreState
from psycopg_pool import PoolTimeout
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.const import (
    CONF_SENTINEL_DISCOVERY_ENABLED,
    CONF_SENTINEL_ENABLED,
    CONF_VIDEO_ANALYZER_MODE,
    DOMAIN,
    VIDEO_ANALYZER_MODE_NOTIFY_ON_ANOMALY,
)
from custom_components.home_generative_agent.core.video_analyzer import VideoAnalyzer
from custom_components.home_generative_agent.sentinel.baseline import (
    SentinelBaselineUpdater,
)
from custom_components.home_generative_agent.sentinel.discovery_engine import (
    SentinelDiscoveryEngine,
)
from custom_components.home_generative_agent.sentinel.engine import SentinelEngine

from .test_fallback_setup import _fallback_setup_data, _patch_setup_dependencies

REMOVE_LISTENER_ERROR = "Unable to remove unknown job listener"


class RecordingStartable:
    """Lifecycle stand-in that records where and when start() ran."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        """Initialize the recording stand-in."""
        self.start_threads: list[threading.Thread] = []
        self.stop_calls = 0

    def start(self) -> None:
        """Record the thread the start ran on."""
        self.start_threads.append(threading.current_thread())

    async def stop(self) -> None:
        """Record the stop."""
        self.stop_calls += 1


class RecordingVideoAnalyzer(RecordingStartable):
    """Video analyzer stand-in."""


class RecordingEngine(RecordingStartable):
    """Sentinel/discovery engine stand-in."""


class RecordingHttpClient:
    """httpx.Client stand-in that records close() calls and their thread."""

    def __init__(self) -> None:
        """Initialize the recording client."""
        self.close_calls = 0
        self.close_threads: list[threading.Thread] = []

    def close(self) -> None:
        """Record the close and the thread it ran on."""
        self.close_calls += 1
        self.close_threads.append(threading.current_thread())


async def _setup_with_deferred_start(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[MockConfigEntry, RecordingVideoAnalyzer]:
    """Set up the entry with HA not yet started and the analyzer enabled."""
    hass.set_state(CoreState.not_running)
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True

    data = _fallback_setup_data()
    data.options[CONF_VIDEO_ANALYZER_MODE] = VIDEO_ANALYZER_MODE_NOTIFY_ON_ANOMALY
    _patch_setup_dependencies(hass, monkeypatch, data)

    analyzers: list[RecordingVideoAnalyzer] = []

    def _make_analyzer(*args: Any, **kwargs: Any) -> RecordingVideoAnalyzer:
        analyzer = RecordingVideoAnalyzer(*args, **kwargs)
        analyzers.append(analyzer)
        return analyzer

    monkeypatch.setattr(hga_component, "VideoAnalyzer", _make_analyzer)

    result = await cast("Any", hga_component).async_setup_entry(hass, entry)
    assert result is True
    assert analyzers, "setup did not construct a video analyzer"
    analyzer = analyzers[0]
    assert analyzer.start_threads == [], "analyzer started before HA finished starting"
    return entry, analyzer


async def _setup_with_deferred_sentinel_start(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[MockConfigEntry, RecordingEngine, RecordingEngine, RecordingHttpClient]:
    """Set up with HA not yet started and both sentinel engines enabled."""
    hass.set_state(CoreState.not_running)
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True

    data = _fallback_setup_data()
    data.options[CONF_SENTINEL_ENABLED] = True
    data.options[CONF_SENTINEL_DISCOVERY_ENABLED] = True
    _patch_setup_dependencies(hass, monkeypatch, data)

    sentinels: list[RecordingEngine] = []
    discoveries: list[RecordingEngine] = []
    clients: list[RecordingHttpClient] = []

    def _make(sink: list[RecordingEngine]) -> Any:
        def _factory(*args: Any, **kwargs: Any) -> RecordingEngine:
            engine = RecordingEngine(*args, **kwargs)
            sink.append(engine)
            return engine

        return _factory

    def _make_client(*_args: Any, **_kwargs: Any) -> RecordingHttpClient:
        client = RecordingHttpClient()
        clients.append(client)
        return client

    monkeypatch.setattr(hga_component, "VideoAnalyzer", RecordingVideoAnalyzer)
    monkeypatch.setattr(hga_component, "SentinelEngine", _make(sentinels))
    monkeypatch.setattr(hga_component, "SentinelDiscoveryEngine", _make(discoveries))
    # Patches the shared httpx module process-wide, not a component-local
    # alias — safe only because the fallback harness stubs every provider
    # constructor, so setup builds exactly one sync client.
    monkeypatch.setattr("httpx.Client", _make_client)

    result = await cast("Any", hga_component).async_setup_entry(hass, entry)
    assert result is True
    assert sentinels, "setup did not build a sentinel engine"
    assert discoveries, "setup did not build a discovery engine"
    assert clients, "setup did not build an OpenAI http client"
    assert sentinels[0].start_threads == [], "sentinel started before HA finished"
    assert discoveries[0].start_threads == [], "discovery started before HA finished"
    # Bind the client by identity, not creation order: if anything else in the
    # process built a sync client first, clients[0] would be the wrong object
    # and every close assertion downstream would silently test nothing.
    client = entry.runtime_data.openai_http_client
    assert client is clients[0], "captured the wrong httpx client"
    return entry, sentinels[0], discoveries[0], client


@pytest.mark.asyncio
async def test_started_listener_runs_inline_on_the_event_loop(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The start listener must run on the loop, in the firing tick.

    ``async_fire`` is deliberately not awaited: an executor-dispatched listener
    would not have run yet, and the guard flag would still be False while the
    listener itself is already unsubscribed.
    """
    analyzer = (await _setup_with_deferred_start(hass, monkeypatch))[1]

    hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)

    assert analyzer.start_threads == [threading.current_thread()]


@pytest.mark.asyncio
async def test_unload_after_start_does_not_remove_a_consumed_listener(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Unloading after the event fired must not re-remove the listener."""
    entry, analyzer = await _setup_with_deferred_start(hass, monkeypatch)

    hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
    await hass.async_block_till_done()
    assert len(analyzer.start_threads) == 1

    with caplog.at_level(logging.ERROR):
        await entry._async_process_on_unload(hass)

    assert REMOVE_LISTENER_ERROR not in caplog.text


@pytest.mark.asyncio
async def test_stop_latches_the_analyzer_against_a_later_deferred_start(
    hass: Any,
) -> None:
    """
    A stopped analyzer must refuse to start, however late the start lands.

    The listener cancel cannot cover this on its own. ``async_unload_entry``
    calls ``stop()`` and then awaits several more times before Home Assistant
    runs the entry's on-unload callbacks, so ``EVENT_HOMEASSISTANT_STARTED``
    firing in one of those windows starts the analyzer and *correctly* leaves
    the cancel with nothing to do. HA then deletes ``runtime_data`` around a
    live capture loop whose retention deque can delete the replacement
    entry's snapshot files. The latch is checked in ``start()``, which is
    the one place downstream of every ordering.
    """
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    analyzer = VideoAnalyzer(hass, entry)

    await analyzer.stop()

    # Without the latch this raises AttributeError, because start() reads
    # self.entry.runtime_data.options for the video-model semaphore and an
    # unloaded entry no longer has runtime_data. That crash is the production
    # symptom; the latch returns before touching it.
    analyzer.start()

    assert not hasattr(analyzer, "_cancel_track"), "stopped analyzer started anyway"


@pytest.mark.asyncio
async def test_unload_before_start_cancels_the_listener(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Unloading before the event fires must cancel the orphaned start."""
    entry, analyzer = await _setup_with_deferred_start(hass, monkeypatch)

    with caplog.at_level(logging.ERROR):
        await entry._async_process_on_unload(hass)

        hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
        await hass.async_block_till_done()

    assert analyzer.start_threads == [], "orphaned analyzer started after unload"
    assert REMOVE_LISTENER_ERROR not in caplog.text


@pytest.mark.asyncio
async def test_sentinel_start_listeners_run_inline_on_the_event_loop(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The sentinel and discovery start listeners must run on the loop.

    They used to be plain sync functions that hopped to the loop with
    ``call_soon_threadsafe``, which means ``HassJob`` dispatched the body to a
    worker thread. ``async_fire`` is deliberately not awaited here: an
    executor-dispatched listener would not have run yet.
    """
    _, sentinel, discovery, _ = await _setup_with_deferred_sentinel_start(
        hass, monkeypatch
    )

    hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)

    assert sentinel.start_threads == [threading.current_thread()]
    assert discovery.start_threads == [threading.current_thread()]


@pytest.mark.asyncio
async def test_unload_before_start_cancels_the_sentinel_listeners(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Unloading before the event fires must cancel both orphaned starts."""
    entry, sentinel, discovery, _ = await _setup_with_deferred_sentinel_start(
        hass, monkeypatch
    )

    with caplog.at_level(logging.ERROR):
        await entry._async_process_on_unload(hass)

        hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
        await hass.async_block_till_done()

    assert sentinel.start_threads == [], "orphaned sentinel started after unload"
    assert discovery.start_threads == [], "orphaned discovery started after unload"
    assert REMOVE_LISTENER_ERROR not in caplog.text


@pytest.mark.asyncio
async def test_unload_after_start_does_not_remove_consumed_sentinel_listeners(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Unloading after the event fired must not re-remove the listeners."""
    entry, sentinel, discovery, _ = await _setup_with_deferred_sentinel_start(
        hass, monkeypatch
    )

    hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
    await hass.async_block_till_done()
    assert len(sentinel.start_threads) == 1
    assert len(discovery.start_threads) == 1

    with caplog.at_level(logging.ERROR):
        await entry._async_process_on_unload(hass)

    assert REMOVE_LISTENER_ERROR not in caplog.text


@pytest.mark.asyncio
async def test_shutdown_of_a_loaded_entry_stops_engines_and_closes_the_client(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The positive half of the pair: STOP on a LOADED entry must still work.

    Home Assistant does not unload config entries at shutdown, so this handler
    is the only stop path a still-loaded entry has. Scoping it to the entry had
    to leave that working — and the removal test below asserts only zeroes,
    which would pass just as happily if the listener were never registered at
    all. Without this control, deleting the registration outright leaves the
    suite green.
    """
    _, sentinel, discovery, client = await _setup_with_deferred_sentinel_start(
        hass, monkeypatch
    )

    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done()

    assert sentinel.stop_calls == 1, "shutdown did not stop the sentinel engine"
    assert discovery.stop_calls == 1, "shutdown did not stop the discovery engine"
    assert client.close_calls == 1, "shutdown did not close the OpenAI client"


@pytest.mark.asyncio
async def test_stop_listener_is_removed_on_unload(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The EVENT_HOMEASSISTANT_STOP handler must not outlive its setup.

    It used to be registered unconditionally on every ``async_setup_entry`` and
    never removed, so each reload added another closure capturing that
    generation's engines and client. After N reloads shutdown fired N+1 of
    them, stopping long-dead objects and keeping every stale generation's
    object graph reachable until the process exited.

    Paired with the positive control above, which proves the listener is
    registered in the first place.
    """
    entry, sentinel, discovery, client = await _setup_with_deferred_sentinel_start(
        hass, monkeypatch
    )

    await entry._async_process_on_unload(hass)
    await hass.async_block_till_done()
    # The unload's own hook closed this client; the point below is that the
    # STOP closure does not fire a SECOND time on the dead generation.
    closes_after_unload = client.close_calls

    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done()

    assert sentinel.stop_calls == 0, "stale STOP closure stopped a dead engine"
    assert discovery.stop_calls == 0, "stale STOP closure stopped a dead engine"
    assert client.close_calls == closes_after_unload, "stale STOP closure re-closed"


@pytest.mark.asyncio
async def test_a_reload_storm_leaves_exactly_one_live_stop_closure(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    N reloads then shutdown must fire ONE stop closure, not N+1.

    This is the leak the entry-scoping exists to fix, and asserting it for a
    single generation does not prove it: the old code accumulated one closure
    per setup, so the bug is only visible across generations. Each dead
    generation's engines must stay untouched at shutdown, and only the live
    generation's may be stopped.
    """
    generations: list[
        tuple[MockConfigEntry, RecordingEngine, RecordingEngine, RecordingHttpClient]
    ] = []
    for _ in range(3):
        entry, sentinel, discovery, client = await _setup_with_deferred_sentinel_start(
            hass, monkeypatch
        )
        generations.append((entry, sentinel, discovery, client))
        # Unload every generation but the last, as a reload would.
        if len(generations) < 3:
            await entry._async_process_on_unload(hass)
            await hass.async_block_till_done()

    # Each dead generation closed its own client at its own unload; record that
    # so the shutdown below can be shown not to touch them again.
    closes_before_stop = [client.close_calls for *_, client in generations[:-1]]

    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done()

    for index, (_, sentinel, discovery, client) in enumerate(generations[:-1]):
        assert sentinel.stop_calls == 0, f"generation {index} sentinel was resurrected"
        assert discovery.stop_calls == 0, (
            f"generation {index} discovery was resurrected"
        )
        assert client.close_calls == closes_before_stop[index], (
            f"generation {index} client was re-closed by a stale STOP closure"
        )

    _, live_sentinel, live_discovery, live_client = generations[-1]
    assert live_sentinel.stop_calls == 1, "live generation was not stopped"
    assert live_discovery.stop_calls == 1, "live generation was not stopped"
    assert live_client.close_calls == 1, "live generation's client was not closed"


@pytest.mark.asyncio
async def test_unload_closes_the_openai_http_client_off_the_loop(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Unload must close the synchronous OpenAI client, and not on the loop.

    A fresh ``httpx.Client`` is built on every setup and handed to the OpenAI
    provider instances. Its only close used to be inside the STOP handler,
    which the wrap above now scopes to the entry — so without an unload-time
    close a reload storm would retain one client, its connection pool, and its
    sockets per generation.

    The thread assertion is the load-bearing half: this is a *sync* client, so
    a regression to a bare ``client.close()`` would still satisfy a call count
    while blocking the event loop.

    The close deliberately does NOT happen inside ``async_unload_entry`` — it
    is an on-unload callback, which Home Assistant runs after that function
    returns, hence after ``async_unload_platforms``. Asserting the ordering is
    the point: closing a transport out from under a still-registered entity is
    how this surfaces as "Cannot send a request, as the client has been
    closed."
    """
    loop_thread = threading.current_thread()
    entry, _, _, client = await _setup_with_deferred_sentinel_start(hass, monkeypatch)

    await cast("Any", hga_component).async_unload_entry(hass, entry)
    assert client.close_calls == 0, "client closed before the platforms were unloaded"

    await entry._async_process_on_unload(hass)
    await hass.async_block_till_done()

    assert client.close_calls == 1
    assert client.close_threads[0] is not loop_thread, "sync close blocked the loop"


@pytest.mark.asyncio
async def test_a_refused_platform_unload_leaves_the_engines_alive(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    If a platform refuses to unload, nothing may be torn down.

    Every teardown in ``async_unload_entry`` is irreversible — the engines set
    a one-way ``_stopped`` latch and the pool cannot be reopened — while
    returning False puts the entry in ``FAILED_UNLOAD``, which Home Assistant
    treats as non-recoverable and which keeps ``runtime_data`` and the
    refusing platform's entities in place. Tearing down first would strand
    live entities on dead resources with no way back short of a restart.
    """
    entry, sentinel, discovery, client = await _setup_with_deferred_sentinel_start(
        hass, monkeypatch
    )

    async def _refuse(*_args: Any, **_kwargs: Any) -> bool:
        return False

    monkeypatch.setattr(hass.config_entries, "async_unload_platforms", _refuse)

    result = await cast("Any", hga_component).async_unload_entry(hass, entry)

    assert result is False, "a refused platform unload must not report success"
    assert sentinel.stop_calls == 0, "engine stopped despite the aborted unload"
    assert discovery.stop_calls == 0, "engine stopped despite the aborted unload"
    assert client.close_calls == 0, "client closed despite the aborted unload"


class _UnreachablePool:
    """AsyncConnectionPool stand-in for a database that is not reachable."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        """Initialize the pool."""
        self.close_calls = 0

    async def open(self) -> None:
        """Open successfully — see the note in the test about wait=False."""

    async def close(self) -> None:
        """Record the close."""
        self.close_calls += 1


@pytest.mark.asyncio
async def test_a_failed_setup_closes_the_openai_http_client(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A setup that aborts must not leak the client it already built.

    The client is constructed early in ``async_setup_entry``, long before
    ``runtime_data`` exists, and Home Assistant never calls
    ``async_unload_entry`` for an entry that failed to load — so an
    unload-time close could never reach this path. It is also the path users
    repeat: reload, reload, reload until the database comes up. The close is
    therefore registered with ``entry.async_on_unload`` at construction, which
    Home Assistant runs on *every* failed setup as well as on unload
    (``ConfigEntry.async_setup``: ``finally: if not result ... await
    self._async_process_on_unload(hass)``), so one hook covers every abort
    path rather than the handful that happen to be ``return False``.

    The failure is injected at the database bootstrap, not at ``pool.open()``:
    psycopg-pool 3.3.0 defaults ``open()`` to ``wait=False``
    (``pool_async.py:406``), so it does not contact the database and the
    ``PoolTimeout`` branch above cannot fire. Faking a timeout there would be
    testing a failure mode the pinned pool does not have. An unreachable
    database really surfaces one block later, when the first connection is
    requested, and lands in the broad ``except Exception`` that closes the
    pool.

    This calls ``_async_process_on_unload`` directly because the test drives
    ``async_setup_entry`` itself rather than going through HA's config-entry
    machinery, which is what would normally invoke it.
    """
    hass.set_state(CoreState.not_running)
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True

    data = _fallback_setup_data()
    _patch_setup_dependencies(hass, monkeypatch, data)

    clients: list[RecordingHttpClient] = []
    pools: list[_UnreachablePool] = []

    def _make_client(*_args: Any, **_kwargs: Any) -> RecordingHttpClient:
        client = RecordingHttpClient()
        clients.append(client)
        return client

    def _make_pool(*args: Any, **kwargs: Any) -> _UnreachablePool:
        pool = _UnreachablePool(*args, **kwargs)
        pools.append(pool)
        return pool

    async def _unreachable(*_args: Any, **_kwargs: Any) -> None:
        raise PoolTimeout

    monkeypatch.setattr("httpx.Client", _make_client)
    # Undo the harness default of "no database" so the DB branch is taken.
    monkeypatch.setattr(
        hga_component, "build_database_uri_from_entry", lambda _e: "postgresql://x/y"
    )
    monkeypatch.setattr(hga_component, "AsyncConnectionPool", _make_pool)
    # Stub the langgraph stores: the real ones spawn background batch tasks the
    # failure path never reaps, which is a separate (pre-existing) leak and
    # would only add noise here.
    monkeypatch.setattr(hga_component, "AsyncPostgresStore", MagicMock())
    monkeypatch.setattr(hga_component, "AsyncPostgresSaver", MagicMock())
    monkeypatch.setattr(hga_component, "_bootstrap_db_once", _unreachable)

    result = await cast("Any", hga_component).async_setup_entry(hass, entry)
    assert result is False, "setup should have failed on the unreachable database"
    assert clients, "setup did not build an OpenAI http client"
    assert clients[0].close_calls == 0, "closed before HA ran the on-unload hooks"

    await entry._async_process_on_unload(hass)
    await hass.async_block_till_done()

    assert clients[0].close_calls == 1, "failed setup leaked the OpenAI http client"
    assert pools, "setup did not build a connection pool"
    assert pools[0].close_calls == 1, "failed setup leaked the connection pool"


@pytest.mark.asyncio
async def test_stop_latches_the_sentinel_engine_against_a_later_start() -> None:
    """
    A stopped sentinel engine must refuse to start, however late.

    The listener cancel cannot cover this on its own: ``async_unload_entry``
    calls ``stop()`` and then awaits several more times before HA runs the
    on-unload callbacks, so ``EVENT_HOMEASSISTANT_STARTED`` firing in one of
    those windows starts the engine and *correctly* leaves the cancel with
    nothing to do — leaving an orphaned evaluation loop with stale options
    running alongside the replacement entry.
    """
    engine = SentinelEngine(
        MagicMock(),
        {},
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        rule_registry=MagicMock(),
        entry_id="entry",
    )

    await engine.stop()
    engine.start()

    assert engine._task is None, "stopped sentinel engine started anyway"


@pytest.mark.asyncio
async def test_stop_latches_the_discovery_engine_against_a_later_start() -> None:
    """A stopped discovery engine must refuse to start, however late."""
    engine = SentinelDiscoveryEngine(
        hass=MagicMock(),
        options={},
        model=MagicMock(),
        store=MagicMock(),
        rule_registry=MagicMock(),
        proposal_store=MagicMock(),
    )

    await engine.stop()
    engine.start()

    assert engine._task is None, "stopped discovery engine started anyway"


@pytest.mark.asyncio
async def test_stop_latches_the_baseline_updater_against_a_later_start() -> None:
    """A stopped baseline updater must refuse to start, however late."""
    updater = SentinelBaselineUpdater(MagicMock(), MagicMock(), {})

    await updater.stop()
    updater.start()

    assert updater._task is None, "stopped baseline updater started anyway"
