# ruff: noqa: S101
"""
Regression tests for the deferred video-analyzer start listener.

The analyzer start is deferred to ``EVENT_HOMEASSISTANT_STARTED`` when the entry
sets up before Home Assistant finishes starting. Two things have to hold at once:

* unloading *before* the event fires must cancel the listener, or the orphaned
  analyzer starts alongside its replacement (duplicate capture loops, and a
  second retention deque with deletion power over live files);
* unloading *after* the event fires must not call the already-consumed remove
  function, or HA logs ``Unable to remove unknown job listener`` with a
  traceback on every post-startup reload.

Both depend on the listener body running on the **event loop**, not an executor
thread: ``_OneTimeListener`` unsubscribes on the loop and then dispatches the
body, so a thread hop would leave a window where the listener is already gone
but the guard flag still says it never fired.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, cast

import pytest
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import CoreState
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.const import (
    CONF_VIDEO_ANALYZER_MODE,
    DOMAIN,
    VIDEO_ANALYZER_MODE_NOTIFY_ON_ANOMALY,
)

from .test_fallback_setup import _fallback_setup_data, _patch_setup_dependencies

REMOVE_LISTENER_ERROR = "Unable to remove unknown job listener"


class RecordingVideoAnalyzer:
    """Video analyzer stand-in that records where and when start() ran."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        """Initialize the recording analyzer."""
        self.start_threads: list[threading.Thread] = []
        self.stop_calls = 0

    def start(self) -> None:
        """Record the thread the start ran on."""
        self.start_threads.append(threading.current_thread())

    async def stop(self) -> None:
        """Record the stop."""
        self.stop_calls += 1


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
