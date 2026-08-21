"""Config-entry lifecycle helpers shared by the integration and its platforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import callback

if TYPE_CHECKING:
    from collections.abc import Callable

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant


@callback
def defer_start_until_hass_started(
    hass: HomeAssistant,
    entry: ConfigEntry,
    start: Callable[[], None],
) -> None:
    """
    Run ``start`` when HA finishes starting, cancelling it if the entry unloads.

    Background engines constructed during ``async_setup_entry`` cannot start
    before Home Assistant has finished starting, so their start is deferred to
    ``EVENT_HOMEASSISTANT_STARTED``. Three details make that safe, and all
    three were regressions at some point:

    * ``@callback`` is load-bearing, not decoration. Without it ``HassJob``
      infers ``Executor`` for a plain sync function and dispatches the body to
      a worker THREAD, while ``_OneTimeListener`` has already unsubscribed on
      the loop thread. That leaves a window where the listener is gone but
      ``fired`` is still False — unload (loop thread) then calls the stale
      remover anyway and logs the very "Unable to remove unknown job listener"
      error the guard exists to prevent. Worse, ``start`` would be queued from
      that thread and could land after ``async_unload_entry`` has already
      stopped the engine and HA has deleted ``entry.runtime_data``.
    * The listener is cancelled on unload. If the entry reloads before HA
      finishes starting, a leaked listener would start the ORPHANED engine
      alongside its replacement — duplicate evaluation, notifications, and
      writes, from an instance holding stale options.
    * ``async_listen_once`` self-unsubscribes once it fires, so calling the
      remove function again on unload would log that same spurious error.
      Only call it if the event has not fired yet.

    The cancel is necessary but NOT sufficient: ``async_unload_entry`` stops
    each engine and then awaits several more times before HA runs the
    on-unload callbacks, so ``EVENT_HOMEASSISTANT_STARTED`` landing in one of
    those windows starts the engine and correctly leaves the cancel with
    nothing to do. Each engine closes that window with a one-way ``_stopped``
    latch checked in its own ``start()``, which is the one place downstream of
    every ordering. Platform callers without a latch must carry their own
    equivalent guard inside ``start`` (e.g. refusing when the entry is no
    longer loaded).
    """
    fired = False

    @callback
    def _run_start(_event: object) -> None:
        nonlocal fired
        fired = True
        start()

    remove_listener = hass.bus.async_listen_once(
        EVENT_HOMEASSISTANT_STARTED, _run_start
    )

    @callback
    def _cancel_listener() -> None:
        if not fired:
            remove_listener()

    entry.async_on_unload(_cancel_listener)
