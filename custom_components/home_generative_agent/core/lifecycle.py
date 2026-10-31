"""Config-entry lifecycle helpers shared by the integration and its platforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.config_entries import (
    SIGNAL_CONFIG_ENTRY_CHANGED,
    ConfigEntryChange,
    ConfigEntryState,
)
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from ..const import DOMAIN  # noqa: TID252
from .subentry_resolver import get_database_subentry

if TYPE_CHECKING:
    from collections.abc import Callable

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

# An entry that is not running because of its database: waiting to retry on
# one it cannot reach, or failed on settings it cannot use. A change to the
# database settings (#748) or their deletion (#883) reloads it at once, since
# it has no update listener and its next retry can be 10 minutes away.
DATABASE_DOWN_STATES = (ConfigEntryState.SETUP_RETRY, ConfigEntryState.SETUP_ERROR)

# The states a Database subentry deletion reloads from: a running entry, which
# would otherwise keep the old pool and memory store until some later restart
# (#870), and one that is down on that database (#883).
_RELOAD_ON_DELETION_STATES = (ConfigEntryState.LOADED, *DATABASE_DOWN_STATES)


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


def _database_subentry_id(hass: HomeAssistant, entry: ConfigEntry) -> str | None:
    subentry = get_database_subentry(hass, entry)
    return subentry.subentry_id if subentry else None


@callback
def async_watch_database_deletion(hass: HomeAssistant) -> None:
    """
    Reload an entry when its Database subentry is deleted from the UI.

    Home Assistant reloads nothing when a subentry is removed, and the setup
    wizard is the one path that adds or changes the database, scheduling its
    own reload (which HA does not deduplicate). So a deletion is the one
    database change to react to, and this watches for it across every entry
    of the domain for the whole run, registered from `async_setup`:

    * A loaded entry would keep the old pool and memory store until some
      later restart (#870).
    * An entry in setup retry ("Cannot reach the PostgreSQL database yet")
      or setup error came up on in-memory storage only at its next retry, up
      to 10 minutes later, or not before a restart. That is the state a user
      is most likely in when they delete the database (#883).
    * A deletion while a setup attempt is in flight (a retry waiting on its
      connect, say) is kept until that attempt ends: the attempt read the
      subentry before it was deleted, so whether it loads or fails, the
      entry reloads then. It is dropped if the entry ends up unloaded, or a
      Database subentry is added back meanwhile (the wizard reloads).

    The same signal is sent for every state change, including those of the
    reload this schedules, so only the change that removes the subentry
    counts: each entry's Database subentry at its last change is remembered.
    """
    last_seen: dict[str, str | None] = {
        entry.entry_id: _database_subentry_id(hass, entry)
        for entry in hass.config_entries.async_entries(DOMAIN)
    }
    pending: set[str] = set()  # Deleted during a setup attempt still running.

    @callback
    def _on_entry_changed(change_type: ConfigEntryChange, entry: ConfigEntry) -> None:
        if entry.domain != DOMAIN:
            return
        entry_id = entry.entry_id
        if change_type == ConfigEntryChange.REMOVED:
            last_seen.pop(entry_id, None)
            pending.discard(entry_id)
            return
        current = _database_subentry_id(hass, entry)
        previous = last_seen.get(entry_id)
        last_seen[entry_id] = current
        if change_type != ConfigEntryChange.UPDATED:
            return
        removed_now = previous is not None and current is None
        if not removed_now and entry_id not in pending:
            return
        if current is not None:
            pending.discard(entry_id)  # Added back: the wizard reloads.
            return
        if entry.state is ConfigEntryState.SETUP_IN_PROGRESS:
            pending.add(entry_id)
            return
        pending.discard(entry_id)
        if entry.state in _RELOAD_ON_DELETION_STATES:
            hass.config_entries.async_schedule_reload(entry_id)
        # Otherwise not loaded: the next setup reads the entry as it is now.

    async_dispatcher_connect(hass, SIGNAL_CONFIG_ENTRY_CHANGED, _on_entry_changed)
