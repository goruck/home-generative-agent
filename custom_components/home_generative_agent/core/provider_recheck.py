"""
Re-check model providers that failed their setup health check (issue #711).

Setup builds a client only for a provider whose health check passed, so a
check that failed once used to leave that provider out until the integration
was reloaded. At boot that can be a provider that is fine: Home Assistant's
event loop is busy starting, a 2 s timeout fires late, and every cloud check
fails together. With the primary also down, every turn then fails until a
manual reload.

Here, each provider that failed is probed again once Home Assistant has
started, then on a backoff, then every few minutes for as long as one stays
down. When one answers, the entry is reloaded, which is the same step that
fixes it by hand and rebuilds every model, fallback chain and embedding with
it in place. A provider that rejects its credentials is dropped: it will not
start working without a configuration change, and changing the configuration
reloads the entry anyway.

Recovery reloads are rate-limited across reloads. A provider that answers the
probe but fails setup's own check again (one that hovers around the 2 s
timeout) would otherwise reload the entry every 30 s forever, each reload
tearing down Sentinel and the video analyzer and re-sending the fallback
notification. The budget lives in ``hass.data`` because each reload builds a
fresh task; it allows one recovery reload per cooldown, doubles the cooldown
after each reload that left a provider down, and resets once a setup passes
every check or when the next recovery comes long after the last one.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import CoreState, callback
from homeassistant.helpers.start import async_at_started

from ..const import DOMAIN  # noqa: TID252
from .utils import (
    CannotConnectError,
    InvalidAuthError,
    health_failure_reason,
    redact_url,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

# First probe as soon as Home Assistant has started, then back off.
RECHECK_DELAYS_S: tuple[float, ...] = (0.0, 30.0, 60.0, 120.0, 300.0)
# Then keep probing at this interval while any provider is still down.
RECHECK_INTERVAL_S = 300.0
# Minimum gap between recovery reloads, doubled after each one that left a
# provider down, up to the maximum.
RELOAD_COOLDOWN_S = 300.0
RELOAD_COOLDOWN_MAX_S = 3600.0

_BUDGET_KEY = "provider_recheck_reload_budget"
# Keep probing only while the entry can still use the result.
_LIVE_STATES = (ConfigEntryState.LOADED, ConfigEntryState.SETUP_IN_PROGRESS)


@dataclass(frozen=True)
class ProviderProbe:
    """One provider to re-check: a log label and a probe that raises on failure."""

    label: str
    probe: Callable[[], Awaitable[None]]


@dataclass
class _ReloadBudget:
    """Recovery reloads of one entry, carried across its reloads."""

    last_reload: float | None = None
    cooldown: float = RELOAD_COOLDOWN_S


def endpoint_label(name: str, url: str) -> str:
    """Name an endpoint for the log without credentials, query or fragment."""
    return f"{name} ({redact_url(url)})"


def _now() -> float:
    return time.monotonic()


def _budget(hass: HomeAssistant, entry: ConfigEntry) -> _ReloadBudget:
    budgets: dict[str, _ReloadBudget] = hass.data.setdefault(DOMAIN, {}).setdefault(
        _BUDGET_KEY, {}
    )
    return budgets.setdefault(entry.entry_id, _ReloadBudget())


def _reset_budget(hass: HomeAssistant, entry: ConfigEntry) -> None:
    hass.data.get(DOMAIN, {}).get(_BUDGET_KEY, {}).pop(entry.entry_id, None)


def _recheck_delays(*, skip_first: bool = False) -> Iterable[float]:
    delays = RECHECK_DELAYS_S[1:] if skip_first else RECHECK_DELAYS_S
    return itertools.chain(delays, itertools.repeat(RECHECK_INTERVAL_S))


async def _run_probe(probe: ProviderProbe) -> bool | None:
    """Return True if it answered, False if still down, None to stop probing it."""
    try:
        await probe.probe()
    except CannotConnectError as err:
        LOGGER.debug(
            "%s is still unavailable: %s", probe.label, health_failure_reason(err)
        )
        return False
    except InvalidAuthError as err:
        LOGGER.warning(
            "%s rejected its credentials (%s); it stays unavailable until its "
            "settings are fixed.",
            probe.label,
            health_failure_reason(err),
        )
        return None
    except Exception:
        LOGGER.exception("Unexpected error re-checking %s", probe.label)
        return False
    return True


def _reload_allowed(budget: _ReloadBudget, labels: str) -> bool:
    """Spend the reload budget, or log why the reload waits."""
    now = _now()
    if budget.last_reload is not None:
        wait = budget.last_reload + budget.cooldown - now
        if wait > 0:
            LOGGER.debug(
                "%s answered, but the last recovery reload was too recent; "
                "reloading in about %.0f s if it still answers.",
                labels,
                wait,
            )
            return False
        if now - budget.last_reload > 2 * budget.cooldown:
            # A separate incident long after the last reload: start over.
            budget.cooldown = RELOAD_COOLDOWN_S
        else:
            # The previous recovery reload left a provider down: back off.
            budget.cooldown = min(budget.cooldown * 2, RELOAD_COOLDOWN_MAX_S)
    budget.last_reload = now
    return True


async def async_recheck_until_recovered(
    hass: HomeAssistant,
    entry: ConfigEntry,
    probes: Iterable[ProviderProbe],
    delays: Iterable[float] | None = None,
) -> None:
    """Probe failed providers until one answers, then reload the entry."""
    pending: list[ProviderProbe] = list(probes)
    for delay in delays if delays is not None else _recheck_delays():
        if not pending:
            return
        if delay:
            await asyncio.sleep(delay)
        if entry.state not in _LIVE_STATES:
            # Unloaded, or an unload that failed and skipped cancelling us.
            return
        results = await asyncio.gather(*(_run_probe(p) for p in pending))
        recovered = [p for p, ok in zip(pending, results, strict=True) if ok]
        # A recovered provider stays pending until the reload is actually
        # scheduled, so one that answers while setup is still running (or
        # inside the reload cooldown) is probed again next round.
        pending = [p for p, ok in zip(pending, results, strict=True) if ok is not None]
        if not recovered:
            continue
        labels = ", ".join(p.label for p in recovered)
        if entry.state is not ConfigEntryState.LOADED:
            LOGGER.debug(
                "%s answered but the entry is %s; not reloading yet.",
                labels,
                entry.state,
            )
            continue
        if not _reload_allowed(_budget(hass, entry), labels):
            continue
        LOGGER.warning(
            "%s answered after failing its health check at setup; reloading "
            "Home Generative Agent so it can be used.",
            labels,
        )
        hass.config_entries.async_schedule_reload(entry.entry_id)
        return


@callback
def async_schedule_provider_recheck(
    hass: HomeAssistant, entry: ConfigEntry, probes: list[ProviderProbe]
) -> None:
    """Start re-checking ``probes`` once Home Assistant has started."""
    if not probes:
        # Every provider passed: any earlier recovery reload did its job.
        _reset_budget(hass, entry)
        return
    LOGGER.info(
        "Will re-check %s after Home Assistant has started and reload "
        "Home Generative Agent when one answers.",
        ", ".join(p.label for p in probes),
    )
    # On a reload (HA already running) the immediate probe would land while
    # this setup is still in progress and be wasted, so start at the backoff.
    skip_first = hass.state is CoreState.running

    # @callback so HassJob runs this on the loop, not on an executor thread
    # (see core/lifecycle.py). If it fires while the entry is unloading, the
    # task it creates is cancelled with the entry's other background tasks,
    # and the reload above is gated on the entry being loaded.
    @callback
    def _start(_hass: HomeAssistant) -> None:
        entry.async_create_background_task(
            hass,
            async_recheck_until_recovered(
                hass, entry, probes, _recheck_delays(skip_first=skip_first)
            ),
            name=f"{entry.domain} provider health recheck",
        )

    entry.async_on_unload(async_at_started(hass, _start))
