"""
Detect a model provider that stopped answering, and fail fast past it (#828).

A provider can stop answering at runtime without refusing connections: an
Ollama server wedged under memory pressure accepts the request and never
replies (#813). Every caller bounds its model call with its own deadline, so
without this module a hung provider used up the caller's whole budget, the
fallback chain never reached its next provider, and nothing said why.

Two pieces work together:

* ``model_call_budget`` records the caller's deadline in a context variable,
  so the fallback wrappers in ``fallback.py`` can give each provider a share
  of it (``member_timeout``) and still leave time for the next one.
* ``ProviderStallTracker`` counts consecutive timeouts per provider. After
  ``STALL_TIMEOUT_THRESHOLD`` in a row the provider is marked stalled: the
  wrappers skip it, a warning is logged, ``sensor.sentinel_health`` shows it,
  and a background probe sends it a tiny request every
  ``STALL_PROBE_INTERVAL_S`` until it answers again.

The probe is a real model request, not the setup-time health check: in #813
the server still answered health checks while generation hung.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any

from homeassistant.util import dt as dt_util
from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from datetime import datetime

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

# Timeouts in a row before a provider counts as stalled. One slow answer is
# not an outage; three in a row is.
STALL_TIMEOUT_THRESHOLD = 3
# How often a stalled provider is probed, and how long a probe may take.
STALL_PROBE_INTERVAL_S = 60.0
STALL_PROBE_TIMEOUT_S = 60.0
# Time the last provider in a chain leaves unused, so its own timeout fires
# (and is recorded) before the caller's deadline cancels the call: a tenth of
# what is left, at most a second.
_LAST_MEMBER_MARGIN_S = 1.0
_LAST_MEMBER_MARGIN_FRACTION = 0.1

_PROBE_PROMPT = "Reply with OK."

_call_deadline: ContextVar[float | None] = ContextVar(
    "hga_model_call_deadline", default=None
)


@contextmanager
def model_call_budget(seconds: float) -> Generator[None]:
    """
    Tell fallback chains how long the caller will wait for this model call.

    Use it around (or before creating the task for) a call that the caller
    bounds with its own timeout. A nested budget never extends an outer one.
    """
    deadline = monotonic() + seconds
    outer = _call_deadline.get()
    if outer is not None:
        deadline = min(deadline, outer)
    token = _call_deadline.set(deadline)
    try:
        yield
    finally:
        _call_deadline.reset(token)


def member_timeout(*, last: bool) -> float | None:
    """
    Return how long one provider in a chain may take, or None for no limit.

    Every provider but the last gets half of what is left of the caller's
    budget, so the next one still has time; the last gets the rest, less a
    small margin. None when the caller set no budget.
    """
    deadline = _call_deadline.get()
    if deadline is None:
        return None
    remaining = deadline - monotonic()
    if last:
        margin = min(_LAST_MEMBER_MARGIN_S, remaining * _LAST_MEMBER_MARGIN_FRACTION)
        return max(remaining - margin, 0.0)
    return max(remaining / 2, 0.0)


@dataclass
class _ProviderState:
    """What the tracker knows about one provider."""

    name: str
    consecutive_timeouts: int = 0
    stalled_since: datetime | None = None
    last_probe: datetime | None = None
    probe_model: Any = None
    probe_task: asyncio.Task[None] | None = None


class ProviderStallTracker:
    """Per-provider stall state shared by every fallback chain of one entry."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        *,
        provider_name: Callable[[str], str],
        on_change: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the tracker for one config entry."""
        self._hass = hass
        self._entry = entry
        self._provider_name = provider_name
        self._on_change = on_change
        self._states: dict[str, _ProviderState] = {}

    def _state(self, provider_id: str) -> _ProviderState:
        state = self._states.get(provider_id)
        if state is None:
            state = _ProviderState(name=self._provider_name(provider_id))
            self._states[provider_id] = state
        return state

    def name(self, provider_id: str) -> str:
        """Return the readable name of a provider."""
        return self._state(provider_id).name

    def is_stalled(self, provider_id: str) -> bool:
        """Return True while the provider is marked stalled."""
        state = self._states.get(provider_id)
        return state is not None and state.stalled_since is not None

    def stalled_since(self, provider_id: str) -> datetime | None:
        """Return when the provider was marked stalled, or None."""
        state = self._states.get(provider_id)
        return state.stalled_since if state else None

    def record_timeout(self, provider_id: str, model: Any) -> None:
        """Count a timeout; mark the provider stalled at the threshold."""
        state = self._state(provider_id)
        state.consecutive_timeouts += 1
        if state.probe_model is None:
            state.probe_model = model
        if (
            state.stalled_since is not None
            or state.consecutive_timeouts < STALL_TIMEOUT_THRESHOLD
        ):
            return
        state.stalled_since = dt_util.utcnow()
        LOGGER.warning(
            "Model provider %s stopped answering (%d timeouts in a row). "
            "Skipping it, and checking it every %.0f s until it answers again. "
            "If it is an Ollama server, restarting it usually clears this.",
            state.name,
            state.consecutive_timeouts,
            STALL_PROBE_INTERVAL_S,
        )
        state.probe_task = self._entry.async_create_background_task(
            self._hass,
            self._async_probe_until_answered(provider_id),
            f"home_generative_agent provider stall probe {provider_id}",
        )
        self._changed()

    def record_success(self, provider_id: str) -> None:
        """Clear the timeout count and any stall for a provider that answered."""
        state = self._states.get(provider_id)
        if state is None:
            return
        state.consecutive_timeouts = 0
        if state.stalled_since is None:
            return
        LOGGER.info(
            "Model provider %s is answering again (stalled since %s).",
            state.name,
            state.stalled_since.isoformat(),
        )
        state.stalled_since = None
        task, state.probe_task = state.probe_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        self._changed()

    def status(self) -> list[dict[str, Any]]:
        """Return the stalled providers, for sensor.sentinel_health."""
        return [
            {
                "provider": state.name,
                "stalled_since": state.stalled_since.isoformat(),
                "consecutive_timeouts": state.consecutive_timeouts,
                "last_probe": state.last_probe.isoformat()
                if state.last_probe
                else None,
            }
            for state in self._states.values()
            if state.stalled_since is not None
        ]

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

    async def _async_probe_until_answered(self, provider_id: str) -> None:
        """Send the stalled provider a tiny request until it answers."""
        state = self._state(provider_id)
        while state.stalled_since is not None:
            await asyncio.sleep(STALL_PROBE_INTERVAL_S)
            if state.stalled_since is None:
                return
            state.last_probe = dt_util.utcnow()
            try:
                async with asyncio.timeout(STALL_PROBE_TIMEOUT_S):
                    await state.probe_model.ainvoke(
                        [HumanMessage(content=_PROBE_PROMPT)]
                    )
            except Exception as err:  # noqa: BLE001
                # Any failure (timeout, refused connection, error reply)
                # means it is not answering yet; the next probe tries again.
                LOGGER.debug(
                    "Stall probe for model provider %s failed: %s",
                    state.name,
                    type(err).__name__,
                )
                self._changed()
                continue
            self.record_success(provider_id)
            return
