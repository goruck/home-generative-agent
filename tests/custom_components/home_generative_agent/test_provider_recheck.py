# ruff: noqa: S101
"""
Re-checking providers that failed their setup health check (issue #711).

A check that failed once used to be final until a manual reload, so a cloud
fallback that only timed out while Home Assistant was booting left every turn
failing for the whole session.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import CoreState
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.const import DOMAIN
from custom_components.home_generative_agent.core import provider_recheck, utils
from custom_components.home_generative_agent.core.provider_recheck import (
    ProviderProbe,
    async_recheck_until_recovered,
    async_schedule_provider_recheck,
    endpoint_label,
)
from custom_components.home_generative_agent.core.utils import (
    CannotConnectError,
    InvalidAuthError,
    health_failure_reason,
    openai_healthy,
)


class _Probe:
    """A probe that plays back a script of outcomes, then keeps the last one."""

    def __init__(self, *outcomes: Exception | None) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    async def __call__(self) -> None:
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if outcome is not None:
            raise outcome


def _loaded_entry(hass: Any) -> MockConfigEntry:
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    entry.mock_state(hass, ConfigEntryState.LOADED)
    return entry


@pytest.fixture
def reloads(hass: Any, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    reload = MagicMock()
    monkeypatch.setattr(hass.config_entries, "async_schedule_reload", reload)
    return reload


@pytest.mark.asyncio
async def test_reloads_once_a_down_provider_answers(
    hass: Any, reloads: MagicMock
) -> None:
    entry = _loaded_entry(hass)
    probe = _Probe(CannotConnectError(), CannotConnectError(), None)

    await async_recheck_until_recovered(
        hass, entry, [ProviderProbe("OpenAI", probe)], delays=[0, 0, 0, 0]
    )

    assert probe.calls == 3
    reloads.assert_called_once_with(entry.entry_id)


@pytest.mark.asyncio
async def test_rejected_credentials_stop_the_recheck_without_a_reload(
    hass: Any, reloads: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    entry = _loaded_entry(hass)
    probe = _Probe(InvalidAuthError("HTTP 401"))

    with caplog.at_level(logging.WARNING):
        await async_recheck_until_recovered(
            hass, entry, [ProviderProbe("Gemini", probe)], delays=[0, 0, 0]
        )

    assert probe.calls == 1
    reloads.assert_not_called()
    assert "Gemini rejected its credentials (HTTP 401)" in caplog.text


@pytest.mark.asyncio
async def test_waits_for_a_loaded_entry_before_reloading(
    hass: Any, reloads: MagicMock
) -> None:
    """A provider that answers while setup is still running is probed again."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    entry.mock_state(hass, ConfigEntryState.SETUP_IN_PROGRESS)
    probe = _Probe(None)

    def _delays() -> Any:
        yield 0
        reloads.assert_not_called()
        entry.mock_state(hass, ConfigEntryState.LOADED)
        yield 0

    await async_recheck_until_recovered(
        hass, entry, [ProviderProbe("Anthropic", probe)], delays=_delays()
    )

    assert probe.calls == 2
    reloads.assert_called_once_with(entry.entry_id)


@pytest.mark.asyncio
async def test_one_recovery_reloads_while_another_provider_is_still_down(
    hass: Any, reloads: MagicMock
) -> None:
    entry = _loaded_entry(hass)
    down = _Probe(CannotConnectError())
    up = _Probe(CannotConnectError(), None)

    await async_recheck_until_recovered(
        hass,
        entry,
        [ProviderProbe("Ollama (http://ollama)", down), ProviderProbe("OpenAI", up)],
        delays=[0, 0, 0],
    )

    assert (down.calls, up.calls) == (2, 2)
    reloads.assert_called_once_with(entry.entry_id)


@pytest.mark.asyncio
async def test_recheck_starts_only_after_home_assistant_has_started(
    hass: Any, reloads: MagicMock
) -> None:
    hass.set_state(CoreState.not_running)
    entry = _loaded_entry(hass)
    probe = _Probe(None)

    async_schedule_provider_recheck(hass, entry, [ProviderProbe("OpenAI", probe)])
    await hass.async_block_till_done()
    assert probe.calls == 0, "probed while Home Assistant was still starting"

    hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
    await hass.async_block_till_done()
    # async_block_till_done does not wait for background tasks.
    await asyncio.gather(*entry._background_tasks)

    assert probe.calls == 1
    reloads.assert_called_once_with(entry.entry_id)


@pytest.mark.asyncio
async def test_recheck_never_starts_for_an_entry_unloaded_before_startup(
    hass: Any, reloads: MagicMock
) -> None:
    hass.set_state(CoreState.not_running)
    entry = _loaded_entry(hass)
    probe = _Probe(None)

    async_schedule_provider_recheck(hass, entry, [ProviderProbe("OpenAI", probe)])
    await entry._async_process_on_unload(hass)
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
    await hass.async_block_till_done()

    assert probe.calls == 0
    reloads.assert_not_called()


@pytest.mark.asyncio
async def test_nothing_is_scheduled_when_every_provider_passed(hass: Any) -> None:
    hass.set_state(CoreState.not_running)
    entry = _loaded_entry(hass)
    before = hass.bus.async_listeners().get(EVENT_HOMEASSISTANT_STARTED, 0)

    async_schedule_provider_recheck(hass, entry, [])

    assert hass.bus.async_listeners().get(EVENT_HOMEASSISTANT_STARTED, 0) == before


def test_failure_reason_names_a_silent_timeout() -> None:
    """``str(TimeoutError())`` is empty: the log line used to end in nothing."""
    err = CannotConnectError()
    err.__cause__ = TimeoutError()
    assert health_failure_reason(err) == "TimeoutError"

    assert health_failure_reason(CannotConnectError("HTTP 503")) == "HTTP 503"
    assert health_failure_reason(InvalidAuthError()) == "InvalidAuthError"


@pytest.mark.asyncio
async def test_health_warning_names_the_cause(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _timeout(*_args: Any, **_kwargs: Any) -> None:
        raise CannotConnectError from TimeoutError()

    monkeypatch.setattr(utils, "validate_openai_key", _timeout)

    with caplog.at_level(logging.WARNING):
        assert await openai_healthy(hass, "sk-test") is False

    assert "OpenAI health check failed: TimeoutError" in caplog.text


@pytest.mark.asyncio
async def test_unexpected_probe_error_keeps_probing(
    hass: Any, reloads: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """One odd exception must not end the recheck for good."""
    entry = _loaded_entry(hass)
    probe = _Probe(RuntimeError("boom"), None)

    await async_recheck_until_recovered(
        hass, entry, [ProviderProbe("OpenAI", probe)], delays=[0, 0]
    )

    assert probe.calls == 2
    reloads.assert_called_once_with(entry.entry_id)
    assert "Unexpected error re-checking OpenAI" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", [ConfigEntryState.NOT_LOADED, ConfigEntryState.FAILED_UNLOAD]
)
async def test_recheck_stops_once_the_entry_cannot_use_it(
    hass: Any, reloads: MagicMock, state: ConfigEntryState
) -> None:
    """A failed unload skips task cancellation; the loop must end on its own."""
    entry = _loaded_entry(hass)
    probe = _Probe(CannotConnectError())

    def _delays() -> Any:
        yield 0
        entry.mock_state(hass, state)
        yield 0
        yield 0

    await async_recheck_until_recovered(
        hass, entry, [ProviderProbe("OpenAI", probe)], delays=_delays()
    )

    assert probe.calls == 1
    reloads.assert_not_called()


@pytest.mark.asyncio
async def test_production_schedule_backs_off_then_polls(
    hass: Any, reloads: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _loaded_entry(hass)
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(provider_recheck.asyncio, "sleep", _sleep)
    probe = _Probe(*([CannotConnectError()] * 7), None)

    await async_recheck_until_recovered(hass, entry, [ProviderProbe("OpenAI", probe)])

    assert sleeps == [30.0, 60.0, 120.0, 300.0, 300.0, 300.0, 300.0]
    reloads.assert_called_once_with(entry.entry_id)


@pytest.mark.asyncio
async def test_a_flapping_provider_cannot_reload_in_a_loop(
    hass: Any, reloads: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A provider that answers the probe but fails setup again would reload forever.

    Each recheck generation stands for one setup after a recovery reload. The
    first reload is free; the next waits out the cooldown, which then doubles.
    """
    clock = [1000.0]
    monkeypatch.setattr(provider_recheck, "_now", lambda: clock[0])
    entry = _loaded_entry(hass)

    async def _generation() -> None:
        await async_recheck_until_recovered(
            hass, entry, [ProviderProbe("Ollama", _Probe(None))], delays=[0]
        )

    await _generation()
    assert reloads.call_count == 1

    clock[0] += 30  # the reload's setup failed again; its probe answers
    await _generation()
    assert reloads.call_count == 1, "reloaded again inside the cooldown"

    clock[0] += 300
    await _generation()
    assert reloads.call_count == 2

    clock[0] += 300  # cooldown is now 600 s
    await _generation()
    assert reloads.call_count == 2

    clock[0] += 300
    await _generation()
    assert reloads.call_count == 3


@pytest.mark.asyncio
async def test_a_clean_setup_resets_the_reload_cooldown(
    hass: Any, reloads: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(provider_recheck, "_now", lambda: clock[0])
    entry = _loaded_entry(hass)
    probes = [ProviderProbe("Ollama", _Probe(None))]

    await async_recheck_until_recovered(hass, entry, probes, delays=[0])
    async_schedule_provider_recheck(hass, entry, [])  # the reload passed every check
    clock[0] += 30
    await async_recheck_until_recovered(hass, entry, probes, delays=[0])

    assert reloads.call_count == 2


@pytest.mark.asyncio
async def test_a_reload_skips_the_probe_that_would_land_during_setup(
    hass: Any, reloads: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HA is already running on a reload: start at the 30 s backoff."""
    entry = _loaded_entry(hass)
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(provider_recheck.asyncio, "sleep", _sleep)

    async_schedule_provider_recheck(
        hass, entry, [ProviderProbe("OpenAI", _Probe(None))]
    )
    await asyncio.gather(*entry._background_tasks)

    assert sleeps == [30.0]
    reloads.assert_called_once_with(entry.entry_id)


def test_failure_reason_never_quotes_a_rejected_header() -> None:
    """h11 quotes the header value, so a key with a stray newline would leak."""
    err = CannotConnectError()
    err.__cause__ = httpx.LocalProtocolError(
        "Illegal header value b'Bearer sk-SECRET\\n'"
    )

    assert health_failure_reason(err) == "LocalProtocolError"

    err.__cause__ = httpx.ConnectError("All connection attempts failed")
    assert health_failure_reason(err) == "ConnectError: All connection attempts failed"


def test_endpoint_label_strips_credentials_and_query() -> None:
    assert (
        endpoint_label("OpenAI-compatible", "https://user:pw@host:8443/v1?token=x#f")
        == "OpenAI-compatible (https://host:8443/v1)"
    )
    assert (
        endpoint_label("Ollama", "http://[::1]:11434") == "Ollama (http://[::1]:11434)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("validator", "status", "error", "reason"),
    [
        ("validate_ollama_url", 503, CannotConnectError, "HTTP 503"),
        ("validate_openai_key", 503, CannotConnectError, "HTTP 503"),
        ("validate_openai_key", 401, InvalidAuthError, "HTTP 401"),
        ("validate_gemini_key", 403, InvalidAuthError, "HTTP 403"),
        ("validate_anthropic_key", 502, CannotConnectError, "HTTP 502"),
    ],
)
async def test_validators_name_the_http_status(  # noqa: PLR0913
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    validator: str,
    status: int,
    error: type[Exception],
    reason: str,
) -> None:
    class _Client:
        async def get(self, *_args: Any, **_kwargs: Any) -> httpx.Response:
            return httpx.Response(status)

    monkeypatch.setattr(utils, "get_async_client", lambda _hass: _Client())

    with pytest.raises(error) as caught:
        await getattr(utils, validator)(
            hass, "http://x" if "ollama" in validator else "k"
        )

    assert health_failure_reason(caught.value) == reason


@pytest.mark.asyncio
async def test_a_later_incident_starts_with_the_base_cooldown(
    hass: Any, reloads: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fallback that stays down keeps the budget alive; old backoff must decay."""
    clock = [1000.0]
    monkeypatch.setattr(provider_recheck, "_now", lambda: clock[0])
    entry = _loaded_entry(hass)
    probes = [ProviderProbe("Ollama", _Probe(None))]

    for step in (0, 300, 600, 1200):  # back off to a 2400 s cooldown
        clock[0] += step
        await async_recheck_until_recovered(hass, entry, probes, delays=[0])
    assert reloads.call_count == 4

    clock[0] += 86400  # a day later, a new outage recovers
    await async_recheck_until_recovered(hass, entry, probes, delays=[0])
    clock[0] += 300  # and the next recovery needs only the base cooldown
    await async_recheck_until_recovered(hass, entry, probes, delays=[0])

    assert reloads.call_count == 6
