# ruff: noqa: S101
"""Tests for vector store bootstrap behavior."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.const import (
    CONF_VECTORS_BOOTSTRAPPED,
    DOMAIN,
)


class DummyStore:
    """Minimal AsyncPostgresStore stand-in."""

    def __init__(self, index_config: object | None) -> None:
        """Initialize dummy store."""
        self.index_config = index_config
        self.setup_calls = 0

    async def setup(self) -> None:
        """Simulate store setup."""
        self.setup_calls += 1


_BOOTSTRAP_VECTORS_ATTR = "_bootstrap_vectors_once"
bootstrap_vectors_once = cast(
    "Any",
    getattr(hga_component, _BOOTSTRAP_VECTORS_ATTR),
)


@pytest.mark.asyncio
async def test_vector_bootstrap_runs_with_index_config(hass: Any) -> None:
    """Vector bootstrap runs when index config is present."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    store = DummyStore(index_config=object())

    await bootstrap_vectors_once(hass, entry, store)

    assert store.setup_calls == 1
    assert entry.data.get(CONF_VECTORS_BOOTSTRAPPED) is True


@pytest.mark.asyncio
async def test_vector_bootstrap_skips_without_index_config(hass: Any) -> None:
    """Vector bootstrap skips when no index config is present."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    store = DummyStore(index_config=None)

    await bootstrap_vectors_once(hass, entry, store)

    assert store.setup_calls == 0
    assert entry.data.get(CONF_VECTORS_BOOTSTRAPPED) is None


@pytest.mark.asyncio
async def test_vector_bootstrap_skips_when_already_bootstrapped(hass: Any) -> None:
    """Vector bootstrap does not rerun once flagged."""
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_VECTORS_BOOTSTRAPPED: True})
    entry.add_to_hass(hass)
    store = DummyStore(index_config=object())

    await bootstrap_vectors_once(hass, entry, store)

    assert store.setup_calls == 0
