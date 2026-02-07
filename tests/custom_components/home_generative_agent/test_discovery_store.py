# ruff: noqa: S101
"""Tests for discovery store behavior."""

from __future__ import annotations

import pytest

from custom_components.home_generative_agent.sentinel.discovery_store import DiscoveryStore


@pytest.mark.asyncio
async def test_discovery_store_max_records(hass) -> None:
    store = DiscoveryStore(hass, max_records=2)
    await store.async_append({"id": 1})
    await store.async_append({"id": 2})
    await store.async_append({"id": 3})

    records = await store.async_get_latest(10)
    assert [r["id"] for r in records] == [3, 2]
