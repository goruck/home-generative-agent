# ruff: noqa: S101
"""Tests for promote_discovery_candidate service."""

from __future__ import annotations

import pytest

from custom_components.home_generative_agent.sentinel.discovery_store import (
    DiscoveryStore,
)
from custom_components.home_generative_agent.sentinel.proposal_store import (
    ProposalStore,
)


@pytest.mark.asyncio
async def test_promote_candidate(hass) -> None:
    discovery_store = DiscoveryStore(hass, max_records=10)
    await discovery_store.async_append(
        {
            "schema_version": 1,
            "generated_at": "2025-01-01T00:00:00+00:00",
            "model": "test",
            "candidates": [
                {
                    "candidate_id": "c1",
                    "title": "Test",
                    "summary": "Summary",
                    "evidence_paths": ["derived.is_night"],
                    "pattern": "pattern",
                    "confidence_hint": 0.5,
                }
            ],
        }
    )

    proposal_store = ProposalStore(hass)
    await proposal_store.async_append(
        {"candidate_id": "existing", "candidate": {}, "notes": ""}
    )

    candidate = discovery_store.find_candidate("c1")
    assert candidate is not None

    await proposal_store.async_append(
        {"candidate_id": "c1", "candidate": candidate, "notes": "note"}
    )

    records = await proposal_store.async_get_latest(10)
    assert records[0]["candidate_id"] == "c1"
