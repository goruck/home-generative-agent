"""Sentinel rules package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import (
        AnomalyFinding,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class StaticRule(Protocol):
    """
    Contract every built-in rule satisfies.

    ``requires`` names the snapshot capabilities (dotted paths in
    ``snapshot["network"]["capabilities"]``) the rule reads; the engine skips a
    rule whose capabilities are missing and reports it as inactive. An empty
    set means the rule always runs. ``cooldown_minutes`` is a per-type cooldown
    floor for rules that describe a standing condition; 0 means the configured
    cooldown applies unchanged.
    """

    rule_id: str
    requires: frozenset[str]
    cooldown_minutes: int

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return the findings for *snapshot*."""
        ...
