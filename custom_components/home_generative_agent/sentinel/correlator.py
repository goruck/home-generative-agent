"""
Sentinel correlator: groups related findings into CompoundFindings.

Each call to :meth:`SentinelCorrelator.correlate` is **scoped to a single
``_run_once()`` cycle**.  No state is carried across calls, which guarantees
that findings from separate evaluation cycles are never merged (cross-run
isolation).

Correlation heuristics
----------------------
Two :class:`~sentinel.models.AnomalyFinding` objects are considered *related*
when **at least one** of the following holds:

1. **Same area** - both findings have a non-empty ``evidence["area"]`` value
   that matches (case-insensitive).
2. **Overlapping entities** - their ``triggering_entities`` sets share at
   least one entity ID.
3. **Complementary rule types** - the pair belongs to a known set of rule
   combinations that frequently indicate the same underlying incident (e.g.
   an open door while away *and* a camera detecting an unknown person).

Findings that cannot be correlated with any other finding in the same call
are returned unchanged as individual :class:`~sentinel.models.AnomalyFinding`
objects.  Emitted :class:`~sentinel.models.CompoundFinding` objects are
frozen dataclasses and therefore immutable after creation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .models import CompoundFinding

if TYPE_CHECKING:
    from .models import AnomalyFinding

LOGGER = logging.getLogger(__name__)

# Rule-type pairs that are considered complementary and should be correlated
# even when they share no area or entity overlap.  Each entry is a frozenset
# of exactly two rule-type strings.
_COMPLEMENTARY_PAIRS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"open_entry_while_away", "unknown_person_camera_no_home"}),
        frozenset({"unlocked_lock_at_night", "camera_entry_unsecured"}),
        frozenset({"open_entry_while_away", "camera_entry_unsecured"}),
        frozenset({"unlocked_lock_at_night", "open_entry_while_away"}),
    }
)


def _area_of(finding: AnomalyFinding) -> str:
    """Return the normalised area string from evidence, or empty string."""
    area = finding.evidence.get("area")
    if isinstance(area, str):
        return area.strip().lower()
    return ""


def _are_related(a: AnomalyFinding, b: AnomalyFinding) -> bool:
    """Return True when two findings should be grouped together."""
    # 1. Same non-empty area.
    area_a = _area_of(a)
    area_b = _area_of(b)
    if area_a and area_b and area_a == area_b:
        return True

    # 2. Overlapping triggering entities.
    if set(a.triggering_entities) & set(b.triggering_entities):
        return True

    # 3. Complementary rule-type pairs.
    pair: frozenset[str] = frozenset({a.type, b.type})
    return pair in _COMPLEMENTARY_PAIRS


def _build_groups(findings: list[AnomalyFinding]) -> list[list[AnomalyFinding]]:
    """
    Union-find grouping: returns a list of equivalence classes.

    Each inner list contains one or more findings that are mutually related.
    Findings that have no relation to any other finding form singleton groups.
    """
    n = len(findings)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if _are_related(findings[i], findings[j]):
                union(i, j)

    # Collect groups.
    groups: dict[int, list[AnomalyFinding]] = {}
    for i, finding in enumerate(findings):
        root = find(i)
        groups.setdefault(root, []).append(finding)

    return list(groups.values())


class SentinelCorrelator:
    """
    Correlate a batch of findings from a single evaluation cycle.

    Instances carry **no persistent state**.  Every call to
    :meth:`correlate` operates independently, ensuring strict cross-run
    isolation.
    """

    def correlate(
        self, findings: list[AnomalyFinding]
    ) -> list[AnomalyFinding | CompoundFinding]:
        """
        Group related findings and return the correlated output list.

        Parameters
        ----------
        findings:
            All :class:`~sentinel.models.AnomalyFinding` objects produced by
            a single ``_run_once()`` evaluation cycle.

        Returns
        -------
        list[AnomalyFinding | CompoundFinding]
            Singleton findings are returned as-is.  Groups of two or more
            related findings are replaced by a single immutable
            :class:`~sentinel.models.CompoundFinding`.

        """
        if not findings:
            return []

        groups = _build_groups(findings)
        output: list[AnomalyFinding | CompoundFinding] = []

        for group in groups:
            if len(group) == 1:
                output.append(group[0])
            else:
                compound = CompoundFinding.from_findings(group)
                LOGGER.info(
                    "Sentinel correlator grouped %d findings into compound %s "
                    "(types=%s).",
                    len(group),
                    compound.compound_id,
                    [f.type for f in group],
                )
                output.append(compound)

        return output
