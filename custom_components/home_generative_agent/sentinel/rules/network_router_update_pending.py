"""Rule: router / gateway integrations report firmware waiting to install."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import posture_cap

from .network_common import POSTURE_COOLDOWN_MINUTES, make_finding, plural, posture

if TYPE_CHECKING:
    from collections.abc import Callable

    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class NetworkRouterUpdatePendingRule:
    """Router firmware updates close the holes attackers reach first."""

    rule_id = "network_router_update_pending"
    requires = frozenset({posture_cap("router_update_pending")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def __init__(
        self, is_entity_excluded: Callable[[str, str], bool] | None = None
    ) -> None:
        """Initialize with the engine's per-rule entity exclusion check."""
        self._is_entity_excluded = is_entity_excluded

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding listing every router update entity in state ``on``."""
        section = posture(snapshot)
        if not section.get("router_update_pending"):
            return []
        ids = sorted(
            entity_id
            for entity_id in section.get("router_update_entities") or []
            if self._is_entity_excluded is None
            or not self._is_entity_excluded(entity_id, self.rule_id)
        )
        if not ids:
            return []
        by_id = {
            e["entity_id"]: e for e in snapshot["entities"] if e["entity_id"] in ids
        }
        parts = []
        versions: dict[str, dict[str, str | None]] = {}
        for entity_id in ids:
            entity = by_id.get(entity_id)
            attrs = entity["attributes"] if entity else {}
            name = (entity["friendly_name"] if entity else None) or entity_id
            installed = attrs.get("installed_version")
            latest = attrs.get("latest_version")
            versions[entity_id] = {"installed": installed, "latest": latest}
            parts.append(
                f"{name} ({installed} -> {latest})" if installed and latest else name
            )
        return [
            make_finding(
                self.rule_id,
                severity="medium",
                triggering_entities=ids,
                evidence={"entity_ids": ids},
                display={"versions": versions},
                summary=(
                    f"Router firmware {plural(len(ids), 'update')} available: "
                    f"{', '.join(parts)}."
                ),
                suggested_actions=[
                    (
                        "Install the update from the router's app or from the update "
                        "entity in Settings > System > Updates"
                    )
                ],
            )
        ]
