"""Rule: radio coordinators, sticks, or Bluetooth proxies have firmware waiting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import radio_cap

from .network_common import (
    POSTURE_COOLDOWN_MINUTES,
    make_finding,
    plural,
    radio_posture,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class RadioCoordinatorUpdatePendingRule:
    """Coordinator and proxy firmware carries every radio device's security."""

    rule_id = "radio_coordinator_update_pending"
    requires = frozenset({radio_cap("posture.coordinator_update_pending")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def __init__(
        self, is_entity_excluded: Callable[[str, str], bool] | None = None
    ) -> None:
        """Initialize with the engine's per-rule entity exclusion check."""
        self._is_entity_excluded = is_entity_excluded

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding listing every pending coordinator update."""
        ids = sorted(
            entity_id
            for entity_id in radio_posture(snapshot).get("coordinator_update_pending")
            or []
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
                    f"Radio coordinator or proxy firmware "
                    f"{plural(len(ids), 'update')} available: {', '.join(parts)}."
                ),
                suggested_actions=[
                    (
                        "Install the update from Settings > System > Updates; keep "
                        "a backup of the Zigbee or Z-Wave network first"
                    )
                ],
            )
        ]
