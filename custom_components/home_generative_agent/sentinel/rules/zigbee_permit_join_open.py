"""Rule: a Zigbee network is accepting new devices."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import radio_cap

from .network_common import anyone_home, make_finding, radio_posture

if TYPE_CHECKING:
    from collections.abc import Callable

    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class ZigbeePermitJoinOpenRule:
    """
    Zigbee2MQTT's permit join is on.

    While it is on, any Zigbee device in radio range can join the network. A
    window opened on purpose closes by itself within 254 seconds; one opened
    while nobody is home, or through the MQTT broker by someone else, is
    exactly what this rule exists to catch. The engine wakes on the bridge
    switch turning on so the short window is not missed between polls.
    """

    rule_id = "zigbee_permit_join_open"
    requires = frozenset({radio_cap("posture.zigbee_permit_join")})
    cooldown_minutes = 0

    def __init__(
        self, is_entity_excluded: Callable[[str, str], bool] | None = None
    ) -> None:
        """Initialize with the engine's per-rule entity exclusion check."""
        self._is_entity_excluded = is_entity_excluded

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding naming the bridge switches that are on."""
        posture = radio_posture(snapshot)
        if not posture.get("zigbee_permit_join"):
            return []
        ids = sorted(
            entity_id
            for entity_id in posture.get("zigbee_permit_join_entity_ids") or []
            if self._is_entity_excluded is None
            or not self._is_entity_excluded(entity_id, self.rule_id)
        )
        if not ids:
            return []
        away = not anyone_home(snapshot)
        context = " while nobody is home" if away else ""
        return [
            make_finding(
                self.rule_id,
                severity="high" if away else "medium",
                triggering_entities=ids,
                evidence={"entity_ids": ids},
                summary=(
                    f"Zigbee2MQTT is accepting new devices{context} "
                    f"(permit join is on: {', '.join(ids)})."
                ),
                suggested_actions=[
                    "Turn off permit join unless you are pairing a device right now"
                ],
            )
        ]
