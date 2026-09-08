"""Rule: locks, alarm panels, or cameras unavailable for too long."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.const import (
    RECOMMENDED_SENTINEL_NETWORK_OFFLINE_DEVICE_MIN,
)
from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import anyone_home, ha_security, make_finding, plural

if TYPE_CHECKING:
    from collections.abc import Callable

    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class SecurityDeviceUnavailableRule:
    """
    Needs no router: Home Assistant already knows when it lost a device.

    Router client data (``network.clients``) corroborates this rule in a later
    phase by saying whether the device is still on the network. One finding
    per cycle lists every device past the threshold, so an integration outage
    that takes several cameras down reads as one event.
    """

    rule_id = "security_device_unavailable"
    requires = frozenset({ha_cap("unavailable_security_devices")})
    cooldown_minutes = 0

    def __init__(
        self,
        offline_minutes: int = RECOMMENDED_SENTINEL_NETWORK_OFFLINE_DEVICE_MIN,
        is_entity_excluded: Callable[[str, str], bool] | None = None,
    ) -> None:
        """Initialize with the minutes a device may be unavailable."""
        self._offline_minutes = max(1, int(offline_minutes))
        self._is_entity_excluded = is_entity_excluded

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding listing every security device past the threshold."""
        unavailable: dict[str, int] = (
            ha_security(snapshot).get("unavailable_security_devices") or {}
        )
        offline = {
            entity_id: int(minutes)
            for entity_id, minutes in unavailable.items()
            if int(minutes) >= self._offline_minutes
            and (
                self._is_entity_excluded is None
                or not self._is_entity_excluded(entity_id, self.rule_id)
            )
        }
        if not offline:
            return []
        names = {
            e["entity_id"]: e["friendly_name"] or e["entity_id"]
            for e in snapshot["entities"]
            if e["entity_id"] in offline
        }
        ids = sorted(offline)
        listed = ", ".join(f"{names.get(e, e)} ({offline[e]} min)" for e in ids)
        return [
            make_finding(
                self.rule_id,
                severity="high",
                triggering_entities=ids,
                evidence={
                    "entity_ids": ids,
                    "threshold_minutes": self._offline_minutes,
                },
                display={
                    "friendly_names": {e: names.get(e, e) for e in ids},
                    "unavailable_minutes": {e: offline[e] for e in ids},
                    "anyone_home": anyone_home(snapshot),
                },
                summary=(
                    f"{plural(len(ids), 'security device')} unavailable longer "
                    f"than {self._offline_minutes} minutes: {listed}."
                ),
                suggested_actions=["check_sensor"],
            )
        ]
