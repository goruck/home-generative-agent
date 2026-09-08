"""Rule: a lock, alarm panel, or camera has been unavailable for too long."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.const import (
    RECOMMENDED_SENTINEL_NETWORK_OFFLINE_DEVICE_MIN,
)
from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import ha_security, make_finding

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class SecurityDeviceUnavailableRule:
    """
    Needs no router: Home Assistant already knows when it lost a device.

    Router client data (``network.clients``) corroborates this rule in a later
    phase by saying whether the device is still on the network.
    """

    rule_id = "security_device_unavailable"
    requires = frozenset({ha_cap("unavailable_security_devices")})

    def __init__(
        self, offline_minutes: int = RECOMMENDED_SENTINEL_NETWORK_OFFLINE_DEVICE_MIN
    ) -> None:
        """Initialize with the minutes a device may be unavailable."""
        self._offline_minutes = max(1, int(offline_minutes))

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding per security device unavailable past the threshold."""
        unavailable: dict[str, int] = (
            ha_security(snapshot).get("unavailable_security_devices") or {}
        )
        names = {
            e["entity_id"]: e["friendly_name"] or e["entity_id"]
            for e in snapshot["entities"]
        }
        findings: list[AnomalyFinding] = []
        for entity_id, minutes in sorted(unavailable.items()):
            if int(minutes) < self._offline_minutes:
                continue
            name = names.get(entity_id, entity_id)
            findings.append(
                make_finding(
                    self.rule_id,
                    severity="high",
                    triggering_entities=[entity_id],
                    evidence={
                        "entity_id": entity_id,
                        "friendly_name": name,
                        "domain": entity_id.partition(".")[0],
                        "unavailable_minutes": int(minutes),
                        "threshold_minutes": self._offline_minutes,
                        "anyone_home": bool(
                            snapshot["derived"].get("anyone_home", False)
                        ),
                    },
                    summary=(
                        f"{name} has been unavailable for {int(minutes)} minutes "
                        f"(threshold {self._offline_minutes})."
                    ),
                    suggested_actions=["check_sensor"],
                )
            )
        return findings
