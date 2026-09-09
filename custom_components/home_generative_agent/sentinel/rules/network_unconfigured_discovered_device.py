"""Rule: Home Assistant discovered devices on the LAN that nobody configured."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import (
    POSTURE_COOLDOWN_MINUTES,
    anyone_home,
    ha_security,
    make_finding,
    plural,
)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

# Config-flow handlers for cameras and recorders: an unconfigured one while
# the home is empty deserves more than an informational note.
CAMERA_HANDLERS: frozenset[str] = frozenset(
    {
        "reolink",
        "unifiprotect",
        "amcrest",
        "hikvision",
        "axis",
        "onvif",
        "frigate",
        "ring",
        "nest",
        "arlo",
        "blink",
        "eufy",
        "wyze",
        "tapo",
        "synology_dsm",
        "doorbird",
    }
)


class NetworkUnconfiguredDiscoveredDeviceRule:
    """In-progress discovery flows from SSDP, Zeroconf, DHCP, or HomeKit."""

    rule_id = "network_unconfigured_discovered_device"
    requires = frozenset({ha_cap("discovered_unconfigured")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding listing every discovered-but-unconfigured device."""
        discovered = ha_security(snapshot).get("discovered_unconfigured") or []
        items = sorted(
            {
                (
                    str(item.get("handler") or ""),
                    str(item.get("source") or ""),
                    str(item.get("title") or ""),
                )
                for item in discovered
            }
        )
        if not items:
            return []
        away = not anyone_home(snapshot)
        cameras = sorted({h for h, _, _ in items if h in CAMERA_HANDLERS})
        listed = ", ".join(
            f"{title or handler} ({handler} via {source})"
            for handler, source, title in items
        )
        return [
            make_finding(
                self.rule_id,
                severity="medium" if (cameras and away) else "low",
                confidence=0.7,
                evidence={
                    "devices": [
                        {"handler": h, "source": s, "title": t} for h, s, t in items
                    ],
                    "camera_handlers": cameras,
                },
                display={"anyone_home": not away},
                summary=(
                    "Home Assistant discovered "
                    f"{plural(len(items), 'unconfigured device')} "
                    f"on your network: {listed}."
                ),
                suggested_actions=[
                    (
                        "Configure them under Settings > Devices & services if they "
                        "are yours, ignore the discovery if you recognize them and "
                        "do not want them, and investigate any you do not recognize"
                    )
                ],
            )
        ]
