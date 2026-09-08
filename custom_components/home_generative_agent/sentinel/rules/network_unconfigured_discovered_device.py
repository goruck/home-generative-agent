"""Rule: Home Assistant discovered a device on the LAN that nobody configured."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import (
    POSTURE_COOLDOWN_MINUTES,
    anyone_home,
    ha_security,
    make_finding,
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
        """Return one finding per discovered-but-unconfigured device."""
        discovered = ha_security(snapshot).get("discovered_unconfigured") or []
        away = not anyone_home(snapshot)
        findings: list[AnomalyFinding] = []
        for item in discovered:
            handler = str(item.get("handler") or "")
            title = str(item.get("title") or "")
            source = str(item.get("source") or "")
            camera = handler in CAMERA_HANDLERS
            label = title or handler
            findings.append(
                make_finding(
                    self.rule_id,
                    severity="medium" if (camera and away) else "low",
                    confidence=0.7,
                    evidence={
                        "handler": handler,
                        "source": source,
                        "title": title,
                        "is_camera_handler": camera,
                        "anyone_home": not away,
                    },
                    summary=(
                        f"Home Assistant discovered {label} ({handler} via "
                        f"{source}) on your network, but it is not configured."
                    ),
                    suggested_actions=[
                        (
                            "Configure it under Settings > Devices & services if it "
                            "is yours, or ignore the discovery if you recognize it "
                            "and do not want it; investigate if you do not recognize "
                            "it."
                        )
                    ],
                )
            )
        return findings
