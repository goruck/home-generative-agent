"""Rule: a device joined a Zigbee, Z-Wave, Bluetooth, or Matter network."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.sentinel.network_inventory import (
    device_key,
)
from custom_components.home_generative_agent.snapshot.network import radio_cap

from .network_common import (
    PROTOCOL_LABELS,
    anyone_home,
    is_night,
    listed,
    make_finding,
    noun,
    radio,
)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import (
        AnomalyFinding,
        Severity,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
        RadioDevice,
    )


def _describe(device: RadioDevice) -> str:
    name = device.get("name") or "Unnamed device"
    detail = " ".join(
        part for part in (device.get("manufacturer"), device.get("model")) if part
    )
    protocol = PROTOCOL_LABELS.get(device["protocol"], device["protocol"])
    return f"{name} ({protocol}{', ' + detail if detail else ''})"


class RadioNewDeviceJoinedRule:
    """
    Devices paired since the device inventory was last committed.

    Pairing is almost always the owner, so this alerts once per device rather
    than standing until the device is trusted. The engine commits the
    inventory after dispatch and holds back devices whose finding was not
    delivered, so a suppressed alert is repeated on a later run.
    """

    rule_id = "radio_new_device_joined"
    requires = frozenset({radio_cap("devices"), radio_cap("new_devices")})
    cooldown_minutes = 0

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding naming every new radio device."""
        section = radio(snapshot)
        new_keys = set(section.get("new_devices") or [])
        if not new_keys:
            return []
        devices = [d for d in section.get("devices") or [] if device_key(d) in new_keys]
        if not devices:
            return []
        away = not anyone_home(snapshot)
        security = [d for d in devices if d.get("is_security_device")]
        severity: Severity = "low"
        if away and security:
            severity = "high"
        elif away or is_night(snapshot):
            severity = "medium"
        keys = sorted(device_key(d) for d in devices)
        count = len(devices)
        head = f"New radio {noun(count, 'device')} joined"
        context = " while nobody is home" if away else ""
        return [
            make_finding(
                self.rule_id,
                severity=severity,
                evidence={"device_keys": keys},
                display={
                    "device_ids": sorted(d["device_id"] for d in devices),
                    "security_device_ids": sorted(d["device_id"] for d in security),
                },
                summary=(
                    f"{head}{context}: {listed([_describe(d) for d in devices])}."
                ),
                suggested_actions=[
                    (
                        "If you did not pair this, remove it from its integration "
                        "and check who can reach your Zigbee, Z-Wave, or Bluetooth "
                        "setup"
                    ),
                    "Tap Trust device if you recognize it",
                ],
            )
        ]
