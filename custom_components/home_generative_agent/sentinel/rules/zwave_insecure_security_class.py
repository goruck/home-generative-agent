"""Rule: a Z-Wave device joined without S2 security where it matters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import radio_cap

from .network_common import (
    POSTURE_COOLDOWN_MINUTES,
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
    )

_CLASS_LABELS = {"none": "no security", "s0": "legacy S0 security"}


class ZwaveInsecureSecurityClassRule:
    """
    Security devices on S0 or no security, and any device still on S0.

    A lock, alarm panel, or entry cover included without S2 can be commanded
    or sniffed far more easily than an S2 one, and there is no fix short of
    re-including it. Sensors and switches legitimately run without security,
    so those are reported only when they use S0, the deprecated scheme whose
    key exchange is weak.
    """

    rule_id = "zwave_insecure_security_class"
    requires = frozenset({radio_cap("devices.security_class")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding listing every weakly secured Z-Wave device."""
        security: list[tuple[str, str, str]] = []
        other: list[tuple[str, str, str]] = []
        for device in radio(snapshot).get("devices") or []:
            if device["protocol"] != "zwave":
                continue
            security_class = device.get("security_class")
            name = device.get("name") or "Unnamed device"
            entry = (name, device["device_id"], str(security_class))
            if device.get("is_security_device") and security_class in {"none", "s0"}:
                security.append(entry)
            elif security_class == "s0":
                other.append(entry)
        if not security and not other:
            return []
        severity: Severity = "high" if security else "low"
        parts: list[str] = []
        if security:
            parts.append(
                f"Z-Wave security {noun(len(security), 'device')} without S2: "
                + listed([f"{n} ({_CLASS_LABELS[c]})" for n, _, c in sorted(security)])
            )
        if other:
            parts.append(
                f"Z-Wave {noun(len(other), 'device')} on legacy S0 security: "
                + listed([n for n, _, _ in sorted(other)])
            )
        return [
            make_finding(
                self.rule_id,
                severity=severity,
                evidence={
                    "devices": {d: c for _, d, c in sorted(security + other)},
                },
                summary="; ".join(parts) + ".",
                suggested_actions=[
                    (
                        "Exclude the device and include it again with S2 security "
                        "from the Z-Wave JS integration; entering the device's DSK "
                        "PIN enables S2"
                    )
                ],
            )
        ]
