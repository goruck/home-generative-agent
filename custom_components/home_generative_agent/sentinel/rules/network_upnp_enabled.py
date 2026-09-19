"""Rule: the router accepts UPnP requests, so any device can open ports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import posture_cap

from .network_common import (
    POSTURE_COOLDOWN_MINUTES,
    listed,
    make_finding,
    plural,
    posture,
)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

_EVIDENCE_TEXT: dict[str, str] = {
    "ssdp": "is advertising UPnP (Internet Gateway Device) on your network",
    "integration": "is answering the UPnP/IGD integration",
    "discovery_flow": "was discovered by Home Assistant as a UPnP gateway",
    "eero": "has UPnP turned on in its settings",
}


# Gateway names are LAN-advertised text; a push headline names at most this many.
MAX_NAMED_GATEWAYS = 2


class NetworkUpnpEnabledRule:
    """UPnP lets any device on the LAN expose itself to the internet unasked."""

    rule_id = "network_upnp_enabled"
    requires = frozenset({posture_cap("upnp_enabled")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one standing finding while UPnP is observed on."""
        section = posture(snapshot)
        if not section.get("upnp_enabled"):
            return []
        evidence_source = str(section.get("upnp_evidence") or "ssdp")
        names = [str(n) for n in section.get("upnp_gateway_names") or []]
        subject = (
            f"Your router ({listed(names, MAX_NAMED_GATEWAYS)})"
            if names
            else "Your router"
        )
        how = _EVIDENCE_TEXT.get(evidence_source, _EVIDENCE_TEXT["ssdp"])
        sentence = (
            f"{subject} {how}. UPnP lets devices on your network open ports to "
            "the internet without asking you, unless the router restricts it."
        )
        count = section.get("upnp_port_mapping_count")
        display: dict[str, object] = {"gateways": names, "evidence": evidence_source}
        if isinstance(count, int) and not isinstance(count, bool):
            display["port_mapping_count"] = count
            sentence += (
                f" {plural(count, 'port mapping')} "
                f"{'is' if count == 1 else 'are'} currently open through UPnP."
            )
        return [
            make_finding(
                self.rule_id,
                severity="medium",
                evidence={"upnp_enabled": True},
                display=display,
                summary=sentence,
                suggested_actions=[
                    (
                        "Turn off UPnP in your router's settings and forward only "
                        "the ports you need by hand"
                    ),
                    (
                        "If a game console or app needs UPnP, check the router's "
                        "port-mapping list for anything you do not recognize"
                    ),
                ],
            )
        ]
