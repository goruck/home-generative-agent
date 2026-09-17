"""Rule: the home's public IP address differs from the previous run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import posture_cap

from .network_common import make_finding, posture

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class NetworkPublicIpChangedRule:
    """Informational: remote access that relies on a fixed address may break."""

    rule_id = "network_public_ip_changed"
    requires = frozenset({posture_cap("public_ip_changed")})
    cooldown_minutes = 0

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding per new address; identity is the pseudonymized key."""
        section = posture(snapshot)
        if not section.get("public_ip_changed"):
            return []
        key = str(section.get("public_ip_key") or "")
        if not key:
            return []
        entity_id = section.get("public_ip_entity_id")
        entities = [str(entity_id)] if entity_id else []
        return [
            make_finding(
                self.rule_id,
                severity="low",
                triggering_entities=entities,
                evidence={"public_ip_key": key},
                display={
                    "previous_public_ip_key": section.get("public_ip_previous_key")
                },
                summary=(
                    "Your home's public IP address changed since the last check. "
                    "Remote access that relies on the old address (dynamic DNS, a "
                    "VPN endpoint, a port forward you reach by IP) may stop working "
                    "until it is updated."
                ),
                suggested_actions=[
                    (
                        "Nothing to do unless you rely on a fixed public address; "
                        "then update your dynamic DNS or remote access settings"
                    )
                ],
            )
        ]
