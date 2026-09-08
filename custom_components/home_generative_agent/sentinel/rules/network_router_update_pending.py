"""Rule: a router / gateway integration reports firmware waiting to install."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import posture_cap

from .network_common import POSTURE_COOLDOWN_MINUTES, make_finding, posture

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class NetworkRouterUpdatePendingRule:
    """Router firmware updates close the holes attackers reach first."""

    rule_id = "network_router_update_pending"
    requires = frozenset({posture_cap("router_update_pending")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding per router update entity in state ``on``."""
        section = posture(snapshot)
        if not section.get("router_update_pending"):
            return []
        by_id = {e["entity_id"]: e for e in snapshot["entities"]}
        findings: list[AnomalyFinding] = []
        for entity_id in sorted(section.get("router_update_entities") or []):
            entity = by_id.get(entity_id)
            attrs = entity["attributes"] if entity else {}
            name = (entity["friendly_name"] if entity else None) or entity_id
            installed = attrs.get("installed_version")
            latest = attrs.get("latest_version")
            versions = f" ({installed} -> {latest})" if installed and latest else ""
            findings.append(
                make_finding(
                    self.rule_id,
                    severity="medium",
                    triggering_entities=[entity_id],
                    evidence={
                        "entity_id": entity_id,
                        "friendly_name": name,
                        "platform": (entity or {}).get("platform"),
                        "installed_version": installed,
                        "latest_version": latest,
                    },
                    summary=f"Router firmware update available for {name}{versions}.",
                    suggested_actions=[
                        (
                            "Install the update from the router's app or from the "
                            "update entity in Settings > System > Updates."
                        )
                    ],
                )
            )
        return findings
