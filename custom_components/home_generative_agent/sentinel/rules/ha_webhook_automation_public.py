"""Rule: an automation has a webhook trigger that is not local-only."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import POSTURE_COOLDOWN_MINUTES, ha_security, make_finding

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class HaWebhookAutomationPublicRule:
    """Public webhooks are reachable from the internet whenever remote access is on."""

    rule_id = "ha_webhook_automation_public"
    requires = frozenset({ha_cap("webhook_automations_public")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding per automation with a public webhook trigger."""
        ha = ha_security(snapshot)
        critical = set(ha.get("webhook_automations_critical") or [])
        findings: list[AnomalyFinding] = []
        for entity_id in sorted(ha.get("webhook_automations_public") or []):
            is_critical = entity_id in critical
            findings.append(
                make_finding(
                    self.rule_id,
                    severity="high" if is_critical else "medium",
                    triggering_entities=[entity_id],
                    evidence={
                        "entity_id": entity_id,
                        "calls_critical_action": is_critical,
                    },
                    summary=(
                        f"Automation {entity_id} accepts webhooks from outside "
                        "the local network"
                        + (" and can unlock or open an entry." if is_critical else ".")
                    ),
                    suggested_actions=[
                        (
                            "Set the webhook trigger to 'Only accessible from the "
                            "local network' unless an external service must call it."
                        )
                    ],
                )
            )
        return findings
