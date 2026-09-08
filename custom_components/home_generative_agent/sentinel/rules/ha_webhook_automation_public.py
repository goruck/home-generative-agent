"""Rule: automations with webhook triggers that are not local-only."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import POSTURE_COOLDOWN_MINUTES, ha_security, make_finding, plural

if TYPE_CHECKING:
    from collections.abc import Callable

    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class HaWebhookAutomationPublicRule:
    """Public webhooks are reachable from the internet whenever remote access is on."""

    rule_id = "ha_webhook_automation_public"
    requires = frozenset({ha_cap("webhook_automations_public")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def __init__(
        self, is_entity_excluded: Callable[[str, str], bool] | None = None
    ) -> None:
        """
        Initialize the rule.

        ``is_entity_excluded(entity_id, anomaly_type)`` mirrors the engine's
        per-rule entity exclusions; automations the user excluded are dropped
        before aggregation, so one exclusion never silences the whole finding.
        """
        self._is_entity_excluded = is_entity_excluded

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding covering every public webhook automation."""
        ha = ha_security(snapshot)
        critical = set(ha.get("webhook_automations_critical") or [])
        public = sorted(
            entity_id
            for entity_id in ha.get("webhook_automations_public") or []
            if self._is_entity_excluded is None
            or not self._is_entity_excluded(entity_id, self.rule_id)
        )
        if not public:
            return []
        critical_public = sorted(e for e in public if e in critical)
        listed = ", ".join(
            f"{e} (can unlock or open an entry)" if e in critical else e for e in public
        )
        return [
            make_finding(
                self.rule_id,
                severity="high" if critical_public else "medium",
                triggering_entities=public,
                evidence={"automations": public, "critical": critical_public},
                summary=(
                    f"{plural(len(public), 'automation')} accepting webhooks from "
                    f"outside the local network: {listed}."
                ),
                suggested_actions=[
                    (
                        "Set each webhook trigger to 'Only accessible from the local "
                        "network' unless an external service must call it"
                    )
                ],
            )
        ]
