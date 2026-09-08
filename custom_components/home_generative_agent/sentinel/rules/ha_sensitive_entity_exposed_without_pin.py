"""Rule: a lock, alarm, or entry cover is exposed to an assistant with no PIN."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import POSTURE_COOLDOWN_MINUTES, ha_security, make_finding, plural

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

_ASSISTANT_LABELS = {
    "conversation": "Assist",
    "cloud.alexa": "Alexa",
    "cloud.google_assistant": "Google Assistant",
}


class HaSensitiveEntityExposedWithoutPinRule:
    """Sensitive entities reachable by voice with no critical-action PIN."""

    rule_id = "ha_sensitive_entity_exposed_without_pin"
    requires = frozenset(
        {ha_cap("exposed_sensitive_entities"), ha_cap("critical_action_pin_enabled")}
    )
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding per assistant that exposes sensitive entities."""
        ha = ha_security(snapshot)
        # An absent PIN capability is "unknown", never "off".
        if ha.get("critical_action_pin_enabled") is not False:
            return []
        findings: list[AnomalyFinding] = []
        exposed: dict[str, list[str]] = ha.get("exposed_sensitive_entities") or {}
        for assistant, entity_ids in sorted(exposed.items()):
            if not entity_ids:
                continue
            label = _ASSISTANT_LABELS.get(assistant, assistant)
            ids = sorted(entity_ids)
            findings.append(
                make_finding(
                    self.rule_id,
                    severity="high",
                    triggering_entities=ids,
                    evidence={
                        "assistant": assistant,
                        "entity_ids": ids,
                        "entity_count": len(ids),
                        "critical_action_pin_enabled": False,
                    },
                    summary=(
                        f"{plural(len(ids), 'sensitive entity', 'sensitive entities')} "
                        f"exposed to {label} with no Critical Action PIN: "
                        f"{', '.join(ids)}."
                    ),
                    suggested_actions=[
                        (
                            "Enable the Critical Action PIN in the integration's "
                            "global options, or unexpose these entities from the "
                            "assistant."
                        )
                    ],
                )
            )
        return findings
