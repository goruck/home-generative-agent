"""Rule: a lock, alarm, or entry cover is exposed to a voice assistant."""

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

# The Critical Action PIN guards this integration's conversation agent only.
# Alexa and Google Assistant reach Home Assistant through the cloud
# integration and never see the PIN, so exposure to them is always reported.
_PIN_GATED_ASSISTANTS = frozenset({"conversation"})


class HaSensitiveEntityExposedWithoutPinRule:
    """
    Sensitive entities reachable by voice without a gate.

    One finding per cycle listing every exposure, so a second assistant or a
    second lock is never starved by the per-type cooldown.
    """

    rule_id = "ha_sensitive_entity_exposed_without_pin"
    requires = frozenset(
        {ha_cap("exposed_sensitive_entities"), ha_cap("critical_action_pin_enabled")}
    )
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding covering every ungated exposure."""
        ha = ha_security(snapshot)
        pin = ha.get("critical_action_pin_enabled")
        if pin is None:
            return []
        exposed: dict[str, list[str]] = ha.get("exposed_sensitive_entities") or {}
        exposures: dict[str, list[str]] = {}
        for assistant, entity_ids in sorted(exposed.items()):
            if not entity_ids or (pin and assistant in _PIN_GATED_ASSISTANTS):
                continue
            exposures[assistant] = sorted(entity_ids)
        if not exposures:
            return []
        all_ids = sorted({e for ids in exposures.values() for e in ids})
        parts = []
        actions = []
        for assistant, ids in exposures.items():
            label = _ASSISTANT_LABELS.get(assistant, assistant)
            gate = (
                " with no Critical Action PIN"
                if assistant in _PIN_GATED_ASSISTANTS
                else ""
            )
            parts.append(f"{', '.join(ids)} to {label}{gate}")
            if assistant in _PIN_GATED_ASSISTANTS:
                actions.append(
                    "Enable the Critical Action PIN in the integration's global "
                    "options, or unexpose these entities from Assist"
                )
            else:
                actions.append(
                    f"Unexpose these entities from {label}, or require the "
                    f"{label} app's own PIN for unlocking"
                )
        return [
            make_finding(
                self.rule_id,
                severity="high",
                triggering_entities=all_ids,
                evidence={"exposures": exposures},
                display={"critical_action_pin_enabled": bool(pin)},
                summary=(
                    f"{plural(len(all_ids), 'sensitive entity', 'sensitive entities')} "
                    f"exposed: {'; '.join(parts)}."
                ),
                suggested_actions=actions,
            )
        ]
