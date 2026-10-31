"""Rule: a lock, alarm, or entry cover is exposed to a voice assistant."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import (
    POSTURE_COOLDOWN_MINUTES,
    ha_security,
    listed,
    make_finding,
    noun,
    plural,
)

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
# Assist exposure is shared by every conversation agent, so the PIN silences
# it only when every Assist pipeline runs on this integration's agent.
_PIN_GATED_ASSISTANTS = frozenset({"conversation"})
# Held members named per group. What a group holds comes after every
# exposure and its assistant and PIN wording, so the push's 220-character cut
# takes the detail, never the warning (#793).
_MAX_HELD_SHOWN = 3


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
        outside_raw = ha.get("assist_agents_outside_pin")
        outside: list[str] | None = (
            sorted(str(a) for a in outside_raw) if outside_raw is not None else None
        )
        # Unknown pipeline coverage never counts as covered.
        assist_gated = bool(pin) and outside == []
        exposed: dict[str, list[str]] = ha.get("exposed_sensitive_entities") or {}
        # An exposed group is followed by what it holds, so the reader sees
        # why "All Covers" counts (#793).
        held_raw = ha.get("exposed_sensitive_groups") or {}
        held = {
            str(group_id): sorted(str(m) for m in members)
            for group_id, members in held_raw.items()
        }
        exposures: dict[str, list[str]] = {}
        for assistant, entity_ids in sorted(exposed.items()):
            if not entity_ids or (assist_gated and assistant in _PIN_GATED_ASSISTANTS):
                continue
            exposures[assistant] = sorted(entity_ids)
        if not exposures:
            return []
        all_ids = sorted({e for ids in exposures.values() for e in ids})
        groups = {
            entity_id: held[entity_id] for entity_id in all_ids if entity_id in held
        }
        holds = "".join(
            f" {group_id} holds {listed(members, _MAX_HELD_SHOWN)}."
            for group_id, members in groups.items()
        )
        parts = []
        actions = []
        for assistant, ids in exposures.items():
            label = _ASSISTANT_LABELS.get(assistant, assistant)
            if assistant not in _PIN_GATED_ASSISTANTS:
                gate = ""
            elif not pin:
                gate = " with no Critical Action PIN"
            elif outside:
                gate = (
                    ", where the Critical Action PIN does not cover the "
                    f"{noun(len(outside), 'pipeline agent')} "
                    f"{', '.join(outside)}"
                )
            else:
                gate = ", where Critical Action PIN coverage could not be verified"
            parts.append(f"{', '.join(ids)} to {label}{gate}")
            if assistant in _PIN_GATED_ASSISTANTS and pin:
                actions.append(
                    "Point every Assist pipeline at this integration's "
                    "conversation agent so the Critical Action PIN applies, or "
                    "unexpose these entities from Assist"
                )
            elif assistant in _PIN_GATED_ASSISTANTS:
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
                # What a group holds is display: the exposure is the group's,
                # and a member briefly losing its class must not mint a new
                # id and a new push. Present only when there are groups.
                display={
                    "critical_action_pin_enabled": bool(pin),
                    "assist_agents_outside_pin": outside,
                    **({"groups": groups} if groups else {}),
                },
                summary=(
                    f"{plural(len(all_ids), 'sensitive entity', 'sensitive entities')} "
                    f"exposed: {'; '.join(parts)}.{holds}"
                ),
                suggested_actions=actions,
            )
        ]
