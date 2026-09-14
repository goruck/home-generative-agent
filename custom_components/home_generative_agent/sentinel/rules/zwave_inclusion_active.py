"""Rule: a Z-Wave controller is in inclusion mode."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import radio_cap

from .network_common import anyone_home, make_finding, radio_posture

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class ZwaveInclusionActiveRule:
    """
    The Z-Wave JS controller is including devices.

    Observed only when a Sentinel run lands inside the inclusion window: the
    integration exposes no entity for it, so there is nothing to wake on.
    SmartStart listening is not counted because only provisioned devices can
    join in that mode.
    """

    rule_id = "zwave_inclusion_active"
    requires = frozenset({radio_cap("posture.zwave_inclusion_active")})
    cooldown_minutes = 0

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return a finding while the controller is including."""
        if not radio_posture(snapshot).get("zwave_inclusion_active"):
            return []
        away = not anyone_home(snapshot)
        context = " while nobody is home" if away else ""
        return [
            make_finding(
                self.rule_id,
                severity="high" if away else "medium",
                evidence={"zwave_inclusion_active": True},
                summary=f"A Z-Wave controller is including new devices{context}.",
                suggested_actions=[
                    (
                        "Stop inclusion in the Z-Wave JS integration unless you are "
                        "adding a device right now"
                    )
                ],
            )
        ]
