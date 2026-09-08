"""Rule: a Supervisor add-on runs with protection mode off."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import POSTURE_COOLDOWN_MINUTES, ha_security, make_finding

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class HaAddonUnprotectedRule:
    """Protection mode off gives an add-on full access to the host."""

    rule_id = "ha_addon_unprotected"
    requires = frozenset({ha_cap("addons_unprotected")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding per unprotected running add-on."""
        ha = ha_security(snapshot)
        names: dict[str, str] = ha.get("addon_names") or {}
        findings: list[AnomalyFinding] = []
        for slug in sorted(ha.get("addons_unprotected") or []):
            name = names.get(slug, slug)
            findings.append(
                make_finding(
                    self.rule_id,
                    severity="high",
                    evidence={"addon_slug": slug, "addon_name": name},
                    summary=(
                        f"Add-on {name} is running with protection mode off, "
                        "which grants it full access to the host system."
                    ),
                    suggested_actions=[
                        (
                            "Turn protection mode back on in the add-on's Info tab "
                            "unless the add-on's documentation requires it off."
                        )
                    ],
                )
            )
        return findings
