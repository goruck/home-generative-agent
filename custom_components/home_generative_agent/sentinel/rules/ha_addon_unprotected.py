"""Rule: Supervisor add-ons run with protection mode off."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import POSTURE_COOLDOWN_MINUTES, ha_security, make_finding, plural

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
        """Return one finding covering every unprotected running add-on."""
        ha = ha_security(snapshot)
        slugs = sorted(ha.get("addons_unprotected") or [])
        if not slugs:
            return []
        names: dict[str, str] = ha.get("addon_names") or {}
        listed = ", ".join(str(names.get(slug, slug)) for slug in slugs)
        return [
            make_finding(
                self.rule_id,
                severity="high",
                evidence={"addons": slugs},
                display={"addon_names": {s: names.get(s, s) for s in slugs}},
                summary=(
                    f"{plural(len(slugs), 'add-on')} running with protection mode "
                    f"off, which grants full access to the host system: {listed}."
                ),
                suggested_actions=[
                    (
                        "Turn protection mode back on in each add-on's Info tab "
                        "unless the add-on's documentation requires it off"
                    )
                ],
            )
        ]
