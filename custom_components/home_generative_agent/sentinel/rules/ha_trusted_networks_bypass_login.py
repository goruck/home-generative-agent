"""Rule: the trusted-networks auth provider skips the login screen."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import POSTURE_COOLDOWN_MINUTES, ha_security, make_finding

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class HaTrustedNetworksBypassLoginRule:
    """``allow_bypass_login`` makes anyone on the trusted network an admin."""

    rule_id = "ha_trusted_networks_bypass_login"
    requires = frozenset({ha_cap("trusted_networks_bypass_login")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return a finding while login bypass is configured."""
        if not ha_security(snapshot).get("trusted_networks_bypass_login"):
            return []
        return [
            make_finding(
                self.rule_id,
                severity="medium",
                evidence={"allow_bypass_login": True},
                summary=(
                    "The trusted-networks auth provider allows login bypass: "
                    "any device on the trusted network signs in without a "
                    "password."
                ),
                suggested_actions=[
                    (
                        "Set allow_bypass_login to false in the auth provider "
                        "configuration, or narrow trusted_networks to specific hosts"
                    )
                ],
            )
        ]
