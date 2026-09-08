"""Rule: Home Assistant Cloud remote UI is enabled (informational)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import POSTURE_COOLDOWN_MINUTES, ha_security, make_finding

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class HaCloudRemoteUiEnabledRule:
    """Remote access is on: every other finding here is reachable from outside."""

    rule_id = "ha_cloud_remote_ui_enabled"
    requires = frozenset({ha_cap("cloud_remote_ui_enabled")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return a low-severity finding while remote UI is enabled."""
        if not ha_security(snapshot).get("cloud_remote_ui_enabled"):
            return []
        return [
            make_finding(
                self.rule_id,
                severity="low",
                evidence={"cloud_remote_ui_enabled": True},
                summary=(
                    "Home Assistant Cloud remote access is enabled, so this "
                    "instance is reachable from the internet."
                ),
                suggested_actions=[
                    (
                        "Keep remote access on only while you need it, and make "
                        "sure every user has a strong password and multi-factor "
                        "authentication."
                    )
                ],
            )
        ]
