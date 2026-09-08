"""Rule: Home Assistant reported failed login attempts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import (
    LOGIN_NOTIFICATION_ID,
    ha_cap,
)

from .network_common import POSTURE_COOLDOWN_MINUTES, ha_security, make_finding

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class HaFailedLoginsRule:
    """The ``http-login`` persistent notification is present."""

    rule_id = "ha_failed_logins"
    requires = frozenset({ha_cap("failed_login_notification_present")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return a finding while the failed-login notification stands."""
        if not ha_security(snapshot).get("failed_login_notification_present"):
            return []
        return [
            make_finding(
                self.rule_id,
                severity="medium",
                evidence={"notification_id": LOGIN_NOTIFICATION_ID},
                summary=(
                    "Home Assistant recorded failed login attempts; see the "
                    "'Login attempt failed' notification for the source address."
                ),
                suggested_actions=[
                    (
                        "Check the address in the notification; if it is not yours, "
                        "confirm IP banning is enabled and consider putting Home "
                        "Assistant behind a VPN"
                    )
                ],
            )
        ]
