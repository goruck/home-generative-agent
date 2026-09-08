"""Rule: long-lived access tokens older than the configured age."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.const import (
    RECOMMENDED_SENTINEL_HA_TOKEN_STALE_DAYS,
)
from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import POSTURE_COOLDOWN_MINUTES, ha_security, make_finding, plural

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class HaLongLivedTokenStaleRule:
    """
    Old long-lived tokens are standing credentials nobody rotates.

    Home Assistant records refresh-token usage only when an access token is
    exchanged, which a long-lived token does once, at creation. Age is the
    only fact available, so this rule reports tokens older than the
    threshold and says plainly that usage cannot be observed.
    """

    rule_id = "ha_long_lived_token_stale"
    requires = frozenset({ha_cap("long_lived_token_age_days")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def __init__(
        self, stale_days: int = RECOMMENDED_SENTINEL_HA_TOKEN_STALE_DAYS
    ) -> None:
        """Initialize with the age threshold in days."""
        self._stale_days = max(1, int(stale_days))

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding listing every token at or past the age threshold."""
        ages: dict[str, int] = (
            ha_security(snapshot).get("long_lived_token_age_days") or {}
        )
        old = {
            label: int(days)
            for label, days in ages.items()
            if int(days) >= self._stale_days
        }
        if not old:
            return []
        listed = ", ".join(f"{label} ({days} d)" for label, days in sorted(old.items()))
        return [
            make_finding(
                self.rule_id,
                severity="low",
                evidence={"tokens": sorted(old), "threshold_days": self._stale_days},
                display={"token_age_days": dict(sorted(old.items()))},
                summary=(
                    f"{plural(len(old), 'long-lived access token')} older than "
                    f"{self._stale_days} days: {listed}. Home Assistant cannot tell "
                    "whether they are still used; consider rotating them."
                ),
                suggested_actions=[
                    (
                        "Revoke tokens you no longer use from your profile page under "
                        "Security > Long-lived access tokens, and recreate the ones "
                        "you still need"
                    )
                ],
            )
        ]
