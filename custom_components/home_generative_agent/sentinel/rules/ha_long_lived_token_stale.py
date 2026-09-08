"""Rule: a long-lived access token has not been used for a long time."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.const import (
    RECOMMENDED_SENTINEL_HA_TOKEN_STALE_DAYS,
)
from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import POSTURE_COOLDOWN_MINUTES, ha_security, make_finding

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class HaLongLivedTokenStaleRule:
    """Unused tokens are standing credentials with no one watching them."""

    rule_id = "ha_long_lived_token_stale"
    requires = frozenset({ha_cap("long_lived_tokens_unused_days")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def __init__(
        self, stale_days: int = RECOMMENDED_SENTINEL_HA_TOKEN_STALE_DAYS
    ) -> None:
        """Initialize with the staleness threshold in days."""
        self._stale_days = max(1, int(stale_days))

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding per token idle for at least the threshold."""
        unused: dict[str, int] = (
            ha_security(snapshot).get("long_lived_tokens_unused_days") or {}
        )
        findings: list[AnomalyFinding] = []
        for label, days in sorted(unused.items()):
            if int(days) < self._stale_days:
                continue
            findings.append(
                make_finding(
                    self.rule_id,
                    severity="low",
                    evidence={
                        "token": label,
                        "unused_days": int(days),
                        "threshold_days": self._stale_days,
                    },
                    summary=(
                        f"Long-lived access token {label} has not been used for "
                        f"{int(days)} days (threshold {self._stale_days})."
                    ),
                    suggested_actions=[
                        (
                            "Revoke tokens you no longer use from your profile page "
                            "(Security > Long-lived access tokens)."
                        )
                    ],
                )
            )
        return findings
