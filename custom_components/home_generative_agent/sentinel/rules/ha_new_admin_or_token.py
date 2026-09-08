"""Rule: a new admin user, long-lived token, or token seen from a new address."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import ha_security, make_finding

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class HaNewAdminOrTokenRule:
    """
    Auth changes against the persistent inventory.

    The inventory commits after every run, so each change is reported once;
    no cooldown floor is needed and none is set.
    """

    rule_id = "ha_new_admin_or_token"
    requires = frozenset(
        {
            ha_cap("new_admin_users"),
            ha_cap("new_long_lived_tokens"),
            ha_cap("refresh_tokens_from_new_ip"),
        }
    )

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding per new admin, new token, or new token address."""
        ha = ha_security(snapshot)
        findings: list[AnomalyFinding] = []
        for user in ha.get("new_admin_users") or []:
            findings.append(  # noqa: PERF401 - three parallel loops read better
                make_finding(
                    self.rule_id,
                    severity="high",
                    evidence={"change": "new_admin_user", "user": user},
                    summary=f"New administrator account: {user}.",
                    suggested_actions=[
                        (
                            "If you did not create this administrator, remove the "
                            "account under Settings > People and change your passwords."
                        )
                    ],
                )
            )
        for token in ha.get("new_long_lived_tokens") or []:
            findings.append(  # noqa: PERF401
                make_finding(
                    self.rule_id,
                    severity="high",
                    evidence={"change": "new_long_lived_token", "token": token},
                    summary=f"New long-lived access token: {token}.",
                    suggested_actions=[
                        (
                            "If you did not create this token, revoke it from your "
                            "profile page (Security > Long-lived access tokens)."
                        )
                    ],
                )
            )
        for token in ha.get("refresh_tokens_from_new_ip") or []:
            findings.append(  # noqa: PERF401
                make_finding(
                    self.rule_id,
                    severity="medium",
                    confidence=0.6,
                    evidence={"change": "token_from_new_address", "token": token},
                    summary=(
                        f"Long-lived access token {token} was used from an "
                        "address not seen before."
                    ),
                    suggested_actions=[
                        (
                            "If nothing of yours moved to a new network, revoke the "
                            "token from your profile page."
                        )
                    ],
                )
            )
        return findings
