"""Rule: a new admin user or a new long-lived access token appeared."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import ha_security, make_finding, noun

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class HaNewAdminOrTokenRule:
    """
    Auth changes against the persistent inventory.

    One finding per cycle carrying every change, so a new admin and a new
    token in the same cycle are reported together rather than one starving
    the other under the per-type cooldown. The engine commits the inventory
    only after this finding was delivered; a suppressed finding leaves the
    changes uncommitted so they are reported again on a later run.
    """

    rule_id = "ha_new_admin_or_token"
    requires = frozenset({ha_cap("new_admin_users"), ha_cap("new_long_lived_tokens")})
    cooldown_minutes = 0

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding covering every new admin and new token."""
        ha = ha_security(snapshot)
        admins = sorted(ha.get("new_admin_users") or [])
        tokens = sorted(ha.get("new_long_lived_tokens") or [])
        if not admins and not tokens:
            return []
        parts: list[str] = []
        actions: list[str] = []
        if admins:
            parts.append(
                f"new administrator {noun(len(admins), 'account')}: {', '.join(admins)}"
            )
            actions.append(
                "If you did not create this administrator, remove the account "
                "under Settings > People and change your passwords"
            )
        if tokens:
            parts.append(
                f"new long-lived access {noun(len(tokens), 'token')}: "
                f"{', '.join(tokens)}"
            )
            actions.append(
                "If you did not create this token, revoke it from your profile "
                "page under Security > Long-lived access tokens"
            )
        summary = "; ".join(parts)
        return [
            make_finding(
                self.rule_id,
                severity="high",
                evidence={"new_admin_users": admins, "new_long_lived_tokens": tokens},
                summary=summary[0].upper() + summary[1:] + ".",
                suggested_actions=actions,
            )
        ]
