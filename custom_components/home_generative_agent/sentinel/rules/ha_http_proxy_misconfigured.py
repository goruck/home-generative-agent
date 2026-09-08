"""Rule: HTTP hardening is off (IP banning disabled)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import POSTURE_COOLDOWN_MINUTES, ha_security, make_finding

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import (
        AnomalyFinding,
        Severity,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class HaHttpProxyMisconfiguredRule:
    """
    HTTP settings that weaken brute-force protection.

    Home Assistant validates ``use_x_forwarded_for`` and ``trusted_proxies``
    as a pair, so the classic "forwarded headers without trusted proxies"
    misconfiguration cannot start; what remains observable is IP banning
    turned off (medium) or, Home Assistant's shipped default, banning on with
    no login-attempt threshold so it never triggers (low).
    """

    rule_id = "ha_http_proxy_misconfigured"
    requires = frozenset({ha_cap("http_ip_ban_enabled")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return a finding while IP banning cannot block anyone."""
        ha = ha_security(snapshot)
        # Fire only on observed facts: an absent key means the HTTP settings
        # could not be read, which is a missing capability, not a
        # misconfiguration.
        ban_enabled = ha.get("http_ip_ban_enabled")
        if ban_enabled is None:
            return []
        threshold = ha.get("http_login_attempts_threshold")
        threshold_set = isinstance(threshold, int) and threshold >= 1
        if ban_enabled and threshold_set:
            return []
        if not ban_enabled:
            reason = "ip_ban_disabled"
            severity: Severity = "medium"
            summary = (
                "IP banning after failed logins is disabled, so repeated "
                "password guesses are never blocked."
            )
        else:
            # Home Assistant's shipped default: banning is on but no
            # login_attempts_threshold is set, so a ban never triggers.
            reason = "no_login_threshold"
            severity = "low"
            summary = (
                "No login-attempt threshold is configured, so IP banning "
                "never triggers after failed logins."
            )
        return [
            make_finding(
                self.rule_id,
                severity=severity,
                evidence={
                    "reason": reason,
                    "ip_ban_enabled": bool(ban_enabled),
                    "trusted_proxies_configured": bool(
                        ha.get("http_trusted_proxies_configured")
                    ),
                },
                display={"login_attempts_threshold": threshold},
                summary=summary,
                suggested_actions=[
                    (
                        "Set ip_ban_enabled to true and login_attempts_threshold "
                        "to a small number such as 5 in the http section of your "
                        "configuration"
                    )
                ],
            )
        ]
