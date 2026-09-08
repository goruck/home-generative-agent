"""Rule: HTTP hardening is off (IP banning disabled)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import POSTURE_COOLDOWN_MINUTES, ha_security, make_finding

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class HaHttpProxyMisconfiguredRule:
    """
    HTTP settings that weaken brute-force protection.

    Home Assistant validates ``use_x_forwarded_for`` and ``trusted_proxies``
    as a pair, so the classic "forwarded headers without trusted proxies"
    misconfiguration cannot start; what remains observable is IP banning
    turned off or the login-attempt threshold disabled.
    """

    rule_id = "ha_http_proxy_misconfigured"
    requires = frozenset({ha_cap("http_ip_ban_enabled")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return a finding while IP banning is off."""
        ha = ha_security(snapshot)
        # Fire only on an observed False: an absent key means the HTTP
        # settings could not be read, which is a missing capability, not a
        # misconfiguration.
        if ha.get("http_ip_ban_enabled") is not False:
            return []
        threshold = ha.get("http_login_attempts_threshold")
        return [
            make_finding(
                self.rule_id,
                severity="medium",
                evidence={
                    "reason": "ip_ban_disabled",
                    "ip_ban_enabled": False,
                    "login_attempts_threshold": threshold,
                    "trusted_proxies_configured": bool(
                        ha.get("http_trusted_proxies_configured")
                    ),
                },
                summary=(
                    "IP banning after failed logins is disabled, so repeated "
                    "password guesses are never blocked."
                ),
                suggested_actions=[
                    (
                        "Set http: ip_ban_enabled: true and a login_attempts_threshold "
                        "in configuration.yaml."
                    )
                ],
            )
        ]
