"""
Rules: router settings that weaken, or fail to protect, the home network.

Four standing posture checks over settings a router integration exposes
(read from eero today, see ``snapshot/eero.py`` and ``snapshot/router.py``;
any adapter that publishes the same posture keys feeds them):

- ``network_guest_network_idle``: the guest Wi-Fi is on and no guest has
  used it for a while. An unused open door is worth closing.
- ``network_wpa3_disabled``: WPA3 is off. Advisory: older devices may need
  WPA2, so this is a nudge, not an alarm.
- ``network_protection_disabled``: the router's malware and threat blocking
  is off (published only when the account has the feature).
- ``network_ddns_enabled``: dynamic DNS gives the home a fixed public name.
  Informational: fine when it is used, worth turning off when it is not.

Each is a standing condition, so each carries the posture cooldown floor
(one alert a day at most) and one stable identity, like
``network_upnp_enabled``. The switch a setting belongs to is the triggering
entity on both eero tiers, so the engine's per-rule entity exclusions apply
(``SentinelEngine._filter_excluded_findings``) without the rules checking
them again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.const import (
    RECOMMENDED_SENTINEL_NETWORK_GUEST_IDLE_DAYS,
)
from custom_components.home_generative_agent.snapshot.network import posture_cap

from .network_common import POSTURE_COOLDOWN_MINUTES, make_finding, plural, posture

if TYPE_CHECKING:
    from typing import Any

    from custom_components.home_generative_agent.sentinel.models import (
        AnomalyFinding,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


def _source_entity(section: dict[str, Any], key: str) -> list[str]:
    """Return the entity a setting was read from, as the triggering entity."""
    entity_id = section.get(f"{key}_entity_id")
    return [str(entity_id)] if isinstance(entity_id, str) and entity_id else []


class NetworkGuestNetworkIdleRule:
    """The guest network is on and nobody has used it for the configured days."""

    rule_id = "network_guest_network_idle"
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES
    requires = frozenset(
        {posture_cap("guest_network_enabled"), posture_cap("guest_network_idle_days")}
    )

    def __init__(
        self, *, idle_days: int = RECOMMENDED_SENTINEL_NETWORK_GUEST_IDLE_DAYS
    ) -> None:
        """Initialize with the idle threshold in days."""
        self._idle_days = max(1, idle_days)

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one standing finding while the guest network sits unused."""
        section = posture(snapshot)
        idle = section.get("guest_network_idle_days")
        if (
            section.get("guest_network_enabled") is not True
            or not isinstance(idle, int)
            or isinstance(idle, bool)
            or idle < self._idle_days
        ):
            return []
        entities = _source_entity(section, "guest_network_enabled")
        return [
            make_finding(
                self.rule_id,
                severity="low",
                evidence={"guest_network_idle": True},
                display={"idle_days": idle, "threshold_days": self._idle_days},
                summary=(
                    "Your guest Wi-Fi network is on and no guest has connected for "
                    f"{plural(idle, 'day')}. A network nobody uses is one more way in."
                ),
                suggested_actions=[
                    "Turn the guest network off in your router app until you need it",
                    "If you keep it on, change its password after guests leave",
                ],
                triggering_entities=entities,
            )
        ]


class NetworkWpa3DisabledRule:
    """WPA3 is off: an advisory nudge, since older devices may need WPA2."""

    rule_id = "network_wpa3_disabled"
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES
    requires = frozenset({posture_cap("wpa3_enabled")})

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one standing finding while WPA3 is observed off."""
        section = posture(snapshot)
        if section.get("wpa3_enabled") is not False:
            return []
        entities = _source_entity(section, "wpa3_enabled")
        return [
            make_finding(
                self.rule_id,
                severity="low",
                evidence={"wpa3_enabled": False},
                display={},
                summary=(
                    "WPA3 is turned off on your Wi-Fi network, so devices connect with "
                    "the older WPA2 protection. WPA3 resists password guessing better."
                ),
                suggested_actions=[
                    (
                        "Turn on WPA3 in your router app if your devices support it; "
                        "older devices may fail to connect, so check them afterwards"
                    ),
                ],
                triggering_entities=entities,
            )
        ]


class NetworkProtectionDisabledRule:
    """The router's malware and threat blocking is available but switched off."""

    rule_id = "network_protection_disabled"
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES
    requires = frozenset({posture_cap("malware_blocking_enabled")})

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one standing finding while threat blocking is observed off."""
        section = posture(snapshot)
        if section.get("malware_blocking_enabled") is not False:
            return []
        ad_blocking = section.get("ad_blocking_enabled")
        summary = (
            "Your router's malware and threat blocking is turned off, so devices "
            "on your network are not stopped from reaching known malicious sites."
        )
        if ad_blocking is False:
            summary += " Ad blocking is off as well."
        entities = _source_entity(section, "malware_blocking_enabled")
        return [
            make_finding(
                self.rule_id,
                severity="medium",
                evidence={"malware_blocking_enabled": False},
                display={"ad_blocking_enabled": ad_blocking},
                summary=summary,
                suggested_actions=["Turn on advanced security in your router app"],
                triggering_entities=entities,
            )
        ]


class NetworkDdnsEnabledRule:
    """Dynamic DNS gives the home a fixed public hostname."""

    rule_id = "network_ddns_enabled"
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES
    requires = frozenset({posture_cap("ddns_enabled")})

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one standing finding while dynamic DNS is observed on."""
        section = posture(snapshot)
        if section.get("ddns_enabled") is not True:
            return []
        entities = _source_entity(section, "ddns_enabled")
        return [
            make_finding(
                self.rule_id,
                severity="low",
                evidence={"ddns_enabled": True},
                display={},
                summary=(
                    "Dynamic DNS is on at your router, so your home network can be "
                    "found from the internet by a fixed name even when its address "
                    "changes. That is useful for remote access and unneeded otherwise."
                ),
                suggested_actions=[
                    (
                        "Keep it if you use it for remote access or a VPN; otherwise "
                        "turn dynamic DNS off in your router app"
                    ),
                ],
                triggering_entities=entities,
            )
        ]
