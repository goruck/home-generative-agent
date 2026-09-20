"""Rule: an untrusted guest Wi-Fi client is connected while away or at night."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.sentinel.network_inventory import (
    client_key,
)
from custom_components.home_generative_agent.snapshot.network import (
    CAP_CLIENTS,
    CAP_GUEST_CLIENTS,
    CAP_NEW_CLIENTS,
)

from .network_common import (
    POSTURE_COOLDOWN_MINUTES,
    anyone_home,
    clients,
    is_night,
    listed,
    make_finding,
    new_clients,
    plural,
)
from .network_unknown_device_joined import describe_client

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any

    from custom_components.home_generative_agent.sentinel.models import (
        AnomalyFinding,
        Severity,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

# A client's first day belongs to ``network_unknown_device_joined``: that rule
# has just told the owner about it, and a second push for the same arrival
# would only be noise.
FIRST_DAY_HOLDOFF = timedelta(days=1)


class NetworkGuestClientPresentRule:
    """
    Guest Wi-Fi clients nobody vouched for, connected when no guest is expected.

    A guest on the guest network while someone is home in daytime is what the
    network is for, so the rule only looks while nobody is home (medium) or
    at night (low: an overnight visitor is the common case). It reports only
    clients the device inventory holds as *not trusted*: devices the owner
    parked on the guest network were trusted when the inventory was
    established, and a visitor's phone stops being reported once the owner
    taps Trust device. What is left is a device that was announced when it
    joined, was never vouched for, and keeps coming back: a neighbour with
    the guest password, or a device left behind.

    A client still owed its ``network_unknown_device_joined`` alert, or first
    seen less than a day ago, is left to that rule. The condition lasts for
    hours, so the rule carries the one-day cooldown floor of the posture
    rules.
    """

    rule_id = "network_guest_client_present"
    requires = frozenset({CAP_CLIENTS, CAP_NEW_CLIENTS, CAP_GUEST_CLIENTS})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def __init__(
        self, *, is_entity_excluded: Callable[[str, str], bool] | None = None
    ) -> None:
        """Initialize with the per-rule entity exclusions."""
        self._is_entity_excluded = is_entity_excluded

    def _excluded(self, client: Mapping[str, Any]) -> bool:
        entity_id = client.get("tracker_entity_id")
        if not entity_id or self._is_entity_excluded is None:
            return False
        return self._is_entity_excluded(str(entity_id), self.rule_id)

    @staticmethod
    def _past_first_day(client: Mapping[str, Any], now: Any) -> bool:
        first_seen = dt_util.parse_datetime(str(client.get("first_seen") or ""))
        if first_seen is None:
            return False
        return dt_util.as_utc(now) - dt_util.as_utc(first_seen) >= FIRST_DAY_HOLDOFF

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding naming every untrusted guest client connected now."""
        away = not anyone_home(snapshot)
        night = is_night(snapshot)
        if not (away or night):
            return []
        owed = new_clients(snapshot)
        now = dt_util.parse_datetime(snapshot["generated_at"]) or dt_util.utcnow()
        guests = [
            c
            for c in clients(snapshot)
            if c.get("is_guest") is True
            and c.get("connected")
            # Only an explicit "not trusted": a client without a verdict has
            # no inventory row yet and belongs to the unknown-device rule.
            and c.get("trusted") is False
            and c.get("key") not in owed
            and self._past_first_day(c, now)
            and not self._excluded(c)
        ]
        if not guests:
            return []
        severity: Severity = "medium" if away else "low"
        count = len(guests)
        context = "while nobody is home" if away else "at night"
        verb = "is" if count == 1 else "are"
        return [
            make_finding(
                self.rule_id,
                severity=severity,
                evidence={"client_keys": sorted(c["key"] for c in guests)},
                display={
                    # Inventory keys, so the Trust button resolves them.
                    "device_ids": sorted(client_key(c["key"]) for c in guests),
                    "names": [describe_client(c) for c in guests],
                    "anyone_home": not away,
                    "is_night": night,
                },
                summary=(
                    f"{plural(count, 'guest Wi-Fi device')} you have not trusted "
                    f"{verb} connected {context}: "
                    f"{listed([describe_client(c) for c in guests])}."
                ),
                suggested_actions=[
                    (
                        "If you do not recognize it, change the guest Wi-Fi "
                        "password in your router app"
                    ),
                    "Tap Trust device if it belongs to a guest you expect",
                    "Turn the guest network off when no guests are staying",
                ],
                triggering_entities=sorted(
                    str(c["tracker_entity_id"])
                    for c in guests
                    if c.get("tracker_entity_id")
                ),
            )
        ]
