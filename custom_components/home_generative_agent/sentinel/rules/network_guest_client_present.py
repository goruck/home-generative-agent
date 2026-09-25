"""Rule: an untrusted guest Wi-Fi client is connected while nobody is home."""

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
    client_excluded,
    client_known_for,
    clients,
    listed,
    make_finding,
    nobody_home_for_sure,
    plural,
    trust_action,
)
from .network_unknown_device_joined import describe_client

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any

    from custom_components.home_generative_agent.sentinel.models import (
        AnomalyFinding,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

# A client's first day belongs to ``network_unknown_device_joined``: that rule
# has just told the owner about it, and a second push for the same arrival
# would only be noise. After a day the client is judged here whether or not
# that alert was ever settled (an exclusion on the other rule, or days of
# quiet hours, must not hide the client from this one).
FIRST_DAY_HOLDOFF = timedelta(days=1)


def _describe(client: Mapping[str, Any]) -> str:
    """Return the client's description, marked when its name is a known one."""
    text = describe_client(client)
    if client.get("hostname_trusted"):
        # A hint, not a verdict: the name is self-reported.
        return f"{text}, its name matches a device you trust"
    return text


class NetworkGuestClientPresentRule:
    """
    Guest Wi-Fi clients nobody vouched for, connected while nobody is home.

    A guest on the guest network while someone is home is what the network
    is for, day or night, so the rule looks only while every tracked person
    is positively away (``nobody_home_for_sure``): an install without
    ``person`` entities, or a presence outage, never reads as "away". Night
    is deliberately not a trigger: the derived flag follows the sun, so it
    would fire on a winter dinner guest.

    Only clients the device inventory holds as *not trusted* are reported:
    devices the owner parked on the guest network were trusted when the
    inventory was established, and a visitor's phone stops being reported
    once the owner taps Trust device. What is left is a device that was
    announced when it joined, was never vouched for, and keeps coming back:
    a neighbour with the guest password, or a device left behind. A client
    the adapter auto-trusts this run is skipped (the inventory records that
    after the run). A rotated random address whose name a trusted row carries
    is NOT skipped, only marked: the name is whatever the device advertises,
    so skipping would let anyone hide by naming a device after a trusted one.

    The Trust device button is offered for a single device only (see
    ``notifier._TRUST_ONE_DEVICE_TYPES``): the push shows at most 220
    characters, and one tap must never trust a device it did not name.

    The condition lasts for hours, so the rule carries the one-day cooldown
    floor of the posture rules. Two accepted limits follow from that and from
    the first-day holdoff, both recorded in TODOS.md: the floor is per type,
    so a second guest that comes and goes inside the day after an alert about
    another is not reported by this rule; and a device that rotates its
    address more often than daily never gets a day old here, and is covered
    by the unknown-device alert each new address raises.
    """

    rule_id = "network_guest_client_present"
    requires = frozenset({CAP_CLIENTS, CAP_NEW_CLIENTS, CAP_GUEST_CLIENTS})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def __init__(
        self, *, is_entity_excluded: Callable[[str, str], bool] | None = None
    ) -> None:
        """Initialize with the per-rule entity exclusions."""
        self._is_entity_excluded = is_entity_excluded

    def _reportable(self, client: Mapping[str, Any], now: Any) -> bool:
        return (
            client.get("is_guest") is True
            and bool(client.get("connected"))
            # Only an explicit "not trusted": a client without a verdict has
            # no inventory row yet and belongs to the unknown-device rule.
            and client.get("trusted") is False
            and not client.get("auto_trust")
            and client_known_for(client, now, FIRST_DAY_HOLDOFF)
            and not client_excluded(client, self.rule_id, self._is_entity_excluded)
        )

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding naming every untrusted guest client connected now."""
        if not nobody_home_for_sure(snapshot):
            return []
        now = dt_util.parse_datetime(snapshot["generated_at"]) or dt_util.utcnow()
        guests = sorted(
            (c for c in clients(snapshot) if self._reportable(c, now)),
            key=lambda c: str(c["key"]),
        )
        if not guests:
            return []
        count = len(guests)
        names = [_describe(c) for c in guests]
        trust_hint = trust_action(
            count, "Tap Trust device if it belongs to a guest you expect"
        )
        return [
            make_finding(
                self.rule_id,
                severity="medium",
                evidence={"client_keys": [c["key"] for c in guests]},
                display={
                    # Inventory keys, so the Trust button resolves them; in
                    # the same order as ``names``.
                    "device_ids": [client_key(c["key"]) for c in guests],
                    "names": names,
                    "randomized_known": [
                        c["key"] for c in guests if c.get("hostname_trusted")
                    ],
                },
                summary=(
                    f"{plural(count, 'guest Wi-Fi device')} you have not trusted "
                    f"{'is' if count == 1 else 'are'} connected while nobody is "
                    f"home: {listed(names)}."
                ),
                suggested_actions=[
                    (
                        "If you do not recognize it, change the guest Wi-Fi "
                        "password in your router app"
                    ),
                    trust_hint,
                    "Turn the guest network off when no guests are staying",
                ],
                triggering_entities=sorted(
                    str(c["tracker_entity_id"])
                    for c in guests
                    if c.get("tracker_entity_id")
                ),
            )
        ]
