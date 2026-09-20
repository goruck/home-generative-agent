"""Rule: a client the device inventory does not know joined the network."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    RECOMMENDED_SENTINEL_NETWORK_UNKNOWN_DEVICE_GRACE_MIN,
)
from custom_components.home_generative_agent.sentinel.network_inventory import (
    client_key,
)
from custom_components.home_generative_agent.sentinel.redaction import (
    client_display_name,
)
from custom_components.home_generative_agent.snapshot.network import (
    CAP_CLIENTS,
    CAP_NEW_CLIENTS,
)

from .network_common import (
    anyone_home,
    clients,
    is_night,
    listed,
    make_finding,
    new_clients,
    noun,
)

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


def describe_client(client: Mapping[str, Any]) -> str:
    """
    Return ``Name (Apple, wireless, guest Wi-Fi, 192.168.1.23)`` for a client.

    The name is the tracker entity's, the label the user already sees in
    Home Assistant; a client without one is named by manufacturer and key.
    The DHCP hostname is never used here: the summary reaches the audit
    tool's model, and a hostname in free text cannot be redacted.
    """
    name = client.get("name") or client_display_name(client)
    details = [
        part
        for part in (
            client.get("manufacturer"),
            client.get("connection_type"),
            "guest Wi-Fi" if client.get("is_guest") is True else None,
            client.get("ip"),
        )
        if part
    ]
    return f"{name} ({', '.join(details)})" if details else str(name)


def describe_joined(client: Mapping[str, Any]) -> str:
    """Return the client's description, marked when it has since gone offline."""
    text = describe_client(client)
    return text if client.get("connected") else f"{text}, not connected now"


class NetworkUnknownDeviceJoinedRule:
    """
    Clients on the router that the device inventory had not recorded.

    Compares against the inventory as it stood before the run (the adapter
    asks it which keys are new); the engine commits after dispatch and holds
    back clients whose finding was not delivered, so a suppressed alert is
    repeated on a later run. A client is reported once its inventory row is
    older than the grace period, whether or not it is still connected: a
    device that was on the network and left is exactly what the owner wants
    to hear about, so an offline client is named with "not connected now"
    rather than withheld (a field test lost a two-hour visitor to the gate
    this replaced). A client the inventory auto-trusted (a registry device
    set up by a non-router integration) is never new. A rotated random
    address on a device whose name a trusted row already carries is reported
    once, at low severity.
    """

    rule_id = "network_unknown_device_joined"
    requires = frozenset({CAP_CLIENTS, CAP_NEW_CLIENTS})
    cooldown_minutes = 0

    def __init__(
        self,
        *,
        grace_minutes: int = RECOMMENDED_SENTINEL_NETWORK_UNKNOWN_DEVICE_GRACE_MIN,
        is_entity_excluded: Callable[[str, str], bool] | None = None,
    ) -> None:
        """Initialize with the grace period and the per-rule entity exclusions."""
        self._grace = timedelta(minutes=max(0, grace_minutes))
        self._is_entity_excluded = is_entity_excluded

    def _excluded(self, client: Mapping[str, Any]) -> bool:
        entity_id = client.get("tracker_entity_id")
        if not entity_id or self._is_entity_excluded is None:
            return False
        return self._is_entity_excluded(str(entity_id), self.rule_id)

    def _past_grace(self, client: Mapping[str, Any], now: Any) -> bool:
        if not self._grace:
            return True
        first_seen = dt_util.parse_datetime(str(client.get("first_seen") or ""))
        if first_seen is None:
            # Not recorded yet: the commit after this run records it, and
            # the grace period runs from there.
            return False
        return dt_util.as_utc(now) - dt_util.as_utc(first_seen) >= self._grace

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding naming every new client past its grace period."""
        keys = new_clients(snapshot)
        if not keys:
            return []
        now = dt_util.parse_datetime(snapshot["generated_at"]) or dt_util.utcnow()
        joined = [
            c
            for c in clients(snapshot)
            if c.get("key") in keys
            and not self._excluded(c)
            and self._past_grace(c, now)
        ]
        if not joined:
            return []
        away = not anyone_home(snapshot)
        night = is_night(snapshot)
        severity: Severity = "high" if away or night else "medium"
        if all(c.get("hostname_trusted") for c in joined):
            severity = "low"
        count = len(joined)
        head = f"New {noun(count, 'device')} on the network"
        context = " while nobody is home" if away else (" at night" if night else "")
        return [
            make_finding(
                self.rule_id,
                severity=severity,
                evidence={"client_keys": sorted(c["key"] for c in joined)},
                display={
                    # Inventory keys, so the Trust button and the trust
                    # service resolve them without a registry device.
                    "device_ids": sorted(client_key(c["key"]) for c in joined),
                    "names": [describe_joined(c) for c in joined],
                    "offline": sorted(
                        c["key"] for c in joined if not c.get("connected")
                    ),
                    "randomized_known": sorted(
                        c["key"] for c in joined if c.get("hostname_trusted")
                    ),
                },
                summary=(
                    f"{head}{context}: {listed([describe_joined(c) for c in joined])}."
                ),
                suggested_actions=[
                    (
                        "If you do not recognize it, block it in your router app "
                        "and change your Wi-Fi password"
                    ),
                    "Tap Trust device if you recognize it",
                ],
                triggering_entities=sorted(
                    str(c["tracker_entity_id"])
                    for c in joined
                    if c.get("tracker_entity_id")
                ),
            )
        ]
