"""Rule: the router reports more UPnP port mappings than on the previous run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import posture_cap

from .network_common import make_finding, plural, posture

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )


class NetworkUpnpPortMappingAddedRule:
    """A device asked the router to expose it to the internet since the last run."""

    rule_id = "network_upnp_port_mapping_added"
    requires = frozenset({posture_cap("upnp_port_mappings_added")})
    cooldown_minutes = 0

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding per increase; identity is the new count and the delta."""
        section = posture(snapshot)
        added = section.get("upnp_port_mappings_added")
        if not isinstance(added, int) or isinstance(added, bool) or added <= 0:
            return []
        count = section.get("upnp_port_mapping_count")
        total = (
            count if isinstance(count, int) and not isinstance(count, bool) else None
        )
        entity_id = section.get("upnp_port_mapping_entity_id")
        entities = [str(entity_id)] if entity_id else []
        sentence = (
            f"{plural(added, 'new port mapping')} "
            f"{'was' if added == 1 else 'were'} opened through UPnP since the "
            "last check"
        )
        if total is not None:
            sentence += f" ({plural(total, 'mapping')} open in total)"
        sentence += (
            ". A device on your network asked the router to make it reachable "
            "from the internet."
        )
        return [
            make_finding(
                self.rule_id,
                severity="medium",
                triggering_entities=entities,
                evidence={"added": added, "count": total},
                display={
                    "previous_count": section.get("upnp_port_mapping_previous_count")
                },
                summary=sentence,
                suggested_actions=[
                    (
                        "Open the router's UPnP or port-forwarding page to see "
                        "which device opened the port, and remove it if you do not "
                        "recognize it"
                    ),
                    "Turn off UPnP on the router if nothing you use needs it",
                ],
            )
        ]
