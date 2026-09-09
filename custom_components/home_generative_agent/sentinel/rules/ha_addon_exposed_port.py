"""Rule: Supervisor add-ons map container ports onto the host."""

from __future__ import annotations

from typing import TYPE_CHECKING

from custom_components.home_generative_agent.snapshot.network import ha_cap

from .network_common import POSTURE_COOLDOWN_MINUTES, ha_security, make_finding, plural

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

# Add-ons that hand out shell, file, or database access when their port is
# reachable. Matched against the slug and display name.
_HIGH_RISK_HINTS: tuple[str, ...] = (
    "ssh",
    "terminal",
    "samba",
    "smb",
    "mariadb",
    "mysql",
    "postgres",
    "mongodb",
    "influxdb",
    "database",
)


def _high_risk(slug: str, name: str) -> bool:
    haystack = f"{slug} {name}".lower()
    return any(h in haystack for h in _HIGH_RISK_HINTS)


class HaAddonExposedPortRule:
    """One finding per cycle listing every running add-on with host ports."""

    rule_id = "ha_addon_exposed_port"
    requires = frozenset({ha_cap("addons_with_host_ports")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding covering every exposed add-on."""
        ha = ha_security(snapshot)
        ports_by_slug: dict[str, list[int]] = ha.get("addons_with_host_ports") or {}
        names: dict[str, str] = ha.get("addon_names") or {}
        exposed = {
            slug: sorted(ports) for slug, ports in ports_by_slug.items() if ports
        }
        if not exposed:
            return []
        high = sorted(
            slug for slug in exposed if _high_risk(slug, names.get(slug, slug))
        )
        listed = ", ".join(
            f"{names.get(slug, slug)} ({', '.join(str(p) for p in ports)})"
            for slug, ports in sorted(exposed.items())
        )
        return [
            make_finding(
                self.rule_id,
                severity="high" if high else "medium",
                evidence={"addons": exposed, "high_risk": high},
                display={"addon_names": {s: names.get(s, s) for s in exposed}},
                summary=(
                    f"{plural(len(exposed), 'add-on')} listening on host ports, "
                    f"reachable by anything on your network: {listed}."
                ),
                suggested_actions=[
                    (
                        "Disable the host port in each add-on's Network settings if "
                        "you only use it through Ingress, or restrict it with a "
                        "firewall"
                    )
                ],
            )
        ]
