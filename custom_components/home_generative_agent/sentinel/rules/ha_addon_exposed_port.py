"""Rule: a Supervisor add-on maps a container port onto the host."""

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


def _severity_for(slug: str, name: str) -> Severity:
    haystack = f"{slug} {name}".lower()
    return "high" if any(h in haystack for h in _HIGH_RISK_HINTS) else "medium"


class HaAddonExposedPortRule:
    """Running add-ons with host-mapped ports."""

    rule_id = "ha_addon_exposed_port"
    requires = frozenset({ha_cap("addons_with_host_ports")})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding per add-on that exposes host ports."""
        ha = ha_security(snapshot)
        ports_by_slug: dict[str, list[int]] = ha.get("addons_with_host_ports") or {}
        names: dict[str, str] = ha.get("addon_names") or {}
        findings: list[AnomalyFinding] = []
        for slug, ports in sorted(ports_by_slug.items()):
            if not ports:
                continue
            name = names.get(slug, slug)
            port_list = ", ".join(str(p) for p in sorted(ports))
            findings.append(
                make_finding(
                    self.rule_id,
                    severity=_severity_for(slug, name),
                    evidence={
                        "addon_slug": slug,
                        "addon_name": name,
                        "host_ports": sorted(ports),
                    },
                    summary=(
                        f"Add-on {name} listens on host port(s) {port_list}, "
                        "reachable by anything on your network."
                    ),
                    suggested_actions=[
                        (
                            "Disable the host port in the add-on's Network settings "
                            "if you only use it through Ingress, or restrict it "
                            "with a firewall."
                        )
                    ],
                )
            )
        return findings
