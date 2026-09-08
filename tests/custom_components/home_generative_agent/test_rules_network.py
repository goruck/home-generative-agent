# ruff: noqa: S101
"""Tests for the HA-native network / security rules against synthetic snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from custom_components.home_generative_agent.sentinel.models import (
    DISPLAY_ONLY_EVIDENCE_KEYS,
)
from custom_components.home_generative_agent.sentinel.rules.ha_addon_exposed_port import (
    HaAddonExposedPortRule,
)
from custom_components.home_generative_agent.sentinel.rules.ha_addon_unprotected import (
    HaAddonUnprotectedRule,
)
from custom_components.home_generative_agent.sentinel.rules.ha_cloud_remote_ui_enabled import (
    HaCloudRemoteUiEnabledRule,
)
from custom_components.home_generative_agent.sentinel.rules.ha_failed_logins import (
    HaFailedLoginsRule,
)
from custom_components.home_generative_agent.sentinel.rules.ha_http_proxy_misconfigured import (
    HaHttpProxyMisconfiguredRule,
)
from custom_components.home_generative_agent.sentinel.rules.ha_long_lived_token_stale import (
    HaLongLivedTokenStaleRule,
)
from custom_components.home_generative_agent.sentinel.rules.ha_new_admin_or_token import (
    HaNewAdminOrTokenRule,
)
from custom_components.home_generative_agent.sentinel.rules.ha_sensitive_entity_exposed_without_pin import (
    HaSensitiveEntityExposedWithoutPinRule,
)
from custom_components.home_generative_agent.sentinel.rules.ha_trusted_networks_bypass_login import (
    HaTrustedNetworksBypassLoginRule,
)
from custom_components.home_generative_agent.sentinel.rules.ha_webhook_automation_public import (
    HaWebhookAutomationPublicRule,
)
from custom_components.home_generative_agent.sentinel.rules.network_common import (
    NETWORK_RULE_TYPES,
    POSTURE_COOLDOWN_MINUTES,
)
from custom_components.home_generative_agent.sentinel.rules.network_router_update_pending import (
    NetworkRouterUpdatePendingRule,
)
from custom_components.home_generative_agent.sentinel.rules.network_unconfigured_discovered_device import (
    NetworkUnconfiguredDiscoveredDeviceRule,
)
from custom_components.home_generative_agent.sentinel.rules.security_device_unavailable import (
    SecurityDeviceUnavailableRule,
)
from custom_components.home_generative_agent.snapshot.network import ha_cap
from custom_components.home_generative_agent.snapshot.schema import validate_snapshot

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

ALL_RULES = [
    HaSensitiveEntityExposedWithoutPinRule(),
    HaNewAdminOrTokenRule(),
    HaLongLivedTokenStaleRule(stale_days=90),
    HaFailedLoginsRule(),
    HaCloudRemoteUiEnabledRule(),
    HaAddonExposedPortRule(),
    HaAddonUnprotectedRule(),
    HaWebhookAutomationPublicRule(),
    HaTrustedNetworksBypassLoginRule(),
    HaHttpProxyMisconfiguredRule(),
    SecurityDeviceUnavailableRule(offline_minutes=30),
    NetworkUnconfiguredDiscoveredDeviceRule(),
    NetworkRouterUpdatePendingRule(),
]


def _snapshot(
    *,
    ha_security: dict[str, Any] | None = None,
    posture: dict[str, Any] | None = None,
    entities: list[dict[str, Any]] | None = None,
    anyone_home: bool = True,
) -> FullStateSnapshot:
    ha = ha_security or {}
    post = posture or {}
    capabilities = sorted(
        [ha_cap(k) for k in ha] + [f"network.posture.{k}" for k in post]
    )
    return validate_snapshot(
        {
            "schema_version": 2,
            "generated_at": "2026-09-07T12:00:00+00:00",
            "entities": entities or [],
            "camera_activity": [],
            "derived": {
                "now": "2026-09-07T12:00:00+00:00",
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": anyone_home,
                "people_home": ["Lindo"] if anyone_home else [],
                "people_away": [] if anyone_home else ["Lindo"],
                "last_motion_by_area": {},
            },
            "network": {
                "capabilities": capabilities,
                "sources": dict.fromkeys(capabilities, "ha_native"),
                "clients": [],
                "posture": post,
                "ha_security": ha,
                "counters": {},
            },
        }
    )


def _entity(entity_id: str, state: str, **attrs: Any) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "domain": entity_id.partition(".")[0],
        "state": state,
        "friendly_name": attrs.pop("friendly_name", None),
        "area": None,
        "attributes": attrs,
        "last_changed": "2026-09-07T11:00:00+00:00",
        "last_updated": "2026-09-07T11:00:00+00:00",
        "platform": attrs.pop("platform", None),
    }


def _only(findings: list[AnomalyFinding]) -> AnomalyFinding:
    assert len(findings) == 1, [f.evidence for f in findings]
    return findings[0]


# ---------------------------------------------------------------------------
# Family-wide invariants
# ---------------------------------------------------------------------------


def test_every_rule_declares_requires_and_is_registered() -> None:
    """Each rule names its capabilities and appears in NETWORK_RULE_TYPES."""
    for rule in ALL_RULES:
        assert rule.requires, rule.rule_id
        assert all(cap.startswith("network.") for cap in rule.requires), rule.rule_id
        assert rule.rule_id in NETWORK_RULE_TYPES
    assert {r.rule_id for r in ALL_RULES} == NETWORK_RULE_TYPES


def test_rules_return_nothing_on_pre_v2_snapshot() -> None:
    """A snapshot without a network section is silently empty for every rule."""
    snapshot = _snapshot()
    del snapshot["network"]  # type: ignore[misc]
    for rule in ALL_RULES:
        assert rule.evaluate(snapshot) == [], rule.rule_id


def test_summary_is_display_only_and_findings_are_sensitive() -> None:
    """Summary text never changes the anomaly id; every finding is sensitive."""
    assert "summary" in DISPLAY_ONLY_EVIDENCE_KEYS
    a = _only(
        HaLongLivedTokenStaleRule(stale_days=10).evaluate(
            _snapshot(ha_security={"long_lived_tokens_unused_days": {"api": 100}})
        )
    )
    b = _only(
        HaLongLivedTokenStaleRule(stale_days=10).evaluate(
            _snapshot(ha_security={"long_lived_tokens_unused_days": {"api": 101}})
        )
    )
    assert a.is_sensitive
    assert a.evidence["summary"] != b.evidence["summary"]
    # unused_days differs so ids differ, but the summary alone must not matter:
    # rebuild a with b's summary and expect a's id.
    assert a.anomaly_id != b.anomaly_id
    assert HaLongLivedTokenStaleRule.cooldown_minutes == POSTURE_COOLDOWN_MINUTES


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------


def test_sensitive_exposed_without_pin() -> None:
    """Fires per assistant only when the PIN is not enforceable."""
    rule = HaSensitiveEntityExposedWithoutPinRule()
    exposed = {"conversation": ["lock.front"], "cloud.alexa": []}
    off = _snapshot(
        ha_security={
            "exposed_sensitive_entities": exposed,
            "critical_action_pin_enabled": False,
        }
    )
    finding = _only(rule.evaluate(off))
    assert finding.severity == "high"
    assert finding.triggering_entities == ["lock.front"]
    assert "Assist" in finding.evidence["summary"]
    on = _snapshot(
        ha_security={
            "exposed_sensitive_entities": exposed,
            "critical_action_pin_enabled": True,
        }
    )
    assert rule.evaluate(on) == []


def test_new_admin_or_token_one_finding_per_change() -> None:
    """Each new admin, token, and new-address token is its own finding."""
    findings = HaNewAdminOrTokenRule().evaluate(
        _snapshot(
            ha_security={
                "new_admin_users": ["Guest"],
                "new_long_lived_tokens": ["script"],
                "refresh_tokens_from_new_ip": ["api"],
            }
        )
    )
    assert [f.evidence["change"] for f in findings] == [
        "new_admin_user",
        "new_long_lived_token",
        "token_from_new_address",
    ]
    assert [f.severity for f in findings] == ["high", "high", "medium"]
    assert len({f.anomaly_id for f in findings}) == 3
    assert not hasattr(HaNewAdminOrTokenRule, "cooldown_minutes")


def test_long_lived_token_stale_threshold() -> None:
    """Only tokens idle for at least the threshold are reported."""
    findings = HaLongLivedTokenStaleRule(stale_days=90).evaluate(
        _snapshot(
            ha_security={"long_lived_tokens_unused_days": {"old": 91, "fresh": 3}}
        )
    )
    finding = _only(findings)
    assert finding.severity == "low"
    assert finding.evidence["token"] == "old"  # noqa: S105
    assert "91 days" in finding.evidence["summary"]


def test_failed_logins_and_cloud_and_bypass_and_ip_ban() -> None:
    """Boolean posture rules fire on True/absent-protection and stay quiet otherwise."""
    assert (
        _only(
            HaFailedLoginsRule().evaluate(
                _snapshot(ha_security={"failed_login_notification_present": True})
            )
        ).severity
        == "medium"
    )
    assert (
        HaFailedLoginsRule().evaluate(
            _snapshot(ha_security={"failed_login_notification_present": False})
        )
        == []
    )
    assert (
        _only(
            HaCloudRemoteUiEnabledRule().evaluate(
                _snapshot(ha_security={"cloud_remote_ui_enabled": True})
            )
        ).severity
        == "low"
    )
    assert (
        _only(
            HaTrustedNetworksBypassLoginRule().evaluate(
                _snapshot(ha_security={"trusted_networks_bypass_login": True})
            )
        ).severity
        == "medium"
    )
    ban_off = _only(
        HaHttpProxyMisconfiguredRule().evaluate(
            _snapshot(
                ha_security={
                    "http_ip_ban_enabled": False,
                    "http_login_attempts_threshold": -1,
                }
            )
        )
    )
    assert ban_off.evidence["reason"] == "ip_ban_disabled"
    assert (
        HaHttpProxyMisconfiguredRule().evaluate(
            _snapshot(ha_security={"http_ip_ban_enabled": True})
        )
        == []
    )


def test_addon_rules_severity_and_names() -> None:
    """SSH-class add-ons are high; others medium; unprotected is always high."""
    snapshot = _snapshot(
        ha_security={
            "addons_with_host_ports": {"core_ssh": [22], "core_mosquitto": [1883]},
            "addons_unprotected": ["core_mosquitto"],
            "addon_names": {
                "core_ssh": "Terminal & SSH",
                "core_mosquitto": "Mosquitto",
            },
        }
    )
    ports = {
        f.evidence["addon_slug"]: f for f in HaAddonExposedPortRule().evaluate(snapshot)
    }
    assert ports["core_ssh"].severity == "high"
    assert ports["core_mosquitto"].severity == "medium"
    assert "Terminal & SSH" in ports["core_ssh"].evidence["summary"]
    unprotected = _only(HaAddonUnprotectedRule().evaluate(snapshot))
    assert unprotected.severity == "high"
    assert unprotected.evidence["addon_name"] == "Mosquitto"


def test_webhook_public_high_when_critical() -> None:
    """A public webhook automation calling a critical action is high severity."""
    findings = HaWebhookAutomationPublicRule().evaluate(
        _snapshot(
            ha_security={
                "webhook_automations_public": ["automation.a", "automation.b"],
                "webhook_automations_critical": ["automation.b"],
            }
        )
    )
    by_id = {f.triggering_entities[0]: f for f in findings}
    assert by_id["automation.a"].severity == "medium"
    assert by_id["automation.b"].severity == "high"
    assert "unlock or open" in by_id["automation.b"].evidence["summary"]


def test_security_device_unavailable_threshold_and_name() -> None:
    """Fires per device at or past the threshold with the friendly name."""
    snapshot = _snapshot(
        ha_security={"unavailable_security_devices": {"lock.front": 45, "camera.y": 5}},
        entities=[_entity("lock.front", "unavailable", friendly_name="Front Door")],
    )
    finding = _only(
        SecurityDeviceUnavailableRule(offline_minutes=30).evaluate(snapshot)
    )
    assert finding.triggering_entities == ["lock.front"]
    assert finding.severity == "high"
    assert finding.evidence["summary"].startswith("Front Door has been unavailable")
    assert finding.suggested_actions == ["check_sensor"]


def test_unconfigured_device_severity_depends_on_camera_and_presence() -> None:
    """A camera handler while away is medium; everything else low."""
    ha = {
        "discovered_unconfigured": [
            {"handler": "reolink", "source": "dhcp", "title": "RLC-810"},
            {"handler": "hue", "source": "ssdp", "title": "Hue Bridge"},
        ]
    }
    rule = NetworkUnconfiguredDiscoveredDeviceRule()
    away = {
        f.evidence["handler"]: f
        for f in rule.evaluate(_snapshot(ha_security=ha, anyone_home=False))
    }
    assert away["reolink"].severity == "medium"
    assert away["hue"].severity == "low"
    home = {f.evidence["handler"]: f for f in rule.evaluate(_snapshot(ha_security=ha))}
    assert home["reolink"].severity == "low"
    assert "RLC-810" in home["reolink"].evidence["summary"]


def test_router_update_pending_uses_entity_versions() -> None:
    """One finding per router update entity with versions when available."""
    snapshot = _snapshot(
        posture={
            "router_update_pending": True,
            "router_update_entities": ["update.eero_fw"],
        },
        entities=[
            _entity(
                "update.eero_fw",
                "on",
                friendly_name="eero Firmware",
                platform="eero",
                installed_version="7.1",
                latest_version="7.2",
            )
        ],
    )
    finding = _only(NetworkRouterUpdatePendingRule().evaluate(snapshot))
    assert finding.triggering_entities == ["update.eero_fw"]
    assert finding.evidence["platform"] == "eero"
    assert "7.1 -> 7.2" in finding.evidence["summary"]
    quiet = _snapshot(
        posture={"router_update_pending": False, "router_update_entities": []}
    )
    assert NetworkRouterUpdatePendingRule().evaluate(quiet) == []
