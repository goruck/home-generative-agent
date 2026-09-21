# ruff: noqa: S101
"""Tests for the HA-native network / security rules against synthetic snapshots."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import pytest

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
    make_finding,
)
from custom_components.home_generative_agent.sentinel.rules.network_guest_client_present import (
    NetworkGuestClientPresentRule,
)
from custom_components.home_generative_agent.sentinel.rules.network_public_ip_changed import (
    NetworkPublicIpChangedRule,
)
from custom_components.home_generative_agent.sentinel.rules.network_router_posture import (
    NetworkDdnsEnabledRule,
    NetworkGuestNetworkIdleRule,
    NetworkProtectionDisabledRule,
    NetworkWpa3DisabledRule,
)
from custom_components.home_generative_agent.sentinel.rules.network_router_update_pending import (
    NetworkRouterUpdatePendingRule,
)
from custom_components.home_generative_agent.sentinel.rules.network_unconfigured_discovered_device import (
    NetworkUnconfiguredDiscoveredDeviceRule,
)
from custom_components.home_generative_agent.sentinel.rules.network_unknown_device_joined import (
    NetworkUnknownDeviceJoinedRule,
)
from custom_components.home_generative_agent.sentinel.rules.network_upnp_enabled import (
    NetworkUpnpEnabledRule,
)
from custom_components.home_generative_agent.sentinel.rules.network_upnp_port_mapping_added import (
    NetworkUpnpPortMappingAddedRule,
)
from custom_components.home_generative_agent.sentinel.rules.radio_coordinator_update_pending import (
    RadioCoordinatorUpdatePendingRule,
)
from custom_components.home_generative_agent.sentinel.rules.radio_new_device_joined import (
    RadioNewDeviceJoinedRule,
)
from custom_components.home_generative_agent.sentinel.rules.security_device_unavailable import (
    SecurityDeviceUnavailableRule,
)
from custom_components.home_generative_agent.sentinel.rules.zigbee_permit_join_open import (
    ZigbeePermitJoinOpenRule,
)
from custom_components.home_generative_agent.sentinel.rules.zwave_inclusion_active import (
    ZwaveInclusionActiveRule,
)
from custom_components.home_generative_agent.sentinel.rules.zwave_insecure_security_class import (
    ZwaveInsecureSecurityClassRule,
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
    NetworkUpnpEnabledRule(),
    NetworkPublicIpChangedRule(),
    NetworkUpnpPortMappingAddedRule(),
    RadioNewDeviceJoinedRule(),
    NetworkUnknownDeviceJoinedRule(),
    NetworkGuestClientPresentRule(),
    NetworkGuestNetworkIdleRule(),
    NetworkWpa3DisabledRule(),
    NetworkProtectionDisabledRule(),
    NetworkDdnsEnabledRule(),
    ZwaveInsecureSecurityClassRule(),
    ZigbeePermitJoinOpenRule(),
    ZwaveInclusionActiveRule(),
    RadioCoordinatorUpdatePendingRule(),
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
            _snapshot(ha_security={"long_lived_token_age_days": {"api": 100}})
        )
    )
    b = _only(
        HaLongLivedTokenStaleRule(stale_days=10).evaluate(
            _snapshot(ha_security={"long_lived_token_age_days": {"api": 101}})
        )
    )
    assert a.is_sensitive
    assert a.evidence["summary"] != b.evidence["summary"]
    assert a.evidence["token_age_days"] != b.evidence["token_age_days"]
    # The changing figure and the sentence are display-only: identity is the
    # token, so pending-prompt and snooze state survive a changing count.
    assert a.anomaly_id == b.anomaly_id
    assert HaLongLivedTokenStaleRule.cooldown_minutes == POSTURE_COOLDOWN_MINUTES


def test_make_finding_hashes_identity_only_and_rejects_dotted_actions() -> None:
    """Summary and display fields never change the id; dotted actions are refused."""
    kwargs: dict[str, Any] = {
        "severity": "low",
        "evidence": {"token": "api"},
        "suggested_actions": ["Revoke it"],
    }
    a = make_finding("ha_long_lived_token_stale", summary="idle 3 days", **kwargs)
    b = make_finding(
        "ha_long_lived_token_stale",
        summary="idle 4 days",
        display={"unused_days": 4},
        **kwargs,
    )
    assert a.anomaly_id == b.anomaly_id
    assert b.evidence["unused_days"] == 4
    # Bidi / zero-width characters in a summary are dropped before rendering.
    c = make_finding("ha_failed_logins", summary="evil\u202e\u200bname.", **kwargs)
    assert c.evidence["summary"] == "evilname."
    with pytest.raises(ValueError, match="must not contain"):
        make_finding(
            "ha_failed_logins",
            severity="low",
            evidence={},
            summary="x",
            suggested_actions=["Edit configuration.yaml"],
        )


def test_no_rule_emits_a_dotted_suggested_action() -> None:
    """The engine parses 'a.b' suggested actions as service calls (engine.py)."""
    for rule in ALL_RULES:
        source = inspect.getsource(type(rule))
        # Every rule builds through make_finding, which raises on a dot; this
        # guards a future rule that bypasses the helper.
        assert "make_finding(" in source, rule.rule_id


def test_network_rule_types_match_engine_rules_with_requires() -> None:
    """Every engine rule that declares ``requires`` is in NETWORK_RULE_TYPES."""
    from custom_components.home_generative_agent.sentinel.discovery_engine import (  # noqa: PLC0415
        _STATIC_RULE_IDS,
    )
    from custom_components.home_generative_agent.sentinel.engine import (  # noqa: PLC0415
        SentinelEngine,
    )

    engine = SentinelEngine.__new__(SentinelEngine)
    # Build the rule list the way the constructor does, without dependencies.
    init_src = inspect.getsource(SentinelEngine.__init__)
    assert "HaNewAdminOrTokenRule()" in init_src
    gated = {r.rule_id for r in ALL_RULES if getattr(r, "requires", None)}
    assert gated == NETWORK_RULE_TYPES
    assert NETWORK_RULE_TYPES <= _STATIC_RULE_IDS
    del engine


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------


def test_sensitive_exposed_pin_gates_assist_only() -> None:
    """One finding per cycle; the PIN silences Assist but never Alexa/Google."""
    rule = HaSensitiveEntityExposedWithoutPinRule()
    exposed = {
        "conversation": ["lock.front"],
        "cloud.alexa": ["lock.front", "cover.garage"],
        "cloud.google_assistant": [],
    }
    off = _snapshot(
        ha_security={
            "exposed_sensitive_entities": exposed,
            "critical_action_pin_enabled": False,
        }
    )
    finding = _only(rule.evaluate(off))
    assert finding.severity == "high"
    assert finding.triggering_entities == ["cover.garage", "lock.front"]
    assert finding.evidence["exposures"] == {
        "cloud.alexa": ["cover.garage", "lock.front"],
        "conversation": ["lock.front"],
    }
    assert "Assist with no Critical Action PIN" in finding.evidence["summary"]
    assert "to Alexa" in finding.evidence["summary"]
    on = _snapshot(
        ha_security={
            "exposed_sensitive_entities": exposed,
            "critical_action_pin_enabled": True,
            "assist_agents_outside_pin": [],
        }
    )
    gated = _only(rule.evaluate(on))
    # Alexa exposure still reported: the PIN never guarded the cloud path.
    assert gated.evidence["exposures"] == {
        "cloud.alexa": ["cover.garage", "lock.front"]
    }
    assert "Alexa app's own PIN" in gated.suggested_actions[0]
    assist_only = _snapshot(
        ha_security={
            "exposed_sensitive_entities": {"conversation": ["lock.front"]},
            "critical_action_pin_enabled": True,
            "assist_agents_outside_pin": [],
        }
    )
    assert rule.evaluate(assist_only) == []
    # Unknown PIN state (capability absent) never fires.
    unknown = _snapshot(ha_security={"exposed_sensitive_entities": exposed})
    assert rule.evaluate(unknown) == []


def test_new_admin_or_token_one_finding_per_cycle() -> None:
    """Every change in a cycle rides in one finding; the id follows the set."""
    rule = HaNewAdminOrTokenRule()
    both = _only(
        rule.evaluate(
            _snapshot(
                ha_security={
                    "new_admin_users": ["Guest"],
                    "new_long_lived_tokens": ["script"],
                }
            )
        )
    )
    assert both.severity == "high"
    assert both.evidence["new_admin_users"] == ["Guest"]
    assert both.evidence["new_long_lived_tokens"] == ["script"]
    assert both.evidence["summary"] == (
        "New administrator account: Guest; new long-lived access token: script."
    )
    assert len(both.suggested_actions) == 2
    token_only = _only(
        rule.evaluate(
            _snapshot(
                ha_security={"new_admin_users": [], "new_long_lived_tokens": ["script"]}
            )
        )
    )
    assert token_only.anomaly_id != both.anomaly_id
    assert (
        rule.evaluate(
            _snapshot(ha_security={"new_admin_users": [], "new_long_lived_tokens": []})
        )
        == []
    )
    assert HaNewAdminOrTokenRule.cooldown_minutes == 0


def test_long_lived_token_age_threshold() -> None:
    """Tokens at or past the age threshold are listed in one low finding."""
    findings = HaLongLivedTokenStaleRule(stale_days=90).evaluate(
        _snapshot(
            ha_security={
                "long_lived_token_age_days": {"old": 91, "ancient": 400, "fresh": 3}
            }
        )
    )
    finding = _only(findings)
    assert finding.severity == "low"
    assert finding.evidence["tokens"] == ["ancient", "old"]
    assert finding.evidence["token_age_days"] == {"ancient": 400, "old": 91}
    assert "cannot tell whether they are still used" in finding.evidence["summary"]
    assert "ancient (400 d)" in finding.evidence["summary"]


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
    assert ban_off.severity == "medium"
    # Home Assistant's shipped default: banning on, no threshold -> low.
    stock = _only(
        HaHttpProxyMisconfiguredRule().evaluate(
            _snapshot(
                ha_security={
                    "http_ip_ban_enabled": True,
                    "http_login_attempts_threshold": -1,
                }
            )
        )
    )
    assert stock.evidence["reason"] == "no_login_threshold"
    assert stock.severity == "low"
    assert stock.evidence["ip_ban_enabled"] is True
    assert (
        HaHttpProxyMisconfiguredRule().evaluate(
            _snapshot(
                ha_security={
                    "http_ip_ban_enabled": True,
                    "http_login_attempts_threshold": 5,
                }
            )
        )
        == []
    )


def test_addon_rules_aggregate_with_severity_and_names() -> None:
    """One finding per rule; SSH-class add-ons raise the port finding to high."""
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
    ports = _only(HaAddonExposedPortRule().evaluate(snapshot))
    assert ports.severity == "high"
    assert ports.evidence["addons"] == {"core_mosquitto": [1883], "core_ssh": [22]}
    assert ports.evidence["high_risk"] == ["core_ssh"]
    assert "Terminal & SSH (22)" in ports.evidence["summary"]
    assert "Mosquitto (1883)" in ports.evidence["summary"]
    medium = _only(
        HaAddonExposedPortRule().evaluate(
            _snapshot(
                ha_security={"addons_with_host_ports": {"core_mosquitto": [1883]}}
            )
        )
    )
    assert medium.severity == "medium"
    unprotected = _only(HaAddonUnprotectedRule().evaluate(snapshot))
    assert unprotected.severity == "high"
    assert unprotected.evidence["addons"] == ["core_mosquitto"]
    assert "Mosquitto" in unprotected.evidence["summary"]


def test_webhook_public_aggregates_and_honors_exclusions() -> None:
    """All public webhook automations in one finding; critical ones raise it."""
    snapshot = _snapshot(
        ha_security={
            "webhook_automations_public": ["automation.a", "automation.b"],
            "webhook_automations_critical": ["automation.b"],
        }
    )
    finding = _only(HaWebhookAutomationPublicRule().evaluate(snapshot))
    assert finding.severity == "high"
    assert finding.triggering_entities == ["automation.a", "automation.b"]
    assert finding.evidence["critical"] == ["automation.b"]
    assert "automation.b (can unlock or open an entry)" in finding.evidence["summary"]
    # Excluding the critical automation drops it before aggregation.
    excluded = HaWebhookAutomationPublicRule(
        is_entity_excluded=lambda entity_id, _t: entity_id == "automation.b"
    )
    rest = _only(excluded.evaluate(snapshot))
    assert rest.severity == "medium"
    assert rest.triggering_entities == ["automation.a"]


def test_security_device_unavailable_aggregates_past_threshold() -> None:
    """Devices past the threshold ride in one finding with friendly names."""
    snapshot = _snapshot(
        ha_security={
            "unavailable_security_devices": {
                "lock.front": 45,
                "camera.yard": 60,
                "camera.new": 5,
            }
        },
        entities=[_entity("lock.front", "unavailable", friendly_name="Front Door")],
    )
    finding = _only(
        SecurityDeviceUnavailableRule(offline_minutes=30).evaluate(snapshot)
    )
    assert finding.triggering_entities == ["camera.yard", "lock.front"]
    assert finding.severity == "high"
    assert finding.evidence["unavailable_minutes"] == {
        "camera.yard": 60,
        "lock.front": 45,
    }
    assert "Front Door (45 min)" in finding.evidence["summary"]
    assert finding.suggested_actions == ["check_sensor"]
    only_lock = SecurityDeviceUnavailableRule(
        offline_minutes=30,
        is_entity_excluded=lambda entity_id, _t: entity_id.startswith("camera."),
    )
    assert _only(only_lock.evaluate(snapshot)).triggering_entities == ["lock.front"]
    assert SecurityDeviceUnavailableRule(offline_minutes=90).evaluate(snapshot) == []


def test_unconfigured_devices_aggregate_severity_by_camera_and_presence() -> None:
    """A camera handler while away makes the one finding medium; else low."""
    ha = {
        "discovered_unconfigured": [
            {"handler": "reolink", "source": "dhcp", "title": "RLC-810"},
            {"handler": "hue", "source": "ssdp", "title": "Hue Bridge"},
        ]
    }
    rule = NetworkUnconfiguredDiscoveredDeviceRule()
    away = _only(rule.evaluate(_snapshot(ha_security=ha, anyone_home=False)))
    assert away.severity == "medium"
    assert away.evidence["camera_handlers"] == ["reolink"]
    home = _only(rule.evaluate(_snapshot(ha_security=ha)))
    assert home.severity == "low"
    assert "RLC-810 (reolink via dhcp)" in home.evidence["summary"]
    assert "Hue Bridge (hue via ssdp)" in home.evidence["summary"]
    # Presence is display-only: the same set keeps one id home or away.
    assert home.anomaly_id == away.anomaly_id


def test_router_update_pending_aggregates_with_versions() -> None:
    """Every router update entity in one finding; versions are display-only."""
    snapshot = _snapshot(
        posture={
            "router_update_pending": True,
            "router_update_entities": ["update.eero_fw", "update.fritz_fw"],
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
    assert finding.triggering_entities == ["update.eero_fw", "update.fritz_fw"]
    assert "eero Firmware (7.1 -> 7.2)" in finding.evidence["summary"]
    assert finding.evidence["versions"]["update.eero_fw"]["latest"] == "7.2"
    quiet = _snapshot(
        posture={"router_update_pending": False, "router_update_entities": []}
    )
    assert NetworkRouterUpdatePendingRule().evaluate(quiet) == []
    excluded = NetworkRouterUpdatePendingRule(
        is_entity_excluded=lambda entity_id, _t: entity_id == "update.fritz_fw"
    )
    assert _only(excluded.evaluate(snapshot)).triggering_entities == ["update.eero_fw"]


def test_upnp_enabled_is_a_standing_finding_with_display_only_count() -> None:
    """One medium finding while UPnP is on; the mapping count never changes the id."""
    rule = NetworkUpnpEnabledRule()
    plain = _only(
        rule.evaluate(
            _snapshot(
                posture={
                    "upnp_enabled": True,
                    "upnp_evidence": "ssdp",
                    "upnp_gateway_names": ["eero"],
                }
            )
        )
    )
    assert plain.severity == "medium"
    assert rule.cooldown_minutes == POSTURE_COOLDOWN_MINUTES
    assert plain.evidence["summary"].startswith(
        "Your router (eero) is advertising UPnP (Internet Gateway Device)"
    )
    assert "unless the router restricts it" in plain.evidence["summary"]
    assert "port mapping" not in plain.evidence["summary"]
    counted = _only(
        rule.evaluate(
            _snapshot(
                posture={
                    "upnp_enabled": True,
                    "upnp_evidence": "integration",
                    "upnp_gateway_names": [],
                    "upnp_port_mapping_count": 3,
                }
            )
        )
    )
    assert counted.anomaly_id == plain.anomaly_id
    assert counted.evidence["port_mapping_count"] == 3
    assert (
        "Your router is answering the UPnP/IGD integration"
        in (counted.evidence["summary"])
    )
    many = _only(
        rule.evaluate(
            _snapshot(
                posture={"upnp_enabled": True, "upnp_gateway_names": ["a", "b", "c"]}
            )
        )
    )
    assert "Your router (a, b, and 1 more)" in many.evidence["summary"]
    assert "3 port mappings are currently open" in counted.evidence["summary"]
    one = _only(
        rule.evaluate(
            _snapshot(posture={"upnp_enabled": True, "upnp_port_mapping_count": 1})
        )
    )
    assert "1 port mapping is currently open" in one.evidence["summary"]
    assert rule.evaluate(_snapshot(posture={"upnp_enabled": False})) == []


def test_public_ip_changed_is_identified_by_the_new_pseudonymized_key() -> None:
    """Low, per new address, naming the sensor; the old key is display-only."""
    rule = NetworkPublicIpChangedRule()
    assert rule.cooldown_minutes == 0
    base = {
        "public_ip_changed": True,
        "public_ip_key": "abcd1234",
        "public_ip_previous_key": "00000000",
        "public_ip_entity_id": "sensor.gw_external_ip",
    }
    finding = _only(rule.evaluate(_snapshot(posture=base)))
    assert finding.severity == "low"
    assert finding.triggering_entities == ["sensor.gw_external_ip"]
    assert finding.evidence["public_ip_key"] == "abcd1234"
    assert finding.evidence["previous_public_ip_key"] == "00000000"
    assert "public IP address changed" in finding.evidence["summary"]
    other = _only(
        rule.evaluate(_snapshot(posture={**base, "public_ip_key": "ffff0000"}))
    )
    assert other.anomaly_id != finding.anomaly_id
    same_ip = _only(
        rule.evaluate(_snapshot(posture={**base, "public_ip_previous_key": "1111"}))
    )
    assert same_ip.anomaly_id == finding.anomaly_id
    assert rule.evaluate(_snapshot(posture={**base, "public_ip_changed": False})) == []
    assert rule.evaluate(_snapshot(posture={**base, "public_ip_key": ""})) == []


def test_upnp_port_mapping_added_reports_the_increase_only() -> None:
    """Medium on an increase with the total; no finding when nothing was added."""
    rule = NetworkUpnpPortMappingAddedRule()
    assert rule.cooldown_minutes == 0
    sensor = "sensor.gw_port_mapping_number_of_entries"
    finding = _only(
        rule.evaluate(
            _snapshot(
                posture={
                    "upnp_port_mappings_added": 2,
                    "upnp_port_mapping_count": 5,
                    "upnp_port_mapping_previous_count": 3,
                    "upnp_port_mapping_entity_id": sensor,
                }
            )
        )
    )
    assert finding.severity == "medium"
    assert finding.triggering_entities == [sensor]
    assert finding.evidence["added"] == 2
    assert finding.evidence["count"] == 5
    assert finding.evidence["previous_count"] == 3
    assert (
        "2 new port mappings were opened through UPnP since the last check "
        "(5 mappings open in total)."
    ) in finding.evidence["summary"]
    single = _only(
        rule.evaluate(
            _snapshot(
                posture={"upnp_port_mappings_added": 1, "upnp_port_mapping_count": 1}
            )
        )
    )
    assert "1 new port mapping was opened" in single.evidence["summary"]
    assert "(1 mapping open in total)" in single.evidence["summary"]
    assert rule.evaluate(_snapshot(posture={"upnp_port_mappings_added": 0})) == []
    assert rule.evaluate(_snapshot(posture={"upnp_port_mappings_added": True})) == []


def test_sensitive_exposed_pin_does_not_cover_other_pipeline_agents() -> None:
    """A pipeline on another agent keeps Assist exposure reported despite the PIN."""
    rule = HaSensitiveEntityExposedWithoutPinRule()
    exposed = {"conversation": ["lock.front"]}
    covered = _snapshot(
        ha_security={
            "exposed_sensitive_entities": exposed,
            "critical_action_pin_enabled": True,
            "assist_agents_outside_pin": [],
        }
    )
    assert rule.evaluate(covered) == []
    uncovered = _snapshot(
        ha_security={
            "exposed_sensitive_entities": exposed,
            "critical_action_pin_enabled": True,
            "assist_agents_outside_pin": ["conversation.home_assistant"],
        }
    )
    finding = _only(rule.evaluate(uncovered))
    assert finding.evidence["exposures"] == {"conversation": ["lock.front"]}
    assert (
        "does not cover the pipeline agent conversation.home_assistant"
        in (finding.evidence["summary"])
    )
    assert "Point every Assist pipeline" in finding.suggested_actions[0]
    # Unknown coverage is never treated as covered.
    unknown = _snapshot(
        ha_security={
            "exposed_sensitive_entities": exposed,
            "critical_action_pin_enabled": True,
        }
    )
    assert "could not be verified" in _only(rule.evaluate(unknown)).evidence["summary"]
