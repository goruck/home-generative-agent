# ruff: noqa: S101, PLR0913
"""Tests for the ha_native network adapter (docs/network-security-plan.md)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from custom_components.home_generative_agent.agent.helpers import (
    resolve_critical_action_policy,
)
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
)
from custom_components.home_generative_agent.sentinel.auth_inventory import (
    TOKEN_TYPE_LONG_LIVED,
    TOKEN_TYPE_NORMAL,
    AuthInventory,
    ObservedToken,
    ObservedUser,
)
from custom_components.home_generative_agent.sentinel.pseudonymizer import (
    Pseudonymizer,
)
from custom_components.home_generative_agent.snapshot import network as network_mod
from custom_components.home_generative_agent.snapshot.network import (
    CAP_CLIENTS,
    AdapterResult,
    AddonInput,
    AutomationInput,
    HaNativeInputs,
    HttpInput,
    NetworkBuildContext,
    _collect_addons,
    _collect_auth_providers,
    _collect_cloud,
    _collect_discovery,
    _collect_failed_login,
    _collect_http,
    _log_input_failure,
    async_build_network_snapshot,
    async_collect_auth_observation,
    automation_calls_critical_action,
    automation_has_public_webhook,
    empty_network_snapshot,
    ha_cap,
    ha_native_adapter,
    is_sensitive_entity,
    merge_adapter_results,
    posture_cap,
    sanitize_label,
)
from custom_components.home_generative_agent.snapshot.schema import validate_snapshot

if TYPE_CHECKING:
    from custom_components.home_generative_agent.snapshot.schema import (
        SnapshotEntity,
    )

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


def _entity(
    entity_id: str,
    state: str = "on",
    *,
    platform: str | None = None,
    attributes: dict[str, Any] | None = None,
    friendly_name: str | None = None,
    last_changed: datetime | None = None,
) -> SnapshotEntity:
    changed = (last_changed or NOW).isoformat()
    return {
        "entity_id": entity_id,
        "domain": entity_id.partition(".")[0],
        "state": state,
        "friendly_name": friendly_name,
        "area": None,
        "attributes": attributes or {},
        "last_changed": changed,
        "last_updated": changed,
        "platform": platform,
    }


def _token(
    token_id: str,
    *,
    token_type: str = TOKEN_TYPE_LONG_LIVED,
    client_name: str | None = "api",
    last_used_days_ago: int | None = 1,
    ip_key: str | None = "ip-a",
) -> ObservedToken:
    last_used = (
        (NOW - timedelta(days=last_used_days_ago)).isoformat()
        if last_used_days_ago is not None
        else None
    )
    return ObservedToken(
        token_id=token_id,
        user_id="u1",
        token_type=token_type,
        client_name=client_name,
        created_at=(NOW - timedelta(days=400)).isoformat(),
        last_used_at=last_used,
        last_used_ip_key=ip_key,
    )


def _user(
    *tokens: ObservedToken, admin: bool = True, name: str = "Lindo"
) -> ObservedUser:
    return ObservedUser(
        user_id="u1",
        name=name,
        is_admin=admin,
        is_active=True,
        system_generated=False,
        tokens=tuple(tokens),
    )


# ---------------------------------------------------------------------------
# Merge / capability list
# ---------------------------------------------------------------------------


def test_merge_derives_capabilities_and_sources() -> None:
    """Every populated key becomes a dotted capability with its provider."""
    merged = merge_adapter_results(
        [
            AdapterResult(name="generic", posture={"upnp_enabled": True}),
            AdapterResult(
                name="router",
                posture={"upnp_enabled": False},
                ha_security={"admin_user_count": 1},
                clients=[],
            ),
        ]
    )
    assert merged["capabilities"] == sorted(
        [posture_cap("upnp_enabled"), ha_cap("admin_user_count"), CAP_CLIENTS]
    )
    # Later adapter wins for the same key and is recorded as the source.
    assert merged["posture"].get("upnp_enabled") is False
    assert merged["sources"][posture_cap("upnp_enabled")] == "router"
    assert merged["sources"][CAP_CLIENTS] == "router"


def test_merge_without_client_adapter_has_no_client_capability() -> None:
    """No adapter asserting clients means the capability is absent, not empty."""
    merged = merge_adapter_results([AdapterResult(name="ha_native")])
    assert CAP_CLIENTS not in merged["capabilities"]
    assert merged["clients"] == []
    assert empty_network_snapshot("off").get("notes") == ["off"]


# ---------------------------------------------------------------------------
# Pure adapter
# ---------------------------------------------------------------------------


def test_adapter_reports_missing_inputs_as_missing_capabilities() -> None:
    """Inputs that could not be read leave their capability paths out."""
    inputs = HaNativeInputs(now=NOW)
    result = ha_native_adapter(inputs, [], NetworkBuildContext(options=None))
    merged = merge_adapter_results([result])
    caps = set(merged["capabilities"])
    for key in (
        "admin_user_count",
        "failed_login_notification_present",
        "exposed_sensitive_entities",
        "critical_action_pin_enabled",
        "cloud_remote_ui_enabled",
        "http_ip_ban_enabled",
        "trusted_networks_bypass_login",
        "addons_with_host_ports",
        "webhook_automations_public",
        "discovered_unconfigured",
    ):
        assert ha_cap(key) not in caps, key
    # Always derivable from the entity snapshot alone.
    assert ha_cap("unavailable_security_devices") in caps
    assert ha_cap("pending_updates") in caps
    assert posture_cap("router_update_pending") not in caps
    assert any("add-on" in note for note in merged.get("notes", []))


def test_adapter_auth_counts_and_stale_days() -> None:
    """Admin and long-lived token counts plus per-token idle days."""
    inputs = HaNativeInputs(
        now=NOW,
        auth=[
            _user(
                _token("t1", client_name="api", last_used_days_ago=120),
                _token("t2", client_name="api", last_used_days_ago=None),
                _token("t3", token_type=TOKEN_TYPE_NORMAL, client_name="browser"),
            )
        ],
    )
    ha = ha_native_adapter(inputs, [], NetworkBuildContext()).ha_security
    assert ha["admin_user_count"] == 1
    assert ha["long_lived_token_count"] == 2
    # Duplicate client names are disambiguated with an id prefix; age counts
    # from creation because HA never logs long-lived token use.
    assert ha["long_lived_token_age_days"] == {"api (t1)": 400, "api (t2)": 400}
    # No inventory -> no change-detection capabilities.
    assert "new_admin_users" not in ha


def test_adapter_change_fields_only_after_inventory_bootstrap() -> None:
    """Change detection is absent before bootstrap and present after."""
    inventory = AuthInventory(MagicMock())
    inputs = HaNativeInputs(now=NOW, auth=[_user(_token("t1"))])
    context = NetworkBuildContext(auth_inventory=inventory)

    before = ha_native_adapter(inputs, [], context).ha_security
    assert "new_long_lived_tokens" not in before

    inventory._data["bootstrapped_at"] = NOW.isoformat()  # bootstrapped, empty
    after = ha_native_adapter(inputs, [], context).ha_security
    assert after["new_long_lived_tokens"] == ["api"]
    assert after["new_admin_users"] == ["Lindo"]
    assert "refresh_tokens_from_new_ip" not in after


def test_adapter_exposed_entities_filters_to_sensitive_ones() -> None:
    """Only locks, alarms, and entry covers count as sensitive exposure."""
    entities = [
        _entity("lock.front", "locked"),
        _entity("cover.garage", "closed", attributes={"device_class": "garage"}),
        _entity("cover.blinds", "open"),
        _entity("light.hall", "on"),
        _entity("alarm_control_panel.home", "disarmed"),
    ]
    inputs = HaNativeInputs(
        now=NOW,
        exposed={
            "conversation": [
                "lock.front",
                "cover.garage",
                "cover.blinds",
                "light.hall",
                "lock.missing",
            ],
            "cloud.alexa": ["alarm_control_panel.home"],
        },
    )
    ha = ha_native_adapter(inputs, entities, NetworkBuildContext()).ha_security
    assert ha["exposed_sensitive_entities"] == {
        "conversation": ["cover.garage", "lock.front"],
        "cloud.alexa": ["alarm_control_panel.home"],
    }


def test_is_sensitive_entity_cover_by_name_hint() -> None:
    """A cover named like a door counts even without a device class."""
    assert is_sensitive_entity(_entity("cover.side_door", "closed"))
    assert not is_sensitive_entity(_entity("cover.kitchen_shade", "closed"))
    assert is_sensitive_entity(_entity("lock.any", "locked"))


def test_adapter_pin_policy_from_options() -> None:
    """The PIN capability follows the resolved critical-action policy."""
    inputs = HaNativeInputs(now=NOW)
    no_pin = ha_native_adapter(
        inputs,
        [],
        NetworkBuildContext(options={CONF_CRITICAL_ACTION_PIN_ENABLED: True}),
    ).ha_security
    # Enabled but not configured is not enforceable.
    assert no_pin["critical_action_pin_enabled"] is False
    with_pin = ha_native_adapter(
        inputs,
        [],
        NetworkBuildContext(
            options={
                CONF_CRITICAL_ACTION_PIN_ENABLED: True,
                CONF_CRITICAL_ACTION_PIN_HASH: "h",
                CONF_CRITICAL_ACTION_PIN_SALT: "s",
            }
        ),
    ).ha_security
    assert with_pin["critical_action_pin_enabled"] is True


def test_adapter_updates_tag_security_and_router_updates() -> None:
    """Pending updates are split by device class and router platform."""
    entities = [
        _entity("update.router_fw", "on", platform="eero"),
        _entity("update.lock_fw", "on", platform="august"),
        _entity("update.bulb_fw", "on", platform="hue"),
        _entity("update.router_ok", "off", platform="fritz"),
    ]
    inputs = HaNativeInputs(
        now=NOW,
        entity_device={"update.lock_fw": "dev-lock", "update.bulb_fw": "dev-bulb"},
        device_domains={"dev-lock": {"lock", "update"}, "dev-bulb": {"light"}},
    )
    result = ha_native_adapter(inputs, entities, NetworkBuildContext())
    assert result.ha_security["pending_updates"] == [
        "update.bulb_fw",
        "update.lock_fw",
        "update.router_fw",
    ]
    assert result.ha_security["pending_security_updates"] == [
        "update.lock_fw",
        "update.router_fw",
    ]
    assert result.posture["router_update_pending"] is True
    assert result.posture["router_update_entities"] == ["update.router_fw"]


def test_adapter_router_update_capability_requires_a_router_update_entity() -> None:
    """Without any router-platform update entity the posture key is absent."""
    result = ha_native_adapter(
        HaNativeInputs(now=NOW),
        [_entity("update.bulb_fw", "on", platform="hue")],
        NetworkBuildContext(),
    )
    assert "router_update_pending" not in result.posture


def test_adapter_addons_only_running_ones_with_ports() -> None:
    """Stopped add-ons and add-ons without host ports are not exposure."""
    inputs = HaNativeInputs(
        now=NOW,
        addons=[
            AddonInput(
                "core_ssh", "Terminal & SSH", (22,), protected=True, running=True
            ),
            AddonInput(
                "a0d7b954_grafana", "Grafana", (3000,), protected=False, running=False
            ),
            AddonInput(
                "core_mosquitto",
                "Mosquitto",
                (1883, 8883),
                protected=False,
                running=True,
            ),
            AddonInput("core_whisper", "Whisper", (), protected=True, running=True),
            # Details not fetched yet: placeholders must not read as "safe".
            AddonInput(
                "fresh", "Fresh", (), protected=True, running=True, info_known=False
            ),
        ],
    )
    result = ha_native_adapter(inputs, [], NetworkBuildContext())
    ha = result.ha_security
    assert ha["addons_with_host_ports"] == {
        "core_ssh": [22],
        "core_mosquitto": [1883, 8883],
    }
    assert ha["addons_unprotected"] == ["core_mosquitto"]
    assert ha["addon_names"]["core_ssh"] == "Terminal & SSH"
    assert "fresh" in ha["addon_names"]
    assert any("Fresh" in note and "not audited yet" in note for note in result.notes)


def test_adapter_assist_agents_outside_pin_passthrough() -> None:
    """Pipeline agents the PIN does not cover are published verbatim."""
    inputs = HaNativeInputs(
        now=NOW, assist_agents_outside_pin=["conversation.home_assistant"]
    )
    ha = ha_native_adapter(inputs, [], NetworkBuildContext()).ha_security
    assert ha["assist_agents_outside_pin"] == ["conversation.home_assistant"]
    absent = ha_native_adapter(
        HaNativeInputs(now=NOW), [], NetworkBuildContext()
    ).ha_security
    assert "assist_agents_outside_pin" not in absent


def test_adapter_webhook_automations_public_and_critical() -> None:
    """Public webhook triggers are listed; those calling critical actions flagged."""
    public_unlock = {
        "triggers": [{"trigger": "webhook", "webhook_id": "x", "local_only": False}],
        "actions": [{"action": "lock.unlock", "target": {"entity_id": "lock.front"}}],
    }
    public_light = {
        "trigger": [{"platform": "webhook", "webhook_id": "y", "local_only": False}],
        "action": [{"service": "light.turn_on", "entity_id": "light.hall"}],
    }
    local_only = {
        "triggers": [{"trigger": "webhook", "webhook_id": "z"}],
        "actions": [{"action": "lock.unlock", "target": {"entity_id": "lock.front"}}],
    }
    inputs = HaNativeInputs(
        now=NOW,
        automations=[
            AutomationInput("automation.unlock", public_unlock),
            AutomationInput("automation.light", public_light),
            AutomationInput("automation.local", local_only),
        ],
    )
    ha = ha_native_adapter(inputs, [], NetworkBuildContext(options={})).ha_security
    assert ha["webhook_automations_public"] == ["automation.light", "automation.unlock"]
    assert ha["webhook_automations_critical"] == ["automation.unlock"]


def test_automation_helpers_handle_nested_actions_and_defaults() -> None:
    """Nested choose blocks are searched; a missing local_only means local."""
    nested = {
        "actions": [
            {
                "choose": [
                    {
                        "conditions": [],
                        "sequence": [
                            {
                                "action": "cover.open_cover",
                                "target": {"area_id": "garage"},
                            }
                        ],
                    }
                ]
            }
        ]
    }
    critical = [{"domain": "cover", "service": "open_cover", "entity_match": "garage"}]
    # Area target = unresolved -> entity_match rule fails closed (matches).
    assert automation_calls_critical_action(nested, critical)
    assert not automation_calls_critical_action({"actions": []}, critical)
    assert not automation_has_public_webhook(
        {"triggers": [{"trigger": "webhook", "webhook_id": "a"}]}
    )
    assert automation_has_public_webhook(
        {"triggers": {"trigger": "webhook", "webhook_id": "a", "local_only": False}}
    )


def test_adapter_unavailable_security_devices_minutes() -> None:
    """Unavailable locks/alarms/cameras carry minutes since last_changed."""
    entities = [
        _entity("lock.front", "unavailable", last_changed=NOW - timedelta(minutes=45)),
        _entity("camera.yard", "unavailable", last_changed=NOW - timedelta(minutes=5)),
        _entity("camera.ok", "idle"),
        _entity("light.hall", "unavailable"),
    ]
    ha = ha_native_adapter(
        HaNativeInputs(now=NOW), entities, NetworkBuildContext()
    ).ha_security
    assert ha["unavailable_security_devices"] == {"lock.front": 45, "camera.yard": 5}


def test_adapter_http_and_auth_provider_fields() -> None:
    """HTTP and auth-provider inputs map straight through."""
    inputs = HaNativeInputs(
        now=NOW,
        http=HttpInput(
            trusted_proxies_configured=True,
            ip_ban_enabled=False,
            login_attempts_threshold=-1,
        ),
        trusted_networks_bypass_login=True,
        cloud_remote_ui_enabled=True,
        failed_login_notification_present=True,
        discovered_unconfigured=[
            {"handler": "reolink", "source": "dhcp", "title": "cam"}
        ],
        discovered_ignored=[{"handler": "hue", "source": "ignore", "title": "Hue"}],
    )
    ha = ha_native_adapter(inputs, [], NetworkBuildContext()).ha_security
    # Core consumes use_x_forwarded_for at setup and never stores it.
    assert "http_use_x_forwarded_for" not in ha
    assert ha["http_trusted_proxies_configured"] is True
    assert ha["http_ip_ban_enabled"] is False
    assert ha["http_login_attempts_threshold"] == -1
    assert ha["trusted_networks_bypass_login"] is True
    assert ha["cloud_remote_ui_enabled"] is True
    assert ha["failed_login_notification_present"] is True
    assert ha["discovered_unconfigured"][0]["handler"] == "reolink"
    assert ha["discovered_ignored"][0]["handler"] == "hue"


# ---------------------------------------------------------------------------
# Auth observation collector (no secrets leave hass.auth)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_auth_observation_pseudonymizes_and_drops_secrets() -> None:
    """The observation carries ids, names, and HMAC'd addresses only."""
    token = MagicMock()
    token.id = "tok1"
    token.token_type = TOKEN_TYPE_LONG_LIVED
    token.client_name = "api"
    token.created_at = NOW - timedelta(days=3)
    token.last_used_at = NOW
    token.last_used_ip = "192.168.1.50"
    token.token = "SECRET"  # noqa: S105
    token.jwt_key = "SECRET-JWT"
    user = MagicMock()
    user.id = "u1"
    user.name = "Lindo"
    user.is_admin = True
    user.is_active = True
    user.system_generated = False
    user.refresh_tokens = {"tok1": token}
    hass = MagicMock()

    async def _users() -> list[Any]:
        return [user]

    hass.auth.async_get_users = _users
    pseudonymizer = Pseudonymizer("salt")

    observation = await async_collect_auth_observation(hass, pseudonymizer)
    assert observation is not None
    observed = observation[0].tokens[0]
    assert observed.last_used_ip_key == pseudonymizer.ip_key("192.168.1.50")
    assert "192.168" not in repr(observation)
    assert "SECRET" not in repr(observation)

    # Without a pseudonymizer the address is dropped, not stored raw.
    plain = await async_collect_auth_observation(hass, None)
    assert plain is not None
    assert plain[0].tokens[0].last_used_ip_key is None


@pytest.mark.asyncio
async def test_collect_auth_observation_degrades_on_failure() -> None:
    """A failing auth read yields None (capability missing), not an exception."""
    hass = MagicMock()

    async def _boom() -> list[Any]:
        msg = "auth store not ready"
        raise RuntimeError(msg)

    hass.auth.async_get_users = _boom
    assert await async_collect_auth_observation(hass, None) is None


# ---------------------------------------------------------------------------
# Snapshot builder integration (real hass fixture)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_network_snapshot_disabled_has_no_capabilities(hass: Any) -> None:
    """The master switch yields a section with a note and no capabilities."""
    section = await async_build_network_snapshot(
        hass,
        [],
        NetworkBuildContext(enabled=False),
        now=NOW,
        entity_device={},
        device_domains={},
    )
    assert section["capabilities"] == []
    assert section.get("notes")


@pytest.mark.asyncio
async def test_full_snapshot_carries_platform_and_network_section(hass: Any) -> None:
    """The builder fills ``platform`` and always attaches a validated section."""
    from custom_components.home_generative_agent.snapshot.builder import (  # noqa: PLC0415
        async_build_full_state_snapshot,
    )

    hass.states.async_set("lock.front", "unavailable", {"friendly_name": "Front"})
    # No context (baseline, discovery, preview callers): an empty section.
    bare = await async_build_full_state_snapshot(hass)
    validate_snapshot(dict(bare))
    assert bare.get("network", {}).get("capabilities") == []
    snapshot = await async_build_full_state_snapshot(
        hass, network=NetworkBuildContext(options={})
    )
    validate_snapshot(dict(snapshot))
    assert snapshot["schema_version"] == 2
    entity = next(e for e in snapshot["entities"] if e["entity_id"] == "lock.front")
    # No registry entry for a bare state -> platform None, key still present.
    assert entity.get("platform") is None
    network = snapshot.get("network")
    assert network is not None
    assert ha_cap("unavailable_security_devices") in network["capabilities"]
    assert "lock.front" in network["ha_security"].get(
        "unavailable_security_devices", {}
    )
    # The test hass has an auth store and no cloud: auth present, cloud absent.
    assert ha_cap("admin_user_count") in network["capabilities"]
    assert ha_cap("cloud_remote_ui_enabled") not in network["capabilities"]
    assert network["sources"][ha_cap("admin_user_count")] == "ha_native"


# ---------------------------------------------------------------------------
# HA-internals collectors (the booleans every plain-install rule acts on)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_logged_failures() -> None:
    """Clear the once-per-process log guard so tests do not leak into each other."""
    network_mod._LOGGED_INPUT_FAILURES.clear()


def test_input_failure_logs_once_per_input_and_exception_class(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Same input + same error class logs once; a new error class logs again."""
    with caplog.at_level("WARNING"):
        _log_input_failure("thing", RuntimeError("a"))
        _log_input_failure("thing", RuntimeError("b"))
        _log_input_failure("thing", KeyError("c"))
        _log_input_failure("other", RuntimeError("d"))
    messages = [
        r.getMessage() for r in caplog.records if "Network audit" in r.getMessage()
    ]
    assert len(messages) == 3


def test_collect_http_ip_ban_states() -> None:
    """Ban manager presence and threshold are reported as separate facts."""
    from homeassistant.components.http.ban import (  # noqa: PLC0415
        KEY_BAN_MANAGER,
        KEY_LOGIN_THRESHOLD,
    )

    server = MagicMock()
    server.trusted_proxies = []
    server.app = {KEY_BAN_MANAGER: object(), KEY_LOGIN_THRESHOLD: 5}
    hass = MagicMock(http=server)
    assert _collect_http(hass) == HttpInput(
        trusted_proxies_configured=False,
        ip_ban_enabled=True,
        login_attempts_threshold=5,
    )
    # Home Assistant's stock configuration: banning on, no threshold.
    server.app = {KEY_BAN_MANAGER: object(), KEY_LOGIN_THRESHOLD: -1}
    stock = _collect_http(hass)
    assert stock is not None
    assert stock.ip_ban_enabled is True
    assert stock.login_attempts_threshold == -1
    server.app = {}
    off = _collect_http(hass)
    assert off is not None
    assert off.ip_ban_enabled is False
    server.trusted_proxies = ["10.0.0.0/8"]
    assert _collect_http(hass).trusted_proxies_configured is True  # type: ignore[union-attr]
    hass.http = None
    assert _collect_http(hass) is None


def test_collect_auth_providers_bypass_flag() -> None:
    """Only a trusted_networks provider with allow_bypass_login counts."""
    homeassistant_provider = MagicMock(type="homeassistant", config={})
    trusted = MagicMock(type="trusted_networks", config={"allow_bypass_login": True})
    hass = MagicMock()
    hass.auth.auth_providers = [homeassistant_provider]
    assert _collect_auth_providers(hass) is False
    hass.auth.auth_providers = [homeassistant_provider, trusted]
    assert _collect_auth_providers(hass) is True
    trusted.config = {"allow_bypass_login": False}
    assert _collect_auth_providers(hass) is False
    hass.auth = None
    assert _collect_auth_providers(hass) is None


def test_collect_cloud_requires_loaded_logged_in_and_enabled() -> None:
    """Remote UI counts only when cloud is loaded, logged in, and enabled."""
    hass = MagicMock()
    hass.config.components = set()
    assert _collect_cloud(hass) is None
    hass.config.components = {"cloud"}
    hass.data = {}
    assert _collect_cloud(hass) is None
    cloud = MagicMock()
    cloud.client.prefs.remote_enabled = True
    cloud.is_logged_in = False
    hass.data = {"cloud": cloud}
    assert _collect_cloud(hass) is False
    cloud.is_logged_in = True
    assert _collect_cloud(hass) is True


@pytest.mark.asyncio
async def test_collect_failed_login_sees_real_http_login_notification(
    hass: Any,
) -> None:
    """The failed-login fact tracks Home Assistant's own notification id."""
    from homeassistant.components import persistent_notification  # noqa: PLC0415

    assert _collect_failed_login(hass) is False
    persistent_notification.async_create(
        hass, "bad", "Login attempt failed", "http-login"
    )
    await hass.async_block_till_done()
    assert _collect_failed_login(hass) is True
    persistent_notification.async_dismiss(hass, "http-login")
    assert _collect_failed_login(hass) is False


def test_collect_discovery_filters_sources_and_sanitizes_titles() -> None:
    """Only LAN discovery sources count; titles are printable and bounded."""
    hass = MagicMock()
    hass.config_entries.flow.async_progress.return_value = [
        {
            "handler": "reolink",
            "context": {
                "source": "dhcp",
                "title_placeholders": {"name": "cam\u202e\u200b " + "x" * 100},
            },
        },
        {"handler": "hue", "context": {"source": "user"}},
        {"handler": "sonos", "context": {"source": "ssdp"}},
    ]
    ignored_entry = MagicMock(source="ignore", domain="tplink", title="Plug")
    hass.config_entries.async_entries.return_value = [
        ignored_entry,
        MagicMock(source="user", domain="hue", title="Hue"),
    ]
    unconfigured, ignored = _collect_discovery(hass)
    assert unconfigured is not None
    assert [d["handler"] for d in unconfigured] == ["reolink", "sonos"]
    title = unconfigured[0]["title"]
    assert "\u202e" not in title
    assert len(title) <= network_mod.MAX_LABEL_CHARS
    assert title.startswith("cam x")
    assert ignored == [{"handler": "tplink", "source": "ignore", "title": "Plug"}]


def test_sanitize_label_strips_controls_and_caps() -> None:
    """Control/format characters go, whitespace collapses, length is capped."""
    assert sanitize_label("  a\u200b b\n\tc  ") == "a b c"
    assert sanitize_label(None) == ""
    long = sanitize_label("y" * 200)
    assert len(long) == network_mod.MAX_LABEL_CHARS
    assert long.endswith("…")


def test_collect_addons_merges_list_and_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running state comes from the fresh list; ports and protection from info."""
    import homeassistant.helpers.hassio as hassio_helper  # noqa: PLC0415
    from homeassistant.components import hassio  # noqa: PLC0415

    monkeypatch.setattr(hassio_helper, "is_hassio", lambda _hass: True)
    monkeypatch.setattr(
        hassio,
        "get_addons_list",
        lambda _hass: [
            {"slug": "core_ssh", "name": "Terminal & SSH", "state": "started"},
            {"slug": "a0d7b954_grafana", "name": "Grafana", "state": "stopped"},
            {"slug": "fresh", "name": "Just installed", "state": "started"},
            {"name": "no slug"},
        ],
    )
    monkeypatch.setattr(
        hassio,
        "get_addons_info",
        lambda _hass: {
            "core_ssh": {
                "network": {"22/tcp": 22, "80/tcp": None, "flag": True},
                "protected": False,
                "state": "stopped",  # stale: the list wins
            },
            "a0d7b954_grafana": {"network": {"3000/tcp": 3000}, "protected": True},
            "fresh": None,
        },
    )
    addons = _collect_addons(MagicMock())
    assert addons == [
        AddonInput("core_ssh", "Terminal & SSH", (22,), protected=False, running=True),
        AddonInput(
            "a0d7b954_grafana", "Grafana", (3000,), protected=True, running=False
        ),
        AddonInput(
            "fresh",
            "Just installed",
            (),
            protected=True,
            running=True,
            info_known=False,
        ),
    ]


def test_collect_addons_not_ready_and_not_hassio(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Supervisor not ready is silent; a non-Supervisor install reports None."""
    import homeassistant.helpers.hassio as hassio_helper  # noqa: PLC0415
    from homeassistant.components import hassio  # noqa: PLC0415

    monkeypatch.setattr(hassio_helper, "is_hassio", lambda _hass: False)
    assert _collect_addons(MagicMock()) is None

    def _not_ready(_hass: Any) -> Any:
        raise hassio.HassioNotReadyError

    monkeypatch.setattr(hassio_helper, "is_hassio", lambda _hass: True)
    monkeypatch.setattr(hassio, "get_addons_list", _not_ready)
    with caplog.at_level("WARNING"):
        assert _collect_addons(MagicMock()) is None
    assert not [r for r in caplog.records if "Network audit" in r.getMessage()]


def test_is_sensitive_entity_word_boundaries() -> None:
    """'indoor' and 'outdoor' covers are not doors."""
    assert not is_sensitive_entity(_entity("cover.outdoor_awning", "closed"))
    assert not is_sensitive_entity(
        _entity("cover.shade_1", "closed", friendly_name="Indoor Shade")
    )
    assert is_sensitive_entity(_entity("cover.side_door", "closed"))
    assert is_sensitive_entity(
        _entity("cover.c1", "closed", friendly_name="Garage Door Left")
    )


def test_webhook_screener_sees_device_actions_and_turn_on() -> None:
    """The shared allowlist screener catches spellings a service blocklist misses."""
    critical = resolve_critical_action_policy({}).critical_actions
    device_action = {
        "actions": [
            {
                "device_id": "abc",
                "domain": "lock",
                "type": "unlock",
                "entity_id": "lock.front",
            }
        ]
    }
    assert automation_calls_critical_action(device_action, critical)
    # A generic call whose targets cannot be resolved fails closed.
    generic_area = {
        "actions": [
            {"action": "homeassistant.turn_on", "target": {"area_id": "garage"}}
        ]
    }
    assert automation_calls_critical_action(generic_area, critical)
    harmless = {
        "actions": [{"action": "light.turn_on", "target": {"entity_id": "light.a"}}]
    }
    assert not automation_calls_critical_action(harmless, critical)


def test_automation_config_prefers_validated_triggers_and_actions() -> None:
    """Validated (blueprint-substituted) config wins over raw_config."""
    from custom_components.home_generative_agent.snapshot.network import (  # noqa: PLC0415
        _automation_config,
    )

    entity = MagicMock()
    entity.raw_config = {"use_blueprint": {"path": "x.yaml"}}
    entity._trigger_config = [
        {"trigger": "webhook", "webhook_id": "w", "local_only": False}
    ]
    entity.action_script.sequence = [
        {"action": "lock.unlock", "target": {"entity_id": "lock.a"}}
    ]
    config = _automation_config(entity)
    assert config is not None
    assert automation_has_public_webhook(config)
    assert automation_calls_critical_action(
        config, resolve_critical_action_policy({}).critical_actions
    )
    bare = MagicMock(spec=["raw_config"])
    bare.raw_config = {"triggers": []}
    assert _automation_config(bare) == {"triggers": []}
    assert _automation_config(MagicMock(spec=[])) is None


@pytest.mark.asyncio
async def test_collect_exposed_uses_default_exposure(hass: Any) -> None:
    """A cover never configured but default-exposed to Assist counts as exposed."""
    from homeassistant.setup import async_setup_component  # noqa: PLC0415

    from custom_components.home_generative_agent.snapshot.network import (  # noqa: PLC0415
        _collect_exposed,
    )

    assert await async_setup_component(hass, "homeassistant", {})
    hass.states.async_set("cover.garage_door", "closed")
    hass.states.async_set("lock.front", "locked")
    exposed = _collect_exposed(hass, ["cover.garage_door", "lock.front"])
    assert exposed is not None
    # Assist exposes covers by default but not locks; the cloud assistants
    # expose nothing until configured.
    assert exposed["conversation"] == ["cover.garage_door"]
    assert exposed["cloud.alexa"] == []


@pytest.mark.asyncio
async def test_salt_is_persisted_and_reused_across_loads(hass: Any) -> None:
    """The salt survives a reload; removing it produces different keys."""
    from custom_components.home_generative_agent.sentinel.pseudonymizer import (  # noqa: PLC0415
        async_load_pseudonymizer,
        async_remove_pseudonymizer_salt,
    )

    first = await async_load_pseudonymizer(hass)
    await hass.async_block_till_done()
    second = await async_load_pseudonymizer(hass)
    assert first.ip_key("10.0.0.7") == second.ip_key("10.0.0.7")
    assert first.fingerprint == second.fingerprint
    await async_remove_pseudonymizer_salt(hass)
    third = await async_load_pseudonymizer(hass)
    assert third.ip_key("10.0.0.7") != first.ip_key("10.0.0.7")


@pytest.mark.asyncio
async def test_salt_load_failure_yields_temporary_salt(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A storage failure degrades to a usable temporary salt with a warning."""
    from custom_components.home_generative_agent.sentinel import (  # noqa: PLC0415
        pseudonymizer as pmod,
    )

    class _BrokenStore:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        async def async_load(self) -> Any:
            msg = "disk"
            raise OSError(msg)

    monkeypatch.setattr(pmod, "Store", _BrokenStore)
    with caplog.at_level("WARNING"):
        temp = await pmod.async_load_pseudonymizer(MagicMock())
    assert len(temp.ip_key("1.2.3.4")) == 8
    assert any("temporary salt" in r.getMessage() for r in caplog.records)


def test_collect_assist_agents_excludes_own_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only pipelines on agents other than this integration's are returned."""
    import sys  # noqa: PLC0415
    from types import SimpleNamespace  # noqa: PLC0415

    from homeassistant.helpers import entity_registry as er  # noqa: PLC0415

    # The real package drags in the voice stack (hassil), which the test
    # venv does not ship; the collector only needs async_get_pipelines.
    assist_pipeline = SimpleNamespace(async_get_pipelines=lambda _hass: [])
    monkeypatch.setitem(
        sys.modules, "homeassistant.components.assist_pipeline", assist_pipeline
    )

    from custom_components.home_generative_agent.const import DOMAIN  # noqa: PLC0415
    from custom_components.home_generative_agent.snapshot.network import (  # noqa: PLC0415
        _collect_assist_agents,
    )

    own = SimpleNamespace(
        entity_id="conversation.hga", domain="conversation", platform=DOMAIN
    )
    other = SimpleNamespace(
        entity_id="conversation.openai", domain="conversation", platform="openai"
    )
    registry = SimpleNamespace(entities={"a": own, "b": other})
    monkeypatch.setattr(er, "async_get", lambda _hass: registry)
    pipelines = [
        SimpleNamespace(conversation_engine="conversation.hga"),
        SimpleNamespace(conversation_engine="conversation.home_assistant"),
        SimpleNamespace(conversation_engine="conversation.openai"),
        SimpleNamespace(conversation_engine="conversation.hga"),
    ]
    monkeypatch.setattr(assist_pipeline, "async_get_pipelines", lambda _hass: pipelines)
    assert _collect_assist_agents(MagicMock()) == [
        "conversation.home_assistant",
        "conversation.openai",
    ]
    monkeypatch.setattr(
        assist_pipeline, "async_get_pipelines", lambda _hass: pipelines[:1]
    )
    assert _collect_assist_agents(MagicMock()) == []

    def _not_loaded(_hass: Any) -> list[Any]:
        key = "assist_pipeline"
        raise KeyError(key)

    monkeypatch.setattr(assist_pipeline, "async_get_pipelines", _not_loaded)
    assert _collect_assist_agents(MagicMock()) == []
