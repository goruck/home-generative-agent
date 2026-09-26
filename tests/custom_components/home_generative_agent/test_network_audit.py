# ruff: noqa: S101
"""On-demand network audit: engine method, report shape, agent tool, service."""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import yaml

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.agent.tools import audit_home_security
from custom_components.home_generative_agent.const import (
    CONF_SENTINEL_NETWORK_AUDIT_SHARE_DETAILS,
    CONF_SENTINEL_NETWORK_ENABLED,
    CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS,
    NETWORK_AUDIT_REPORT_NOTIFICATION_ID,
    NETWORK_AUDIT_TOOL_DIGEST_COVERAGE_NOTE,
    NETWORK_AUDIT_TOOL_DIGEST_NOTE,
    NETWORK_AUDIT_TOOL_DIGEST_NOTE_NOT_ADMIN,
    NETWORK_AUDIT_TOOL_DIGEST_NOTE_UNDELIVERED,
    NETWORK_AUDIT_TOOL_LABEL_NOTE,
    NETWORK_AUDIT_TOOL_MAX_ENTITIES,
    NETWORK_AUDIT_TOOL_MAX_SUMMARY_CHARS,
    NETWORK_AUDIT_TOOL_PROMPT,
)
from custom_components.home_generative_agent.sentinel.engine import SentinelEngine
from custom_components.home_generative_agent.sentinel.network_audit import (
    PRIVACY_NOTES,
    build_report,
    capability_reason,
    digest,
    empty_report,
    finding_title,
    render_report_markdown,
    summarize,
)
from custom_components.home_generative_agent.sentinel.notifier import escape_markdown
from custom_components.home_generative_agent.sentinel.rules.network_common import (
    NETWORK_RULE_TYPES,
    make_finding,
)
from custom_components.home_generative_agent.sentinel.suppression import (
    SuppressionManager,
    SuppressionState,
)
from custom_components.home_generative_agent.snapshot.network import (
    CAP_CLIENT_DATA_DAY,
    CAP_CLIENTS,
    CAP_COUNTER_BASELINES,
    CAP_NEW_CLIENTS,
    NetworkBuildContext,
    ha_cap,
    posture_cap,
    radio_cap,
)
from custom_components.home_generative_agent.snapshot.schema import validate_snapshot

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.audit.store import AuditStore
    from custom_components.home_generative_agent.sentinel.notifier import (
        SentinelNotifier,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

NOW = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)
_COMPONENT_DIR = (
    Path(__file__).resolve().parents[3] / "custom_components" / "home_generative_agent"
)


class DummySuppression(SuppressionManager):
    """Suppression manager stub."""

    def __init__(self) -> None:  # type: ignore[override]
        self._state = SuppressionState()

    @property
    def state(self) -> SuppressionState:  # type: ignore[override]
        return self._state

    @property
    def is_read_only(self) -> bool:  # type: ignore[override]
        return False

    async def async_save(self) -> None:  # type: ignore[override]
        return None


class DummyNotifier:
    """Notifier stub recording every dispatched finding."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def async_notify(self, finding, snapshot, explanation) -> None:  # type: ignore[no-untyped-def]
        self.calls.append(finding)


class DummyAudit:
    """Audit store stub recording every appended finding."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def async_append_finding(  # type: ignore[no-untyped-def]
        self, snapshot, finding, explanation, **kwargs: Any
    ) -> None:
        self.calls.append(finding)


def _snapshot(
    ha_security: dict[str, Any] | None = None,
    *,
    entities: list[dict[str, Any]] | None = None,
    notes: list[str] | None = None,
) -> FullStateSnapshot:
    ha = ha_security or {}
    caps = sorted(ha_cap(k) for k in ha)
    return validate_snapshot(
        {
            "schema_version": 2,
            "generated_at": NOW.isoformat(),
            "entities": entities or [],
            "camera_activity": [],
            "derived": {
                "now": NOW.isoformat(),
                "timezone": "UTC",
                # Night + nobody home would fire the non-network static rules
                # on a lock entity; the audit must never include those.
                "is_night": True,
                "anyone_home": False,
                "people_home": [],
                "people_away": ["Lindo"],
                "last_motion_by_area": {},
            },
            "network": {
                "capabilities": caps,
                "sources": dict.fromkeys(caps, "ha_native"),
                "clients": [],
                "posture": {},
                "ha_security": ha,
                "counters": {},
                "notes": notes or [],
            },
        }
    )


def _unlocked_lock() -> dict[str, Any]:
    return {
        "entity_id": "lock.front_door",
        "domain": "lock",
        "state": "unlocked",
        "friendly_name": "Front Door",
        "area": None,
        "attributes": {},
        "last_changed": "2026-09-11T01:00:00+00:00",
        "last_updated": "2026-09-11T01:00:00+00:00",
        "platform": None,
    }


def _engine(
    monkeypatch: pytest.MonkeyPatch,
    snapshot: FullStateSnapshot,
    *,
    options: dict[str, Any] | None = None,
    baseline_updater: Any = None,
) -> tuple[SentinelEngine, DummyNotifier, DummyAudit, list[NetworkBuildContext]]:
    contexts: list[NetworkBuildContext] = []

    async def _fake_build(
        _hass: HomeAssistant, *, network: NetworkBuildContext | None = None
    ) -> FullStateSnapshot:
        assert network is not None
        contexts.append(network)
        return snapshot

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine."
        "async_build_full_state_snapshot",
        _fake_build,
    )
    # _timed_run fires the run-complete dispatcher signal, which needs a real
    # hass; the stub engine has none.
    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine.async_dispatcher_send",
        lambda *_args, **_kwargs: None,
    )
    notifier = DummyNotifier()
    audit = DummyAudit()
    engine = SentinelEngine(
        hass=cast("HomeAssistant", object()),
        options={
            "sentinel_cooldown_minutes": 0,
            "sentinel_entity_cooldown_minutes": 0,
            "sentinel_interval_seconds": 60,
            "explain_enabled": False,
            **(options or {}),
        },
        suppression=DummySuppression(),
        notifier=cast("SentinelNotifier", notifier),
        audit_store=cast("AuditStore", audit),
        explainer=None,
        baseline_updater=baseline_updater,
    )
    return engine, notifier, audit, contexts


# ---------------------------------------------------------------------------
# Report module
# ---------------------------------------------------------------------------


def test_capability_reasons_name_what_would_unlock_the_check() -> None:
    assert "router integration" in capability_reason(
        posture_cap("router_update_pending")
    )
    assert "Supervisor" in capability_reason(ha_cap("addons_with_host_ports"))
    assert "Supervisor" in capability_reason(ha_cap("addons_unprotected"))
    assert "Cloud" in capability_reason(ha_cap("cloud_remote_ui_enabled"))
    assert "inventory" in capability_reason(ha_cap("new_admin_users"))
    assert "did not return" in capability_reason(
        ha_cap("failed_login_notification_present")
    )
    assert "source_type router" in capability_reason(CAP_CLIENTS)
    assert "device inventory" in capability_reason(CAP_NEW_CLIENTS)
    # More specific than the clients prefix it shares, so it is matched first.
    assert "guest" in capability_reason("network.clients.is_guest")
    assert "router or DNS" in capability_reason(posture_cap("wpa3_enabled"))
    assert "SSDP" in capability_reason(posture_cap("upnp_enabled"))
    assert "port-mapping count sensor" in capability_reason(
        posture_cap("upnp_port_mappings_added")
    )
    assert "external IP sensor" in capability_reason(posture_cap("public_ip_changed"))
    assert "Zigbee2MQTT" in capability_reason(radio_cap("posture.zigbee_permit_join"))
    assert "ZHA" in capability_reason(radio_cap("posture.zigbee_permit_join"))
    assert "Z-Wave JS" in capability_reason(radio_cap("devices.security_class"))
    assert "Z-Wave JS" in capability_reason(radio_cap("posture.zwave_inclusion_active"))
    assert "Bluetooth proxy" in capability_reason(
        radio_cap("posture.coordinator_update_pending")
    )
    assert "inventory" in capability_reason(radio_cap("new_devices"))
    assert "did not return" in capability_reason(radio_cap("devices"))
    assert "not provided" in capability_reason("network.something_else")


def test_build_report_orders_by_severity_and_explains_inactive_rules() -> None:
    low = make_finding(
        "ha_cloud_remote_ui_enabled",
        severity="low",
        evidence={"enabled": True},
        summary="Cloud remote UI is on.",
        suggested_actions=["Turn it off"],
    )
    high = make_finding(
        "ha_addon_unprotected",
        severity="high",
        evidence={"addons": ["ssh"]},
        summary="1 add-on runs unprotected: SSH.",
        suggested_actions=["Enable protection mode"],
    )
    medium = make_finding(
        "ha_failed_logins",
        severity="medium",
        evidence={"present": True},
        summary="Failed logins seen.",
        suggested_actions=["Check the notification"],
    )
    report = build_report(
        now=NOW,
        findings=[low, high, medium],
        checks_run=["ha_failed_logins", "ha_addon_unprotected"],
        inactive_rules={
            "network_router_update_pending": [posture_cap("router_update_pending")],
            "ha_addon_exposed_port": [ha_cap("addons_with_host_ports")],
        },
        failed_rules=["broken_rule"],
        capabilities={ha_cap("failed_login_notification_present")},
        notes=["Home Assistant Cloud is not loaded; remote UI not audited."],
    )
    assert report["status"] == "ok"
    assert [f["severity"] for f in report["findings"]] == ["high", "medium", "low"]
    first = report["findings"][0]
    assert first["summary"] == "1 add-on runs unprotected: SSH."
    # The summary is lifted out of evidence so it is not repeated.
    assert "summary" not in first["evidence"]
    assert first["evidence"] == {"addons": ["ssh"]}
    assert report["checks_run"] == ["ha_addon_unprotected", "ha_failed_logins"]
    assert "Supervisor" in report["checks_not_run"]["ha_addon_exposed_port"]
    assert "router" in report["checks_not_run"]["network_router_update_pending"]
    assert "raised an error" in report["checks_not_run"]["broken_rule"]
    assert set(report["missing_capabilities"]) == {
        posture_cap("router_update_pending"),
        ha_cap("addons_with_host_ports"),
    }
    assert report["privacy_notes"] == list(PRIVACY_NOTES)
    assert summarize(report) == (
        "3 findings (1 high, 1 medium, 1 low); 2 checks ran, 3 could not run."
    )


def test_summarize_clean_report_and_empty_report() -> None:
    clean = build_report(
        now=NOW, findings=[], checks_run=["a"], inactive_rules={}, capabilities=[]
    )
    assert summarize(clean) == "No findings; 1 check ran."
    disabled = empty_report("disabled", NOW, "off")
    assert disabled["status"] == "disabled"
    assert disabled["notes"] == ["off"]
    assert summarize(disabled) == "No findings; 0 checks ran."


# ---------------------------------------------------------------------------
# Engine method
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_reports_disabled_without_building_a_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _, _, contexts = _engine(
        monkeypatch, _snapshot(), options={CONF_SENTINEL_NETWORK_ENABLED: False}
    )
    report = await engine.async_audit_network()
    assert report["status"] == "disabled"
    assert contexts == []
    assert "turned off" in report["notes"][0]


@pytest.mark.asyncio
async def test_audit_runs_only_network_rules_live_and_dispatches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Findings come back in the report; no notification, audit row, or cooldown."""
    snapshot = _snapshot(
        {"failed_login_notification_present": True, "cloud_remote_ui_enabled": True},
        entities=[_unlocked_lock()],
        notes=["Supervisor unavailable; add-ons not audited."],
    )
    engine, notifier, audit, contexts = _engine(monkeypatch, snapshot)

    report = await engine.async_audit_network()

    assert report["status"] == "ok"
    assert len(contexts) == 1
    assert contexts[0].enabled is True
    types = [f["type"] for f in report["findings"]]
    assert types == ["ha_failed_logins", "ha_cloud_remote_ui_enabled"]
    # The unlocked lock at night while away would fire unlocked_lock_at_night
    # in a scheduled run; the audit is the network family only.
    assert "unlocked_lock_at_night" not in types
    assert set(report["checks_run"]) == {
        "ha_failed_logins",
        "ha_cloud_remote_ui_enabled",
    }
    assert set(report["checks_not_run"]) == NETWORK_RULE_TYPES - set(
        report["checks_run"]
    )
    assert report["notes"] == ["Supervisor unavailable; add-ons not audited."]
    assert report["capabilities"] == [
        ha_cap("cloud_remote_ui_enabled"),
        ha_cap("failed_login_notification_present"),
    ]
    assert all(f["summary"] for f in report["findings"])
    # Side-effect free.
    assert notifier.calls == []
    assert audit.calls == []
    assert engine.run_stats.get("inactive_rules") is None
    assert cast("Any", engine)._suppression.state.last_by_type == {}


@pytest.mark.asyncio
async def test_failed_rule_is_reported_the_same_way_by_cycle_and_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rule whose evaluate() raises is 'failed' on the sensor and in the report."""
    engine, _, _, _ = _engine(
        monkeypatch, _snapshot({"failed_login_notification_present": True})
    )
    rule = next(
        r for r in cast("Any", engine)._rules if r.rule_id == "ha_failed_logins"
    )

    def _boom(_snapshot: Any) -> Any:
        raise KeyError

    monkeypatch.setattr(rule, "evaluate", _boom)

    await engine._timed_run()
    assert engine.run_stats["failed_rules"] == ["ha_failed_logins"]
    assert "ha_failed_logins" not in engine.run_stats["inactive_rules"]

    report = await engine.async_audit_network()
    assert report["status"] == "ok"
    assert "ha_failed_logins" not in report["checks_run"]
    assert "raised an error" in report["checks_not_run"]["ha_failed_logins"]
    assert "ha_failed_logins" not in report["missing_capabilities"]


@pytest.mark.asyncio
async def test_audit_applies_entity_exclusions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(
        {
            "webhook_automations_public": ["automation.gate"],
            "webhook_automations_critical": [],
        }
    )
    engine, _, _, _ = _engine(
        monkeypatch,
        snapshot,
        options={
            CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS: {
                "ha_webhook_automation_public": ["automation.gate"]
            }
        },
    )
    report = await engine.async_audit_network()
    assert "ha_webhook_automation_public" in report["checks_run"]
    assert report["findings"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [ValueError, RuntimeError, AttributeError, OSError])
async def test_audit_reports_unavailable_when_anything_fails(
    monkeypatch: pytest.MonkeyPatch, exc: type[Exception]
) -> None:
    """The boundary degrades every ordinary failure to the documented shape."""
    engine, _, _, _ = _engine(monkeypatch, _snapshot())

    async def _boom(_hass: Any, *, network: Any = None) -> Any:
        _ = network
        raise exc

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine."
        "async_build_full_state_snapshot",
        _boom,
    )
    report = await engine.async_audit_network()
    assert report["status"] == "unavailable"
    assert report["findings"] == []
    assert set(report) == set(
        build_report(
            now=NOW, findings=[], checks_run=[], inactive_rules={}, capabilities=[]
        )
    )


@pytest.mark.asyncio
async def test_audit_is_cancellable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation is not swallowed by the failure boundary."""
    engine, _, _, _ = _engine(monkeypatch, _snapshot())

    async def _hang(_hass: Any, *, network: Any = None) -> Any:
        _ = network
        await asyncio.sleep(60)

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine."
        "async_build_full_state_snapshot",
        _hang,
    )
    task = asyncio.create_task(engine.async_audit_network())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_audit_waits_for_a_cycle_in_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit takes the scheduler's single-flight lock instead of racing it."""
    engine, _, _, contexts = _engine(
        monkeypatch, _snapshot({"failed_login_notification_present": True})
    )
    scheduler = cast("Any", engine)._trigger_scheduler
    await scheduler._lock.acquire()
    task = asyncio.create_task(engine.async_audit_network())
    await asyncio.sleep(0.01)
    assert not task.done()
    assert contexts == []
    scheduler._lock.release()
    report = await asyncio.wait_for(task, timeout=2)
    assert report["status"] == "ok"
    assert len(contexts) == 1
    assert not scheduler._lock.locked()


# ---------------------------------------------------------------------------
# Agent tool
# ---------------------------------------------------------------------------


def _tool_config(sentinel: Any) -> Any:
    return {"configurable": {"hga_runtime_data": SimpleNamespace(sentinel=sentinel)}}


async def _run_tool(config: Any) -> str:
    return await audit_home_security.coroutine(config=config)  # type: ignore[misc]


@pytest.mark.asyncio
async def test_tool_explains_when_sentinel_is_not_enabled() -> None:
    assert "Sentinel is not enabled" in await _run_tool(_tool_config(None))
    assert "Sentinel is not enabled" in await _run_tool({})


@pytest.mark.asyncio
async def test_tool_explains_when_audit_is_disabled_or_unavailable() -> None:
    async def _disabled() -> Any:
        return empty_report("disabled", NOW, "off")

    async def _unavailable() -> Any:
        return empty_report("unavailable", NOW, "snapshot failed")

    text = await _run_tool(_tool_config(SimpleNamespace(async_audit_network=_disabled)))
    # The engine's note is echoed, so the tool and the service agree.
    assert text == "The security audit could not run: off"
    text = await _run_tool(
        _tool_config(SimpleNamespace(async_audit_network=_unavailable))
    )
    assert text == "The security audit could not run: snapshot failed"


@pytest.mark.asyncio
async def test_tool_renders_findings_by_severity_with_checks_not_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(
        {
            "failed_login_notification_present": True,
            "addons_unprotected": ["ssh"],
            "addon_names": {"ssh": "Terminal & SSH"},
        }
    )
    engine, _, _, _ = _engine(monkeypatch, snapshot)

    text = await _run_tool(_tool_config(engine))
    payload = yaml.safe_load(text)

    assert payload["summary"].startswith("2 findings (1 high, 1 medium)")
    assert [f["severity"] for f in payload["findings"]] == ["high", "medium"]
    assert payload["findings"][0]["type"] == "ha_addon_unprotected"
    assert "Terminal & SSH" in payload["findings"][0]["summary"]
    assert payload["findings"][0]["suggested_actions"]
    # The tool hands the model summaries and actions, not raw evidence.
    assert "evidence" not in payload["findings"][0]
    assert "ha_addon_unprotected" in payload["checks_run"]
    # No host-port data was provided, so its sibling rule is reported, not
    # assumed to pass.
    assert "Supervisor" in payload["checks_not_run"]["ha_addon_exposed_port"]
    assert "router" in payload["checks_not_run"]["network_router_update_pending"]
    assert payload["privacy_notes"] == list(PRIVACY_NOTES)
    assert payload["notes"][0] == NETWORK_AUDIT_TOOL_LABEL_NOTE
    assert set(payload) == {
        "generated_at",
        "summary",
        "findings",
        "checks_run",
        "checks_not_run",
        "notes",
        "privacy_notes",
    }


@pytest.mark.asyncio
async def test_tool_caps_what_reaches_the_model_and_labels_names_as_data() -> None:
    long_summary = "x" * (NETWORK_AUDIT_TOOL_MAX_SUMMARY_CHARS + 50)
    entities = [f"light.l{i}" for i in range(NETWORK_AUDIT_TOOL_MAX_ENTITIES + 5)]
    finding = make_finding(
        "security_device_unavailable",
        severity="high",
        evidence={"entities": entities},
        summary=long_summary,
        suggested_actions=["Check them"],
        triggering_entities=entities,
    )
    report = build_report(
        now=NOW,
        findings=[finding],
        checks_run=["security_device_unavailable"],
        inactive_rules={},
        capabilities=[],
        notes=["n" * 2000, *["note"] * 40],
    )

    async def _audit() -> Any:
        return report

    sentinel = SimpleNamespace(
        async_audit_network=_audit, network_audit_share_details=True
    )
    text = await _run_tool(_tool_config(sentinel))
    payload = yaml.safe_load(text)
    rendered = payload["findings"][0]
    assert len(rendered["summary"]) == NETWORK_AUDIT_TOOL_MAX_SUMMARY_CHARS
    assert rendered["summary"].endswith("…")
    assert len(rendered["triggering_entities"]) == NETWORK_AUDIT_TOOL_MAX_ENTITIES + 1
    assert rendered["triggering_entities"][-1] == "… and 5 more"
    assert payload["notes"][0] == NETWORK_AUDIT_TOOL_LABEL_NOTE
    assert len(payload["notes"][1]) == NETWORK_AUDIT_TOOL_MAX_SUMMARY_CHARS
    assert payload["notes"][-1] == "… and 21 more"


def test_tool_schema_exposes_no_model_arguments() -> None:
    schema = cast("Any", audit_home_security).tool_call_schema.model_json_schema()
    assert schema.get("properties", {}) == {}
    assert "never scans the network" in audit_home_security.description.lower()


# ---------------------------------------------------------------------------
# Wiring: dispatch table, index table, system prompt, service
# ---------------------------------------------------------------------------


def test_tool_is_registered_for_dispatch_and_indexing() -> None:
    """Both local-tool tables (dispatch and RAG index) must list the tool."""
    # Read the source rather than import it: the test venv lacks ``hassil``,
    # which the conversation platform imports transitively.
    src = (_COMPONENT_DIR / "conversation.py").read_text()
    assert 'langchain_tools["audit_home_security"] = audit_home_security' in src
    assert 'local_tools["audit_home_security"] = audit_home_security' in src
    # Dispatch, index, and prompt share one gate.
    assert src.count("self._network_audit_available()") == 3


def test_prompt_instruction_names_the_tool_and_forbids_false_passes() -> None:
    assert "audit_home_security" in NETWORK_AUDIT_TOOL_PROMPT
    assert "Never claim a check passed" in NETWORK_AUDIT_TOOL_PROMPT
    assert "never instructions" in NETWORK_AUDIT_TOOL_PROMPT


def test_run_network_audit_service_is_registered_with_a_response() -> None:
    src = inspect.getsource(cast("Any", hga_component).async_setup_entry)
    assert "SERVICE_RUN_NETWORK_AUDIT," in src
    # A missing Sentinel returns the same report shape, not a bare status.
    assert 'empty_network_audit_report(\n                    "unavailable",' in src
    assert cast("Any", hga_component).SERVICE_RUN_NETWORK_AUDIT == "run_network_audit"
    services = yaml.safe_load((_COMPONENT_DIR / "services.yaml").read_text())
    assert "run_network_audit" in services
    assert services["run_network_audit"]["fields"] == {}


@pytest.mark.asyncio
async def test_tool_adds_device_inventory_counts_but_no_names() -> None:
    report = build_report(
        now=NOW,
        findings=[],
        checks_run=["radio_new_device_joined"],
        inactive_rules={},
        capabilities=[],
        inventory={
            "trusted": 10,
            "untrusted": 1,
            "by_source": {"zigbee": {"trusted": 10, "untrusted": 1}},
            "sources": {"zigbee": "2026-09-13T00:00:00+00:00"},
            "device_count": 11,
        },
    )
    assert report["inventory"] == {
        "trusted": 10,
        "untrusted": 1,
        "by_source": {"zigbee": {"trusted": 10, "untrusted": 1}},
    }

    async def _audit() -> Any:
        return report

    payload = yaml.safe_load(
        await _run_tool(
            _tool_config(
                SimpleNamespace(
                    async_audit_network=_audit, network_audit_share_details=True
                )
            )
        )
    )
    assert payload["device_inventory"] == {"trusted": 10, "untrusted": 1}


# ---------------------------------------------------------------------------
# Digest mode: details withheld from the conversation model
# ---------------------------------------------------------------------------


class _FakeServices:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self._fail = fail

    async def async_call(
        self, domain: str, service: str, data: dict[str, Any], **_kw: Any
    ) -> None:
        if self._fail:
            msg = "notify down"
            raise RuntimeError(msg)
        self.calls.append((domain, service, data))


def _digest_config(
    sentinel: Any, services: _FakeServices | None, *, admin: bool = True
) -> Any:
    hass = SimpleNamespace(services=services) if services is not None else None
    return {
        "configurable": {
            "hga_runtime_data": SimpleNamespace(sentinel=sentinel),
            "hass": hass,
            "requester_is_admin": admin,
        }
    }


_LEAKY_SNAPSHOT_FIELDS: dict[str, Any] = {
    "failed_login_notification_present": True,
    "addons_unprotected": ["ssh"],
    "addon_names": {"ssh": "Terminal & SSH"},
}


@pytest.mark.asyncio
async def test_digest_mode_gives_the_model_nothing_copied_from_the_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _, _, _ = _engine(
        monkeypatch,
        _snapshot(_LEAKY_SNAPSHOT_FIELDS),
        options={CONF_SENTINEL_NETWORK_AUDIT_SHARE_DETAILS: False},
    )
    services = _FakeServices()

    text = await _run_tool(_digest_config(engine, services))
    payload = yaml.safe_load(text)

    assert payload["details"] == "withheld"
    assert payload["summary"].startswith("2 findings (1 high, 1 medium)")
    assert payload["findings"][0] == {
        "severity": "high",
        "type": "ha_addon_unprotected",
        "title": finding_title("ha_addon_unprotected"),
    }
    # Fixed per-rule copy only: no summary, action, entity id, or note.
    assert all(set(f) == {"severity", "type", "title"} for f in payload["findings"])
    assert "Terminal" not in text
    assert payload["notes"] == [NETWORK_AUDIT_TOOL_DIGEST_NOTE]
    assert "ha_addon_unprotected" in payload["checks_run"]
    assert "Supervisor" in payload["checks_not_run"]["ha_addon_exposed_port"]
    assert payload["privacy_notes"] == list(PRIVACY_NOTES)

    # The owner gets the report itself, names included, as one replaceable
    # persistent notification.
    ((domain, service, data),) = services.calls
    assert (domain, service) == ("persistent_notification", "create")
    assert data["notification_id"] == NETWORK_AUDIT_REPORT_NOTIFICATION_ID
    assert data["title"] == "Security audit report"
    assert "Terminal & SSH" in data["message"].replace("\\", "")
    assert "**High**" in data["message"]
    assert "Checks that could not run" in data["message"]


@pytest.mark.asyncio
async def test_digest_mode_posts_the_report_only_for_an_administrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Persistent notifications reach every signed-in user, and a guest at a
    # voice satellite must not be able to publish or overwrite the report.
    engine, _, _, _ = _engine(
        monkeypatch,
        _snapshot(_LEAKY_SNAPSHOT_FIELDS),
        options={CONF_SENTINEL_NETWORK_AUDIT_SHARE_DETAILS: False},
    )
    services = _FakeServices()
    config = _digest_config(engine, services, admin=False)
    payload = yaml.safe_load(await _run_tool(config))
    assert services.calls == []
    assert payload["notes"][0] == NETWORK_AUDIT_TOOL_DIGEST_NOTE_NOT_ADMIN
    assert payload["details"] == "withheld"
    # No identity at all (the key missing) is not an administrator either.
    del config["configurable"]["requester_is_admin"]
    await _run_tool(config)
    assert services.calls == []


@pytest.mark.asyncio
async def test_the_privacy_gate_fails_closed() -> None:
    # A sentinel object that cannot say details may be shared gets the digest.
    finding = make_finding(
        "ha_addon_unprotected",
        severity="high",
        evidence={"addons": ["ssh"]},
        summary="Terminal & SSH runs unprotected.",
        suggested_actions=["Turn protection mode on"],
    )

    async def _audit() -> Any:
        return build_report(
            now=NOW,
            findings=[finding],
            checks_run=["ha_addon_unprotected"],
            inactive_rules={},
            capabilities=[],
        )

    for sentinel in (
        SimpleNamespace(async_audit_network=_audit),
        SimpleNamespace(async_audit_network=_audit, network_audit_share_details="yes"),
    ):
        text = await _run_tool(_digest_config(sentinel, _FakeServices()))
        assert yaml.safe_load(text)["details"] == "withheld"
        assert "Terminal" not in text


def test_digest_counts_coverage_notes_without_repeating_them() -> None:
    # A check that ran on incomplete data says so in a note; the note names
    # things from the home, so the model is told how many there are, and not
    # to call the home fully checked.
    report = build_report(
        now=NOW,
        findings=[],
        checks_run=["ha_addon_unprotected", "ha_addon_exposed_port"],
        inactive_rules={},
        capabilities=[],
        notes=["Add-on 'Secret Vault' details were not fetched; not audited."],
    )
    payload = digest(report, NETWORK_AUDIT_TOOL_DIGEST_NOTE)
    assert payload["summary"].startswith("No findings")
    assert payload["coverage_notes_withheld"] == 1
    assert payload["notes"] == [
        NETWORK_AUDIT_TOOL_DIGEST_NOTE,
        NETWORK_AUDIT_TOOL_DIGEST_COVERAGE_NOTE.format(count=1),
    ]
    assert "Secret Vault" not in yaml.dump(payload)
    clean = build_report(
        now=NOW, findings=[], checks_run=["a"], inactive_rules={}, capabilities=[]
    )
    assert "coverage_notes_withheld" not in digest(clean, "n")


@pytest.mark.asyncio
async def test_digest_mode_says_so_when_the_report_could_not_be_posted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _, _, _ = _engine(
        monkeypatch,
        _snapshot(_LEAKY_SNAPSHOT_FIELDS),
        options={CONF_SENTINEL_NETWORK_AUDIT_SHARE_DETAILS: False},
    )
    for config in (
        _digest_config(engine, _FakeServices(fail=True)),
        _digest_config(engine, None),
    ):
        payload = yaml.safe_load(await _run_tool(config))
        assert payload["notes"] == [NETWORK_AUDIT_TOOL_DIGEST_NOTE_UNDELIVERED]
        assert payload["details"] == "withheld"
        assert "Terminal" not in yaml.dump(payload)


@pytest.mark.asyncio
async def test_details_are_shared_by_default_and_nothing_is_posted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _, _, _ = _engine(monkeypatch, _snapshot(_LEAKY_SNAPSHOT_FIELDS))
    services = _FakeServices()
    payload = yaml.safe_load(await _run_tool(_digest_config(engine, services)))
    assert "details" not in payload
    assert "Terminal & SSH" in payload["findings"][0]["summary"]
    assert services.calls == []


def test_every_network_rule_has_a_fixed_title_for_the_digest() -> None:
    # A missing label would fall back to the rule id with spaces, which is
    # still name-free, but the digest should read like the notifications do.
    for rule_id in sorted(NETWORK_RULE_TYPES):
        assert finding_title(rule_id) != rule_id.replace("_", " "), rule_id


def test_report_markdown_escapes_names_from_the_lan_and_is_capped() -> None:
    finding = make_finding(
        "network_unconfigured_discovered_device",
        severity="low",
        evidence={"handlers": ["x"]},
        summary="Found [click](http://evil) <b>cam</b>.",
        suggested_actions=["Configure or ignore it"],
    )
    report = build_report(
        now=NOW,
        findings=[finding],
        checks_run=["network_unconfigured_discovered_device"],
        inactive_rules={},
        failed_rules=[],
        capabilities=[],
        notes=["Gateway *Evil* seen"],
        inventory=None,
    )
    text = render_report_markdown(report, None, escape_markdown, 12000)
    assert "\\[click\\](http://evil)" in text
    assert "\\<b\\>cam\\</b\\>" in text
    assert "Gateway \\*Evil\\* seen" in text
    assert "  - Configure or ignore it" in text
    assert "**Notes**\n- Gateway" in text
    # The chrome follows Home Assistant's language; the facts stay as written.
    czech = SimpleNamespace(config=SimpleNamespace(language="cs"))
    text_cs = render_report_markdown(report, cast("Any", czech), escape_markdown, 12000)
    assert "**Nízká**" in text_cs
    # Cut between lines, so no bold or escape pair is left half open.
    short = render_report_markdown(report, None, escape_markdown, 120)
    assert len(short) <= 120
    assert short.endswith("run the run_network_audit service for the rest.")
    assert short.count("**") % 2 == 0
    assert "[click" not in short


def test_report_markdown_keeps_a_finding_whose_severity_it_does_not_know() -> None:
    report = build_report(
        now=NOW,
        findings=[
            make_finding(
                "ha_failed_logins",
                severity=cast("Any", "critical"),
                evidence={},
                summary="Someone failed to log in.",
                suggested_actions=[],
            )
        ],
        checks_run=["ha_failed_logins"],
        inactive_rules={},
        capabilities=[],
        notes=["Gateway seen\n# Not a heading"],
    )
    text = render_report_markdown(report, None, escape_markdown, 12000)
    assert "**Other**\n- **Failed login attempts.** Someone failed to log in." in text
    # Text from the home cannot open a new Markdown block.
    assert "- Gateway seen # Not a heading" in text


# ---------------------------------------------------------------------------
# Network counters and their baselines (plan step 10)
# ---------------------------------------------------------------------------


class _FakeUpdater:
    def __init__(self, baselines: dict[str, dict[str, float]]) -> None:
        self.baselines = baselines
        self.offered: list[dict[str, float]] = []
        self.fetched_metrics: list[list[str]] = []

    def offer_counters(self, counters: Any) -> int:
        self.offered.append(dict(counters))
        return len(counters)

    async def async_fetch_counter_baselines(
        self, metrics: Any
    ) -> dict[str, dict[str, float]]:
        self.fetched_metrics.append(list(metrics))
        return self.baselines

    def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _usage_client(key: str, name: str, up: int) -> dict[str, Any]:
    return {
        "key": key,
        "connected": True,
        "name": name,
        "data_up_day_bytes": up,
        "data_down_day_bytes": 100_000_000,
    }


def _usage_snapshot(*clients: dict[str, Any]) -> FullStateSnapshot:
    snapshot = _snapshot()
    section = cast("Any", snapshot)["network"]
    section["clients"] = list(clients) or [
        _usage_client("3fa2c1b0", "Living room TV", 3_200_000_000)
    ]
    section["counters"] = {
        f"network.client.{c['key']}.data_up_day_bytes": float(c["data_up_day_bytes"])
        for c in section["clients"]
    }
    section["capabilities"] = sorted(
        {*section["capabilities"], CAP_CLIENTS, CAP_CLIENT_DATA_DAY}
    )
    section["sources"][CAP_CLIENTS] = "eero_runtime"
    section["sources"][CAP_CLIENT_DATA_DAY] = "eero_runtime"
    return snapshot


def _usual_up(key: str, value: float) -> dict[str, dict[str, float]]:
    # Keyed by the snapshot's hour: the engine and the rule both read the
    # buckets for generated_at, never the wall clock.
    return {
        f"network.client.{key}.data_up_day_bytes": {f"hourly_avg_{NOW.hour}": value}
    }


@pytest.mark.asyncio
async def test_audit_injects_counter_baselines_without_offering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updater = _FakeUpdater(_usual_up("3fa2c1b0", 8e8))
    engine, _, _, _ = _engine(monkeypatch, _usage_snapshot(), baseline_updater=updater)
    report = await engine.async_audit_network()
    assert report["status"] == "ok"
    assert CAP_COUNTER_BASELINES in report["capabilities"]
    assert "network_client_usage_anomaly" in report["checks_run"]
    usage = [
        f for f in report["findings"] if f["type"] == "network_client_usage_anomaly"
    ]
    assert len(usage) == 1
    assert "Living room TV" in usage[0]["summary"]
    # The on-demand audit never hands a sample over; the fetch is bounded to
    # the two hour buckets the rule reads.
    assert updater.offered == []
    assert updater.fetched_metrics == [
        [f"hourly_avg_{NOW.hour}", f"hourly_avg_{(NOW.hour + 1) % 24}"]
    ]


@pytest.mark.asyncio
async def test_cycle_offers_counters_and_lists_the_rule_as_waiting_without_baselines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updater = _FakeUpdater({})
    engine, _, _, _ = _engine(monkeypatch, _usage_snapshot(), baseline_updater=updater)
    await engine._run_once()
    assert updater.offered == [{"network.client.3fa2c1b0.data_up_day_bytes": 3.2e9}]
    inactive = engine.run_stats["inactive_rules"]
    assert inactive["network_client_usage_anomaly"] == [CAP_COUNTER_BASELINES]
    assert "baseline collection" in capability_reason(CAP_COUNTER_BASELINES)
    assert "Data Usage (Day)" in capability_reason(CAP_CLIENT_DATA_DAY)


@pytest.mark.asyncio
async def test_counter_baselines_capability_needs_a_covered_device_and_notes_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A baseline for some other counter is not coverage for the usage check.
    other = {"network.client_count": {f"hourly_avg_{NOW.hour}": 38.0}}
    snapshot = _usage_snapshot(
        _usage_client("a", "TV", 3_200_000_000), _usage_client("b", "Phone", 1)
    )
    engine, _, _, _ = _engine(
        monkeypatch, snapshot, baseline_updater=_FakeUpdater(other)
    )
    report = await engine.async_audit_network()
    assert CAP_COUNTER_BASELINES not in report["capabilities"]
    assert "network_client_usage_anomaly" in report["checks_not_run"]
    assert any("cover 0 of 2 devices" in n for n in report["notes"])
    # One covered device grants the capability; the other is still noted.
    engine, _, _, _ = _engine(
        monkeypatch, snapshot, baseline_updater=_FakeUpdater(_usual_up("a", 8e8))
    )
    report = await engine.async_audit_network()
    assert CAP_COUNTER_BASELINES in report["capabilities"]
    assert any("cover 1 of 2 devices" in n for n in report["notes"])


@pytest.mark.asyncio
async def test_a_slow_or_failing_baseline_read_leaves_the_rule_not_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Hanging(_FakeUpdater):
        async def async_fetch_counter_baselines(self, metrics: Any) -> Any:
            msg = "pool closed"
            raise RuntimeError(msg)

    engine, _, _, _ = _engine(
        monkeypatch, _usage_snapshot(), baseline_updater=_Hanging({})
    )
    report = await engine.async_audit_network()
    assert report["status"] == "ok"
    assert CAP_COUNTER_BASELINES not in report["capabilities"]


@pytest.mark.asyncio
async def test_usage_findings_cool_down_per_device_not_per_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A benign alert on one device must not hide another device's for a day."""
    baselines = {**_usual_up("a", 8e8), **_usual_up("b", 8e8), **_usual_up("c", 8e8)}
    two = _usage_snapshot(
        _usage_client("a", "TV", 3_200_000_000),
        _usage_client("b", "NAS", 4_000_000_000),
    )
    engine, notifier, _, _ = _engine(
        monkeypatch, two, baseline_updater=_FakeUpdater(baselines)
    )
    await engine._run_once()
    delivered = [f.evidence["client_key"] for f in notifier.calls]
    assert delivered == ["a", "b"]
    # Same devices, next cycle: each is on its own daily cooldown.
    await engine._run_once()
    assert len(notifier.calls) == 2
    # A third device reported later is not held by the first two.
    three = _usage_snapshot(
        _usage_client("a", "TV", 3_300_000_000),
        _usage_client("b", "NAS", 4_100_000_000),
        _usage_client("c", "Cam", 5_000_000_000),
    )
    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine."
        "async_build_full_state_snapshot",
        _build_returning(three),
    )
    await engine._run_once()
    assert [f.evidence["client_key"] for f in notifier.calls] == ["a", "b", "c"]


def _build_returning(snapshot: FullStateSnapshot) -> Any:
    async def _fake_build(_hass: Any, *, network: Any = None) -> FullStateSnapshot:
        del network
        return snapshot

    return _fake_build
