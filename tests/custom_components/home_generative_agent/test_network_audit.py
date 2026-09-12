# ruff: noqa: S101
"""On-demand network audit: engine method, report shape, agent tool, service."""

from __future__ import annotations

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
    CONF_SENTINEL_NETWORK_ENABLED,
    CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS,
    NETWORK_AUDIT_TOOL_PROMPT,
)
from custom_components.home_generative_agent.sentinel.engine import SentinelEngine
from custom_components.home_generative_agent.sentinel.network_audit import (
    PRIVACY_NOTES,
    build_report,
    capability_reason,
    empty_report,
    summarize,
)
from custom_components.home_generative_agent.sentinel.rules.network_common import (
    NETWORK_RULE_TYPES,
    make_finding,
)
from custom_components.home_generative_agent.sentinel.suppression import (
    SuppressionManager,
    SuppressionState,
)
from custom_components.home_generative_agent.snapshot.network import (
    CAP_CLIENTS,
    NetworkBuildContext,
    ha_cap,
    posture_cap,
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
    assert "router integration adapter" in capability_reason(CAP_CLIENTS)
    assert "router or DNS" in capability_reason(posture_cap("upnp_enabled"))
    assert "not provided" in capability_reason("network.radio.devices")


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
            "broken_rule": [],
        },
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
async def test_audit_reports_unavailable_when_snapshot_build_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _, _, _ = _engine(monkeypatch, _snapshot())

    async def _boom(_hass: Any, *, network: Any = None) -> Any:
        _ = network
        raise ValueError

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine."
        "async_build_full_state_snapshot",
        _boom,
    )
    report = await engine.async_audit_network()
    assert report["status"] == "unavailable"
    assert report["findings"] == []


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
    assert "turned off" in text
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
    assert set(payload) == {
        "generated_at",
        "summary",
        "findings",
        "checks_run",
        "checks_not_run",
        "notes",
        "privacy_notes",
    }


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
    assert src.count('"audit_home_security": audit_home_security,') == 2
    assert "audit_prompt = NETWORK_AUDIT_TOOL_PROMPT if has_tools else" in src


def test_prompt_instruction_names_the_tool_and_forbids_false_passes() -> None:
    assert "audit_home_security" in NETWORK_AUDIT_TOOL_PROMPT
    assert "Never claim a check passed" in NETWORK_AUDIT_TOOL_PROMPT


def test_run_network_audit_service_is_registered_with_a_response() -> None:
    src = inspect.getsource(cast("Any", hga_component).async_setup_entry)
    assert "SERVICE_RUN_NETWORK_AUDIT," in src
    assert cast("Any", hga_component).SERVICE_RUN_NETWORK_AUDIT == "run_network_audit"
    services = yaml.safe_load((_COMPONENT_DIR / "services.yaml").read_text())
    assert "run_network_audit" in services
    assert services["run_network_audit"]["fields"] == {}
