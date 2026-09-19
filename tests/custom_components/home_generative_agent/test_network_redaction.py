# ruff: noqa: S101
"""Tests for the network-identifier gate in front of every model call."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from custom_components.home_generative_agent.agent.tools import audit_home_security
from custom_components.home_generative_agent.explain.llm_explain import LLMExplainer
from custom_components.home_generative_agent.sentinel.discovery_engine import (
    SentinelDiscoveryEngine,
)
from custom_components.home_generative_agent.sentinel.network_audit import (
    build_report,
)
from custom_components.home_generative_agent.sentinel.redaction import (
    LAN_IP_REDACTED,
    MAC_REDACTED,
    PUBLIC_IP_REDACTED,
    client_display_name,
    redact_network_identifiers,
)
from custom_components.home_generative_agent.sentinel.rules.network_common import (
    make_finding,
)
from custom_components.home_generative_agent.sentinel.triage import _build_prompt

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.sentinel.models import (
        AnomalyFinding,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

NOW = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)

# Every address shape a rule or adapter could fold into text. A hostname
# (``Lindos-iPhone`` below) cannot be recognised by pattern, so it only ever
# appears in the structural positions the walk removes by key.
RAW = (
    "router 192.168.1.1, public 8.8.8.8, v6 2606:4700::1111, link fe80::1, "
    "mac aa:bb:cc:dd:ee:ff, ieee 00:0d:6f:00:0a:bb:cc:dd"
)
IDENTIFIERS = (
    "192.168.1.1",
    "8.8.8.8",
    "2606:4700::1111",
    "fe80::1",
    "aa:bb:cc:dd:ee:ff",
    "00:0d:6f:00:0a:bb:cc:dd",
)


def _assert_clean(text: str) -> None:
    for identifier in IDENTIFIERS:
        assert identifier not in text, identifier


# ---------------------------------------------------------------------------
# Textual layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("lan 192.168.1.1", f"lan {LAN_IP_REDACTED}"),
        ("lan 10.0.0.5 and 172.16.4.2", f"lan {LAN_IP_REDACTED} and {LAN_IP_REDACTED}"),
        ("loop 127.0.0.1", f"loop {LAN_IP_REDACTED}"),
        ("wan 8.8.8.8", f"wan {PUBLIC_IP_REDACTED}"),
        # Carrier-grade NAT is the address the home is reached by.
        ("cgnat 100.64.1.2", f"cgnat {PUBLIC_IP_REDACTED}"),
        # End of sentence keeps its period; the address still goes.
        ("changed to 8.8.4.4.", f"changed to {PUBLIC_IP_REDACTED}."),
        ("url http://192.168.1.5:8123/api", f"url http://{LAN_IP_REDACTED}:8123/api"),
        ("v6 2606:4700::1111", f"v6 {PUBLIC_IP_REDACTED}"),
        ("link fe80::1%eth0", f"link {LAN_IP_REDACTED}%eth0"),
        ("loop ::1", f"loop {LAN_IP_REDACTED}"),
        ("bracket [2606:4700::1111]:443", f"bracket [{PUBLIC_IP_REDACTED}]:443"),
        (
            "full 2001:0db8:0000:0000:0000:ff00:0042:8329",
            f"full {LAN_IP_REDACTED}",
        ),
        ("mac aa:bb:cc:dd:ee:ff", f"mac {MAC_REDACTED}"),
        ("mac AA-BB-CC-DD-EE-FF", f"mac {MAC_REDACTED}"),
        ("cisco 0011.2233.4455", f"cisco {MAC_REDACTED}"),
        # An EUI-64 (Zigbee IEEE) address is one token, not two halves.
        ("ieee 00:0d:6f:00:0a:bb:cc:dd", f"ieee {MAC_REDACTED}"),
        ("bt F4:5C:89:AB:CD:EF", f"bt {MAC_REDACTED}"),
    ],
)
def test_addresses_in_text_are_replaced_by_class(text: str, expected: str) -> None:
    assert redact_network_identifiers(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "out of range 999.1.1.1",
        "version 7.57.2",
        "five groups 1.2.3.4.5",
        "entity sensor.eero_192_168_1_1",
        "clock 12:30:45",
        "hash 3fa2c1b0aabbccdd",
        "seven octets aa:bb:cc:dd:ee:ff:00",
        "bare hex aabbccddeeff",
        "prefixed v1.2.3.4",
    ],
)
def test_lookalikes_are_left_alone(text: str) -> None:
    assert redact_network_identifiers(text) == text


def test_four_part_version_is_redacted_by_design() -> None:
    """A dotted quad that parses as an address is treated as one (documented)."""
    assert redact_network_identifiers("firmware 1.2.3.4") == (
        f"firmware {PUBLIC_IP_REDACTED}"
    )


# ---------------------------------------------------------------------------
# Structural layer
# ---------------------------------------------------------------------------


def test_client_dict_keeps_key_and_display_name_only() -> None:
    client = {
        "key": "3fa2c1b0",
        "connected": True,
        "mac": "aa:bb:cc:dd:ee:ff",
        "ip": "192.168.1.5",
        "hostname": "Lindos-iPhone",
        "manufacturer": "Apple",
        "signal": -50,
    }
    redacted = redact_network_identifiers(client)
    assert redacted == {
        "key": "3fa2c1b0",
        "connected": True,
        "hostname": "Apple 3fa2c1b0",
        "manufacturer": "Apple",
        "signal": -50,
    }


def test_address_keys_are_dropped_whatever_their_value_or_case() -> None:
    value = {
        "MAC": None,
        "Ip_Address": "",
        "ipv6": ["fe80::1"],
        "ieee": "00:0d:6f:00:0a:bb:cc:dd",
        "address": "F4:5C:89:AB:CD:EF",
        "bluetooth_address": "x",
        "kept": 1,
    }
    assert redact_network_identifiers(value) == {"kept": 1}


def test_hostname_without_client_key_is_dropped() -> None:
    """A hostname cannot be pattern-matched, so by position it can only go."""
    value = {"host_name": "Lindos-iPhone", "hostname": "nas", "state": "home"}
    assert redact_network_identifiers(value) == {"state": "home"}


def test_walk_covers_keys_containers_and_nesting() -> None:
    value = {
        "192.168.1.1": ("10.0.0.1", 3),
        "gateways": ["eero (192.168.1.1)", {"name": "aa:bb:cc:dd:ee:ff"}],
        "tags": frozenset({"8.8.8.8"}),
        "nested": {"clients": [{"key": "k1", "hostname": "h", "ip": "1.1.1.1"}]},
    }
    redacted = redact_network_identifiers(value)
    assert redacted == {
        LAN_IP_REDACTED: (LAN_IP_REDACTED, 3),
        "gateways": [f"eero ({LAN_IP_REDACTED})", {"name": MAC_REDACTED}],
        "tags": frozenset({PUBLIC_IP_REDACTED}),
        "nested": {"clients": [{"key": "k1", "hostname": "device k1"}]},
    }
    assert isinstance(redacted["192.168.1.1" and LAN_IP_REDACTED], tuple)
    assert isinstance(redacted["tags"], frozenset)


def test_redaction_never_mutates_its_input() -> None:
    client = {"key": "k", "mac": "aa:bb:cc:dd:ee:ff", "nested": ["192.168.1.1"]}
    before = json.dumps(client, sort_keys=True)
    redact_network_identifiers(client)
    assert json.dumps(client, sort_keys=True) == before


@pytest.mark.parametrize("value", [None, 3, 1.5, True])
def test_non_string_scalars_pass_through(value: Any) -> None:
    assert redact_network_identifiers(value) == value


def test_client_display_name_prefers_manufacturer_and_sanitizes() -> None:
    assert client_display_name({"key": "3fa2c1b0", "manufacturer": "Apple"}) == (
        "Apple 3fa2c1b0"
    )
    assert client_display_name({"key": "3fa2c1b0"}) == "device 3fa2c1b0"
    assert client_display_name({"manufacturer": "Apple"}) == "unknown device"
    assert client_display_name("not a client") == "unknown device"
    # Manufacturer strings come from the OUI table but pass sanitize_label
    # like every other label: control characters are dropped.
    zero_width = "Ap\u200bple"
    assert client_display_name({"key": "k", "manufacturer": zero_width}) == "Apple k"


# ---------------------------------------------------------------------------
# Model boundaries
# ---------------------------------------------------------------------------


class _CapturingModel:
    def __init__(self, content: str) -> None:
        self._content = content
        self.messages: list[Any] | None = None

    async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
        self.messages = messages
        return SimpleNamespace(content=self._content)


def _network_finding() -> AnomalyFinding:
    return make_finding(
        "network_unknown_device",
        severity="medium",
        evidence={
            "device_keys": ["3fa2c1b0"],
            "clients": [
                {
                    "key": "3fa2c1b0",
                    "mac": "aa:bb:cc:dd:ee:ff",
                    "ip": "192.168.1.1",
                    "hostname": "Lindos-iPhone",
                    "manufacturer": "Apple",
                }
            ],
        },
        display={"gateways": ["eero (8.8.8.8)"]},
        summary=RAW,
        suggested_actions=["Trust or block the device"],
        triggering_entities=["device_tracker.lindos_iphone"],
    )


@pytest.mark.asyncio
async def test_explainer_prompt_carries_no_identifier() -> None:
    model = _CapturingModel("A new device joined.")
    explainer = LLMExplainer(model, deployment="edge")
    finding = _network_finding()
    before = json.dumps(finding.evidence, sort_keys=True, default=str)

    await explainer.async_explain(finding)

    assert model.messages is not None
    prompt = cast("str", model.messages[1].content)
    _assert_clean(prompt)
    assert "Lindos-iPhone" not in prompt
    assert "Apple 3fa2c1b0" in prompt
    # The finding itself is untouched: its evidence feeds the anomaly id and
    # the notifier's own copy.
    assert json.dumps(finding.evidence, sort_keys=True, default=str) == before


def test_triage_prompt_carries_no_identifier() -> None:
    snapshot = cast(
        "FullStateSnapshot",
        {"derived": {"is_night": False, "anyone_home": True}},
    )
    prompt = _build_prompt(_network_finding(), snapshot)
    _assert_clean(prompt)
    assert "Lindos-iPhone" not in prompt
    assert "type: network_unknown_device" in prompt


@pytest.mark.asyncio
async def test_discovery_prompt_carries_no_identifier(hass: HomeAssistant) -> None:
    """The reducer output passes the gate even once it grows a network section."""
    model = _CapturingModel(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-01-01T00:00:00Z",
                "model": "test",
                "candidates": [],
            }
        )
    )

    class _Store:
        async def async_get_latest(self, _limit: int) -> list[dict[str, Any]]:
            return []

        async def async_append(self, _payload: Any) -> None:
            pass

    engine = SentinelDiscoveryEngine(
        hass=hass, options={}, model=model, store=cast("Any", _Store())
    )

    async def _fake_run(model: Any, messages: Any, **_kw: Any) -> Any:
        return await model.ainvoke(messages)

    reduced = {
        "entities": [{"entity_id": "sensor.rack", "area": "Rack 192.168.1.1"}],
        "camera_activity": [],
        "derived": {"is_night": False},
        "network": {
            "clients": [
                {"key": "k", "mac": "aa:bb:cc:dd:ee:ff", "hostname": "Lindos-iPhone"}
            ],
            "posture": {"upnp_gateway_names": [RAW]},
        },
    }
    engine_module = "custom_components.home_generative_agent.sentinel.discovery_engine"
    with (
        patch.object(
            engine,
            "_existing_semantic_context",
            new_callable=AsyncMock,
            return_value=(set(), set(), set()),
        ),
        patch(
            f"{engine_module}.async_build_full_state_snapshot",
            new_callable=AsyncMock,
            return_value={
                "entities": [],
                "camera_activity": [],
                "derived": {"is_night": False, "now": "2026-01-01T00:00:00Z"},
                "generated_at": "2026-01-01T00:00:00Z",
            },
        ),
        patch(f"{engine_module}.reduce_snapshot_for_discovery", return_value=reduced),
        patch(f"{engine_module}.run_sentinel_model_call", side_effect=_fake_run),
    ):
        await engine._run_once()

    assert model.messages is not None
    prompt = "".join(str(m.content) for m in model.messages)
    _assert_clean(prompt)
    assert "Lindos-iPhone" not in prompt
    assert "device k" in prompt


@pytest.mark.asyncio
async def test_audit_tool_payload_carries_no_identifier() -> None:
    report = build_report(
        now=NOW,
        findings=[_network_finding()],
        checks_run=["network_unknown_device"],
        inactive_rules={},
        capabilities=[],
        notes=[f"gateway eero at 192.168.1.1 ({RAW})"],
    )

    async def _audit() -> Any:
        return report

    config = {
        "configurable": {
            "hga_runtime_data": SimpleNamespace(
                sentinel=SimpleNamespace(async_audit_network=_audit)
            )
        }
    }
    text = await audit_home_security.coroutine(config=config)  # type: ignore[misc]

    _assert_clean(text)
    payload = yaml.safe_load(text)
    assert LAN_IP_REDACTED in payload["findings"][0]["summary"]
    assert MAC_REDACTED in payload["findings"][0]["summary"]
    assert LAN_IP_REDACTED in payload["notes"][1]
    # The service-side report the tool was built from is not touched.
    assert "192.168.1.1" in report["notes"][0]
