# ruff: noqa: S101
"""Discovery over the network section: reducer, templates, evaluators, keys."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from custom_components.home_generative_agent.sentinel.discovery_semantic import (
    candidate_semantic_key,
    rule_semantic_key,
)
from custom_components.home_generative_agent.sentinel.dynamic_rules import (
    evaluate_dynamic_rules,
)
from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
from custom_components.home_generative_agent.sentinel.notifier import (
    _network_summary,
    is_security_copy,
)
from custom_components.home_generative_agent.sentinel.proposal_templates import (
    NETWORK_TEMPLATES,
    SUPPORTED_TEMPLATES,
    explain_normalize_candidate,
)
from custom_components.home_generative_agent.sentinel.redaction import (
    redact_network_identifiers,
)
from custom_components.home_generative_agent.snapshot.discovery_reducer import (
    reduce_snapshot_for_discovery,
)
from custom_components.home_generative_agent.snapshot.schema import validate_snapshot

from .test_discovery_engine_dedupe import _engine as _discovery_engine

if TYPE_CHECKING:
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

KEY = "3fa2c1b0"


def _client(key: str = KEY, **extra: Any) -> dict[str, Any]:
    client: dict[str, Any] = {
        "key": key,
        "connected": True,
        "name": "Living room TV",
        "ip": "192.168.1.23",
        "hostname": "lr-tv.lan",
        "tracker_entity_id": "device_tracker.living_room_tv",
        "manufacturer": "Sony",
    }
    client.update(extra)
    return client


def _snapshot(
    *,
    clients: list[dict[str, Any]] | None = None,
    posture: dict[str, Any] | None = None,
    anyone_home: bool = True,
    is_night: bool = False,
    network: bool = True,
) -> FullStateSnapshot:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "generated_at": "2026-09-26T12:00:00+00:00",
        "entities": [],
        "camera_activity": [],
        "derived": {
            "now": "2026-09-26T12:00:00+00:00",
            "timezone": "UTC",
            "is_night": is_night,
            "anyone_home": anyone_home,
            "people_home": ["person.sam"] if anyone_home else [],
            "people_away": [] if anyone_home else ["person.sam"],
            "last_motion_by_area": {},
        },
    }
    if network:
        caps = ["network.clients"] if clients is not None else []
        payload["network"] = {
            "capabilities": caps,
            "sources": dict.fromkeys(caps, "eero_runtime"),
            "clients": clients or [],
            "posture": posture or {},
            "ha_security": {},
            "counters": {},
        }
    return validate_snapshot(payload)


def _candidate(paths: list[str], summary: str, **extra: Any) -> dict[str, Any]:
    candidate = {
        "candidate_id": "net_candidate",
        "title": summary,
        "summary": summary,
        "pattern": "network",
        "confidence_hint": 0.8,
        "evidence_paths": paths,
    }
    candidate.update(extra)
    return candidate


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------


def test_reducer_adds_a_pseudonymized_network_section() -> None:
    snapshot = _snapshot(
        clients=[
            _client(),
            _client(
                "b1b1b1b1", name=None, connected=False, is_guest=True, trusted=False
            ),
        ],
        posture={
            "upnp_enabled": True,
            "wpa3_enabled": False,
            "upnp_enabled_entity_id": "switch.kro_upnp",
            "public_ip_key": "deadbeef",
            "guest_client_count": 0,
        },
    )
    reduced = redact_network_identifiers(reduce_snapshot_for_discovery(snapshot))
    network = reduced["network"]
    assert [c["key"] for c in network["clients"]] == [KEY, "b1b1b1b1"]
    tv, guest = network["clients"]
    assert tv == {"key": KEY, "name": "Living room TV", "connected": True}
    assert guest["name"] == "Sony b1b1b1b1"
    assert guest["is_guest"] is True
    assert guest["trusted"] is False
    # Only the boolean settings, never the twins or the public-IP key.
    assert network["posture"] == {"upnp_enabled": True, "wpa3_enabled": False}
    text = json.dumps(reduced)
    for secret in ("192.168.1.23", "lr-tv", "switch.kro_upnp", "deadbeef"):
        assert secret not in text


def test_reducer_leaves_the_section_out_without_network_facts() -> None:
    assert "network" not in reduce_snapshot_for_discovery(_snapshot(network=False))
    assert "network" not in reduce_snapshot_for_discovery(_snapshot(clients=[]))


def test_reducer_drops_clients_first_when_over_budget() -> None:
    from custom_components.home_generative_agent.snapshot import (  # noqa: PLC0415
        discovery_reducer,
    )

    snapshot = _snapshot(
        clients=[_client(f"{i:08x}") for i in range(60)],
        posture={"upnp_enabled": True},
    )
    reduced = reduce_snapshot_for_discovery(snapshot)
    assert len(reduced["network"]["clients"]) == discovery_reducer._MAX_NETWORK_CLIENTS
    original = discovery_reducer._TOKEN_BUDGET_CHARS
    discovery_reducer._TOKEN_BUDGET_CHARS = 200
    try:
        squeezed = reduce_snapshot_for_discovery(snapshot)
    finally:
        discovery_reducer._TOKEN_BUDGET_CHARS = original
    assert squeezed["network"] == {"posture": {"upnp_enabled": True}}


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


def test_templates_are_registered() -> None:
    assert NETWORK_TEMPLATES <= SUPPORTED_TEMPLATES


def test_client_present_while_away_normalizes_with_context() -> None:
    result = explain_normalize_candidate(
        _candidate(
            [f"network.clients[key={KEY}].connected", "not derived.anyone_home"],
            "Living room TV is connected while nobody is home",
        )
    )
    rule = result.normalized
    assert rule is not None
    assert rule.template_id == "network_client_present_when"
    assert rule.params == {
        "client_key": KEY,
        "require_away": True,
        "require_home": False,
        "require_night": False,
    }
    assert rule.severity == "medium"
    assert rule.is_sensitive


def test_client_absent_at_night_normalizes() -> None:
    rule = explain_normalize_candidate(
        _candidate(
            [f"network.clients[key='{KEY}'].connected", "derived.is_night"],
            "The security camera is missing from the network at night",
            candidate_id="camera_missing_at_night",
        )
    ).normalized
    assert rule is not None
    assert rule.template_id == "network_client_absent_when"
    assert rule.rule_id == "camera_missing_at_night"
    assert rule.params["require_night"] is True
    assert rule.params["require_away"] is False
    assert rule.severity == "low"


def test_posture_normalizes_with_the_value_the_prose_names() -> None:
    upnp = explain_normalize_candidate(
        _candidate(["network.posture.upnp_enabled"], "Alert when UPnP is turned on")
    ).normalized
    assert upnp is not None
    assert upnp.template_id == "network_posture_equals"
    assert upnp.params == {"posture_key": "upnp_enabled", "expected": True}
    wpa3 = explain_normalize_candidate(
        _candidate(["network.posture.wpa3_enabled"], "WPA3 protection was disabled")
    ).normalized
    assert wpa3 is not None
    assert wpa3.params == {"posture_key": "wpa3_enabled", "expected": False}
    # No value in the prose: the value that opens the network.
    guest = explain_normalize_candidate(
        _candidate(["network.posture.guest_network_enabled"], "Guest network change")
    ).normalized
    assert guest is not None
    assert guest.params["expected"] is True
    # A setting the rules do not watch is not a network candidate.
    assert (
        explain_normalize_candidate(
            _candidate(["network.posture.public_ip_key"], "Public IP changed")
        ).normalized
        is None
    )


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


def _rule(template_id: str, **params: Any) -> dict[str, Any]:
    return {
        "rule_id": f"test_{template_id}",
        "template_id": template_id,
        "params": params,
        "severity": "medium",
        "confidence": 0.8,
        "is_sensitive": True,
        "suggested_actions": ["check_device"],
    }


def test_client_present_when_fires_only_in_context() -> None:
    rule = _rule("network_client_present_when", client_key=KEY, require_away=True)
    home = _snapshot(clients=[_client()], anyone_home=True)
    assert evaluate_dynamic_rules(home, [rule]) == []
    away = _snapshot(clients=[_client()], anyone_home=False)
    (finding,) = evaluate_dynamic_rules(away, [rule])
    assert finding.evidence["summary"] == (
        "Living room TV (Sony, 192.168.1.23) is connected to the network while "
        "nobody is home."
    )
    assert finding.triggering_entities == ["device_tracker.living_room_tv"]
    assert finding.is_sensitive
    offline = _snapshot(clients=[_client(connected=False)], anyone_home=False)
    assert evaluate_dynamic_rules(offline, [rule]) == []


def test_client_absent_when_fires_for_offline_or_vanished_clients() -> None:
    rule = _rule(
        "network_client_absent_when", client_key=KEY, require_night=True, name="Cam"
    )
    night_offline = _snapshot(clients=[_client(connected=False)], is_night=True)
    (finding,) = evaluate_dynamic_rules(night_offline, [rule])
    assert finding.evidence["summary"] == (
        "Living room TV (Sony, 192.168.1.23) is not on the network at night."
    )
    # Gone from the router's list entirely: still absent, named from the rule.
    night_gone = _snapshot(clients=[_client("other000")], is_night=True)
    (finding,) = evaluate_dynamic_rules(night_gone, [rule])
    assert finding.evidence["summary"] == "Cam is not on the network at night."
    assert finding.triggering_entities == []
    # Daytime, or no router data this run: nothing.
    assert (
        evaluate_dynamic_rules(_snapshot(clients=[_client(connected=False)]), [rule])
        == []
    )
    no_router = _snapshot(network=False, is_night=True)
    assert evaluate_dynamic_rules(no_router, [rule]) == []


def test_posture_equals_fires_on_the_watched_value() -> None:
    rule = _rule("network_posture_equals", posture_key="upnp_enabled", expected=True)
    on = _snapshot(posture={"upnp_enabled": True, "upnp_enabled_entity_id": "switch.u"})
    (finding,) = evaluate_dynamic_rules(on, [rule])
    assert finding.evidence["summary"] == "The router's upnp setting is on."
    assert finding.triggering_entities == ["switch.u"]
    assert (
        evaluate_dynamic_rules(_snapshot(posture={"upnp_enabled": False}), [rule]) == []
    )
    assert evaluate_dynamic_rules(_snapshot(posture={}), [rule]) == []
    bad = _rule("network_posture_equals", posture_key="upnp_enabled", expected="yes")
    assert evaluate_dynamic_rules(on, [bad]) == []


# ---------------------------------------------------------------------------
# Semantic keys, engine gate, notifier copy
# ---------------------------------------------------------------------------


def test_candidate_and_activated_rule_share_a_semantic_key() -> None:
    candidate = _candidate(
        [f"network.clients[key={KEY}].connected", "not derived.anyone_home"],
        "Living room TV is connected while nobody is home",
    )
    rule = explain_normalize_candidate(candidate).normalized
    assert rule is not None
    assert candidate_semantic_key(candidate) == rule_semantic_key(rule.as_dict())
    assert candidate_semantic_key(candidate) == (
        f"v1|subject=network_client|predicate=present|night=any|home=0|scope=any|"
        f"entities={KEY}"
    )
    posture = _candidate(["network.posture.wpa3_enabled"], "WPA3 turned off")
    posture_rule = explain_normalize_candidate(posture).normalized
    assert posture_rule is not None
    assert candidate_semantic_key(posture) == rule_semantic_key(posture_rule.as_dict())
    assert "network_posture" in str(candidate_semantic_key(posture))


def test_network_paths_pass_the_derived_only_gate() -> None:
    engine = _discovery_engine()
    candidate = _candidate(
        [f"network.clients[key={KEY}].connected", "not derived.anyone_home"],
        "Living room TV is connected while nobody is home",
    )
    filtered, dropped = engine._filter_novel_candidates([candidate], set())
    assert len(filtered) == 1
    assert dropped == []


def test_network_template_findings_use_their_summary_and_are_security_copy() -> None:
    finding = AnomalyFinding(
        anomaly_id="n1",
        type="living_room_tv_while_away",
        severity="medium",
        confidence=0.8,
        triggering_entities=[],
        evidence={
            "template_id": "network_client_present_when",
            "summary": "Living room TV is connected to the network while nobody is home.",
        },
        suggested_actions=[],
        is_sensitive=True,
    )
    assert _network_summary(finding) == finding.evidence["summary"]
    assert is_security_copy(finding)
    assert cast("Any", finding).evidence["template_id"] in NETWORK_TEMPLATES
