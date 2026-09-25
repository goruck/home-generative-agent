# ruff: noqa: S101
"""The network_client_usage_anomaly rule."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from custom_components.home_generative_agent.sentinel.baseline import (
    METRIC_HOURLY_PREFIX,
)
from custom_components.home_generative_agent.sentinel.rules.network_client_usage_anomaly import (
    NetworkClientUsageAnomalyRule,
    human_bytes,
)
from custom_components.home_generative_agent.sentinel.rules.network_common import (
    POSTURE_COOLDOWN_MINUTES,
)
from custom_components.home_generative_agent.snapshot.network import (
    CAP_CLIENT_DATA_DAY,
    CAP_CLIENTS,
    CAP_COUNTER_BASELINES,
    client_counter_id,
)
from custom_components.home_generative_agent.snapshot.schema import validate_snapshot

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import (
        AnomalyFinding,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

NOW = datetime(2026, 9, 25, 14, 30, tzinfo=UTC)
HOUR = f"{METRIC_HOURLY_PREFIX}14"
MB = 1_000_000


def _client(
    key: str, *, up: int | None = None, down: int | None = None, **extra: Any
) -> dict[str, Any]:
    client: dict[str, Any] = {
        "key": key,
        "connected": True,
        "name": f"TV {key}",
        "connection_type": "wireless",
        "tracker_entity_id": f"device_tracker.{key}",
    }
    if up is not None:
        client["data_up_day_bytes"] = up
    if down is not None:
        client["data_down_day_bytes"] = down
    client.update(extra)
    return client


def _snapshot(
    clients: list[dict[str, Any]],
    baselines: dict[str, dict[str, float]] | None,
) -> FullStateSnapshot:
    caps = [CAP_CLIENTS, CAP_CLIENT_DATA_DAY]
    network: dict[str, Any] = {
        "capabilities": caps,
        "sources": dict.fromkeys(caps, "eero_runtime"),
        "clients": clients,
        "posture": {},
        "ha_security": {},
        "counters": {},
    }
    if baselines is not None:
        network["counter_baselines"] = baselines
        network["capabilities"].append(CAP_COUNTER_BASELINES)
    return validate_snapshot(
        {
            "schema_version": 2,
            "generated_at": NOW.isoformat(),
            "entities": [],
            "camera_activity": [],
            "derived": {
                "now": NOW.isoformat(),
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": True,
                "people_home": ["person.sam"],
                "people_away": [],
                "last_motion_by_area": {},
            },
            "network": network,
        }
    )


def _usual(key: str, figure: str, value: float) -> dict[str, dict[str, float]]:
    return {client_counter_id(key, figure): {HOUR: value}}


def _only(findings: list[AnomalyFinding]) -> AnomalyFinding:
    assert len(findings) == 1
    return findings[0]


def test_rule_declares_its_capabilities_and_the_daily_floor() -> None:
    assert NetworkClientUsageAnomalyRule.requires == {
        CAP_CLIENTS,
        CAP_CLIENT_DATA_DAY,
        CAP_COUNTER_BASELINES,
    }
    assert NetworkClientUsageAnomalyRule.cooldown_minutes == POSTURE_COOLDOWN_MINUTES


def test_upload_far_above_usual_is_medium_with_figures() -> None:
    rule = NetworkClientUsageAnomalyRule()
    snapshot = _snapshot(
        [_client("a", up=3_200 * MB)], _usual("a", "data_up_day_bytes", 800 * MB)
    )
    finding = _only(rule.evaluate(snapshot))
    assert finding.type == "network_client_usage_anomaly"
    assert finding.severity == "medium"
    assert finding.is_sensitive
    assert finding.evidence["client_keys"] == ["a"]
    assert finding.evidence["directions"] == ["a:upload"]
    assert finding.evidence["figures"] == [
        {
            "key": "a",
            "direction": "upload",
            "today_bytes": 3_200 * MB,
            "usual_bytes": 800 * MB,
        }
    ]
    assert finding.evidence["summary"] == (
        "A device on the network has moved far more data than usual for this "
        "hour: TV a (wireless) has uploaded 3.2 GB so far today, about 4.0x its "
        "usual 800 MB."
    )
    assert finding.triggering_entities == ["device_tracker.a"]
    assert all("." not in a for a in finding.suggested_actions)


def test_download_alone_is_low() -> None:
    rule = NetworkClientUsageAnomalyRule()
    snapshot = _snapshot(
        [_client("a", down=9_000 * MB)], _usual("a", "data_down_day_bytes", 2_000 * MB)
    )
    finding = _only(rule.evaluate(snapshot))
    assert finding.severity == "low"
    assert "downloaded 9.0 GB" in finding.evidence["summary"]


def test_a_multiple_without_the_byte_floor_is_not_news() -> None:
    # 3 MB -> 30 MB is 10x and nothing.
    rule = NetworkClientUsageAnomalyRule()
    snapshot = _snapshot(
        [_client("a", up=30 * MB)], _usual("a", "data_up_day_bytes", 3 * MB)
    )
    assert rule.evaluate(snapshot) == []


def test_the_floor_without_the_multiple_is_not_news() -> None:
    # +300 MB on a device that usually moves 2 GB is 15% more.
    rule = NetworkClientUsageAnomalyRule()
    snapshot = _snapshot(
        [_client("a", up=2_300 * MB)], _usual("a", "data_up_day_bytes", 2_000 * MB)
    )
    assert rule.evaluate(snapshot) == []


def test_a_device_that_usually_moves_nothing_by_now() -> None:
    rule = NetworkClientUsageAnomalyRule()
    snapshot = _snapshot(
        [_client("a", up=600 * MB)], _usual("a", "data_up_day_bytes", 0.0)
    )
    finding = _only(rule.evaluate(snapshot))
    assert "when it usually has moved nothing by now" in finding.evidence["summary"]


def test_no_baseline_for_this_hour_no_figure_or_offline_means_no_finding() -> None:
    rule = NetworkClientUsageAnomalyRule()
    other_hour = {client_counter_id("a", "data_up_day_bytes"): {"hourly_avg_3": 1.0}}
    assert rule.evaluate(_snapshot([_client("a", up=3_200 * MB)], other_hour)) == []
    assert rule.evaluate(_snapshot([_client("a", up=3_200 * MB)], {})) == []
    usual = _usual("a", "data_up_day_bytes", 800 * MB)
    assert rule.evaluate(_snapshot([_client("a")], usual)) == []
    offline = _client("a", up=3_200 * MB, connected=False)
    assert rule.evaluate(_snapshot([offline], usual)) == []
    garbled = {client_counter_id("a", "data_up_day_bytes"): {HOUR: float("nan")}}
    assert rule.evaluate(_snapshot([_client("a", up=3_200 * MB)], garbled)) == []


def test_identity_is_the_client_and_direction_set_not_the_figures() -> None:
    rule = NetworkClientUsageAnomalyRule()
    usual = _usual("a", "data_up_day_bytes", 800 * MB)
    first = _only(rule.evaluate(_snapshot([_client("a", up=3_200 * MB)], usual)))
    later = _only(rule.evaluate(_snapshot([_client("a", up=4_800 * MB)], usual)))
    assert first.anomaly_id == later.anomaly_id
    both = {**usual, **_usual("a", "data_down_day_bytes", 100 * MB)}
    two = _only(
        rule.evaluate(_snapshot([_client("a", up=3_200 * MB, down=2_000 * MB)], both))
    )
    assert two.anomaly_id != first.anomaly_id
    assert two.evidence["directions"] == ["a:download", "a:upload"]
    assert two.evidence["client_keys"] == ["a"]


def test_several_devices_in_one_finding_and_exclusions() -> None:
    calls: list[tuple[str, str]] = []

    def excluded(entity_id: str, rule_id: str) -> bool:
        calls.append((entity_id, rule_id))
        return entity_id == "device_tracker.b"

    rule = NetworkClientUsageAnomalyRule(is_entity_excluded=excluded)
    baselines = {
        **_usual("a", "data_up_day_bytes", 800 * MB),
        **_usual("b", "data_up_day_bytes", 800 * MB),
        **_usual("c", "data_down_day_bytes", 800 * MB),
    }
    clients = [
        _client("a", up=3_200 * MB),
        _client("b", up=3_200 * MB),
        _client("c", down=3_200 * MB),
    ]
    finding = _only(rule.evaluate(_snapshot(clients, baselines)))
    assert finding.evidence["client_keys"] == ["a", "c"]
    assert ("device_tracker.b", "network_client_usage_anomaly") in calls
    assert finding.evidence["summary"].startswith(
        "Devices on the network have moved far more data than usual for this hour: "
    )


def test_human_bytes() -> None:
    assert human_bytes(999) == "999 B"
    assert human_bytes(12_300) == "12 kB"
    assert human_bytes(800 * MB) == "800 MB"
    assert human_bytes(3_200 * MB) == "3.2 GB"
    assert human_bytes(2.5e12) == "2.5 TB"
