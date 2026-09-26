# ruff: noqa: S101
"""The network_client_usage_anomaly rule."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from custom_components.home_generative_agent.sentinel.baseline import (
    METRIC_HOURLY_PREFIX,
)
from custom_components.home_generative_agent.sentinel.rules.network_client_usage_anomaly import (
    USAGE_ENTITY_COOLDOWN_MINUTES,
    NetworkClientUsageAnomalyRule,
    hour_metrics,
    human_bytes,
    usage_coverage,
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
NEXT_HOUR = f"{METRIC_HOURLY_PREFIX}15"
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


def _usual(
    key: str, figure: str, value: float, metric: str = HOUR
) -> dict[str, dict[str, float]]:
    return {client_counter_id(key, figure): {metric: value}}


def _only(findings: list[AnomalyFinding]) -> AnomalyFinding:
    assert len(findings) == 1
    return findings[0]


def test_rule_declares_its_capabilities_and_a_per_device_daily_cooldown() -> None:
    assert NetworkClientUsageAnomalyRule.requires == {
        CAP_CLIENTS,
        CAP_CLIENT_DATA_DAY,
        CAP_COUNTER_BASELINES,
    }
    # Per device, not per type: one device's alert never hides another's.
    assert NetworkClientUsageAnomalyRule.cooldown_minutes == 0
    assert NetworkClientUsageAnomalyRule.entity_cooldown_minutes == (
        USAGE_ENTITY_COOLDOWN_MINUTES
    )


def test_hour_metrics_are_this_hour_and_the_next_in_utc() -> None:
    assert hour_metrics(NOW) == (HOUR, NEXT_HOUR)
    late = datetime(2026, 9, 25, 23, 50, tzinfo=UTC)
    assert hour_metrics(late) == (
        f"{METRIC_HOURLY_PREFIX}23",
        f"{METRIC_HOURLY_PREFIX}0",
    )


def test_upload_far_above_usual_is_medium_with_figures() -> None:
    rule = NetworkClientUsageAnomalyRule()
    snapshot = _snapshot(
        [_client("a", up=3_200 * MB)], _usual("a", "data_up_day_bytes", 800 * MB)
    )
    finding = _only(rule.evaluate(snapshot))
    assert finding.type == "network_client_usage_anomaly"
    assert finding.severity == "medium"
    assert finding.is_sensitive
    assert finding.evidence["client_key"] == "a"
    assert finding.evidence["direction"] == "upload"
    assert finding.evidence["today_bytes"] == 3_200 * MB
    assert finding.evidence["usual_bytes"] == 800 * MB
    assert finding.evidence["summary"] == (
        "TV a (wireless) has uploaded 3.2 GB so far today, about 4.0x its usual "
        "800 MB by this hour."
    )
    # The per-device pseudo entity carries the daily cooldown; the tracker
    # lets exclusions and snoozes address the device.
    assert finding.triggering_entities == [
        "network.client.a.upload",
        "device_tracker.a",
    ]
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
    assert (
        "when it usually has moved nothing by this hour"
        in (finding.evidence["summary"])
    )


def test_usual_is_the_larger_of_this_hour_and_the_next() -> None:
    # A nightly backup that always lands at :50 is never in this hour's
    # samples (taken at :00, :15, :30, :45) but is in the next hour's.
    rule = NetworkClientUsageAnomalyRule()
    baselines = {
        client_counter_id("a", "data_up_day_bytes"): {HOUR: 0.0, NEXT_HOUR: 1_100 * MB}
    }
    assert rule.evaluate(_snapshot([_client("a", up=1_000 * MB)], baselines)) == []
    finding = _only(rule.evaluate(_snapshot([_client("a", up=5_000 * MB)], baselines)))
    assert finding.evidence["usual_bytes"] == 1_100 * MB
    # The next hour alone is enough too (the current one not yet sampled).
    only_next = _usual("a", "data_up_day_bytes", 800 * MB, NEXT_HOUR)
    assert len(rule.evaluate(_snapshot([_client("a", up=3_200 * MB)], only_next))) == 1


def test_no_baseline_for_these_hours_no_figure_or_offline_means_no_finding() -> None:
    rule = NetworkClientUsageAnomalyRule()
    other_hour = _usual("a", "data_up_day_bytes", 1.0, "hourly_avg_3")
    assert rule.evaluate(_snapshot([_client("a", up=3_200 * MB)], other_hour)) == []
    assert rule.evaluate(_snapshot([_client("a", up=3_200 * MB)], {})) == []
    usual = _usual("a", "data_up_day_bytes", 800 * MB)
    assert rule.evaluate(_snapshot([_client("a")], usual)) == []
    offline = _client("a", up=3_200 * MB, connected=False)
    assert rule.evaluate(_snapshot([offline], usual)) == []
    garbled = _usual("a", "data_up_day_bytes", float("nan"))
    assert rule.evaluate(_snapshot([_client("a", up=3_200 * MB)], garbled)) == []


def test_identity_is_the_device_and_direction_not_the_figures() -> None:
    rule = NetworkClientUsageAnomalyRule()
    usual = _usual("a", "data_up_day_bytes", 800 * MB)
    first = _only(rule.evaluate(_snapshot([_client("a", up=3_200 * MB)], usual)))
    later = _only(rule.evaluate(_snapshot([_client("a", up=4_800 * MB)], usual)))
    assert first.anomaly_id == later.anomaly_id
    # Upload and download are two findings, each with its own identity.
    both = {**usual, **_usual("a", "data_down_day_bytes", 100 * MB)}
    two = rule.evaluate(_snapshot([_client("a", up=3_200 * MB, down=2_000 * MB)], both))
    assert [f.evidence["direction"] for f in two] == ["upload", "download"]
    assert two[0].anomaly_id == first.anomaly_id
    assert two[1].anomaly_id != first.anomaly_id
    assert two[1].triggering_entities[0] == "network.client.a.download"


def test_one_finding_per_device_and_exclusions() -> None:
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
    findings = rule.evaluate(_snapshot(clients, baselines))
    assert [f.evidence["client_key"] for f in findings] == ["a", "c"]
    assert [f.severity for f in findings] == ["medium", "low"]
    assert ("device_tracker.b", "network_client_usage_anomaly") in calls


def test_usage_coverage_counts_devices_with_a_readable_baseline() -> None:
    section = cast(
        "Any",
        _snapshot(
            [
                _client("a", up=1),
                _client("b", up=1),
                _client("c", up=1, connected=False),
                _client("d"),  # no traffic figure
            ],
            None,
        ),
    )["network"]
    baselines = {
        **_usual("a", "data_up_day_bytes", 5.0),
        **_usual("b", "data_down_day_bytes", 5.0, "hourly_avg_3"),
    }
    assert usage_coverage(section, baselines, NOW) == (1, 2)
    assert usage_coverage(section, {}, NOW) == (0, 2)


def test_human_bytes() -> None:
    assert human_bytes(999) == "999 B"
    assert human_bytes(12_300) == "12 kB"
    assert human_bytes(800 * MB) == "800 MB"
    assert human_bytes(3_200 * MB) == "3.2 GB"
    assert human_bytes(2.5e12) == "2.5 TB"
