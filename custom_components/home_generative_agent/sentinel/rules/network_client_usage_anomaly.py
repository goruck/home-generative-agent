"""Rule: a client has moved far more data today than it usually has by now."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    SENTINEL_NETWORK_USAGE_MIN_EXCESS_BYTES,
    SENTINEL_NETWORK_USAGE_THRESHOLD_PCT,
)
from custom_components.home_generative_agent.sentinel.baseline import (
    METRIC_HOURLY_PREFIX,
)
from custom_components.home_generative_agent.snapshot.network import (
    CAP_CLIENT_DATA_DAY,
    CAP_CLIENTS,
    CAP_COUNTER_BASELINES,
    client_counter_id,
)

from .network_common import (
    client_excluded,
    clients,
    make_finding,
    network_section,
)
from .network_unknown_device_joined import describe_client

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from datetime import datetime
    from typing import Any

    from custom_components.home_generative_agent.sentinel.models import (
        AnomalyFinding,
        Severity,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
        NetworkSnapshot,
    )

# (direction word, client figure) for the two counters the rule compares.
DIRECTIONS: tuple[tuple[str, str], ...] = (
    ("upload", "data_up_day_bytes"),
    ("download", "data_down_day_bytes"),
)
# Minutes a device stays reported before it is judged again (the figure only
# grows until midnight); per device and direction, not per rule type, so one
# device's alert never hides another's.
USAGE_ENTITY_COOLDOWN_MINUTES = 24 * 60


def hour_metrics(now: datetime) -> tuple[str, str]:
    """
    Return the two hourly metrics "usual by now" is read from.

    The updater keys the hourly profile by UTC hour. The current bucket
    holds samples taken earlier in past hours like this one, so a transfer
    that always lands late in the hour (a nightly backup at :50) would never
    be in it; the next bucket is, and the figure only grows within a day, so
    the larger of the two is "what this device had usually moved by the end
    of this hour".
    """
    hour = dt_util.as_utc(now).hour
    return (
        f"{METRIC_HOURLY_PREFIX}{hour}",
        f"{METRIC_HOURLY_PREFIX}{(hour + 1) % 24}",
    )


def usual_by_now(
    baselines: Mapping[str, Mapping[str, float]],
    key: str,
    figure: str,
    metrics: Iterable[str],
) -> float | None:
    """Return the baseline figure for *key*'s *figure*, or None without one."""
    stats = baselines.get(client_counter_id(key, figure)) or {}
    values = [
        float(v)
        for m in metrics
        if (v := stats.get(m)) is not None and math.isfinite(float(v)) and v >= 0
    ]
    return max(values) if values else None


def usage_coverage(
    section: NetworkSnapshot,
    baselines: Mapping[str, Mapping[str, float]],
    now: datetime,
) -> tuple[int, int]:
    """
    Return ``(covered, reporting)`` connected clients that report today's traffic.

    *reporting* is how many there are, *covered* how many of them have a
    baseline the rule can read now. The engine publishes the rule's
    capability only when at least one is covered, and notes the rest, so
    "no findings" is never said over devices whose baselines do not exist
    yet.
    """
    metrics = hour_metrics(now)
    reporting = covered = 0
    for client in section.get("clients") or []:
        if not client.get("connected") or client.get("data_up_day_bytes") is None:
            continue
        reporting += 1
        if any(
            usual_by_now(baselines, str(client["key"]), figure, metrics) is not None
            for _direction, figure in DIRECTIONS
        ):
            covered += 1
    return covered, reporting


def human_bytes(count: float) -> str:
    """Return ``count`` bytes as "3.2 GB" / "800 MB" / "12 kB"."""
    value = float(count)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":  # noqa: PLR2004
            break
        value /= 1000
    if unit == "B":
        return f"{int(value)} B"
    return f"{value:.1f} {unit}" if value < 10 else f"{value:.0f} {unit}"  # noqa: PLR2004


class NetworkClientUsageAnomalyRule:
    """
    A connected client whose traffic so far today is far above its usual.

    eero reports each client's cumulative download and upload for the day.
    The baseline updater keeps an hourly profile of those figures per client
    (``network.client.<key>.data_*_day_bytes``, keyed by the pseudonymized
    key), so "usual" means what this device had typically moved by the end
    of this hour (see ``hour_metrics``). A device is reported when it
    exceeds that by the threshold percentage AND by at least the byte
    floor: an idle device going from 3 MB to 30 MB is a multiple but not
    news, a TV going from 800 MB to 3 GB is. Upload is medium (a device
    sending a lot is the shape of a compromise), download low.

    One finding per device and direction, each with its own identity and
    a one-day cooldown on that identity (``entity_cooldown_minutes``, keyed
    by a per-device pseudo entity), rather than one aggregated finding
    under the rule-type floor: a benign download alert on one device must
    not hide an upload alert on another for a day, and a device that goes
    from downloading to uploading is a new finding.
    """

    rule_id = "network_client_usage_anomaly"
    requires = frozenset({CAP_CLIENTS, CAP_CLIENT_DATA_DAY, CAP_COUNTER_BASELINES})
    cooldown_minutes = 0
    entity_cooldown_minutes = USAGE_ENTITY_COOLDOWN_MINUTES

    def __init__(
        self,
        *,
        threshold_pct: float = SENTINEL_NETWORK_USAGE_THRESHOLD_PCT,
        min_excess_bytes: int = SENTINEL_NETWORK_USAGE_MIN_EXCESS_BYTES,
        is_entity_excluded: Callable[[str, str], bool] | None = None,
    ) -> None:
        """Initialize with the thresholds and the per-rule entity exclusions."""
        self._threshold = max(0.0, threshold_pct)
        self._min_excess = max(0, min_excess_bytes)
        self._is_entity_excluded = is_entity_excluded

    def _excess(
        self, client: Mapping[str, Any], figure: str, usual: float | None
    ) -> tuple[float, float] | None:
        """Return ``(current, usual)`` when the figure is far above usual."""
        current = client.get(figure)
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            return None
        if usual is None or not math.isfinite(float(current)):
            return None
        if float(current) - usual < self._min_excess:
            return None
        if usual > 0 and float(current) < usual * (1 + self._threshold / 100):
            return None
        return float(current), usual

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding per client and direction far above its usual."""
        section = network_section(snapshot)
        if section is None:
            return []
        baselines = section.get("counter_baselines") or {}
        now = dt_util.parse_datetime(snapshot["generated_at"]) or dt_util.utcnow()
        metrics = hour_metrics(now)
        findings: list[AnomalyFinding] = []
        for client in clients(snapshot):
            if not client.get("connected") or client_excluded(
                client, self.rule_id, self._is_entity_excluded
            ):
                continue
            key = str(client["key"])
            for direction, figure in DIRECTIONS:
                usual = usual_by_now(baselines, key, figure, metrics)
                excess = self._excess(client, figure, usual)
                if excess is not None:
                    findings.append(self._finding(client, direction, *excess))
        return findings

    def _finding(
        self, client: Mapping[str, Any], direction: str, current: float, usual: float
    ) -> AnomalyFinding:
        key = str(client["key"])
        severity: Severity = "medium" if direction == "upload" else "low"
        return make_finding(
            self.rule_id,
            severity=severity,
            evidence={"client_key": key, "direction": direction},
            display={
                "name": describe_client(client),
                "today_bytes": int(current),
                "usual_bytes": int(usual),
            },
            summary=f"{self._describe(client, direction, current, usual)}.",
            suggested_actions=[
                (
                    "If you do not expect this, check what the device is doing "
                    "and block it in your router app"
                ),
                "Open the device's insights in the eero app to see where it went",
            ],
            # The per-device pseudo entity carries the daily cooldown; the
            # tracker, when there is one, lets per-rule exclusions and
            # snoozes address the device the way the other client rules do.
            triggering_entities=[
                client_counter_id(key, direction),
                *(
                    [str(client["tracker_entity_id"])]
                    if client.get("tracker_entity_id")
                    else []
                ),
            ],
        )

    @staticmethod
    def _describe(
        client: Mapping[str, Any], direction: str, current: float, usual: float
    ) -> str:
        verb = "uploaded" if direction == "upload" else "downloaded"
        if usual <= 0:
            return (
                f"{describe_client(client)} has {verb} {human_bytes(current)} so far "
                "today, when it usually has moved nothing by this hour"
            )
        ratio = current / usual
        times = f"{ratio:.1f}" if ratio < 10 else f"{ratio:.0f}"  # noqa: PLR2004
        return (
            f"{describe_client(client)} has {verb} {human_bytes(current)} so far "
            f"today, about {times}x its usual {human_bytes(usual)} by this hour"
        )
