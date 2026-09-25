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
    POSTURE_COOLDOWN_MINUTES,
    client_excluded,
    clients,
    listed,
    make_finding,
    network_section,
    noun,
)
from .network_unknown_device_joined import describe_client

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any

    from custom_components.home_generative_agent.sentinel.models import (
        AnomalyFinding,
        Severity,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

# (direction word, client figure) for the two counters the rule compares.
_DIRECTIONS: tuple[tuple[str, str], ...] = (
    ("upload", "data_up_day_bytes"),
    ("download", "data_down_day_bytes"),
)


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
    key), so "usual" means what this device had typically moved by this
    hour. A device is reported when it exceeds that by the threshold
    percentage AND by at least the byte floor: an idle device going from
    3 MB to 30 MB is a multiple but not news, a TV going from 800 MB to
    3 GB is. Upload is medium (a device sending a lot is the shape of a
    compromise), download low.

    One aggregated finding per run, at most once a day (the standing-
    condition cooldown floor): the figures only grow until midnight.
    """

    rule_id = "network_client_usage_anomaly"
    requires = frozenset({CAP_CLIENTS, CAP_CLIENT_DATA_DAY, CAP_COUNTER_BASELINES})
    cooldown_minutes = POSTURE_COOLDOWN_MINUTES

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
        if usual is None or not math.isfinite(usual) or usual < 0:
            return None
        if float(current) - usual < self._min_excess:
            return None
        if usual > 0 and float(current) < usual * (1 + self._threshold / 100):
            return None
        return float(current), usual

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding naming every client far above its usual today."""
        section = network_section(snapshot)
        if section is None:
            return []
        baselines = section.get("counter_baselines") or {}
        now = dt_util.parse_datetime(snapshot["generated_at"]) or dt_util.utcnow()
        # The updater keys the hourly profile by UTC hour.
        hour_metric = f"{METRIC_HOURLY_PREFIX}{dt_util.as_utc(now).hour}"
        hits: list[tuple[dict[str, Any], str, float, float]] = []
        for client in clients(snapshot):
            if not client.get("connected") or client_excluded(
                client, self.rule_id, self._is_entity_excluded
            ):
                continue
            for direction, figure in _DIRECTIONS:
                stats = baselines.get(client_counter_id(str(client["key"]), figure))
                usual = (stats or {}).get(hour_metric)
                excess = self._excess(client, figure, usual)
                if excess is not None:
                    hits.append((client, direction, *excess))
        if not hits:
            return []
        severity: Severity = (
            "medium" if any(d == "upload" for _c, d, _cur, _u in hits) else "low"
        )
        names = [self._describe(*hit) for hit in hits]
        count = len({c["key"] for c, *_rest in hits})
        return [
            make_finding(
                self.rule_id,
                severity=severity,
                evidence={
                    "client_keys": sorted({c["key"] for c, *_rest in hits}),
                    "directions": sorted(f"{c['key']}:{d}" for c, d, *_r in hits),
                },
                display={
                    "names": names,
                    "figures": [
                        {
                            "key": c["key"],
                            "direction": d,
                            "today_bytes": int(cur),
                            "usual_bytes": int(usual),
                        }
                        for c, d, cur, usual in hits
                    ],
                },
                summary=(
                    f"{noun(count, 'A device', 'Devices')} on the network "
                    f"{'has' if count == 1 else 'have'} moved far more data than "
                    f"usual for this hour: {listed(names)}."
                ),
                suggested_actions=[
                    (
                        "If you do not expect this, check what the device is doing "
                        "and block it in your router app"
                    ),
                    "Open the device's insights in the eero app to see where it went",
                ],
                triggering_entities=sorted(
                    {
                        str(c["tracker_entity_id"])
                        for c, *_rest in hits
                        if c.get("tracker_entity_id")
                    }
                ),
            )
        ]

    @staticmethod
    def _describe(
        client: Mapping[str, Any], direction: str, current: float, usual: float
    ) -> str:
        verb = "uploaded" if direction == "upload" else "downloaded"
        if usual <= 0:
            return (
                f"{describe_client(client)} has {verb} {human_bytes(current)} so far "
                "today, when it usually has moved nothing by now"
            )
        ratio = current / usual
        times = f"{ratio:.1f}" if ratio < 10 else f"{ratio:.0f}"  # noqa: PLR2004
        return (
            f"{describe_client(client)} has {verb} {human_bytes(current)} so far "
            f"today, about {times}x its usual {human_bytes(usual)}"
        )
