"""Performance metrics collection and reporting for video analysis."""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

LOGGER = logging.getLogger(__name__)


@dataclass
class _Metrics:
    """Per-camera metrics storage."""

    captured: int = 0
    enqueued: int = 0
    skipped_duplicate: int = 0
    dropped_stale: int = 0
    analyzed: int = 0
    timeouts: int = 0
    lat_ms: deque[float] = field(default_factory=lambda: deque(maxlen=512))


class VideoAnalyzerMetrics:
    """Collects and reports performance metrics."""

    def __init__(
        self,
        report_interval_sec: int = 3600,
        latency_history_size: int = 512,
    ) -> None:
        """Initialize metrics collector.

        Args:
            report_interval_sec: How often to flush and report metrics
            latency_history_size: Number of latency samples to keep per camera
        """
        self._report_interval = report_interval_sec
        self._history_size = latency_history_size
        self._metrics: dict[str, _Metrics] = defaultdict(_Metrics)

    def increment(self, camera_id: str, metric: str, count: int = 1) -> None:
        """Increment a counter metric.

        Args:
            camera_id: Camera identifier
            metric: Metric name (captured, enqueued, skipped_duplicate, etc.)
            count: Amount to increment by
        """
        m = self._metrics[camera_id]

        if metric == "captured":
            m.captured += count
        elif metric == "enqueued":
            m.enqueued += count
        elif metric == "skipped_duplicate":
            m.skipped_duplicate += count
        elif metric == "dropped_stale":
            m.dropped_stale += count
        elif metric == "analyzed":
            m.analyzed += count
        elif metric == "timeouts":
            m.timeouts += count
        # Unknown metrics are silently ignored

    def record_latency(self, camera_id: str, ms: float) -> None:
        """Record a latency sample in milliseconds.

        Args:
            camera_id: Camera identifier
            ms: Latency in milliseconds
        """
        self._metrics[camera_id].lat_ms.append(float(ms))

    async def flush_and_report(self, _now: datetime) -> None:
        """Log aggregated metrics and reset counters.

        Args:
            _now: Current datetime (unused, for compatibility with time interval callback)
        """
        for cam, m in self._metrics.items():
            lat_list = list(m.lat_ms)
            avg_ms = statistics.fmean(lat_list) if lat_list else 0.0
            p95_ms = self._percentile(lat_list, 95.0) if lat_list else 0.0

            msg = (
                "[%s] Metrics (last interval): "
                "captured=%d enqueued=%d skipped_duplicate=%d "
                "dropped_stale=%d analyzed=%d timeouts=%d "
                "avg_latency_ms=%.1f p95_latency_ms=%.1f"
            )
            LOGGER.info(
                msg,
                cam,
                m.captured,
                m.enqueued,
                m.skipped_duplicate,
                m.dropped_stale,
                m.analyzed,
                m.timeouts,
                avg_ms,
                p95_ms,
            )

            # Reset counters and samples
            m.captured = 0
            m.enqueued = 0
            m.skipped_duplicate = 0
            m.dropped_stale = 0
            m.analyzed = 0
            m.timeouts = 0
            m.lat_ms.clear()

    @staticmethod
    def _percentile(values: Iterable[float], q: float) -> float:
        """Calculate nearest-rank percentile.

        Args:
            values: Iterable of numeric values
            q: Percentile (0-100)

        Returns:
            Percentile value, or 0.0 if no data
        """
        xs = list(values)
        if not xs:
            return 0.0

        xs.sort()

        if len(xs) == 1:
            return float(xs[0])

        # Clamp rank to valid range
        k = max(0, min(len(xs) - 1, round((q / 100.0) * (len(xs) - 1))))
        return float(xs[k])
