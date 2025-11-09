"""Tool calling metrics and execution tracking for Home Generative Agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class ToolCallMetrics:
    """Track execution metrics for a single tool call."""

    tool_name: str
    call_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    duration_ms: float | None = None
    success: bool | None = None
    error_type: str | None = (
        None  # "validation" | "execution" | "timeout" | "not_found"
    )
    error_message: str | None = None
    inputs_hash: str | None = None  # Sanitized hash of inputs for tracking
    response_size_bytes: int | None = None

    def finalize(
        self,
        success: bool,
        error_type: str | None = None,
        error_message: str | None = None,
        response_size_bytes: int | None = None,
    ) -> None:
        """Mark tool call as complete with results."""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.success = success
        self.error_type = error_type
        self.error_message = error_message
        self.response_size_bytes = response_size_bytes

    def is_timeout(self, timeout_ms: float) -> bool:
        """Check if execution exceeded timeout."""
        return self.duration_ms is not None and self.duration_ms > timeout_ms


class ToolCallRateLimiter:
    """Rate limit tool calls using sliding window algorithm."""

    def __init__(self, max_calls_per_minute: int = 30, window_minutes: int = 1):
        """Initialize rate limiter.

        Args:
            max_calls_per_minute: Maximum calls allowed within window
            window_minutes: Sliding window size in minutes
        """
        self.max_calls = max_calls_per_minute
        self.window = timedelta(minutes=window_minutes)
        self.calls: list[tuple[str, datetime]] = []  # (tool_name, timestamp)

    def can_call(self, tool_name: str) -> tuple[bool, int, float]:
        """Check if tool call is allowed.

        Args:
            tool_name: Name of tool to call

        Returns:
            Tuple of (allowed, current_call_count, reset_seconds)
        """
        now = datetime.now()
        cutoff = now - self.window

        # Prune old calls outside window
        self.calls = [(name, ts) for name, ts in self.calls if ts > cutoff]

        current_count = len(self.calls)
        if current_count >= self.max_calls:
            # Calculate when next slot opens
            oldest = self.calls[0][1]
            reset_time = oldest + self.window
            reset_seconds = (reset_time - now).total_seconds()
            return False, current_count, max(0, reset_seconds)

        return True, current_count, 0.0

    def record_call(self, tool_name: str, timestamp: datetime | None = None) -> None:
        """Record a tool call."""
        if timestamp is None:
            timestamp = datetime.now()
        self.calls.append((tool_name, timestamp))

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        now = datetime.now()
        cutoff = now - self.window
        active_calls = [(name, ts) for name, ts in self.calls if ts > cutoff]

        return {
            "total_recorded": len(self.calls),
            "active_in_window": len(active_calls),
            "max_allowed": self.max_calls,
            "window_minutes": self.window.total_seconds() / 60,
        }


class ToolMetricsCollector:
    """Collect and aggregate tool call metrics."""

    def __init__(self, retention_minutes: int = 60):
        """Initialize metrics collector.

        Args:
            retention_minutes: How long to retain metrics
        """
        self.retention = timedelta(minutes=retention_minutes)
        self.metrics: list[ToolCallMetrics] = []

    def add_metric(self, metric: ToolCallMetrics) -> None:
        """Add a tool call metric."""
        self.metrics.append(metric)
        self._cleanup_old_metrics()

    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff = datetime.now() - self.retention
        self.metrics = [m for m in self.metrics if m.start_time > cutoff]

    def get_summary(self, tool_name: str | None = None) -> dict[str, Any]:
        """Get aggregated metrics summary.

        Args:
            tool_name: Filter by tool name, None for all

        Returns:
            Dictionary with aggregated statistics
        """
        metrics = self.metrics
        if tool_name:
            metrics = [m for m in metrics if m.tool_name == tool_name]

        if not metrics:
            return {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "error_breakdown": {},
            }

        successful = sum(1 for m in metrics if m.success)
        failed = sum(1 for m in metrics if not m.success)
        total = len(metrics)

        errors: dict[str, int] = {}
        for m in metrics:
            if m.error_type:
                errors[m.error_type] = errors.get(m.error_type, 0) + 1

        durations = [m.duration_ms for m in metrics if m.duration_ms is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "total_calls": total,
            "successful_calls": successful,
            "failed_calls": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_duration_ms": avg_duration,
            "error_breakdown": errors,
        }

    def get_tool_list_summary(self) -> dict[str, Any]:
        """Get per-tool summaries."""
        tool_names = {m.tool_name for m in self.metrics}
        return {name: self.get_summary(name) for name in sorted(tool_names)}
