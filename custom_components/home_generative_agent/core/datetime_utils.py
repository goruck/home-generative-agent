"""Datetime utilities for consistent timestamp handling."""

from __future__ import annotations

from datetime import datetime

import homeassistant.util.dt as dt_util
from homeassistant.exceptions import HomeAssistantError


class DateTimeUtils:
    """Centralized datetime operations for consistency."""

    SNAPSHOT_FORMAT = "%Y%m%d_%H%M%S"

    @staticmethod
    def parse_or_default(
        datetime_str: str | None,
        default: datetime,
        error_message: str | None = None,
    ) -> datetime:
        """
        Parse datetime string or return default.

        Args:
            datetime_str: String to parse (ISO format or None)
            default: Default datetime if parsing fails
            error_message: Optional error message to raise on failure

        Returns:
            Parsed datetime in UTC

        Raises:
            HomeAssistantError: If error_message provided and parsing fails

        """
        if datetime_str is None:
            return default

        parsed = dt_util.parse_datetime(datetime_str)
        if parsed is None:
            if error_message:
                raise HomeAssistantError(error_message)
            return default

        return dt_util.as_utc(parsed)

    @staticmethod
    def snapshot_timestamp(dt: datetime | None = None) -> str:
        """
        Get standardized snapshot filename timestamp.

        Args:
            dt: Datetime to format (defaults to now)

        Returns:
            Timestamp string in YYYYMMDD_HHMMSS format

        """
        if dt is None:
            dt = dt_util.utcnow()
        return dt_util.as_local(dt).strftime(DateTimeUtils.SNAPSHOT_FORMAT)

    @staticmethod
    def parse_snapshot_timestamp(filename: str) -> datetime:
        """
        Extract datetime from snapshot filename.

        Args:
            filename: Filename like "snapshot_20250426_002804.jpg" or just "20250426_002804"

        Returns:
            Datetime in local timezone

        Raises:
            ValueError: If filename format is invalid

        """
        # Extract timestamp from "snapshot_20250426_002804.jpg"
        timestamp_str = filename.replace("snapshot_", "").replace(".jpg", "")

        # Parse in local timezone (as files were created with as_local)
        dt_naive = datetime.strptime(timestamp_str, DateTimeUtils.SNAPSHOT_FORMAT)

        # Attach default timezone
        dt_local = dt_naive.replace(tzinfo=dt_util.DEFAULT_TIME_ZONE)

        return dt_local

    @staticmethod
    def to_epoch(dt: datetime) -> int:
        """
        Convert datetime to UTC epoch seconds.

        Args:
            dt: Datetime to convert

        Returns:
            Unix timestamp (seconds since epoch)

        """
        return int(dt_util.as_timestamp(dt))

    @staticmethod
    def epoch_from_snapshot_path(filename: str) -> int:
        """
        Parse snapshot filename and return epoch timestamp.

        Args:
            filename: Snapshot filename (with or without "snapshot_" prefix)

        Returns:
            Unix timestamp in seconds

        Raises:
            ValueError: If filename format is invalid

        """
        dt_local = DateTimeUtils.parse_snapshot_timestamp(filename)
        return DateTimeUtils.to_epoch(dt_local)
