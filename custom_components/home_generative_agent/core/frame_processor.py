"""Frame processing with VLM and face recognition."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any

import aiofiles
import async_timeout
import homeassistant.util.dt as dt_util
from homeassistant.exceptions import HomeAssistantError

from ..agent.tools import analyze_image  # noqa: TID252
from .datetime_utils import DateTimeUtils

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .face_recognition_service import FaceRecognitionService

LOGGER = logging.getLogger(__name__)


class FrameProcessor:
    """Processes individual frames and batches."""

    def __init__(
        self,
        hass: HomeAssistant,
        vision_model: Any,
        face_service: FaceRecognitionService,
        vision_semaphore: asyncio.Semaphore,
        face_timeout: int = 10,
        vision_timeout: int = 60,
        frame_deadline: int = 240,
    ) -> None:
        """
        Initialize frame processor.

        Args:
            hass: Home Assistant instance
            vision_model: VLM model for scene analysis
            face_service: Face recognition service
            vision_semaphore: Global semaphore for VLM concurrency
            face_timeout: Timeout for face recognition in seconds
            vision_timeout: Timeout for VLM analysis in seconds
            frame_deadline: Skip frames older than this in seconds

        """
        self.hass = hass
        self.vision_model = vision_model
        self.face_service = face_service
        self.vision_sem = vision_semaphore
        self._face_timeout = face_timeout
        self._vision_timeout = vision_timeout
        self._frame_deadline = frame_deadline

    def order_by_timestamp(self, batch: list[Path]) -> list[tuple[Path, int]]:
        """
        Sort paths by embedded timestamp.

        Args:
            batch: List of snapshot paths

        Returns:
            List of (path, epoch_timestamp) tuples, sorted by timestamp

        """
        ordered = []
        for path in batch:
            try:
                epoch = DateTimeUtils.epoch_from_snapshot_path(path.name)
                ordered.append((path, epoch))
            except ValueError:
                LOGGER.warning("Could not parse timestamp from %s", path.name)
                continue

        return sorted(ordered, key=lambda x: x[1])

    async def process_single_frame(
        self, path: Path, camera_id: str, prev_description: str | None = None
    ) -> dict[str, list[str]]:
        """
        Process one snapshot: faces + VLM description.

        Args:
            path: Path to snapshot
            camera_id: Camera entity ID
            prev_description: Description of previous frame for context

        Returns:
            Dict mapping frame description to list of recognized names,
            or empty dict if processing failed/skipped

        """
        # Freshness check
        try:
            epoch = DateTimeUtils.epoch_from_snapshot_path(path.name)
        except ValueError:
            LOGGER.warning("[%s] Invalid snapshot filename: %s", camera_id, path.name)
            return {}

        age = dt_util.utcnow().timestamp() - float(epoch)
        if age > self._frame_deadline:
            LOGGER.debug(
                "[%s] Skipping stale snapshot (%ds): %s", camera_id, int(age), path
            )
            return {}

        start_time = monotonic()

        try:
            # Load image
            async with aiofiles.open(path, "rb") as file:
                data = await file.read()

            # Face recognition with timeout
            try:
                async with async_timeout.timeout(self._face_timeout):
                    faces_in_frame = await self.face_service.recognize_faces(
                        data, camera_id
                    )
            except TimeoutError:
                LOGGER.warning(
                    "[%s] Face recognition timed out for %s", camera_id, path
                )
                faces_in_frame = []

            # VLM analysis with global concurrency limit and timeout
            try:
                async with (
                    self.vision_sem,
                    async_timeout.timeout(self._vision_timeout),
                ):
                    frame_description = await analyze_image(
                        self.vision_model, data, None, prev_text=prev_description
                    )
            except TimeoutError:
                LOGGER.warning("[%s] VLM analysis timed out for %s", camera_id, path)
                return {}

        except (FileNotFoundError, HomeAssistantError) as exc:
            if isinstance(exc, FileNotFoundError):
                LOGGER.warning("[%s] Snapshot not found: %s", camera_id, path)
            else:
                LOGGER.exception("[%s] Error analyzing %s", camera_id, path)
            return {}

        else:
            duration_ms = (monotonic() - start_time) * 1000.0
            LOGGER.debug(
                "[%s] Processed frame in %.1fms: %s", camera_id, duration_ms, path.name
            )
            return {frame_description: faces_in_frame}

    async def process_batch(
        self, camera_id: str, ordered_paths: list[tuple[Path, int]]
    ) -> tuple[list[dict[str, list[str]]], list[str]]:
        """
        Process multiple frames in temporal order.

        Args:
            camera_id: Camera entity ID
            ordered_paths: List of (path, epoch) tuples sorted by time

        Returns:
            Tuple of:
                - List of frame descriptions with people
                - List of unique recognized person names

        """
        if not ordered_paths:
            return [], []

        t0 = ordered_paths[0][1]
        frame_descriptions: list[dict[str, list[str]]] = []
        prev_text: str | None = None

        # Process each frame with context from previous
        for path, ts in ordered_paths:
            result = await self.process_single_frame(path, camera_id, prev_text)
            if not result:
                continue

            # Extract description and people
            frame_desc, faces = next(iter(result.items()))

            # Add relative timestamp
            frame_descriptions.append({f"t+{ts - t0}s. {frame_desc}": faces})

            # Use this description as context for next frame
            prev_text = frame_desc

        # Deduplicate near-identical descriptions
        frame_descriptions = self._deduplicate_descriptions(frame_descriptions)

        # Cap to last 8 frames
        frame_descriptions = frame_descriptions[-8:]

        # Extract unique recognized people
        recognized = sorted(
            {
                person
                for desc in frame_descriptions
                for people in desc.values()
                for person in people
                if person != "None"
            }
        )

        return frame_descriptions, recognized

    @staticmethod
    def _deduplicate_descriptions(
        descs: list[dict[str, list[str]]],
    ) -> list[dict[str, list[str]]]:
        """
        Collapse near-duplicate frame texts to reduce prompt size.

        Args:
            descs: List of {description: people} dicts

        Returns:
            Deduplicated list

        """
        out: list[dict[str, list[str]]] = []
        last_norm: str | None = None

        for d in descs:
            # Get description text (single key in dict)
            text = next(iter(d.keys()))

            # Normalize for comparison
            norm = re.sub(r"\s+", " ", text.lower()).strip()

            if norm != last_norm:
                out.append(d)
                last_norm = norm

        return out
