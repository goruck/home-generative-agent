"""Face detection and recognition via external API."""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urljoin

import aiofiles
import httpx
from PIL import Image

from .datetime_utils import DateTimeUtils
from .path_utils import PathUtils

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .person_gallery import PersonGalleryDAO

LOGGER = logging.getLogger(__name__)


class FaceRecognitionService:
    """Handles face detection and recognition via external API."""

    def __init__(
        self,
        hass: HomeAssistant,
        api_url: str,
        person_dao: PersonGalleryDAO,
        snapshot_root: Path,
        timeout: int = 10,
        save_debug_crops: bool = False,
    ) -> None:
        """Initialize face recognition service.

        Args:
            hass: Home Assistant instance
            api_url: Face recognition API base URL
            person_dao: Person gallery database access
            snapshot_root: Root directory for snapshots
            timeout: HTTP request timeout in seconds
            save_debug_crops: Whether to save debug face crops
        """
        self.hass = hass
        self.api_url = api_url
        self.person_dao = person_dao
        self.snapshot_root = snapshot_root
        self._timeout = timeout
        self._save_debug_crops = save_debug_crops
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Initialize HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
            LOGGER.debug("Face recognition service started")

    async def stop(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None
            LOGGER.debug("Face recognition service stopped")

    async def recognize_faces(
        self, image_bytes: bytes, camera_id: str
    ) -> list[str]:
        """Detect and recognize faces in image.

        Args:
            image_bytes: Image data (JPEG)
            camera_id: Camera entity ID for debug output

        Returns:
            List of recognized person names, or ["Indeterminate"] if no faces
        """
        client = self._client
        if client is None:
            # Fallback if start() wasn't called
            client = httpx.AsyncClient(timeout=self._timeout)

        # Call face API
        try:
            resp = await client.post(
                urljoin(self.api_url.rstrip("/") + "/", "analyze"),
                files={"file": ("snapshot.jpg", image_bytes, "image/jpeg")},
            )
            resp.raise_for_status()
            face_res = resp.json()
        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as err:
            LOGGER.warning("Face API HTTP %s: %s", err.response.status_code, err)
            return []
        except httpx.RequestError as err:
            LOGGER.warning("Face API request error: %s", err)
            return []
        except ValueError:
            LOGGER.warning("Face API returned invalid JSON")
            return []

        faces = face_res.get("faces", [])
        if not faces:
            return ["Indeterminate"]

        # Recognize each face
        recognized: list[str] = []
        for idx, face in enumerate(faces):
            embedding = face["embedding"]
            name = await self.person_dao.recognize_person(embedding)
            recognized.append(name)

            # Optional debug crop
            if self._save_debug_crops:
                await self._save_face_crop(
                    image_bytes, face.get("bbox"), camera_id, idx, name
                )

        return recognized

    async def _save_face_crop(
        self,
        image_bytes: bytes,
        bbox: list[int] | None,
        camera_id: str,
        face_idx: int,
        person_name: str,
    ) -> None:
        """Save debug face crop to disk.

        Args:
            image_bytes: Original image data
            bbox: Bounding box [x1, y1, x2, y2]
            camera_id: Camera entity ID
            face_idx: Face index in this frame
            person_name: Recognized person name
        """
        if not bbox or len(bbox) != 4:
            return

        # Helper to do PIL work off the event loop
        def _crop_and_encode(
            data: bytes, box: list[int], pad: float, min_px: int
        ) -> bytes | None:
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1

                if w <= 0 or h <= 0:
                    return None

                # Add padding
                dx, dy = int(w * pad), int(h * pad)
                x1 = max(0, x1 - dx)
                y1 = max(0, y1 - dy)
                x2 = min(img.width, x2 + dx)
                y2 = min(img.height, y2 + dy)

                if x2 <= x1 or y2 <= y1:
                    return None

                crop = img.crop((x1, y1, x2, y2))

                # Resize if too small
                if crop.width < min_px or crop.height < min_px:
                    crop = crop.resize(
                        (min_px, min_px), resample=Image.Resampling.LANCZOS
                    )

                # Encode to JPEG
                out = io.BytesIO()
                crop.save(out, format="JPEG", quality=95, subsampling=0)
                return out.getvalue()
            except (Image.UnidentifiedImageError, OSError):
                return None

        try:
            # Prepare directory
            face_dir = PathUtils.face_debug_dir(self.snapshot_root, camera_id)
            await self.hass.async_add_executor_job(PathUtils.ensure_dir, face_dir)

            # Crop and encode
            jpeg_bytes = await self.hass.async_add_executor_job(
                _crop_and_encode, image_bytes, bbox, 0.3, 128
            )
            if not jpeg_bytes:
                return

            # Save
            timestamp = DateTimeUtils.snapshot_timestamp()
            face_file = face_dir / f"face_{timestamp}_{face_idx}_{person_name}.jpg"
            async with aiofiles.open(face_file, "wb") as f:
                await f.write(jpeg_bytes)

            LOGGER.debug("Saved face crop: %s", face_file)

        except asyncio.CancelledError:
            raise
        except OSError as err:
            LOGGER.warning("Failed to save face crop: %s", err)
