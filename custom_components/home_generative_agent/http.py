"""HTTP endpoints for Home Generative Agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from aiohttp import multipart, web
from homeassistant.helpers.http import HomeAssistantView

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .core.runtime import HGAConfigEntry

LOGGER = logging.getLogger(__name__)

MAX_UPLOAD_BYTES: Final = 10 * 1024 * 1024
ALLOWED_EXTENSIONS: Final = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _is_allowed_image(part: multipart.BodyPartReader) -> bool:
    content_type = part.headers.get("Content-Type", "")
    if content_type.startswith("image/"):
        return True
    if part.filename:
        return Path(part.filename).suffix.lower() in ALLOWED_EXTENSIONS
    return False


class EnrollPersonView(HomeAssistantView):
    """Accept an image upload and enroll a person."""

    url = "/api/home_generative_agent/enroll"
    name = "api:home_generative_agent:enroll_person"
    requires_auth = True

    def __init__(self, hass: HomeAssistant, entry: HGAConfigEntry) -> None:
        """Initialize enroll person view."""
        self._hass = hass
        self._entry = entry

    async def post(self, request: web.Request) -> web.Response:  # noqa: PLR0911, PLR0912
        """Handle POST request to enroll a person."""
        try:
            reader = await request.multipart()
        except Exception as err:  # noqa: BLE001
            try:
                form = await request.post()
            except Exception:  # noqa: BLE001
                return web.json_response(
                    {"status": "error", "message": f"Invalid upload: {err}"},
                    status=400,
                )
            name = str(form.get("name", "")).strip() if "name" in form else ""
            if name:
                return web.json_response(
                    {"status": "error", "message": "Image is required."},
                    status=400,
                )
            return web.json_response(
                {"status": "error", "message": "Name is required."},
                status=400,
            )

        name: str | None = None
        images: list[bytes] = []

        async for part in reader:
            if part is None:
                continue
            part = cast("multipart.BodyPartReader", part)
            if part.name == "name":
                name = (await part.text()).strip()
            elif part.name == "file":
                if not _is_allowed_image(part):
                    return web.json_response(
                        {"status": "error", "message": "Unsupported file type."},
                        status=400,
                    )
                data = bytearray()
                while True:
                    chunk = await part.read_chunk(64 * 1024)
                    if not chunk:
                        break
                    data.extend(chunk)
                    if len(data) > MAX_UPLOAD_BYTES:
                        return web.json_response(
                            {
                                "status": "error",
                                "message": "File is too large.",
                            },
                            status=413,
                        )
                images.append(bytes(data))

        if not name:
            return web.json_response(
                {"status": "error", "message": "Name is required."},
                status=400,
            )
        if not images:
            return web.json_response(
                {"status": "error", "message": "Image is required."},
                status=400,
            )

        dao = self._entry.runtime_data.person_gallery
        enrolled = 0
        skipped = 0
        for img_bytes in images:
            ok = await dao.enroll_from_image(
                self._entry.runtime_data.face_api_url, name, img_bytes
            )
            if not ok:
                skipped += 1
                continue
            enrolled += 1

        if enrolled == 0:
            return web.json_response(
                {"status": "error", "message": "No face found in image(s)."},
                status=400,
            )

        return web.json_response(
            {
                "status": "ok",
                "message": f"Enrolled {enrolled} image(s). Skipped {skipped}.",
                "name": name,
                "count": enrolled,
            }
        )
