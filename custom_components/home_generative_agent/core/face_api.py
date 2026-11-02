"""Requests to the face-recognition service, with a retry for dead keep-alives."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import urljoin

import httpx
from homeassistant.exceptions import HomeAssistantError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

LOGGER = logging.getLogger(__name__)

# One retry when a pooled keep-alive connection is dead at send time (#900).
# uvicorn closes idle connections after 5 s by default while Home Assistant's
# shared httpx client keeps them for 15 s, and HA's client factory does not
# let a caller shorten that. A request sent right at the server's idle
# deadline races its FIN and fails with RemoteProtocolError, or with ReadError
# when the server resets instead. httpcore drops that connection, so the retry
# goes out on a new one unless another pooled connection went stale at the
# same moment. Every face API call only reads, so sending one twice is safe.
FACE_API_STALE_CONN_RETRIES = 1


class FaceServiceError(HomeAssistantError):
    """The face service could not be reached, failed, or sent no readable answer."""


_STALE_CONN_ERRORS = (httpx.RemoteProtocolError, httpx.ReadError)


def face_api_url(base_url: str, path: str) -> str:
    """Join a face-service base URL and an endpoint path."""
    return urljoin(base_url.rstrip("/") + "/", path)


async def face_api_request(
    send: Callable[[], Awaitable[httpx.Response]],
) -> httpx.Response:
    """Run one face API request, retrying once on a dead pooled connection."""
    for _ in range(FACE_API_STALE_CONN_RETRIES):
        try:
            return await send()
        except _STALE_CONN_ERRORS as err:
            LOGGER.debug(
                "Face API connection dropped (%r); retrying on a new connection",
                err,
            )
    return await send()


async def post_face_analyze(
    client: httpx.AsyncClient,
    base_url: str,
    image_bytes: bytes,
    *,
    request_timeout: float | None = None,
) -> httpx.Response:
    """
    POST an image to the face service's /analyze endpoint.

    ``request_timeout`` applies to each attempt; ``None`` keeps the client's
    own timeout. A caller-side deadline still covers both attempts.
    """
    url = face_api_url(base_url, "analyze")
    files = {"file": ("snapshot.jpg", image_bytes, "image/jpeg")}
    if request_timeout is None:
        return await face_api_request(lambda: client.post(url, files=files))
    return await face_api_request(
        lambda: client.post(url, files=files, timeout=request_timeout)
    )
