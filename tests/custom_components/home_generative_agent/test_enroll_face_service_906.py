# ruff: noqa: S101
"""
Enrollment names the face service when it fails (issue #906).

A face service that is down, answers an HTTP error or sends no readable
answer raises ``FaceServiceError`` naming the service and what it did, so
the enroll service and the upload view can say so instead of a traceback or
"Home Generative Agent may be reloading".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest

from custom_components.home_generative_agent.core.face_api import FaceServiceError
from custom_components.home_generative_agent.core.person_gallery import (
    PersonGalleryDAO,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from homeassistant.core import HomeAssistant


async def _enroll(
    hass: HomeAssistant,
    handler: Callable[[httpx.Request], httpx.Response],
    url: str = "http://face-api:8000",
) -> bool:
    dao = PersonGalleryDAO(cast("Any", object()), hass)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        dao._client = client
        return await dao.enroll_from_image(url, "Alice", b"img")


def _refused(request: httpx.Request) -> httpx.Response:
    msg = "Connection refused"
    raise httpx.ConnectError(msg, request=request)


@pytest.mark.asyncio
async def test_a_down_face_service_is_named(hass: HomeAssistant) -> None:
    with pytest.raises(FaceServiceError) as info:
        await _enroll(hass, _refused)
    assert str(info.value) == (
        "Could not reach the face service at http://face-api:8000: "
        "ConnectError: Connection refused"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 500, 503])
async def test_an_http_error_from_the_face_service_is_named(
    hass: HomeAssistant, status: int
) -> None:
    with pytest.raises(FaceServiceError, match=f"answered HTTP {status}"):
        await _enroll(hass, lambda _request: httpx.Response(status))


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 413, 415, 422])
async def test_an_image_the_service_rejects_is_skipped(
    hass: HomeAssistant, status: int
) -> None:
    """The image, not the service, is at fault: skip it, keep the batch going."""
    assert not await _enroll(hass, lambda _request: httpx.Response(status))


@pytest.mark.asyncio
async def test_the_url_keeps_its_case(hass: HomeAssistant) -> None:
    with pytest.raises(FaceServiceError) as info:
        await _enroll(
            hass, lambda _request: httpx.Response(500), url="http://Face-Box:8000/API/"
        )
    assert str(info.value) == (
        "The face service at http://face-box:8000/API/ answered HTTP 500"
    )


@pytest.mark.asyncio
async def test_an_invalid_url_is_named(hass: HomeAssistant) -> None:
    def _invalid(_request: httpx.Request) -> httpx.Response:
        msg = "Invalid port"
        raise httpx.InvalidURL(msg)

    with pytest.raises(FaceServiceError, match="Could not reach the face service"):
        await _enroll(hass, _invalid)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"<html>proxy error</html>"),
        httpx.Response(200, json=["not", "an", "object"]),
    ],
)
async def test_an_unreadable_answer_is_named(
    hass: HomeAssistant, response: httpx.Response
) -> None:
    with pytest.raises(FaceServiceError, match=r"(?i)the face service at"):
        await _enroll(hass, lambda _request: response)


@pytest.mark.asyncio
async def test_the_message_never_shows_url_credentials(hass: HomeAssistant) -> None:
    with pytest.raises(FaceServiceError) as info:
        await _enroll(hass, _refused, url="http://user:s3cret@face-api:8000/?t=x")
    assert "s3cret" not in str(info.value)
    assert "http://face-api:8000/" in str(info.value)


@pytest.mark.asyncio
async def test_no_face_is_still_a_plain_false(hass: HomeAssistant) -> None:
    """A working service that found no face is not a service failure."""
    assert not await _enroll(
        hass, lambda _request: httpx.Response(200, json={"faces": []})
    )


def test_it_is_a_home_assistant_error() -> None:
    """The enroll service lets it through as is: HA shows its message."""
    from homeassistant.exceptions import HomeAssistantError  # noqa: PLC0415

    assert issubclass(FaceServiceError, HomeAssistantError)
