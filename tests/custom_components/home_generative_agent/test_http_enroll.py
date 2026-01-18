# ruff: noqa: S101
"""Tests for the enroll person HTTP endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest
from aiohttp import FormData
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.const import (
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_OK,
    HTTP_STATUS_REQUEST_TOO_LARGE,
)
from custom_components.home_generative_agent.core.runtime import HGAConfigEntry, HGAData
from custom_components.home_generative_agent.http import (
    MAX_UPLOAD_BYTES,
    EnrollPersonView,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aiohttp.test_utils import TestClient
    from homeassistant.core import HomeAssistant


@dataclass
class DummyDAO:
    """Minimal DAO stub for enroll_from_image."""

    result: bool = True
    results: list[bool] | None = None
    last_args: tuple[str, str, bytes] | None = None

    async def enroll_from_image(
        self, face_api_url: str, name: str, image_bytes: bytes
    ) -> bool:
        """enroll_from_image dummy implementation."""
        self.last_args = (face_api_url, name, image_bytes)
        if self.results is not None:
            return self.results.pop(0)
        return self.result


class DummyEntry:
    """Minimal config entry stand-in."""

    def __init__(self, dao: DummyDAO) -> None:
        """Initialize the dummy entry."""
        self.runtime_data = HGAData(
            options={},
            chat_model=None,
            chat_model_options={},
            vision_model=None,
            summarization_model=None,
            pool=None,
            store=None,
            checkpointer=None,
            video_analyzer=None,  # type: ignore[arg-type]
            face_api_url="http://face-api",
            face_recognition=False,
            person_gallery=dao,
            pending_actions={},
        )


@pytest.mark.asyncio
async def test_enroll_missing_name(
    hass: HomeAssistant, hass_client: Callable[[], Awaitable[TestClient]]
) -> None:
    """Return 400 when the name field is missing."""
    await async_setup_component(hass, "http", {})
    entry = cast("HGAConfigEntry", DummyEntry(DummyDAO()))
    hass.http.register_view(EnrollPersonView(hass, entry))
    client = await hass_client()

    form = FormData()
    form.add_field("file", b"img", filename="face.jpg", content_type="image/jpeg")

    response = await client.post("/api/home_generative_agent/enroll", data=form)
    data = await response.json()

    assert response.status == HTTP_STATUS_BAD_REQUEST
    assert "Name is required" in data["message"]


@pytest.mark.asyncio
async def test_enroll_missing_file(
    hass: HomeAssistant, hass_client: Callable[[], Awaitable[TestClient]]
) -> None:
    """Return 400 when the image field is missing."""
    await async_setup_component(hass, "http", {})
    entry = cast("HGAConfigEntry", DummyEntry(DummyDAO()))
    hass.http.register_view(EnrollPersonView(hass, entry))
    client = await hass_client()

    form = FormData()
    form.add_field("name", "Alice")

    response = await client.post("/api/home_generative_agent/enroll", data=form)
    data = await response.json()

    assert response.status == HTTP_STATUS_BAD_REQUEST
    assert "Image is required" in data["message"]


@pytest.mark.asyncio
async def test_enroll_invalid_file_type(
    hass: HomeAssistant, hass_client: Callable[[], Awaitable[TestClient]]
) -> None:
    """Return 400 when the file type is unsupported."""
    await async_setup_component(hass, "http", {})
    entry = cast("HGAConfigEntry", DummyEntry(DummyDAO()))
    hass.http.register_view(EnrollPersonView(hass, entry))
    client = await hass_client()

    form = FormData()
    form.add_field("name", "Alice")
    form.add_field("file", b"nope", filename="face.txt", content_type="text/plain")

    response = await client.post("/api/home_generative_agent/enroll", data=form)
    data = await response.json()

    assert response.status == HTTP_STATUS_BAD_REQUEST
    assert "Unsupported file type" in data["message"]


@pytest.mark.asyncio
async def test_enroll_too_large(
    hass: HomeAssistant, hass_client: Callable[[], Awaitable[TestClient]]
) -> None:
    """Return 413 when the upload exceeds the size limit."""
    await async_setup_component(hass, "http", {})
    entry = cast("HGAConfigEntry", DummyEntry(DummyDAO()))
    hass.http.register_view(EnrollPersonView(hass, entry))
    client = await hass_client()

    form = FormData()
    form.add_field("name", "Alice")
    form.add_field(
        "file",
        b"a" * (MAX_UPLOAD_BYTES + 1),
        filename="face.jpg",
        content_type="image/jpeg",
    )

    response = await client.post("/api/home_generative_agent/enroll", data=form)
    data = await response.json()

    assert response.status == HTTP_STATUS_REQUEST_TOO_LARGE
    assert "File is too large" in data["message"]


@pytest.mark.asyncio
async def test_enroll_success(
    hass: HomeAssistant, hass_client: Callable[[], Awaitable[TestClient]]
) -> None:
    """Return 200 when a valid image enrolls successfully."""
    await async_setup_component(hass, "http", {})
    dao = DummyDAO(result=True)
    entry = cast("HGAConfigEntry", DummyEntry(dao))
    hass.http.register_view(EnrollPersonView(hass, entry))
    client = await hass_client()

    form = FormData()
    form.add_field("name", "Alice")
    form.add_field("file", b"img", filename="face.jpg", content_type="image/jpeg")

    response = await client.post("/api/home_generative_agent/enroll", data=form)
    data = await response.json()

    assert response.status == HTTP_STATUS_OK
    assert data["status"] == "ok"
    assert dao.last_args is not None


@pytest.mark.asyncio
async def test_enroll_skips_failed_images(
    hass: HomeAssistant, hass_client: Callable[[], Awaitable[TestClient]]
) -> None:
    """Return 200 when some images fail but at least one enrolls."""
    await async_setup_component(hass, "http", {})
    dao = DummyDAO(results=[False, True, False])
    entry = cast("HGAConfigEntry", DummyEntry(dao))
    hass.http.register_view(EnrollPersonView(hass, entry))
    client = await hass_client()

    form = FormData()
    form.add_field("name", "Alice")
    form.add_field("file", b"img1", filename="face1.jpg", content_type="image/jpeg")
    form.add_field("file", b"img2", filename="face2.jpg", content_type="image/jpeg")
    form.add_field("file", b"img3", filename="face3.jpg", content_type="image/jpeg")

    response = await client.post("/api/home_generative_agent/enroll", data=form)
    data = await response.json()

    assert response.status == HTTP_STATUS_OK
    assert data["count"] == 1


@pytest.mark.asyncio
async def test_enroll_all_failures(
    hass: HomeAssistant, hass_client: Callable[[], Awaitable[TestClient]]
) -> None:
    """Return 400 when no images contain a detectable face."""
    await async_setup_component(hass, "http", {})
    dao = DummyDAO(results=[False, False])
    entry = cast("HGAConfigEntry", DummyEntry(dao))
    hass.http.register_view(EnrollPersonView(hass, entry))
    client = await hass_client()

    form = FormData()
    form.add_field("name", "Alice")
    form.add_field("file", b"img1", filename="face1.jpg", content_type="image/jpeg")
    form.add_field("file", b"img2", filename="face2.jpg", content_type="image/jpeg")

    response = await client.post("/api/home_generative_agent/enroll", data=form)
    data = await response.json()

    assert response.status == HTTP_STATUS_BAD_REQUEST
    assert "No face found" in data["message"]
