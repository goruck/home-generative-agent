# ruff: noqa: S101
"""Tests for the enroll person HTTP endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from aiohttp import FormData
from homeassistant.config_entries import ConfigEntryState
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.const import (
    DOMAIN,
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_OK,
    HTTP_STATUS_REQUEST_TOO_LARGE,
    HTTP_STATUS_SERVICE_UNAVAILABLE,
)
from custom_components.home_generative_agent.core.runtime import HGAData
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


def _add_loaded_entry(hass: HomeAssistant, dao: DummyDAO) -> MockConfigEntry:
    """Register a LOADED config entry the view can resolve per request."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    entry.mock_state(hass, ConfigEntryState.LOADED)
    entry.runtime_data = HGAData(
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
        suppression=None,
        sentinel=None,
        notifier=None,
        action_handler=None,
        audit_store=None,
        explainer=None,
        discovery_store=None,
        discovery_engine=None,
        proposal_store=None,
        rule_registry=None,
    )
    return entry


@pytest.mark.asyncio
async def test_enroll_missing_name(
    hass: HomeAssistant, hass_client: Callable[[], Awaitable[TestClient]]
) -> None:
    """Return 400 when the name field is missing."""
    await async_setup_component(hass, "http", {})
    _add_loaded_entry(hass, DummyDAO())
    hass.http.register_view(EnrollPersonView(hass))
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
    _add_loaded_entry(hass, DummyDAO())
    hass.http.register_view(EnrollPersonView(hass))
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
    _add_loaded_entry(hass, DummyDAO())
    hass.http.register_view(EnrollPersonView(hass))
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
    _add_loaded_entry(hass, DummyDAO())
    hass.http.register_view(EnrollPersonView(hass))
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
    _add_loaded_entry(hass, dao)
    hass.http.register_view(EnrollPersonView(hass))
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
    _add_loaded_entry(hass, dao)
    hass.http.register_view(EnrollPersonView(hass))
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
    _add_loaded_entry(hass, dao)
    hass.http.register_view(EnrollPersonView(hass))
    client = await hass_client()

    form = FormData()
    form.add_field("name", "Alice")
    form.add_field("file", b"img1", filename="face1.jpg", content_type="image/jpeg")
    form.add_field("file", b"img2", filename="face2.jpg", content_type="image/jpeg")

    response = await client.post("/api/home_generative_agent/enroll", data=form)
    data = await response.json()

    assert response.status == HTTP_STATUS_BAD_REQUEST
    assert "No face found" in data["message"]


@pytest.mark.asyncio
async def test_enroll_no_loaded_entry_returns_503(
    hass: HomeAssistant, hass_client: Callable[[], Awaitable[TestClient]]
) -> None:
    """Return 503 when no config entry is currently loaded."""
    await async_setup_component(hass, "http", {})
    hass.http.register_view(EnrollPersonView(hass))
    client = await hass_client()

    form = FormData()
    form.add_field("name", "Alice")
    form.add_field("file", b"img", filename="face.jpg", content_type="image/jpeg")

    response = await client.post("/api/home_generative_agent/enroll", data=form)
    data = await response.json()

    assert response.status == HTTP_STATUS_SERVICE_UNAVAILABLE
    assert "not loaded" in data["message"]


@pytest.mark.asyncio
async def test_enroll_returns_503_while_the_only_entry_is_reloading(
    hass: HomeAssistant, hass_client: Callable[[], Awaitable[TestClient]]
) -> None:
    """
    A present-but-not-LOADED entry must be skipped, yielding 503.

    This is the mid-reload window: the entry still exists, but its
    runtime_data is being torn down. The zero-entry 503 test cannot catch a
    regression that returns any entry regardless of state — only an entry
    that is present and NOT loaded exercises the state check in the loop.
    """
    await async_setup_component(hass, "http", {})
    dao = DummyDAO()
    entry = _add_loaded_entry(hass, dao)
    entry.mock_state(hass, ConfigEntryState.UNLOAD_IN_PROGRESS)
    hass.http.register_view(EnrollPersonView(hass))
    client = await hass_client()

    form = FormData()
    form.add_field("name", "Alice")
    form.add_field("file", b"img", filename="face.jpg", content_type="image/jpeg")

    response = await client.post("/api/home_generative_agent/enroll", data=form)
    data = await response.json()

    assert response.status == HTTP_STATUS_SERVICE_UNAVAILABLE
    assert "not loaded" in data["message"]
    assert dao.last_args is None, "the unloading entry's DAO was reached"


@pytest.mark.asyncio
async def test_enroll_resolves_current_entry_not_registering_entry(
    hass: HomeAssistant, hass_client: Callable[[], Awaitable[TestClient]]
) -> None:
    """
    The view reaches the live entry generation, not the one at register time.

    Regression shape: remove-and-re-add builds a NEW ConfigEntry, and a view
    pinning the registering entry would dereference deleted runtime_data.
    """
    await async_setup_component(hass, "http", {})
    old_dao = DummyDAO(result=True)
    old_entry = _add_loaded_entry(hass, old_dao)
    hass.http.register_view(EnrollPersonView(hass))
    client = await hass_client()

    # Simulate remove-and-re-add: old generation gone, new entry loaded.
    old_entry.mock_state(hass, ConfigEntryState.NOT_LOADED)
    await hass.config_entries.async_remove(old_entry.entry_id)
    new_dao = DummyDAO(result=True)
    _add_loaded_entry(hass, new_dao)

    form = FormData()
    form.add_field("name", "Alice")
    form.add_field("file", b"img", filename="face.jpg", content_type="image/jpeg")

    response = await client.post("/api/home_generative_agent/enroll", data=form)

    assert response.status == HTTP_STATUS_OK
    assert new_dao.last_args is not None
    assert old_dao.last_args is None


@pytest.mark.asyncio
async def test_enroll_returns_503_when_teardown_races_the_upload(
    hass: HomeAssistant, hass_client: Callable[[], Awaitable[TestClient]]
) -> None:
    """
    A reload tearing the entry down mid-upload returns 503, not a 500.

    The DAO raising between images stands in for the real shape: unload
    closes the pool (and deletes runtime_data) under an in-flight multi-image
    enroll. aiohttp would otherwise convert the raise into a 500 traceback.
    """
    await async_setup_component(hass, "http", {})

    class _TearingDAO(DummyDAO):
        async def enroll_from_image(
            self, face_api_url: str, name: str, image_bytes: bytes
        ) -> bool:
            if self.last_args is not None:
                msg = "the pool was closed by a mid-upload reload"
                raise RuntimeError(msg)
            self.last_args = (face_api_url, name, image_bytes)
            return True

    _add_loaded_entry(hass, _TearingDAO())
    hass.http.register_view(EnrollPersonView(hass))
    client = await hass_client()

    form = FormData()
    form.add_field("name", "Alice")
    form.add_field("file", b"img1", filename="face1.jpg", content_type="image/jpeg")
    form.add_field("file", b"img2", filename="face2.jpg", content_type="image/jpeg")

    response = await client.post("/api/home_generative_agent/enroll", data=form)
    data = await response.json()

    assert response.status == HTTP_STATUS_SERVICE_UNAVAILABLE
    assert "reloading" in data["message"]
