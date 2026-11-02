# ruff: noqa: S101
"""
A dead keep-alive connection must not drop a frame's face results (issue #900).

uvicorn closes idle keep-alive connections after 5 s by default while Home
Assistant's shared httpx client pools them for 15 s. A POST sent right at the
server's idle deadline races the server's FIN and fails with
``RemoteProtocolError`` ("Server disconnected without sending a response.") or
``ReadError``. ``recognize_faces`` used to log a warning and return ``[]`` for
that frame, and enrollment failed the same way. Both now retry once.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest

from custom_components.home_generative_agent.core import utils as core_utils
from custom_components.home_generative_agent.core.person_gallery import (
    FACE_EMBEDDING_DIMS,
    PersonGalleryDAO,
)
from custom_components.home_generative_agent.core.video_analyzer import (
    VideoAnalyzer,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from homeassistant.core import HomeAssistant

_DISCONNECTED = "Server disconnected without sending a response."


def _transport(
    failures: list[Callable[[httpx.Request], Exception]],
    payload: dict[str, object],
    seen: list[httpx.Request],
) -> httpx.MockTransport:
    """Raise each queued failure in turn, then answer 200 with ``payload``."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if failures:
            raise failures.pop(0)(request)
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def _remote_protocol(request: httpx.Request) -> Exception:
    return httpx.RemoteProtocolError(_DISCONNECTED, request=request)


def _read_error(request: httpx.Request) -> Exception:
    return httpx.ReadError("", request=request)


def _connect_error(request: httpx.Request) -> Exception:
    return httpx.ConnectError("Connection refused", request=request)


def _analyzer(hass: HomeAssistant, client: httpx.AsyncClient) -> VideoAnalyzer:
    runtime_data = SimpleNamespace(
        face_recognition=True,
        face_api_url="http://face-api",
        person_gallery=None,
    )
    analyzer = VideoAnalyzer(
        hass, cast("Any", SimpleNamespace(runtime_data=runtime_data))
    )
    analyzer._httpx_client = client
    return analyzer


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [_remote_protocol, _read_error])
async def test_recognize_faces_retries_a_dead_pooled_connection(
    hass: HomeAssistant,
    failure: Callable[[httpx.Request], Exception],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One stale connection costs a retry, not the frame's face results."""
    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(
        transport=_transport([failure], {"faces": []}, seen)
    ) as client:
        with caplog.at_level(logging.WARNING):
            hits = await _analyzer(hass, client).recognize_faces(b"img", "camera.test")

    assert [hit.name for hit in hits] == ["Indeterminate"]
    assert len(seen) == 2
    assert all(str(r.url) == "http://face-api/analyze" for r in seen)
    assert "Face API request error" not in caplog.text


@pytest.mark.asyncio
async def test_recognize_faces_gives_up_after_one_retry(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """A face API that keeps dropping connections still degrades to []."""
    seen: list[httpx.Request] = []
    failures = [_remote_protocol, _remote_protocol, _remote_protocol]
    async with httpx.AsyncClient(
        transport=_transport(failures, {"faces": []}, seen)
    ) as client:
        with caplog.at_level(logging.WARNING):
            hits = await _analyzer(hass, client).recognize_faces(b"img", "camera.test")

    assert hits == []
    assert len(seen) == 2
    assert "Face API request error" in caplog.text


@pytest.mark.asyncio
async def test_recognize_faces_does_not_retry_a_refused_connection(
    hass: HomeAssistant,
) -> None:
    """A down face service is not a stale connection; fail fast as before."""
    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(
        transport=_transport([_connect_error], {"faces": []}, seen)
    ) as client:
        hits = await _analyzer(hass, client).recognize_faces(b"img", "camera.test")

    assert hits == []
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_recognize_faces_gives_each_attempt_the_face_timeout(
    hass: HomeAssistant,
) -> None:
    """
    Each attempt carries the 10 s face timeout, not the client default.

    The caller's own 10 s deadline still bounds both attempts together; a
    stale connection fails at once, so the retry has nearly all of it.
    """
    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(
        transport=_transport([_read_error], {"faces": []}, seen), timeout=99
    ) as client:
        await _analyzer(hass, client).recognize_faces(b"img", "camera.test")

    assert [r.extensions["timeout"]["read"] for r in seen] == [10, 10]


@pytest.mark.asyncio
async def test_enrollment_retries_a_dead_pooled_connection(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enrollment posts to the same /analyze and gets the same one retry."""
    seen: list[httpx.Request] = []
    added: list[str] = []
    payload: dict[str, object] = {"faces": [{"embedding": [0.1] * FACE_EMBEDDING_DIMS}]}
    dao = PersonGalleryDAO(cast("Any", object()), hass)

    async def _add_person(name: str, _embedding: object) -> None:
        added.append(name)

    monkeypatch.setattr(dao, "add_person", _add_person)
    async with httpx.AsyncClient(
        transport=_transport([_remote_protocol], payload, seen), timeout=7
    ) as client:
        dao._client = client
        assert await dao.enroll_from_image("http://face-api/", "Alice", b"img")

    assert added == ["Alice"]
    assert len(seen) == 2
    # Enrollment keeps the client's own timeout, as before.
    assert [r.extensions["timeout"]["read"] for r in seen] == [7, 7]


@pytest.mark.asyncio
async def test_face_service_url_check_retries_a_dead_pooled_connection(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saving the face URL must not report a running service as unreachable."""
    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(
        transport=_transport([_remote_protocol], {"status": "ok"}, seen)
    ) as client:
        monkeypatch.setattr(core_utils, "get_async_client", lambda _hass: client)
        await core_utils.validate_face_api_url(hass, "http://face-api")

    assert [str(r.url) for r in seen] == ["http://face-api/status"] * 2
