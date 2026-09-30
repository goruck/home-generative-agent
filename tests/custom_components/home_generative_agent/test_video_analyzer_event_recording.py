# ruff: noqa: S101, EM101, TRY003
"""
Tests for the ring-mqtt event-recording frame ingest (issue #491).

Covers the hass-free helpers in core/event_recording.py (URL reading, host
vetting, thinning, download with size cap / redirect refusal / not-ready
classification, real ffmpeg extraction and cancellation when a binary is
present) and the analyzer's scheduling, task bookkeeping, URL re-read per
attempt, retry, failure accounting, buffer replacement, window generations,
dedupe history reset, name-collision placement, orphan sweep, and shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import stat
import subprocess
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

import custom_components.home_generative_agent.core.event_recording as er_mod
import custom_components.home_generative_agent.core.video_analyzer as va_mod
from custom_components.home_generative_agent.config_flow import DEFAULT_OPTIONS
from custom_components.home_generative_agent.const import (
    CONF_VIDEO_ANALYZER_EVENT_RECORDING_ENABLED,
    CONF_VIDEO_ANALYZER_MOTION_CAMERA_MAP,
)
from custom_components.home_generative_agent.core.event_recording import (
    RecordingError,
    RecordingNotReadyError,
    assert_public_host,
    download_recording,
    extract_frames,
    ffmpeg_binary,
    is_valid_event_id,
    recording_url_from_state,
    thin_frames,
)
from custom_components.home_generative_agent.core.video_analyzer import (
    VideoAnalyzer,
    _is_analyzer_snapshot_name,
)
from custom_components.home_generative_agent.core.video_helpers import (
    epoch_from_path,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

URL = "https://download-eu.prod.phoenix.devices.amazon.dev/v1/download/abc.mp4?sig=1"
SELECT = "select.front_door_event_select"
CAMERA = "camera.front_door"
FFMPEG = shutil.which("ffmpeg")
STARTED = datetime(2026, 9, 28, 10, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def enable_event_loop_debug() -> None:
    """No-op override: pure-asyncio tests don't need HA's debug-mode hook."""


@pytest.fixture(autouse=True)
def verify_cleanup() -> None:
    """No-op override: all tasks are awaited explicitly below."""


@pytest.fixture(autouse=True)
def public_host() -> Any:
    """Skip DNS in analyzer-level tests; assert_public_host has its own tests."""
    with patch.object(va_mod, "assert_public_host", AsyncMock()) as m:
        yield m


@pytest.fixture
def hass() -> MagicMock:
    mock = MagicMock()
    mock.data = {}

    async def _run(func: Any, *args: Any) -> Any:
        return func(*args)

    mock.async_add_executor_job = AsyncMock(side_effect=_run)
    return mock


@pytest.fixture
def entry() -> MagicMock:
    e = MagicMock()
    e.runtime_data.options = {CONF_VIDEO_ANALYZER_EVENT_RECORDING_ENABLED: True}
    return e


@pytest.fixture
def va(hass: MagicMock, entry: MagicMock, tmp_path: Path) -> VideoAnalyzer:
    analyzer = VideoAnalyzer(hass, entry)
    analyzer._httpx_client = MagicMock()  # type: ignore[attr-defined]
    analyzer._get_snapshot_dir = AsyncMock(return_value=tmp_path)  # type: ignore[method-assign]
    analyzer._is_unique_enough = AsyncMock(return_value=True)  # type: ignore[method-assign]
    return analyzer


def _state(event_id: str | None, url: Any = URL) -> MagicMock:
    attrs: dict[str, Any] = {}
    if event_id is not None:
        attrs["eventId"] = event_id
    if url is not None:
        attrs["recordingUrl"] = url
    return MagicMock(attributes=attrs, state="event")


def _select_event(old: MagicMock | None, new: MagicMock | None) -> MagicMock:
    event = MagicMock()
    event.data = {"entity_id": SELECT, "old_state": old, "new_state": new}
    return event


def _stub_bg_task(va: VideoAnalyzer) -> MagicMock:
    def _consume(coro: Any, _name: str) -> MagicMock:
        coro.close()
        task = MagicMock()
        task.done.return_value = False
        return task

    stub = MagicMock(side_effect=_consume)
    va._create_background_task = stub  # type: ignore[method-assign]
    return stub


def _fake_extract(count: int) -> Any:
    """Return an extract_frames stand-in that writes `count` frames."""

    async def _extract(_binary: str, _clip: Path, out_dir: Path) -> list[Path]:
        frames = []
        for i in range(1, count + 1):
            f = out_dir / f"frame_{i:04d}.jpg"
            f.write_bytes(b"\xff\xd8" + bytes([i]))
            frames.append(f)
        return frames

    return _extract


async def _fake_download(_client: Any, _url: str, dest: Path) -> int:
    await asyncio.to_thread(dest.write_bytes, b"mp4")
    return 3


def _work_dirs(root: Path) -> list[Path]:
    return [p for p in root.iterdir() if p.name.startswith(er_mod.EVENT_WORK_PREFIX)]


async def _run_ingest(
    va: VideoAnalyzer,
    *,
    url: str = URL,
    event_id: str = "222",
    generation: int = 0,
    started: datetime = STARTED,
) -> None:
    await va._ingest_event_recording(  # type: ignore[attr-defined]
        CAMERA, SELECT, event_id, url, started, generation
    )


# ---------------------------------------------------------------------------
# Helpers: URL reading, event id shape, thinning, binary resolution
# ---------------------------------------------------------------------------


def test_recording_url_requires_https_string() -> None:
    assert recording_url_from_state(None) is None
    assert recording_url_from_state(_state("1", url=None)) is None
    assert recording_url_from_state(_state("1", url="")) is None
    assert recording_url_from_state(_state("1", url=42)) is None
    assert recording_url_from_state(_state("1", url="http://x/y.mp4?sig=1")) is None
    assert recording_url_from_state(_state("1", url=f"  {URL} ")) == URL


def test_event_id_shape() -> None:
    assert is_valid_event_id("7663194728557204991")
    assert is_valid_event_id("abc-DEF_1.2:3")
    assert not is_valid_event_id("")
    assert not is_valid_event_id("../../..")
    assert not is_valid_event_id("a/b")
    assert not is_valid_event_id("x\ny")
    assert not is_valid_event_id("9" * 129)


def test_thin_frames_keeps_all_when_under_cap() -> None:
    frames = ["a", "b", "c"]
    assert thin_frames(frames, 8) == [(0, "a"), (1, "b"), (2, "c")]


def test_thin_frames_spreads_evenly_first_and_last() -> None:
    frames = list(range(20))
    kept = thin_frames(frames, 8)
    indices = [i for i, _ in kept]
    assert len(kept) == 8
    assert indices[0] == 0
    assert indices[-1] == 19
    assert indices == sorted(set(indices))
    assert all(frames[i] == f for i, f in kept)


def test_thin_frames_edges() -> None:
    assert thin_frames([], 8) == []
    assert thin_frames([1, 2, 3], 0) == []
    assert thin_frames([1, 2, 3], 1) == [(0, 1)]


def test_ffmpeg_binary_prefers_ha_manager(hass: MagicMock) -> None:
    assert ffmpeg_binary(hass) == "ffmpeg"
    hass.data["ffmpeg"] = MagicMock(binary="/usr/local/bin/ffmpeg")
    assert ffmpeg_binary(hass) == "/usr/local/bin/ffmpeg"
    hass.data["ffmpeg"] = MagicMock(binary="")
    assert ffmpeg_binary(hass) == "ffmpeg"


# ---------------------------------------------------------------------------
# assert_public_host
# ---------------------------------------------------------------------------


def _addrinfo(*ips: str) -> list[tuple[Any, ...]]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443)) for ip in ips]


async def testpublic_host_rejects_address_literals() -> None:
    with pytest.raises(RecordingError, match="address literal"):
        await assert_public_host("https://192.168.1.1/x.mp4")
    with pytest.raises(RecordingError, match="address literal"):
        await assert_public_host("https://[::1]/x.mp4")
    with pytest.raises(RecordingError, match="no host"):
        await assert_public_host("https:///x.mp4")


@pytest.mark.parametrize(
    "ip",
    ["10.1.2.3", "192.168.1.10", "172.16.0.5", "127.0.0.1", "169.254.1.1", "0.0.0.0"],  # noqa: S104,
)
async def testpublic_host_rejects_local_resolution(ip: str) -> None:
    loop = asyncio.get_running_loop()
    with (
        patch.object(loop, "getaddrinfo", AsyncMock(return_value=_addrinfo(ip))),
        pytest.raises(RecordingError, match="non-public"),
    ):
        await assert_public_host(URL)


async def testpublic_host_rejects_mixed_resolution() -> None:
    loop = asyncio.get_running_loop()
    with (
        patch.object(
            loop,
            "getaddrinfo",
            AsyncMock(return_value=_addrinfo("52.1.2.3", "10.0.0.1")),
        ),
        pytest.raises(RecordingError, match="non-public"),
    ):
        await assert_public_host(URL)


async def testpublic_host_accepts_public_resolution() -> None:
    loop = asyncio.get_running_loop()
    with patch.object(
        loop, "getaddrinfo", AsyncMock(return_value=_addrinfo("52.1.2.3"))
    ):
        await assert_public_host(URL)


async def testpublic_host_dns_failure_is_retryable() -> None:
    loop = asyncio.get_running_loop()
    with (
        patch.object(loop, "getaddrinfo", AsyncMock(side_effect=socket.gaierror("nx"))),
        pytest.raises(RecordingNotReadyError, match="resolve"),
    ):
        await assert_public_host(URL)


# ---------------------------------------------------------------------------
# download_recording
# ---------------------------------------------------------------------------


@respx.mock
async def test_download_writes_body_and_returns_size(tmp_path: Path) -> None:
    respx.get(URL).mock(return_value=httpx.Response(200, content=b"x" * 1000))
    dest = tmp_path / "clip.mp4"
    async with httpx.AsyncClient() as client:
        size = await download_recording(client, URL, dest)
    assert size == 1000
    assert dest.read_bytes() == b"x" * 1000


@respx.mock
@pytest.mark.parametrize("status", [403, 404, 429])
async def test_download_not_ready_statuses_are_retryable(
    tmp_path: Path, status: int
) -> None:
    respx.get(URL).mock(return_value=httpx.Response(status))
    async with httpx.AsyncClient() as client:
        with pytest.raises(RecordingNotReadyError):
            await download_recording(client, URL, tmp_path / "c.mp4")


@respx.mock
async def test_download_server_error_is_final(tmp_path: Path) -> None:
    respx.get(URL).mock(return_value=httpx.Response(500))
    async with httpx.AsyncClient() as client:
        with pytest.raises(RecordingError) as excinfo:
            await download_recording(client, URL, tmp_path / "c.mp4")
    assert not isinstance(excinfo.value, RecordingNotReadyError)


EDGE = URL.replace("https://download-eu", "https://51-49-199-144.download-eu")


@respx.mock
async def test_download_follows_same_host_subdomain_redirect(tmp_path: Path) -> None:
    """Ring's transcoding URL 302s to <edge-node>.<same host>: followed and vetted."""
    respx.get(URL).mock(return_value=httpx.Response(302, headers={"location": EDGE}))
    respx.get(EDGE).mock(return_value=httpx.Response(206, content=b"ftyp" * 10))
    dest = tmp_path / "clip.mp4"
    with patch.object(er_mod, "assert_public_host", AsyncMock()) as vet:
        async with httpx.AsyncClient() as client:
            size = await download_recording(client, URL, dest)
    assert size == 40
    assert dest.read_bytes() == b"ftyp" * 10
    vet.assert_awaited_once_with(EDGE)


@respx.mock
async def test_download_follows_relative_redirect(tmp_path: Path) -> None:
    respx.get(URL).mock(
        return_value=httpx.Response(302, headers={"location": "/v1/download/final.mp4"})
    )
    final = "https://download-eu.prod.phoenix.devices.amazon.dev/v1/download/final.mp4"
    respx.get(final).mock(return_value=httpx.Response(200, content=b"x" * 5))
    with patch.object(er_mod, "assert_public_host", AsyncMock()):
        async with httpx.AsyncClient() as client:
            assert await download_recording(client, URL, tmp_path / "c.mp4") == 5


@respx.mock
@pytest.mark.parametrize(
    ("location", "match"),
    [
        ("https://evil.example.net/x.mp4", "leaves the host"),
        ("https://amazon.dev.evil.example.net/x.mp4", "leaves the host"),
        ("https://prod.phoenix.devices.amazon.dev/x.mp4", "leaves the host"),
        (
            "http://51-49-199-144.download-eu.prod.phoenix.devices.amazon.dev/x",
            "non-https",
        ),
        ("", "without a Location"),
    ],
)
async def test_download_refuses_off_rule_redirects(
    tmp_path: Path, location: str, match: str
) -> None:
    headers = {"location": location} if location else {}
    respx.get(URL).mock(return_value=httpx.Response(302, headers=headers))
    other = respx.route().mock(return_value=httpx.Response(200, content=b"x"))
    with patch.object(er_mod, "assert_public_host", AsyncMock()):
        async with httpx.AsyncClient() as client:
            with pytest.raises(RecordingError, match=match):
                await download_recording(client, URL, tmp_path / "c.mp4")
    assert not other.called


@respx.mock
async def test_download_redirect_hop_budget(tmp_path: Path) -> None:
    respx.get(URL).mock(return_value=httpx.Response(302, headers={"location": EDGE}))
    respx.get(EDGE).mock(return_value=httpx.Response(302, headers={"location": EDGE}))
    with patch.object(er_mod, "assert_public_host", AsyncMock()) as vet:
        async with httpx.AsyncClient() as client:
            with pytest.raises(RecordingError, match="redirects"):
                await download_recording(client, URL, tmp_path / "c.mp4")
    assert vet.await_count == er_mod.RECORDING_MAX_REDIRECTS


@respx.mock
async def test_download_redirect_target_must_resolve_public(tmp_path: Path) -> None:
    respx.get(URL).mock(return_value=httpx.Response(302, headers={"location": EDGE}))
    edge = respx.get(EDGE).mock(return_value=httpx.Response(200, content=b"x"))
    with patch.object(
        er_mod,
        "assert_public_host",
        AsyncMock(side_effect=RecordingError("non-public")),
    ):
        async with httpx.AsyncClient() as client:
            with pytest.raises(RecordingError, match="non-public"):
                await download_recording(client, URL, tmp_path / "c.mp4")
    assert not edge.called


@respx.mock
async def test_download_rejects_declared_oversize(tmp_path: Path) -> None:
    respx.get(URL).mock(
        return_value=httpx.Response(200, content=b"ab", headers={"content-length": "2"})
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(RecordingError, match="limit"):
            await download_recording(client, URL, tmp_path / "c.mp4", max_bytes=1)


@respx.mock
async def test_download_rejects_streamed_oversize(tmp_path: Path) -> None:
    respx.get(URL).mock(
        return_value=httpx.Response(200, stream=httpx.ByteStream(b"x" * 50))
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(RecordingError, match="exceeds"):
            await download_recording(client, URL, tmp_path / "c.mp4", max_bytes=10)


@respx.mock
async def test_download_empty_body_is_not_ready(tmp_path: Path) -> None:
    respx.get(URL).mock(return_value=httpx.Response(200, content=b""))
    async with httpx.AsyncClient() as client:
        with pytest.raises(RecordingNotReadyError):
            await download_recording(client, URL, tmp_path / "c.mp4")


@respx.mock
async def test_download_transport_error_is_not_ready(tmp_path: Path) -> None:
    respx.get(URL).mock(side_effect=httpx.ConnectError("boom"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(RecordingNotReadyError):
            await download_recording(client, URL, tmp_path / "c.mp4")


async def test_download_malformed_url_is_final(tmp_path: Path) -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(RecordingError, match="malformed"):
            await download_recording(client, "https://x\x00y/", tmp_path / "c")


class _Trickle(httpx.AsyncByteStream):
    """One byte every 50 ms, forever."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            await asyncio.sleep(0.05)
            yield b"x"


async def test_download_wall_clock_bound(tmp_path: Path) -> None:
    """A trickling server is cut off by the total deadline, not per read."""

    async def _trickle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_Trickle())

    with (
        respx.mock,
        patch.object(er_mod, "RECORDING_DOWNLOAD_TIMEOUT_SEC", 0.3),
    ):
        respx.get(URL).mock(side_effect=_trickle)
        async with httpx.AsyncClient() as client:
            with pytest.raises(RecordingError, match="exceeded"):
                await download_recording(client, URL, tmp_path / "c.mp4")


# ---------------------------------------------------------------------------
# extract_frames (real ffmpeg when available)
# ---------------------------------------------------------------------------


def _make_clip(path: Path, seconds: int) -> None:
    assert FFMPEG is not None
    subprocess.run(  # noqa: S603
        [
            FFMPEG,
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={seconds}:size=1600x900:rate=5",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg binary not installed")
async def test_extract_frames_real_ffmpeg_one_per_second_scaled(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    clip = tmp_path / "clip.mp4"
    _make_clip(clip, seconds=5)
    out = tmp_path / "frames"
    out.mkdir()
    frames = await extract_frames(FFMPEG or "ffmpeg", clip, out)
    # fps=1 over a 5 s clip yields 5 frames (ffmpeg may emit one extra at the
    # boundary depending on build); every one must be a JPEG no wider than
    # the cap (source is 1600 px wide).
    assert 5 <= len(frames) <= 6
    assert frames == sorted(frames)
    for f in frames:
        with Image.open(f) as img:
            assert img.width == er_mod.RECORDING_MAX_WIDTH


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg binary not installed")
async def test_extract_frames_garbage_input_raises(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not a video")
    out = tmp_path / "frames"
    out.mkdir()
    with pytest.raises(RecordingError, match="exited"):
        await extract_frames(FFMPEG or "ffmpeg", clip, out)


async def test_extract_frames_missing_binary_raises(tmp_path: Path) -> None:
    with pytest.raises(RecordingError, match="could not start"):
        await extract_frames("/nonexistent/ffmpeg-hga", tmp_path / "x.mp4", tmp_path)


def _fake_ffmpeg(tmp_path: Path) -> Path:
    """Return a stand-in binary that records its pid then sleeps."""
    script = tmp_path / "fake-ffmpeg"
    script.write_text(f"#!/bin/sh\necho $$ > {tmp_path / 'pid'}\nsleep 30\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_extract_frames_cancellation_kills_child(tmp_path: Path) -> None:
    script = _fake_ffmpeg(tmp_path)
    task = asyncio.create_task(
        extract_frames(str(script), tmp_path / "x.mp4", tmp_path)
    )
    pid_file = tmp_path / "pid"
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        await asyncio.sleep(0.02)
    pid = int(pid_file.read_text().strip())
    assert _alive(pid)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(50):
        if not _alive(pid):
            break
        await asyncio.sleep(0.02)
    assert not _alive(pid)


async def test_extract_frames_timeout_kills_child(tmp_path: Path) -> None:
    script = _fake_ffmpeg(tmp_path)
    with (
        patch.object(er_mod, "RECORDING_FFMPEG_TIMEOUT_SEC", 0.2),
        pytest.raises(RecordingError, match="did not finish"),
    ):
        await extract_frames(str(script), tmp_path / "x.mp4", tmp_path)
    pid = int((tmp_path / "pid").read_text().strip())
    assert not _alive(pid)


# ---------------------------------------------------------------------------
# Filename contract
# ---------------------------------------------------------------------------


def test_recording_frame_names_are_analyzer_owned() -> None:
    assert _is_analyzer_snapshot_name("snapshot_20260928_101010.jpg")
    assert _is_analyzer_snapshot_name("snapshot_20260928_101010_r07.jpg")
    assert not _is_analyzer_snapshot_name("snapshot_20260928_101010_r7.jpg")
    assert not _is_analyzer_snapshot_name("snapshot_20260928_101010_r123.jpg")
    assert not _is_analyzer_snapshot_name("snapshot_20260928_101010_x07.jpg")


def test_recording_frame_epoch_ignores_suffix() -> None:
    plain = epoch_from_path(Path("snapshot_20260928_101010.jpg"))
    suffixed = epoch_from_path(Path("snapshot_20260928_101010_r07.jpg"))
    assert plain == suffixed


def test_option_defaults_off() -> None:
    assert DEFAULT_OPTIONS[CONF_VIDEO_ANALYZER_EVENT_RECORDING_ENABLED] is False


# ---------------------------------------------------------------------------
# Scheduling: _maybe_ingest_event_recording via the event_select handler
# ---------------------------------------------------------------------------


def _wire_handler(va: VideoAnalyzer, hass: MagicMock, new_state: MagicMock) -> None:
    """Make the select resolve to CAMERA via the override map (no registry)."""
    va.entry.runtime_data.options[CONF_VIDEO_ANALYZER_MOTION_CAMERA_MAP] = {
        SELECT: CAMERA
    }
    hass.states.get.side_effect = lambda eid: (
        MagicMock() if eid == CAMERA else new_state if eid == SELECT else None
    )


def test_handler_schedules_ingest_when_enabled(
    va: VideoAnalyzer, hass: MagicMock
) -> None:
    new = _state("222")
    _wire_handler(va, hass, new)
    stub = _stub_bg_task(va)
    with patch.object(va_mod, "async_call_later"):
        va._handle_event_select_change(_select_event(_state("111"), new))  # type: ignore[attr-defined]
    names = [c.args[1] for c in stub.call_args_list]
    assert any("event_select snapshot loop" in n for n in names)
    assert f"hga video event recording {CAMERA}:222" in names
    assert f"{CAMERA}:222" in va._event_recording_tasks  # type: ignore[attr-defined]


def test_handler_skips_ingest_when_disabled(
    va: VideoAnalyzer, hass: MagicMock, entry: MagicMock
) -> None:
    entry.runtime_data.options[CONF_VIDEO_ANALYZER_EVENT_RECORDING_ENABLED] = False
    new = _state("222")
    _wire_handler(va, hass, new)
    stub = _stub_bg_task(va)
    with patch.object(va_mod, "async_call_later"):
        va._handle_event_select_change(_select_event(_state("111"), new))  # type: ignore[attr-defined]
    names = [c.args[1] for c in stub.call_args_list]
    assert not any("event recording" in n for n in names)


def test_no_recording_url_degrades_silently(
    va: VideoAnalyzer, hass: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    new = _state("222", url=None)
    _wire_handler(va, hass, new)
    stub = _stub_bg_task(va)
    with (
        patch.object(va_mod, "async_call_later"),
        caplog.at_level(logging.DEBUG, logger=va_mod.LOGGER.name),
    ):
        va._handle_event_select_change(_select_event(_state("111"), new))  # type: ignore[attr-defined]
    names = [c.args[1] for c in stub.call_args_list]
    assert not any("event recording" in n for n in names)
    assert "Ring Protect" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_hostile_event_id_is_ignored_and_not_logged_raw(
    va: VideoAnalyzer, caplog: pytest.LogCaptureFixture
) -> None:
    stub = _stub_bg_task(va)
    hostile = "../../../../.."
    with caplog.at_level(logging.DEBUG, logger=va_mod.LOGGER.name):
        va._maybe_ingest_event_recording(CAMERA, SELECT, _state(hostile), hostile)  # type: ignore[attr-defined]
    assert stub.call_count == 0
    assert hostile not in caplog.text


def test_same_event_id_is_ingested_once(va: VideoAnalyzer) -> None:
    stub = _stub_bg_task(va)
    va._maybe_ingest_event_recording(CAMERA, SELECT, _state("5"), "5")  # type: ignore[attr-defined]
    va._maybe_ingest_event_recording(CAMERA, SELECT, _state("5"), "5")  # type: ignore[attr-defined]
    va._maybe_ingest_event_recording(CAMERA, SELECT, _state("6"), "6")  # type: ignore[attr-defined]
    assert stub.call_count == 2


def test_a_b_a_does_not_double_book_the_task_key(va: VideoAnalyzer) -> None:
    """A re-selected older event neither re-fetches nor clobbers bookkeeping."""
    stub = _stub_bg_task(va)
    for eid in ("A", "B", "A"):
        va._maybe_ingest_event_recording(CAMERA, SELECT, _state(eid), eid)  # type: ignore[attr-defined]
    assert stub.call_count == 2
    tasks = va._event_recording_tasks  # type: ignore[attr-defined]
    assert set(tasks) == {f"{CAMERA}:A", f"{CAMERA}:B"}
    first_a = tasks[f"{CAMERA}:A"]
    # A finished task only removes its own slot, never a newer task's.
    va._on_event_recording_done(f"{CAMERA}:A", MagicMock())  # type: ignore[attr-defined]
    assert tasks[f"{CAMERA}:A"] is first_a
    va._on_event_recording_done(f"{CAMERA}:A", first_a)  # type: ignore[attr-defined]
    assert f"{CAMERA}:A" not in tasks


def test_pending_cap_drops_new_events_as_counted_failures(
    va: VideoAnalyzer, caplog: pytest.LogCaptureFixture
) -> None:
    stub = _stub_bg_task(va)
    cap = er_mod.RECORDING_MAX_PENDING_PER_CAMERA
    with caplog.at_level(logging.WARNING, logger=va_mod.LOGGER.name):
        for i in range(cap + 2):
            va._maybe_ingest_event_recording(CAMERA, SELECT, _state(str(i)), str(i))  # type: ignore[attr-defined]
    assert stub.call_count == cap
    assert va._metrics[CAMERA].recording_failures == 2  # type: ignore[attr-defined]
    assert "in flight" in caplog.text


# ---------------------------------------------------------------------------
# ffmpeg resolution inside the task
# ---------------------------------------------------------------------------


async def test_missing_ffmpeg_warns_once_and_falls_back(
    va: VideoAnalyzer, caplog: pytest.LogCaptureFixture
) -> None:
    download = AsyncMock()
    with (
        patch.object(va_mod, "ffmpeg_available", return_value=False),
        patch.object(va_mod, "download_recording", download),
        caplog.at_level(logging.WARNING, logger=va_mod.LOGGER.name),
    ):
        await _run_ingest(va, event_id="5")
        await _run_ingest(va, event_id="6")
    warnings = [r for r in caplog.records if "ffmpeg" in r.getMessage()]
    assert len(warnings) == 1
    assert download.await_count == 0


async def test_ffmpeg_probe_cached_and_reprobed_hourly(va: VideoAnalyzer) -> None:
    with patch.object(va_mod, "ffmpeg_available", return_value=True) as avail:
        assert await va._resolve_ffmpeg() == "ffmpeg"  # type: ignore[attr-defined]
        assert await va._resolve_ffmpeg() == "ffmpeg"  # type: ignore[attr-defined]
        assert avail.call_count == 1
        ok, at = va._event_recording_binary_ok["ffmpeg"]  # type: ignore[attr-defined]
        va._event_recording_binary_ok["ffmpeg"] = (  # type: ignore[attr-defined]
            ok,
            at - va_mod._STALE_REREPORT_INTERVAL_SEC - 1,
        )
        assert await va._resolve_ffmpeg() == "ffmpeg"  # type: ignore[attr-defined]
        assert avail.call_count == 2


# ---------------------------------------------------------------------------
# Ingest: download → extract → stamp → admit
# ---------------------------------------------------------------------------


@pytest.fixture
def ffmpeg_present() -> Any:
    with patch.object(va_mod, "ffmpeg_available", return_value=True):
        yield


@pytest.mark.usefixtures("ffmpeg_present")
async def test_ingest_replaces_held_frames_with_clip_frames(
    va: VideoAnalyzer, hass: MagicMock, tmp_path: Path
) -> None:
    hass.states.get.return_value = _state("222")
    va._active_motion_cameras[CAMERA] = MagicMock()  # type: ignore[attr-defined]
    stale = tmp_path / "snapshot_20260928_095000.jpg"
    stale.write_bytes(b"old")
    va._event_snapshot_buffers[CAMERA] = deque([stale])  # type: ignore[attr-defined]

    with (
        patch.object(va_mod, "download_recording", _fake_download),
        patch.object(va_mod, "extract_frames", _fake_extract(12)),
    ):
        await _run_ingest(va)

    buffer = list(va._event_snapshot_buffers[CAMERA])  # type: ignore[attr-defined]
    assert stale not in buffer
    assert len(buffer) == er_mod.RECORDING_MAX_FRAMES
    assert all(_is_analyzer_snapshot_name(p.name) for p in buffer)
    assert all(p.exists() for p in buffer)
    epochs = [epoch_from_path(p) for p in buffer]
    assert epochs == sorted(epochs)
    # A 12 s clip ended when the eventId flipped: frame 0 is stamped 12 s
    # before the trigger, the last frame (second 11) one second before it.
    assert epochs[0] == int(STARTED.timestamp()) - 12
    assert epochs[-1] == int(STARTED.timestamp()) - 1
    assert buffer[0].name.endswith("_r00.jpg")
    assert buffer[-1].name.endswith("_r11.jpg")
    # Work dir is gone, frames are in retention, metrics counted.
    assert _work_dirs(tmp_path) == []
    retention = va._retention_deques[CAMERA]  # type: ignore[attr-defined]
    assert all(p in retention for p in buffer)
    metrics = va._metrics[CAMERA]  # type: ignore[attr-defined]
    assert metrics.recording_frames == er_mod.RECORDING_MAX_FRAMES
    assert metrics.replaced_by_recording == 1
    assert metrics.recording_failures == 0


@pytest.mark.usefixtures("ffmpeg_present")
async def test_clip_frames_sort_before_later_snapshots(
    va: VideoAnalyzer, hass: MagicMock, tmp_path: Path
) -> None:
    """Snapshots the loop captures after the trigger order after the clip."""
    hass.states.get.return_value = _state("222")
    va._active_motion_cameras[CAMERA] = MagicMock()  # type: ignore[attr-defined]
    with (
        patch.object(va_mod, "download_recording", _fake_download),
        patch.object(va_mod, "extract_frames", _fake_extract(30)),
    ):
        await _run_ingest(va)
    later = tmp_path / "snapshot_20260928_100003.jpg"  # trigger + 3 s
    buffer = va._event_snapshot_buffers[CAMERA]  # type: ignore[attr-defined]
    buffer.append(later)
    ordered = va_mod.order_batch(list(buffer))
    assert ordered[-1][0] == later
    assert all(e <= int(STARTED.timestamp()) for _, e in ordered[:-1])


@pytest.mark.usefixtures("ffmpeg_present")
async def test_second_clip_keeps_earlier_recording_frames(
    va: VideoAnalyzer, hass: MagicMock, tmp_path: Path
) -> None:
    """Two events inside one window: clip 2 drops snapshots, not clip 1."""
    hass.states.get.return_value = _state("222")
    va._active_motion_cameras[CAMERA] = MagicMock()  # type: ignore[attr-defined]
    snapshot = tmp_path / "snapshot_20260928_100003.jpg"
    snapshot.write_bytes(b"s")

    with (
        patch.object(va_mod, "download_recording", _fake_download),
        patch.object(va_mod, "extract_frames", _fake_extract(2)),
    ):
        await _run_ingest(va, event_id="222")
        first = list(va._event_snapshot_buffers[CAMERA])  # type: ignore[attr-defined]
        va._event_snapshot_buffers[CAMERA].append(snapshot)  # type: ignore[attr-defined]
        await _run_ingest(va, event_id="223", started=STARTED + timedelta(seconds=5))

    buffer = list(va._event_snapshot_buffers[CAMERA])  # type: ignore[attr-defined]
    assert len(first) == 2
    assert buffer[:2] == first
    assert snapshot not in buffer
    assert len(buffer) == 4
    assert va._metrics[CAMERA].replaced_by_recording == 1  # type: ignore[attr-defined]


@pytest.mark.usefixtures("ffmpeg_present")
async def test_same_second_events_never_overwrite_frames(
    va: VideoAnalyzer, hass: MagicMock
) -> None:
    hass.states.get.return_value = _state("222")
    va._active_motion_cameras[CAMERA] = MagicMock()  # type: ignore[attr-defined]
    with (
        patch.object(va_mod, "download_recording", _fake_download),
        patch.object(va_mod, "extract_frames", _fake_extract(2)),
    ):
        await _run_ingest(va, event_id="222")
        await _run_ingest(va, event_id="223")
    buffer = list(va._event_snapshot_buffers[CAMERA])  # type: ignore[attr-defined]
    assert len(buffer) == 4
    assert len({p.name for p in buffer}) == 4
    assert all(p.exists() for p in buffer)
    # Second clip's frames were bumped by one second, suffix intact.
    assert buffer[2].name.endswith("_r00.jpg")
    assert epoch_from_path(buffer[2]) == epoch_from_path(buffer[0]) + 1


@pytest.mark.usefixtures("ffmpeg_present")
async def test_ingest_after_window_flush_is_its_own_batch(
    va: VideoAnalyzer, hass: MagicMock
) -> None:
    """A late clip is analyzed as one ordered batch, never via the live queue."""
    hass.states.get.return_value = _state("222")
    analyze = AsyncMock()
    va._analyze_and_finalize = analyze  # type: ignore[method-assign]
    va._get_snapshot_queue = MagicMock(side_effect=AssertionError("live queue used"))  # type: ignore[method-assign]
    created: list[Any] = []

    def _run(coro: Any, _name: str) -> MagicMock:
        created.append(asyncio.ensure_future(coro))
        return MagicMock()

    va._create_background_task = MagicMock(side_effect=_run)  # type: ignore[method-assign]
    with (
        patch.object(va_mod, "download_recording", _fake_download),
        patch.object(va_mod, "extract_frames", _fake_extract(8)),
    ):
        await _run_ingest(va)
    await asyncio.gather(*created)
    assert analyze.await_count == 1
    assert analyze.await_args is not None
    camera, ordered = analyze.await_args.args
    assert camera == CAMERA
    assert len(ordered) == 8
    assert [e for _, e in ordered] == sorted(e for _, e in ordered)
    assert CAMERA not in va._event_snapshot_buffers  # type: ignore[attr-defined]


@pytest.mark.usefixtures("ffmpeg_present")
async def test_stale_generation_does_not_touch_a_later_window(
    va: VideoAnalyzer, hass: MagicMock, tmp_path: Path
) -> None:
    """Event A's clip landing during event B's window leaves B's frames alone."""
    hass.states.get.return_value = _state("A")
    va._active_motion_cameras[CAMERA] = MagicMock()  # type: ignore[attr-defined]
    b_frame = tmp_path / "snapshot_20260928_100100.jpg"
    va._event_snapshot_buffers[CAMERA] = deque([b_frame])  # type: ignore[attr-defined]
    va._window_generation[CAMERA] = 3  # type: ignore[attr-defined]
    analyze = AsyncMock()
    va._analyze_and_finalize = analyze  # type: ignore[method-assign]
    created: list[Any] = []
    va._create_background_task = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda coro, _n: created.append(asyncio.ensure_future(coro))
    )
    with (
        patch.object(va_mod, "download_recording", _fake_download),
        patch.object(va_mod, "extract_frames", _fake_extract(2)),
    ):
        await _run_ingest(va, event_id="A", generation=2)
    await asyncio.gather(*created)
    assert list(va._event_snapshot_buffers[CAMERA]) == [b_frame]  # type: ignore[attr-defined]
    assert analyze.await_count == 1
    assert va._metrics[CAMERA].replaced_by_recording == 0  # type: ignore[attr-defined]


def test_window_end_bumps_generation(va: VideoAnalyzer) -> None:
    _stub_bg_task(va)
    assert va._window_generation.get(CAMERA, 0) == 0  # type: ignore[attr-defined]
    va._stop_motion_loop_and_flush(CAMERA)  # type: ignore[attr-defined]
    assert va._window_generation[CAMERA] == 1  # type: ignore[attr-defined]


@pytest.mark.usefixtures("ffmpeg_present")
async def test_ingest_dedupe_keeps_at_least_first_frame(
    va: VideoAnalyzer, hass: MagicMock
) -> None:
    hass.states.get.return_value = _state("222")
    va._active_motion_cameras[CAMERA] = MagicMock()  # type: ignore[attr-defined]
    va._is_unique_enough = AsyncMock(return_value=False)  # type: ignore[method-assign]

    with (
        patch.object(va_mod, "download_recording", _fake_download),
        patch.object(va_mod, "extract_frames", _fake_extract(4)),
    ):
        await _run_ingest(va)

    buffer = list(va._event_snapshot_buffers[CAMERA])  # type: ignore[attr-defined]
    assert len(buffer) == 1
    assert buffer[0].name.endswith("_r00.jpg")
    assert va._metrics[CAMERA].skipped_duplicate == 4  # type: ignore[attr-defined]


@pytest.mark.usefixtures("ffmpeg_present")
async def test_ingest_resets_hash_history_before_gating(
    va: VideoAnalyzer, hass: MagicMock
) -> None:
    """Clip frames dedupe among themselves, not against displaced snapshots."""
    hass.states.get.return_value = _state("222")
    va._active_motion_cameras[CAMERA] = MagicMock()  # type: ignore[attr-defined]
    va._last_hashes[CAMERA] = deque([0xDEADBEEF], maxlen=2)  # type: ignore[attr-defined]
    seen_history: list[bool] = []

    async def _gate(camera_id: str, _path: Path) -> bool:
        seen_history.append(camera_id in va._last_hashes)  # type: ignore[attr-defined]
        return True

    va._is_unique_enough = _gate  # type: ignore[method-assign]
    with (
        patch.object(va_mod, "download_recording", _fake_download),
        patch.object(va_mod, "extract_frames", _fake_extract(2)),
    ):
        await _run_ingest(va)
    assert seen_history == [False, False]


@pytest.mark.usefixtures("ffmpeg_present")
async def test_ingest_rereads_url_each_attempt_while_event_current(
    va: VideoAnalyzer, hass: MagicMock
) -> None:
    urls = iter([URL.replace("sig=1", "sig=2"), URL.replace("sig=1", "sig=3")])
    hass.states.get.side_effect = lambda _eid: _state("222", url=next(urls))
    va._create_background_task = MagicMock(side_effect=lambda c, _n: c.close())  # type: ignore[method-assign]
    seen: list[str] = []

    async def _dl(_c: Any, url: str, dest: Path) -> int:
        seen.append(url)
        if len(seen) == 1:
            raise RecordingNotReadyError("HTTP 404")
        return await _fake_download(_c, url, dest)

    with (
        patch.object(va_mod, "RECORDING_FETCH_BACKOFF_SEC", (0.0, 0.0, 0.0)),
        patch.object(va_mod, "download_recording", _dl),
        patch.object(va_mod, "extract_frames", _fake_extract(1)),
    ):
        await _run_ingest(va, url=URL)
    assert seen == [URL.replace("sig=1", "sig=2"), URL.replace("sig=1", "sig=3")]


@pytest.mark.usefixtures("ffmpeg_present")
async def test_ingest_keeps_trigger_url_when_entity_moved_on(
    va: VideoAnalyzer, hass: MagicMock, public_host: AsyncMock
) -> None:
    hass.states.get.return_value = _state("333", url=URL.replace("sig=1", "sig=3"))
    va._create_background_task = MagicMock(side_effect=lambda c, _n: c.close())  # type: ignore[method-assign]
    seen: list[str] = []

    async def _dl(_c: Any, url: str, dest: Path) -> int:
        seen.append(url)
        return await _fake_download(_c, url, dest)

    with (
        patch.object(va_mod, "download_recording", _dl),
        patch.object(va_mod, "extract_frames", _fake_extract(1)),
    ):
        await _run_ingest(va, url=URL, event_id="222")
    assert seen == [URL]
    public_host.assert_awaited_once_with(URL)


@pytest.mark.usefixtures("ffmpeg_present")
async def test_ingest_retries_not_ready_then_succeeds(
    va: VideoAnalyzer, hass: MagicMock
) -> None:
    hass.states.get.return_value = _state("222")
    va._create_background_task = MagicMock(side_effect=lambda c, _n: c.close())  # type: ignore[method-assign]
    calls = {"n": 0}

    async def _dl(_c: Any, url: str, dest: Path) -> int:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RecordingNotReadyError("HTTP 404")
        return await _fake_download(_c, url, dest)

    with (
        patch.object(va_mod, "RECORDING_FETCH_BACKOFF_SEC", (0.0, 0.0, 0.0, 0.0)),
        patch.object(va_mod, "download_recording", _dl),
        patch.object(va_mod, "extract_frames", _fake_extract(2)),
    ):
        await _run_ingest(va)
    assert calls["n"] == 3
    assert va._metrics[CAMERA].recording_frames == 2  # type: ignore[attr-defined]
    assert va._metrics[CAMERA].recording_failures == 0  # type: ignore[attr-defined]


@pytest.mark.usefixtures("ffmpeg_present")
async def test_ingest_gives_up_after_backoff_and_warns_hourly(
    va: VideoAnalyzer,
    hass: MagicMock,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    hass.states.get.return_value = _state("222")
    va._active_motion_cameras[CAMERA] = MagicMock()  # type: ignore[attr-defined]
    held = tmp_path / "snapshot_20260928_095000.jpg"
    va._event_snapshot_buffers[CAMERA] = deque([held])  # type: ignore[attr-defined]

    async def _dl(_c: Any, _url: str, _dest: Path) -> int:
        raise RecordingNotReadyError("HTTP 404")

    with (
        patch.object(va_mod, "RECORDING_FETCH_BACKOFF_SEC", (0.0, 0.0)),
        patch.object(va_mod, "download_recording", _dl),
        caplog.at_level(logging.DEBUG, logger=va_mod.LOGGER.name),
    ):
        await _run_ingest(va, event_id="222")
        await _run_ingest(va, event_id="223")

    # Held snapshot frames stand in untouched; nothing leaked on disk.
    assert list(va._event_snapshot_buffers[CAMERA]) == [held]  # type: ignore[attr-defined]
    assert _work_dirs(tmp_path) == []
    assert va._metrics[CAMERA].recording_failures == 2  # type: ignore[attr-defined]
    failures = [r for r in caplog.records if "could not be analyzed" in r.getMessage()]
    assert [r.levelno for r in failures] == [logging.WARNING, logging.DEBUG]
    assert "never became fetchable" in failures[0].getMessage()


@pytest.mark.usefixtures("ffmpeg_present")
async def test_ingest_host_rejection_is_final(
    va: VideoAnalyzer, hass: MagicMock, public_host: AsyncMock
) -> None:
    hass.states.get.return_value = _state("222")
    public_host.side_effect = RecordingError("resolves to a non-public address")
    download = AsyncMock()
    with (
        patch.object(va_mod, "RECORDING_FETCH_BACKOFF_SEC", (0.0, 0.0, 0.0)),
        patch.object(va_mod, "download_recording", download),
    ):
        await _run_ingest(va)
    assert download.await_count == 0
    assert public_host.await_count == 1
    assert va._metrics[CAMERA].recording_failures == 1  # type: ignore[attr-defined]


@pytest.mark.usefixtures("ffmpeg_present")
async def test_ingest_extract_failure_is_counted(
    va: VideoAnalyzer, hass: MagicMock, tmp_path: Path
) -> None:
    hass.states.get.return_value = _state("222")

    async def _boom(_b: str, _clip: Path, _out: Path) -> list[Path]:
        raise RecordingError("ffmpeg exited 1: moov atom not found")

    with (
        patch.object(va_mod, "download_recording", _fake_download),
        patch.object(va_mod, "extract_frames", _boom),
    ):
        await _run_ingest(va)
    assert va._metrics[CAMERA].recording_failures == 1  # type: ignore[attr-defined]
    assert va._metrics[CAMERA].recording_frames == 0  # type: ignore[attr-defined]
    assert _work_dirs(tmp_path) == []
    assert CAMERA not in va._event_snapshot_buffers  # type: ignore[attr-defined]


@pytest.mark.usefixtures("ffmpeg_present")
async def test_ingest_deadline_is_counted(
    va: VideoAnalyzer, hass: MagicMock, tmp_path: Path
) -> None:
    hass.states.get.return_value = _state("222")

    async def _slow(_c: Any, _url: str, _dest: Path) -> int:
        await asyncio.sleep(5)
        return 1

    with (
        patch.object(va_mod, "RECORDING_INGEST_DEADLINE_SEC", 0.1),
        patch.object(va_mod, "download_recording", _slow),
    ):
        await _run_ingest(va)
    assert va._metrics[CAMERA].recording_failures == 1  # type: ignore[attr-defined]
    assert _work_dirs(tmp_path) == []


@pytest.mark.usefixtures("ffmpeg_present")
async def test_ingest_uses_configured_ffmpeg_binary(
    va: VideoAnalyzer, hass: MagicMock
) -> None:
    hass.states.get.return_value = _state("222")
    hass.data["ffmpeg"] = MagicMock(binary="/opt/ffmpeg")
    va._create_background_task = MagicMock(side_effect=lambda c, _n: c.close())  # type: ignore[method-assign]
    seen: list[str] = []

    async def _extract(binary: str, clip: Path, out_dir: Path) -> list[Path]:
        seen.append(binary)
        return await _fake_extract(1)(binary, clip, out_dir)

    with (
        patch.object(va_mod, "download_recording", _fake_download),
        patch.object(va_mod, "extract_frames", _extract),
    ):
        await _run_ingest(va)
    assert seen == ["/opt/ffmpeg"]


@pytest.mark.usefixtures("ffmpeg_present")
async def test_partial_stamp_failure_still_registers_placed_frames(
    va: VideoAnalyzer, hass: MagicMock, tmp_path: Path
) -> None:
    hass.states.get.return_value = _state("222")
    real = va_mod._place_recording_frame
    calls = {"n": 0}

    def _flaky(src: Path, snapshot_dir: Path, base: datetime, second: int) -> Path:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real(src, snapshot_dir, base, second)

    with (
        patch.object(va_mod, "_place_recording_frame", _flaky),
        patch.object(va_mod, "download_recording", _fake_download),
        patch.object(va_mod, "extract_frames", _fake_extract(3)),
    ):
        await _run_ingest(va)
    retention = va._retention_deques[CAMERA]  # type: ignore[attr-defined]
    assert len(retention) == 1
    assert retention[0].exists()
    assert va._metrics[CAMERA].recording_failures == 1  # type: ignore[attr-defined]
    assert _work_dirs(tmp_path) == []


# ---------------------------------------------------------------------------
# Metrics, orphan sweep, shutdown
# ---------------------------------------------------------------------------


async def test_metrics_report_resets_recording_counters(va: VideoAnalyzer) -> None:
    va._m_inc(CAMERA, "recording_frames", 3)  # type: ignore[attr-defined]
    va._m_inc(CAMERA, "recording_failures", 1)  # type: ignore[attr-defined]
    va._m_inc(CAMERA, "replaced_by_recording", 2)  # type: ignore[attr-defined]
    await va._metrics_flush_report(datetime.now(tz=UTC))  # type: ignore[attr-defined]
    m = va._metrics[CAMERA]  # type: ignore[attr-defined]
    assert (m.recording_frames, m.recording_failures, m.replaced_by_recording) == (
        0,
        0,
        0,
    )


async def test_retention_seed_sweeps_orphaned_work_dirs(
    va: VideoAnalyzer, tmp_path: Path
) -> None:
    cam = tmp_path / "camera_front_door"
    cam.mkdir()
    orphan = cam / f"{er_mod.EVENT_WORK_PREFIX}abc123"
    orphan.mkdir()
    (orphan / "recording.mp4").write_bytes(b"mp4")
    keep = cam / "snapshot_20250101_120000.jpg"
    keep.write_bytes(b"jpeg")
    with patch.object(va_mod, "VIDEO_ANALYZER_SNAPSHOT_ROOT", str(tmp_path)):
        await va._seed_retention_from_disk()  # type: ignore[attr-defined]
    assert not orphan.exists()
    assert keep.exists()


async def test_stop_cancels_inflight_ingest(va: VideoAnalyzer) -> None:
    started = asyncio.Event()

    async def _slow() -> None:
        started.set()
        await asyncio.sleep(60)

    task = asyncio.create_task(_slow())
    await started.wait()
    va._event_recording_tasks[f"{CAMERA}:1"] = task  # type: ignore[attr-defined]
    va._event_recording_recent[CAMERA] = deque(["1"])  # type: ignore[attr-defined]
    va._cancel_track = MagicMock()  # type: ignore[attr-defined]
    va._cancel_listen = MagicMock()  # type: ignore[attr-defined]

    await va.stop()

    assert task.cancelled()
    assert va._event_recording_tasks == {}  # type: ignore[attr-defined]
    assert va._event_recording_recent == {}  # type: ignore[attr-defined]


@pytest.mark.usefixtures("ffmpeg_present")
async def test_admit_after_stop_is_a_noop(va: VideoAnalyzer, hass: MagicMock) -> None:
    hass.states.get.return_value = _state("222")
    va._stopped = True  # type: ignore[attr-defined]
    download = AsyncMock()
    with patch.object(va_mod, "download_recording", download):
        await _run_ingest(va)
    assert download.await_count == 0


def test_module_constants_are_sane() -> None:
    assert er_mod.RECORDING_MAX_FRAMES >= 1
    assert er_mod.RECORDING_FETCH_BACKOFF_SEC[0] == 0.0
    assert sum(er_mod.RECORDING_FETCH_BACKOFF_SEC) <= 30
    assert er_mod.RECORDING_INGEST_DEADLINE_SEC > (
        er_mod.RECORDING_DOWNLOAD_TIMEOUT_SEC + er_mod.RECORDING_FFMPEG_TIMEOUT_SEC
    )


# ---------------------------------------------------------------------------
# Stale-snapshot leak (issue #491 field report): deferred flush + hash history
# ---------------------------------------------------------------------------


def _run_bg(va: VideoAnalyzer) -> list[asyncio.Future[Any]]:
    created: list[asyncio.Future[Any]] = []

    def _run(coro: Any, _name: str) -> asyncio.Future[Any]:
        fut = asyncio.ensure_future(coro)
        created.append(fut)
        return fut

    va._create_background_task = MagicMock(side_effect=_run)  # type: ignore[method-assign]
    return created


def _clip_frame(tmp_path: Path, second: int) -> Path:
    path = tmp_path / f"snapshot_20260928_0959{50 + second:02d}_r{second:02d}.jpg"
    path.write_bytes(b"clip")
    return path


async def test_window_close_waits_for_its_recording(
    va: VideoAnalyzer, tmp_path: Path
) -> None:
    """A clip landing after the window closed still displaces its snapshots."""
    created = _run_bg(va)
    analyze = AsyncMock()
    va._analyze_and_finalize = analyze  # type: ignore[method-assign]
    stale = tmp_path / "snapshot_20260928_095000.jpg"
    va._event_snapshot_buffers[CAMERA] = deque([stale])  # type: ignore[attr-defined]
    va._active_motion_cameras[CAMERA] = MagicMock()  # type: ignore[attr-defined]
    ingest: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    va._event_recording_tasks[f"{CAMERA}:222"] = ingest  # type: ignore[attr-defined]
    va._event_recording_task_gen[f"{CAMERA}:222"] = 0  # type: ignore[attr-defined]

    va._stop_motion_loop_and_flush(CAMERA)  # type: ignore[attr-defined]
    await asyncio.sleep(0)
    assert analyze.await_count == 0
    assert CAMERA not in va._event_snapshot_buffers  # type: ignore[attr-defined]

    frames = [_clip_frame(tmp_path, 0), _clip_frame(tmp_path, 1)]
    await va._admit_recording_frames(CAMERA, frames, 0)  # type: ignore[attr-defined]
    ingest.set_result(None)
    await asyncio.gather(*created)

    assert analyze.await_count == 1
    assert analyze.await_args is not None
    _, ordered = analyze.await_args.args
    assert [p for p, _ in ordered] == frames
    assert va._metrics[CAMERA].replaced_by_recording == 1  # type: ignore[attr-defined]
    assert va._deferred_flush == {}  # type: ignore[attr-defined]


async def test_deferred_flush_falls_back_to_snapshots_after_grace(
    va: VideoAnalyzer, tmp_path: Path
) -> None:
    created = _run_bg(va)
    analyze = AsyncMock()
    va._analyze_and_finalize = analyze  # type: ignore[method-assign]
    stale = tmp_path / "snapshot_20260928_095000.jpg"
    va._event_snapshot_buffers[CAMERA] = deque([stale])  # type: ignore[attr-defined]
    va._active_motion_cameras[CAMERA] = MagicMock()  # type: ignore[attr-defined]
    ingest: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    va._event_recording_tasks[f"{CAMERA}:222"] = ingest  # type: ignore[attr-defined]
    va._event_recording_task_gen[f"{CAMERA}:222"] = 0  # type: ignore[attr-defined]

    with patch.object(va_mod, "RECORDING_FLUSH_GRACE_SEC", 0.01):
        va._stop_motion_loop_and_flush(CAMERA)  # type: ignore[attr-defined]
        await asyncio.gather(*created)

    assert analyze.await_count == 1
    assert analyze.await_args is not None
    _, ordered = analyze.await_args.args
    assert [p for p, _ in ordered] == [stale]
    assert va._deferred_flush == {}  # type: ignore[attr-defined]
    ingest.cancel()


async def test_older_windows_recording_does_not_defer_the_flush(
    va: VideoAnalyzer, tmp_path: Path
) -> None:
    created = _run_bg(va)
    va._analyze_and_finalize = AsyncMock()  # type: ignore[method-assign]
    va._event_snapshot_buffers[CAMERA] = deque(  # type: ignore[attr-defined]
        [tmp_path / "snapshot_20260928_095000.jpg"]
    )
    va._active_motion_cameras[CAMERA] = MagicMock()  # type: ignore[attr-defined]
    va._window_generation[CAMERA] = 1  # type: ignore[attr-defined]
    ingest: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    va._event_recording_tasks[f"{CAMERA}:111"] = ingest  # type: ignore[attr-defined]
    va._event_recording_task_gen[f"{CAMERA}:111"] = 0  # type: ignore[attr-defined]

    va._stop_motion_loop_and_flush(CAMERA)  # type: ignore[attr-defined]
    await asyncio.gather(*created)

    assert va._analyze_and_finalize.await_count == 1  # type: ignore[attr-defined]
    assert va._deferred_flush == {}  # type: ignore[attr-defined]
    ingest.cancel()


def _noise_jpeg(path: Path, seed: int) -> Path:
    import random  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    rng = random.Random(seed)  # noqa: S311
    img = Image.new("L", (64, 64))
    img.putdata([rng.randrange(256) for _ in range(64 * 64)])
    img.save(path, "JPEG")
    return path


async def test_frozen_frame_recapture_is_rejected_after_clip_admit(
    va: VideoAnalyzer, tmp_path: Path
) -> None:
    """A battery cam's re-served frozen frame never joins the clip's batch."""
    va._is_unique_enough = VideoAnalyzer._is_unique_enough.__get__(va)  # type: ignore[method-assign]
    va._event_select_dedupe.add(CAMERA)  # type: ignore[attr-defined]
    va._active_motion_cameras[CAMERA] = MagicMock()  # type: ignore[attr-defined]
    frozen = _noise_jpeg(tmp_path / "snapshot_20260928_100001.jpg", 1)
    assert await va._is_unique_enough(CAMERA, frozen)
    va._event_snapshot_buffers[CAMERA] = deque([frozen])  # type: ignore[attr-defined]

    clip = [
        _noise_jpeg(tmp_path / f"snapshot_20260928_09595{i}_r0{i}.jpg", 10 + i)
        for i in range(3)
    ]
    await va._admit_recording_frames(CAMERA, clip, 0)  # type: ignore[attr-defined]
    assert list(va._event_snapshot_buffers[CAMERA]) == clip  # type: ignore[attr-defined]

    recapture = tmp_path / "snapshot_20260928_100004.jpg"
    shutil.copy(frozen, recapture)
    assert not await va._is_unique_enough(CAMERA, recapture)
    fresh = _noise_jpeg(tmp_path / "snapshot_20260928_100007.jpg", 99)
    assert await va._is_unique_enough(CAMERA, fresh)


async def test_late_clip_leaves_a_later_windows_history_alone(
    va: VideoAnalyzer, tmp_path: Path
) -> None:
    _run_bg(va)
    va._analyze_and_finalize = AsyncMock()  # type: ignore[method-assign]
    va._active_motion_cameras[CAMERA] = MagicMock()  # type: ignore[attr-defined]
    va._window_generation[CAMERA] = 2  # type: ignore[attr-defined]
    va._last_hashes[CAMERA] = deque([0xDEADBEEF], maxlen=2)  # type: ignore[attr-defined]

    async def _gate(camera_id: str, _path: Path) -> bool:
        va._last_hashes.setdefault(camera_id, deque(maxlen=2)).append(0x1)  # type: ignore[attr-defined]
        return True

    va._is_unique_enough = _gate  # type: ignore[method-assign]
    await va._admit_recording_frames(CAMERA, [_clip_frame(tmp_path, 0)], 1)  # type: ignore[attr-defined]
    assert CAMERA not in va._last_hashes  # type: ignore[attr-defined]
