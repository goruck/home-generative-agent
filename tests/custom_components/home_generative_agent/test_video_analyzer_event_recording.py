# ruff: noqa: S101, EM101, TRY003
"""
Tests for the ring-mqtt event-recording frame ingest (issue #491).

Covers the hass-free helpers in core/event_recording.py (URL reading, thinning,
download with size cap and not-ready classification, real ffmpeg extraction
when a binary is present) and the analyzer's scheduling, URL re-read, retry,
failure accounting, buffer replacement, dedupe floor, and shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
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
    download_recording,
    extract_frames,
    ffmpeg_binary,
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

URL = "https://download-eu.prod.phoenix.devices.amazon.dev/v1/download/abc.mp4?sig=1"
SELECT = "select.front_door_event_select"
CAMERA = "camera.front_door"
FFMPEG = shutil.which("ffmpeg")


@pytest.fixture(autouse=True)
def enable_event_loop_debug() -> None:
    """No-op override: pure-asyncio tests don't need HA's debug-mode hook."""


@pytest.fixture(autouse=True)
def verify_cleanup() -> None:
    """No-op override: all tasks are awaited explicitly below."""


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


# ---------------------------------------------------------------------------
# Helpers: URL reading, thinning, binary resolution
# ---------------------------------------------------------------------------


def test_recording_url_requires_https_string() -> None:
    assert recording_url_from_state(None) is None
    assert recording_url_from_state(_state("1", url=None)) is None
    assert recording_url_from_state(_state("1", url="")) is None
    assert recording_url_from_state(_state("1", url=42)) is None
    assert recording_url_from_state(_state("1", url="http://x/y.mp4?sig=1")) is None
    assert recording_url_from_state(_state("1", url=f"  {URL} ")) == URL


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
            f"testsrc=duration={seconds}:size=160x120:rate=5",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg binary not installed")
async def test_extract_frames_real_ffmpeg_one_per_second(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _make_clip(clip, seconds=5)
    out = tmp_path / "frames"
    out.mkdir()
    frames = await extract_frames(FFMPEG or "ffmpeg", clip, out)
    # fps=1 over a 5 s clip yields 5 frames (ffmpeg may emit one extra at the
    # boundary depending on build); every one must be a JPEG.
    assert 5 <= len(frames) <= 6
    assert frames == sorted(frames)
    assert all(f.read_bytes()[:2] == b"\xff\xd8" for f in frames)


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
    with (
        patch.object(va_mod, "async_call_later"),
        patch.object(va_mod, "ffmpeg_available", return_value=True),
    ):
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
    with (
        patch.object(va_mod, "async_call_later"),
        patch.object(va_mod, "ffmpeg_available", return_value=True),
    ):
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
        patch.object(va_mod, "ffmpeg_available", return_value=True),
        caplog.at_level(logging.DEBUG, logger=va_mod.LOGGER.name),
    ):
        va._handle_event_select_change(_select_event(_state("111"), new))  # type: ignore[attr-defined]
    names = [c.args[1] for c in stub.call_args_list]
    assert not any("event recording" in n for n in names)
    assert "Ring Protect" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_same_event_id_is_ingested_once(va: VideoAnalyzer) -> None:
    stub = _stub_bg_task(va)
    with patch.object(va_mod, "ffmpeg_available", return_value=True):
        va._maybe_ingest_event_recording(CAMERA, SELECT, _state("5"), "5")  # type: ignore[attr-defined]
        va._maybe_ingest_event_recording(CAMERA, SELECT, _state("5"), "5")  # type: ignore[attr-defined]
        va._maybe_ingest_event_recording(CAMERA, SELECT, _state("6"), "6")  # type: ignore[attr-defined]
    assert stub.call_count == 2


def test_missing_ffmpeg_warns_once_and_falls_back(
    va: VideoAnalyzer, caplog: pytest.LogCaptureFixture
) -> None:
    stub = _stub_bg_task(va)
    with (
        patch.object(va_mod, "ffmpeg_available", return_value=False),
        caplog.at_level(logging.WARNING, logger=va_mod.LOGGER.name),
    ):
        va._maybe_ingest_event_recording(CAMERA, SELECT, _state("5"), "5")  # type: ignore[attr-defined]
        va._maybe_ingest_event_recording(CAMERA, SELECT, _state("6"), "6")  # type: ignore[attr-defined]
    assert stub.call_count == 0
    warnings = [r for r in caplog.records if "ffmpeg" in r.getMessage()]
    assert len(warnings) == 1
    # The lookup is cached for the analyzer's lifetime (a reload re-checks),
    # and the failed attempts must not mark the events as ingested.
    assert va._event_recording_last_id == {}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Ingest: download → extract → stamp → admit
# ---------------------------------------------------------------------------


STARTED = datetime(2026, 9, 28, 10, 0, 0, tzinfo=UTC)


async def _run_ingest(
    va: VideoAnalyzer, *, url: str = URL, event_id: str = "222"
) -> None:
    await va._ingest_event_recording(  # type: ignore[attr-defined]
        CAMERA, SELECT, event_id, url, "ffmpeg", STARTED
    )


async def test_ingest_replaces_held_frames_with_clip_frames(
    va: VideoAnalyzer, hass: MagicMock, tmp_path: Path
) -> None:
    hass.states.get.return_value = _state("222")
    va._active_motion_cameras[CAMERA] = MagicMock()  # type: ignore[attr-defined]
    stale = tmp_path / "snapshot_20260928_095000.jpg"
    stale.write_bytes(b"old")
    va._event_snapshot_buffers[CAMERA] = deque([stale])  # type: ignore[attr-defined]
    va._is_unique_enough = AsyncMock(return_value=True)  # type: ignore[method-assign]

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
    assert epochs[0] == int(STARTED.timestamp())
    assert epochs[-1] == int(STARTED.timestamp()) + 11
    # Work dir is gone, frames are in retention, metrics counted.
    assert not (tmp_path / "_event_222").exists()
    retention = va._retention_deques[CAMERA]  # type: ignore[attr-defined]
    assert all(p in retention for p in buffer)
    metrics = va._metrics[CAMERA]  # type: ignore[attr-defined]
    assert metrics.recording_frames == er_mod.RECORDING_MAX_FRAMES
    assert metrics.replaced_by_recording == 1
    assert metrics.recording_failures == 0


async def test_second_clip_keeps_earlier_recording_frames(
    va: VideoAnalyzer, hass: MagicMock, tmp_path: Path
) -> None:
    """Two events inside one window: clip 2 drops snapshots, not clip 1."""
    hass.states.get.return_value = _state("222")
    va._active_motion_cameras[CAMERA] = MagicMock()  # type: ignore[attr-defined]
    va._is_unique_enough = AsyncMock(return_value=True)  # type: ignore[method-assign]
    snapshot = tmp_path / "snapshot_20260928_100003.jpg"
    snapshot.write_bytes(b"s")

    with (
        patch.object(va_mod, "download_recording", _fake_download),
        patch.object(va_mod, "extract_frames", _fake_extract(2)),
    ):
        await _run_ingest(va, event_id="222")
        first = list(va._event_snapshot_buffers[CAMERA])  # type: ignore[attr-defined]
        va._event_snapshot_buffers[CAMERA].append(snapshot)  # type: ignore[attr-defined]
        await _run_ingest(va, event_id="223")

    buffer = list(va._event_snapshot_buffers[CAMERA])  # type: ignore[attr-defined]
    assert len(first) == 2
    assert buffer[:2] == first
    assert snapshot not in buffer
    assert len(buffer) == 4
    assert va._metrics[CAMERA].replaced_by_recording == 1  # type: ignore[attr-defined]


async def test_ffmpeg_lookup_is_cached(va: VideoAnalyzer) -> None:
    _stub_bg_task(va)
    with patch.object(va_mod, "ffmpeg_available", return_value=True) as avail:
        va._maybe_ingest_event_recording(CAMERA, SELECT, _state("5"), "5")  # type: ignore[attr-defined]
        va._maybe_ingest_event_recording(CAMERA, SELECT, _state("6"), "6")  # type: ignore[attr-defined]
    assert avail.call_count == 1


async def test_ingest_after_window_flush_goes_to_live_queue(
    va: VideoAnalyzer, hass: MagicMock
) -> None:
    hass.states.get.return_value = _state("222")
    va._is_unique_enough = AsyncMock(return_value=True)  # type: ignore[method-assign]
    queue: asyncio.Queue[Any] = asyncio.Queue()
    va._get_snapshot_queue = MagicMock(return_value=queue)  # type: ignore[method-assign]

    with (
        patch.object(va_mod, "download_recording", _fake_download),
        patch.object(va_mod, "extract_frames", _fake_extract(3)),
    ):
        await _run_ingest(va)

    assert queue.qsize() == 3
    assert CAMERA not in va._event_snapshot_buffers  # type: ignore[attr-defined]


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


async def test_ingest_rereads_url_for_same_event(
    va: VideoAnalyzer, hass: MagicMock
) -> None:
    fresh = URL.replace("sig=1", "sig=2")
    hass.states.get.return_value = _state("222", url=fresh)
    va._is_unique_enough = AsyncMock(return_value=True)  # type: ignore[method-assign]
    va._get_snapshot_queue = MagicMock(return_value=asyncio.Queue())  # type: ignore[method-assign]
    seen: list[str] = []

    async def _dl(_c: Any, url: str, dest: Path) -> int:
        seen.append(url)
        return await _fake_download(_c, url, dest)

    with (
        patch.object(va_mod, "download_recording", _dl),
        patch.object(va_mod, "extract_frames", _fake_extract(1)),
    ):
        await _run_ingest(va, url=URL)
    assert seen == [fresh]


async def test_ingest_keeps_trigger_url_when_entity_moved_on(
    va: VideoAnalyzer, hass: MagicMock
) -> None:
    hass.states.get.return_value = _state("333", url=URL.replace("sig=1", "sig=3"))
    va._is_unique_enough = AsyncMock(return_value=True)  # type: ignore[method-assign]
    va._get_snapshot_queue = MagicMock(return_value=asyncio.Queue())  # type: ignore[method-assign]
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


async def test_ingest_retries_not_ready_then_succeeds(
    va: VideoAnalyzer, hass: MagicMock
) -> None:
    hass.states.get.return_value = _state("222")
    va._is_unique_enough = AsyncMock(return_value=True)  # type: ignore[method-assign]
    va._get_snapshot_queue = MagicMock(return_value=asyncio.Queue())  # type: ignore[method-assign]
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
    assert not (tmp_path / "_event_222").exists()
    assert va._metrics[CAMERA].recording_failures == 2  # type: ignore[attr-defined]
    failures = [r for r in caplog.records if "could not be analyzed" in r.getMessage()]
    assert [r.levelno for r in failures] == [logging.WARNING, logging.DEBUG]
    assert "never became fetchable" in failures[0].getMessage()


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
    assert not (tmp_path / "_event_222").exists()
    assert CAMERA not in va._event_snapshot_buffers  # type: ignore[attr-defined]


async def test_ingest_uses_configured_ffmpeg_binary(
    va: VideoAnalyzer, hass: MagicMock
) -> None:
    hass.states.get.return_value = _state("222")
    va._is_unique_enough = AsyncMock(return_value=True)  # type: ignore[method-assign]
    va._get_snapshot_queue = MagicMock(return_value=asyncio.Queue())  # type: ignore[method-assign]
    seen: list[str] = []

    async def _extract(binary: str, clip: Path, out_dir: Path) -> list[Path]:
        seen.append(binary)
        return await _fake_extract(1)(binary, clip, out_dir)

    with (
        patch.object(va_mod, "download_recording", _fake_download),
        patch.object(va_mod, "extract_frames", _extract),
    ):
        await va._ingest_event_recording(  # type: ignore[attr-defined]
            CAMERA, SELECT, "222", URL, "/opt/ffmpeg", STARTED
        )
    assert seen == ["/opt/ffmpeg"]


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


async def test_stop_cancels_inflight_ingest(va: VideoAnalyzer) -> None:
    started = asyncio.Event()

    async def _slow() -> None:
        started.set()
        await asyncio.sleep(60)

    task = asyncio.create_task(_slow())
    await started.wait()
    va._event_recording_tasks[f"{CAMERA}:1"] = task  # type: ignore[attr-defined]
    va._event_recording_last_id[CAMERA] = "1"  # type: ignore[attr-defined]
    va._cancel_track = MagicMock()  # type: ignore[attr-defined]
    va._cancel_listen = MagicMock()  # type: ignore[attr-defined]

    await va.stop()

    assert task.cancelled()
    assert va._event_recording_tasks == {}  # type: ignore[attr-defined]
    assert va._event_recording_last_id == {}  # type: ignore[attr-defined]


def test_module_constants_are_sane() -> None:
    assert er_mod.RECORDING_MAX_FRAMES >= 1
    assert er_mod.RECORDING_FETCH_BACKOFF_SEC[0] == 0.0
    assert sum(er_mod.RECORDING_FETCH_BACKOFF_SEC) <= 30
