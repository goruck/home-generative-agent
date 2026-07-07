# ruff: noqa: S101, S108, PYI034
"""
Tests for video analyzer priority management.

Covers (test plan items):
- Queue item record preserves path and adds enqueue timestamp
- Stale-frame backlog policy: frames beyond threshold dropped, newest kept
- Stale-frame evaluation occurs before semaphore acquisition
- Semaphore acquisition timeout causes frame drop rather than 90 s wait
- Semaphore timeout increments the semaphore_timeouts counter
- VLM calls include bounded num_predict matching VIDEO_VLM_NUM_PREDICT
- VLM calls pass reasoning=False via configurable
- Summarization calls include bounded num_predict matching VIDEO_SUMMARY_NUM_PREDICT
- Summarization calls pass reasoning=False via configurable
- VLM frame analysis and summary generation share the same semaphore concurrency limit
- Startup logs video_model_semaphore size and uncontended flag
- Startup capability probe logs model/memory data when available
- Startup capability probe falls back silently when probe fails
- Motion snapshot loop uses VIDEO_ANALYZER_MOTION_SCAN_INTERVAL, not VIDEO_ANALYZER_SCAN_INTERVAL
- _resolve_camera_from_motion: direct match, VMD strip, _motion strip (Reolink + Ring-MQTT), no match, override precedence
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import custom_components.home_generative_agent as hga_mod
import custom_components.home_generative_agent.core.utils as utils_mod
import custom_components.home_generative_agent.core.video_analyzer as va_mod
from custom_components.home_generative_agent.const import (
    VIDEO_ANALYZER_MOTION_SCAN_INTERVAL,
    VIDEO_ANALYZER_SCAN_INTERVAL,
    VIDEO_SUMMARY_NUM_PREDICT,
    VIDEO_VLM_NUM_PREDICT,
)
from custom_components.home_generative_agent.core.video_analyzer import (
    _VIDEO_QUEUE_BACKLOG_THRESHOLD,
    VideoAnalyzer,
    _SnapshotItem,
)
from custom_components.home_generative_agent.core.video_helpers import (
    put_with_backpressure,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

hga_symbols = cast("Mapping[str, Any]", vars(hga_mod))
NullChat = cast("type[Any]", hga_symbols["NullChat"])
_log_ollama_server_info = cast(
    "Callable[[Any, dict[str, str]], Awaitable[None]]",
    hga_symbols["_log_ollama_server_info"],
)


def _model_deployment_get(va: VideoAnalyzer) -> MagicMock:
    """Return the MagicMock backing runtime_data.model_deployments.get."""
    return cast("MagicMock", va.entry.runtime_data.model_deployments.get)


def _entry_model_deployment_get(entry: MagicMock) -> MagicMock:
    """Return the MagicMock backing an entry fixture's model deployment getter."""
    return cast("MagicMock", entry.runtime_data.model_deployments.get)


# ---------------------------------------------------------------------------
# Override autouse fixtures from pytest-homeassistant-custom-component
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def enable_event_loop_debug() -> None:
    """No-op override: pure-asyncio tests don't need HA's debug-mode hook."""


@pytest.fixture(autouse=True)
def verify_cleanup() -> None:
    """No-op override: all tasks explicitly awaited; no HA resources to clean up."""


# ---------------------------------------------------------------------------
# Reset gate primitives that video sessions touch
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def _fresh_gate_primitives(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset admission-control globals before each test."""
    video_idle = asyncio.Event()
    video_idle.set()
    chat_idle = asyncio.Event()
    chat_idle.set()

    monkeypatch.setattr(utils_mod, "_video_idle", video_idle)
    monkeypatch.setattr(utils_mod, "_video_active_count", 0)
    monkeypatch.setattr(utils_mod, "_chat_idle", chat_idle)
    monkeypatch.setattr(utils_mod, "_chat_active_count", 0)


# ---------------------------------------------------------------------------
# Minimal fake async file for aiofiles.open mocking
# ---------------------------------------------------------------------------


class _FakeImageFile:
    """Async context manager that mimics an aiofiles file handle."""

    BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 20  # minimal JPEG-like header

    async def read(self) -> bytes:
        return self.BYTES

    async def __aenter__(self) -> _FakeImageFile:
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False


# ---------------------------------------------------------------------------
# Fixtures: minimal VideoAnalyzer
# ---------------------------------------------------------------------------


@pytest.fixture
def entry() -> MagicMock:
    """Config entry stub with runtime_data defaults."""
    e = MagicMock()
    e.runtime_data.options = {}
    e.runtime_data.model_deployments.get.return_value = "cloud"
    e.runtime_data.face_recognition = False

    configured_vlm: Any = MagicMock()
    configured_vlm.ainvoke = AsyncMock(return_value=MagicMock(content="frame desc"))
    e.runtime_data.vision_model.with_config.return_value = configured_vlm
    # Empty existing config so the video override dict is predictable in tests.
    e.runtime_data.vision_model.config = {}

    configured_sum: Any = MagicMock()
    configured_sum.ainvoke = AsyncMock(
        return_value=MagicMock(content="A person walks by.")
    )
    e.runtime_data.summarization_model.with_config.return_value = configured_sum
    e.runtime_data.summarization_model.config = {}

    return e


@pytest.fixture
def hass() -> MagicMock:
    return MagicMock()


@pytest.fixture
def va(hass: MagicMock, entry: MagicMock) -> VideoAnalyzer:
    """VideoAnalyzer with a pre-built semaphore (simulates post-start state)."""
    analyzer = VideoAnalyzer(hass, entry)
    analyzer._video_model_sem = asyncio.Semaphore(1)  # type: ignore[attr-defined]
    return analyzer


# ---------------------------------------------------------------------------
# _SnapshotItem: queue item record
# ---------------------------------------------------------------------------


def test_snapshot_item_has_path_and_enqueued() -> None:
    """Queue item record preserves path and adds enqueue timestamp."""
    item = _SnapshotItem(path=Path("/tmp/snap.jpg"), enqueued=1_234_567.0)
    assert item.path == Path("/tmp/snap.jpg")
    assert item.enqueued == pytest.approx(1_234_567.0)


def test_snapshot_item_path_remains_a_path() -> None:
    """Path attribute is a Path, not a str."""
    item = _SnapshotItem(path=Path("snapshot_20250101_120000.jpg"), enqueued=0.0)
    assert isinstance(item.path, Path)


# ---------------------------------------------------------------------------
# put_with_backpressure: generic over _SnapshotItem
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_put_with_backpressure_accepts_snapshot_item() -> None:
    """put_with_backpressure works with the new _SnapshotItem queue type."""
    q: asyncio.Queue[_SnapshotItem] = asyncio.Queue(maxsize=2)
    items = [
        _SnapshotItem(path=Path(f"/tmp/snap_{i}.jpg"), enqueued=float(i))
        for i in range(3)
    ]
    for item in items:
        await put_with_backpressure(q, item)

    # Queue held 2; the 3rd call should have dropped the oldest (index 0)
    assert q.qsize() == 2
    remaining = [q.get_nowait() for _ in range(2)]
    assert remaining[0].path == Path("/tmp/snap_1.jpg")
    assert remaining[1].path == Path("/tmp/snap_2.jpg")


@pytest.mark.asyncio
async def test_put_with_backpressure_empty_queue_no_drop() -> None:
    """When queue is not full no item is dropped."""
    q: asyncio.Queue[_SnapshotItem] = asyncio.Queue(maxsize=5)
    item = _SnapshotItem(path=Path("/tmp/snap.jpg"), enqueued=1.0)
    await put_with_backpressure(q, item)
    assert q.qsize() == 1
    assert q.get_nowait().path == Path("/tmp/snap.jpg")


# ---------------------------------------------------------------------------
# _get_batch: backlog policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_batch_deep_queue_keeps_newest_frame(va: VideoAnalyzer) -> None:
    """When queue depth exceeds threshold after first get, only newest is kept."""
    q: asyncio.Queue[_SnapshotItem] = asyncio.Queue(maxsize=50)
    t = 1_000.0
    # 4 items: 1 is dequeued first, leaving 3 > _VIDEO_QUEUE_BACKLOG_THRESHOLD (2)
    for i in range(4):
        await q.put(_SnapshotItem(path=Path(f"/tmp/snap_{i}.jpg"), enqueued=t + i))

    batch = await va._get_batch(q, "camera.test")  # type: ignore[attr-defined]

    assert len(batch) == 1
    assert batch[0] == Path("/tmp/snap_3.jpg")  # newest by enqueued time
    assert va._metrics["camera.test"].dropped_backlog == 3  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_get_batch_shallow_queue_returns_all_frames(va: VideoAnalyzer) -> None:
    """With queue at or below threshold, all frames are returned normally."""
    q: asyncio.Queue[_SnapshotItem] = asyncio.Queue(maxsize=50)
    t = 1_000.0
    # 3 items: 1 is dequeued first, leaving 2 == threshold — no backlog drop
    for i in range(3):
        await q.put(_SnapshotItem(path=Path(f"/tmp/snap_{i}.jpg"), enqueued=t + i))

    batch = await va._get_batch(q, "camera.test")  # type: ignore[attr-defined]

    assert len(batch) == 3
    assert va._metrics["camera.test"].dropped_backlog == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_get_batch_backlog_threshold_constant_value() -> None:
    """_VIDEO_QUEUE_BACKLOG_THRESHOLD is 2 (plan spec)."""
    assert _VIDEO_QUEUE_BACKLOG_THRESHOLD == 2


@pytest.mark.asyncio
async def test_get_batch_single_item_no_backlog(va: VideoAnalyzer) -> None:
    """Single-item queue is returned as-is without any drop."""
    q: asyncio.Queue[_SnapshotItem] = asyncio.Queue(maxsize=50)
    await q.put(_SnapshotItem(path=Path("/tmp/only.jpg"), enqueued=1.0))

    batch = await va._get_batch(q, "camera.test")  # type: ignore[attr-defined]

    assert batch == [Path("/tmp/only.jpg")]
    assert va._metrics["camera.test"].dropped_backlog == 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _process_snapshot: stale frame check before semaphore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_frame_skips_semaphore(va: VideoAnalyzer) -> None:
    """Stale frame returns {} immediately without calling semaphore.acquire."""
    # Use a mock semaphore so we can assert acquire was never called
    mock_sem = MagicMock()
    mock_sem.acquire = AsyncMock()
    va._video_model_sem = mock_sem  # type: ignore[attr-defined]
    _model_deployment_get(va).return_value = "edge"

    # Invalid path name → epoch_from_path raises ValueError → epoch=0 → stale
    result = await va._process_snapshot(Path("not_a_snapshot.jpg"), "camera.test")  # type: ignore[attr-defined]

    assert result == {}
    assert va._metrics["camera.test"].dropped_stale == 1  # type: ignore[attr-defined]
    mock_sem.acquire.assert_not_called()


@pytest.mark.asyncio
async def test_stale_frame_does_not_open_file(va: VideoAnalyzer) -> None:
    """Stale-frame path returns before trying to open the snapshot file."""
    _model_deployment_get(va).return_value = "edge"

    with patch("aiofiles.open", return_value=_FakeImageFile()) as mock_open:
        await va._process_snapshot(Path("not_a_snapshot.jpg"), "camera.test")  # type: ignore[attr-defined]

    mock_open.assert_not_called()


# ---------------------------------------------------------------------------
# _process_snapshot: semaphore timeout causes frame drop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semaphore_timeout_causes_frame_drop(
    va: VideoAnalyzer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Semaphore acquisition timeout causes _process_snapshot to return {}."""
    # 0-permit semaphore → any acquire blocks
    va._video_model_sem = asyncio.Semaphore(0)  # type: ignore[attr-defined]
    _model_deployment_get(va).return_value = "edge"
    monkeypatch.setattr(va_mod, "_VIDEO_MODEL_SEMAPHORE_WAIT_SEC", 0.05)

    with (
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.epoch_from_path",
            return_value=int(time.time()),
        ),
        patch("aiofiles.open", return_value=_FakeImageFile()),
    ):
        result = await va._process_snapshot(  # type: ignore[attr-defined]
            Path("snapshot_20250101_120000.jpg"), "camera.test"
        )

    assert result == {}


@pytest.mark.asyncio
async def test_semaphore_timeout_increments_counter(
    va: VideoAnalyzer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Semaphore timeout increments semaphore_timeouts metric."""
    va._video_model_sem = asyncio.Semaphore(0)  # type: ignore[attr-defined]
    _model_deployment_get(va).return_value = "edge"
    monkeypatch.setattr(va_mod, "_VIDEO_MODEL_SEMAPHORE_WAIT_SEC", 0.05)

    with (
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.epoch_from_path",
            return_value=int(time.time()),
        ),
        patch("aiofiles.open", return_value=_FakeImageFile()),
    ):
        await va._process_snapshot(  # type: ignore[attr-defined]
            Path("snapshot_20250101_120000.jpg"), "camera.test"
        )

    assert va._metrics["camera.test"].semaphore_timeouts == 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _process_snapshot: VLM num_predict and reasoning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vlm_call_uses_video_vlm_num_predict(va: VideoAnalyzer) -> None:
    """VLM frame analysis is configured with VIDEO_VLM_NUM_PREDICT."""
    # Cloud deployment bypasses the semaphore gate
    _model_deployment_get(va).return_value = "cloud"

    with (
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.epoch_from_path",
            return_value=int(time.time()),
        ),
        patch("aiofiles.open", return_value=_FakeImageFile()),
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.analyze_image",
            new=AsyncMock(return_value="a person is at the door"),
        ),
    ):
        await va._process_snapshot(  # type: ignore[attr-defined]
            Path("snapshot_20250101_120000.jpg"), "camera.test"
        )

    call_cfg = va.entry.runtime_data.vision_model.with_config.call_args.kwargs["config"]
    assert call_cfg["configurable"]["num_predict"] == VIDEO_VLM_NUM_PREDICT
    assert "reasoning" not in call_cfg["configurable"]


@pytest.mark.asyncio
async def test_vlm_call_disables_reasoning(va: VideoAnalyzer) -> None:
    """Edge VLM frame analysis binds reasoning=False to suppress Ollama thinking."""
    _model_deployment_get(va).return_value = "edge"

    with (
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.epoch_from_path",
            return_value=int(time.time()),
        ),
        patch("aiofiles.open", return_value=_FakeImageFile()),
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.analyze_image",
            new=AsyncMock(return_value="desc"),
        ),
    ):
        await va._process_snapshot(  # type: ignore[attr-defined]
            Path("snapshot_20250101_120000.jpg"), "camera.test"
        )

    call_cfg = va.entry.runtime_data.vision_model.with_config.call_args.kwargs["config"]
    assert call_cfg["configurable"]["num_predict"] == VIDEO_VLM_NUM_PREDICT
    assert call_cfg["configurable"]["reasoning"] is False


# ---------------------------------------------------------------------------
# _generate_summary: summarization num_predict and reasoning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_uses_video_summary_num_predict(va: VideoAnalyzer) -> None:
    """Multi-frame summary is configured with VIDEO_SUMMARY_NUM_PREDICT."""
    _model_deployment_get(va).return_value = "cloud"

    frame_descs = [
        {"A person is at the gate. t+0s.": []},
        {"The person walks away. t+3s.": []},
    ]
    result = await va._generate_summary(frame_descs)  # type: ignore[attr-defined]

    call_cfg = va.entry.runtime_data.summarization_model.with_config.call_args.kwargs[
        "config"
    ]
    assert call_cfg["configurable"]["num_predict"] == VIDEO_SUMMARY_NUM_PREDICT
    assert "reasoning" not in call_cfg["configurable"]
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_summary_disables_reasoning(va: VideoAnalyzer) -> None:
    """Edge summarization passes reasoning=False directly to ainvoke."""
    _model_deployment_get(va).return_value = "edge"

    frame_descs = [
        {"Scene A.": []},
        {"Scene B.": []},
    ]
    await va._generate_summary(frame_descs)  # type: ignore[attr-defined]

    call_cfg = va.entry.runtime_data.summarization_model.with_config.call_args.kwargs[
        "config"
    ]
    assert call_cfg["configurable"]["num_predict"] == VIDEO_SUMMARY_NUM_PREDICT
    assert call_cfg["configurable"]["reasoning"] is False
    # reasoning is in the config, not in ainvoke kwargs
    configured_sum = va.entry.runtime_data.summarization_model.with_config.return_value
    assert configured_sum.ainvoke.call_args.kwargs.get("reasoning") is None


# ---------------------------------------------------------------------------
# NullChat fallback: with_config(config=...) must not raise TypeError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vlm_null_chat_with_config_does_not_raise(
    hass: MagicMock, entry: MagicMock
) -> None:
    """NullChat.with_config accepts a positional config arg without TypeError."""
    entry.runtime_data.vision_model = NullChat()
    _entry_model_deployment_get(entry).return_value = "cloud"
    va = VideoAnalyzer(hass, entry)
    va._video_model_sem = asyncio.Semaphore(1)  # type: ignore[attr-defined]

    with (
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.epoch_from_path",
            return_value=int(time.time()),
        ),
        patch("aiofiles.open", return_value=_FakeImageFile()),
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.analyze_image",
            new=AsyncMock(return_value="desc"),
        ),
    ):
        result = await va._process_snapshot(  # type: ignore[attr-defined]
            Path("snapshot_20250101_120000.jpg"), "camera.test"
        )

    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_summary_null_chat_with_config_does_not_raise(
    hass: MagicMock, entry: MagicMock
) -> None:
    """NullChat.with_config accepts a positional config arg without TypeError."""
    entry.runtime_data.summarization_model = NullChat()
    _entry_model_deployment_get(entry).return_value = "cloud"
    va = VideoAnalyzer(hass, entry)

    frame_descs = [{"Scene A.": []}, {"Scene B.": []}]
    # NullChat.ainvoke returns a str, not an AIMessage, so _generate_summary
    # raises ValueError (empty content) — the correct fallback path.  A TypeError
    # from with_config would propagate uncaught and fail this test.
    with contextlib.suppress(ValueError):
        await va._generate_summary(frame_descs)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _resolve_camera_from_motion: motion sensor → camera entity resolution
# ---------------------------------------------------------------------------


def _make_va_with_states(
    hass: MagicMock, entry: MagicMock, existing_camera_ids: list[str]
) -> VideoAnalyzer:
    """Return a VideoAnalyzer whose hass.states.get mirrors existing_camera_ids."""
    existing = set(existing_camera_ids)
    hass.states.get.side_effect = lambda eid: MagicMock() if eid in existing else None
    return VideoAnalyzer(hass, entry)


def test_resolve_direct_name_match(hass: MagicMock, entry: MagicMock) -> None:
    """binary_sensor.X maps to camera.X when that entity exists."""
    va = _make_va_with_states(hass, entry, ["camera.frontgate"])
    result = va._resolve_camera_from_motion("binary_sensor.frontgate")  # type: ignore[attr-defined]
    assert result == "camera.frontgate"


def test_resolve_vmd_suffix_stripped(hass: MagicMock, entry: MagicMock) -> None:
    """UniFi Protect VMD sensors (binary_sensor.X_vmd1) resolve to camera.X."""
    va = _make_va_with_states(hass, entry, ["camera.frontgate"])
    result = va._resolve_camera_from_motion("binary_sensor.frontgate_vmd1")  # type: ignore[attr-defined]
    assert result == "camera.frontgate"


def test_resolve_motion_suffix_reolink(hass: MagicMock, entry: MagicMock) -> None:
    """Reolink: binary_sensor.X_motion resolves to camera.X."""
    va = _make_va_with_states(hass, entry, ["camera.front_door"])
    result = va._resolve_camera_from_motion("binary_sensor.front_door_motion")  # type: ignore[attr-defined]
    assert result == "camera.front_door"


def test_resolve_motion_suffix_ring_mqtt(hass: MagicMock, entry: MagicMock) -> None:
    """Ring-MQTT: binary_sensor.X_motion resolves to camera.X_snapshot."""
    va = _make_va_with_states(hass, entry, ["camera.front_door_snapshot"])
    result = va._resolve_camera_from_motion("binary_sensor.front_door_motion")  # type: ignore[attr-defined]
    assert result == "camera.front_door_snapshot"


def test_resolve_no_match_returns_none(hass: MagicMock, entry: MagicMock) -> None:
    """Returns None when no camera entity can be resolved."""
    va = _make_va_with_states(hass, entry, [])
    result = va._resolve_camera_from_motion("binary_sensor.unknown_sensor_motion")  # type: ignore[attr-defined]
    assert result is None


def test_resolve_override_map_takes_precedence(
    hass: MagicMock, entry: MagicMock
) -> None:
    """Explicit override map is checked before any inference."""
    va = _make_va_with_states(hass, entry, ["camera.override_cam", "camera.inferred"])
    override = {"binary_sensor.front_door_motion": "camera.override_cam"}
    with patch(
        "custom_components.home_generative_agent.core.video_analyzer.VIDEO_ANALYZER_MOTION_CAMERA_MAP",
        override,
    ):
        result = va._resolve_camera_from_motion("binary_sensor.front_door_motion")  # type: ignore[attr-defined]
    assert result == "camera.override_cam"


def test_motion_scan_interval_is_longer_than_recording_interval() -> None:
    """Motion loop uses a longer interval than the recording-camera poll to avoid
    notification bursts on cameras with extended motion windows (e.g. Ring-MQTT)."""
    assert VIDEO_ANALYZER_MOTION_SCAN_INTERVAL > VIDEO_ANALYZER_SCAN_INTERVAL


@pytest.mark.asyncio
async def test_null_chat_ainvoke_accepts_positional_config() -> None:
    """NullChat.ainvoke accepts config as a positional arg (LangChain convention)."""
    result = await NullChat().ainvoke([], {"configurable": {}})
    assert result == "LLM unavailable."


@pytest.mark.asyncio
async def test_null_chat_astream_accepts_positional_config() -> None:
    """NullChat.astream accepts config as a positional arg (LangChain convention)."""
    chunks = [chunk async for chunk in NullChat().astream([], {"configurable": {}})]
    assert chunks == ["LLM unavailable."]


# ---------------------------------------------------------------------------
# Shared semaphore: VLM and summary share the same concurrency limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semaphore_limits_concurrent_vlm_calls(
    va: VideoAnalyzer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple camera workers share one semaphore; max one VLM call at a time."""
    _model_deployment_get(va).return_value = "edge"
    va._video_model_sem = asyncio.Semaphore(1)  # type: ignore[attr-defined]

    concurrent = 0
    max_concurrent = 0

    async def _slow_analyze(*_args: object, **_kwargs: object) -> str:
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.05)
        concurrent -= 1
        return "frame description"

    monkeypatch.setattr(va_mod, "_VIDEO_MODEL_SEMAPHORE_WAIT_SEC", 5)

    with (
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.epoch_from_path",
            return_value=int(time.time()),
        ),
        patch("aiofiles.open", return_value=_FakeImageFile()),
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.analyze_image",
            new=AsyncMock(side_effect=_slow_analyze),
        ),
    ):
        tasks = [
            asyncio.create_task(
                va._process_snapshot(  # type: ignore[attr-defined]
                    Path("snapshot_20250101_120000.jpg"), f"camera.test_{i}"
                )
            )
            for i in range(3)
        ]
        await asyncio.gather(*tasks)

    assert max_concurrent == 1


# ---------------------------------------------------------------------------
# start(): semaphore size and uncontended flag logged at startup
# ---------------------------------------------------------------------------


def test_start_logs_semaphore_size(
    va: VideoAnalyzer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """start() emits an INFO log with video_model_semaphore size and uncontended flag."""
    va.entry.runtime_data.options = {
        "video_model_semaphore": 2,
        "model_provider_uncontended": True,
    }

    with (
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.async_track_time_interval",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.get_async_client",
            return_value=MagicMock(),
        ),
        caplog.at_level(
            logging.INFO,
            logger="custom_components.home_generative_agent.core.video_analyzer",
        ),
    ):
        va.start()

    messages = [r.message for r in caplog.records]
    assert any("video_model_semaphore=2" in m for m in messages)
    assert any("uncontended=True" in m for m in messages)

    cast("MagicMock", va.hass.bus.async_listen).assert_called()


def test_start_builds_semaphore_from_config(va: VideoAnalyzer) -> None:
    """start() reads CONF_VIDEO_MODEL_SEMAPHORE from options to size the semaphore."""
    va.entry.runtime_data.options = {"video_model_semaphore": 3}

    with (
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.async_track_time_interval",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.get_async_client",
            return_value=MagicMock(),
        ),
    ):
        va.start()

    assert va._video_model_sem is not None  # type: ignore[attr-defined]
    # Semaphore with 3 permits: internal counter should equal the configured size
    assert va._video_model_sem._value == 3  # type: ignore[attr-defined]


def test_start_uses_home_assistant_shared_httpx_client(va: VideoAnalyzer) -> None:
    """start() must not create a private httpx client in the event loop."""
    client = MagicMock()

    with (
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.async_track_time_interval",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.get_async_client",
            return_value=client,
        ) as get_client,
    ):
        va.start()

    get_client.assert_called_once_with(va.hass)
    assert va._httpx_client is client  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _log_ollama_server_info: startup capability probe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_ollama_server_info_logs_model_and_vram(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Startup probe logs model name and VRAM info when /api/ps responds."""
    hass = MagicMock()
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "models": [
            {
                "name": "qwen3:8b",
                "size": 5_000_000_000,
                "size_vram": 4_000_000_000,
            }
        ]
    }
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    with (
        patch(
            "custom_components.home_generative_agent.get_async_client",
            return_value=mock_client,
        ),
        caplog.at_level(
            logging.INFO,
            logger="custom_components.home_generative_agent",
        ),
    ):
        await _log_ollama_server_info(hass, {"vlm": "http://ollama:11434"})

    messages = [r.message for r in caplog.records]
    assert any("qwen3:8b" in m for m in messages)
    assert any("vram_mb" in m for m in messages)


@pytest.mark.asyncio
async def test_log_ollama_server_info_falls_back_silently_on_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Startup probe logs a WARNING and does not raise when /api/ps is unreachable."""
    hass = MagicMock()
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("connection refused"))

    with (
        patch(
            "custom_components.home_generative_agent.get_async_client",
            return_value=mock_client,
        ),
        caplog.at_level(
            logging.WARNING,
            logger="custom_components.home_generative_agent",
        ),
    ):
        # Must not raise
        await _log_ollama_server_info(hass, {"vlm": "http://ollama:11434"})

    messages = [r.message for r in caplog.records]
    assert any("probe_failed" in m for m in messages)


@pytest.mark.asyncio
async def test_log_ollama_server_info_logs_shared_url_swap_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Probe logs a swap-risk note when multiple categories share a URL."""
    hass = MagicMock()
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "models": [{"name": "llama3:8b", "size": 0, "size_vram": 0}]
    }
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    shared_url = "http://ollama:11434"
    with (
        patch(
            "custom_components.home_generative_agent.get_async_client",
            return_value=mock_client,
        ),
        caplog.at_level(
            logging.INFO,
            logger="custom_components.home_generative_agent",
        ),
    ):
        await _log_ollama_server_info(
            hass, {"vlm": shared_url, "summarization": shared_url}
        )

    messages = [r.message for r in caplog.records]
    assert any("model_swapping_may_add_latency" in m for m in messages)
