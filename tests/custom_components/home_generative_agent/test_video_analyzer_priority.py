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
- VLM calls force reasoning=False only when the base config carries a reasoning key
- Summarization calls include bounded num_predict matching VIDEO_SUMMARY_NUM_PREDICT
- Summarization calls force reasoning=False only when the base config carries one
- Worker survives unexpected exceptions (e.g. ollama.ResponseError) and keeps consuming
- Worker exit removes the queue entry so the next snapshot respawns a worker
- analyze_image propagates ollama.ResponseError; _process_snapshot and the
  camera chat tool handle it at their own boundaries (issue #473)
- VLM frame analysis and summary generation share the same semaphore concurrency limit
- Startup logs video_model_semaphore size and uncontended flag
- Startup capability probe logs model/memory data when available
- Startup capability probe falls back silently when probe fails
- Motion snapshot loop runs at VIDEO_ANALYZER_MOTION_SCAN_INTERVAL until cancelled
- Recording poll skips cameras already tracked by the motion loop (mutual exclusion)
- _resolve_camera_from_motion: direct match, VMD strip, _motion strip (Reolink + Ring-MQTT), no match, override precedence
- Event lifecycle: motion-held snapshots are not analyzed until the OFF/exit
  flush, which processes the whole window as one ordered batch
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ollama import ResponseError as OllamaResponseError

import custom_components.home_generative_agent as hga_mod
import custom_components.home_generative_agent.core.utils as utils_mod
import custom_components.home_generative_agent.core.video_analyzer as va_mod
from custom_components.home_generative_agent.agent.tools import (
    analyze_image,
    get_and_analyze_camera_image,
)
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
    """Edge VLM analysis forces reasoning=False for thinking-capable models."""
    _model_deployment_get(va).return_value = "edge"
    # Base config carries a reasoning key (reasoning_field() for thinking models).
    va.entry.runtime_data.vision_model.config = {"configurable": {"reasoning": True}}

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


@pytest.mark.asyncio
async def test_vlm_call_omits_reasoning_for_non_thinking_model(
    va: VideoAnalyzer,
) -> None:
    """
    Edge VLM analysis must not inject a reasoning key for non-thinking models.

    reasoning_field() returns {} for models like gemma3/qwen2.5vl; sending
    think=false anyway makes some Ollama builds reject the request (issue #473).
    """
    _model_deployment_get(va).return_value = "edge"
    # Base config has no reasoning key — the model does not support thinking.
    va.entry.runtime_data.vision_model.config = {"configurable": {}}

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
    assert "reasoning" not in call_cfg["configurable"]


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
    """Edge summarization forces reasoning=False for thinking-capable models."""
    _model_deployment_get(va).return_value = "edge"
    # Base config carries a reasoning key (reasoning_field() for thinking models).
    va.entry.runtime_data.summarization_model.config = {
        "configurable": {"reasoning": True}
    }

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


@pytest.mark.asyncio
async def test_summary_omits_reasoning_for_non_thinking_model(
    va: VideoAnalyzer,
) -> None:
    """Edge summarization must not inject a reasoning key for non-thinking models."""
    _model_deployment_get(va).return_value = "edge"
    # Base config has no reasoning key — the model does not support thinking.
    va.entry.runtime_data.summarization_model.config = {"configurable": {}}

    frame_descs = [
        {"Scene A.": []},
        {"Scene B.": []},
    ]
    await va._generate_summary(frame_descs)  # type: ignore[attr-defined]

    call_cfg = va.entry.runtime_data.summarization_model.with_config.call_args.kwargs[
        "config"
    ]
    assert "reasoning" not in call_cfg["configurable"]


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


def test_motion_scan_interval_exceeds_recording_interval() -> None:
    """Motion loop interval exceeds recording-camera poll interval."""
    assert VIDEO_ANALYZER_MOTION_SCAN_INTERVAL > VIDEO_ANALYZER_SCAN_INTERVAL


@pytest.mark.asyncio
async def test_recording_poll_skips_motion_tracked_cameras(
    hass: MagicMock, entry: MagicMock
) -> None:
    """_take_snapshots_from_recording_cameras skips cameras already in the motion loop."""
    va = _make_va_with_states(hass, entry, ["camera.frontgate"])
    va._active_motion_cameras["camera.frontgate"] = MagicMock()  # type: ignore[attr-defined]

    hass.states.async_all.return_value = [
        MagicMock(entity_id="camera.frontgate", state="recording"),
    ]

    with patch.object(va, "_take_single_snapshot") as mock_snap:
        await va._take_snapshots_from_recording_cameras(  # type: ignore[attr-defined]
            datetime.now(tz=dt.UTC)
        )
        mock_snap.assert_not_called()


# ---------------------------------------------------------------------------
# Event lifecycle: held snapshots flush as one batch at motion OFF
# ---------------------------------------------------------------------------


def _prepare_snapshot_capture(va: VideoAnalyzer, tmp_path: Path) -> AsyncMock:
    """Stub HA service calls and snapshot dir so _take_single_snapshot runs."""
    mock_hass = cast("MagicMock", va.hass)
    mock_hass.services.async_call = AsyncMock()
    mock_hass.async_add_executor_job = AsyncMock(return_value=True)
    return AsyncMock(return_value=tmp_path)


@pytest.mark.asyncio
async def test_held_snapshots_not_analyzed_until_flush(
    va: VideoAnalyzer, tmp_path: Path
) -> None:
    """Motion-held snapshots wait for the event flush; no per-frame analysis."""
    camera_id = "camera.test"
    mock_dir = _prepare_snapshot_capture(va, tmp_path)
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.UTC)

    with (
        patch.object(va, "_get_snapshot_dir", new=mock_dir),
        patch.object(va, "_analyze_and_finalize", new=AsyncMock()) as mock_analyze,
    ):
        for i in range(3):
            now = base + dt.timedelta(seconds=3 * i)
            path = await va._take_single_snapshot(  # type: ignore[attr-defined]
                camera_id, now, hold_for_batch=True
            )
            assert path is not None

        # While motion is on: nothing analyzed, no live worker spawned.
        mock_analyze.assert_not_called()
        assert camera_id not in va._snapshot_queues  # type: ignore[attr-defined]
        assert len(va._event_snapshot_buffers[camera_id]) == 3  # type: ignore[attr-defined]

        # Motion OFF → flush: exactly one batch containing the whole window.
        await va._process_snapshot_queue(camera_id)  # type: ignore[attr-defined]

        mock_analyze.assert_called_once()
        ordered = mock_analyze.call_args.args[1]
        assert len(ordered) == 3
        epochs = [epoch for _, epoch in ordered]
        assert epochs == sorted(epochs)
        assert camera_id not in va._event_snapshot_buffers  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_motion_loop_holds_snapshots_for_batch(va: VideoAnalyzer) -> None:
    """The motion snapshot loop captures with hold_for_batch=True."""
    seen: list[bool] = []

    async def fake_take(
        _camera_id: str, _now: datetime, *, hold_for_batch: bool = False
    ) -> Path | None:
        seen.append(hold_for_batch)
        raise asyncio.CancelledError

    with patch.object(va, "_take_single_snapshot", new=fake_take):
        await va._motion_snapshot_loop("camera.test")  # type: ignore[attr-defined]

    assert seen == [True]


@pytest.mark.asyncio
async def test_unheld_snapshot_goes_to_live_queue(
    va: VideoAnalyzer, tmp_path: Path
) -> None:
    """Poll-path snapshots (no active loop) still feed the live worker queue."""
    camera_id = "camera.test"
    mock_dir = _prepare_snapshot_capture(va, tmp_path)
    # Close the worker coroutine instead of scheduling it (hass is a MagicMock).
    cast("MagicMock", va.hass).async_create_task = MagicMock(
        side_effect=lambda coro: (coro.close(), MagicMock())[1]  # type: ignore[no-any-return]
    )

    with patch.object(va, "_get_snapshot_dir", new=mock_dir):
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.UTC)
        path = await va._take_single_snapshot(camera_id, now)  # type: ignore[attr-defined]

    assert path is not None
    assert camera_id not in va._event_snapshot_buffers  # type: ignore[attr-defined]
    assert va._snapshot_queues[camera_id].qsize() == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_flush_combines_held_and_queued_frames(va: VideoAnalyzer) -> None:
    """The event flush drains both the hold buffer and queued leftovers."""
    camera_id = "camera.test"
    held = Path("snapshot_20250101_120000.jpg")
    queued = Path("snapshot_20250101_120003.jpg")

    va._event_snapshot_buffers[camera_id] = deque([held])  # type: ignore[attr-defined]
    q: asyncio.Queue[_SnapshotItem] = asyncio.Queue()
    q.put_nowait(_SnapshotItem(path=queued, enqueued=0.0))
    va._snapshot_queues[camera_id] = q  # type: ignore[attr-defined]

    with patch.object(va, "_analyze_and_finalize", new=AsyncMock()) as mock_analyze:
        await va._process_snapshot_queue(camera_id)  # type: ignore[attr-defined]

    mock_analyze.assert_called_once()
    ordered = mock_analyze.call_args.args[1]
    assert [p for p, _ in ordered] == [held, queued]


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


# ---------------------------------------------------------------------------
# Issue #473: un-pulled Ollama model must not kill the snapshot worker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_survives_unexpected_exception(
    va: VideoAnalyzer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ResponseError from analysis drops the batch but keeps the worker alive."""
    camera_id = "camera.test"
    queue: asyncio.Queue[_SnapshotItem] = asyncio.Queue()
    va._snapshot_queues[camera_id] = queue  # type: ignore[attr-defined]
    monkeypatch.setattr(va_mod, "_WORKER_ERROR_BACKOFF_SEC", 0)

    calls: list[object] = []
    first_call = asyncio.Event()
    second_call = asyncio.Event()

    not_found_msg = "model 'ghost:latest' not found"

    async def flaky_analyze(_camera_id: str, ordered: object) -> None:
        calls.append(ordered)
        if len(calls) == 1:
            first_call.set()
            raise OllamaResponseError(not_found_msg, 404)
        second_call.set()

    with patch.object(
        va, "_analyze_and_finalize", new=AsyncMock(side_effect=flaky_analyze)
    ):
        task = asyncio.create_task(va._snapshot_worker(camera_id))  # type: ignore[attr-defined]

        await queue.put(
            _SnapshotItem(path=Path("snapshot_20250101_120000.jpg"), enqueued=1.0)
        )
        await asyncio.wait_for(first_call.wait(), timeout=5)

        # Worker must still be consuming after the exception.
        await queue.put(
            _SnapshotItem(path=Path("snapshot_20250101_120003.jpg"), enqueued=2.0)
        )
        await asyncio.wait_for(second_call.wait(), timeout=5)

        assert not task.done()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert len(calls) == 2


@pytest.mark.asyncio
async def test_worker_exit_removes_queue_entry(va: VideoAnalyzer) -> None:
    """Worker exit clears its queue entry so the next snapshot respawns a worker."""
    camera_id = "camera.test"
    queue: asyncio.Queue[_SnapshotItem] = asyncio.Queue()
    va._snapshot_queues[camera_id] = queue  # type: ignore[attr-defined]

    task = asyncio.create_task(va._snapshot_worker(camera_id))  # type: ignore[attr-defined]
    va._active_queue_tasks[camera_id] = task  # type: ignore[attr-defined]
    await asyncio.sleep(0)  # let the worker start

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert camera_id not in va._snapshot_queues  # type: ignore[attr-defined]
    assert camera_id not in va._active_queue_tasks  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_worker_cleanup_spares_respawned_queue(va: VideoAnalyzer) -> None:
    """A stale worker's cleanup must not remove a respawned worker's entries."""
    camera_id = "camera.test"
    old_queue: asyncio.Queue[_SnapshotItem] = asyncio.Queue()
    va._snapshot_queues[camera_id] = old_queue  # type: ignore[attr-defined]

    task = asyncio.create_task(va._snapshot_worker(camera_id))  # type: ignore[attr-defined]
    await asyncio.sleep(0)  # let the worker capture its queue

    # Simulate a respawn landing before the old worker's cancellation completes.
    new_queue: asyncio.Queue[_SnapshotItem] = asyncio.Queue()
    new_task = MagicMock()
    va._snapshot_queues[camera_id] = new_queue  # type: ignore[attr-defined]
    va._active_queue_tasks[camera_id] = new_task  # type: ignore[attr-defined]

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert va._snapshot_queues[camera_id] is new_queue  # type: ignore[attr-defined]
    assert va._active_queue_tasks[camera_id] is new_task  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_analyze_image_propagates_ollama_response_error() -> None:
    """analyze_image lets ResponseError escape so callers can skip the frame."""
    model = MagicMock()
    model.ainvoke = AsyncMock(
        side_effect=OllamaResponseError("model 'ghost:latest' not found", 404)
    )

    with pytest.raises(OllamaResponseError):
        await analyze_image(model, _FakeImageFile.BYTES, None)


@pytest.mark.asyncio
async def test_camera_tool_swallows_ollama_response_error() -> None:
    """The chat tool returns an error string instead of raising to the graph."""
    model = MagicMock()
    model.ainvoke = AsyncMock(
        side_effect=OllamaResponseError("model 'ghost:latest' not found", 404)
    )
    config = {"configurable": {"hass": MagicMock(), "vlm_model": model}}

    with patch(
        "custom_components.home_generative_agent.agent.tools._get_camera_image",
        new=AsyncMock(return_value=_FakeImageFile.BYTES),
    ):
        result = await get_and_analyze_camera_image.coroutine(  # type: ignore[misc]
            camera_name="camera.test", detection_keywords=None, config=config
        )

    assert result == "Error analyzing image with VLM model."


@pytest.mark.asyncio
async def test_process_snapshot_swallows_ollama_response_error(
    va: VideoAnalyzer,
) -> None:
    """_process_snapshot drops the frame when a ResponseError escapes analysis."""
    _model_deployment_get(va).return_value = "cloud"

    with (
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.epoch_from_path",
            return_value=int(time.time()),
        ),
        patch("aiofiles.open", return_value=_FakeImageFile()),
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.analyze_image",
            new=AsyncMock(
                side_effect=OllamaResponseError("model 'ghost:latest' not found", 404)
            ),
        ),
    ):
        result = await va._process_snapshot(  # type: ignore[attr-defined]
            Path("snapshot_20250101_120000.jpg"), "camera.test"
        )

    assert result == {}
