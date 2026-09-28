"""
Event-recording frame ingest for ring-mqtt cameras (issue #491).

Battery Ring cameras refresh their interval snapshot at best every 600 s, so a
capture window opened on an ``eventId`` change structurally cannot deliver
imagery *of the event*: the newest frame can predate it by up to ten minutes.
The one source of actual event imagery on this device class is the event
recording itself, which ring-mqtt exposes as a signed ``recordingUrl``
attribute on the same ``select.*_event_select`` entity that carries the
``eventId``. Field data (n=8, #491): the URL is present the instant the
``eventId`` flips and the MP4 is fetchable 0.4-1.1 s later.

This module holds the pieces that do not need the analyzer's state: reading
the URL off a state object, downloading the clip with a size cap, extracting
frames with the ffmpeg binary Home Assistant ships, and thinning them to a
bounded count. The analyzer owns scheduling, dedupe, and where the frames go.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from typing import TYPE_CHECKING, Any, Final

import aiofiles
import httpx

if TYPE_CHECKING:
    from pathlib import Path

    from homeassistant.core import HomeAssistant, State

LOGGER = logging.getLogger(__name__)

# --- Tuning ---
# The download is retried on "not there yet" responses only. Field data puts
# availability at ~1 s; the schedule below tolerates a slower Ring backend (or
# a slower transcode on longer clips) for about 15 s before giving up.
RECORDING_FETCH_BACKOFF_SEC: Final[tuple[float, ...]] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
# Ring clips are a few MB; a 64 MiB cap bounds a pathological (or hostile)
# response without cutting any real recording short.
RECORDING_MAX_BYTES: Final[int] = 64 * 1024 * 1024
RECORDING_DOWNLOAD_TIMEOUT_SEC: Final[float] = 60.0
RECORDING_FFMPEG_TIMEOUT_SEC: Final[float] = 60.0
# One frame per second of clip, then thinned to at most this many frames so a
# long (subscription-length) recording costs a bounded number of VLM calls.
RECORDING_FRAME_FPS: Final[int] = 1
RECORDING_MAX_FRAMES: Final[int] = 8
_RECORDING_URL_ATTR: Final[str] = "recordingUrl"
# Home Assistant's ffmpeg integration stores its manager under this hass.data
# key (homeassistant.components.ffmpeg.DATA_FFMPEG). Read by literal rather
# than imported: the component is not in default_config, and importing it on
# an install where it was never set up would pull its `ha-ffmpeg`
# requirement, which may be absent outside the official image.
_DATA_FFMPEG: Final[str] = "ffmpeg"
_DEFAULT_FFMPEG_BINARY: Final[str] = "ffmpeg"
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({403, 404, 409, 423, 425, 429})
_FFMPEG_STDERR_TAIL: Final[int] = 400


class RecordingError(Exception):
    """The recording could not be turned into frames; not worth retrying."""


class RecordingNotReadyError(RecordingError):
    """The recording is not (yet) fetchable; the caller may retry."""


def recording_url_from_state(state: State | None) -> str | None:
    """
    Return the signed recording URL carried by an event_select state, if any.

    Only ``https`` URLs are accepted: the signature travels in the query
    string, so a plaintext scheme would leak it, and nothing ring-mqtt
    publishes is ever plain http. Missing or empty means the camera has no
    recording for this event (typically: no Ring Protect subscription).
    """
    if state is None:
        return None
    url = state.attributes.get(_RECORDING_URL_ATTR)
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url.lower().startswith("https://"):
        return None
    return url


def ffmpeg_binary(hass: HomeAssistant) -> str:
    """
    Return the ffmpeg binary to run.

    Prefers the path the ffmpeg integration was configured with (it honours a
    user's ``ffmpeg_bin`` override) and falls back to ``ffmpeg`` on PATH,
    which the official Home Assistant image provides.
    """
    manager: Any = hass.data.get(_DATA_FFMPEG)
    binary = getattr(manager, "binary", None)
    if isinstance(binary, str) and binary:
        return binary
    return _DEFAULT_FFMPEG_BINARY


def ffmpeg_available(binary: str) -> bool:
    """Return True when the binary resolves to an executable."""
    return shutil.which(binary) is not None


async def download_recording(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    *,
    max_bytes: int = RECORDING_MAX_BYTES,
) -> int:
    """
    Stream the recording at ``url`` to ``dest`` and return the byte count.

    Raises RecordingNotReadyError on responses that mean "not yet" (403/404 and
    friends: the signed object may not have landed at the CDN), and
    RecordingError on anything else, including a body over ``max_bytes``.
    """
    written = 0
    try:
        async with (
            client.stream(
                "GET",
                url,
                timeout=RECORDING_DOWNLOAD_TIMEOUT_SEC,
                follow_redirects=True,
            ) as resp,
            aiofiles.open(dest, "wb") as out,
        ):
            if resp.status_code in _RETRYABLE_STATUS:
                msg = f"HTTP {resp.status_code}"
                raise RecordingNotReadyError(msg)
            if resp.status_code >= 400:  # noqa: PLR2004
                msg = f"HTTP {resp.status_code}"
                raise RecordingError(msg)
            declared = resp.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                msg = f"recording is {declared} bytes (limit {max_bytes})"
                raise RecordingError(msg)
            async for chunk in resp.aiter_bytes():
                written += len(chunk)
                if written > max_bytes:
                    msg = f"recording exceeds {max_bytes} bytes"
                    raise RecordingError(msg)
                await out.write(chunk)
    except httpx.HTTPError as err:
        msg = f"download failed: {err.__class__.__name__}: {err}"
        raise RecordingNotReadyError(msg) from err
    if written == 0:
        msg = "download returned an empty body"
        raise RecordingNotReadyError(msg)
    return written


def _list_frames(out_dir: Path) -> list[Path]:
    """Return the extracted frames in clip order (blocking; run off-loop)."""
    return sorted(out_dir.glob("frame_*.jpg"))


async def extract_frames(
    binary: str,
    recording: Path,
    out_dir: Path,
    *,
    fps: int = RECORDING_FRAME_FPS,
) -> list[Path]:
    """
    Decode ``recording`` to JPEG frames at ``fps`` in ``out_dir``.

    Returns the frame paths in clip order (ffmpeg numbers them from 1). Raises
    RecordingError when ffmpeg fails, times out, or produces no frame.
    """
    pattern = out_dir / "frame_%04d.jpg"
    cmd = (
        binary,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(recording),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "3",
        "-f",
        "image2",
        str(pattern),
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as err:
        msg = f"could not start {binary}: {err}"
        raise RecordingError(msg) from err
    try:
        async with asyncio.timeout(RECORDING_FFMPEG_TIMEOUT_SEC):
            _, stderr = await proc.communicate()
    except TimeoutError as err:
        proc.kill()
        await proc.wait()
        msg = f"{binary} did not finish within {RECORDING_FFMPEG_TIMEOUT_SEC:.0f} s"
        raise RecordingError(msg) from err
    if proc.returncode != 0:
        tail = stderr.decode(errors="replace").strip()[-_FFMPEG_STDERR_TAIL:]
        msg = f"{binary} exited {proc.returncode}: {tail or 'no stderr'}"
        raise RecordingError(msg)
    frames = await asyncio.to_thread(_list_frames, out_dir)
    if not frames:
        msg = f"{binary} produced no frames"
        raise RecordingError(msg)
    return frames


def thin_frames[T](
    frames: list[T], max_frames: int = RECORDING_MAX_FRAMES
) -> list[tuple[int, T]]:
    """
    Keep at most ``max_frames`` evenly spaced frames, first and last included.

    Returns ``(index, frame)`` pairs so the caller can keep each frame's
    position in the clip (its second, at 1 fps) for timestamping.
    """
    if max_frames <= 0 or not frames:
        return []
    n = len(frames)
    if n <= max_frames:
        return list(enumerate(frames))
    if max_frames == 1:
        return [(0, frames[0])]
    step = (n - 1) / (max_frames - 1)
    indices = sorted({round(i * step) for i in range(max_frames)})
    return [(i, frames[i]) for i in indices]
