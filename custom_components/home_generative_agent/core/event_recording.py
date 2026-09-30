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
the URL off a state object, checking where it points, downloading the clip
with a size cap, extracting frames with the ffmpeg binary Home Assistant
ships, and thinning them to a bounded count. The analyzer owns scheduling,
dedupe, and where the frames go.

Trust boundary: ``eventId`` and ``recordingUrl`` arrive over MQTT, so nothing
here treats them as safe. The event id never reaches a filesystem path, and
the URL is fetched only when it is https, names a host (not an address), and
that host resolves to public addresses only. Ring's signed URL is a
transcoding endpoint that 302-redirects to a node-specific subdomain of the
same host (field data, #491), so redirects are followed — but only to https
targets on the same host or a subdomain of it, each vetted the same way, and
at most RECORDING_MAX_REDIRECTS hops.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import shutil
import socket
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urljoin, urlsplit

import aiofiles
import httpx
from homeassistant.util.network import is_invalid, is_ip_address, is_local

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
# Wall-clock bound on one download attempt (httpx's own timeout is per read,
# so a trickling server would otherwise hold the camera's lock indefinitely).
RECORDING_DOWNLOAD_TIMEOUT_SEC: Final[float] = 60.0
RECORDING_FFMPEG_TIMEOUT_SEC: Final[float] = 60.0
# Bound on one whole ingest (all retries + decode + placement); the analyzer
# wraps the task in it so a stuck ingest can never pin a camera.
RECORDING_INGEST_DEADLINE_SEC: Final[float] = 180.0
# How long a closing capture window waits for its own recording before it
# flushes the snapshots alone. Battery-cam clips land ~5 s after the window
# closes (field data, issue #491); flushing first analyzed (and notified on)
# the retained snapshot, which predates the event. Covers the not-ready retry
# schedule plus a typical download and decode; a slower clip still arrives,
# as its own batch, after the snapshot fallback.
RECORDING_FLUSH_GRACE_SEC: Final[float] = 20.0
# One frame per second of clip, then thinned to at most this many frames so a
# long (subscription-length) recording costs a bounded number of VLM calls.
RECORDING_FRAME_FPS: Final[int] = 1
RECORDING_MAX_FRAMES: Final[int] = 8
# Decode budget: Ring recordings are at most 120 s, so `-t` at 180 s never
# truncates a real clip but stops a hostile hour-long one; frames are scaled
# down to this width so the transient JPEGs stay a few hundred KB each.
RECORDING_MAX_CLIP_SEC: Final[int] = 180
RECORDING_MAX_WIDTH: Final[int] = 1280
# Concurrency: ffmpeg decoders running at once across all cameras, and how
# many ingests may be in flight per camera before new events are dropped
# (counted as failures, so the hourly metrics show it).
RECORDING_MAX_CONCURRENT: Final[int] = 2
RECORDING_MAX_PENDING_PER_CAMERA: Final[int] = 3
# mkdtemp prefix for the per-ingest work dir inside the camera snapshot dir.
# The retention seed removes any such dir left by a crash or shutdown.
EVENT_WORK_PREFIX: Final[str] = "_event_"
# Ring event ids are 19-digit decimals; anything outside this generous shape
# is not an event this code understands and is never logged raw.
EVENT_ID_RE: Final = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_RECORDING_URL_ATTR: Final[str] = "recordingUrl"
# Home Assistant's ffmpeg integration stores its manager under this hass.data
# key (homeassistant.components.ffmpeg.DATA_FFMPEG). Read by literal rather
# than imported: the component is not in default_config, and importing it on
# an install where it was never set up would pull its `ha-ffmpeg`
# requirement, which may be absent outside the official image.
_DATA_FFMPEG: Final[str] = "ffmpeg"
_DEFAULT_FFMPEG_BINARY: Final[str] = "ffmpeg"
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({403, 404, 409, 423, 425, 429})
# Ring's `req_type=TranscodingRequest` URL answers 302 to
# `<edge-node>.<same host>/<same path>`; one hop is the observed shape, three
# leaves room for a CDN change without letting a redirect chain run away.
RECORDING_MAX_REDIRECTS: Final[int] = 3
_HTTP_REDIRECT_MIN: Final[int] = 300
_HTTP_ERROR_MIN: Final[int] = 400
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


def is_valid_event_id(event_id: str) -> bool:
    """Return True for an event id shaped like something Ring would publish."""
    return EVENT_ID_RE.fullmatch(event_id) is not None


async def assert_public_host(url: str) -> None:
    """
    Refuse a URL whose host is an address literal or resolves to a local one.

    The URL is attacker-influenced (it arrives over MQTT), and Home Assistant
    sits on the LAN, so a fetch must never become a probe of the LAN or of
    the host itself. Ring's download links are always hostnames on public
    CDNs, so an address literal is rejected outright. A resolution failure is
    retryable (DNS hiccup); a local or invalid answer is final. The check runs
    before every attempt; a rebinding between check and fetch would need an
    attacker who controls both the MQTT attribute and the resolver.
    """
    try:
        host = urlsplit(url).hostname
    except ValueError as err:
        msg = f"recording URL is malformed: {err}"
        raise RecordingError(msg) from err
    if not host:
        msg = "recording URL has no host"
        raise RecordingError(msg)
    if is_ip_address(host):
        msg = f"recording URL names an address literal ({host}), not a host"
        raise RecordingError(msg)
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            host, 443, type=socket.SOCK_STREAM
        )
    except socket.gaierror as err:
        msg = f"could not resolve {host}: {err}"
        raise RecordingNotReadyError(msg) from err
    addresses = {ipaddress.ip_address(info[4][0]) for info in infos}
    if not addresses:
        msg = f"{host} resolved to no address"
        raise RecordingNotReadyError(msg)
    for address in addresses:
        if (
            is_local(address)
            or is_invalid(address)
            or address.is_multicast
            or address.is_reserved
        ):
            msg = f"{host} resolves to a non-public address ({address})"
            raise RecordingError(msg)


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
    """Return True when the binary resolves to an executable (blocking)."""
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

    Redirects are followed by hand rather than by httpx, so every hop is
    held to the same rule as the signed URL: https, same host or a subdomain
    of the host the redirect came from, and a public resolution — Ring's
    transcoding endpoint 302s to ``<edge-node>.<same host>`` (field data,
    #491), and nothing else is a shape this code should follow. Raises
    RecordingNotReadyError on responses that mean "not yet" (403/404 and
    friends: the signed object may not have landed at the CDN), on transport
    errors, and on an empty body; RecordingError on anything else, including
    a malformed URL, a redirect off-host or over the hop budget, a body over
    ``max_bytes``, or an attempt over the wall-clock bound.
    """
    written = 0
    try:
        async with asyncio.timeout(RECORDING_DOWNLOAD_TIMEOUT_SEC):
            for hop in range(RECORDING_MAX_REDIRECTS + 1):
                next_url: str | None = None
                async with client.stream("GET", url, follow_redirects=False) as resp:
                    status = resp.status_code
                    if status in _RETRYABLE_STATUS:
                        msg = f"HTTP {status}"
                        raise RecordingNotReadyError(msg)
                    if _HTTP_REDIRECT_MIN <= status < _HTTP_ERROR_MIN:
                        next_url = _redirect_target(url, resp.headers.get("location"))
                    elif status >= _HTTP_ERROR_MIN:
                        msg = f"HTTP {status}"
                        raise RecordingError(msg)
                    else:
                        written = await _write_body(resp, dest, max_bytes)
                if next_url is None:
                    break
                if hop >= RECORDING_MAX_REDIRECTS:
                    msg = f"more than {RECORDING_MAX_REDIRECTS} redirects"
                    raise RecordingError(msg)
                # Vet the hop before fetching it (outside the response
                # context so the redirect connection is released first).
                await assert_public_host(next_url)
                url = next_url
    except TimeoutError as err:
        msg = f"download exceeded {RECORDING_DOWNLOAD_TIMEOUT_SEC:.0f} s"
        raise RecordingError(msg) from err
    except (httpx.InvalidURL, ValueError) as err:
        msg = f"recording URL is malformed: {err}"
        raise RecordingError(msg) from err
    except httpx.HTTPError as err:
        msg = f"download failed: {err.__class__.__name__}: {err}"
        raise RecordingNotReadyError(msg) from err
    if written == 0:
        msg = "download returned an empty body"
        raise RecordingNotReadyError(msg)
    return written


async def _write_body(resp: httpx.Response, dest: Path, max_bytes: int) -> int:
    """Stream a 2xx body to ``dest`` under the size cap; return bytes written."""
    declared = resp.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        msg = f"recording is {declared} bytes (limit {max_bytes})"
        raise RecordingError(msg)
    written = 0
    async with aiofiles.open(dest, "wb") as out:
        async for chunk in resp.aiter_bytes():
            written += len(chunk)
            if written > max_bytes:
                msg = f"recording exceeds {max_bytes} bytes"
                raise RecordingError(msg)
            await out.write(chunk)
    return written


def _redirect_target(current: str, location: str | None) -> str:
    """
    Resolve and vet a redirect ``Location`` against the URL that sent it.

    The target must be https and its host must be the current host or a
    subdomain of it (``a.b.example.dev`` for ``b.example.dev``); anything
    else — another domain, a plaintext scheme, a missing header — is a final
    error, because the signed URL never legitimately points elsewhere.
    """
    if not location:
        msg = "redirect without a Location header"
        raise RecordingError(msg)
    target = urljoin(current, location.strip())
    try:
        cur, new = urlsplit(current), urlsplit(target)
    except ValueError as err:
        msg = f"redirect target is malformed: {err}"
        raise RecordingError(msg) from err
    cur_host = (cur.hostname or "").lower()
    new_host = (new.hostname or "").lower()
    if new.scheme.lower() != "https":
        msg = f"redirect to a non-https URL ({new.scheme or 'no scheme'})"
        raise RecordingError(msg)
    if not cur_host or not new_host:
        msg = "redirect without a host"
        raise RecordingError(msg)
    if new_host != cur_host and not new_host.endswith("." + cur_host):
        msg = f"redirect leaves the host ({cur_host} -> {new_host})"
        raise RecordingError(msg)
    return target


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

    Decoding stops at RECORDING_MAX_CLIP_SEC and frames are scaled to at most
    RECORDING_MAX_WIDTH wide, so the transient output is bounded whatever the
    clip claims to be. The child is killed and reaped on timeout and on
    cancellation, so a stop() mid-decode never leaves an ffmpeg writing into
    a directory the caller is about to remove.

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
        "-t",
        str(RECORDING_MAX_CLIP_SEC),
        "-i",
        str(recording),
        "-vf",
        f"fps={fps},scale='min({RECORDING_MAX_WIDTH},iw)':-2",
        "-frames:v",
        str(RECORDING_MAX_CLIP_SEC * fps),
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
        await _reap(proc)
        msg = f"{binary} did not finish within {RECORDING_FFMPEG_TIMEOUT_SEC:.0f} s"
        raise RecordingError(msg) from err
    except asyncio.CancelledError:
        await _reap(proc)
        raise
    if proc.returncode != 0:
        tail = stderr.decode(errors="replace").strip()[-_FFMPEG_STDERR_TAIL:]
        msg = f"{binary} exited {proc.returncode}: {tail or 'no stderr'}"
        raise RecordingError(msg)
    frames = await asyncio.to_thread(_list_frames, out_dir)
    if not frames:
        msg = f"{binary} produced no frames"
        raise RecordingError(msg)
    return frames


async def _reap(proc: asyncio.subprocess.Process) -> None:
    """Kill a still-running child and wait for it (never raises)."""
    if proc.returncode is not None:
        return
    try:
        proc.kill()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except TimeoutError:
        LOGGER.warning("ffmpeg (pid %s) did not exit after kill", proc.pid)


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
