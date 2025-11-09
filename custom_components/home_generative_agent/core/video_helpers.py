"""Helpers for video analyzer."""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import re
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Final

from homeassistant.util import dt as dt_util
from PIL import Image

from ..const import (  # noqa: TID252
    VIDEO_ANALYZER_LATEST_NAME,
    VIDEO_ANALYZER_LATEST_SUBFOLDER,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Video snapshot naming constants
_LATEST_NAME: Final[str] = VIDEO_ANALYZER_LATEST_NAME
_LATEST_SUBFOLDER: Final[str] = VIDEO_ANALYZER_LATEST_SUBFOLDER

# dHash parameters
_UNIQUENESS_HASH_SIZE: Final[int] = 8  # dHash grid size (8 -> 64-bit)

# Single-frame heuristic helpers
_HUMAN_TERMS = {"person", "people", "man", "woman", "boy", "girl", "child", "children"}
_NEG_IDENTS = {"Indeterminate", "Unknown Person", ""}
_MAX_SENTENCES = 2
_MAX_CHARS = 300
_MAX_NAMES = 2


def load_image_rgb(buf: bytes) -> Image.Image:
    """
    Decode bytes into an RGB Pillow Image.

    Keep this sync so it can be offloaded with hass.async_add_executor_job.
    """
    return Image.open(io.BytesIO(buf)).convert("RGB")


def ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if missing."""
    path.mkdir(parents=True, exist_ok=True)


def crop_resize_encode_jpeg(
    img: Image.Image,
    bbox: Sequence[int],
    pad: float,
    min_px: int,
) -> bytes | None:
    """
    Crop a padded face bbox, ensure minimum size, and return JPEG bytes.

    Args:
        img: Source RGB image.
        bbox: [x1, y1, x2, y2] rectangle (inclusive-exclusive semantics not assumed).
        pad: Fractional padding to apply on each side (e.g., 0.3 == 30%).
        min_px: Minimum width/height of the output; will upscale if smaller.

    Returns:
        JPEG bytes if crop succeeds; otherwise None.

    """
    num_args = 4
    if len(bbox) != num_args:
        return None

    x1, y1, x2, y2 = (int(v) for v in bbox)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None

    dx, dy = int(w * pad), int(h * pad)
    x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
    x2, y2 = min(img.width, x2 + dx), min(img.height, y2 + dy)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = img.crop((x1, y1, x2, y2))
    if crop.width < min_px or crop.height < min_px:
        crop = crop.resize((min_px, min_px), resample=Image.Resampling.LANCZOS)

    out = io.BytesIO()
    crop.save(out, format="JPEG", quality=95, subsampling=0)
    return out.getvalue()


def dedupe_desc(descs: list[dict[str, list[str]]]) -> list[dict[str, list[str]]]:
    """Collapse near-duplicate frame texts to reduce prompt size."""
    out: list[dict[str, list[str]]] = []
    last_norm: str | None = None
    for d in descs:
        # dict has single key
        text = next(iter(d.keys()))
        norm = re.sub(r"\s+", " ", text.lower()).strip()
        if norm != last_norm:
            out.append(d)
            last_norm = norm
    return out


def hamming64(a: int, b: int) -> int:
    """Hamming distance for up to 64-bit integers."""
    x = (a ^ b) & ((1 << 64) - 1)
    # builtin bit_count is fast in Py3.8+
    return x.bit_count()


def dhash_bytes(buf: bytes, size: int = _UNIQUENESS_HASH_SIZE) -> int:
    """
    Compute 64-bit (size=8) or larger dHash from JPEG/PNG bytes using PIL only.

    dHash compares adjacent pixels horizontally.
    """
    with Image.open(io.BytesIO(buf)) as img:
        im = img.convert("L")  # grayscale
        # Resize to (size+1, size) to have adjacent pairs horizontally
        im = im.resize((size + 1, size), Image.Resampling.LANCZOS)
        pixels = im.getdata()
        # Build bitstring by comparing horizontally
        bits = 0
        bitpos = 0
        width = size + 1
        for y in range(size):
            row_off = y * width
            for x in range(size):
                left = pixels[row_off + x]
                right = pixels[row_off + x + 1]
                if left > right:  # set bit if left > right
                    bits |= 1 << bitpos
                bitpos += 1
        return bits  # up to size*size bits; with size=8 it's 64-bit


def epoch_from_path(path: Path) -> int:
    """Extract epoch seconds from snapshot filename."""
    s = path.stem.removeprefix("snapshot_")  # "YYYYMMDD_HHMMSS"
    y, mo, d = int(s[0:4]), int(s[4:6]), int(s[6:8])
    hh, mm, ss = int(s[9:11]), int(s[11:13]), int(s[13:15])

    # Filename was created with dt_util.as_local(...), so attach local tz
    dt_local = datetime(y, mo, d, hh, mm, ss, tzinfo=dt_util.DEFAULT_TIME_ZONE)
    # Return UTC epoch seconds
    return int(dt_util.as_timestamp(dt_local))


def order_batch(batch: list[Path]) -> list[tuple[Path, int]]:
    """Order snapshot paths by epoch extracted from filename."""
    return sorted(((p, epoch_from_path(p)) for p in batch), key=lambda x: x[1])


async def put_with_backpressure(q: asyncio.Queue[Path], p: Path) -> None:
    """Put item into queue, dropping oldest if full."""
    if q.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            _ = q.get_nowait()
            q.task_done()
        _LOGGER.debug("Queue full; dropped oldest to enqueue %s", p)
    await q.put(p)


def _camera_fs_dir(root: Path, camera_id: str) -> Path:
    """Return filesystem directory for a camera under snapshot root."""
    return root / camera_id.replace(".", "_")


def latest_target(root: Path, camera_id: str) -> Path:
    """Return the target 'latest.jpg' path for a camera under the snapshot root."""
    return _camera_fs_dir(root, camera_id) / _LATEST_SUBFOLDER / _LATEST_NAME


async def publish_latest_atomic(hass: HomeAssistant, src: Path, dst: Path) -> None:
    """Atomically copy a snapshot from src to dst."""

    def _copy() -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with (
            src.open("rb") as f_in,
            NamedTemporaryFile(dir=dst.parent, delete=False) as tmp,
        ):
            tmp.write(f_in.read())
            tmp.flush()
            Path(tmp.name).replace(dst)

    await hass.async_add_executor_job(_copy)


async def mirror_to_www(hass: HomeAssistant, src: Path, camera_id: str) -> Path:
    """Copy src to /config/www/hga_notify/<camera>/latest.jpg."""
    dest = Path(hass.config.path("www")) / "hga_notify" / camera_id.replace(".", "_")
    target = dest / "latest.jpg"

    def _copy() -> None:
        dest.mkdir(parents=True, exist_ok=True)
        data = src.read_bytes()
        target.write_bytes(data)

    await hass.async_add_executor_job(_copy)
    return target


def www_notify_path(hass: HomeAssistant, camera_id: str) -> Path:
    """Resolve /config/www location for the camera's latest.jpg."""
    slug = camera_id.replace(".", "_")
    base = Path(hass.config.path("www")) / "hga_notify" / slug
    return base / "latest.jpg"


# --- Single-frame heuristic helpers ---


def clean_frame_text(text: str) -> str:
    """Remove timestamps/labels and collapse whitespace."""
    s = re.sub(r"\bt\+\d+s\.\s*", "", text, flags=re.IGNORECASE)  # drop "t+3s." etc.
    s = re.sub(
        r"\b(frame|camera)\s*\d*:?\s*", "", s, flags=re.IGNORECASE
    )  # drop "Frame 1:"
    return re.sub(r"\s+", " ", s).strip()


def limit_sentences_and_chars(
    text: str,
    *,
    max_chars: int = _MAX_CHARS,
    max_sentences: int = _MAX_SENTENCES,
) -> str:
    """Split on sentence enders, keep up to 2 sentences, then enforce char cap."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    clipped = " ".join(parts[:max_sentences]).strip()
    return clipped[:max_chars].rstrip(" ,;")


def has_human_terms(text: str) -> bool:
    """Return True if text contains any human-related terms."""
    lt = text.lower()
    return any(term in lt for term in _HUMAN_TERMS)


def format_subject(identities: Iterable[str], text: str) -> str | None:
    """Return subject phrase per rules, else None if no human present."""
    idents = [i for i in identities if i not in _NEG_IDENTS]
    if idents:
        # Up to two known names
        uniq = []
        for i in idents:
            if i not in uniq:
                uniq.append(i)
            if len(uniq) == _MAX_NAMES:
                break
        return " and ".join(uniq)
    # Unknown but human mentioned?
    if has_human_terms(text) or any(i == "Unknown Person" for i in identities):
        return "a person"
    return None


def apply_name_substitution(text: str, subject: str, *, had_known_name: bool) -> str:
    """Swap generic 'the/a person/man/woman' with the known name when possible."""
    if not had_known_name:
        return text
    # Replace only the first generic mention to keep phrasing natural
    pattern = r"\b(?:the|a)\s+(?:person|man|woman)\b"
    return re.sub(pattern, subject, text, count=1, flags=re.IGNORECASE)
