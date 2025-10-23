"""Helpers for video analyzer."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Final

from ..const import (  # noqa: TID252
    VIDEO_ANALYZER_LATEST_NAME,
    VIDEO_ANALYZER_LATEST_SUBFOLDER,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

LATEST_NAME: Final[str] = VIDEO_ANALYZER_LATEST_NAME
LATEST_SUBFOLDER: Final[str] = VIDEO_ANALYZER_LATEST_SUBFOLDER


def _camera_fs_dir(root: Path, camera_id: str) -> Path:
    """Return filesystem directory for a camera under snapshot root."""
    return root / camera_id.replace(".", "_")


def latest_target(root: Path, camera_id: str) -> Path:
    """Return the target 'latest.jpg' path for a camera under the snapshot root."""
    return _camera_fs_dir(root, camera_id) / LATEST_SUBFOLDER / LATEST_NAME


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
