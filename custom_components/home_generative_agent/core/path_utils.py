"""Path construction utilities for consistent file naming."""

from __future__ import annotations

from pathlib import Path


class PathUtils:
    """Utilities for consistent path construction."""

    @staticmethod
    def sanitize_entity_id(entity_id: str) -> str:
        """
        Convert entity_id to safe filesystem name.

        Args:
            entity_id: Entity ID like "camera.front_door"

        Returns:
            Safe filename like "camera_front_door"

        """
        return entity_id.replace(".", "_")

    @staticmethod
    def snapshot_filename(timestamp_str: str) -> str:
        """
        Generate standard snapshot filename.

        Args:
            timestamp_str: Timestamp in YYYYMMDD_HHMMSS format

        Returns:
            Filename like "snapshot_20250426_002804.jpg"

        """
        return f"snapshot_{timestamp_str}.jpg"

    @staticmethod
    def camera_snapshot_dir(root: Path, camera_id: str) -> Path:
        """
        Get camera-specific snapshot directory.

        Args:
            root: Snapshot root directory
            camera_id: Camera entity ID

        Returns:
            Path like root/camera_front_door

        """
        safe_name = PathUtils.sanitize_entity_id(camera_id)
        return root / safe_name

    @staticmethod
    def face_debug_dir(root: Path, camera_id: str) -> Path:
        """
        Get camera-specific face debug directory.

        Args:
            root: Snapshot root directory
            camera_id: Camera entity ID

        Returns:
            Path like root/faces/camera_front_door

        """
        safe_name = PathUtils.sanitize_entity_id(camera_id)
        return root / "faces" / safe_name

    @staticmethod
    def ensure_dir(path: Path) -> None:
        """
        Create directory if it doesn't exist.

        Args:
            path: Directory path to create

        """
        path.mkdir(parents=True, exist_ok=True)
