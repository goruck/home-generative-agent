"""HGA integration runtime data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres import AsyncPostgresStore
    from psycopg import AsyncConnection
    from psycopg.rows import DictRow
    from psycopg_pool import AsyncConnectionPool

    from .video_analyzer import VideoAnalyzer


@dataclass
class HGAData:
    """HGA integration data."""

    chat_model: Any
    chat_model_options: dict[str, Any]
    vision_model: Any
    summarization_model: Any
    pool: AsyncConnectionPool[AsyncConnection[DictRow]]
    store: AsyncPostgresStore
    checkpointer: AsyncPostgresSaver
    video_analyzer: VideoAnalyzer
    face_api_url: str
    face_recognition: bool
    person_gallery: Any


type HGAConfigEntry = ConfigEntry[HGAData]
