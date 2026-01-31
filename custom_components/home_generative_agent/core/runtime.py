"""HGA integration runtime data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from psycopg import AsyncConnection
    from psycopg.rows import DictRow
    from psycopg_pool import AsyncConnectionPool

    from ..audit.store import AuditStore
    from ..explain.llm_explain import LLMExplainer
    from ..notify.actions import ActionHandler
    from ..notify.dispatcher import NotificationDispatcher
    from ..sentinel.engine import SentinelEngine
    from ..sentinel.suppression import SuppressionManager
    from .video_analyzer import VideoAnalyzer


@dataclass
class HGAData:
    """HGA integration data."""

    options: dict[str, Any]
    chat_model: Any
    chat_model_options: dict[str, Any]
    vision_model: Any
    summarization_model: Any
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] | None
    store: Any
    checkpointer: Any
    video_analyzer: VideoAnalyzer
    face_api_url: str
    face_recognition: bool
    person_gallery: Any
    pending_actions: dict[str, dict[str, Any]]
    suppression: SuppressionManager | None
    sentinel: SentinelEngine | None
    notifier: NotificationDispatcher | None
    action_handler: ActionHandler | None
    audit_store: AuditStore | None
    explainer: LLMExplainer | None


type HGAConfigEntry = ConfigEntry[HGAData]
