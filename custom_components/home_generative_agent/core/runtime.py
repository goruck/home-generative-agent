"""HGA integration runtime data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from psycopg import AsyncConnection
    from psycopg.rows import DictRow
    from psycopg_pool import AsyncConnectionPool

    from custom_components.home_generative_agent.audit.store import AuditStore
    from custom_components.home_generative_agent.explain.llm_explain import LLMExplainer
    from custom_components.home_generative_agent.notify.actions import ActionHandler
    from custom_components.home_generative_agent.notify.dispatcher import (
        NotificationDispatcher,
    )
    from custom_components.home_generative_agent.sentinel.discovery_engine import (
        SentinelDiscoveryEngine,
    )
    from custom_components.home_generative_agent.sentinel.discovery_store import (
        DiscoveryStore,
    )
    from custom_components.home_generative_agent.sentinel.engine import SentinelEngine
    from custom_components.home_generative_agent.sentinel.proposal_store import (
        ProposalStore,
    )
    from custom_components.home_generative_agent.sentinel.rule_registry import (
        RuleRegistry,
    )
    from custom_components.home_generative_agent.sentinel.suppression import (
        SuppressionManager,
    )

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
    discovery_store: DiscoveryStore | None
    discovery_engine: SentinelDiscoveryEngine | None
    proposal_store: ProposalStore | None
    rule_registry: RuleRegistry | None


type HGAConfigEntry = ConfigEntry[HGAData]
