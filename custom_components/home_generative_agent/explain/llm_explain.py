"""Optional LLM explanation layer for anomaly findings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding

LOGGER = logging.getLogger(__name__)


class LLMExplainer:
    """Generate non-authoritative explanation text for findings."""

    def __init__(self, model: Any) -> None:
        """Initialize the explainer with an optional LLM model."""
        self._model = model

    async def async_explain(self, finding: AnomalyFinding) -> str | None:
        """Return explanation text or None on failure."""
        if self._model is None:
            return None

        prompt = USER_PROMPT_TEMPLATE.format(
            anomaly_type=finding.type,
            severity=finding.severity,
            evidence=finding.evidence,
            suggested_actions=finding.suggested_actions,
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

        try:
            result = await self._model.ainvoke(messages)
        except (ValueError, TypeError, RuntimeError) as err:
            LOGGER.warning("LLM explanation failed: %s", err)
            return None

        content = getattr(result, "content", None)
        if not content:
            return None
        return str(content)
