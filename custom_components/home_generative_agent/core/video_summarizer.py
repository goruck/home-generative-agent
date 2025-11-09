"""Multi-frame video summarization using LLM."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import async_timeout
from langchain_core.messages import HumanMessage, SystemMessage

from ..const import (  # noqa: TID252
    REASONING_DELIMITERS,
    VIDEO_ANALYZER_PROMPT,
    VIDEO_ANALYZER_SYSTEM_MESSAGE,
)

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger(__name__)


class VideoSummarizer:
    """Generates multi-frame summaries using LLM."""

    def __init__(self, summarization_model: Any, timeout: int = 60) -> None:
        """Initialize video summarizer.

        Args:
            summarization_model: LLM model for summarization
            timeout: Timeout in seconds for summary generation
        """
        self.model = summarization_model
        self._timeout = timeout

    async def generate_summary(
        self, frame_descriptions: list[dict[str, list[str]]]
    ) -> str:
        """Create narrative summary from frame descriptions.

        Args:
            frame_descriptions: List of {description: [people]} dicts

        Returns:
            Summary text

        Raises:
            ValueError: If frame_descriptions is empty
        """
        await asyncio.sleep(0)  # Yield control

        if not frame_descriptions:
            msg = "At least one frame description required."
            raise ValueError(msg)

        # Build prompt with frame descriptions and person identities
        ftag = "\n<frame description>\n{}\n</frame description>"
        ptag = "\n<person identity>\n{}\n</person identity>"

        prompt_parts = [VIDEO_ANALYZER_PROMPT]
        for entry in frame_descriptions:
            for frame, people in entry.items():
                frame_part = ftag.format(frame)
                people_part = "".join(ptag.format(p) for p in people)
                prompt_parts.append(frame_part + people_part)

        prompt = " ".join(prompt_parts)
        LOGGER.debug("Summary prompt: %s", prompt)

        # Generate summary
        messages = [
            SystemMessage(content=VIDEO_ANALYZER_SYSTEM_MESSAGE),
            HumanMessage(content=prompt),
        ]

        response = await self.model.ainvoke(messages)
        LOGGER.debug("Video summary response: %s", response)

        # Extract summary content, removing any reasoning delimiters
        content = response.content
        first, sep, last = content.partition(REASONING_DELIMITERS.get("end", ""))
        summary = (last if sep else first).strip("\n")

        return summary

    async def summarize_with_timeout(
        self, camera_id: str, frame_descriptions: list[dict[str, list[str]]]
    ) -> str | None:
        """Generate summary with timeout protection.

        Args:
            camera_id: Camera identifier for logging
            frame_descriptions: List of {description: [people]} dicts

        Returns:
            Summary text, or None if timeout/failure
        """
        if not frame_descriptions:
            return None

        try:
            async with async_timeout.timeout(self._timeout):
                summary = await self.generate_summary(frame_descriptions)
            LOGGER.info("[%s] Video analysis: %s", camera_id, summary)
            return summary
        except TimeoutError:
            LOGGER.warning(
                "[%s] Summary generation timed out after %ds",
                camera_id,
                self._timeout,
            )
            return None
        except ValueError as err:
            LOGGER.warning("[%s] Summary generation failed: %s", camera_id, err)
            return None
