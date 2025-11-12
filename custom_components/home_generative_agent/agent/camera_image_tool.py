"""Camera image analysis tool for Home Generative Agent."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import TYPE_CHECKING, Annotated

from homeassistant.components import camera
from homeassistant.exceptions import HomeAssistantError
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import InjectedToolArg, tool

from ..const import (  # noqa: TID252
    VLM_IMAGE_HEIGHT,
    VLM_IMAGE_WIDTH,
    VLM_SYSTEM_PROMPT,
    VLM_USER_KW_TEMPLATE,
    VLM_USER_PROMPT,
)
from ..core.utils import extract_final  # noqa: TID252

if TYPE_CHECKING:
    from typing import Any

    from homeassistant.core import HomeAssistant
    from langchain_core.language_models import BaseMessage, LanguageModelInput
    from langchain_core.runnables.base import RunnableSerializable

LOGGER = logging.getLogger(__name__)


async def _get_camera_image(hass: HomeAssistant, camera_name: str) -> bytes | None:
    """Get an image from a given camera."""
    camera_entity_id: str = f"camera.{camera_name.lower()}"
    try:
        image = await camera.async_get_image(
            hass=hass,
            entity_id=camera_entity_id,
            width=VLM_IMAGE_WIDTH,
            height=VLM_IMAGE_HEIGHT,
        )
    except HomeAssistantError:
        LOGGER.exception("Error getting image from camera %s", camera_entity_id)
        return None

    return image.content


def _prompt_func(data: dict[str, Any]) -> list[AnyMessage]:
    system = data["system"]
    text = data["text"]
    image = data["image"]
    prev_text = data.get("prev_text")

    # Build the user content (text first, then optional previous frame text, then image)
    content_parts: list[str | dict[str, Any]] = []

    # Main instruction text
    content_parts.append({"type": "text", "text": text})

    # OPTIONAL: previous frame's one-line description to aid motion/direction grounding
    if prev_text:
        # Keep it short and explicit that it is text-only context, not metadata
        content_parts.append(
            {"type": "text", "text": f'Previous frame (text only): "{prev_text}"'}
        )

    # Image payload last
    content_parts.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        }
    )

    return [SystemMessage(content=system), HumanMessage(content=content_parts)]


async def analyze_image(
    vlm_model: RunnableSerializable[LanguageModelInput, BaseMessage],
    image: bytes,
    detection_keywords: list[str] | None = None,
    prev_text: str | None = None,
) -> str:
    """Analyze an image with the preconfigured VLM model."""
    await asyncio.sleep(0)  # keep the event loop snappy

    image_data = base64.b64encode(image).decode("utf-8")
    chain = _prompt_func | vlm_model

    if detection_keywords is not None:
        prompt = VLM_USER_KW_TEMPLATE.format(key_words=" or ".join(detection_keywords))
    else:
        prompt = VLM_USER_PROMPT

    try:
        resp = await chain.ainvoke(
            {
                "system": VLM_SYSTEM_PROMPT,
                "text": prompt,
                "image": image_data,
                "prev_text": prev_text,
            }
        )
    except HomeAssistantError:
        msg = "Error analyzing image with VLM model."
        LOGGER.exception(msg)
        return msg

    LOGGER.debug("Raw VLM model response: %s", resp)

    return extract_final(getattr(resp, "content", "") or "")


@tool(parse_docstring=True)
async def get_and_analyze_camera_image(  # noqa: D417
    camera_name: str,
    detection_keywords: list[str] | None = None,
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> str:
    """
    Get a camera image and perform scene analysis on it.

    Args:
        camera_name: Name of the camera for scene analysis.
        detection_keywords: Specific objects to look for in image, if any.
            For example, If user says "check the front porch camera for
            boxes and dogs", detection_keywords would be ["boxes", "dogs"].

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    hass = config["configurable"]["hass"]
    vlm_model = config["configurable"]["vlm_model"]
    image = await _get_camera_image(hass, camera_name)
    if image is None:
        return "Error getting image from camera."
    return await analyze_image(vlm_model, image, detection_keywords)
