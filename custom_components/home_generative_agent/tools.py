"""Langgraph tools for Home Generative Agent."""
from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING, Annotated, Any

from homeassistant.components import camera
from homeassistant.exceptions import (
    HomeAssistantError,
)
from homeassistant.util import ulid
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig  # noqa: TCH002
from langchain_core.tools import InjectedToolArg, tool
from langchain_ollama import ChatOllama  # noqa: TCH002
from langgraph.prebuilt import InjectedStore  # noqa: TCH002
from langgraph.store.base import BaseStore  # noqa: TCH002
from pydantic import BaseModel, Field
from ulid import ULID  # noqa: TCH002

from .const import (
    CONF_VISION_MODEL_TEMPERATURE,
    CONF_VLM,
    RECOMMENDED_VISION_MODEL_TEMPERATURE,
    RECOMMENDED_VLM,
    VISION_MODEL_IMAGE_HEIGHT,
    VISION_MODEL_IMAGE_WIDTH,
    VISION_MODEL_SYSTEM_PROMPT,
    VISION_MODEL_USER_PROMPT_TEMPLATE,
    VLM_NUM_PREDICT,
)

if TYPE_CHECKING:
    from types import MappingProxyType

    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

async def _get_camera_image(hass: HomeAssistant, camera_name: str) -> bytes:
    """Get an image from a given camera."""
    camera_entity_id: str = f"camera.{camera_name.lower()}"
    try:
        image = await camera.async_get_image(
            hass=hass,
            entity_id=camera_entity_id,
            width=VISION_MODEL_IMAGE_WIDTH,
            height=VISION_MODEL_IMAGE_HEIGHT
        )
    except HomeAssistantError as err:
        LOGGER.error(
            "Error getting image from camera '%s' with error: %s",
            camera_entity_id, err
        )

    return image.content

async def _analyze_image(
        vlm_model: ChatOllama,
        options: dict[str, Any] | MappingProxyType[str, Any],
        image: bytes
    ) -> str:
    """Analyze an image."""
    encoded_image = base64.b64encode(image).decode("utf-8")

    def prompt_func(data: dict[str, Any]) -> list[AnyMessage]:
        system = data["system"]
        text = data["text"]
        image = data["image"]

        text_part = {"type": "text", "text": text}
        image_part = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        }

        content_parts = []
        content_parts.append(text_part)
        content_parts.append(image_part)

        return [SystemMessage(content=system), HumanMessage(content=content_parts)]

    class ObjectTypeAndLocation(BaseModel):
        """Get type and location of objects in image."""

        object_type: str = Field(
            description="the type of obect in the immage"
        )
        object_location: str = Field(
            description="the location of the object in the image"
        )

    class ImageSceneAnalysis(BaseModel):
        """
        Get image scene analysis.

        Includes a description of the image, type and location of objects present,
        number of people present and number of animals present in the image.
        """

        description: str = Field(
            description="description of the image scene"
        )
        objects: list[ObjectTypeAndLocation] = Field(
            description="object type and location in image"
        )
        people: int = Field(
            description="number of people in the image"
        )
        animals: int = Field(
            description="number of aniamls in the image"
        )

    schema = json.dumps(ImageSceneAnalysis.model_json_schema())

    model = vlm_model
    model_with_config = model.with_config(
        {"configurable":
            {
                "model": options.get(
                    CONF_VLM,
                    RECOMMENDED_VLM,
                ),
                "format": "json",
                "temperature": options.get(
                    CONF_VISION_MODEL_TEMPERATURE,
                    RECOMMENDED_VISION_MODEL_TEMPERATURE,
                ),
                "num_predict": VLM_NUM_PREDICT,
            }
        }
    )

    chain = prompt_func | model_with_config

    try:
        response =  await chain.ainvoke(
            {
                "system": VISION_MODEL_SYSTEM_PROMPT,
                "text": VISION_MODEL_USER_PROMPT_TEMPLATE.format(schema=schema),
                "image": encoded_image
            }
        )
    except HomeAssistantError as err: #TODO: add validation error handling and retry prompt
        LOGGER.error("Error analyzing image %s", err)

    return response

@tool
async def get_and_analyze_camera_image(
        camera_name: str,
        *,
        # Hide these arguments from the model.
        config: Annotated[RunnableConfig, InjectedToolArg()],
        store: Annotated[BaseStore, InjectedStore()],
    ) -> str:
    """Get an image from a given camera and analyze it."""
    hass = config["configurable"]["hass"]
    vlm_model = config["configurable"]["vlm_model"]
    options = config["configurable"]["options"]
    image = await _get_camera_image(hass, camera_name)
    return await _analyze_image(vlm_model, options, image)

@tool(parse_docstring=False)
async def upsert_memory(
    content: str,
    context: str,
    *,
    memory_id: ULID | None = None,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """
    Upsert a memory in the database.

    If a memory conflicts with an existing one, then just UPDATE the
    existing one by passing in memory_id - don't create two memories
    that are the same. If the user corrects a memory, UPDATE it.

    Args:
        content: The main content of the memory. For example:
            "User expressed interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
        memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
            The memory to overwrite

    Returns:
        A string containing the stored memory id.

    """
    mem_id = memory_id or ulid.ulid_now()
    await store.aput(
        ("memories", config["configurable"]["user_id"]),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"
