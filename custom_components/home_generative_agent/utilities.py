"""Utility functions for Home Generative Assist."""
import logging
from collections.abc import Callable, Generator
from datetime import datetime
from typing import Any

import homeassistant.util.dt as dt_util
from homeassistant.exceptions import (
    HomeAssistantError,
)
from homeassistant.helpers import llm
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import OllamaEmbeddings
from tiktoken.core import Encoding
from voluptuous_openapi import convert

LOGGER = logging.getLogger(__name__)

async def generate_embeddings(
        texts: list[str],
        model: OllamaEmbeddings
    ) -> list[list[float]]:
    """Generate embeddings from a list of text."""
    response = await model.aembed_documents(texts)
    return list(response)

def gen_dict_extract(key: str, var: dict) -> Generator[str, None, None]:
    """Find a key in nested dict."""
    if hasattr(var,"items"):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result

def format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format Home Assistant LLM tools to be compatible with OpenAI format."""
    tool_spec = {
        "name": tool.name,
        "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
    }
    # Add dummy arg descriptions if needed to make function token counter work.
    for arg in list(tool_spec["parameters"]["properties"]):
        try:
            _ = tool_spec["parameters"]["properties"][arg]["description"]
        except KeyError:
            tool_spec["parameters"]["properties"][arg]["description"] = ""
    if tool.description:
        tool_spec["description"] = tool.description
    return {"type": "function", "function": tool_spec}

def as_utc(dattim: str, default: datetime, error_message: str) -> datetime:
        """
        Convert a string representing a datetime into a datetime.datetime.

        Args:
            dattim: String representing a datetime.
            default: datatime.datetime to use as default.
            error_message: Message to raise in case of error.

        Raises:
            Homeassistant error if datetime cannot be parsed.

        Returns:
            A datetime.datetime of the string in UTC.

        """
        if dattim is None:
            return default

        parsed_datetime = dt_util.parse_datetime(dattim)
        if parsed_datetime is None:
            raise HomeAssistantError(error_message)

        return dt_util.as_utc(parsed_datetime)

def token_counter(messages: list[BaseMessage], encoding: Encoding | Any) -> int:
    """
    Approximately reproduce https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb.

    For simplicity only supports str Message.contents.

    Currently not used.
    """
    def _num_tokens_in_text(text: str, enc: Encoding | Any) -> int:
        return len(enc.encode(text))

    num_tokens = 3  # every reply is primed with <|start|>assistant<|message|>
    tokens_per_message = 3
    tokens_per_name = 1
    for msg in messages:
        LOGGER.debug("MSG: %s", msg)
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported messages type {msg.__class__}")
        LOGGER.debug("MSG CONTENT: %s", msg.content)
        num_tokens += (
            tokens_per_message
            + _num_tokens_in_text(role, encoding)
            + _num_tokens_in_text(msg.content, encoding)
        )
        if msg.name:
            LOGGER.debug("MESSAGE NAME: %s", msg.name)
            num_tokens += tokens_per_name + _num_tokens_in_text(msg.name, encoding)
        try:
            if msg.tool_calls:
                LOGGER.debug("TOOL CALLS: %s", msg.tool_calls)
        except AttributeError:
            pass
    LOGGER.debug("NUM TOKENS: %s", num_tokens)
    return num_tokens

def num_tokens_for_tools(
        functions: list,
        encoding: Encoding,
        model: str="gpt-4o"
    ) -> int:
    """
    See https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb.

    Currently not used.
    """
    # Initialize function settings to 0
    func_init = 0
    prop_init = 0
    prop_key = 0
    enum_init = 0
    enum_item = 0
    func_end = 0

    if model in [
        "gpt-4o",
        "gpt-4o-mini"
    ]:

        # Set function settings for the above models
        func_init = 7
        prop_init = 3
        prop_key = 3
        enum_init = -3
        enum_item = 3
        func_end = 12
    elif model in [
        "gpt-3.5-turbo",
        "gpt-4"
    ]:
        # Set function settings for the above models
        func_init = 10
        prop_init = 3
        prop_key = 3
        enum_init = -3
        enum_item = 3
        func_end = 12
    else:
        msg = f"""num_tokens_for_tools() is not implemented for model {model}."""
        raise NotImplementedError(
            msg
        )

    func_token_count = 0
    if len(functions) > 0:
        for f in functions:
            func_token_count += func_init  # Add tokens for start of each function
            function = f["function"]
            f_name = function["name"]
            f_desc = function["description"]
            if f_desc.endswith("."):
                f_desc = f_desc[:-1]
            line = f_name + ":" + f_desc
            func_token_count += len(encoding.encode(line))  # Add tokens for set name and description
            if len(function["parameters"]["properties"]) > 0:
                func_token_count += prop_init  # Add tokens for start of each property
                for key in list(function["parameters"]["properties"].keys()):
                    LOGGER.debug("KEY: %s", key)
                    func_token_count += prop_key  # Add tokens for each set property
                    p_name = key
                    p_type = function["parameters"]["properties"][key]["type"]
                    p_desc = function["parameters"]["properties"][key]["description"]
                    if "enum" in function["parameters"]["properties"][key].keys():
                        func_token_count += enum_init  # Add tokens if property has enum list
                        for item in function["parameters"]["properties"][key]["enum"]:
                            func_token_count += enum_item
                            func_token_count += len(encoding.encode(item))
                    if p_desc.endswith("."):
                        p_desc = p_desc[:-1]
                    line = f"{p_name}:{p_type}:{p_desc}"
                    func_token_count += len(encoding.encode(line))
        func_token_count += func_end
    LOGGER.debug("FUNC TOKENS: %s", func_token_count)
    total_tokens = func_token_count

    return total_tokens
