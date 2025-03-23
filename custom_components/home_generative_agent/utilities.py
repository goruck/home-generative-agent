"""Utility functions for Home Generative Assist."""
import logging
from typing import Any

from homeassistant.helpers import llm
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from tiktoken.core import Encoding

LOGGER = logging.getLogger(__name__)

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
