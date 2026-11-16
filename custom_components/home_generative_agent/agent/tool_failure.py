"""A plain-text tool answer that says the tool did not do what was asked."""

from __future__ import annotations

from langchain_core.messages.tool import ToolOutputMixin


class ToolFailure(str, ToolOutputMixin):
    """
    The text a tool returns when it did not do what was asked (#1023).

    LangChain gives every string a tool returns the status "success", so a
    refusal written as text ("Camera 'X' was not found", "Not stored") read as
    done, and an empty model reply after one said "Done.". As a
    `ToolOutputMixin`, LangChain hands this back unchanged instead of wrapping
    it in a `ToolMessage`, so `_run_langchain_tool` (graph.py) can give it the
    error status. It is still a `str`, so a caller comparing the text, or
    calling the tool's coroutine directly, sees the same words.
    """

    __slots__ = ()
