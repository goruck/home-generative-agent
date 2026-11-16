"""A plain-text tool answer that says the tool did not do what was asked."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages.tool import ToolOutputMixin

if TYPE_CHECKING:
    from langchain_core.messages import ToolMessage

# Set in a ToolMessage's additional_kwargs by `_run_langchain_tool` (graph.py)
# for a ToolFailure. Its error status says the tool did not act; this says the
# tool ran and answered, so readers that skip errors to mean "the call never
# ran" (a PIN still pending, a tool not worth carrying to the next turn) can
# tell the two apart, as they did when these answers had the status "success".
TOOL_REFUSAL_KWARG = "hga_tool_refusal"


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

    # For ruff's SLOT000 only: ToolOutputMixin has no __slots__, so instances
    # still carry a __dict__.
    __slots__ = ()


def is_tool_refusal(msg: ToolMessage) -> bool:
    """Return True for one of our tools' answers that it did not act."""
    return bool(msg.additional_kwargs.get(TOOL_REFUSAL_KWARG))
