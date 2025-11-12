"""Web search tool for Home Generative Agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import InjectedToolArg, tool

from ..const import (  # noqa: TID252
    CONF_BROWSERLESS_URL,
    CONF_SEARXNG_URL,
    RECOMMENDED_BROWSERLESS_URL,
    RECOMMENDED_SEARXNG_URL,
    WEB_SEARCH_MAX_CONTENT_LENGTH,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_TIMEOUT_SECONDS,
)
from ..core.utils import extract_final  # noqa: TID252

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from langchain_core.language_models import BaseMessage, LanguageModelInput
    from langchain_core.runnables.base import RunnableSerializable

LOGGER = logging.getLogger(__name__)


async def _query_searxng(
    searxng_url: str, query: str, max_results: int = WEB_SEARCH_MAX_RESULTS
) -> list[dict[str, str]]:
    """Query searxng and return search results.

    Args:
        searxng_url: URL of the searxng instance
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        List of search results with 'url', 'title', and 'content' keys
    """
    try:
        async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT_SECONDS) as client:
            response = await client.get(
                f"{searxng_url}/search",
                params={"q": query, "format": "json"},
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for result in data.get("results", [])[:max_results]:
                results.append(
                    {
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "content": result.get("content", ""),
                    }
                )
            return results
    except Exception:
        LOGGER.exception("Error querying searxng at %s", searxng_url)
        return []


async def _fetch_page_with_browserless(
    browserless_url: str, url: str, timeout: int = WEB_SEARCH_TIMEOUT_SECONDS
) -> str:
    """Fetch page content using Browserless /chromium/scrape endpoint.

    Args:
        browserless_url: Base URL of the Browserless server
        url: URL to fetch
        timeout: Timeout in seconds

    Returns:
        Extracted text content from the page, filtered for common body tags
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Use browserless /chromium/scrape endpoint with element filtering
            # Filter for common body tags that contain relevant content
            response = await client.post(
                f"{browserless_url}/chromium/scrape",
                json={
                    "url": url,
                    "elements": [
                        {"selector": "article", "timeout": 2500},
                        {"selector": "main", "timeout": 2500},
                        {"selector": "p", "timeout": 2500},
                        {"selector": "h1", "timeout": 2500},
                        {"selector": "h2", "timeout": 2500},
                        {"selector": "h3", "timeout": 2500},
                        {"selector": "h4", "timeout": 2500},
                        {"selector": "h5", "timeout": 2500},
                        {"selector": "h6", "timeout": 2500},
                        {"selector": "li", "timeout": 2500},
                        {"selector": "blockquote", "timeout": 2500},
                        {"selector": "pre", "timeout": 2500},
                        {"selector": "code", "timeout": 2500},
                    ],
                    "bestAttempt": True,
                },
            )
            LOGGER.debug("Browserless response status: %d", response.status_code)
            data = response.json()

            # Extract text from the elements
            text_parts = []
            for element_data in data.get("data", []):
                if isinstance(element_data, dict):
                    # Extract text from results array
                    for result in element_data.get("results", []):
                        if isinstance(result, dict):
                            text = result.get("text", "")
                            if text and text.strip():
                                text_parts.append(text.strip())

            # Join all text parts with newlines
            combined_text = "\n".join(text_parts)
            LOGGER.debug("Extracted %d characters from page", len(combined_text))
            return combined_text
    except Exception:
        LOGGER.exception("Error fetching page %s with Browserless", url)
        return ""


async def _summarize_search_results(
    chat_model: RunnableSerializable[LanguageModelInput, BaseMessage],
    query: str,
    results: list[dict[str, Any]],
) -> str:
    """Summarize search results using the chat model.

    Args:
        chat_model: Language model to use for summarization
        query: Original search query
        results: List of search results with content

    Returns:
        Summarized content
    """
    # Prepare content for summarization
    content_parts = []
    for i, result in enumerate(results, 1):
        content_parts.append(
            f"Result {i}: {result['title']}\n"
            f"URL: {result['url']}\n"
            f"Content: {result['content'][:WEB_SEARCH_MAX_CONTENT_LENGTH]}\n"
        )

    combined_content = "\n\n".join(content_parts)

    messages = [
        SystemMessage(
            content="You are a helpful assistant that summarizes web search results. "
            "Provide a concise summary of the key information found."
        ),
        HumanMessage(
            content=f"Search query: {query}\n\n"
            f"Search results:\n{combined_content}\n\n"
            f"Please provide a brief summary of the most relevant information."
        ),
    ]

    try:
        response = await chat_model.ainvoke(messages)
        return extract_final(getattr(response, "content", "") or "")
    except Exception:
        LOGGER.exception("Error summarizing search results")
        return "Error: Could not summarize search results."


@tool(parse_docstring=True)
async def web_search(  # noqa: D417
    query: str,
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> str:
    """
    Perform a web search using searxng and fetch detailed content with Browserless.

    This tool first searches the web using a searxng instance, then fetches
    and analyzes the content of the top results using Browserless, and finally
    provides a summary of the findings.

    Args:
        query: The search query to execute.

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    hass: HomeAssistant = config["configurable"]["hass"]
    chat_model = config["configurable"]["chat_model"]

    # Get configuration
    entry_data = hass.config_entries.async_entries("home_generative_agent")
    if not entry_data:
        return "Integration not configured properly."

    entry = entry_data[0]
    merged_config = {**entry.data, **entry.options}

    searxng_url = merged_config.get(CONF_SEARXNG_URL, RECOMMENDED_SEARXNG_URL)
    browserless_url = merged_config.get(
        CONF_BROWSERLESS_URL, RECOMMENDED_BROWSERLESS_URL
    )

    LOGGER.info("Performing web search for query: %s", query)

    # Step 1: Query searxng
    search_results = await _query_searxng(searxng_url, query, WEB_SEARCH_MAX_RESULTS)

    if not search_results:
        return f"No search results found for query: {query}"

    LOGGER.debug("Found %d search results", len(search_results))

    # Step 2: Fetch detailed content from top results using Browserless
    for result in search_results:
        url = result["url"]
        LOGGER.debug("Fetching content from: %s", url)
        content = await _fetch_page_with_browserless(browserless_url, url)
        if content:
            result["content"] = content[:WEB_SEARCH_MAX_CONTENT_LENGTH]
        # If Browserless fails, we'll use the snippet from searxng (already in result['content'])

    # Step 3: Summarize the results
    summary = await _summarize_search_results(chat_model, query, search_results)

    return summary
