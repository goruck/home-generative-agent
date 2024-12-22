"""Utility functions for Home Generative Assist."""
import logging

from langchain_ollama import OllamaEmbeddings

LOGGER = logging.getLogger(__name__)

async def generate_embeddings(
        texts: list[str],
        model: OllamaEmbeddings
    ) -> list[list[float]]:
    """Generate embeddings from a list of text."""
    response = await model.aembed_documents(texts)
    return list(response)
