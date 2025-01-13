"""Utility functions for Home Generative Assist."""
import logging
from collections.abc import Generator

from langchain_ollama import OllamaEmbeddings

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
