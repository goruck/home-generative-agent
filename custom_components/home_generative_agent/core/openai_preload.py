"""
Resolve the OpenAI SDK's lazily imported resources off the event loop.

From openai 3.26 (pinned by Home Assistant 2026.11), the SDK's resource
packages import their modules on first attribute access, through a module
``__getattr__`` that calls ``importlib.import_module`` with a relative name
(``".chat"``). Home Assistant flags any ``import_module`` call made on the
event loop unless the name is already in ``sys.modules``, and a relative name
never is. So the first ``ChatOpenAI(...)``, ``OpenAIEmbeddings(...)`` or
``client.audio.speech`` on the loop logs a "Detected blocking call to
import_module" warning that tells the user to report a bug against this
integration, and loads the module on the loop while doing it.

Importing the modules ahead of time is not enough: the warning fires on the
relative name, so each package's lazy attributes have to be resolved too.
After resolution the package caches them in its globals and never calls
``import_module`` again. :func:`preload_openai_resources` does this for every
package on this integration's paths and is meant to run in the executor once,
before any provider is built. Older SDKs import eagerly, so for them it does
nothing beyond resolving attributes that already exist.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_LOGGER = logging.getLogger(__name__)

# Packages whose every lazy attribute this integration can reach: chat
# completions (ChatOpenAI), the Responses API (ChatOpenAI can pick it per
# model), and audio (the STT and TTS platforms).
_LAZY_PACKAGES = (
    "openai.resources.chat",
    "openai.resources.chat.completions",
    "openai.resources.responses",
    "openai.resources.audio",
)

# From the top-level resources package, only the classes the clients build for
# the resources this integration uses. Resolving all of it would import the
# whole SDK (admin, fine-tuning, vector stores, ...).
_RESOURCE_NAMES = (
    "Chat",
    "AsyncChat",
    "Responses",
    "AsyncResponses",
    "Audio",
    "AsyncAudio",
    "Embeddings",
    "AsyncEmbeddings",
    "Models",
    "AsyncModels",
)

_preloaded = False


def _resolve(module: ModuleType, names: tuple[str, ...] | list[str]) -> None:
    for name in names:
        try:
            getattr(module, name)
        except Exception:  # noqa: BLE001 - best effort; a miss only costs a warning later
            _LOGGER.debug("Could not preload %s.%s", module.__name__, name)


def preload_openai_resources() -> None:
    """
    Resolve the OpenAI SDK's lazy resource attributes. Blocking; run in the executor.

    Idempotent and never raises: a name a future SDK drops is skipped, and the
    cost of a miss is the warning this exists to prevent, not a failed setup.
    """
    global _preloaded  # noqa: PLW0603
    if _preloaded:
        return
    try:
        resources = importlib.import_module("openai.resources")
    except Exception:  # noqa: BLE001
        _LOGGER.debug("Could not import openai.resources to preload it")
        return
    _resolve(resources, _RESOURCE_NAMES)
    for package_name in _LAZY_PACKAGES:
        try:
            package = importlib.import_module(package_name)
        except Exception:  # noqa: BLE001
            _LOGGER.debug("Could not import %s to preload it", package_name)
            continue
        _resolve(package, [n for n in dir(package) if not n.startswith("_")])
    _preloaded = True
