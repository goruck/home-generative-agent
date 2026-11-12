# ruff: noqa: S101
"""Tests for resolving the OpenAI SDK's lazy resources off the event loop."""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from typing import Any

import pytest

import custom_components.home_generative_agent as hga_init
from custom_components.home_generative_agent.core import openai_preload

_PACKAGES = (
    "openai.resources",
    "openai.resources.chat",
    "openai.resources.chat.completions",
    "openai.resources.responses",
    "openai.resources.audio",
)


def _lazy_package(name: str, exports: list[str], calls: list[str]) -> Any:
    """Build a module that resolves names like openai 3.26's lazy packages do."""
    module = types.ModuleType(name)

    def __getattr__(attr: str) -> Any:  # noqa: N807
        if attr not in exports:
            raise AttributeError(attr)
        calls.append(f"{name}.{attr}")
        value = object()
        setattr(module, attr, value)  # what the SDK's globals() cache does
        return value

    module.__getattr__ = __getattr__  # type: ignore[method-assign]
    module.__dir__ = lambda: sorted(exports)  # type: ignore[method-assign]
    return module


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Install lazy stand-ins for the SDK packages and reset the once-flag."""
    calls: list[str] = []
    monkeypatch.setattr(openai_preload, "_preloaded", False)
    for name in _PACKAGES:
        exports = (
            ["Chat", "AsyncChat", "Audio", "AsyncAudio", "Embeddings", "Batches"]
            if name == "openai.resources"
            else ["Thing", "AsyncThing"]
        )
        monkeypatch.setitem(sys.modules, name, _lazy_package(name, exports, calls))
    return calls


def test_preload_resolves_every_lazy_name_on_the_integrations_paths(
    fake_sdk: list[str],
) -> None:
    """Each package's names resolve, so the loop never imports them later."""
    openai_preload.preload_openai_resources()

    for name in _PACKAGES[1:]:
        assert f"{name}.Thing" in fake_sdk
        assert f"{name}.AsyncThing" in fake_sdk
    assert "openai.resources.Chat" in fake_sdk
    assert "openai.resources.AsyncAudio" in fake_sdk
    # Only the resources the integration uses, not the whole SDK.
    assert "openai.resources.Batches" not in fake_sdk


def test_preload_runs_once(fake_sdk: list[str]) -> None:
    """A second setup (a reload) does not walk the packages again."""
    openai_preload.preload_openai_resources()
    first = list(fake_sdk)
    openai_preload.preload_openai_resources()

    assert fake_sdk == first


def test_preload_survives_a_name_that_will_not_resolve(
    fake_sdk: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A future SDK that drops or breaks a name costs a warning, not setup."""
    broken = sys.modules["openai.resources.chat"]

    def _boom(attr: str) -> Any:
        raise ImportError(attr)

    monkeypatch.setattr(broken, "__getattr__", _boom)

    openai_preload.preload_openai_resources()

    assert "openai.resources.audio.Thing" in fake_sdk


def test_preload_survives_a_missing_package(
    fake_sdk: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An SDK without one of the packages still preloads the rest."""
    real_import = importlib.import_module

    def _import(name: str, package: str | None = None) -> Any:
        if name == "openai.resources.responses":
            raise ModuleNotFoundError(name)
        return real_import(name, package)

    monkeypatch.setattr(openai_preload.importlib, "import_module", _import)

    openai_preload.preload_openai_resources()

    assert "openai.resources.audio.Thing" in fake_sdk
    assert not [c for c in fake_sdk if c.startswith("openai.resources.responses")]


def test_preload_against_the_installed_sdk() -> None:
    """The real SDK, whatever version is pinned, preloads without raising."""
    openai_preload._preloaded = False
    try:
        openai_preload.preload_openai_resources()
    finally:
        openai_preload._preloaded = True
    assert openai_preload._preloaded


def test_setup_preloads_in_the_executor_before_building_providers() -> None:
    """Pin the wiring: off the loop, and before the first OpenAI model."""
    full = inspect.getsource(hga_init)
    # Setup builds every provider; build_openai_chat is defined above it and
    # only called from it.
    src = full[full.index("async def async_setup_entry(") :]
    preload = src.index("async_add_executor_job(preload_openai_resources)")
    assert preload < src.index("build_openai_chat(")
    assert preload < src.index("ChatOpenAI(")
    assert preload < src.index("OpenAIEmbeddings(")
    assert preload < src.index("async_forward_entry_setups(")
    builder = full.index("def build_openai_chat(")
    assert full.count("build_openai_chat(") == full[builder:].count(
        "build_openai_chat("
    ), "build_openai_chat is called only from setup"
