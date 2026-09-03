# ruff: noqa: S101
"""
Tests for picking the OpenAPI schema converter.

Home Assistant 2026.9 swapped voluptuous-openapi for probatio and aliased
probatio under the ``voluptuous`` name, so the converter has to follow whichever
library the running core builds its schemas with. ``helpers`` prefers core's
own ``llm`` re-exports (which cannot diverge from core) and falls back to
sniffing ``vol.Schema.__module__`` on a core that stops re-exporting them.

These tests must collect and pass on BOTH cores: nothing here may import
voluptuous_openapi or probatio at module level — whichever one the running
core was not built against may be absent from the environment.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest
import voluptuous as vol
from homeassistant.helpers import llm

from custom_components.home_generative_agent.agent import helpers

if TYPE_CHECKING:
    from collections.abc import Callable

# Core re-exports the sentinel of whichever converter library it was built
# against, and that is exactly the object its own serializers hand back.
_CORE_UNSUPPORTED: Any = llm.UNSUPPORTED  # pyright: ignore[reportPrivateImportUsage]

_IS_PROBATIO_CORE = vol.Schema.__module__.startswith("probatio")


class _Opaque:
    """A hashable object no branch of the robust serializer claims."""


def test_converter_and_sentinel_match_core_llm_exports() -> None:
    """On any supported core, the converter pair is core's own re-exports."""
    core_convert = getattr(llm, "to_openapi", None) or getattr(llm, "convert", None)
    assert core_convert is not None
    assert helpers.convert is core_convert
    assert helpers.UNSUPPORTED is _CORE_UNSUPPORTED


@pytest.mark.skipif(
    _IS_PROBATIO_CORE, reason="core is probatio; the voluptuous-openapi path is n/a"
)
def test_voluptuous_openapi_is_used_on_a_voluptuous_core() -> None:
    """A core whose ``vol`` is real voluptuous renders with voluptuous-openapi."""
    voluptuous_openapi = pytest.importorskip("voluptuous_openapi")
    assert helpers.convert is voluptuous_openapi.convert
    assert helpers.UNSUPPORTED is voluptuous_openapi.UNSUPPORTED


@pytest.mark.skipif(
    not _IS_PROBATIO_CORE, reason="core is voluptuous; the probatio path is n/a"
)
def test_probatio_is_used_on_a_probatio_core() -> None:
    """A 2026.9 core renders with probatio's ``to_openapi``."""
    probatio = pytest.importorskip("probatio")
    assert helpers.convert is probatio.to_openapi
    assert helpers.UNSUPPORTED is probatio.UNSUPPORTED


def test_core_sentinel_counts_as_a_deferral() -> None:
    """Core's own sentinel is recognised even if it comes from the other lib."""
    assert helpers._is_unsupported(helpers.UNSUPPORTED)
    assert helpers._is_unsupported(_CORE_UNSUPPORTED)
    assert not helpers._is_unsupported({"type": "string"})
    assert not helpers._is_unsupported(None)


def test_sibling_library_sentinel_counts_as_a_deferral() -> None:
    """
    The sentinel of every importable converter library is accepted.

    A third-party ``APIInstance.custom_serializer`` can be built against the
    sibling library (pip does not uninstall voluptuous-openapi when a core
    upgrade brings probatio); its sentinel must read as a deferral, not as a
    rendered schema.
    """
    for module_name in ("voluptuous_openapi", "probatio"):
        module = sys.modules.get(module_name)
        if module is None:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
        sentinel = getattr(module, "UNSUPPORTED", None)
        if sentinel is not None:
            assert helpers._is_unsupported(sentinel)


def _fake_probatio_core() -> tuple[ModuleType, ModuleType, list[Callable[[Any], Any]]]:
    """Build fake ``probatio`` and aliased ``voluptuous`` modules."""
    probatio = ModuleType("probatio")
    probatio.UNSUPPORTED = object()  # type: ignore[attr-defined]
    seen: list[Callable[[Any], Any]] = []

    def to_openapi(
        schema: Any,  # noqa: ARG001
        *,
        custom_serializer: Callable[[Any], Any] | None = None,
    ) -> dict[str, Any]:
        if custom_serializer is not None:
            seen.append(custom_serializer)
        return {"type": "object"}

    probatio.to_openapi = to_openapi  # type: ignore[attr-defined]

    class _ProbatioSchema:
        """Stand-in for the ``Schema`` core's aliased ``voluptuous`` exposes."""

    _ProbatioSchema.__module__ = "probatio.schema"
    aliased_vol = ModuleType("voluptuous")
    aliased_vol.Schema = _ProbatioSchema  # type: ignore[attr-defined]
    return probatio, aliased_vol, seen


def test_llm_exports_win_over_the_module_sniff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Core's ``llm`` re-exports beat the ``__module__`` heuristic.

    This is the divergence-proofing: even if a future probatio stamps a
    voluptuous-compatible ``__module__`` on its shim (or vice versa), the
    converter still matches core because it IS core's.
    """
    probatio, aliased_vol, _seen = _fake_probatio_core()
    try:
        # The reload sits inside the try: if it raises partway, the finally
        # still restores a coherent helpers module for the rest of the session.
        monkeypatch.setitem(sys.modules, "probatio", probatio)
        monkeypatch.setitem(sys.modules, "voluptuous", aliased_vol)
        importlib.reload(helpers)
        core_convert = getattr(llm, "to_openapi", None) or getattr(llm, "convert", None)
        assert helpers.convert is core_convert
        assert helpers.UNSUPPORTED is _CORE_UNSUPPORTED
        # The fake probatio's sentinel is importable, so it is accepted too.
        assert helpers._is_unsupported(probatio.UNSUPPORTED)
    finally:
        monkeypatch.undo()
        importlib.reload(helpers)

    assert helpers.UNSUPPORTED is _CORE_UNSUPPORTED


def test_module_sniff_fallback_when_llm_stops_reexporting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Without the ``llm`` re-exports, the sniff picks the aliasing library.

    Foreign sentinels still defer: a serializer built against the *other*
    library hands back its own sentinel, and treating that as a rendered
    schema would corrupt the tool.
    """
    probatio, aliased_vol, seen = _fake_probatio_core()
    try:
        monkeypatch.setitem(sys.modules, "probatio", probatio)
        monkeypatch.setitem(sys.modules, "voluptuous", aliased_vol)
        monkeypatch.delattr(llm, "convert", raising=False)
        monkeypatch.delattr(llm, "to_openapi", raising=False)
        monkeypatch.delattr(llm, "UNSUPPORTED", raising=False)
        importlib.reload(helpers)
        assert helpers.convert is probatio.to_openapi
        assert helpers.UNSUPPORTED is probatio.UNSUPPORTED

        helpers.safe_convert(
            aliased_vol.Schema(),  # type: ignore[attr-defined]
            custom_serializer=lambda _obj: _CORE_UNSUPPORTED,
        )
        robust = seen[-1]
        assert robust(_Opaque()) is probatio.UNSUPPORTED
    finally:
        monkeypatch.undo()
        importlib.reload(helpers)

    assert helpers.UNSUPPORTED is _CORE_UNSUPPORTED
