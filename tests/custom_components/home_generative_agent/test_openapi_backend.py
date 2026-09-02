# ruff: noqa: S101
"""
Tests for picking the OpenAPI schema converter.

Home Assistant 2026.9 swapped voluptuous-openapi for probatio and aliased
probatio under the ``voluptuous`` name, so the converter has to follow whichever
library the running core builds its schemas with.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

import voluptuous as vol
import voluptuous_openapi
from homeassistant.helpers import llm

from custom_components.home_generative_agent.agent import helpers

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

# Core re-exports the sentinel of whichever converter library it was built
# against, and that is exactly the object its own serializers hand back.
_CORE_UNSUPPORTED: Any = llm.UNSUPPORTED  # pyright: ignore[reportPrivateImportUsage]


class _Opaque:
    """A hashable object no branch of the robust serializer claims."""


def test_voluptuous_openapi_is_used_on_a_voluptuous_core() -> None:
    """A core whose ``vol`` is real voluptuous renders with voluptuous-openapi."""
    assert not vol.Schema.__module__.startswith("probatio")
    assert helpers.convert is voluptuous_openapi.convert
    assert helpers.UNSUPPORTED is voluptuous_openapi.UNSUPPORTED


def test_core_sentinel_counts_as_a_deferral() -> None:
    """Core's own sentinel is recognised even if it comes from the other lib."""
    assert helpers._is_unsupported(helpers.UNSUPPORTED)
    assert helpers._is_unsupported(_CORE_UNSUPPORTED)
    assert not helpers._is_unsupported({"type": "string"})
    assert not helpers._is_unsupported(None)


def test_probatio_is_used_when_core_aliases_it_as_voluptuous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 2026.9 core renders with probatio, and foreign sentinels still defer."""
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

    monkeypatch.setitem(sys.modules, "probatio", probatio)
    monkeypatch.setitem(sys.modules, "voluptuous", aliased_vol)
    importlib.reload(helpers)
    try:
        assert helpers.convert is probatio.to_openapi
        assert helpers.UNSUPPORTED is probatio.UNSUPPORTED

        # A serializer built against the *other* library hands back its own
        # sentinel; treating that as a rendered schema would corrupt the tool.
        helpers.safe_convert(
            _ProbatioSchema(), custom_serializer=lambda _obj: _CORE_UNSUPPORTED
        )
        robust = seen[-1]
        assert robust(_Opaque()) is probatio.UNSUPPORTED
    finally:
        monkeypatch.undo()
        importlib.reload(helpers)

    assert helpers.convert is voluptuous_openapi.convert
