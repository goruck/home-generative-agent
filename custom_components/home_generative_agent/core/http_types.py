"""
Type names for the HTTP client and response Home Assistant hands out.

From Home Assistant 2026.11, ``import homeassistant`` calls
``httpx2.alias_httpx()``, so at runtime ``import httpx`` loads ``httpx2`` and
``get_async_client`` returns an ``httpx2.AsyncClient``; before 2026.11 it is
an ``httpx.AsyncClient``. HGA keeps importing ``httpx`` at runtime, which is
right on every supported release (2026.8 has no ``httpx2``), but a type
checker reads the real ``httpx`` package and sees two unrelated classes.
Annotate a client or response that can come from Home Assistant with these
names, which accept either. They are for annotations only and exist only
while type checking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx
    import httpx2

    type HAAsyncClient = httpx.AsyncClient | httpx2.AsyncClient
    type HAResponse = httpx.Response | httpx2.Response
