"""
Anthropic chat model whose SDK client is built once, off the event loop.

``ChatAnthropic`` builds its ``anthropic.AsyncClient`` lazily in a
``cached_property`` on first use. The provider is wrapped in
``configurable_fields`` so the chat, vision and summarization models can each
select their own model name and sampling parameters, and
``RunnableConfigurableFields._prepare`` honours those by constructing a *fresh*
``ChatAnthropic`` through ``__init__`` on every call. A ``cached_property``
lives in the instance ``__dict__`` and is not a model field, so it never
carries across: every request constructed a new SDK client on the calling
thread, i.e. inside the event loop.

Since anthropic 0.98 the client constructor auto-discovers credentials on disk
(``~/.config/anthropic/active_config``) so it can warn when an environment
variable shadows them. That is blocking file I/O. Home Assistant's loop
protection reports it on the first turn after every start and deduplicates the
report per call site, so the log understated how often it happened: it was
every turn (issues #587 and #618).

``_prepare`` copies every *declared* model field into the fresh instance, so
this subclass declares the client as one. Setup builds it once in an executor
through :func:`async_prime_async_client`; every per-request copy then inherits
the built client and touches neither the disk nor the SSL context. The copy
also inherits the client parameters the client was built from and compares
them with its own before reusing it, so a copy that somehow differs in API
key, base URL, timeout, retries, headers or proxy builds its own client the
way the base class would instead of silently talking to the wrong endpoint.
Today that cannot happen: only model, temperature, thinking and max_tokens
are configurable, and a credential change reloads the entry. The client dies
with the provider on unload, so a rotated key is not retained.

The fields are typed ``Any`` on purpose. ``anthropic`` is imported only for
type checking, and a pydantic field annotated with an unresolvable name fails
at class creation, which happens when ``__init__`` imports this module,
outside the per-provider try/except. That would take down the whole
integration rather than one provider.

Only the async client is shared. The integration never calls the sync client
(token counting is approximated locally), so priming it would spend a second
SSL context load at startup for a path that is never taken. A primed model
cannot be deep-copied (the httpx client holds a lock); nothing in the
integration deep-copies models, and the base class has the same property
after its first request.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from functools import cached_property
from typing import TYPE_CHECKING, Any

from langchain_anthropic import ChatAnthropic
from pydantic import Field

if TYPE_CHECKING:
    import anthropic
    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)


def _on_event_loop() -> bool:
    """Return True when called from a thread that is running an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class SharedClientChatAnthropic(ChatAnthropic):
    """``ChatAnthropic`` that shares its async SDK client across per-request copies."""

    shared_async_client: Any = Field(default=None, exclude=True, repr=False)
    """The primed ``anthropic.AsyncClient``; copied into every per-request instance."""

    shared_client_params: Any = Field(default=None, exclude=True, repr=False)
    """The client parameters ``shared_async_client`` was built from."""

    def _shared_client_params(self) -> dict[str, Any]:
        """Return everything the base class feeds into the SDK client constructor."""
        return {**self._client_params, "anthropic_proxy": self.anthropic_proxy}

    @cached_property
    def _async_client(self) -> anthropic.AsyncClient:
        try:
            params = self._shared_client_params()
            if (
                self.shared_async_client is not None
                and self.shared_client_params == params
            ):
                return self.shared_async_client
            client = super()._async_client
        except AttributeError as exc:
            # pydantic's __getattr__ turns an AttributeError raised inside a
            # property into "no attribute '_async_client'", hiding the real
            # cause (typically an SDK or langchain-anthropic release renaming
            # something this module reaches into). Keep the whole chain.
            causes: list[str] = []
            cur: BaseException | None = exc
            while cur is not None:
                causes.append(str(cur))
                cur = cur.__cause__ or cur.__context__
            msg = "Anthropic client construction failed: " + " <- ".join(causes)
            raise RuntimeError(msg) from exc
        if _on_event_loop():
            LOGGER.warning(
                "Built an Anthropic SDK client on the event loop; setup did not "
                "prime it (client parameters changed or priming failed)"
            )
        else:
            LOGGER.debug(
                "Built the Anthropic SDK client on thread %s",
                threading.current_thread().name,
            )
        self.shared_async_client = client
        self.shared_client_params = params
        return client


async def async_prime_async_client(
    hass: HomeAssistant, model: SharedClientChatAnthropic
) -> None:
    """
    Build ``model``'s SDK client in an executor so no request has to.

    Also computes the SDK's platform headers there: the SDK resolves them on
    the first request per process by reading ``/etc/os-release`` and probing
    the interpreter, both blocking, and memoises the result process-wide.
    """

    def _build() -> None:
        model._async_client.platform_headers()  # noqa: SLF001

    await hass.async_add_executor_job(_build)
