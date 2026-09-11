# ruff: noqa: S101
"""
Issue #618: the Anthropic SDK client is built once, off the event loop.

``configurable_fields`` rebuilds ``ChatAnthropic`` through ``__init__`` on
every call, which discards the lazily cached SDK client; the SDK constructor
reads ``~/.config/anthropic/active_config`` (anthropic >= 0.98), so each turn
blocked the loop on file I/O. These tests pin the per-call rebuild that the
workaround exists for, and the sharing that neutralises it.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import threading
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import ConfigurableField
from pydantic import SecretStr

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

from custom_components.home_generative_agent.core.anthropic_client import (
    LOGGER,
    SharedClientChatAnthropic,
    async_prime_async_client,
)

_CONFIG = {"configurable": {"model": "claude-sonnet-4-6", "temperature": 0.7}}
_INSTANCES: list[_FakeSdkClient] = []


class _FakeSdkClient:
    """Stands in for ``anthropic.AsyncClient``; records the constructing thread."""

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.thread = threading.current_thread().name
        self.platform_headers_thread: str | None = None
        _INSTANCES.append(self)

    def platform_headers(self) -> dict[str, str]:
        self.platform_headers_thread = threading.current_thread().name
        return {}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # ChatAnthropic reads the base URL from the environment; a developer box
    # pointing at a proxy must not change what these tests assert.
    for var in ("ANTHROPIC_API_URL", "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    _INSTANCES.clear()
    yield
    _INSTANCES.clear()


@pytest.fixture
def sdk_ctor() -> Iterator[Any]:
    with patch("anthropic.AsyncClient", side_effect=_FakeSdkClient) as ctor:
        yield ctor


def _model(cls: type[ChatAnthropic] = SharedClientChatAnthropic, **kwargs: Any) -> Any:
    kwargs.setdefault("anthropic_api_key", SecretStr("sk-test"))
    kwargs.setdefault("model", "claude-sonnet-4-6")
    return cls(streaming=True, **kwargs)  # type: ignore[call-arg]


def _provider(cls: type[ChatAnthropic] = SharedClientChatAnthropic, **kw: Any) -> Any:
    return _model(cls, **kw).configurable_fields(
        model=ConfigurableField(id="model"),
        temperature=ConfigurableField(id="temperature"),
        thinking=ConfigurableField(id="thinking"),
        max_tokens=ConfigurableField(id="max_tokens"),
    )


def _prime_in_thread(model: Any) -> None:
    thread = threading.Thread(target=lambda: model._async_client, name="executor")
    thread.start()
    thread.join()


def _per_call_copy(provider: Any) -> Any:
    """Return the concrete model a single ``ainvoke`` would run against."""
    bound = provider.with_config(config=_CONFIG)
    prepared, _ = bound._prepare(bound.config)
    return prepared


def test_plain_chat_anthropic_rebuilds_the_sdk_client_every_call(sdk_ctor: Any) -> None:
    """
    Pin the upstream behaviour the workaround exists for: one client per call.

    If this starts failing, langchain-core carries the cached client across
    ``_prepare`` and ``SharedClientChatAnthropic`` may no longer be needed.
    """
    provider = _provider(ChatAnthropic)
    _prime_in_thread(provider.default)
    assert sdk_ctor.call_count == 1

    for _ in range(3):
        copy = _per_call_copy(provider)
        assert copy is not provider.default
        _ = copy._async_client
    assert sdk_ctor.call_count == 4
    assert [c.thread for c in _INSTANCES] == ["executor"] + ["MainThread"] * 3


def test_shared_client_is_built_once_and_reused_by_every_copy(
    sdk_ctor: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    provider = _provider()
    _prime_in_thread(provider.default)
    assert sdk_ctor.call_count == 1
    primed = provider.default._async_client
    assert primed.thread == "executor"

    for _ in range(3):
        copy = _per_call_copy(provider)
        assert isinstance(copy, SharedClientChatAnthropic)
        assert copy is not provider.default
        assert "_async_client" not in copy.__dict__
        assert copy._async_client is primed

    assert sdk_ctor.call_count == 1
    assert "on the event loop" not in caplog.text
    assert caplog.text.count("Built the Anthropic SDK client") == 1


def test_bind_tools_and_model_copy_keep_the_shared_client(sdk_ctor: Any) -> None:
    """Every LangChain path that derives a model from the primed one shares it."""
    provider = _provider()
    _prime_in_thread(provider.default)
    primed = provider.default._async_client

    bound = provider.with_config(config=_CONFIG).bind_tools(
        [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    )
    assert bound.bound._async_client is primed
    assert provider.default.model_copy(update={"temperature": 0.1})._async_client is (
        primed
    )
    assert sdk_ctor.call_count == 1


def test_equal_params_share_across_independent_models(sdk_ctor: Any) -> None:
    """Sharing is by value of the client parameters, not by object identity."""
    primed = _model(default_headers={"X-A": "1", "X-B": "2"})
    _prime_in_thread(primed)
    same = _model(
        default_headers={"X-B": "2", "X-A": "1"},
        shared_async_client=primed.shared_async_client,
        shared_client_params=primed.shared_client_params,
    )
    assert same._async_client is primed._async_client
    assert sdk_ctor.call_count == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"anthropic_api_key": SecretStr("sk-other")},
        {"anthropic_api_url": "https://proxy.local"},
        {"anthropic_proxy": "http://proxy:3128"},
        {"max_retries": 9},
        {"default_request_timeout": 30.0},
        {"default_headers": {"X-A": "1"}},
    ],
)
async def test_copy_with_different_client_params_builds_its_own_client(
    sdk_ctor: Any, caplog: pytest.LogCaptureFixture, kwargs: dict[str, Any]
) -> None:
    """A copy must never reuse a client built for another endpoint or key."""
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    primed = _model()
    _prime_in_thread(primed)
    other = _model(
        shared_async_client=primed.shared_async_client,
        shared_client_params=primed.shared_client_params,
        **kwargs,
    )

    own = other._async_client

    assert own is not primed._async_client
    assert sdk_ctor.call_count == 2
    assert other.shared_async_client is own
    assert "on the event loop" in caplog.text


def test_shared_client_passes_the_base_class_parameters(sdk_ctor: Any) -> None:
    """Sharing changes where the client is built, not how."""
    assert sdk_ctor.call_count == 0
    provider = _provider()
    client = _per_call_copy(provider)._async_client
    assert client.params["api_key"] == "sk-test"
    assert client.params["base_url"] == "https://api.anthropic.com"
    assert "http_client" in client.params
    assert client.params["default_headers"]["User-Agent"]


@pytest.mark.usefixtures("sdk_ctor")
def test_serialization_omits_the_client() -> None:
    provider = _provider()
    _prime_in_thread(provider.default)
    dumped = provider.default.to_json()
    assert "shared_async_client" not in dumped.get("kwargs", {})
    assert "shared_client_params" not in dumped.get("kwargs", {})
    assert "shared_async_client" not in repr(provider.default)


def test_attribute_error_inside_construction_is_not_masked_by_pydantic() -> None:
    """Pydantic turns a swallowed AttributeError into 'no attribute _async_client'."""
    model = _model()

    def _renamed(_self: Any) -> Any:
        msg = "renamed"
        raise AttributeError(msg)

    with (
        patch.object(ChatAnthropic, "_client_params", property(_renamed)),
        pytest.raises(RuntimeError, match="renamed"),
    ):
        _ = model._async_client


async def test_async_prime_builds_the_client_in_the_executor(sdk_ctor: Any) -> None:
    executor_threads: list[str] = []

    class _Hass:
        async def async_add_executor_job(
            self, func: Callable[..., Any], *args: Any
        ) -> Any:
            def _run() -> Any:
                executor_threads.append(threading.current_thread().name)
                return func(*args)

            return await asyncio.get_running_loop().run_in_executor(None, _run)

    provider = _provider()
    await async_prime_async_client(_Hass(), provider.default)  # type: ignore[arg-type]

    assert sdk_ctor.call_count == 1
    primed = provider.default._async_client
    assert primed.thread == executor_threads[0] != "MainThread"
    # The SDK's platform headers read /etc/os-release on first use; the prime
    # takes that read off the loop as well.
    assert primed.platform_headers_thread == executor_threads[0]

    copy = _per_call_copy(provider)
    assert copy._async_client is primed
    assert sdk_ctor.call_count == 1


@pytest.mark.skipif(
    importlib.util.find_spec("anthropic.lib.credentials") is None,
    reason="anthropic < 0.98 has no credential discovery; nothing to observe",
)
async def test_real_sdk_reads_the_credential_file_once_off_the_loop() -> None:
    """
    Issue #618 against the real SDK: no mocks around the client constructor.

    Counts every read of ``~/.config/anthropic/active_config`` while priming
    and then serving five per-request copies, and asserts one read total, on
    an executor thread. This is the only test that fails on the SDK behaviour
    itself; the others patch ``anthropic.AsyncClient`` and cannot.
    """
    from anthropic.lib.credentials import _constants as credentials  # noqa: PLC0415

    reads: list[str] = []
    real_read = credentials._read_active_config_pointer

    def _counting_read() -> str | None:
        reads.append(threading.current_thread().name)
        return real_read()

    class _Hass:
        async def async_add_executor_job(
            self, func: Callable[..., Any], *args: Any
        ) -> Any:
            return await asyncio.get_running_loop().run_in_executor(None, func, *args)

    provider = _provider()
    with patch.object(credentials, "_read_active_config_pointer", _counting_read):
        await async_prime_async_client(_Hass(), provider.default)  # type: ignore[arg-type]
        primed = provider.default._async_client
        for _ in range(5):
            copy = _per_call_copy(provider)
            assert copy._async_client is primed
            copy._async_client.platform_headers()

    assert len(reads) == 1
    assert reads[0] != threading.current_thread().name
    assert "MainThread" not in reads
