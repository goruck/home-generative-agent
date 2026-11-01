"""Shared test configuration."""

import os

# langchain-openai >= 1.6 probes a throwaway socket when it builds a
# ChatOpenAI client, to drop TCP keepalive options the kernel rejects. The
# library tolerates a blocked socket, but the HA test plugin fails any test
# that tried to open one. The probe never connects, so turning the keepalive
# options off in tests loses nothing.
os.environ.setdefault("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "0")


import pytest


@pytest.fixture(autouse=True)
def _no_chat_grace_between_tests() -> None:
    """
    Reset the chat-grace clock before every test (#866).

    ``core.utils._chat_last_end`` is a module global set by every edge chat
    turn; left over from one test it would defer Sentinel admission in an
    unrelated test for 30 s of wall-clock time.
    """
    import custom_components.home_generative_agent.core.utils as utils_mod  # noqa: PLC0415

    utils_mod._chat_last_end = 0.0
    utils_mod._uncontended = False
    utils_mod._sentinel_last_call_end = 0.0


@pytest.fixture(autouse=True)
def default_database_reachable(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """
    Skip the network half of every database test the flows run (#765).

    Without it every test that runs Basic setup would open a real PostgreSQL
    connection to ``localhost``, which may reach a developer's own database.
    Only ``core.utils._db_probe`` (connect, query, close) is replaced, so the
    URI checks ``validate_db_uri`` makes before connecting still run; a URI
    that passes them looks reachable. A test that needs it unreachable patches
    ``validate_db_uri`` or ``_db_probe`` itself. Returns the URIs probed.
    """
    from custom_components.home_generative_agent.core import (  # noqa: PLC0415
        utils,
    )

    probed: list[str] = []

    async def _reachable(db_uri: str) -> None:
        probed.append(db_uri)

    monkeypatch.setattr(utils, "_db_probe", _reachable)
    return probed
