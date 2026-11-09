# ruff: noqa: S101
"""Tests for the database connections' client-side keepalive settings (#937)."""

from __future__ import annotations

import psycopg
import pytest
from psycopg.conninfo import make_conninfo
from psycopg.pq import Conninfo

from custom_components.home_generative_agent.core.db_utils import (
    _DB_CLIENT_KEEPALIVES,
    db_client_keepalives,
)

BASE_URI = "postgresql://ha@db:5432/hga"


def test_every_setting_is_known_to_libpq() -> None:
    """Libpq rejects a connection naming an option it does not know."""
    known = {option.keyword.decode() for option in Conninfo.get_defaults()}
    assert _DB_CLIENT_KEEPALIVES.keys() <= known


def test_uri_without_keepalives_gets_the_defaults() -> None:
    """A dead server is given up on within about a minute."""
    settings = db_client_keepalives(f"{BASE_URI}?sslmode=disable")

    assert settings == _DB_CLIENT_KEEPALIVES
    probes = settings["keepalives_interval"] * settings["keepalives_count"]
    assert settings["tcp_user_timeout"] == (settings["keepalives_idle"] + probes) * 1000


def test_defaults_are_a_copy() -> None:
    """A caller changing its settings cannot change the next caller's."""
    db_client_keepalives(BASE_URI)["keepalives_idle"] = 1

    assert _DB_CLIENT_KEEPALIVES["keepalives_idle"] == 30


@pytest.mark.parametrize(
    "option",
    ["keepalives_idle=600", "keepalives_interval=5", "keepalives_count=9"],
)
def test_own_probe_timing_is_not_cut_short(option: str) -> None:
    """On Linux tcp_user_timeout would end the probes the URI asked for early."""
    uri = f"{BASE_URI}?{option}"
    key = option.split("=", maxsplit=1)[0]

    settings = db_client_keepalives(uri)

    expected = set(_DB_CLIENT_KEEPALIVES) - {key, "tcp_user_timeout"}
    assert set(settings) == expected
    assert option in make_conninfo(uri, **settings)


@pytest.mark.parametrize(
    "option", ["keepalives=0", "keepalives=1", "tcp_user_timeout=0"]
)
def test_other_own_settings_keep_the_remaining_defaults(option: str) -> None:
    """An option that sets no probe timing costs only itself."""
    uri = f"{BASE_URI}?{option}"
    key = option.split("=", maxsplit=1)[0]

    settings = db_client_keepalives(uri)

    assert set(settings) == set(_DB_CLIENT_KEEPALIVES) - {key}
    assert option in make_conninfo(uri, **settings)


def test_blank_option_counts_as_unset() -> None:
    """A blank row in Database connection params must not disable the defaults."""
    assert db_client_keepalives(f"{BASE_URI}?keepalives_idle=") == _DB_CLIENT_KEEPALIVES


def test_keyword_value_connection_string() -> None:
    """A libpq keyword/value string is read too, not only a URI."""
    assert "keepalives_idle" not in db_client_keepalives("host=db keepalives_idle=5")
    assert db_client_keepalives("host=db dbname=hga") == _DB_CLIENT_KEEPALIVES


def test_unix_socket_uri_is_accepted() -> None:
    """Libpq ignores these over a socket, so the defaults are harmless."""
    uri = "postgresql://ha@/hga?host=/var/run/postgresql"

    conninfo = make_conninfo(uri, **db_client_keepalives(uri))

    assert "host=/var/run/postgresql" in conninfo
    assert "keepalives_idle=30" in conninfo


def test_unparseable_uri_raises_what_connecting_would() -> None:
    """Every caller already handles this error from the connection."""
    with pytest.raises(psycopg.ProgrammingError):
        db_client_keepalives("postgresql://ha@db/hga?nonsense")
