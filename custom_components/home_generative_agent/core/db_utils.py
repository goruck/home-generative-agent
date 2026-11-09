"""Various pgvector database utility functions."""

from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import (
    parse_qsl,
    quote,
    unquote,
    urlencode,
    urlparse,
    urlunsplit,
)

from homeassistant.const import (
    CONF_HOST,
    CONF_PASSWORD,
    CONF_PORT,
    CONF_USERNAME,
)
from psycopg.conninfo import conninfo_to_dict

from ..const import CONF_DB_NAME, CONF_DB_PARAMS  # noqa: TID252

# Defaults for noticing a server that vanished without closing the connection,
# which the OS otherwise takes about two hours to do (#937). Keepalives cover a
# connection waiting for an answer; they do not run while sent data is
# unacknowledged, which tcp_user_timeout covers. libpq ignores all of these
# over a Unix socket. A timeout in asyncio does not replace them: psycopg waits
# on, with no limit, after a cancelled query on a connection that is gone.
_DB_CLIENT_KEEPALIVES = {
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 3,
    "tcp_user_timeout": 60000,
}
_DB_KEEPALIVE_TIMING = {"keepalives_idle", "keepalives_interval", "keepalives_count"}


def db_client_keepalives(db_uri: str) -> dict[str, Any]:
    """
    Return the keepalive settings to add to a connection made with ``db_uri``.

    psycopg lets connection arguments override the URI, so an option the URI
    sets is left out; a blank one counts as unset, as it does in libpq. On
    Linux ``tcp_user_timeout`` also decides when keepalive probes give up, so
    it is left out too when the URI sets its own probe timing.
    """
    set_by_uri = {key for key, value in conninfo_to_dict(db_uri).items() if value}
    defaults = dict(_DB_CLIENT_KEEPALIVES)
    if set_by_uri & _DB_KEEPALIVE_TIMING:
        del defaults["tcp_user_timeout"]
    return {key: value for key, value in defaults.items() if key not in set_by_uri}


def parse_postgres_uri(uri: str) -> dict[str, Any]:
    """
    Parse a postgres URI into components.

    Returns keys: username, password, host, port, dbname, params
                  (list of {"key","value"}).
    """
    parsed = urlparse(uri)
    username = unquote(parsed.username) if parsed.username else None
    password = unquote(parsed.password) if parsed.password else None
    host = parsed.hostname
    port = parsed.port
    dbname = (
        parsed.path[1:]
        if parsed.path and parsed.path.startswith("/")
        else (parsed.path or None)
    )

    params: list[dict[str, str]] = []
    if parsed.query:
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            params.append({"key": key, "value": value})

    return {
        "username": username,
        "password": password,
        "host": host,
        "port": port,
        "dbname": dbname,
        "params": params,
    }


def _build_postgres_params(
    params_list: Iterable[Mapping[str, Any]] | None,
) -> dict[str, str]:
    """Build a PostgreSQL Params dict from validated database form data."""
    params: dict[str, str] = {}
    if params_list:
        for item in params_list:
            key = str(item["key"]).strip()
            value = str(item["value"]).strip()
            if key:
                params[key] = value
    return params


def build_postgres_uri(data: dict[str, Any]) -> str:
    """Build a PostgreSQL URI from validated database form data."""
    # Percent-encoded so an '@', ':', '/', '?' or '#' in either cannot end the
    # field early: libpq would read the rest as the host and echo it, part of
    # the password included, in its connection error (#748).
    username = quote(str(data[CONF_USERNAME]), safe="")
    password = quote(str(data[CONF_PASSWORD]), safe="")
    host = data[CONF_HOST]
    port = data.get(CONF_PORT)
    db_name = data[CONF_DB_NAME]
    params = _build_postgres_params(data.get(CONF_DB_PARAMS))

    if port is not None:
        netloc = f"{username}:{password}@{host}:{int(port)}"
    else:
        netloc = f"{username}:{password}@{host}"

    return urlunsplit(
        (  # scheme, netloc, path, query, fragment
            "postgresql",
            netloc,
            f"/{db_name}",
            urlencode(params) if params else "",
            "",
        )
    )
