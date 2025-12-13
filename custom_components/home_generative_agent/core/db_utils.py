from typing import Any
from collections.abc import Iterable, Mapping

from urllib.parse import parse_qsl, unquote, urlparse, urlencode, urlunsplit
from homeassistant.const import (
    CONF_USERNAME,
    CONF_PASSWORD,
    CONF_HOST,
    CONF_PORT,
)
from ..const import (
    CONF_DB_NAME,
    CONF_DB_PARAMS,
)

def parse_postgres_uri(uri: str) -> dict[str, Any]:
    """
    Parse a postgres URI into components.

    Returns keys: username, password, host, port, dbname, params (list of {"key","value"}).
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
    username = data[CONF_USERNAME]
    password = data[CONF_PASSWORD]
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
