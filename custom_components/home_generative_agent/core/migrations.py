"""Migrations for person_gallery feature."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from psycopg import AsyncConnection, AsyncCursor
    from psycopg.rows import DictRow
    from psycopg_pool import AsyncConnectionPool


async def _migration_1(cur: AsyncCursor[DictRow]) -> None:
    """Migration 1: create person_gallery and bump schema version."""
    await cur.execute(
        """
        CREATE TABLE IF NOT EXISTS person_gallery (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            embedding VECTOR(512),
            added_at TIMESTAMP DEFAULT NOW()
        )
        """
    )
    await cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_person_gallery_embedding
        ON person_gallery USING ivfflat (embedding vector_l2_ops)
        WITH (lists = 100)
        """
    )

    # bump version to 1
    await cur.execute(
        "INSERT INTO hga_schema_version (id, version) VALUES (1, 1) "
        "ON CONFLICT(id) DO UPDATE SET version = 1"
    )


MIGRATIONS: dict[int, Callable] = {1: _migration_1}


async def migrate_person_gallery(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
) -> None:
    """Run pending migrations for person_gallery."""
    async with pool.connection() as conn, conn.cursor() as cur:
        # --- Schema version table ---
        await cur.execute(
            """
            CREATE TABLE IF NOT EXISTS hga_schema_version (
                id INTEGER PRIMARY KEY DEFAULT 1,
                version INTEGER NOT NULL
            )
            """
        )
        await cur.execute("SELECT version FROM hga_schema_version WHERE id = 1")

        row: dict[str, int] | None = await cur.fetchone()
        current_version: int = row["version"] if row else 0

        # --- Run pending migrations ---
        for version in sorted(MIGRATIONS.keys()):
            if current_version < version:
                await MIGRATIONS[version](cur)
                current_version = version
