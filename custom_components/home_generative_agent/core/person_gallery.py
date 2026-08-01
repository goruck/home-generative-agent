"""Data access layer for person recognition using pgvector cosine distance."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING
from urllib.parse import urljoin

import numpy as np
from homeassistant.helpers.httpx_client import get_async_client
from psycopg.rows import DictRow, dict_row

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from psycopg import AsyncConnection
    from psycopg_pool import AsyncConnectionPool

LOGGER = logging.getLogger(__name__)
Embedding = Sequence[float]
FACE_EMBEDDING_DIMS = 512


class PersonGalleryDAO:
    """Access layer for person recognition using pgvector cosine distance."""

    def __init__(
        self,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]],
        hass: HomeAssistant,
    ) -> None:
        """Initialize with a psycopg_pool AsyncConnectionPool."""
        self.pool = pool
        self._client = get_async_client(hass)

    def _normalize(self, embedding: Embedding) -> list[float]:
        """Return L2-normalized list of floats."""
        v = np.array(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            msg = "Zero vector cannot be normalized"
            raise ValueError(msg)
        return (v / norm).tolist()

    def _as_pgvector(self, embedding: Embedding) -> str:
        """Format floats as pgvector literal string with full precision."""
        if len(embedding) != FACE_EMBEDDING_DIMS:
            msg = f"Expected {FACE_EMBEDDING_DIMS} dims, got {len(embedding)}"
            raise ValueError(msg)
        return "[" + ",".join(format(float(x), ".17g") for x in embedding) + "]"

    async def enroll_from_image(
        self, face_api_url: str, name: str, image_bytes: bytes
    ) -> bool:
        """Detect face in image, extract embedding, and add to gallery."""
        resp = await self._client.post(
            urljoin(face_api_url.rstrip("/") + "/", "analyze"),
            files={"file": ("snapshot.jpg", image_bytes, "image/jpeg")},
        )
        resp.raise_for_status()
        data = resp.json()

        faces = data.get("faces", [])
        if not isinstance(faces, list) or not faces:
            LOGGER.warning("No face detected for enrollment of %s", name)
            return False

        emb = faces[0].get("embedding") if isinstance(faces[0], dict) else None
        if not isinstance(emb, list) or len(emb) != FACE_EMBEDDING_DIMS:
            LOGGER.warning("Invalid face embedding received for %s", name)
            return False
        await self.add_person(name, emb)
        LOGGER.info("Enrolled new person '%s' with embedding.", name)
        return True

    async def add_person(self, name: str, embedding: Embedding) -> None:
        """Insert normalized embedding into gallery (cosine distance ready)."""
        normed = self._normalize(embedding)
        vec_str = self._as_pgvector(normed)

        sql = """
            INSERT INTO public.person_gallery (name, embedding)
            VALUES (%s, %s::vector(512))
        """
        async with (
            self.pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cur,
        ):
            await cur.execute(sql, (name, vec_str))
            LOGGER.debug("Inserted %s into person_gallery", name)

    async def recognize_person(
        self, embedding: Embedding, threshold: float = 0.7
    ) -> str:
        """Return best cosine match or 'Unknown Person'."""
        normed = self._normalize(embedding)
        vec_str = self._as_pgvector(normed)

        sql = """
            SELECT name, embedding <=> %s::vector(512) AS distance
            FROM public.person_gallery
            ORDER BY distance
            LIMIT 1
        """
        async with (
            self.pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cur,
        ):
            await cur.execute(sql, (vec_str,))
            row = await cur.fetchone()

            if not row:
                LOGGER.error("Recognition query returned no rows")
                return "Unknown Person"

            dist = float(row["distance"])
            LOGGER.debug("Closest match=%s cosine_distance=%.6f", row["name"], dist)
            return row["name"] if dist < threshold else "Unknown Person"

    async def list_people(self) -> list[str]:
        """Return list of distinct enrolled person names."""
        async with (
            self.pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cur,
        ):
            await cur.execute(
                "SELECT DISTINCT name FROM public.person_gallery ORDER BY name"
            )
            rows = await cur.fetchall()
            return [r["name"] for r in rows]
