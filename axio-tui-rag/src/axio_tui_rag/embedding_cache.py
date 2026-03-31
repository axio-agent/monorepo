"""CachedEmbeddingTransport: SQLite-backed embedding cache."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import aiosqlite
from axio.transport import EmbeddingTransport

logger = logging.getLogger(__name__)

_CREATE_TABLE = "CREATE TABLE IF NOT EXISTS embedding_cache (  key TEXT PRIMARY KEY,  embedding TEXT NOT NULL)"


def _cache_key(model_id: str, text: str) -> str:
    return hashlib.sha256(f"{model_id}\0{text}".encode()).hexdigest()


class CachedEmbeddingTransport:
    """Transparent caching wrapper around an EmbeddingTransport.

    Computes ``sha256(model_id + '\\0' + text)`` as the cache key and stores
    the embedding vector as JSON in SQLite.  Uncached texts are forwarded to
    the inner transport in a single batch call.
    """

    def __init__(self, transport: EmbeddingTransport, db_path: Path, model_id: str) -> None:
        self._transport = transport
        self._db_path = db_path
        self._model_id = model_id
        self._conn: aiosqlite.Connection | None = None

    async def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(str(self._db_path))
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA busy_timeout=5000")
            await self._conn.execute("PRAGMA synchronous=NORMAL")
            await self._conn.execute(_CREATE_TABLE)
        return self._conn

    async def _lookup(self, conn: aiosqlite.Connection, keys: list[str]) -> dict[str, list[float]]:
        """Batch-fetch cached embeddings by key."""
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        query = f"SELECT key, embedding FROM embedding_cache WHERE key IN ({placeholders})"  # noqa: S608
        async with conn.execute(query, keys) as cursor:
            rows = await cursor.fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    async def _store(self, conn: aiosqlite.Connection, items: list[tuple[str, list[float]]]) -> None:
        """Batch-insert embeddings into cache."""
        if not items:
            return
        await conn.executemany(
            "INSERT OR IGNORE INTO embedding_cache (key, embedding) VALUES (?, ?)",
            [(key, json.dumps(vec)) for key, vec in items],
        )
        await conn.commit()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        conn = await self._ensure_conn()

        keys = [_cache_key(self._model_id, t) for t in texts]
        cached = await self._lookup(conn, keys)

        miss_indices = [i for i, k in enumerate(keys) if k not in cached]

        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            vectors = await self._transport.embed(miss_texts)
            to_store: list[tuple[str, list[float]]] = []
            for idx, vec in zip(miss_indices, vectors):
                cached[keys[idx]] = vec
                to_store.append((keys[idx], vec))
            await self._store(conn, to_store)
            logger.debug("Embedding cache: %d hits, %d misses", len(texts) - len(miss_indices), len(miss_indices))
        else:
            logger.debug("Embedding cache: %d hits, 0 misses", len(texts))

        return [cached[k] for k in keys]

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
