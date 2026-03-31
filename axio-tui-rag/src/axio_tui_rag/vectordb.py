"""LanceDB-backed vector store for semantic search."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lancedb  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
from axio.transport import EmbeddingTransport


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks by paragraphs, then lines, then chars."""
    if not text.strip():
        return []

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If a single paragraph exceeds chunk_size, split by lines
            if len(para) > chunk_size:
                lines = para.split("\n")
                current = ""
                for line in lines:
                    line_candidate = f"{current}\n{line}".strip() if current else line
                    if len(line_candidate) <= chunk_size:
                        current = line_candidate
                    else:
                        if current:
                            chunks.append(current)
                        # If a single line exceeds chunk_size, split by chars
                        if len(line) > chunk_size:
                            for i in range(0, len(line), chunk_size - overlap):
                                chunks.append(line[i : i + chunk_size])
                            current = ""
                        else:
                            current = line
            else:
                current = para

    if current:
        chunks.append(current)

    # Apply overlap between paragraph-level chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            prefix = prev[-overlap:] if len(prev) > overlap else prev
            overlapped.append(prefix + "\n" + chunks[i])
        chunks = overlapped

    return chunks


@dataclass(frozen=True, slots=True)
class SearchResult:
    path: str
    chunk: str
    score: float


@dataclass(slots=True)
class VectorStore:
    transport: EmbeddingTransport
    db_path: Path = field(default_factory=lambda: Path.home() / ".local" / "share" / "axio-vectors")
    _db: Any = field(default=None, init=False, repr=False)
    _table: Any = field(default=None, init=False, repr=False)
    _dim: int = field(default=0, init=False, repr=False)

    _TABLE_NAME = "vectors"

    async def _get_db(self) -> Any:
        if self._db is None:
            self.db_path.mkdir(parents=True, exist_ok=True)
            self._db = await lancedb.connect_async(str(self.db_path))
        return self._db

    async def _get_or_create_table(self, dim: int) -> Any:
        db = await self._get_db()
        if self._table is not None:
            return self._table
        schema = pa.schema(
            [
                pa.field("path", pa.utf8()),
                pa.field("chunk", pa.utf8()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ]
        )
        try:
            self._table = await db.open_table(self._TABLE_NAME)
        except Exception:
            self._table = await db.create_table(self._TABLE_NAME, schema=schema)
        return self._table

    async def index_file(self, path: str, content: str) -> int:
        """Chunk, embed, and upsert file content. Returns number of chunks indexed."""
        chunks = chunk_text(content)
        if not chunks:
            return 0

        vectors = await self.transport.embed(chunks)
        if self._dim == 0:
            self._dim = len(vectors[0])

        table = await self._get_or_create_table(self._dim)
        # Delete old entries for this path
        try:
            await table.delete(f"path = '{path}'")
        except Exception:
            pass
        rows = [{"path": path, "chunk": c, "vector": v} for c, v in zip(chunks, vectors)]
        await table.add(rows)
        return len(rows)

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Embed query and perform vector search."""
        vectors = await self.transport.embed([query])
        query_vec = vectors[0]
        if self._dim == 0:
            self._dim = len(query_vec)

        try:
            table = await self._get_or_create_table(self._dim)
        except Exception:
            return []
        results = await table.vector_search(query_vec).limit(limit).to_list()
        return [SearchResult(path=row["path"], chunk=row["chunk"], score=float(row["_distance"])) for row in results]
