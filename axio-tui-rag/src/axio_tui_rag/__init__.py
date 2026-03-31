"""RAG tools for Axio TUI."""

from __future__ import annotations

from typing import Any, ClassVar

from axio.tool import ToolHandler
from pydantic import Field

from .vectordb import VectorStore


def _short(value: Any, limit: int = 60) -> str:
    s = str(value)
    return s if len(s) <= limit else s[:limit] + "..."


class IndexFiles(ToolHandler):
    """Index one or more files into the vector store for later semantic
    search. Files are chunked and embedded. Re-indexing a file replaces
    its old chunks. Use this before semantic_search to make file
    contents searchable."""

    paths: list[str] = Field(default_factory=list)
    _vector_store: ClassVar[VectorStore | None] = None

    def __repr__(self) -> str:
        return f"IndexFiles(paths={self.paths!r})"

    async def __call__(self) -> str:
        import asyncio
        import os
        from pathlib import Path

        if IndexFiles._vector_store is None:
            return "Vector store is not configured — no embedding model selected."
        results: list[str] = []
        for p in self.paths:
            file_path = Path(os.getcwd()) / p
            if not file_path.is_file():
                results.append(f"{p}: file not found")
                continue
            content = await asyncio.to_thread(file_path.read_text)
            count = await IndexFiles._vector_store.index_file(p, content)
            results.append(f"{p}: {count} chunks indexed")
        return "\n".join(results)


class SemanticSearch(ToolHandler):
    """Search previously indexed files using semantic similarity.
    Returns the most relevant text chunks with file paths and
    similarity scores. Files must be indexed first with index_files."""

    query: str = ""
    limit: int = 5
    _vector_store: ClassVar[VectorStore | None] = None

    def __repr__(self) -> str:
        return f"SemanticSearch(query={_short(self.query)!r})"

    async def __call__(self) -> str:
        if SemanticSearch._vector_store is None:
            return "Vector store is not configured — no embedding model selected."
        results = await SemanticSearch._vector_store.search(self.query, limit=self.limit)
        if not results:
            return "No results found."
        lines: list[str] = []
        for r in results:
            lines.append(f"[{r.path}] (score: {r.score:.4f})\n{r.chunk[:500]}")
        return "\n---\n".join(lines)


__all__ = ["IndexFiles", "SemanticSearch", "VectorStore"]
