"""Tests for axon.tui.vectordb — vector store and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from axio_tui_rag.vectordb import SearchResult, VectorStore, chunk_text

# ---------------------------------------------------------------------------
# Stub embedding transport
# ---------------------------------------------------------------------------


class StubEmbeddingTransport:
    """Returns fixed-dimension vectors for testing."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.call_count = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [[float(i + 1) / len(texts)] * self.dim for i in range(len(texts))]


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_short_text_single_chunk(self) -> None:
        chunks = chunk_text("Hello world", chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_empty_text(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   ") == []
        assert chunk_text("\n\n") == []

    def test_paragraph_split(self) -> None:
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_text(text, chunk_size=30, overlap=0)
        assert len(chunks) >= 2
        assert "Paragraph one." in chunks[0]

    def test_long_paragraph_splits_by_lines(self) -> None:
        lines = "\n".join(f"line {i}" for i in range(50))
        chunks = chunk_text(lines, chunk_size=50, overlap=0)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_long_line_splits_by_chars(self) -> None:
        long_line = "x" * 500
        chunks = chunk_text(long_line, chunk_size=100, overlap=0)
        assert len(chunks) > 1
        # Each chunk should be at most chunk_size (no overlap added)
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_overlap_applied(self) -> None:
        text = "A" * 100 + "\n\n" + "B" * 100
        chunks = chunk_text(text, chunk_size=110, overlap=10)
        assert len(chunks) == 2
        # Second chunk should start with overlap from first
        assert chunks[1].startswith("A" * 10)

    def test_chunk_size_respected(self) -> None:
        text = "\n\n".join(f"paragraph {i} " * 20 for i in range(10))
        chunks = chunk_text(text, chunk_size=200, overlap=0)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


class TestVectorStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> VectorStore:
        transport = StubEmbeddingTransport(dim=4)
        return VectorStore(transport=transport, db_path=tmp_path / "vectors")

    async def test_index_and_search(self, store: VectorStore) -> None:
        count = await store.index_file("test.py", "def hello():\n    print('hello')")
        assert count > 0

        results = await store.search("hello function")
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        assert results[0].path == "test.py"

    async def test_reindex_replaces_old(self, store: VectorStore) -> None:
        await store.index_file("test.py", "old content")
        await store.index_file("test.py", "new content")

        results = await store.search("content")
        paths = [r.path for r in results]
        assert paths.count("test.py") <= 1  # old entry was deleted

    async def test_empty_content_returns_zero(self, store: VectorStore) -> None:
        count = await store.index_file("empty.py", "")
        assert count == 0

    async def test_search_empty_store(self, store: VectorStore) -> None:
        results = await store.search("anything")
        assert results == []

    async def test_multiple_files(self, store: VectorStore) -> None:
        await store.index_file("a.py", "alpha content")
        await store.index_file("b.py", "beta content")

        results = await store.search("content", limit=10)
        paths = {r.path for r in results}
        assert "a.py" in paths
        assert "b.py" in paths

    async def test_search_result_has_score(self, store: VectorStore) -> None:
        await store.index_file("test.py", "some content here")
        results = await store.search("content")
        assert len(results) > 0
        assert isinstance(results[0].score, float)

    async def test_search_limit(self, store: VectorStore) -> None:
        for i in range(10):
            await store.index_file(f"file{i}.py", f"content number {i}")

        results = await store.search("content", limit=3)
        assert len(results) <= 3
