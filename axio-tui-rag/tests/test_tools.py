"""Tests for axio-tui-rag — IndexFiles and SemanticSearch tool handlers."""

from __future__ import annotations

import os
from pathlib import Path

from axio_tui_rag import IndexFiles, SemanticSearch
from axio_tui_rag.vectordb import VectorStore


class _StubEmbeddingTransport:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 4 for _ in texts]


class TestIndexFiles:
    async def test_no_store_returns_error(self) -> None:
        IndexFiles._vector_store = None
        handler = IndexFiles(paths=["test.py"])
        result = await handler()
        assert "not configured" in result

    async def test_file_not_found(self, tmp_path: Path) -> None:
        transport = _StubEmbeddingTransport()
        store = VectorStore(transport=transport, db_path=tmp_path / "vectors")
        IndexFiles._vector_store = store
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            handler = IndexFiles(paths=["missing.py"])
            result = await handler()
            assert "file not found" in result
        finally:
            os.chdir(old_cwd)
            IndexFiles._vector_store = None

    async def test_index_success(self, tmp_path: Path) -> None:
        (tmp_path / "hello.py").write_text("def hello(): pass")
        transport = _StubEmbeddingTransport()
        store = VectorStore(transport=transport, db_path=tmp_path / "vectors")
        IndexFiles._vector_store = store
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            handler = IndexFiles(paths=["hello.py"])
            result = await handler()
            assert "chunks indexed" in result
        finally:
            os.chdir(old_cwd)
            IndexFiles._vector_store = None

    async def test_repr(self) -> None:
        handler = IndexFiles(paths=["a.py", "b.py"])
        assert "a.py" in repr(handler)


class TestSemanticSearch:
    async def test_no_store_returns_error(self) -> None:
        SemanticSearch._vector_store = None
        handler = SemanticSearch(query="test")
        result = await handler()
        assert "not configured" in result

    async def test_search_empty_store(self, tmp_path: Path) -> None:
        transport = _StubEmbeddingTransport()
        store = VectorStore(transport=transport, db_path=tmp_path / "vectors")
        SemanticSearch._vector_store = store
        try:
            handler = SemanticSearch(query="anything")
            result = await handler()
            assert "No results" in result
        finally:
            SemanticSearch._vector_store = None

    async def test_search_returns_results(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("def greet(): print('hi')")
        transport = _StubEmbeddingTransport()
        store = VectorStore(transport=transport, db_path=tmp_path / "vectors")
        IndexFiles._vector_store = store
        SemanticSearch._vector_store = store
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            await IndexFiles(paths=["code.py"])()
            handler = SemanticSearch(query="greeting function")
            result = await handler()
            assert "code.py" in result
        finally:
            os.chdir(old_cwd)
            IndexFiles._vector_store = None
            SemanticSearch._vector_store = None

    async def test_repr(self) -> None:
        handler = SemanticSearch(query="find something")
        assert "find something" in repr(handler)
