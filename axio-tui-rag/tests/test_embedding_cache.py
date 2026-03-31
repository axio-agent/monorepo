"""Tests for axon.tui.embedding_cache — CachedEmbeddingTransport."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from axio_tui_rag.embedding_cache import CachedEmbeddingTransport, _cache_key


class StubEmbeddingTransport:
    """Records calls and returns deterministic vectors."""

    def __init__(self) -> None:
        self.call_count = 0
        self.texts_received: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        self.texts_received.append(texts)
        return [[float(ord(t[0])), float(len(t))] for t in texts]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_embed_cache.db"


@pytest.fixture
def inner() -> StubEmbeddingTransport:
    return StubEmbeddingTransport()


def test_cache_key_deterministic() -> None:
    k1 = _cache_key("model-a", "hello")
    k2 = _cache_key("model-a", "hello")
    assert k1 == k2


def test_cache_key_differs_by_model() -> None:
    k1 = _cache_key("model-a", "hello")
    k2 = _cache_key("model-b", "hello")
    assert k1 != k2


def test_cache_key_differs_by_text() -> None:
    k1 = _cache_key("model-a", "hello")
    k2 = _cache_key("model-a", "world")
    assert k1 != k2


def test_cache_key_is_sha256() -> None:
    key = _cache_key("m", "t")
    expected = hashlib.sha256(b"m\0t").hexdigest()
    assert key == expected


async def test_cache_miss_calls_transport(db_path: Path, inner: StubEmbeddingTransport) -> None:
    cache = CachedEmbeddingTransport(inner, db_path, "model-a")
    try:
        result = await cache.embed(["hello"])
        assert len(result) == 1
        assert inner.call_count == 1
        assert inner.texts_received == [["hello"]]
    finally:
        await cache.close()


async def test_cache_hit_skips_transport(db_path: Path, inner: StubEmbeddingTransport) -> None:
    cache = CachedEmbeddingTransport(inner, db_path, "model-a")
    try:
        first = await cache.embed(["hello"])
        second = await cache.embed(["hello"])
        assert first == second
        assert inner.call_count == 1
    finally:
        await cache.close()


async def test_mixed_batch_partial_hit(db_path: Path, inner: StubEmbeddingTransport) -> None:
    cache = CachedEmbeddingTransport(inner, db_path, "model-a")
    try:
        await cache.embed(["hello"])
        assert inner.call_count == 1

        result = await cache.embed(["hello", "world"])
        assert inner.call_count == 2
        # Only "world" should have been sent to the transport
        assert inner.texts_received[1] == ["world"]
        assert len(result) == 2
    finally:
        await cache.close()


async def test_different_model_causes_miss(db_path: Path, inner: StubEmbeddingTransport) -> None:
    cache_a = CachedEmbeddingTransport(inner, db_path, "model-a")
    cache_b = CachedEmbeddingTransport(inner, db_path, "model-b")
    try:
        await cache_a.embed(["hello"])
        assert inner.call_count == 1

        await cache_b.embed(["hello"])
        assert inner.call_count == 2
    finally:
        await cache_a.close()
        await cache_b.close()


async def test_cache_persists_across_instances(db_path: Path, inner: StubEmbeddingTransport) -> None:
    cache1 = CachedEmbeddingTransport(inner, db_path, "model-a")
    try:
        first = await cache1.embed(["hello"])
    finally:
        await cache1.close()
    assert inner.call_count == 1

    cache2 = CachedEmbeddingTransport(inner, db_path, "model-a")
    try:
        second = await cache2.embed(["hello"])
    finally:
        await cache2.close()
    assert inner.call_count == 1
    assert first == second


async def test_preserves_order(db_path: Path, inner: StubEmbeddingTransport) -> None:
    cache = CachedEmbeddingTransport(inner, db_path, "model-a")
    try:
        result = await cache.embed(["abc", "xy"])
        assert result[0] == [float(ord("a")), 3.0]
        assert result[1] == [float(ord("x")), 2.0]
    finally:
        await cache.close()


async def test_all_cached_no_transport_call(db_path: Path, inner: StubEmbeddingTransport) -> None:
    cache = CachedEmbeddingTransport(inner, db_path, "model-a")
    try:
        await cache.embed(["a", "b", "c"])
        assert inner.call_count == 1

        await cache.embed(["b", "a", "c"])
        assert inner.call_count == 1
    finally:
        await cache.close()


async def test_close_idempotent(db_path: Path, inner: StubEmbeddingTransport) -> None:
    cache = CachedEmbeddingTransport(inner, db_path, "model-a")
    await cache.embed(["test"])
    await cache.close()
    await cache.close()


async def test_empty_input(db_path: Path, inner: StubEmbeddingTransport) -> None:
    cache = CachedEmbeddingTransport(inner, db_path, "model-a")
    try:
        result = await cache.embed([])
        assert result == []
        assert inner.call_count == 0
    finally:
        await cache.close()
