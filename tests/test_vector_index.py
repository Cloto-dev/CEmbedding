"""VectorIndex tests.

Ported from clotohub-servers servers/tests/test_embedding_index.py when the
monorepo copy was retired in favor of this repository (2026-07-09), plus the
v0.6.0 matrix-search equivalence/invalidation suite.
"""

import sqlite3
import struct

import numpy as np
import pytest

from cembedding.server import VectorIndex


class HashProvider:
    """Deterministic unit vectors keyed by text hash."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    async def embed(self, texts):
        results = []
        for text in texts:
            vec = np.random.default_rng(hash(text) % 2**31).random(self._dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            results.append(vec.tolist())
        return results

    def dimensions(self):
        return self._dim


async def make_index() -> VectorIndex:
    import aiosqlite

    idx = VectorIndex(":memory:")
    idx._db = await aiosqlite.connect(":memory:")
    await idx._db.executescript(
        """
        CREATE TABLE vectors (
            namespace TEXT NOT NULL,
            item_id   TEXT NOT NULL,
            vector    BLOB NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (namespace, item_id)
        );
        """
    )
    await idx._db.commit()
    idx._index = {}
    return idx


@pytest.mark.asyncio
async def test_lifecycle():
    """Index → search → remove lifecycle."""
    idx = await make_index()
    provider = HashProvider()

    count = await idx.index("test:ns", [{"id": "a", "text": "hello"}, {"id": "b", "text": "world"}], provider)
    assert count == 2
    assert await idx.count("test:ns") == 2

    results = await idx.search("test:ns", "hello", 10, 0.0, provider)
    assert len(results) > 0
    assert results[0]["id"] in ("a", "b")
    assert "score" in results[0]

    removed = await idx.remove("test:ns", ["a"])
    assert removed == 1
    assert await idx.count("test:ns") == 1

    results = await idx.search("test:ns", "hello", 10, 0.0, provider)
    assert all(r["id"] != "a" for r in results)

    await idx._db.close()


@pytest.mark.asyncio
async def test_namespace_isolation():
    idx = await make_index()
    provider = HashProvider()

    await idx.index("cpersona:alice", [{"id": "m1", "text": "alice memory"}], provider)
    await idx.index("cpersona:bob", [{"id": "m1", "text": "bob memory"}], provider)

    assert await idx.count("cpersona:alice") == 1
    assert await idx.count("cpersona:bob") == 1

    await idx.remove("cpersona:alice", ["m1"])
    assert await idx.count("cpersona:alice") == 0
    assert await idx.count("cpersona:bob") == 1

    await idx._db.close()


@pytest.mark.asyncio
async def test_upsert_replaces_vector():
    idx = await make_index()
    call_count = 0

    class BumpProvider(HashProvider):
        async def embed(self, texts):
            nonlocal call_count
            call_count += 1
            results = []
            for text in texts:
                vec = np.random.default_rng(hash(text) % 2**31 + call_count).random(384).astype(np.float32)
                results.append((vec / np.linalg.norm(vec)).tolist())
            return results

    provider = BumpProvider()
    await idx.index("ns", [{"id": "x", "text": "first"}], provider)
    vec1 = idx._index["ns"]["x"].copy()
    await idx.index("ns", [{"id": "x", "text": "second"}], provider)
    vec2 = idx._index["ns"]["x"]

    assert not np.allclose(vec1, vec2)
    assert await idx.count("ns") == 1

    await idx._db.close()


def test_blob_pack_unpack_roundtrip():
    original = np.random.default_rng(42).random(384).astype(np.float32)
    blob = struct.pack(f"<{len(original)}f", *original)
    restored = np.frombuffer(blob, dtype=np.float32)
    np.testing.assert_array_equal(original, restored)


@pytest.mark.asyncio
async def test_empty_namespace_search():
    import aiosqlite

    idx = VectorIndex(":memory:")
    idx._db = await aiosqlite.connect(":memory:")
    idx._index = {}
    results = await idx.search("nonexistent", "query", 10, 0.0, HashProvider())
    assert results == []
    await idx._db.close()


# ── v0.6.0 matrix search ──


@pytest.mark.asyncio
async def test_matrix_search_matches_per_item_loop():
    """v0.6.0 matrix search returns exactly what the per-item np.dot loop did."""
    idx = await make_index()
    provider = HashProvider()

    items = [{"id": f"m{i}", "text": f"memory number {i}"} for i in range(200)]
    await idx.index("ns", items, provider)

    query = "memory number 42"
    got = await idx.search("ns", query, 10, 0.1, provider)

    # Reference: the pre-v0.6.0 per-item loop over the same in-memory index.
    import heapq

    qvec = np.array((await provider.embed([query]))[0], dtype=np.float32)
    candidates = []
    for item_id, vec in idx._index["ns"].items():
        sim = float(np.dot(qvec, vec))
        if sim >= 0.1:
            candidates.append((sim, item_id))
    expected = [
        {"id": item_id, "score": round(score, 4)}
        for score, item_id in heapq.nlargest(10, candidates, key=lambda x: x[0])
    ]

    assert [r["id"] for r in got] == [r["id"] for r in expected]
    for g, e in zip(got, expected):
        assert abs(g["score"] - e["score"]) <= 1e-4  # BLAS vs per-row rounding

    await idx._db.close()


@pytest.mark.asyncio
async def test_matrix_cache_invalidated_on_writes():
    """index/remove/purge each drop the namespace's cached search matrix."""
    idx = await make_index()
    provider = HashProvider()

    await idx.index("ns", [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}], provider)
    await idx.search("ns", "alpha", 10, 0.0, provider)
    assert "ns" in idx._matrix_cache

    await idx.index("ns", [{"id": "c", "text": "gamma"}], provider)
    assert "ns" not in idx._matrix_cache
    results = await idx.search("ns", "gamma", 10, 0.0, provider)
    assert "c" in [r["id"] for r in results]

    await idx.remove("ns", ["c"])
    assert "ns" not in idx._matrix_cache
    results = await idx.search("ns", "gamma", 10, 0.0, provider)
    assert "c" not in [r["id"] for r in results]

    await idx.purge_namespace("ns")
    assert "ns" not in idx._matrix_cache
    assert await idx.search("ns", "alpha", 10, 0.0, provider) == []

    await idx._db.close()


@pytest.mark.asyncio
async def test_matrix_search_mixed_dimensions():
    """Vectors of a different dimension are excluded, like the old len() skip."""
    idx = await make_index()

    await idx.index("ns", [{"id": "d384", "text": "small"}], HashProvider(dim=384))
    await idx.index("ns", [{"id": "d768", "text": "large"}], HashProvider(dim=768))

    results = await idx.search("ns", "anything", 10, -1.0, HashProvider(dim=384))
    assert [r["id"] for r in results] == ["d384"]
    results = await idx.search("ns", "anything", 10, -1.0, HashProvider(dim=768))
    assert [r["id"] for r in results] == ["d768"]

    await idx._db.close()


@pytest.mark.asyncio
async def test_persisted_index_loads_and_searches(tmp_path):
    """initialize() loads SQLite-persisted vectors into the search path."""
    db_path = str(tmp_path / "index.db")
    rng = np.random.default_rng(3)
    v = rng.random(384).astype(np.float32)
    v /= np.linalg.norm(v)

    db = sqlite3.connect(db_path)
    db.executescript(
        """
        CREATE TABLE vectors (
            namespace TEXT NOT NULL,
            item_id   TEXT NOT NULL,
            vector    BLOB NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (namespace, item_id)
        );
        """
    )
    db.execute(
        "INSERT INTO vectors (namespace, item_id, vector) VALUES (?, ?, ?)",
        ("cpersona:agent", "mem:1", struct.pack(f"<{len(v)}f", *v)),
    )
    db.commit()
    db.close()

    idx = VectorIndex(db_path)
    await idx.initialize()
    assert await idx.count("cpersona:agent") == 1

    class EchoProvider:
        async def embed(self, texts):
            return [v.tolist()]

        def dimensions(self):
            return 384

    results = await idx.search("cpersona:agent", "q", 5, 0.0, EchoProvider())
    assert results[0]["id"] == "mem:1"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-3)
    await idx.shutdown()
