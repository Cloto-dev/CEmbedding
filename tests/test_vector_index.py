"""VectorIndex tests.

Ported from clotohub-servers servers/tests/test_embedding_index.py when the
monorepo copy was retired in favor of this repository (2026-07-09), plus the
matrix-search equivalence suite and the single-resident-matrix layout tests.
"""

import sqlite3
import struct
import sys
import types

import numpy as np
import pytest

from cembedding.server import VectorIndex, _VectorGroup


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


class ConstantProvider:
    """Every text maps to one shared unit vector, so all scores tie exactly."""

    def __init__(self, dim: int = 384):
        self._dim = dim
        vec = np.random.default_rng(7).random(dim).astype(np.float32)
        self._vec = (vec / np.linalg.norm(vec)).tolist()

    async def embed(self, texts):
        return [list(self._vec) for _ in texts]

    def dimensions(self):
        return self._dim


def live_rows(group: _VectorGroup):
    """(item_id, vector) of every live row, in row order — the search order."""
    return [(group.ids[row], group.matrix[row]) for row in range(group.n) if group.live[row]]


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
    group = idx._groups["ns"][384]
    row = group.rows["x"]
    vec1 = group.matrix[row].copy()
    await idx.index("ns", [{"id": "x", "text": "second"}], provider)
    vec2 = group.matrix[group.rows["x"]]

    assert not np.allclose(vec1, vec2)
    assert group.rows["x"] == row  # overwritten in place, so it keeps its position
    assert group.n == 1  # and does not consume a second row
    assert await idx.count("ns") == 1

    await idx._db.close()


def test_blob_pack_unpack_roundtrip():
    original = np.random.default_rng(42).random(384).astype(np.float32)
    blob = struct.pack(f"<{len(original)}f", *original)
    restored = np.frombuffer(blob, dtype=np.float32)
    np.testing.assert_array_equal(original, restored)
    # index() writes blobs with tobytes(); on a little-endian host that is byte
    # for byte what struct.pack("<f") produced, so old rows stay readable.
    assert original.astype(np.float32, copy=False).tobytes() == blob


@pytest.mark.asyncio
async def test_empty_namespace_search():
    import aiosqlite

    idx = VectorIndex(":memory:")
    idx._db = await aiosqlite.connect(":memory:")
    results = await idx.search("nonexistent", "query", 10, 0.0, HashProvider())
    assert results == []
    await idx._db.close()


# ── matrix search ──


@pytest.mark.asyncio
async def test_matrix_search_matches_per_item_loop():
    """Matrix search returns exactly what the per-item np.dot loop did."""
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
    for item_id, vec in live_rows(idx._groups["ns"][384]):
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
async def test_writes_are_visible_without_rebuilding_the_matrix():
    """index/remove reach the next search while reusing the same allocation."""
    idx = await make_index()
    provider = HashProvider()

    await idx.index("ns", [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}], provider)
    await idx.search("ns", "alpha", 10, 0.0, provider)
    group = idx._groups["ns"][384]
    matrix = group.matrix
    assert group.n < group.capacity  # there is room to append without reallocating

    await idx.index("ns", [{"id": "c", "text": "gamma"}], provider)
    assert idx._groups["ns"][384] is group
    assert group.matrix is matrix  # appended into spare capacity, nothing re-stacked
    results = await idx.search("ns", "gamma", 10, 0.0, provider)
    assert "c" in [r["id"] for r in results]

    await idx.remove("ns", ["c"])
    assert group.matrix is matrix  # tombstoned, still no rebuild
    results = await idx.search("ns", "gamma", 10, 0.0, provider)
    assert "c" not in [r["id"] for r in results]

    await idx.purge_namespace("ns")
    assert "ns" not in idx._groups
    assert await idx.search("ns", "alpha", 10, 0.0, provider) == []

    await idx._db.close()


@pytest.mark.asyncio
async def test_readd_after_remove_lands_at_the_end_of_the_order():
    """A removed id that comes back is the newest row, so it loses ties again."""
    idx = await make_index()
    provider = ConstantProvider()  # identical vectors, so every score ties exactly

    items = [{"id": item_id, "text": item_id} for item_id in ("a", "b", "c")]
    await idx.index("ns", items, provider)
    results = await idx.search("ns", "q", 10, 0.0, provider)
    assert [r["id"] for r in results] == ["a", "b", "c"]

    await idx.remove("ns", ["a"])
    await idx.index("ns", [{"id": "a", "text": "a"}], provider)
    results = await idx.search("ns", "q", 10, 0.0, provider)
    assert [r["id"] for r in results] == ["b", "c", "a"]
    assert await idx.count("ns") == 3

    await idx._db.close()


@pytest.mark.asyncio
async def test_compaction_preserves_results_and_order():
    """Once tombstones outnumber live rows the matrix is compacted in place."""
    idx = await make_index()
    provider = ConstantProvider()  # identical vectors, so order is the only variable

    items = [{"id": f"m{i}", "text": f"m{i}"} for i in range(10)]
    await idx.index("ns", items, provider)
    group = idx._groups["ns"][384]
    assert group.n == 10

    # Re-adding m0 moves it to the back, so row order is no longer id order.
    await idx.remove("ns", ["m0"])
    await idx.index("ns", [{"id": "m0", "text": "m0"}], provider)
    expected = ["m2", "m4", "m6", "m8", "m0"]

    dropped = ["m1", "m3", "m5", "m7", "m9"]  # 6 dead rows of 11, past the half mark
    assert await idx.remove("ns", dropped) == len(dropped)
    assert (group.n, group.dead) == (5, 0)  # compacted
    assert group.ids[: group.n] == expected
    assert [group.rows[item_id] for item_id in expected] == [0, 1, 2, 3, 4]

    results = await idx.search("ns", "q", 10, 0.0, provider)
    assert [r["id"] for r in results] == expected
    assert await idx.count("ns") == 5

    await idx._db.close()


@pytest.mark.asyncio
async def test_remove_counts_only_the_rows_that_existed():
    """The return value is SQLite's delete count: unknown and repeated ids do not add to it."""
    idx = await make_index()
    provider = HashProvider()

    await idx.index("ns", [{"id": f"m{i}", "text": f"m{i}"} for i in range(4)], provider)

    assert await idx.remove("ns", ["m0", "never-indexed"]) == 1
    assert await idx.remove("ns", ["m1", "m1"]) == 1  # a repeated id deletes one row
    assert await idx.remove("ns", ["never-indexed"]) == 0
    assert await idx.remove("other-ns", ["m2"]) == 0  # namespaces do not share ids
    assert await idx.count("ns") == 2

    await idx._db.close()


@pytest.mark.asyncio
async def test_capacity_growth_keeps_every_vector_searchable():
    """Appending past the initial allocation reallocates without losing rows."""
    idx = await make_index()
    provider = HashProvider()

    items = [{"id": f"m{i}", "text": f"memory {i}"} for i in range(200)]
    await idx.index("ns", items, provider)
    group = idx._groups["ns"][384]
    assert group.capacity > _VectorGroup.MIN_CAPACITY
    assert group.n == 200

    results = await idx.search("ns", "memory 7", 200, -1.0, provider)
    assert {r["id"] for r in results} == {item["id"] for item in items}
    assert await idx.count("ns") == 200

    # Rows from before and after each reallocation still hold their vectors:
    # a text's own row is its nearest neighbour, at a similarity of 1.
    for row in (3, 70, 150):
        top = await idx.search("ns", f"memory {row}", 1, -1.0, provider)
        assert top == [{"id": f"m{row}", "score": 1.0}]

    await idx._db.close()


@pytest.mark.asyncio
async def test_upsert_with_new_dimension_moves_the_id_between_groups():
    """One id holds one vector, so a re-index at another width retires the old row."""
    idx = await make_index()

    await idx.index("ns", [{"id": "x", "text": "small"}], HashProvider(dim=384))
    await idx.index("ns", [{"id": "y", "text": "other"}], HashProvider(dim=384))
    await idx.index("ns", [{"id": "x", "text": "large"}], HashProvider(dim=768))

    assert "x" not in idx._groups["ns"][384].rows
    assert "x" in idx._groups["ns"][768].rows
    assert await idx.count("ns") == 2

    results = await idx.search("ns", "anything", 10, -1.0, HashProvider(dim=384))
    assert [r["id"] for r in results] == ["y"]
    results = await idx.search("ns", "anything", 10, -1.0, HashProvider(dim=768))
    assert [r["id"] for r in results] == ["x"]

    await idx._db.close()


@pytest.mark.asyncio
async def test_indexed_vectors_survive_a_reopen(tmp_path):
    """index() writes what initialize() reads back: same vectors, same results."""
    db_path = str(tmp_path / "roundtrip.db")
    provider = HashProvider()

    idx = VectorIndex(db_path)
    await idx.initialize()
    await idx.index("ns", [{"id": f"m{i}", "text": f"memory {i}"} for i in range(5)], provider)
    await idx.index("ns", [{"id": "wide", "text": "other width"}], HashProvider(dim=768))
    before = await idx.search("ns", "memory 3", 10, 0.0, provider)
    matrix_before = idx._groups["ns"][384].matrix[:5].copy()
    assert len(before) == 5
    await idx.shutdown()

    reopened = VectorIndex(db_path)
    await reopened.initialize()
    assert await reopened.count("ns") == 6
    group = reopened._groups["ns"][384]
    assert group.ids == [f"m{i}" for i in range(5)]
    np.testing.assert_array_equal(group.matrix[:5], matrix_before)
    assert await reopened.search("ns", "memory 3", 10, 0.0, provider) == before
    await reopened.shutdown()


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


# ── search backend ──


class _FakeArray:
    """Stand-in for a backend array: not an ndarray, but convertible to one."""

    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    def __array__(self, dtype=None, copy=None):
        return self.data if dtype is None else self.data.astype(dtype)


def _install_fake_mlx(monkeypatch):
    """Register a numpy-backed stand-in for mlx and count matrix conversions."""
    core = types.ModuleType("mlx.core")
    core.matrix_conversions = 0

    def array(data):
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:
            core.matrix_conversions += 1
        return _FakeArray(arr)

    core.array = array
    core.matmul = lambda a, b: _FakeArray(a.data @ b.data)

    package = types.ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return core


@pytest.mark.asyncio
async def test_mlx_backend_converts_the_matrix_only_after_a_write(monkeypatch):
    """The resident backend copy is rebuilt lazily, and only when the matrix changed."""
    idx = await make_index()
    provider = HashProvider()
    await idx.index("ns", [{"id": f"m{i}", "text": f"memory {i}"} for i in range(20)], provider)
    expected = await idx.search("ns", "memory 5", 5, 0.0, provider)  # numpy backend

    core = _install_fake_mlx(monkeypatch)
    monkeypatch.setattr("cembedding.server.EMBEDDING_SEARCH_BACKEND", "mlx")

    assert await idx.search("ns", "memory 5", 5, 0.0, provider) == expected
    assert core.matrix_conversions == 1
    assert await idx.search("ns", "memory 5", 5, 0.0, provider) == expected
    assert core.matrix_conversions == 1  # no write in between, so no rebuild

    await idx.index("ns", [{"id": "late", "text": "memory 5"}], provider)
    results = await idx.search("ns", "memory 5", 5, 0.0, provider)
    assert core.matrix_conversions == 2  # the write invalidated the backend copy
    assert "late" in [r["id"] for r in results]

    await idx._db.close()
