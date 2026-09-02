"""Contiguous sidecar tests.

What the sidecar promises is that nothing observable changes: a search through
(mapped base + in-memory tail) returns exactly what the same database returns
when every vector is read from SQLite, the live set after a start is SQLite's
set, a file that cannot be trusted is ignored with a reason, and the file only
appears where the startup policy says it should.

The equivalence test compares ids, order and scores with ``==`` rather than a
tolerance: the point of writing the rows in rowid order and concatenating the
segments base-first is that the arithmetic and the tie order are unchanged, and
a tolerance would hide exactly the difference worth catching. Its corpus is
built from vectors whose dot products are exactly representable (see
``GridProvider``), so the comparison is decided by row order rather than by
which BLAS kernel a segment's row count happened to select.
"""

import asyncio
import hashlib
import json
import os
import pathlib
import random
import sqlite3
import struct

import numpy as np
import pytest

from cembedding import sidecar
from cembedding.server import VectorIndex

NS_MAIN = "corpus:main"
NS_SIDE = "corpus:side"
NS_TIES = "corpus:ties"


class TextProvider:
    """Deterministic unit vectors keyed by text, so identical texts score identically."""

    def __init__(self, dim: int):
        self._dim = dim
        self._cache: dict[str, list[float]] = {}

    def vector(self, text: str) -> list[float]:
        vec = self._cache.get(text)
        if vec is None:
            digest = hashlib.blake2b(f"{self._dim}:{text}".encode(), digest_size=8).digest()
            arr = np.random.default_rng(int.from_bytes(digest, "little")).standard_normal(self._dim).astype(np.float32)
            arr /= np.linalg.norm(arr)
            vec = arr.tolist()
            self._cache[text] = vec
        return vec

    async def embed(self, texts):
        return [list(self.vector(text)) for text in texts]

    def dimensions(self) -> int:
        return self._dim


class GridProvider:
    """Deterministic vectors whose dot products do not depend on summation order.

    Splitting one matmul into two hands the smaller part to a different BLAS
    kernel, and a kernel that accumulates in another order can return the
    neighbouring float: on this machine, rows 28-29 of a 30-row matmul and the
    same two rows multiplied alone already differ in the last bit. That is
    invisible in a reported score, but it decides the order of two rows that
    would otherwise tie exactly, which is the property the equivalence test is
    about. Coordinates here are multiples of 1/128 within [-1/16, 1/16], so
    every product and every partial sum is exactly representable in float32 and
    the two paths cannot disagree about a score at all.
    """

    def __init__(self, dim: int):
        self._dim = dim
        self._cache: dict[str, list[float]] = {}

    def vector(self, text: str) -> list[float]:
        vec = self._cache.get(text)
        if vec is None:
            digest = hashlib.blake2b(f"grid:{self._dim}:{text}".encode(), digest_size=8).digest()
            rng = np.random.default_rng(int.from_bytes(digest, "little"))
            arr = rng.integers(-8, 9, size=self._dim).astype(np.float32) / np.float32(128)
            vec = arr.tolist()
            self._cache[text] = vec
        return vec

    async def embed(self, texts):
        return [list(self.vector(text)) for text in texts]

    def dimensions(self) -> int:
        return self._dim


P384 = TextProvider(384)
P768 = TextProvider(768)
G384 = GridProvider(384)
G768 = GridProvider(768)


# ── helpers ──

_OPENED: list = []


@pytest.fixture(autouse=True)
def close_indexes_left_open():
    """Close the database of every index a test opened, pass or fail.

    An index holds an aiosqlite connection, and that connection runs a thread
    the interpreter waits for on the way out. A test that fails before its own
    shutdown would otherwise leave the thread alive and the process would hang
    after reporting the failure — which is exactly when the report matters.
    The connection is closed directly rather than through shutdown() so that
    cleaning up cannot write a sidecar the test did not ask for.
    """
    _OPENED.clear()
    yield
    leftover = [index for index in _OPENED if index._db is not None]
    _OPENED.clear()
    if leftover:

        async def close_all():
            for index in leftover:
                await index._db.close()
                index._db = None

        asyncio.run(close_all())


async def open_index(monkeypatch, db_path, *, mode="auto", min_rows=10**9, max_tail=10.0**6) -> VectorIndex:
    """Open an index with the sidecar settings stated.

    The defaults are deliberately unreachable thresholds: a test that is not
    about the build policy never triggers a build, so the state it set up is
    the state it examines. Tests about the policy state their own numbers.
    """
    monkeypatch.setenv("EMBEDDING_SIDECAR", mode)
    monkeypatch.setenv("EMBEDDING_SIDECAR_MIN_ROWS", str(min_rows))
    monkeypatch.setenv("EMBEDDING_SIDECAR_MAX_TAIL", str(max_tail))
    index = VectorIndex(str(db_path))
    await index.initialize()
    _OPENED.append(index)
    return index


def groups_of(index: VectorIndex):
    return [group for ns_groups in index._groups.values() for group in ns_groups.values()]


def base_live_total(index: VectorIndex) -> int:
    return sum(len(group.base_rows) for group in groups_of(index))


def tail_live_total(index: VectorIndex) -> int:
    return sum(len(group.rows) for group in groups_of(index))


def resident_items(index: VectorIndex) -> set:
    """Every (namespace, item_id) the resident segments still hold."""
    items = set()
    for namespace, ns_groups in index._groups.items():
        for group in ns_groups.values():
            items |= {(namespace, item_id) for item_id in group.base_rows}
            items |= {(namespace, item_id) for item_id in group.rows}
    return items


def sqlite_items(db_path) -> set:
    conn = sqlite3.connect(str(db_path))
    try:
        return set(conn.execute("SELECT namespace, item_id FROM vectors"))
    finally:
        conn.close()


def sqlite_count(db_path, namespace=None) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        if namespace is None:
            return conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM vectors WHERE namespace = ?", (namespace,)).fetchone()[0]
    finally:
        conn.close()


def read_header(path) -> dict:
    raw = pathlib.Path(path).read_bytes()
    _, _, header_len = struct.unpack("<8sII", raw[:16])
    return json.loads(raw[16 : 16 + header_len])


def rewrite_watermark(path, watermark: int) -> None:
    """Re-encode a built sidecar with a different watermark, blocks untouched.

    The file stays internally consistent — same layout rules, same blocks, so
    the length check still passes — which is what makes it a test of the
    reconciliation rather than of the format checks.
    """
    raw = pathlib.Path(path).read_bytes()
    header = read_header(path)
    entries, blocks = [], []
    for group in header["groups"]:
        entry = sidecar._Entry(group["namespace"], group["dim"])
        entry.ids = list(group["ids"])
        entry.rowids = list(group["rowids"])
        entries.append(entry)
        start = group["offset"]
        blocks.append(raw[start : start + group["count"] * group["dim"] * 4])
    pathlib.Path(path).write_bytes(sidecar._encode_header(watermark, entries) + b"".join(blocks))


async def build_corpus(monkeypatch, db_path) -> None:
    """Three namespaces, two widths, and groups of items whose scores tie exactly."""
    index = await open_index(monkeypatch, db_path, mode="off")
    await index.index(NS_MAIN, [{"id": f"main-{i}", "text": f"document {i}"} for i in range(3000)], G384)
    await index.index(NS_MAIN, [{"id": f"wide-{i}", "text": f"wide document {i}"} for i in range(200)], G768)
    await index.index(NS_SIDE, [{"id": f"side-{i}", "text": f"side note {i}"} for i in range(300)], G384)
    ties = [{"id": f"tie-{g}-{k}", "text": f"tied text {g}"} for g in range(6) for k in range(5)]
    await index.index(NS_TIES, ties, G384)
    await index.shutdown()


async def apply_writes_after_build(monkeypatch, db_path) -> None:
    """Everything a caller can do to a corpus the sidecar has already snapshotted.

    No removal takes the row that currently holds the largest rowid, because
    SQLite hands a freed largest rowid to the next insert and the index would
    then (correctly) refuse the file and read everything from SQLite, which is
    a different path from the one this fixture is here to exercise.
    """
    index = await open_index(monkeypatch, db_path, mode="off")
    # Overwrites of base items: each lands in the tail with a new vector.
    await index.index(NS_MAIN, [{"id": f"main-{i}", "text": f"revised document {i}"} for i in range(0, 600, 3)], G384)
    # Removals of base items.
    await index.remove(NS_MAIN, [f"main-{i}" for i in range(1000, 1200)])
    # Items that exist only in the tail, two of which are then removed.
    await index.index(NS_MAIN, [{"id": f"late-{i}", "text": f"late document {i}"} for i in range(120)], G384)
    await index.remove(NS_MAIN, ["late-5", "late-6"])
    # A removed base item that comes back.
    await index.index(NS_MAIN, [{"id": "main-1000", "text": "restored document 1000"}], G384)
    # Base items that change width, so they move to another group entirely.
    await index.index(NS_MAIN, [{"id": f"main-{i}", "text": f"switched {i}"} for i in range(2000, 2020)], G768)
    # Ties spanning both segments: one member removed, one overwritten into the
    # tail, one new member added after them.
    await index.remove(NS_TIES, ["tie-1-2"])
    await index.index(NS_TIES, [{"id": "tie-0-0", "text": "tied text 0"}], G384)
    await index.index(NS_TIES, [{"id": "tie-new", "text": "tied text 0"}], G384)
    await index.shutdown()


QUERY_PARAMS = [(10, 0.0), (5, 0.03), (100, -1.0), (1, 0.0), (3, 0.5), (0, 0.02), (25, 0.01)]


async def run_queries(index: VectorIndex, count: int = 100) -> list:
    """The same seeded query sequence for any index, so two runs are comparable."""
    rng = random.Random(20260902)
    results = []
    for i in range(count):
        limit, floor = QUERY_PARAMS[i % len(QUERY_PARAMS)]
        namespace = rng.choice([NS_MAIN, NS_MAIN, NS_MAIN, NS_SIDE, NS_TIES, "corpus:absent"])
        wide = namespace == NS_MAIN and i % 5 == 0
        text = rng.choice(
            [
                f"document {rng.randrange(3000)}",
                f"revised document {rng.randrange(0, 600, 3)}",
                f"late document {rng.randrange(120)}",
                f"tied text {rng.randrange(6)}",
                f"side note {rng.randrange(300)}",
                f"wide document {rng.randrange(200)}" if wide else f"switched {rng.randrange(2000, 2020)}",
            ]
        )
        results.append(await index.search(namespace, text, limit, floor, G768 if wide else G384))
    return results


# ── equivalence ──


@pytest.mark.asyncio
async def test_sidecar_search_matches_the_sqlite_only_index(tmp_path, monkeypatch):
    """Mapped base + tail returns the same ids, order and scores as SQLite alone."""
    db_path = tmp_path / "index.db"
    await build_corpus(monkeypatch, db_path)
    sidecar.build(str(db_path))
    await apply_writes_after_build(monkeypatch, db_path)

    mapped = await open_index(monkeypatch, db_path)
    plain = await open_index(monkeypatch, db_path, mode="off")

    # Without this the comparison could pass by having both sides fall back.
    assert base_live_total(mapped) > 3000
    assert tail_live_total(mapped) > 300
    assert base_live_total(mapped) + tail_live_total(mapped) == sqlite_count(db_path)
    assert base_live_total(plain) == 0

    from_sidecar = await run_queries(mapped)
    from_sqlite = await run_queries(plain)
    assert sum(len(result) for result in from_sqlite) > 500  # the comparison is not vacuous
    assert from_sidecar == from_sqlite

    await mapped.shutdown()
    await plain.shutdown()


@pytest.mark.asyncio
async def test_live_items_match_sqlite_after_random_writes(tmp_path, monkeypatch):
    """After a scripted-random session, the resident set is exactly SQLite's set."""
    db_path = tmp_path / "index.db"
    index = await open_index(monkeypatch, db_path, mode="off")
    await index.index(NS_MAIN, [{"id": f"main-{i}", "text": f"document {i}"} for i in range(400)], P384)
    await index.index(NS_SIDE, [{"id": f"side-{i}", "text": f"side note {i}"} for i in range(150)], P384)
    await index.shutdown()
    sidecar.build(str(db_path))

    rng = random.Random(4242)
    index = await open_index(monkeypatch, db_path, mode="off")
    newest = None  # never removed: freeing the largest rowid hands it to the next insert
    for step in range(60):
        namespace = rng.choice([NS_MAIN, NS_SIDE])
        action = rng.choice(["add", "add", "overwrite", "remove", "remove"])
        if action == "add":
            newest = f"extra-{step}"
            await index.index(namespace, [{"id": newest, "text": f"extra text {step}"}], P384)
        elif action == "overwrite":
            target = f"main-{rng.randrange(400)}" if namespace == NS_MAIN else f"side-{rng.randrange(150)}"
            newest = target
            await index.index(namespace, [{"id": target, "text": f"rewritten {step}"}], P384)
        else:
            victims = [f"main-{rng.randrange(400)}" for _ in range(4)]
            await index.remove(namespace, [victim for victim in victims if victim != newest])
    await index.shutdown()

    reopened = await open_index(monkeypatch, db_path)
    assert base_live_total(reopened) > 0  # the file was used, not silently bypassed
    assert resident_items(reopened) == sqlite_items(db_path)
    for namespace in (NS_MAIN, NS_SIDE):
        assert await reopened.count(namespace) == sqlite_count(db_path, namespace)
    await reopened.shutdown()


# ── freshness ──


@pytest.mark.asyncio
async def test_a_watermark_past_the_real_rows_does_not_hide_rows(tmp_path, monkeypatch):
    """A watermark that claims rows the file does not hold must not lose them.

    Rows above the watermark are the ones a start reads from SQLite, so a
    watermark moved past them says "the base already covers these". It does
    not, and every one of them has to still be searchable.
    """
    db_path = tmp_path / "index.db"
    index = await open_index(monkeypatch, db_path, mode="off")
    await index.index(NS_MAIN, [{"id": f"main-{i}", "text": f"document {i}"} for i in range(50)], P384)
    await index.shutdown()
    sidecar.build(str(db_path))

    index = await open_index(monkeypatch, db_path, mode="off")
    await index.index(NS_MAIN, [{"id": f"late-{i}", "text": f"late document {i}"} for i in range(20)], P384)
    await index.shutdown()

    path = sidecar.default_path(str(db_path))
    rewrite_watermark(path, read_header(path)["watermark"] + 1000)

    reopened = await open_index(monkeypatch, db_path)
    assert resident_items(reopened) == sqlite_items(db_path)
    assert await reopened.count(NS_MAIN) == 70
    for i in (0, 7, 19):
        top = await reopened.search(NS_MAIN, f"late document {i}", 1, 0.0, P384)
        assert top == [{"id": f"late-{i}", "score": 1.0}]
    # The rows the file does not hold cannot be served from it, so the whole
    # resident set has to come from SQLite.
    assert base_live_total(reopened) == 0
    await reopened.shutdown()


@pytest.mark.asyncio
async def test_an_overwrite_after_the_build_serves_the_new_vector(tmp_path, monkeypatch):
    """An id in the base that is written again moves to the tail, old row tombstoned."""
    db_path = tmp_path / "index.db"
    index = await open_index(monkeypatch, db_path, mode="off")
    await index.index(NS_MAIN, [{"id": f"m{i}", "text": f"original {i}"} for i in range(5)], P384)
    await index.shutdown()
    sidecar.build(str(db_path))

    index = await open_index(monkeypatch, db_path, mode="off")
    await index.index(NS_MAIN, [{"id": "m2", "text": "replacement text"}], P384)
    await index.shutdown()

    reopened = await open_index(monkeypatch, db_path)
    group = reopened._groups[NS_MAIN][384]
    assert group.base_count == 5  # the snapshot still holds every original row
    assert "m2" not in group.base_rows and group.base_dead == 1  # but not as a live one
    assert "m2" in group.rows  # the current vector is in the tail
    assert await reopened.count(NS_MAIN) == 5

    assert await reopened.search(NS_MAIN, "replacement text", 1, 0.0, P384) == [{"id": "m2", "score": 1.0}]
    stale = await reopened.search(NS_MAIN, "original 2", 5, 0.99, P384)
    assert stale == []  # the old vector is still mapped, and still unreachable
    await reopened.shutdown()


@pytest.mark.asyncio
async def test_the_build_writes_an_overwritten_item_once_at_its_newest_position(tmp_path, monkeypatch):
    """INSERT OR REPLACE moves an item's rowid, and the build follows that order.

    All three items score identically here, so their order in the results is
    their row order and nothing else: an item that was rewritten belongs at the
    end, exactly where a load straight from SQLite would put it.
    """
    db_path = tmp_path / "index.db"
    index = await open_index(monkeypatch, db_path, mode="off")
    tied = "tied text"
    await index.index(NS_TIES, [{"id": item, "text": tied} for item in ("a", "b", "c")], P384)
    await index.index(NS_TIES, [{"id": "b", "text": tied}], P384)  # b is rewritten, so b moves last
    await index.shutdown()
    sidecar.build(str(db_path))

    reopened = await open_index(monkeypatch, db_path)
    group = reopened._groups[NS_TIES][384]
    assert group.base_ids == ["a", "c", "b"]  # written once, at its newest position
    assert group.n == 0  # and nothing was left for the tail to re-read
    results = await reopened.search(NS_TIES, tied, 10, 0.0, P384)
    assert [hit["id"] for hit in results] == ["a", "c", "b"]
    await reopened.shutdown()


@pytest.mark.asyncio
async def test_the_mlx_backend_converts_each_segment_once(tmp_path, monkeypatch):
    """A non-numpy backend converts the base when it is first touched, and no more.

    The base is a read-only mapping that no write can change — a removal only
    clears a flag applied to the scores — so its converted copy stays valid for
    the process, while the tail's is rebuilt after every write as before.
    """
    from test_vector_index import _install_fake_mlx  # one fake backend for the suite

    db_path = tmp_path / "index.db"
    index = await open_index(monkeypatch, db_path, mode="off")
    await index.index(NS_MAIN, [{"id": f"m{i}", "text": f"document {i}"} for i in range(20)], P384)
    await index.shutdown()
    sidecar.build(str(db_path))
    index = await open_index(monkeypatch, db_path, mode="off")
    await index.index(NS_MAIN, [{"id": "late", "text": "late document"}], P384)
    await index.shutdown()

    reopened = await open_index(monkeypatch, db_path)
    expected = await reopened.search(NS_MAIN, "document 3", 5, 0.0, P384)  # numpy backend

    core = _install_fake_mlx(monkeypatch)
    monkeypatch.setattr("cembedding.server.EMBEDDING_SEARCH_BACKEND", "mlx")
    assert await reopened.search(NS_MAIN, "document 3", 5, 0.0, P384) == expected
    assert core.matrix_conversions == 2  # base and tail, one each
    assert await reopened.search(NS_MAIN, "document 3", 5, 0.0, P384) == expected
    assert core.matrix_conversions == 2  # nothing changed, so nothing was rebuilt

    await reopened.remove(NS_MAIN, ["m4"])  # tombstones a base row
    assert await reopened.search(NS_MAIN, "document 3", 5, 0.0, P384) == expected
    assert core.matrix_conversions == 2  # the base rows themselves did not change

    await reopened.index(NS_MAIN, [{"id": "later", "text": "later document"}], P384)
    await reopened.search(NS_MAIN, "document 3", 5, 0.0, P384)
    assert core.matrix_conversions == 3  # the write invalidated the tail copy only
    await reopened.shutdown()


@pytest.mark.asyncio
async def test_a_namespace_created_after_the_build_has_no_base(tmp_path, monkeypatch):
    """A group the file never saw is simply a group whose rows are all tail."""
    db_path = tmp_path / "index.db"
    index = await open_index(monkeypatch, db_path, mode="off")
    await index.index(NS_MAIN, [{"id": f"m{i}", "text": f"document {i}"} for i in range(20)], P384)
    await index.shutdown()
    sidecar.build(str(db_path))

    index = await open_index(monkeypatch, db_path, mode="off")
    await index.index(NS_SIDE, [{"id": f"s{i}", "text": f"side note {i}"} for i in range(7)], P384)
    await index.index(NS_MAIN, [{"id": "wide", "text": "wide document"}], P768)  # a new width, too
    await index.shutdown()

    reopened = await open_index(monkeypatch, db_path)
    late = reopened._groups[NS_SIDE][384]
    assert (late.base_count, late.n) == (0, 7)
    assert (reopened._groups[NS_MAIN][768].base_count, reopened._groups[NS_MAIN][768].n) == (0, 1)
    assert (reopened._groups[NS_MAIN][384].base_count, reopened._groups[NS_MAIN][384].n) == (20, 0)
    assert await reopened.count(NS_SIDE) == 7
    assert await reopened.search(NS_SIDE, "side note 3", 1, 0.0, P384) == [{"id": "s3", "score": 1.0}]
    assert await reopened.search(NS_MAIN, "wide document", 1, 0.0, P768) == [{"id": "wide", "score": 1.0}]
    await reopened.shutdown()


# ── corruption ──


def truncate_file(path) -> str:
    raw = pathlib.Path(path).read_bytes()
    pathlib.Path(path).write_bytes(raw[: len(raw) - 4096])
    return "length mismatch"


def wrong_magic(path) -> str:
    with open(path, "r+b") as handle:
        handle.write(b"NOTASIDE")
    return "bad magic"


def foreign_version(path) -> str:
    with open(path, "r+b") as handle:
        handle.seek(8)
        handle.write(struct.pack("<I", sidecar.FORMAT_VERSION + 1))
    return "format version"


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt", [truncate_file, wrong_magic, foreign_version])
async def test_a_file_that_cannot_be_trusted_is_ignored_and_reported(tmp_path, monkeypatch, corrupt):
    db_path = tmp_path / "index.db"
    await build_corpus(monkeypatch, db_path)
    sidecar.build(str(db_path))
    rows = sqlite_count(db_path)

    path = sidecar.default_path(str(db_path))
    expected_reason = corrupt(path)

    report = sidecar.status(str(db_path))
    assert report["present"] is True
    assert report["ok"] is False
    assert expected_reason in report["reason"]
    assert report["sqlite_rows"] == rows

    reopened = await open_index(monkeypatch, db_path)
    assert base_live_total(reopened) == 0  # nothing was mapped
    assert tail_live_total(reopened) == rows
    plain = await open_index(monkeypatch, db_path, mode="off")
    results = await run_queries(reopened, count=30)
    assert sum(len(result) for result in results) > 100
    assert results == await run_queries(plain, count=30)
    await reopened.shutdown()
    await plain.shutdown()


@pytest.mark.asyncio
async def test_a_corrupt_file_is_replaced_once_the_corpus_earns_one(tmp_path, monkeypatch):
    """The fallback is not permanent: a start that would build one builds over it."""
    db_path = tmp_path / "index.db"
    index = await open_index(monkeypatch, db_path, mode="off")
    await index.index(NS_MAIN, [{"id": f"main-{i}", "text": f"document {i}"} for i in range(300)], P384)
    await index.shutdown()
    sidecar.build(str(db_path))
    path = sidecar.default_path(str(db_path))
    wrong_magic(path)

    reopened = await open_index(monkeypatch, db_path, min_rows=100)
    assert sidecar.status(str(db_path))["ok"] is True
    assert base_live_total(reopened) == 300
    assert await reopened.count(NS_MAIN) == 300
    await reopened.shutdown()


def test_status_describes_a_healthy_file(tmp_path):
    db_path = tmp_path / "index.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE vectors (
            namespace TEXT NOT NULL, item_id TEXT NOT NULL, vector BLOB NOT NULL,
            created_at TEXT DEFAULT (datetime('now')), PRIMARY KEY (namespace, item_id)
        );
        """
    )
    vector = np.arange(4, dtype=np.float32).tobytes()
    conn.executemany(
        "INSERT INTO vectors (namespace, item_id, vector) VALUES (?, ?, ?)",
        [(NS_MAIN, f"m{i}", vector) for i in range(6)],
    )
    conn.commit()
    conn.close()

    absent = sidecar.status(str(db_path))
    assert absent["present"] is False and absent["ok"] is False and absent["reason"] == "no sidecar file"
    assert absent["sqlite_rows"] == 6

    sidecar.build(str(db_path), chunk_rows=4)  # more than one chunk, to exercise the loop
    report = sidecar.status(str(db_path))
    assert report["ok"] is True and report["reason"] is None
    assert (report["base_rows"], report["watermark"], report["tail_rows"]) == (6, 6, 0)
    assert report["bytes"] == os.path.getsize(report["path"])

    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO vectors (namespace, item_id, vector) VALUES (?, ?, ?)", (NS_MAIN, "m6", vector))
    conn.commit()
    conn.close()
    assert sidecar.status(str(db_path))["tail_rows"] == 1


def test_command_line_builds_and_reports(tmp_path, capsys):
    db_path = tmp_path / "index.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE vectors (
            namespace TEXT NOT NULL, item_id TEXT NOT NULL, vector BLOB NOT NULL,
            created_at TEXT DEFAULT (datetime('now')), PRIMARY KEY (namespace, item_id)
        );
        """
    )
    conn.execute(
        "INSERT INTO vectors (namespace, item_id, vector) VALUES (?, ?, ?)",
        (NS_MAIN, "m0", np.arange(8, dtype=np.float32).tobytes()),
    )
    conn.commit()
    conn.close()

    assert sidecar.main(["status", "--db", str(db_path)]) == 0  # absent is not an error
    assert sidecar.main(["build", "--db", str(db_path)]) == 0
    capsys.readouterr()
    assert sidecar.main(["status", "--db", str(db_path)]) == 0
    out = capsys.readouterr().out
    assert "usable:   yes" in out and "1 rows" in out

    wrong_magic(sidecar.default_path(str(db_path)))
    assert sidecar.main(["status", "--db", str(db_path)]) == 1  # present but unusable
    assert "bad magic" in capsys.readouterr().out


# ── startup and shutdown policy ──


@pytest.mark.asyncio
async def test_below_the_minimum_no_file_is_created(tmp_path, monkeypatch):
    db_path = tmp_path / "index.db"
    path = sidecar.default_path(str(db_path))
    index = await open_index(monkeypatch, db_path, min_rows=100)
    await index.index(NS_MAIN, [{"id": f"m{i}", "text": f"document {i}"} for i in range(40)], P384)
    assert not os.path.exists(path)
    await index.shutdown()  # a tail alone does not earn a file for a corpus this small
    assert not os.path.exists(path)

    reopened = await open_index(monkeypatch, db_path, min_rows=100)
    assert await reopened.count(NS_MAIN) == 40
    assert not os.path.exists(path)
    await reopened.shutdown()


@pytest.mark.asyncio
async def test_at_the_minimum_a_start_builds_and_maps_the_file(tmp_path, monkeypatch):
    db_path = tmp_path / "index.db"
    path = sidecar.default_path(str(db_path))
    index = await open_index(monkeypatch, db_path, mode="off")
    await index.index(NS_MAIN, [{"id": f"m{i}", "text": f"document {i}"} for i in range(40)], P384)
    await index.shutdown()
    assert not os.path.exists(path)

    reopened = await open_index(monkeypatch, db_path, min_rows=40)
    assert os.path.exists(path)
    # The rows this start read from SQLite were released for the mapping.
    assert base_live_total(reopened) == 40
    assert tail_live_total(reopened) == 0
    assert await reopened.count(NS_MAIN) == 40
    await reopened.shutdown()


@pytest.mark.asyncio
async def test_a_grown_tail_triggers_a_rebuild_at_startup(tmp_path, monkeypatch):
    db_path = tmp_path / "index.db"
    index = await open_index(monkeypatch, db_path, mode="off")
    await index.index(NS_MAIN, [{"id": f"m{i}", "text": f"document {i}"} for i in range(100)], P384)
    await index.shutdown()
    sidecar.build(str(db_path))
    index = await open_index(monkeypatch, db_path, mode="off")
    await index.index(NS_MAIN, [{"id": f"late-{i}", "text": f"late document {i}"} for i in range(40)], P384)
    await index.shutdown()

    path = sidecar.default_path(str(db_path))
    kept = await open_index(monkeypatch, db_path, max_tail=1.0)  # 40 tail rows over 100 base: within budget
    assert tail_live_total(kept) == 40
    assert sidecar.status(str(db_path))["tail_rows"] == 40
    await kept.shutdown()

    rewritten = await open_index(monkeypatch, db_path, max_tail=0.25)  # over budget, so rebuild
    assert base_live_total(rewritten) == 140
    assert tail_live_total(rewritten) == 0
    assert sidecar.status(str(db_path))["tail_rows"] == 0
    assert os.path.getsize(path) > 0
    await rewritten.shutdown()


@pytest.mark.asyncio
async def test_shutdown_refreshes_a_file_the_session_moved_past(tmp_path, monkeypatch):
    db_path = tmp_path / "index.db"
    index = await open_index(monkeypatch, db_path, mode="off")
    await index.index(NS_MAIN, [{"id": f"m{i}", "text": f"document {i}"} for i in range(30)], P384)
    await index.shutdown()
    sidecar.build(str(db_path))
    assert sidecar.status(str(db_path))["base_rows"] == 30

    index = await open_index(monkeypatch, db_path)
    await index.index(NS_MAIN, [{"id": f"late-{i}", "text": f"late document {i}"} for i in range(5)], P384)
    await index.remove(NS_MAIN, ["m0"])
    assert sidecar.status(str(db_path))["base_rows"] == 30  # no build while running
    await index.shutdown()

    report = sidecar.status(str(db_path))
    assert (report["base_rows"], report["tail_rows"]) == (34, 0)  # the next start reads nothing extra

    reopened = await open_index(monkeypatch, db_path)
    assert base_live_total(reopened) == 34
    assert tail_live_total(reopened) == 0
    assert resident_items(reopened) == sqlite_items(db_path)
    await reopened.shutdown()


@pytest.mark.asyncio
async def test_shutdown_refreshes_a_file_a_purge_emptied(tmp_path, monkeypatch):
    """A purged namespace takes its group away, and with it the report that its rows died."""
    db_path = tmp_path / "index.db"
    index = await open_index(monkeypatch, db_path, mode="off")
    await index.index(NS_MAIN, [{"id": f"m{i}", "text": f"document {i}"} for i in range(30)], P384)
    await index.index(NS_SIDE, [{"id": f"s{i}", "text": f"side note {i}"} for i in range(10)], P384)
    await index.shutdown()
    sidecar.build(str(db_path))

    index = await open_index(monkeypatch, db_path)
    assert await index.purge_namespace(NS_MAIN) == 30
    await index.shutdown()

    report = sidecar.status(str(db_path))
    assert (report["base_rows"], report["tail_rows"]) == (10, 0)
    reopened = await open_index(monkeypatch, db_path)
    assert resident_items(reopened) == sqlite_items(db_path)
    await reopened.shutdown()


@pytest.mark.asyncio
async def test_sidecar_off_neither_reads_nor_writes_the_file(tmp_path, monkeypatch):
    db_path = tmp_path / "index.db"
    index = await open_index(monkeypatch, db_path, mode="off")
    await index.index(NS_MAIN, [{"id": f"m{i}", "text": f"document {i}"} for i in range(30)], P384)
    await index.shutdown()
    sidecar.build(str(db_path))
    path = sidecar.default_path(str(db_path))
    before = pathlib.Path(path).read_bytes()

    index = await open_index(monkeypatch, db_path, mode="off", min_rows=1)
    assert base_live_total(index) == 0  # the file is there and was not read
    await index.index(NS_MAIN, [{"id": "late", "text": "late document"}], P384)
    await index.shutdown()
    assert pathlib.Path(path).read_bytes() == before  # nor written


@pytest.mark.asyncio
async def test_an_in_memory_database_never_builds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EMBEDDING_SIDECAR", "auto")
    monkeypatch.setenv("EMBEDDING_SIDECAR_MIN_ROWS", "1")
    index = VectorIndex(":memory:")
    await index.initialize()
    assert index._sidecar_enabled is False
    await index.index(NS_MAIN, [{"id": f"m{i}", "text": f"document {i}"} for i in range(5)], P384)
    await index.shutdown()
    assert os.listdir(tmp_path) == []


@pytest.mark.asyncio
async def test_the_sidecar_path_can_be_moved(tmp_path, monkeypatch):
    db_path = tmp_path / "index.db"
    elsewhere = tmp_path / "cache" / "vectors.sidecar"
    elsewhere.parent.mkdir()
    monkeypatch.setenv("EMBEDDING_SIDECAR_PATH", str(elsewhere))

    index = await open_index(monkeypatch, db_path, min_rows=10)
    await index.index(NS_MAIN, [{"id": f"m{i}", "text": f"document {i}"} for i in range(20)], P384)
    await index.shutdown()
    assert elsewhere.exists()
    assert not os.path.exists(str(db_path) + ".sidecar")

    reopened = await open_index(monkeypatch, db_path, min_rows=10)
    assert base_live_total(reopened) == 20
    await reopened.shutdown()


def test_an_empty_database_builds_a_file_that_maps_to_nothing(tmp_path):
    db_path = tmp_path / "index.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE vectors (
            namespace TEXT NOT NULL, item_id TEXT NOT NULL, vector BLOB NOT NULL,
            created_at TEXT DEFAULT (datetime('now')), PRIMARY KEY (namespace, item_id)
        );
        """
    )
    conn.commit()
    conn.close()

    result = sidecar.build(str(db_path))
    assert (result["rows"], result["groups"], result["watermark"]) == (0, 0, 0)
    mapped = sidecar.load(result["path"])
    assert mapped.groups == [] and mapped.row_count == 0
    assert os.path.getsize(result["path"]) == result["bytes"]


def test_a_missing_file_is_not_an_error(tmp_path):
    assert sidecar.load(str(tmp_path / "absent.sidecar")) is None


def test_a_file_shorter_than_its_prefix_is_reported(tmp_path):
    path = tmp_path / "stub.sidecar"
    path.write_bytes(b"CEMB")
    with pytest.raises(sidecar.SidecarUnusable, match="shorter than"):
        sidecar.load(str(path))
