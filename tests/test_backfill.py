"""End-to-end test: backfill script -> VectorIndex load -> search semantics.

Pins the migration contract: existing cpersona embedding blobs are copied
verbatim into the index DB, the server loads them at startup, and a /search
over the backfilled namespace returns the right item ids.
"""

import importlib.util
import os
import sqlite3
import struct

import numpy as np
import pytest

from cembedding.server import VectorIndex

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "backfill_embedding_index.py")


def _load_backfill():
    spec = importlib.util.spec_from_file_location("backfill_mod", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_cpersona_db(path: str, vectors: dict[str, np.ndarray]) -> None:
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT, embedding BLOB
        );
        CREATE TABLE episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT, embedding BLOB
        );
        """
    )
    for _mem_id, vec in sorted(vectors.items()):
        db.execute(
            "INSERT INTO memories (agent_id, embedding) VALUES (?, ?)",
            ("claude-code", struct.pack(f"<{len(vec)}f", *vec)),
        )
    db.commit()
    db.close()


@pytest.mark.asyncio
async def test_backfill_then_search_roundtrip(tmp_path):
    rng = np.random.default_rng(7)
    vecs = {}
    for i in range(1, 4):
        v = rng.random(384).astype(np.float32)
        vecs[f"mem:{i}"] = v / np.linalg.norm(v)

    src_db = str(tmp_path / "cpersona.db")
    idx_db = str(tmp_path / "embedding_index.db")
    _make_cpersona_db(src_db, vecs)

    backfill_mod = _load_backfill()
    assert backfill_mod.backfill(src_db, idx_db, agent_id=None, expect_dim=384, dry_run=False) == 0

    idx = VectorIndex(idx_db)
    await idx.initialize()
    assert await idx.count("cpersona:claude-code") == 3

    target = vecs["mem:2"]

    class EchoProvider:
        async def embed(self, texts):
            return [target.tolist()]

        def dimensions(self):
            return 384

    results = await idx.search("cpersona:claude-code", "whatever", 3, 0.0, EchoProvider())
    assert results[0]["id"] == "mem:2"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-3)
    assert {r["id"] for r in results} == {"mem:1", "mem:2", "mem:3"}

    await idx.shutdown()


def test_backfill_rejects_non_cpersona_db(tmp_path):
    empty = str(tmp_path / "empty.db")
    sqlite3.connect(empty).close()
    backfill_mod = _load_backfill()
    assert backfill_mod.backfill(empty, str(tmp_path / "idx.db"), None, None, False) == 1


def test_backfill_expect_dim_guard(tmp_path):
    """Vectors of the wrong dimension are skipped, not imported."""
    rng = np.random.default_rng(11)
    vecs = {
        "mem:1": rng.random(384).astype(np.float32),
        "mem:2": rng.random(768).astype(np.float32),
    }
    src_db = str(tmp_path / "cpersona.db")
    idx_db = str(tmp_path / "embedding_index.db")
    _make_cpersona_db(src_db, vecs)

    backfill_mod = _load_backfill()
    assert backfill_mod.backfill(src_db, idx_db, None, expect_dim=768, dry_run=False) == 0

    db = sqlite3.connect(idx_db)
    rows = db.execute("SELECT item_id, length(vector) FROM vectors").fetchall()
    db.close()
    assert len(rows) == 1
    assert rows[0][1] == 768 * 4
