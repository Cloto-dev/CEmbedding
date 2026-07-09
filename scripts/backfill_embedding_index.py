"""Backfill the embedding server's VectorIndex from an existing cpersona DB.

CPersona only pushes NEW stores to the remote index (`POST /index` in
do_store). Flipping an existing deployment to CPERSONA_VECTOR_SEARCH_MODE=remote
with an empty index would silently lose semantic recall over everything stored
before the flip (the remote branch falls back to local only on HTTP errors,
not on empty results). This script copies the already-computed embedding blobs
straight into the index DB — no re-embedding, no HTTP.

IMPORTANT: the index will serve /search queries embedded by the embedding
server's CURRENT provider. The backfilled vectors must come from the same
model, or scores are meaningless. Verify with --expect-dim (e.g. 768 for
jina-v5-nano) before pointing production at the result.

Usage:
  python scripts/backfill_embedding_index.py \
      --cpersona-db ~/.claude/cpersona.db \
      --index-db data/embedding_index.db \
      [--agent-id claude-code] [--expect-dim 768] [--dry-run]

The embedding server loads the index into memory at startup, so restart it
(or start it for the first time) after backfilling.
"""

import argparse
import os
import sqlite3
import sys


def backfill(cpersona_db: str, index_db: str, agent_id: str | None, expect_dim: int | None, dry_run: bool) -> int:
    src = sqlite3.connect(f"file:{cpersona_db}?mode=ro", uri=True)

    tables = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "memories" not in tables:
        print(f"error: {cpersona_db} has no 'memories' table (empty or not a cpersona DB)", file=sys.stderr)
        src.close()
        return 1

    where = "embedding IS NOT NULL"
    params: tuple = ()
    if agent_id:
        where += " AND agent_id = ?"
        params = (agent_id,)

    rows = src.execute(f"SELECT agent_id, id, embedding FROM memories WHERE {where}", params).fetchall()
    ep_rows = src.execute(f"SELECT agent_id, id, embedding FROM episodes WHERE {where}", params).fetchall()
    src.close()

    items: list[tuple[str, str, bytes]] = []
    skipped_dim = 0
    for prefix, batch in (("mem", rows), ("ep", ep_rows)):
        for aid, rid, blob in batch:
            if not blob or len(blob) % 4 != 0:
                continue
            if expect_dim is not None and len(blob) != expect_dim * 4:
                skipped_dim += 1
                continue
            items.append((f"cpersona:{aid}", f"{prefix}:{rid}", blob))

    print(f"source rows: {len(rows)} memories + {len(ep_rows)} episodes -> {len(items)} vectors to backfill")
    if skipped_dim:
        print(f"WARNING: skipped {skipped_dim} rows whose dimension != --expect-dim (mixed-model corpus?)")
    if dry_run:
        print("dry-run: no writes performed")
        return 0
    if not items:
        print("nothing to backfill")
        return 0

    os.makedirs(os.path.dirname(index_db) or ".", exist_ok=True)
    dst = sqlite3.connect(index_db)
    dst.execute("PRAGMA journal_mode=WAL")
    dst.executescript(
        """
        CREATE TABLE IF NOT EXISTS vectors (
            namespace TEXT NOT NULL,
            item_id   TEXT NOT NULL,
            vector    BLOB NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (namespace, item_id)
        );
        CREATE INDEX IF NOT EXISTS idx_vectors_ns ON vectors (namespace);
        """
    )
    dst.executemany("INSERT OR REPLACE INTO vectors (namespace, item_id, vector) VALUES (?, ?, ?)", items)
    dst.commit()
    total = dst.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    namespaces = dst.execute("SELECT COUNT(DISTINCT namespace) FROM vectors").fetchone()[0]
    dst.close()
    print(f"done: index now holds {total} vectors across {namespaces} namespaces")
    print("restart the embedding server so it reloads the index into memory")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpersona-db", required=True, help="Path to the source cpersona.db (opened read-only)")
    parser.add_argument(
        "--index-db", required=True, help="Path to the embedding server's index DB (EMBEDDING_INDEX_DB_PATH)"
    )
    parser.add_argument("--agent-id", default=None, help="Backfill only this agent (default: all agents)")
    parser.add_argument(
        "--expect-dim", type=int, default=None, help="Skip vectors whose dimension differs (model-consistency guard)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing")
    args = parser.parse_args()

    if not os.path.exists(args.cpersona_db):
        print(f"error: cpersona DB not found: {args.cpersona_db}", file=sys.stderr)
        return 1
    return backfill(args.cpersona_db, args.index_db, args.agent_id, args.expect_dim, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
