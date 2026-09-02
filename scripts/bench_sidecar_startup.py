"""Local benchmark: what a start costs with and without the sidecar.

Fills a throwaway database with N vectors, then measures time-to-first-search
and peak resident memory twice: once with EMBEDDING_SIDECAR=off (every vector
decoded out of SQLite, the behaviour before the sidecar) and once with the file
mapped.

Each arm runs in its own process, because peak resident memory is a high-water
mark that never falls: measured in one process, the second arm would inherit
the first arm's peak and both numbers would be the larger one.

    uv run python scripts/bench_sidecar_startup.py --rows 100000

This is a benchmark, not a test: it writes hundreds of megabytes and takes
minutes at the default size, so nothing in the suite runs it.
"""

import argparse
import asyncio
import json
import os
import resource
import sqlite3
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cembedding import sidecar  # noqa: E402

NAMESPACE = "bench:corpus"


class FixedProvider:
    """One canned query vector: the benchmark is about the start, not the model."""

    def __init__(self, dim: int):
        self._vector = (np.random.default_rng(1).standard_normal(dim) / dim**0.5).astype(np.float32).tolist()

    async def embed(self, texts):
        return [list(self._vector) for _ in texts]

    def dimensions(self) -> int:
        return len(self._vector)


def peak_rss_bytes() -> int:
    """ru_maxrss is bytes on macOS and kilobytes on Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def fill(db_path: str, rows: int, dim: int, chunk: int = 2000) -> None:
    """Write the corpus in chunks, so filling it does not need the corpus in memory."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS vectors (
            namespace TEXT NOT NULL,
            item_id   TEXT NOT NULL,
            vector    BLOB NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (namespace, item_id)
        );
        """
    )
    rng = np.random.default_rng(7)
    written = 0
    while written < rows:
        batch = min(chunk, rows - written)
        block = rng.standard_normal((batch, dim)).astype(np.float32)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        conn.executemany(
            "INSERT OR REPLACE INTO vectors (namespace, item_id, vector) VALUES (?, ?, ?)",
            [(NAMESPACE, f"item-{written + i}", block[i].tobytes()) for i in range(batch)],
        )
        conn.commit()
        written += batch
    conn.close()


async def measure(db_path: str, dim: int) -> dict:
    """Time an index from cold to its first answered query, in this process."""
    from cembedding.server import VectorIndex

    provider = FixedProvider(dim)
    index = VectorIndex(db_path)
    started = time.perf_counter()
    await index.initialize()
    hits = await index.search(NAMESPACE, "query", 10, -1.0, provider)
    elapsed = time.perf_counter() - started
    resident = sum(group.base_count + group.n for ns_groups in index._groups.values() for group in ns_groups.values())
    await index.shutdown()
    return {"seconds": elapsed, "peak_rss": peak_rss_bytes(), "hits": len(hits), "resident_rows": resident}


def run_arm(arm: str, db_path: str, dim: int) -> dict:
    """Run one arm in a child process and return what it measured."""
    env = dict(os.environ)
    env["EMBEDDING_SIDECAR"] = "off" if arm == "off" else "auto"
    # Thresholds that never fire: this measures mapping an existing file, not
    # the cost of writing one.
    env["EMBEDDING_SIDECAR_MIN_ROWS"] = str(10**12)
    env["EMBEDDING_SIDECAR_MAX_TAIL"] = str(10.0**9)
    completed = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--arm", arm, "--db", db_path, "--dim", str(dim)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def human_bytes(count: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if count < 1024 or unit == "GiB":
            return f"{count:,.1f} {unit}"
        count /= 1024
    return f"{count:,.1f} GiB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--db", default="", help="reuse an existing corpus instead of writing a new one")
    parser.add_argument("--arm", choices=("off", "sidecar"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.arm:  # child process: one measurement, reported as JSON on stdout
        print(json.dumps(asyncio.run(measure(args.db, args.dim))))
        return 0

    directory = tempfile.mkdtemp(prefix="sidecar-bench-")
    db_path = args.db or os.path.join(directory, "bench.db")
    if not args.db:
        print(f"filling {db_path} with {args.rows:,} x {args.dim} vectors ...", flush=True)
        started = time.perf_counter()
        fill(db_path, args.rows, args.dim)
        print(f"  wrote in {time.perf_counter() - started:,.1f}s ({human_bytes(os.path.getsize(db_path))})")

    print("building the sidecar ...", flush=True)
    started = time.perf_counter()
    built = sidecar.build(db_path)
    print(f"  wrote {built['path']} in {time.perf_counter() - started:,.1f}s ({human_bytes(built['bytes'])})")

    results = {arm: run_arm(arm, db_path, args.dim) for arm in ("off", "sidecar")}
    print()
    print(f"{args.rows:,} rows x {args.dim} dimensions")
    print(f"{'arm':<10} {'time to first search':>21} {'peak RSS':>12} {'resident rows':>14}")
    for arm, result in results.items():
        label = "SQLite" if arm == "off" else "sidecar"
        print(
            f"{label:<10} {result['seconds']:>20,.2f}s {human_bytes(result['peak_rss']):>12} "
            f"{result['resident_rows']:>14,}"
        )
    speedup = results["off"]["seconds"] / results["sidecar"]["seconds"]
    saved = results["off"]["peak_rss"] - results["sidecar"]["peak_rss"]
    print(f"\n{speedup:,.1f}x faster to first search, {human_bytes(saved)} less peak resident memory")
    return 0


if __name__ == "__main__":
    sys.exit(main())
