"""Contiguous sidecar file for the vector index.

The index keeps one float32 matrix per (namespace, dimension) resident in
memory. Building those matrices from SQLite costs one Python ``bytes`` object
per vector and decodes the whole corpus at every start, which is linear in the
corpus and peaks at roughly twice the payload. The sidecar is a derived file
whose bytes are already the layout numpy wants, so the same matrices can be
mapped instead of decoded.

The file is derived state and nothing depends on it: SQLite stays the durable
record, the sidecar is never written between builds, and deleting it only costs
the next start a full read. A file that cannot be trusted -- a foreign format
version, a length that disagrees with its own header -- is reported and ignored
rather than guessed at.

Layout::

    magic (8 bytes) | format version (uint32 LE) | header length (uint32 LE)
    header: UTF-8 JSON, padded so the first block starts 64-byte aligned
    per group: float32[count][dim], little-endian, contiguous

The header carries the watermark (the largest SQLite rowid present at build
time) and, per group, the item ids and the rowids their vectors were read from.
Those are variable-length and are read exactly once at startup, so they live in
the JSON header rather than in a second binary section; the rowids are what
lets a start tell a base row that is still current from one that was replaced.

This module deliberately depends on nothing but the standard library and numpy:
a build or a status check should not need an HTTP stack or an inference runtime,
so it uses the synchronous ``sqlite3`` module rather than the server's aiosqlite
connection, and callers inside an event loop run it in a worker thread.
"""

import argparse
import json
import logging
import os
import sqlite3
import struct
import sys

import numpy as np

logger = logging.getLogger(__name__)

#: File signature. Deliberately not version-bearing: the version is a separate
#: field so a reader can say "format 2, I read 1" instead of "not a sidecar".
MAGIC = b"CEMBSCAR"
FORMAT_VERSION = 1

_PREFIX = struct.Struct("<8sII")
#: Blocks start on a cache line, which costs at most 63 bytes of padding once.
_ALIGNMENT = 64
DEFAULT_CHUNK_ROWS = 4096

#: Same default as the server's EMBEDDING_INDEX_DB_PATH. Repeated rather than
#: imported to keep this module free of the server's provider imports.
DEFAULT_DB_PATH = "data/embedding_index.db"


class SidecarUnusable(Exception):
    """A sidecar file exists but this build will not map it.

    Raised for a bad magic, a foreign format version, and a file whose length
    disagrees with its header. The caller's answer is always the same: log the
    reason and load from SQLite.
    """


class SidecarGroup:
    """One (namespace, dimension) block of a mapped sidecar."""

    __slots__ = ("dim", "ids", "matrix", "namespace", "rowids")

    def __init__(self, namespace: str, dim: int, ids: list, rowids: list, matrix: np.ndarray):
        self.namespace = namespace
        self.dim = dim
        self.ids = ids
        self.rowids = rowids
        self.matrix = matrix

    @property
    def count(self) -> int:
        return len(self.ids)


class Sidecar:
    """A mapped sidecar file: its watermark and one entry per group."""

    __slots__ = ("groups", "path", "version", "watermark")

    def __init__(self, path: str, version: int, watermark: int, groups: list):
        self.path = path
        self.version = version
        self.watermark = watermark
        self.groups = groups

    @property
    def row_count(self) -> int:
        return sum(group.count for group in self.groups)


class _Entry:
    """One group under construction during a build."""

    __slots__ = ("dim", "ids", "namespace", "offset", "rowids")

    def __init__(self, namespace: str, dim: int):
        self.namespace = namespace
        self.dim = dim
        self.ids: list[str] = []
        self.rowids: list[int] = []
        self.offset = 0

    @property
    def count(self) -> int:
        return len(self.ids)

    @property
    def nbytes(self) -> int:
        return self.count * self.dim * 4


def default_path(db_path: str) -> str:
    """Where the sidecar of a database lives, unless EMBEDDING_SIDECAR_PATH overrides it."""
    return os.environ.get("EMBEDDING_SIDECAR_PATH") or f"{db_path}.sidecar"


# ── build ──


def build(db_path: str, out_path: str | None = None, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> dict:
    """Write the sidecar for a database; return what was written.

    Rows are read in rowid order, so an item that was overwritten after its
    first insert (``INSERT OR REPLACE`` gives it a new rowid) is written once,
    at its newest position, with its newest vector. Blobs are streamed in
    chunks, so the build costs one chunk of memory rather than the corpus.

    The file is written to a temporary name, fsynced and renamed into place, so
    a concurrent reader sees either the previous sidecar or the new one and
    never a half-written file.
    """
    out_path = out_path or default_path(db_path)
    tmp_path = f"{out_path}.tmp"
    conn = sqlite3.connect(db_path)
    written = 0
    try:
        # One read transaction spans the watermark and both passes, so the ids
        # in the header describe exactly the rows the blocks hold.
        conn.execute("BEGIN")
        watermark = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM vectors").fetchone()[0]
        entries = _scan(conn, watermark, chunk_rows)
        header = _encode_header(watermark, entries)
        try:
            with open(tmp_path, "wb") as handle:
                handle.write(header)
                for entry in entries:
                    written += _write_block(handle, conn, entry, watermark, chunk_rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, out_path)
        except BaseException:
            _discard(tmp_path)
            raise
    finally:
        conn.close()
    _fsync_dir(os.path.dirname(os.path.abspath(out_path)))
    result = {
        "path": out_path,
        "rows": written,
        "groups": len(entries),
        "watermark": watermark,
        "bytes": os.path.getsize(out_path),
    }
    logger.info(
        "sidecar built: %s (%d rows in %d groups, watermark %d, %d bytes)",
        result["path"],
        result["rows"],
        result["groups"],
        result["watermark"],
        result["bytes"],
    )
    return result


def _scan(conn: sqlite3.Connection, watermark: int, chunk_rows: int) -> list:
    """Collect the ids and rowids of every group, in rowid order.

    This pass reads no vector: ``length(vector)`` gives the dimension, and the
    ids are what the header has to carry anyway.
    """
    entries: dict[tuple[str, int], _Entry] = {}
    cursor = conn.execute(
        "SELECT rowid, namespace, item_id, length(vector) FROM vectors WHERE rowid <= ? ORDER BY rowid",
        (watermark,),
    )
    while True:
        rows = cursor.fetchmany(chunk_rows)
        if not rows:
            break
        for rowid, namespace, item_id, byte_len in rows:
            if byte_len % 4:
                raise ValueError(f"vector blob for {namespace}/{item_id} is not a whole number of float32 values")
            key = (namespace, byte_len // 4)
            entry = entries.get(key)
            if entry is None:
                entry = entries[key] = _Entry(namespace, byte_len // 4)
            entry.ids.append(item_id)
            entry.rowids.append(rowid)
    return list(entries.values())


def _encode_header(watermark: int, entries: list) -> bytes:
    """Serialize the header and give every group its absolute byte offset.

    The offsets live inside the header, so their digits change the header's own
    length and move the offsets again. The layout is settled by re-encoding
    until it stops growing, which takes two passes in practice.
    """
    data_start = _align(_PREFIX.size)
    for _ in range(8):
        offset = data_start
        for entry in entries:
            entry.offset = offset
            offset += entry.nbytes
        body = json.dumps(
            {
                "format": FORMAT_VERSION,
                "watermark": watermark,
                "data_start": data_start,
                "groups": [
                    {
                        "namespace": entry.namespace,
                        "dim": entry.dim,
                        "count": entry.count,
                        "offset": entry.offset,
                        "ids": entry.ids,
                        "rowids": entry.rowids,
                    }
                    for entry in entries
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        settled = _align(_PREFIX.size + len(body))
        if settled == data_start:
            padding = data_start - _PREFIX.size - len(body)
            return _PREFIX.pack(MAGIC, FORMAT_VERSION, len(body)) + body + b"\0" * padding
        data_start = settled
    raise RuntimeError("sidecar header layout did not settle")


def _write_block(handle, conn: sqlite3.Connection, entry: _Entry, watermark: int, chunk_rows: int) -> int:
    """Stream one group's vectors into the file, in the order the header lists them."""
    position = handle.tell()
    if position != entry.offset:
        raise RuntimeError(f"block for {entry.namespace}/{entry.dim} starts at {position}, header says {entry.offset}")
    row_bytes = entry.dim * 4
    cursor = conn.execute(
        "SELECT vector FROM vectors WHERE namespace = ? AND length(vector) = ? AND rowid <= ? ORDER BY rowid",
        (entry.namespace, row_bytes, watermark),
    )
    written = 0
    while True:
        chunk = cursor.fetchmany(chunk_rows)
        if not chunk:
            break
        buffer = b"".join(blob for (blob,) in chunk)
        if len(buffer) != len(chunk) * row_bytes:
            raise ValueError(f"vector width changed under the build in {entry.namespace}")
        handle.write(buffer)
        written += len(chunk)
    if written != entry.count:
        raise ValueError(f"{entry.namespace}/{entry.dim}: header says {entry.count} rows, wrote {written}")
    return written


def _align(offset: int) -> int:
    return -(-offset // _ALIGNMENT) * _ALIGNMENT


def _discard(path: str) -> None:
    """Drop a half-written temporary file; its absence is the desired state."""
    try:
        os.remove(path)
    except OSError:
        pass


def _fsync_dir(directory: str) -> None:
    """Make the rename itself durable where the platform supports it."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


# ── load ──


def load(path: str) -> Sidecar | None:
    """Map a sidecar file.

    Returns None when there is no file, and raises :class:`SidecarUnusable`
    when there is one this build will not trust. Mapping costs nothing
    proportional to the corpus: the header is read, the blocks are mapped, and
    the pages are faulted in on first touch.
    """
    try:
        size = os.path.getsize(path)
    except FileNotFoundError:
        # No file is the ordinary case, not a fault. Anything else (a directory
        # in the way, no permission) is left to the caller to report.
        return None

    with open(path, "rb") as handle:
        prefix = handle.read(_PREFIX.size)
        if len(prefix) < _PREFIX.size:
            raise SidecarUnusable(f"file is {size} bytes, shorter than the {_PREFIX.size}-byte prefix")
        magic, version, header_len = _PREFIX.unpack(prefix)
        if magic != MAGIC:
            raise SidecarUnusable(f"bad magic {magic!r}, expected {MAGIC!r}")
        if version != FORMAT_VERSION:
            raise SidecarUnusable(f"format version {version}, this build reads {FORMAT_VERSION}")
        if _PREFIX.size + header_len > size:
            raise SidecarUnusable(f"header claims {header_len} bytes, the file holds {size - _PREFIX.size}")
        raw = handle.read(header_len)

    try:
        header = json.loads(raw)
        watermark = int(header["watermark"])
        data_start = int(header["data_start"])
        raw_groups = list(header["groups"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SidecarUnusable(f"header is not readable: {exc}") from exc

    # The length check is the one that matters: a build killed halfway, a
    # truncated copy and a header that disagrees with its own body all show up
    # as a file whose size is not what the header describes.
    parsed = []
    expected_size = data_start
    for raw_group in raw_groups:
        try:
            namespace = raw_group["namespace"]
            dim = int(raw_group["dim"])
            count = int(raw_group["count"])
            offset = int(raw_group["offset"])
            ids = list(raw_group["ids"])
            rowids = list(raw_group["rowids"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SidecarUnusable(f"group entry is not readable: {exc}") from exc
        if dim < 0 or count < 0 or offset < data_start:
            raise SidecarUnusable(f"group {namespace}/{dim} claims an impossible extent")
        if len(ids) != count or len(rowids) != count:
            raise SidecarUnusable(
                f"group {namespace}/{dim} lists {len(ids)} ids and {len(rowids)} rowids for {count} rows"
            )
        expected_size = max(expected_size, offset + count * dim * 4)
        parsed.append((namespace, dim, count, offset, ids, rowids))
    if size != expected_size:
        raise SidecarUnusable(f"length mismatch: the file is {size} bytes, its header describes {expected_size}")

    groups = []
    for namespace, dim, count, offset, ids, rowids in parsed:
        if count and dim:
            matrix = np.memmap(path, dtype=np.float32, mode="r", offset=offset, shape=(count, dim))
        else:
            # A group with no rows, or rows with no coordinates, maps nothing;
            # its ids still have to be counted, so it keeps an empty matrix.
            matrix = np.zeros((count, dim), dtype=np.float32)
        groups.append(SidecarGroup(namespace, dim, ids, rowids, matrix))
    return Sidecar(path, version, watermark, groups)


# ── status ──


def status(db_path: str, path: str | None = None) -> dict:
    """Report whether the sidecar of a database is present, usable and how stale.

    An index that has quietly fallen back to SQLite for a week should be
    visible as such rather than as "somehow not faster", so the reason a file
    was ignored is part of the answer.
    """
    path = path or default_path(db_path)
    info = {
        "path": path,
        "present": os.path.exists(path),
        "ok": False,
        "reason": None,
        "watermark": None,
        "base_rows": None,
        "tail_rows": None,
        "sqlite_rows": None,
        "bytes": os.path.getsize(path) if os.path.exists(path) else None,
    }

    try:
        mapped = load(path)
    except (SidecarUnusable, OSError) as exc:
        mapped = None
        info["reason"] = str(exc)
    else:
        if mapped is None:
            info["reason"] = "no sidecar file"
        else:
            info["ok"] = True
            info["watermark"] = mapped.watermark
            info["base_rows"] = mapped.row_count

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        try:
            info["sqlite_rows"] = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            if mapped is not None:
                info["tail_rows"] = conn.execute(
                    "SELECT COUNT(*) FROM vectors WHERE rowid > ?", (mapped.watermark,)
                ).fetchone()[0]
        except sqlite3.Error:
            # No vectors table yet: the database side simply has nothing to say.
            pass
        finally:
            conn.close()
    return info


# ── command line ──


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m cembedding.sidecar`` and ``cembedding-sidecar``."""
    parser = argparse.ArgumentParser(
        prog="cembedding-sidecar",
        description="Build or inspect the contiguous sidecar of a vector index database.",
    )
    parser.add_argument("command", choices=("build", "status"), help="build the sidecar, or report on it")
    parser.add_argument(
        "--db",
        default=os.environ.get("EMBEDDING_INDEX_DB_PATH", DEFAULT_DB_PATH),
        help="index database (default: $EMBEDDING_INDEX_DB_PATH or %(default)s)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "build":
        try:
            build(args.db)
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
            print(f"build failed: {exc}", file=sys.stderr)
            return 1
        return 0

    info = status(args.db)
    print(f"sidecar:  {info['path']}")
    print(f"present:  {'yes' if info['present'] else 'no'}" + (f" ({info['bytes']} bytes)" if info["present"] else ""))
    print(f"usable:   {'yes' if info['ok'] else 'no'}" + ("" if info["ok"] else f" - {info['reason']}"))
    print(f"watermark: {info['watermark'] if info['watermark'] is not None else '-'}")
    print(f"base rows: {info['base_rows'] if info['base_rows'] is not None else '-'}")
    sqlite_rows = info["sqlite_rows"]
    tail = info["tail_rows"]
    print(
        f"database:  {sqlite_rows if sqlite_rows is not None else '-'} rows"
        + (f", {tail} above the watermark" if tail is not None else "")
    )
    # A file that is present but unusable is the one state worth a non-zero
    # exit: the index is running, just not on the file someone thinks it is.
    return 0 if info["ok"] or not info["present"] else 1


if __name__ == "__main__":
    sys.exit(main())
