# Contiguous Sidecar for the Vector Index

Status: implemented in the 0.7 line. The SQLite table keeps its schema and stays
the durable record; the sidecar is a derived file that can be deleted at any
time. Search results are identical to the in-memory matrix search, including
the order of equally-similar rows.

## 1. The problem

`VectorIndex.initialize()` reads every row of the `vectors` table and copies
every blob into a matrix. That is a full materialization of the corpus through
SQLite's row cursor: one Python `bytes` object per vector, joined and decoded.
Measured on a comparable workload (100,000 rows of 768 dimensions), the row
read is about three quarters of the cost and the decode most of the rest; the
matrix multiply that actually answers a query is under 3%. The startup cost is
linear in the corpus, the peak memory during load is roughly twice the payload,
and on an 8 GB host a corpus of a million 768-dimensional vectors does not
start at all.

The search itself is already one matrix multiply per query over a resident
matrix. The slow part is getting the matrix into memory.

## 2. What changes and what does not

The **source** of the resident matrix moves from SQLite rows to a file whose
bytes are already laid out as the `float32[count][dim]` numpy wants. The file
is opened with `numpy.memmap`, so opening it costs nothing proportional to its
size, pages are read on first touch, and the operating system is free to evict
them under memory pressure and read them back.

Nothing about the arithmetic changes. The query is still `matrix @ query_vec`
over the same rows in the same order, so scores are bit-identical to what the
same rows produce when loaded from SQLite, and the tie-break — row order, then
a stable sort — is unchanged. The equivalence tests that already compare the
matrix search to a per-item loop continue to apply as they are.

SQLite stays what it is: the durable record of every vector, written on every
`index()` and `remove()` exactly as today. The sidecar never receives a write
between builds. If the sidecar and SQLite disagree, SQLite is right and the
sidecar is rebuilt.

## 3. Layout

One resident group per (namespace, dimension), as before. A group now has two
segments:

| Segment | Backing | Rows | Writes |
| --- | --- | --- | --- |
| **base** | `numpy.memmap`, read-only | the rows the sidecar was built from | never; a removal flips a liveness flag held in memory |
| **tail** | in-memory `float32` matrix with spare capacity | every row written to SQLite after the build | as today: append into spare capacity, overwrite in place, tombstone, compact |

A search multiplies both segments and concatenates the scores base-first,
tail-second. Because the build writes rows in the order they were resident, and
the tail appends in insertion order, this concatenation reproduces the single
matrix's row order and therefore its tie order.

An item lives in exactly one row. An overwrite of an item that is in the base
tombstones its base row and appends the new vector to the tail (a memmap opened
read-only cannot be written in place, and a base that could be written would
no longer be a snapshot). Compaction applies to the tail only; base tombstones
are reclaimed by the next build.

## 4. The file

One sidecar per SQLite database, next to it (`<db path>.sidecar`), holding all
groups:

| Part | Contents |
| --- | --- |
| Header | magic, format version, watermark, and one entry per group: namespace, dimension, row count, byte offsets of its arrays |
| per group `embeddings` | `float32[count][dim]`, little-endian, contiguous |
| per group `ids` | the item ids, as a UTF-8 JSON array in the header (ids are strings of unbounded length, and the header is read once) |

The reader validates the file's length against what the header claims before
mapping anything: that one check catches a build killed halfway, a truncated
copy, and a header that disagrees with its own body. A file that fails it is
reported and ignored, and the index starts from SQLite as it does today.

The build writes to a temporary file, fsyncs, and renames it into place, so a
reader never sees a partial sidecar.

## 5. Freshness: the watermark

The sidecar records the largest SQLite `rowid` present when it was built. Rows
with a `rowid` above the watermark are read from SQLite at startup — with their
blobs, exactly as today — and become the initial tail. That is the whole
freshness mechanism: a vector indexed after the build is never invisible,
whether or not a rebuild has run since, and the cost of a late rebuild is a
longer tail, never a wrong answer.

`INSERT OR REPLACE` gives an overwritten item a new `rowid`, so an overwrite
after the build lands in the tail with the new vector while the base still
holds the old one. The reconciliation below resolves it in the tail's favour.

## 6. Reconciliation at startup

The watermark says which rows existed at build time. It cannot say which of
the base rows still exist, or still carry the vector the base holds. Both
questions are answered at startup from SQLite without touching a blob:

```sql
SELECT rowid, namespace, item_id FROM vectors
```

For every base row, the item is live if and only if its (namespace, item_id)
is present with a `rowid` at or below the watermark. An item that is absent was
removed after the build; an item whose `rowid` is above the watermark was
overwritten after the build and its current vector is in the tail. Either way
the base row is tombstoned before the first search.

This read is ids only. On the workload above it is about a tenth of the cost of
reading the blobs, and it keeps the index correct without relying on any
consumer to filter unknown ids out of its results.

## 7. When the sidecar is built

The build streams SQLite in bounded chunks into the temporary file, so its peak
memory is one chunk, not the corpus.

- **Explicitly**: `python -m cembedding.sidecar build|status [--db PATH]`.
- **At startup**, when `EMBEDDING_SIDECAR=auto` (the default) and either there
  is no sidecar and the table holds at least `EMBEDDING_SIDECAR_MIN_ROWS`
  rows, or the tail would exceed `EMBEDDING_SIDECAR_MAX_TAIL` of the base.
  Below the minimum the in-memory index is cheap enough that a file adds
  nothing, and small deployments keep exactly the behaviour they have today.
- **At shutdown**, when the tail is non-empty and the sidecar is enabled, so the
  next start is O(1). A build that fails at shutdown is logged and ignored: the
  next start falls back to the tail read.

`EMBEDDING_SIDECAR=off` disables the file entirely: no build, no read, the
in-memory index from SQLite as before.

## 8. Absence, corruption, and dimension mismatch

The sidecar is optional in the strongest sense. Three conditions ignore it and
load from SQLite: the file is missing, it fails its length check, or its format
version is not the one this code writes. The condition and the reason are
logged at startup and surfaced by the `status` subcommand, so an index that has
silently fallen back for a week is visible as such rather than as "somehow not
faster".

A group present in SQLite but absent from the sidecar (a namespace or dimension
created after the build) is simply a group with an empty base and a tail
holding all of its rows.

## 9. What this design leaves out

- **Approximate search.** The sidecar keeps exact brute force. Sub-linear
  search is a different feature with a different correctness contract.
- **Writing the sidecar on every index() call.** Appending to a mapped file
  while searches run on it, and keeping the header's row counts coherent, is a
  concurrency problem the watermark makes unnecessary.
- **Compacting the base.** Tombstoned base rows cost a masked multiply until
  the next build, which is bounded by the tail ratio that triggers a rebuild.
- **Shrinking the tail's allocation.** As today for the in-memory matrix.

## 10. How it is gated

- **Equivalence.** For a corpus and a scripted sequence of writes, searches
  through (sidecar base + tail) return exactly what the same corpus returns
  when loaded from SQLite alone: ids, order and scores compared for equality.
- **One-directional containment.** For a random corpus, random writes after the
  build, and random removals, the set of live items after startup equals the
  set of (namespace, item_id) in SQLite. A test that deliberately advances the
  recorded watermark past real rows must go red, and one that deliberately
  skips the reconciliation must go red on an overwrite.
- **Corruption.** A truncated sidecar, a wrong magic, and a foreign format
  version each fall back to SQLite and report why.
- **Startup cost.** A local-only benchmark records the time to first search and
  the peak resident memory for the SQLite path and the sidecar path at a stated
  corpus size, so the claim in §1 is a number in the repository rather than a
  sentence.
