# Contiguous Sidecar for the Vector Index

Status: implemented in the 0.7 line. The SQLite table keeps its schema and stays
the durable record; the sidecar is a derived file that can be deleted at any
time. Search results are identical to the in-memory matrix search, with one
named exception for exact ties across the segment boundary (§3).

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

Nothing about the arithmetic changes: the query is still `matrix @ query_vec`
over the same rows in the same order, and the tie-break — row order, then a
stable sort — is unchanged. What does change is that the rows are now two
matrices instead of one (§3), and that has a measurable consequence for exact
ties, described there.

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
tail-second. Because the build writes rows in rowid order, the tail is read in
rowid order and appends in insertion order, this concatenation reproduces the
single matrix's row order.

**Two multiplies are not one multiply.** A BLAS library picks its kernel by
the operand's shape, and a small segment can be handed a kernel whose
accumulation order differs from the one a large matrix gets. Measured with
numpy on Accelerate: the last two rows of a 30-row matrix score differently in
their last bit when multiplied as a 2-row matrix than as rows 28–29 of the
whole. The reported score is rounded to four decimals, so the payload does not
show it; what can change is the *order* of rows that tie exactly, and at a
top-k boundary which tied row is returned. Real embeddings tie exactly only
when two items hold the same vector. The equivalence gate (§10) therefore uses
vectors whose dot products are exactly representable, so it measures row order
and not kernel choice; that limit is stated in the fixture rather than hidden
in a tolerance.

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
| Header | magic, format version, watermark, the byte offset where blocks start, and one entry per group: namespace, dimension, row count, block offset, the item ids, and the SQLite `rowid` each row was read from |
| per group `embeddings` | `float32[count][dim]`, little-endian, contiguous, 64-byte aligned |

Ids and rowids are variable-length and are read exactly once at startup, so
they live in the JSON header rather than in a binary section. The rowids are
not decoration: they are what lets a start tell a base row that is still
current from one that was replaced (§6).

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
is still present **at the `rowid` the row was read from**. An item that is
absent was removed after the build; an item at a higher `rowid` was overwritten
after the build and its current vector is in the tail. Either way the base row
is tombstoned before the first search.

Comparing rowids rather than just "present at or below the watermark" is
necessary, not cautious: SQLite reuses the largest rowid once the row holding
it is deleted. Delete the newest item and insert another, and the newcomer
takes the watermark's own rowid — it is neither a base row nor above the
watermark, so a presence test would keep the old base row and never see the
new one. With rowids recorded, the old row fails its rowid comparison. The
newcomer is still not in the tail, so the start also counts SQLite's rows at
or below the watermark and compares that to the live base rows; if they
disagree, the mapping is dropped and everything is read from SQLite, with the
reason logged. Repairing it in place would mean appending a row that belongs
*before* the base after it, which breaks the tie order the layout exists to
preserve.

One case stays undetectable: the newest item is removed and then re-added,
under the same id, before any other insert, by a writer that is not this
index (another process, or a run with `EMBEDDING_SIDECAR=off`). SQLite hands
the item its own rowid back, the (key, rowid) pair is intact, and the base's
old vector is served until the next build. A running index cannot produce it —
its removal tombstones the base row and its re-add lands in the tail — and a
clean shutdown rebuilds after any removal. It is recorded here because it is
real, not because it is likely.

This read is ids only. On the workload above it is about a tenth of the cost of
reading the blobs, and it keeps the index correct without relying on any
consumer to filter unknown ids out of its results.

## 7. When the sidecar is built

The build streams SQLite in bounded chunks into the temporary file, so its peak
memory is one chunk, not the corpus.

- **Explicitly**: `python -m cembedding.sidecar build|status [--db PATH]`.
- **At startup**, when `EMBEDDING_SIDECAR=auto` (the default) and either there
  is no *usable* sidecar and the table holds at least
  `EMBEDDING_SIDECAR_MIN_ROWS` rows, or the tail exceeds
  `EMBEDDING_SIDECAR_MAX_TAIL` of the base. "Usable" matters: a truncated or
  foreign file is replaced once the corpus earns one, instead of being ignored
  forever. Below the minimum the in-memory index is cheap enough that a file
  adds nothing — measured at 2,000 rows the ids-only reconciliation costs more
  than it saves — and small deployments keep exactly the behaviour they have
  today, including leaving no file behind.
- **At shutdown**, when something the current file does not hold exists (a
  non-empty tail, tombstoned base rows, or a group dropped whole by a purge)
  and either a file already exists or the corpus has reached the minimum. A
  build that fails at shutdown is logged and ignored: the next start falls
  back to the tail read.

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
  the next build. The startup trigger looks at the tail, not at dead base rows,
  so a session that removes most of its base keeps paying that multiply until
  a shutdown or an explicit build.
- **Keeping the memory win under `EMBEDDING_SEARCH_BACKEND=mlx`.** The base is
  converted to an mlx array once, which materialises it in unified memory.
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
- **Startup cost.** `scripts/bench_sidecar_startup.py` records the time to
  first search and the peak resident memory for both paths at a stated corpus
  size. At 100,000 × 768 on an Apple laptop: 0.23 s and 1.1 GiB from SQLite,
  0.09 s and 402 MiB from the sidecar.
