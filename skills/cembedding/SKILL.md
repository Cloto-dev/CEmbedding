---
name: cembedding
description: >-
  Install, run, and operate CEmbedding — a local-first embedding server
  (on-device ONNX or OpenAI-compatible API) that serves vectors over a tiny
  HTTP contract and is the reference /embed backend for CPersona. Use this
  skill when the user wants local embeddings, is setting up CPersona's
  vector-search layer, sees degraded / keyword-only recall, or needs to pick
  or change an embedding provider. Covers install, model download, provider
  choice, CPersona wiring, troubleshooting, and persisting the operating
  policy into the user's CLAUDE.md.
---

# CEmbedding — local-first embedding server

CEmbedding turns text into vectors over a minimal HTTP contract:

```
POST /embed
Request:  { "texts": ["string", ...] }                 # non-empty, max 100 per batch
Response: { "embeddings": [[float, ...], ...], "dimensions": <int> }
```

It runs a model **on-device** via ONNX (no API key, no network) or proxies an
**OpenAI-compatible API**. Its primary consumer is
[CPersona](https://github.com/Cloto-dev/CPersona), whose hybrid search uses it
for the vector-similarity layer; anything that can POST JSON can use it. It
also exposes an MCP (stdio) surface and an optional persistent vector index
(`/index`, `/search`), but `/embed` is all CPersona needs.

- MIT licensed. Repo: <https://github.com/Cloto-dev/CEmbedding>

## When to use this skill

- The user wants **local / private / free** embeddings, or asks to set up
  "an embedding server" (often as step 2 of a CPersona install).
- CPersona recall is **degraded** (an `advisory` field on recall responses,
  or vector search silently disabled) and the embedding backend is the
  suspect.
- The user wants to **choose or change** the embedding model/provider.

## Setup

**Prerequisites:** Python 3.10+.

### 1. Download a model and run the server (on-device ONNX)

The one-command path (needs [uv](https://docs.astral.sh/uv/)); run both
commands from the same directory — the model lands in `./data/models`:

```bash
uvx --from "cembedding[onnx]" cembedding-download-model --model jina-v5-nano
EMBEDDING_PROVIDER=onnx_jina_v5_nano uvx --from "cembedding[onnx]" cembedding
```

Or on PATH: `pip install "cembedding[onnx]"`, then `cembedding-download-model
--model jina-v5-nano` and `EMBEDDING_PROVIDER=onnx_jina_v5_nano cembedding`.
From source: `git clone https://github.com/Cloto-dev/CEmbedding.git`,
`pip install ".[onnx]"`, `python -m cembedding.download_model --model
jina-v5-nano`, run with `EMBEDDING_PROVIDER=onnx_jina_v5_nano python -m cembedding`.

Model choice: `jina-v5-nano` (768d) is what CPersona is tuned and benchmarked
against — default to it. `bge-m3` is larger/multilingual, `miniml` is the
smallest (384d). For an OpenAI-compatible API instead of on-device, use
`EMBEDDING_PROVIDER=api_openai` with `EMBEDDING_API_KEY` (no model download).

> **Trap:** the provider **defaults to `api_openai`** — running with no
> `EMBEDDING_PROVIDER` and no `EMBEDDING_API_KEY` fails. On-device users must
> set an `onnx_*` provider explicitly.

### 2. Verify

Expect `HTTP embedding endpoint started on http://127.0.0.1:8401/embed` in
the log, then:

```bash
curl -s http://127.0.0.1:8401/embed \
  -H 'content-type: application/json' \
  -d '{"texts":["hello world"]}' | head -c 200
```

A JSON body with `"dimensions"` means it works.

### 3. Wire consumers (CPersona)

```bash
# in CPersona's MCP config env
CPERSONA_EMBEDDING_MODE=http
CPERSONA_EMBEDDING_URL=http://127.0.0.1:8401/embed
```

### 4. Keep it running

The process **exits on stdin EOF** (it is also an MCP stdio server), so a
naive `nohup ... &` or `... < /dev/null` dies immediately. Run it under a
service manager (systemd / launchd), or keep stdin held open for a manual
background boot.

### 5. Persist the policy into CLAUDE.md (recommended)

This skill only loads when a conversation activates it, but the operating
policy below matters in **every** session that touches embeddings, so offer
to persist it per the
[CLAUDE.md Policy Generation Standard](https://github.com/Cloto-dev/CPersona/blob/master/docs/CLAUDE_MD_POLICY_STANDARD.md):

- **Ask first** — show the exact block, get approval before writing.
- **Default target: `~/.claude/CLAUDE.md`** (embeddings are cross-project
  infrastructure); offer a project-level `CLAUDE.md` as the scoped
  alternative.
- **Replace, don't duplicate** — if a `BEGIN cembedding-policy` marker
  already exists, replace everything between the markers; never touch
  content outside them.
- Substitute `<PROVIDER>` and `<RUN_COMMAND>` with the values chosen above
  (e.g. `onnx_jina_v5_nano` and `uvx --from "cembedding[onnx]" cembedding`),
  and the port if not 8401.

```markdown
<!-- BEGIN cembedding-policy v1 (managed by the cembedding skill; re-run the skill to update) -->
## CEmbedding policy

A local CEmbedding server provides vector embeddings at `http://127.0.0.1:8401/embed`
(provider `<PROVIDER>`). Keep these behaviors:

- **Liveness first.** When semantic recall looks degraded (e.g. CPersona attaches an
  `advisory` field, or vector results are silently thin), check this server before anything
  else: `curl -s http://127.0.0.1:8401/embed -H 'content-type: application/json'
  -d '{"texts":["ping"]}'`. If it fails, restart it: `EMBEDDING_PROVIDER=<PROVIDER> <RUN_COMMAND>`.
- **Keep stdin open when daemonizing.** The process exits on stdin EOF; launch it under a
  service manager, or hold stdin open for manual background boots.
- **Model stability.** The model defines the vector space — never switch `EMBEDDING_PROVIDER`
  casually. After any deliberate model change, recalibrate consumers (CPersona:
  `calibrate_threshold`).
- `/embed` accepts at most 100 texts per batch — chunk larger jobs.

Details, setup, and troubleshooting: the `cembedding` skill.
<!-- END cembedding-policy -->
```

## Operations

- **Providers** (`EMBEDDING_PROVIDER`): `onnx_jina_v5_nano` / `onnx_bge_m3` /
  `onnx_miniml` (local CPU), `mlx_bge_m3` (Apple Silicon, `pip install
  ".[mlx]"`), `api_openai` (needs `EMBEDDING_API_KEY`, optional
  `EMBEDDING_API_URL` / `EMBEDDING_MODEL`).
- **Key env:** `EMBEDDING_HTTP_PORT` (default 8401),
  `EMBEDDING_INDEX_ENABLED` / `EMBEDDING_INDEX_DB_PATH` (persistent vector
  index behind `/index` `/search` `/remove` `/purge`),
  `EMBEDDING_SEARCH_BACKEND` (`numpy` default, `mlx` for Apple-GPU),
  `ONNX_MODEL_DIR`, `ONNX_MAX_SEQ_LEN`. Full table: the repository README.
- **CPersona remote vector search** (`CPERSONA_VECTOR_SEARCH_MODE=remote`)
  stores vectors in this server's index; when flipping an existing CPersona
  deployment, migrate already-stored vectors first with
  `scripts/backfill_embedding_index.py` (see README).

## Troubleshooting

- **Exits immediately after start** — stdin closed (see Setup step 4). Use a
  service manager or keep stdin open.
- **`api_openai` errors on a local-only setup** — `EMBEDDING_PROVIDER` was
  not set; pick an `onnx_*` provider.
- **Model not found** — `cembedding-download-model` writes to `./data/models`
  relative to the **current directory**; run the server from the same
  directory or set `ONNX_MODEL_DIR`.
- **CPersona recall degraded** — verify liveness with the curl above, confirm
  CPersona's `CPERSONA_EMBEDDING_URL` points at this host/port, then recall
  again. If the model/dimension changed, run CPersona's
  `calibrate_threshold`.
