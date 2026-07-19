<div align="center">

# CEmbedding

### Local-first embedding server

Vector embeddings over a tiny HTTP contract.
On-device ONNX or any OpenAI-compatible API. The reference `/embed` server for [CPersona](https://github.com/Cloto-dev/CPersona).

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()

</div>

---

> **Standalone repository** — extracted from the (now private) `clotohub-servers` monorepo so it can be used on its own. [ClotoCore](https://github.com/Cloto-dev/ClotoCore) users get this through the in-app marketplace ([ClotoHub](https://hub.cloto.dev)); everyone else can run it directly as described below.

## What it is

A small server that turns text into vectors. It speaks a minimal HTTP contract so anything can call it — its primary consumer is [CPersona](https://github.com/Cloto-dev/CPersona), whose hybrid search uses it for the vector-similarity layer. It can run a model **on-device** via ONNX (no API key, no network) or proxy an **OpenAI-compatible API**.

It also exposes an MCP (stdio) surface and an optional persistent vector index (`/index`, `/search`), but the HTTP `/embed` endpoint is all CPersona needs.

## The `/embed` contract

```
POST /embed
Request:  { "texts": ["string", ...] }                 # non-empty array, max 100 per batch
Response: { "embeddings": [[float, ...], ...], "dimensions": <int> }
```

Point any client (e.g. CPersona's `CPERSONA_EMBEDDING_URL` / generic `EMBEDDING_HTTP_URL`) at `http://127.0.0.1:8401/embed`.

## Quick Start (on-device ONNX)

**Prerequisites:** Python 3.10+

```bash
# Download a model into ./data/models (jina-v5-nano is what CPersona is tuned for)
uvx --from "cembedding[onnx]" cembedding-download-model --model jina-v5-nano

# Run the server (reads ./data/models from the current directory)
EMBEDDING_PROVIDER=onnx_jina_v5_nano uvx --from "cembedding[onnx]" cembedding
```

Or install it onto your PATH with `pip install "cembedding[onnx]"`, then run
`cembedding-download-model --model jina-v5-nano` and `cembedding`.

From source (development):

```bash
git clone https://github.com/Cloto-dev/CEmbedding.git
cd CEmbedding
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install ".[onnx]"
python -m cembedding.download_model --model jina-v5-nano
EMBEDDING_PROVIDER=onnx_jina_v5_nano python -m cembedding   # or: python server.py
```

You should see `HTTP embedding endpoint started on http://127.0.0.1:8401/embed`. Verify it:

```bash
curl -s http://127.0.0.1:8401/embed \
  -H 'content-type: application/json' \
  -d '{"texts":["hello world"]}' | head -c 200
```

## Providers

Set `EMBEDDING_PROVIDER`:

| Value | Model | Notes |
|-------|-------|-------|
| `onnx_jina_v5_nano` | jina-v5-nano (33M, 768d) | Local CPU, what CPersona is benchmarked against |
| `onnx_bge_m3` | bge-m3 | Local CPU, larger / multilingual |
| `onnx_miniml` | all-MiniLM-L6-v2 (22M, 384d) | Local CPU, smallest |
| `mlx_bge_m3` | bge-m3 (MLX) | Apple Silicon only — `pip install ".[mlx]"` |
| `auto_bge_m3` | bge-m3 | Auto-selects MLX on Apple Silicon, ONNX elsewhere |
| `api_openai` | provider's model | OpenAI-compatible API; needs `EMBEDDING_API_KEY` (+ optional `EMBEDDING_API_URL`, `EMBEDDING_MODEL`) |

Download a local model with `cembedding-download-model --model {miniml,jina-v5-nano,bge-m3}` (or `python -m cembedding.download_model ...` from a source checkout; fetched from HuggingFace into `./data/models`, not committed to this repo).

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_PROVIDER` | `api_openai` | Provider (see table above) |
| `EMBEDDING_HTTP_PORT` | `8401` | HTTP port for `/embed` |
| `EMBEDDING_INDEX_ENABLED` | `true` | Enable the persistent vector index endpoints (`/index`, `/search`, `/remove`, `/purge`) |
| `EMBEDDING_INDEX_DB_PATH` | `data/embedding_index.db` | SQLite file backing the vector index |
| `EMBEDDING_SEARCH_BACKEND` | `numpy` | `/search` matmul backend. `numpy` (Accelerate BLAS) or `mlx` (Apple-GPU resident matrix; falls back to numpy when mlx is absent) |
| `ONNX_MODEL_DIR` | (auto) | Override the model directory for ONNX providers |
| `ONNX_EP_PREFERENCE` | (auto) | ONNX execution providers, comma-separated. Empty = auto (CoreML on macOS, DirectML on Windows, else CPU; CPU always ensured) |
| `ONNX_MAX_SEQ_LEN` | `2048` | Max tokenization length (1–8192; MiniLM clamped to 512 internally) |
| `EMBEDDING_API_KEY` | — | Required for `api_openai` |
| `EMBEDDING_API_URL` | `https://api.openai.com/v1/embeddings` | API endpoint for `api_openai` |
| `CEMBEDDING_AUTH_TOKEN` | — | Inbound bearer token. Unset = no authentication (see below) |
| `CEMBEDDING_REQUIRE_AUTH` | `false` | Refuse to start when no token is configured |

## Authentication (v0.6.2)

Both HTTP surfaces — the REST endpoints (`/embed`, `/index`, `/search`,
`/remove`, `/purge`) and the Streamable HTTP MCP transport — accept an inbound
bearer token:

```bash
CEMBEDDING_AUTH_TOKEN=$(openssl rand -hex 32)
```

With the token set, every request must carry `Authorization: Bearer <token>`;
a missing header, a wrong scheme and a wrong token are all rejected with `401`.
Comparison is constant-time. With the token unset, requests are served exactly
as in earlier versions and a warning is logged — set `CEMBEDDING_REQUIRE_AUTH=true`
to turn that warning into a startup error instead. Requiring a token is opt-in
in this release so existing deployments keep working; a later release will make
it the default.

**Do not treat the bind address as the security boundary.** The REST surface
binds loopback and the MCP transport binds `0.0.0.0` by default, but a tunnel or
reverse proxy forwards to loopback all the same, so a loopback bind is no
evidence that requests are local. If the process is reachable through a tunnel,
a proxy, or any non-loopback interface, configure a token.

## Use with CPersona

Run this server, then tell CPersona to use it:

```bash
# CPersona MCP config env
CPERSONA_EMBEDDING_MODE=http
CPERSONA_EMBEDDING_URL=http://127.0.0.1:8401/embed
```

Without an embedding server CPersona still works (FTS5 + keyword search); adding one enables the vector-similarity layer.

To serve CPersona's *remote vector search* (`CPERSONA_VECTOR_SEARCH_MODE=remote`),
this server's `/index` + `/search` endpoints hold the vectors. v0.6.0 searches a
per-namespace resident matrix (one matmul per query, ~21x faster than v0.5.0 at
237k x 384: 131 ms -> 6 ms/query), so a full-corpus semantic recall stays fast at
memory-corpus scale. When flipping an existing CPersona deployment to remote mode,
first migrate its already-stored vectors:

```bash
python scripts/backfill_embedding_index.py \
    --cpersona-db ~/.claude/cpersona.db \
    --index-db data/embedding_index.db \
    --expect-dim 768   # your embedding model's dimension
```

then restart this server so it reloads the index. Skipping the backfill silently
drops every pre-flip memory from semantic recall (the remote branch only falls
back to local search on HTTP errors, not on empty results).

## License

MIT — see [LICENSE](LICENSE).
