"""
Cloto MCP Server: Vector Embedding
Pluggable embedding provider with HTTP endpoint for inter-server communication.
Providers: api_openai (OpenAI-compatible API), onnx_miniml (local MiniLM ONNX).

v0.2.0: Vector Index — persistent index + search endpoints for centralized vector search.

Design: docs/CPERSONA_MEMORY_DESIGN.md Section 5
"""

import asyncio
import logging
import os
import platform as _platform
import sys
from abc import ABC, abstractmethod

import httpx
import numpy as np
from aiohttp import web
from mcp.server.stdio import stdio_server


from cembedding._vendored_mcp_common.mcp_utils import ToolRegistry
from cembedding.auth import (
    BearerTokenMiddleware,
    aiohttp_bearer_middleware,
    check_startup,
    resolve_auth_token,
)

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "api_openai")
EMBEDDING_HTTP_PORT = int(os.environ.get("EMBEDDING_HTTP_PORT", "8401"))
if not (1 <= EMBEDDING_HTTP_PORT <= 65535):
    raise ValueError(f"EMBEDDING_HTTP_PORT must be 1-65535, got {EMBEDDING_HTTP_PORT}")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL", "https://api.openai.com/v1/embeddings")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "")  # provider-dependent default
EMBEDDING_TIMEOUT = int(os.environ.get("EMBEDDING_TIMEOUT_SECS", "30"))

# Vector Index (v0.2.0)
EMBEDDING_INDEX_ENABLED = os.environ.get("EMBEDDING_INDEX_ENABLED", "true").lower() == "true"
EMBEDDING_INDEX_DB_PATH = os.environ.get("EMBEDDING_INDEX_DB_PATH", "data/embedding_index.db")
# Search matmul backend (v0.5.0): "numpy" (Accelerate BLAS, default) or "mlx"
# (Apple-GPU resident matrix; falls back to numpy when mlx is not installed).
EMBEDDING_SEARCH_BACKEND = os.environ.get("EMBEDDING_SEARCH_BACKEND", "numpy").lower()
if EMBEDDING_SEARCH_BACKEND not in ("numpy", "mlx"):
    raise ValueError(f"EMBEDDING_SEARCH_BACKEND must be numpy or mlx, got {EMBEDDING_SEARCH_BACKEND}")

# ONNX tokenization max sequence length (1-8192). MiniLM is clamped to 512
# internally (positional embeddings cap). Jina-v5-nano supports up to 8192.
ONNX_MAX_SEQ_LEN = int(os.environ.get("ONNX_MAX_SEQ_LEN", "2048"))
if not (1 <= ONNX_MAX_SEQ_LEN <= 8192):
    raise ValueError(f"ONNX_MAX_SEQ_LEN must be 1-8192, got {ONNX_MAX_SEQ_LEN}")

# ONNX-specific — resolve relative paths against CLOTO_PROJECT_DIR when running
# inside a sandbox (isolation changes the working directory).
_project_dir = os.environ.get("CLOTO_PROJECT_DIR", "")
_MODEL_DIRS = {
    "onnx_miniml": "data/models/all-MiniLM-L6-v2",
    "onnx_jina_v5_nano": "data/models/jina-embeddings-v5-text-nano",
    "onnx_bge_m3": "data/models/bge-m3",
    "mlx_bge_m3": "data/models/bge-m3-mlx",
}
_default_model_dir = _MODEL_DIRS.get(EMBEDDING_PROVIDER, "data/models/all-MiniLM-L6-v2")
ONNX_MODEL_DIR = os.environ.get("ONNX_MODEL_DIR", "")
if not ONNX_MODEL_DIR:
    if _project_dir and not os.path.isabs(_default_model_dir):
        ONNX_MODEL_DIR = os.path.join(_project_dir, _default_model_dir)
    else:
        ONNX_MODEL_DIR = _default_model_dir


def _select_ort_providers() -> list:
    """Select ONNX Runtime execution providers with cross-platform fallback.

    Priority when ONNX_EP_PREFERENCE is empty (auto-detect):
      1. CoreMLExecutionProvider (macOS — standard onnxruntime)
      2. DmlExecutionProvider   (Windows — standard onnxruntime)
      3. CUDAExecutionProvider  (Linux/Windows NVIDIA — requires onnxruntime-gpu)
      4. ROCmExecutionProvider  (Linux AMD — requires onnxruntime ROCm build)
      5. CPUExecutionProvider   (always appended as terminal fallback)

    When ONNX_EP_PREFERENCE is set, use the comma-separated list but always
    filter against get_available_providers() and always ensure CPUExecutionProvider
    is present so session creation cannot fail for lack of any provider.

    Fail-open: if onnxruntime import or get_available_providers() raises,
    return ["CPUExecutionProvider"] so the caller can still attempt to load.
    """
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]

    preference = os.environ.get("ONNX_EP_PREFERENCE", "").strip()

    if preference:
        requested = [p.strip() for p in preference.split(",") if p.strip()]
        providers = [p for p in requested if p in available]
    else:
        providers = []
        for candidate in (
            "CoreMLExecutionProvider",  # macOS
            "DmlExecutionProvider",  # Windows (DirectML)
            "CUDAExecutionProvider",  # Linux/Windows NVIDIA (onnxruntime-gpu)
            "ROCmExecutionProvider",  # Linux AMD (onnxruntime ROCm)
        ):
            if candidate in available:
                providers.append(candidate)

    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    return providers


# ============================================================
# Provider Abstraction
# ============================================================


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (load model, create client, etc.)."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""

    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensionality."""

    async def shutdown(self) -> None:
        """Clean up resources."""


# ============================================================
# api_openai Provider
# ============================================================


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embedding API provider."""

    def __init__(self, api_key: str, api_url: str, model: str, timeout: int):
        self._api_key = api_key
        self._api_url = api_url
        self._model = model or "text-embedding-3-small"
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._dimensions = int(os.environ.get("EMBEDDING_DIMENSIONS", "1536"))

    async def initialize(self) -> None:
        if not self._api_key:
            raise ValueError("EMBEDDING_API_KEY is required for api_openai provider")
        self._client = httpx.AsyncClient(timeout=self._timeout)
        logger.info(
            "OpenAI embedding provider initialized (model=%s, url=%s)",
            self._model,
            self._api_url,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._client:
            raise RuntimeError("Provider not initialized")

        response = await self._client.post(
            self._api_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self._model, "input": texts},
        )
        response.raise_for_status()

        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]

        # Update dimensions from actual response
        if embeddings:
            self._dimensions = len(embeddings[0])

        # L2-normalize for consistent cosine similarity via dot product
        result = []
        for emb in embeddings:
            vec = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            result.append(vec.tolist())

        return result

    def dimensions(self) -> int:
        return self._dimensions

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


# ============================================================
# onnx_miniml Provider
# ============================================================


class OnnxMiniLMProvider(EmbeddingProvider):
    """Local all-MiniLM-L6-v2 ONNX embedding provider."""

    def __init__(self, model_dir: str):
        self._model_dir = model_dir
        self._session = None
        self._tokenizer = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError(
                "onnx_miniml provider requires: uv pip install onnxruntime tokenizers\n"
                "Or: uv pip install cloto-mcp-embedding[onnx]"
            )

        model_path = os.path.join(self._model_dir, "model.onnx")
        tokenizer_path = os.path.join(self._model_dir, "tokenizer.json")

        # Auto-download model if missing
        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            logger.info("ONNX model not found, downloading automatically...")
            try:
                from cembedding.download_model import download

                if not download():
                    raise FileNotFoundError(f"Failed to download ONNX model to {self._model_dir}")
            except ImportError:
                raise FileNotFoundError(
                    f"ONNX model not found at {model_path}. Download with: python -m cembedding.download_model"
                )

        providers = _select_ort_providers()

        self._session = ort.InferenceSession(model_path, providers=providers)
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        # MiniLM positional embeddings cap at 512 — clamp ONNX_MAX_SEQ_LEN.
        miniml_seq_len = min(ONNX_MAX_SEQ_LEN, 512)
        if miniml_seq_len < ONNX_MAX_SEQ_LEN:
            logger.warning(
                "MiniLM max_position=512, clamping ONNX_MAX_SEQ_LEN=%d to 512",
                ONNX_MAX_SEQ_LEN,
            )
        # Pad to the longest sequence in the batch (length=None), not to a fixed
        # max length: fixed-length padding makes every forward pass cost the
        # full max_seq_len regardless of input size (~14x slower for short
        # texts). Output is unchanged — pooling honors the attention mask.
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tokenizer.enable_truncation(max_length=miniml_seq_len)

        logger.info(
            "ONNX MiniLM provider initialized (dir=%s, seq_len=%d, requested=%s, active=%s)",
            self._model_dir,
            miniml_seq_len,
            providers,
            self._session.get_providers(),
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._session or not self._tokenizer:
            raise RuntimeError("Provider not initialized")

        async with self._lock:
            return await asyncio.get_event_loop().run_in_executor(None, self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding (run in executor to avoid blocking)."""
        encodings = self._tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        token_embeddings = outputs[0]  # (batch, seq_len, hidden_dim)

        # Mean pooling + L2 normalization
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        mean_pooled = sum_embeddings / sum_mask

        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        normalized = mean_pooled / norms

        return normalized.tolist()

    def dimensions(self) -> int:
        return 384

    async def shutdown(self) -> None:
        self._session = None
        self._tokenizer = None


# ============================================================
# Platform detection helpers
# ============================================================


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and _platform.machine() == "arm64"


def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        import mlx_embeddings  # noqa: F401

        return True
    except ImportError:
        return False


# ============================================================
# CoreML-aware session factory (shared by ONNX providers)
# ============================================================


def _create_ort_session(model_path: str, providers: list):
    """Create InferenceSession with CoreML-aware 3-stage fallback.

    Stage 1: CoreML with MLProgram + dynamic shapes (ort 1.18+).
    Stage 2: CoreML without provider_options (ort version mismatch tolerance).
    Stage 3: CPU only (guaranteed fallback).
    """
    import onnxruntime as ort

    if "CoreMLExecutionProvider" in providers:
        rest = [p for p in providers if p != "CoreMLExecutionProvider"]
        providers_with_opts = [
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "0",
                },
            ),
            *rest,
        ]
        try:
            return ort.InferenceSession(model_path, providers=providers_with_opts)
        except Exception as e:
            logger.warning("CoreML init with provider_options failed, retrying without options: %s", e)
        try:
            return ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            logger.warning("CoreML plain init failed, falling back to CPU-only: %s", e)
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    return ort.InferenceSession(model_path, providers=providers)


# ============================================================
# mlx_bge_m3 Provider (macOS Apple Silicon only)
# ============================================================

MLX_BGE_M3_REPO = "mlx-community/bge-m3-mlx-fp16"


class MlxBgeM3Provider(EmbeddingProvider):
    """Apple Silicon MLX bge-m3 provider (fp16, Metal/ANE accelerated).

    Uses mlx-embeddings to run BAAI/bge-m3 natively on Metal GPU.
    macOS ARM only — auto_bge_m3 falls back to OnnxBgeM3Provider on other platforms.
    """

    def __init__(self, model_path: str = MLX_BGE_M3_REPO):
        # model_path: HuggingFace repo ID or absolute local directory
        self._model_path = model_path
        self._model = None
        self._tokenizer = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        try:
            from mlx_embeddings.utils import load as mlx_load
        except ImportError:
            raise ImportError("mlx_bge_m3 requires: uv pip install mlx-embeddings")

        self._model, self._tokenizer = await asyncio.get_event_loop().run_in_executor(None, mlx_load, self._model_path)
        logger.info("MLX BGE-M3 provider initialized (path=%s)", self._model_path)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._model or not self._tokenizer:
            raise RuntimeError("Provider not initialized")

        async with self._lock:
            return await asyncio.get_event_loop().run_in_executor(None, self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        import mlx.core as mx

        inputs = self._tokenizer.batch_encode_plus(
            texts,
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=min(ONNX_MAX_SEQ_LEN, 8192),
        )
        outputs = self._model(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        embs = outputs.text_embeds  # already L2-normalized
        mx.eval(embs)
        return embs.tolist()

    def dimensions(self) -> int:
        return 1024

    async def shutdown(self) -> None:
        self._model = None
        self._tokenizer = None


# ============================================================
# onnx_bge_m3 Provider
# ============================================================


class OnnxBgeM3Provider(EmbeddingProvider):
    """Local BAAI/bge-m3 ONNX embedding provider (int8 quantized).

    CLS-token pooling. 1024-dim dense output, 8K context, 100+ languages.
    Based on XLM-RoBERTa — no token_type_ids.
    """

    def __init__(self, model_dir: str):
        self._model_dir = model_dir
        self._session = None
        self._tokenizer = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        try:
            import onnxruntime  # noqa: F401
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError("onnx_bge_m3 provider requires: uv pip install onnxruntime tokenizers")

        model_dir_abs = os.path.abspath(self._model_dir)
        model_path = os.path.join(model_dir_abs, "model.onnx")
        tokenizer_path = os.path.join(model_dir_abs, "tokenizer.json")

        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            logger.info("BGE-M3 ONNX model not found, downloading...")
            try:
                from cembedding.download_model import download_bge_m3

                if not download_bge_m3(model_dir_abs):
                    raise FileNotFoundError(f"Failed to download BGE-M3 model to {model_dir_abs}")
            except ImportError:
                raise FileNotFoundError(
                    f"ONNX model not found at {model_path}. "
                    f"Download with: python -m cembedding.download_model --model bge-m3"
                )

        providers = _select_ort_providers()
        self._session = _create_ort_session(model_path, providers)
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        bge_seq_len = min(ONNX_MAX_SEQ_LEN, 8192)
        # XLM-RoBERTa pad token is <pad> (id=1)
        # Pad to longest in batch (see the MiniLM provider note): fixed-length
        # padding cost the full max_seq_len on every call; output unchanged.
        self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        self._tokenizer.enable_truncation(max_length=bge_seq_len)

        logger.info(
            "ONNX BGE-M3 provider initialized (dir=%s, seq_len=%d, requested=%s, active=%s)",
            self._model_dir,
            bge_seq_len,
            providers,
            self._session.get_providers(),
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._session or not self._tokenizer:
            raise RuntimeError("Provider not initialized")

        async with self._lock:
            return await asyncio.get_event_loop().run_in_executor(None, self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding with CLS-token pooling."""
        encodings = self._tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        input_names = [inp.name for inp in self._session.get_inputs()]
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self._session.run(None, inputs)
        token_embeddings = outputs[0]  # (batch, seq_len, 1024)

        # CLS-token pooling: first token of each sequence
        cls_embs = token_embeddings[:, 0, :].astype(np.float32, copy=False)

        # L2 normalization
        norms = np.linalg.norm(cls_embs, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        normalized = cls_embs / norms

        return normalized.tolist()

    def dimensions(self) -> int:
        return 1024

    async def shutdown(self) -> None:
        self._session = None
        self._tokenizer = None


# ============================================================
# onnx_jina_v5_nano Provider
# ============================================================


class OnnxJinaV5NanoProvider(EmbeddingProvider):
    """Local jina-embeddings-v5-text-nano ONNX embedding provider.

    Uses Last-Token pooling (different from MiniLM's mean pooling).
    768-dim output, 8K context, retrieval-optimized merged LoRA.
    """

    def __init__(self, model_dir: str):
        self._model_dir = model_dir
        self._session = None
        self._tokenizer = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        try:
            import onnxruntime  # noqa: F401  # availability check — session is created in _create_session_with_fallback
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError(
                "onnx_jina_v5_nano provider requires: uv pip install onnxruntime tokenizers\n"
                "Or: uv pip install cloto-mcp-embedding[onnx]"
            )

        model_path = os.path.join(self._model_dir, "model.onnx")
        tokenizer_path = os.path.join(self._model_dir, "tokenizer.json")

        # Auto-download model if missing
        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            logger.info("Jina-v5-nano ONNX model not found, downloading...")
            try:
                from cembedding.download_model import download_jina_v5_nano

                if not download_jina_v5_nano(self._model_dir):
                    raise FileNotFoundError(f"Failed to download model to {self._model_dir}")
            except ImportError:
                raise FileNotFoundError(
                    f"ONNX model not found at {model_path}. "
                    f"Download with: python -m cembedding.download_model --model jina-v5-nano"
                )

        providers = _select_ort_providers()
        self._session = self._create_session_with_fallback(model_path, providers)
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        # jina-v5-nano supports up to 8K context via RoPE.
        # Pad to longest in batch (see the MiniLM provider note): fixed-length
        # padding cost the full max_seq_len on every call; output unchanged.
        self._tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
        self._tokenizer.enable_truncation(max_length=ONNX_MAX_SEQ_LEN)

        logger.info(
            "ONNX Jina-v5-nano provider initialized (dir=%s, seq_len=%d, requested=%s, active=%s)",
            self._model_dir,
            ONNX_MAX_SEQ_LEN,
            providers,
            self._session.get_providers(),
        )

    def _create_session_with_fallback(self, model_path: str, providers: list):
        return _create_ort_session(model_path, providers)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._session or not self._tokenizer:
            raise RuntimeError("Provider not initialized")

        async with self._lock:
            return await asyncio.get_event_loop().run_in_executor(None, self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding with Last-Token pooling."""
        encodings = self._tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        # jina-v5-nano may not use token_type_ids — check model inputs
        input_names = [inp.name for inp in self._session.get_inputs()]
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self._session.run(None, inputs)
        token_embeddings = outputs[0]  # (batch, seq_len, hidden_dim=768)

        # Last-Token pooling (vectorized): gather embedding at the last non-padding token per row.
        last_indices = np.maximum(np.sum(attention_mask, axis=1) - 1, 0).astype(np.int64)
        batch_indices = np.arange(token_embeddings.shape[0])
        last_token_embs = token_embeddings[batch_indices, last_indices].astype(np.float32, copy=False)

        # L2 normalization
        norms = np.linalg.norm(last_token_embs, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        normalized = last_token_embs / norms

        return normalized.tolist()

    def dimensions(self) -> int:
        return 768

    async def shutdown(self) -> None:
        self._session = None
        self._tokenizer = None


# ============================================================
# Vector Index (v0.2.0)
# ============================================================


class _VectorGroup:
    """The resident vectors of one (namespace, dimension), in a single matrix.

    Rows ``[0, n)`` are in use and ``live`` says which of them still belong to
    an item. Writes never permute rows: an overwrite rewrites the item's own
    row, an append takes the next free row, and a removal only clears the
    liveness flag. That matters beyond speed — search breaks score ties by row
    order, so keeping positions stable keeps results stable. Tombstoned rows
    are reclaimed by ``maybe_compact`` once they outnumber the live ones, which
    bounds the wasted memory at half the matrix while still preserving the
    relative order of everything that survives.
    """

    __slots__ = ("dim", "matrix", "ids", "rows", "live", "n", "dead", "_backend_matrix", "_dirty")

    #: Smallest matrix allocation; below this the doubling is not worth the copies.
    MIN_CAPACITY = 64

    def __init__(self, dim: int, capacity: int = 0):
        self.dim = dim
        self.matrix = np.zeros((max(capacity, self.MIN_CAPACITY), dim), dtype=np.float32)
        self.ids: list[str] = []
        self.rows: dict[str, int] = {}
        self.live = np.zeros(self.matrix.shape[0], dtype=bool)
        self.n = 0
        self.dead = 0
        # Resident copy for a non-numpy backend, rebuilt lazily after writes.
        self._backend_matrix = None
        self._dirty = True

    @property
    def capacity(self) -> int:
        return self.matrix.shape[0]

    def put(self, item_id: str, vec: np.ndarray) -> None:
        """Insert a vector, or overwrite an existing id in place at its own row."""
        row = self.rows.get(item_id)
        if row is None:
            if self.n == self.capacity:
                self._grow()
            row = self.n
            self.n += 1
            self.ids.append(item_id)
            self.rows[item_id] = row
            self.live[row] = True
        self.matrix[row] = vec
        self._dirty = True

    def tombstone(self, item_id: str) -> bool:
        """Mark an id's row dead. Returns False when the id is not in this group."""
        row = self.rows.pop(item_id, None)
        if row is None:
            return False
        self.live[row] = False
        self.dead += 1
        self._dirty = True
        return True

    def _grow(self) -> None:
        """Double the allocation, so appending N items costs O(N) copies overall."""
        capacity = max(self.MIN_CAPACITY, self.capacity * 2)
        matrix = np.zeros((capacity, self.dim), dtype=np.float32)
        matrix[: self.n] = self.matrix[: self.n]
        live = np.zeros(capacity, dtype=bool)
        live[: self.n] = self.live[: self.n]
        self.matrix = matrix
        self.live = live

    def maybe_compact(self) -> None:
        """Reclaim tombstoned rows once they outnumber the live ones.

        Live rows keep their relative order, so search results and tie order are
        unaffected. The allocation itself is kept for the next appends.
        """
        if self.dead == 0 or self.dead <= self.n // 2:
            return
        keep = np.nonzero(self.live[: self.n])[0]
        self.matrix[: keep.size] = self.matrix[keep]
        self.live[: keep.size] = True
        self.live[keep.size : self.n] = False
        self.ids = [self.ids[row] for row in keep]
        self.rows = {item_id: row for row, item_id in enumerate(self.ids)}
        self.n = int(keep.size)
        self.dead = 0
        self._dirty = True

    def load(self, ids: list[str], block: np.ndarray) -> None:
        """Fill an empty group from one contiguous block of rows (startup path)."""
        if block.shape[0]:
            self.matrix[: block.shape[0]] = block
        self.ids = list(ids)
        self.rows = {item_id: row for row, item_id in enumerate(self.ids)}
        self.live[: len(self.ids)] = True
        self.n = len(self.ids)
        self.dead = 0
        self._dirty = True

    def operand(self):
        """Return the used rows in the form the search backend multiplies."""
        used = self.matrix[: self.n]
        if EMBEDDING_SEARCH_BACKEND != "mlx":
            return used
        # The numpy matrix stays the source of truth; the backend copy is
        # rebuilt only when a write since the last search invalidated it.
        if self._dirty or self._backend_matrix is None:
            self._backend_matrix = VectorIndex._to_backend(used)
            self._dirty = False
        return self._backend_matrix


class VectorIndex:
    """Persistent vector index with in-memory search.

    Vectors are stored in SQLite for durability and held in memory as one
    float32 matrix per (namespace, dimension); a query is a single matmul over
    that matrix instead of a per-item Python loop, which used to dominate
    wall-clock from a few thousand vectors up.

    There is exactly one resident copy of every vector. An earlier layout kept
    two — a per-item ndarray map plus a stacked search matrix rebuilt lazily —
    which doubled resident memory and made every write O(N), because the write
    discarded the stack and the next search re-stacked the whole namespace.
    Now an append is amortised O(1) into spare capacity, an overwrite touches
    one row, a removal flips one flag, and no write rebuilds the matrix.

    ``EMBEDDING_SEARCH_BACKEND=mlx`` opts into Apple-GPU matmul (falls back to
    numpy when mlx is unavailable); numpy (Accelerate BLAS) is the default.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db = None
        # Resident vectors: {namespace: {dimension: _VectorGroup}}
        self._groups: dict[str, dict[int, _VectorGroup]] = {}

    async def initialize(self) -> None:
        import aiosqlite

        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.executescript(
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
        await self._db.commit()

        # Load all vectors into memory. Blobs are collected per group first and
        # decoded in one pass per group: decoding them one by one would allocate
        # an array object per vector only to copy it into the matrix and drop it.
        rows = await self._db.execute_fetchall("SELECT namespace, item_id, vector FROM vectors")
        grouped: dict[tuple[str, int], tuple[list[str], list[bytes]]] = {}
        for ns, item_id, blob in rows:
            if len(blob) % 4:
                raise ValueError(f"vector blob for {ns}/{item_id} is not a whole number of float32 values")
            entry = grouped.get((ns, len(blob)))
            if entry is None:
                entry = grouped[(ns, len(blob))] = ([], [])
            entry[0].append(item_id)
            entry[1].append(blob)

        total = 0
        for (ns, byte_len), (ids, blobs) in grouped.items():
            dim = byte_len // 4
            group = _VectorGroup(dim, capacity=len(ids))
            # A zero-dimension blob carries no coordinates, so there is nothing
            # to decode; the ids are still tracked so counts stay accurate.
            block = (
                np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(-1, dim)
                if dim
                else np.zeros((0, 0), dtype=np.float32)
            )
            group.load(ids, block)
            self._groups.setdefault(ns, {})[dim] = group
            total += group.n

        logger.info("VectorIndex loaded: %d vectors across %d namespaces", total, len(self._groups))

    async def index(self, namespace: str, items: list[dict], provider: "EmbeddingProvider") -> int:
        """Index items. Each item has 'id' and 'text'. Returns count indexed."""
        if not self._db:
            raise RuntimeError("VectorIndex not initialized")
        if not items:
            return 0

        texts = [item["text"] for item in items]
        embeddings = await provider.embed(texts)
        if not embeddings:
            return 0

        # One conversion for the whole batch: the per-item vectors below are
        # views into it, so a batch of B vectors allocates one array, not B.
        batch = np.asarray(embeddings, dtype=np.float32)
        if batch.ndim != 2:
            raise ValueError(f"provider returned {batch.ndim}-D embeddings, expected one vector per text")

        ns_groups = self._groups.setdefault(namespace, {})
        params = []
        indexed = 0
        for item, vec in zip(items, batch):
            item_id = item["id"]
            # Native float32 bytes, i.e. the little-endian layout the loader reads back.
            params.append((namespace, item_id, vec.astype(np.float32, copy=False).tobytes()))
            self._put(ns_groups, item_id, vec)
            indexed += 1

        # One statement for the batch: a per-row execute paid a round trip and a
        # statement reset for every vector.
        await self._db.executemany(
            "INSERT OR REPLACE INTO vectors (namespace, item_id, vector) VALUES (?, ?, ?)",
            params,
        )
        await self._db.commit()
        return indexed

    def _put(self, ns_groups: dict[int, _VectorGroup], item_id: str, vec: np.ndarray) -> None:
        """Store one vector in its dimension's group, moving it if the dimension changed."""
        dim = int(vec.shape[0])
        group = ns_groups.get(dim)
        if group is None:
            group = ns_groups[dim] = _VectorGroup(dim)
        if item_id not in group.rows:
            # An id may hold only one vector, so a re-index at a new dimension
            # has to retire the row it occupies in the old dimension's group.
            for other_dim, other in ns_groups.items():
                if other_dim != dim and other.tombstone(item_id):
                    other.maybe_compact()
                    break
        group.put(item_id, vec)

    async def search(
        self,
        namespace: str,
        query: str,
        limit: int,
        min_similarity: float,
        provider: "EmbeddingProvider",
    ) -> list[dict]:
        """Search for similar vectors. Returns [{id, score}, ...] sorted by score desc."""
        ns_groups = self._groups.get(namespace)
        if not ns_groups or not any(group.rows for group in ns_groups.values()):
            return []

        embeddings = await provider.embed([query])
        if not embeddings or not embeddings[0]:
            return []

        query_vec = np.asarray(embeddings[0], dtype=np.float32)
        # Vectors of another dimension are not comparable and are skipped, the
        # same exclusion the per-item loop made with a length check.
        group = ns_groups.get(int(query_vec.shape[0]))
        if group is None or group.n == 0:
            return []
        sims = self._matmul(group.operand(), query_vec)

        # Vectorized threshold + top-K. Tombstoned rows still take part in the
        # matmul (skipping them would cost a copy of the matrix) and are dropped
        # here. A stable descending sort keeps row order among equal scores —
        # the same tie behavior as the heapq.nlargest over an insertion-ordered
        # candidate list it replaced.
        hits = np.nonzero((sims >= min_similarity) & group.live[: group.n])[0]
        if limit <= 0 or hits.size == 0:
            return []
        if hits.size > limit:
            hits = hits[np.argpartition(sims[hits], -limit)[-limit:]]
            hits.sort()  # restore row order so the stable sort's tie order matches
        order = hits[np.argsort(-sims[hits], kind="stable")]
        ids = group.ids
        return [{"id": ids[i], "score": round(float(sims[i]), 4)} for i in order]

    @staticmethod
    def _to_backend(matrix: np.ndarray):
        """Convert a matrix to the search backend's resident form.

        mlx arrays live in unified memory, so converting when the matrix changes
        (not per query) is what makes the GPU path pay off.
        """
        if EMBEDDING_SEARCH_BACKEND == "mlx":
            try:
                import mlx.core as mx

                return mx.array(matrix)
            except ImportError:
                logger.warning("EMBEDDING_SEARCH_BACKEND=mlx but mlx is not installed — using numpy")
        return matrix

    @staticmethod
    def _matmul(matrix, query_vec: np.ndarray) -> np.ndarray:
        """One similarity pass over the namespace matrix (numpy or mlx resident)."""
        if not isinstance(matrix, np.ndarray):  # mlx resident matrix
            import mlx.core as mx

            return np.asarray(mx.matmul(matrix, mx.array(query_vec)))
        return matrix @ query_vec

    async def remove(self, namespace: str, ids: list[str]) -> int:
        """Remove items from index. Returns count removed."""
        if not self._db:
            raise RuntimeError("VectorIndex not initialized")
        if not ids:
            return 0

        ns_groups = self._groups.get(namespace, {})
        for item_id in ids:
            for group in ns_groups.values():
                if group.tombstone(item_id):
                    break

        cursor = await self._db.executemany(
            "DELETE FROM vectors WHERE namespace = ? AND item_id = ?",
            [(namespace, item_id) for item_id in ids],
        )
        # executemany reports the rows deleted by the whole batch. With
        # (namespace, item_id) as the primary key each id deletes at most one
        # row, so this equals the number of ids that existed — the same figure
        # the per-id loop counted, including a repeated id counting once.
        removed = cursor.rowcount
        await self._db.commit()

        for dim in [dim for dim, group in ns_groups.items() if not group.rows]:
            del ns_groups[dim]
        for group in ns_groups.values():
            group.maybe_compact()
        return removed

    async def purge_namespace(self, namespace: str) -> int:
        """Remove all vectors in a namespace. Returns count removed."""
        if not self._db:
            raise RuntimeError("VectorIndex not initialized")

        cursor = await self._db.execute("DELETE FROM vectors WHERE namespace = ?", (namespace,))
        await self._db.commit()
        removed = sum(len(group.rows) for group in self._groups.pop(namespace, {}).values())
        return max(cursor.rowcount, removed)

    async def count(self, namespace: str) -> int:
        """Count vectors in a namespace."""
        return sum(len(group.rows) for group in self._groups.get(namespace, {}).values())

    async def shutdown(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
        self._groups.clear()


_vector_index: VectorIndex | None = None


# ============================================================
# Provider Factory
# ============================================================


def _resolve_model_dir(provider_key: str) -> str:
    """Resolve model directory for a provider key, respecting CLOTO_PROJECT_DIR."""
    rel = _MODEL_DIRS.get(provider_key, "data/models/bge-m3")
    if _project_dir and not os.path.isabs(rel):
        return os.path.join(_project_dir, rel)
    return rel


def create_provider() -> EmbeddingProvider:
    """Create an embedding provider based on configuration."""
    if EMBEDDING_PROVIDER == "api_openai":
        return OpenAIEmbeddingProvider(
            api_key=EMBEDDING_API_KEY,
            api_url=EMBEDDING_API_URL,
            model=EMBEDDING_MODEL,
            timeout=EMBEDDING_TIMEOUT,
        )
    elif EMBEDDING_PROVIDER == "onnx_miniml":
        return OnnxMiniLMProvider(model_dir=ONNX_MODEL_DIR)
    elif EMBEDDING_PROVIDER == "onnx_jina_v5_nano":
        return OnnxJinaV5NanoProvider(model_dir=ONNX_MODEL_DIR)
    elif EMBEDDING_PROVIDER == "onnx_bge_m3":
        return OnnxBgeM3Provider(model_dir=ONNX_MODEL_DIR)
    elif EMBEDDING_PROVIDER == "mlx_bge_m3":
        mlx_path = os.environ.get("MLX_MODEL_DIR", "")
        if not mlx_path:
            local = _resolve_model_dir("mlx_bge_m3")
            mlx_path = local if os.path.isdir(local) else MLX_BGE_M3_REPO
        return MlxBgeM3Provider(model_path=mlx_path)
    elif EMBEDDING_PROVIDER == "auto_bge_m3":
        if _is_apple_silicon() and _mlx_available():
            mlx_path = os.environ.get("MLX_MODEL_DIR", "") or MLX_BGE_M3_REPO
            logger.info("auto_bge_m3: Apple Silicon detected, using MLX provider")
            return MlxBgeM3Provider(model_path=mlx_path)
        logger.info("auto_bge_m3: falling back to ONNX CPU provider")
        return OnnxBgeM3Provider(model_dir=ONNX_MODEL_DIR)
    else:
        raise ValueError(
            f"Unknown embedding provider: {EMBEDDING_PROVIDER}. "
            f"Supported: api_openai, onnx_miniml, onnx_jina_v5_nano, onnx_bge_m3, "
            f"mlx_bge_m3, auto_bge_m3"
        )


# ============================================================
# HTTP Endpoint (for CPersona inter-server communication)
# ============================================================

_provider: EmbeddingProvider | None = None


async def handle_embed(request: web.Request) -> web.Response:
    """POST /embed — Generate embeddings for input texts."""
    if _provider is None:
        return web.json_response({"error": "Provider not initialized"}, status=503)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    texts = body.get("texts")
    if not isinstance(texts, list) or not texts:
        return web.json_response(
            {"error": "'texts' must be a non-empty array of strings"},
            status=400,
        )

    # Limit batch size to prevent OOM
    if len(texts) > 100:
        return web.json_response({"error": "Batch size exceeds limit (max 100)"}, status=400)

    try:
        embeddings = await _provider.embed(texts)
        return web.json_response(
            {
                "embeddings": embeddings,
                "dimensions": _provider.dimensions(),
            }
        )
    except Exception as e:
        logger.exception("Embedding failed")
        return web.json_response({"error": f"Embedding failed: {e}"}, status=500)


async def handle_index(request: web.Request) -> web.Response:
    """POST /index — Index vectors for later search."""
    if _provider is None or _vector_index is None:
        return web.json_response({"error": "Not initialized"}, status=503)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    namespace = body.get("namespace", "default")
    items = body.get("items")
    if not isinstance(items, list) or not items:
        return web.json_response({"error": "'items' must be a non-empty array"}, status=400)
    if len(items) > 100:
        return web.json_response({"error": "Batch size exceeds limit (max 100)"}, status=400)

    for item in items:
        if not isinstance(item, dict) or "id" not in item or "text" not in item:
            return web.json_response({"error": "Each item must have 'id' and 'text'"}, status=400)

    try:
        indexed = await _vector_index.index(namespace, items, _provider)
        return web.json_response({"ok": True, "indexed": indexed})
    except Exception as e:
        logger.exception("Index failed")
        return web.json_response({"error": f"Index failed: {e}"}, status=500)


async def handle_search(request: web.Request) -> web.Response:
    """POST /search — Search indexed vectors by similarity."""
    if _provider is None or _vector_index is None:
        return web.json_response({"error": "Not initialized"}, status=503)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    namespace = body.get("namespace", "default")
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return web.json_response({"error": "'query' must be a non-empty string"}, status=400)

    limit = min(int(body.get("limit", 10)), 500)
    min_similarity = float(body.get("min_similarity", 0.3))

    try:
        results = await _vector_index.search(namespace, query, limit, min_similarity, _provider)
        return web.json_response({"results": results})
    except Exception as e:
        logger.exception("Search failed")
        return web.json_response({"error": f"Search failed: {e}"}, status=500)


async def handle_remove(request: web.Request) -> web.Response:
    """POST /remove — Remove vectors from index."""
    if _vector_index is None:
        return web.json_response({"error": "Not initialized"}, status=503)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    namespace = body.get("namespace", "default")
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids:
        return web.json_response({"error": "'ids' must be a non-empty array"}, status=400)

    try:
        removed = await _vector_index.remove(namespace, ids)
        return web.json_response({"ok": True, "removed": removed})
    except Exception as e:
        logger.exception("Remove failed")
        return web.json_response({"error": f"Remove failed: {e}"}, status=500)


async def handle_purge(request: web.Request) -> web.Response:
    """POST /purge — Remove all vectors in a namespace."""
    if _vector_index is None:
        return web.json_response({"error": "Not initialized"}, status=503)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    namespace = body.get("namespace")
    if not isinstance(namespace, str) or not namespace.strip():
        return web.json_response({"error": "'namespace' must be a non-empty string"}, status=400)

    try:
        removed = await _vector_index.purge_namespace(namespace)
        return web.json_response({"ok": True, "removed": removed})
    except Exception as e:
        logger.exception("Purge failed")
        return web.json_response({"error": f"Purge failed: {e}"}, status=500)


async def run_http_server(port: int) -> None:
    """Run the HTTP embedding endpoint alongside MCP stdio."""
    auth_token = resolve_auth_token()
    check_startup("REST endpoint", f"127.0.0.1:{port}", auth_token)

    middlewares = [aiohttp_bearer_middleware(auth_token)] if auth_token else []
    app = web.Application(middlewares=middlewares)
    app.router.add_post("/embed", handle_embed)
    if EMBEDDING_INDEX_ENABLED and _vector_index is not None:
        app.router.add_post("/index", handle_index)
        app.router.add_post("/search", handle_search)
        app.router.add_post("/remove", handle_remove)
        app.router.add_post("/purge", handle_purge)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    logger.info("HTTP embedding endpoint started on http://127.0.0.1:%d/embed", port)

    try:
        # Block until cancelled
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


# ============================================================
# MCP Server
# ============================================================

registry = ToolRegistry("cloto-mcp-embedding")


@registry.tool(
    "embed",
    "Generate vector embeddings for input texts.",
    {
        "type": "object",
        "properties": {
            "texts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Texts to embed (batch, max 100)",
            }
        },
        "required": ["texts"],
    },
)
async def handle_embed_tool(arguments: dict) -> dict:
    if _provider is None:
        return {"error": "Provider not initialized"}

    texts = arguments.get("texts", [])
    if not isinstance(texts, list) or not texts:
        return {"error": "'texts' must be a non-empty array"}

    if len(texts) > 100:
        return {"error": "Batch size exceeds limit (max 100)"}

    try:
        embeddings = await _provider.embed(texts)
        return {
            "embeddings": embeddings,
            "dimensions": _provider.dimensions(),
        }
    except Exception as e:
        return {"error": str(e)}


@registry.tool(
    "index",
    "Index text items for vector similarity search. Each item gets embedded and stored persistently.",
    {
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": "Namespace for isolation (e.g., 'cpersona:agent-id')",
                "default": "default",
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["id", "text"],
                },
                "description": "Items to index (max 100)",
            },
        },
        "required": ["items"],
    },
)
async def handle_index_tool(arguments: dict) -> dict:
    if _provider is None or _vector_index is None:
        return {"error": "Not initialized or index disabled"}

    namespace = arguments.get("namespace", "default")
    items = arguments.get("items", [])
    if not items or len(items) > 100:
        return {"error": "items must be 1-100 entries"}

    try:
        indexed = await _vector_index.index(namespace, items, _provider)
        return {"ok": True, "indexed": indexed}
    except Exception as e:
        return {"error": str(e)}


@registry.tool(
    "search",
    "Search indexed vectors by semantic similarity. Returns top-K results with scores.",
    {
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": "Namespace to search within",
                "default": "default",
            },
            "query": {
                "type": "string",
                "description": "Search query text",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default: 10)",
                "default": 10,
            },
            "min_similarity": {
                "type": "number",
                "description": "Minimum cosine similarity threshold (default: 0.3)",
                "default": 0.3,
            },
        },
        "required": ["query"],
    },
)
async def handle_search_tool(arguments: dict) -> dict:
    if _provider is None or _vector_index is None:
        return {"error": "Not initialized or index disabled"}

    namespace = arguments.get("namespace", "default")
    query = arguments.get("query", "")
    limit = min(int(arguments.get("limit", 10)), 500)
    min_similarity = float(arguments.get("min_similarity", 0.3))

    if not query.strip():
        return {"error": "query must be non-empty"}

    try:
        results = await _vector_index.search(namespace, query, limit, min_similarity, _provider)
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}


@registry.tool(
    "remove",
    "Remove items from the vector index by ID.",
    {
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": "Namespace containing the items",
                "default": "default",
            },
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Item IDs to remove",
            },
        },
        "required": ["ids"],
    },
)
async def handle_remove_tool(arguments: dict) -> dict:
    if _vector_index is None:
        return {"error": "Index not initialized or disabled"}

    namespace = arguments.get("namespace", "default")
    ids = arguments.get("ids", [])
    if not ids:
        return {"error": "ids must be non-empty"}

    try:
        removed = await _vector_index.remove(namespace, ids)
        return {"ok": True, "removed": removed}
    except Exception as e:
        return {"error": str(e)}


@registry.tool(
    "purge",
    "Remove ALL vectors in a namespace. Use for bulk cleanup (e.g., agent deletion).",
    {
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": "Namespace to purge completely",
            },
        },
        "required": ["namespace"],
    },
)
async def handle_purge_tool(arguments: dict) -> dict:
    if _vector_index is None:
        return {"error": "Index not initialized or disabled"}

    namespace = arguments.get("namespace", "")
    if not namespace:
        return {"error": "namespace is required"}

    try:
        removed = await _vector_index.purge_namespace(namespace)
        return {"ok": True, "removed": removed}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Main
# ============================================================


async def _run_streamable_http() -> None:
    """Run embedding as a Streamable HTTP MCP server (no stdio, no REST /embed).

    Enabled by setting EMBEDDING_TRANSPORT=streamable-http. Listens on
    EMBEDDING_MCP_HTTP_PORT (default 8403) and mounts the MCP endpoint at
    /embedding/mcp and /embedding so it can coexist with other services
    behind path-based reverse proxies.
    """
    global _provider, _vector_index

    import contextlib
    from collections.abc import AsyncIterator

    import uvicorn
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.routing import Mount

    host = os.environ.get("EMBEDDING_MCP_HTTP_HOST", "0.0.0.0")
    port = int(os.environ.get("EMBEDDING_MCP_HTTP_PORT", "8403"))
    # Checked before the provider loads a model: an auth misconfiguration
    # should fail in milliseconds, not after an ONNX warm-up.
    auth_token = resolve_auth_token()
    check_startup("Streamable HTTP MCP", f"{host}:{port}", auth_token)

    _provider = create_provider()
    await _provider.initialize()

    if EMBEDDING_INDEX_ENABLED:
        _vector_index = VectorIndex(EMBEDDING_INDEX_DB_PATH)
        await _vector_index.initialize()

    session_manager = StreamableHTTPSessionManager(
        app=registry.server,
        stateless=True,
    )

    async def mcp_endpoint(scope, receive, send):
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            logger.info("Embedding Streamable HTTP server ready")
            yield

    app = Starlette(
        routes=[
            Mount("/embedding/mcp", app=mcp_endpoint),
            Mount("/embedding", app=mcp_endpoint),
            Mount("/mcp", app=mcp_endpoint),
            Mount("/", app=mcp_endpoint),
        ],
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=["https://claude.ai", "https://www.claude.ai"],
                allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
                allow_headers=[
                    "Authorization",
                    "Content-Type",
                    "Mcp-Session-Id",
                    "Mcp-Protocol-Version",
                    "Last-Event-Id",
                ],
                expose_headers=["Mcp-Session-Id"],
            ),
            # Inside CORS so 401s still carry the CORS headers a browser
            # client needs to read them.
            Middleware(BearerTokenMiddleware, auth_token=auth_token),
        ],
        lifespan=lifespan,
    )

    logger.info("Starting Embedding Streamable HTTP MCP on %s:%d", host, port)

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    try:
        await server.serve()
    finally:
        if _vector_index:
            await _vector_index.shutdown()
        await _provider.shutdown()
        logger.info("Embedding Streamable HTTP server shut down")


async def main():
    global _provider, _vector_index

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    transport = os.environ.get("EMBEDDING_TRANSPORT", "stdio")
    if transport == "streamable-http":
        await _run_streamable_http()
        return

    logger.info(
        "Starting embedding server (provider=%s, http_port=%d, index=%s)",
        EMBEDDING_PROVIDER,
        EMBEDDING_HTTP_PORT,
        "enabled" if EMBEDDING_INDEX_ENABLED else "disabled",
    )

    _provider = create_provider()
    await _provider.initialize()

    # Initialize vector index if enabled
    if EMBEDDING_INDEX_ENABLED:
        _vector_index = VectorIndex(EMBEDDING_INDEX_DB_PATH)
        await _vector_index.initialize()

    # Start HTTP endpoint as background task
    http_task = asyncio.create_task(run_http_server(EMBEDDING_HTTP_PORT))

    try:
        async with stdio_server() as (read_stream, write_stream):
            await registry.server.run(
                read_stream,
                write_stream,
                registry.server.create_initialization_options(),
            )
    finally:
        http_task.cancel()
        try:
            await http_task
        except asyncio.CancelledError:
            pass
        if _vector_index:
            await _vector_index.shutdown()
        await _provider.shutdown()
        logger.info("Embedding server shut down")


def run():
    """Console-script / ``python -m cembedding`` entry point (sync wrapper)."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
