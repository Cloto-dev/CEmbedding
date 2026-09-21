"""The identity of the backend behind the vectors.

A caller stores vectors next to the name of whatever produced them, and later
compares that name to decide whether a stored vector may still be compared with
a fresh one. Over HTTP the caller never learns the name: it posts texts and the
server picks the model, so the caller falls back to a configured default and two
different models read as one.

These tests pin what makes the report usable for that decision: every component
that moves the numbers moves the fingerprint, an identity the server cannot
complete is reported as incomplete rather than fingerprinted over the half it
knows, a model replaced in place gets a new digest, and the response `/embed`
returns is the one it returned before this endpoint existed.
"""

import os
import sys
import types

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from cembedding import server
from cembedding.server import _TokenCounter, backend_identity

COMPLETE = {
    "model": "bge-m3",
    "dimensions": 1024,
    "window": 512,
    "pooling": "cls",
    "normalized": True,
}


class FakeProvider:
    """A provider with no model behind it, reporting whatever it is handed."""

    def __init__(self, identity=None, files=None):
        self._identity = COMPLETE if identity is None else identity
        self._files = files or {}
        self.embed_calls = 0

    async def embed(self, texts):
        self.embed_calls += 1
        return [[0.0, 1.0] for _ in texts]

    def dimensions(self):
        return 2

    def identity(self):
        return dict(self._identity)

    def identity_files(self):
        return dict(self._files)

    token_counter = server.EmbeddingProvider.token_counter


@pytest.fixture(autouse=True)
def clean_digest_cache():
    """Each test starts with an empty cache: a digest kept from another test's
    file would hide both the caching test and the re-read test."""
    server._DIGEST_CACHE.clear()
    yield
    server._DIGEST_CACHE.clear()


@pytest.fixture
def model_files(tmp_path):
    graph = tmp_path / "model.onnx"
    tokenizer = tmp_path / "tokenizer.json"
    graph.write_bytes(b"graph-bytes")
    tokenizer.write_bytes(b"tokenizer-bytes")
    return {"graph": str(graph), "tokenizer": str(tokenizer)}


def _client(auth_token=None):
    return TestClient(TestServer(server.build_http_app(auth_token)))


# ------------------------------------------------------------- the fingerprint


async def test_a_complete_identity_is_fingerprinted(model_files, monkeypatch):
    monkeypatch.setattr(server, "EMBEDDING_PROVIDER", "onnx_bge_m3")
    report = await backend_identity(FakeProvider(files=model_files))

    assert report["incomplete"] == []
    assert report["fingerprint"].startswith("1:")
    assert report["identity"]["provider"] == "onnx_bge_m3"
    assert set(report["identity"]["digests"]) == {"graph", "tokenizer"}


async def test_the_same_backend_fingerprints_the_same_twice(model_files, monkeypatch):
    monkeypatch.setattr(server, "EMBEDDING_PROVIDER", "onnx_bge_m3")
    first = await backend_identity(FakeProvider(files=model_files))
    server._DIGEST_CACHE.clear()  # recomputed from the files, not replayed
    second = await backend_identity(FakeProvider(files=model_files))

    assert first["fingerprint"] == second["fingerprint"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("model", "bge-m3-other"),
        ("dimensions", 768),
        ("window", 8192),
        ("pooling", "mean"),
        ("normalized", False),
    ],
)
async def test_every_field_that_moves_the_vectors_moves_the_fingerprint(field, value, model_files, monkeypatch):
    """The point of the endpoint. A component that changes the numbers and not
    the fingerprint would let a caller compare vectors from two backends."""
    monkeypatch.setattr(server, "EMBEDDING_PROVIDER", "onnx_bge_m3")
    base = await backend_identity(FakeProvider(files=model_files))
    changed = await backend_identity(FakeProvider({**COMPLETE, field: value}, files=model_files))

    assert changed["fingerprint"] is not None
    assert changed["fingerprint"] != base["fingerprint"]


async def test_the_provider_is_part_of_the_identity(model_files, monkeypatch):
    monkeypatch.setattr(server, "EMBEDDING_PROVIDER", "onnx_bge_m3")
    base = await backend_identity(FakeProvider(files=model_files))
    monkeypatch.setattr(server, "EMBEDDING_PROVIDER", "mlx_bge_m3")
    changed = await backend_identity(FakeProvider(files=model_files))

    assert changed["fingerprint"] != base["fingerprint"]


@pytest.mark.parametrize("label", ["graph", "tokenizer"])
async def test_replacing_a_file_in_place_changes_the_fingerprint(label, model_files, monkeypatch):
    """Same path, same configuration, different bytes — the case a configured
    model name cannot see at all."""
    monkeypatch.setattr(server, "EMBEDDING_PROVIDER", "onnx_bge_m3")
    base = await backend_identity(FakeProvider(files=model_files))

    with open(model_files[label], "wb") as f:
        f.write(b"replaced-bytes-of-a-different-length")
    changed = await backend_identity(FakeProvider(files=model_files))

    assert changed["fingerprint"] != base["fingerprint"]


async def test_a_second_weights_file_is_part_of_the_identity(model_files, tmp_path, monkeypatch):
    """A graph that names its weights in a sibling file: hashing the graph alone
    would give one digest to two different sets of weights."""
    monkeypatch.setattr(server, "EMBEDDING_PROVIDER", "onnx_jina_v5_nano")
    weights = tmp_path / "model.onnx_data"
    weights.write_bytes(b"fp32-weights")
    with_weights = {**model_files, "weights": str(weights)}

    base = await backend_identity(FakeProvider(files=with_weights))
    weights.write_bytes(b"int8-weights")
    changed = await backend_identity(FakeProvider(files=with_weights))

    assert base["incomplete"] == []
    assert changed["fingerprint"] != base["fingerprint"]


# ------------------------------------------------------------ what is not known


@pytest.mark.parametrize("field", list(COMPLETE))
async def test_a_missing_field_is_reported_and_not_fingerprinted(field, model_files, monkeypatch):
    monkeypatch.setattr(server, "EMBEDDING_PROVIDER", "onnx_bge_m3")
    partial = {k: v for k, v in COMPLETE.items() if k != field}
    report = await backend_identity(FakeProvider(partial, files=model_files))

    assert report["fingerprint"] is None
    assert field in report["incomplete"]


@pytest.mark.parametrize("label", ["graph", "tokenizer"])
async def test_a_file_that_cannot_be_read_is_missing_not_empty(label, model_files, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "EMBEDDING_PROVIDER", "onnx_bge_m3")
    files = {**model_files, label: str(tmp_path / "absent")}
    report = await backend_identity(FakeProvider(files=files))

    assert report["fingerprint"] is None
    assert f"digests.{label}" in report["incomplete"]
    assert label not in report["identity"].get("digests", {})


async def test_a_backend_with_no_files_names_both_digests(monkeypatch):
    """A remote API: the graph and the tokenizer are on the other side of the
    network, so the identity stays incomplete however complete the fields are."""
    monkeypatch.setattr(server, "EMBEDDING_PROVIDER", "api_openai")
    report = await backend_identity(FakeProvider(files={}))

    assert report["fingerprint"] is None
    assert report["incomplete"] == ["digests.graph", "digests.tokenizer"]


async def test_two_incomplete_backends_are_not_told_apart_by_null(monkeypatch):
    """Both answer null, which is why null is not a value a caller may compare.
    The report says so by naming the missing parts instead."""
    monkeypatch.setattr(server, "EMBEDDING_PROVIDER", "api_openai")
    one = await backend_identity(FakeProvider({**COMPLETE, "model": "one"}))
    other = await backend_identity(FakeProvider({**COMPLETE, "model": "other"}))

    assert one["fingerprint"] is other["fingerprint"] is None
    assert one["identity"]["model"] != other["identity"]["model"]


# ------------------------------------------------------------------- the digest


async def test_the_digest_is_computed_once_per_file_state(model_files, monkeypatch):
    monkeypatch.setattr(server, "EMBEDDING_PROVIDER", "onnx_bge_m3")
    reads = []
    original = server._digest_file
    monkeypatch.setattr(server, "_digest_file", lambda p: reads.append(p) or original(p))

    provider = FakeProvider(files=model_files)
    await backend_identity(provider)
    await backend_identity(provider)
    assert len(reads) == 2  # graph and tokenizer, once each

    with open(model_files["graph"], "wb") as f:
        f.write(b"replaced")
    await backend_identity(provider)
    assert len(reads) == 3  # the replaced file, and only it


# ----------------------------------------------------------------- the endpoint


async def test_capabilities_reports_over_http(model_files, monkeypatch):
    monkeypatch.setattr(server, "EMBEDDING_PROVIDER", "onnx_bge_m3")
    monkeypatch.setattr(server, "_provider", FakeProvider(files=model_files))
    async with _client() as client:
        resp = await client.get("/capabilities")
        body = await resp.json()

    assert resp.status == 200
    assert body["identity"]["model"] == "bge-m3"
    assert body["fingerprint"].startswith("1:")


async def test_capabilities_answers_503_before_the_model_is_loaded(monkeypatch):
    monkeypatch.setattr(server, "_provider", None)
    async with _client() as client:
        resp = await client.get("/capabilities")
    assert resp.status == 503


async def test_capabilities_is_behind_the_same_bearer_check(model_files, monkeypatch):
    monkeypatch.setattr(server, "_provider", FakeProvider(files=model_files))
    async with _client(auth_token="secret") as client:
        assert (await client.get("/capabilities")).status == 401
        resp = await client.get("/capabilities", headers={"Authorization": "Bearer secret"})
        assert resp.status == 200


async def test_embed_answers_exactly_as_it_did_before(model_files, monkeypatch):
    """The identity is on its own route. A caller that never asks for it receives
    the response it received before this endpoint existed."""
    monkeypatch.setattr(server, "_provider", FakeProvider(files=model_files))
    async with _client() as client:
        resp = await client.post("/embed", json={"texts": ["alpha beta"]})
        assert await resp.json() == {"embeddings": [[0.0, 1.0]], "dimensions": 2}


# ------------------------------------------------- what each provider declares


@pytest.mark.parametrize(
    "cls, model, pooling",
    [
        (server.OnnxMiniLMProvider, "all-MiniLM-L6-v2", "mean"),
        (server.OnnxBgeM3Provider, "bge-m3", "cls"),
        (server.OnnxJinaV5NanoProvider, "jina-embeddings-v5-text-nano-retrieval", "last_token"),
    ],
)
def test_each_onnx_provider_declares_the_pooling_it_performs(cls, model, pooling):
    """The declared reduction and the performed one are separate statements of
    one fact. Changing `_embed_sync` without changing the declaration would leave
    the fingerprint equal across vectors that are not, so both have to move."""
    provider = cls.__new__(cls)
    provider._counter = _TokenCounter(None, 512, None)
    identity = provider.identity()

    assert identity["model"] == model
    assert identity["pooling"] == pooling
    assert identity["window"] == 512


def test_a_provider_that_cannot_see_its_tokens_omits_the_window():
    """No window reported is a missing component, never an unbounded one."""
    provider = server.OnnxBgeM3Provider.__new__(server.OnnxBgeM3Provider)
    assert "window" not in provider.identity()


def test_a_provider_reports_the_files_it_opened(tmp_path):
    provider = server.OnnxBgeM3Provider.__new__(server.OnnxBgeM3Provider)
    assert provider.identity_files() == {}

    provider._loaded_files = {"graph": str(tmp_path / "model.onnx")}
    assert provider.identity_files() == {"graph": str(tmp_path / "model.onnx")}


def _toy_tokenizer() -> Tokenizer:
    tok = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return tok


def _stub_onnx_load(monkeypatch):
    """Let a provider's initialize() run without a model on disk.

    ``os.path.exists`` is left alone: whether the sibling weights file is there
    is the thing under test, so a stub answering True for every path would make
    the assertion pass on a provider that never looked.
    """
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    session = type("Session", (), {"get_providers": lambda self: ["CPUExecutionProvider"]})()
    monkeypatch.setattr(server, "_create_ort_session", lambda *a, **k: session)
    monkeypatch.setattr(Tokenizer, "from_file", staticmethod(lambda _p: _toy_tokenizer()))


@pytest.mark.parametrize("with_weights", [True, False])
async def test_the_jina_provider_records_the_weights_beside_its_graph(monkeypatch, tmp_path, with_weights):
    """The graph of this model is a stub naming its weights in a sibling file.
    Recording only the graph would give one identity to two precisions, so this
    pins the provider that opens them rather than the helper that hashes them."""
    from cembedding.download_model import JINA_VARIANT_STEMS

    _stub_onnx_load(monkeypatch)
    stem = JINA_VARIANT_STEMS[server.EMBEDDING_MODEL_VARIANT]
    graph = tmp_path / f"{stem}.onnx"
    graph.write_bytes(b"graph")
    (tmp_path / "tokenizer.json").write_bytes(b"{}")
    weights = tmp_path / f"{stem}.onnx_data"
    if with_weights:
        weights.write_bytes(b"weights")

    provider = server.OnnxJinaV5NanoProvider(str(tmp_path))
    await provider.initialize()
    files = provider.identity_files()

    assert files["graph"] == str(graph)
    assert files["tokenizer"] == str(tmp_path / "tokenizer.json")
    assert files.get("weights") == (str(weights) if with_weights else None)


@pytest.mark.parametrize(
    "cls, stem",
    [(server.OnnxMiniLMProvider, "model"), (server.OnnxBgeM3Provider, "model")],
)
async def test_a_single_file_provider_records_its_graph_and_tokenizer(monkeypatch, tmp_path, cls, stem):
    _stub_onnx_load(monkeypatch)
    (tmp_path / f"{stem}.onnx").write_bytes(b"graph")
    (tmp_path / "tokenizer.json").write_bytes(b"{}")

    provider = cls(str(tmp_path))
    await provider.initialize()

    assert provider.identity_files() == {
        "graph": os.path.join(str(tmp_path), f"{stem}.onnx"),
        "tokenizer": os.path.join(str(tmp_path), "tokenizer.json"),
    }
