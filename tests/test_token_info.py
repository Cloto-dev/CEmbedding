"""Token accounting against the embedding window.

A caller storing long records has to know where the vector stops seeing them.
These tests pin the three things that make the report trustworthy: the prefix
it names is exactly the prefix the embedding path truncates to (same token ids,
same end offset), offsets are characters rather than bytes, and a caller who
does not ask for the report receives the response it received before.

The parity tests against a real tokenizer need the jina-v5-nano assets on disk
and self-skip where the model was never downloaded, like the padding tests.
"""

import os

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from cembedding import server
from cembedding.server import _TokenCounter

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "models",
    "jina-embeddings-v5-text-nano",
)
JINA_TOKENIZER = os.path.join(MODEL_DIR, "tokenizer.json")

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


def _toy_tokenizer() -> Tokenizer:
    """Word-level tokenizer that wraps every text in [CLS] ... [SEP] (two specials)."""
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    return tok


def _truncating(tok_factory, window):
    tok = tok_factory()
    tok.enable_truncation(max_length=window)
    return tok


# ------------------------------------------------------------------ counting


def test_a_text_inside_the_window_is_reported_whole():
    counter = _TokenCounter(_toy_tokenizer(), window=8)
    text = "alpha beta gamma"
    [info] = counter.count([text])
    assert info == {"count": 5, "window": 8, "truncated": False, "window_end_char": len(text)}


def test_exactly_filling_the_window_is_not_truncated_and_one_more_token_is():
    counter = _TokenCounter(_toy_tokenizer(), window=5)
    fits, over = counter.count(["alpha beta gamma", "alpha beta gamma delta"])
    assert (fits["count"], fits["truncated"]) == (5, False)
    assert (over["count"], over["truncated"]) == (6, True)
    # Two specials leave three content tokens: the embedded prefix ends after "gamma".
    assert over["window_end_char"] == len("alpha beta gamma")


def test_the_reported_prefix_is_the_prefix_the_embedding_path_keeps():
    window = 6
    counter = _TokenCounter(_toy_tokenizer(), window=window)
    truncating = _truncating(_toy_tokenizer, window)
    text = " ".join(WORDS)
    [info] = counter.count([text])
    kept = truncating.encode(text)
    content = [i for i, special in enumerate(kept.special_tokens_mask) if not special]
    assert info["window_end_char"] == kept.offsets[content[-1]][1]
    # And re-tokenizing that prefix yields exactly the kept content tokens.
    assert _toy_tokenizer().encode(text[: info["window_end_char"]]).ids == kept.ids


@pytest.mark.skipif(not os.path.exists(JINA_TOKENIZER), reason="jina-v5-nano assets not downloaded")
@pytest.mark.parametrize(
    "text",
    [
        "Retrieval-augmented memory keeps the tail of a long record out of reach unless it is split.",
        "長い記憶の尾はベクトル検索から見えないので、分割して各ノードに埋め込みを持たせる必要がある。",
    ],
)
def test_parity_with_the_real_truncating_tokenizer(text):
    window = 16
    counter = _TokenCounter.from_file(JINA_TOKENIZER, window)
    truncating = _truncating(lambda: Tokenizer.from_file(JINA_TOKENIZER), window)
    [info] = counter.count([text])
    kept = truncating.encode(text)
    assert info["truncated"] is True and len(kept.ids) == window
    content = [i for i, special in enumerate(kept.special_tokens_mask) if not special]
    assert info["window_end_char"] == kept.offsets[content[-1]][1]


@pytest.mark.skipif(not os.path.exists(JINA_TOKENIZER), reason="jina-v5-nano assets not downloaded")
def test_offsets_are_characters_not_bytes():
    text = "長い記憶の尾はベクトル検索から見えない。"
    [info] = _TokenCounter.from_file(JINA_TOKENIZER, 8).count([text])
    # 8 tokens cannot hold this text, and its UTF-8 form is three times its length:
    # a byte offset would point past the end of the string.
    assert info["truncated"] is True
    assert 0 < info["window_end_char"] < len(text)
    [whole] = _TokenCounter.from_file(JINA_TOKENIZER, 512).count([text])
    assert whole["window_end_char"] == len(text)


# ---------------------------------------------------------------------- HTTP


class FakeProvider:
    def __init__(self, counter):
        self._counter = counter
        self.embed_calls = 0

    async def embed(self, texts):
        self.embed_calls += 1
        return [[0.0, 1.0] for _ in texts]

    def dimensions(self):
        return 2

    token_counter = server.EmbeddingProvider.token_counter


@pytest.fixture
def provider(monkeypatch):
    fake = FakeProvider(_TokenCounter(_toy_tokenizer(), window=5))
    monkeypatch.setattr(server, "_provider", fake)
    return fake


def _client(auth_token=None):
    return TestClient(TestServer(server.build_http_app(auth_token)))


async def test_embed_without_the_flag_answers_as_before(provider):
    async with _client() as client:
        resp = await client.post("/embed", json={"texts": ["alpha beta"]})
        assert resp.status == 200
        assert await resp.json() == {"embeddings": [[0.0, 1.0]], "dimensions": 2}


async def test_embed_with_the_flag_carries_one_report_per_text(provider):
    async with _client() as client:
        resp = await client.post("/embed", json={"texts": ["alpha", "alpha beta gamma delta"], "token_info": True})
        body = await resp.json()
    assert resp.status == 200
    assert [i["truncated"] for i in body["token_info"]] == [False, True]
    assert body["embeddings"] == [[0.0, 1.0], [0.0, 1.0]]


@pytest.mark.parametrize("flag", ["true", 1, None])
async def test_the_flag_must_be_a_boolean(provider, flag):
    async with _client() as client:
        resp = await client.post("/embed", json={"texts": ["alpha"], "token_info": flag})
    assert resp.status == 400
    assert provider.embed_calls == 0


async def test_count_tokens_does_not_run_the_model(provider):
    async with _client() as client:
        resp = await client.post("/count_tokens", json={"texts": ["alpha beta gamma delta"]})
        body = await resp.json()
    assert resp.status == 200
    assert body["token_info"][0]["count"] == 6
    assert provider.embed_calls == 0


@pytest.mark.parametrize("texts", [[], "alpha", ["alpha", 3], ["alpha"] * 101])
async def test_count_tokens_validates_its_input(provider, texts):
    async with _client() as client:
        resp = await client.post("/count_tokens", json={"texts": texts})
    assert resp.status == 400


async def test_a_provider_that_cannot_see_its_tokens_reports_unknown(monkeypatch):
    monkeypatch.setattr(server, "_provider", FakeProvider(None))
    async with _client() as client:
        counted = await (await client.post("/count_tokens", json={"texts": ["alpha"]})).json()
        embedded = await (await client.post("/embed", json={"texts": ["alpha"], "token_info": True})).json()
    assert counted == {"token_info": None}
    assert embedded["token_info"] is None


async def test_count_tokens_sits_behind_the_same_token_as_embed(provider):
    async with _client("s3cret") as client:
        denied = await client.post("/count_tokens", json={"texts": ["alpha"]})
        allowed = await client.post(
            "/count_tokens", json={"texts": ["alpha"]}, headers={"Authorization": "Bearer s3cret"}
        )
    assert denied.status == 401
    assert allowed.status == 200


@pytest.mark.parametrize(
    ("cls", "seq_len", "expected_window"),
    [
        (server.OnnxMiniLMProvider, 1000, 512),  # MiniLM clamps to its 512 positions
        (server.OnnxBgeM3Provider, 1000, 1000),
        (server.OnnxJinaV5NanoProvider, 1000, 1000),
    ],
)
async def test_each_local_provider_counts_against_the_window_it_truncates_to(
    monkeypatch, tmp_path, cls, seq_len, expected_window
):
    """The counter's window and the embedding tokenizer's truncation must be one number."""
    monkeypatch.setattr(server, "ONNX_MAX_SEQ_LEN", seq_len)
    session = type("Session", (), {"get_providers": lambda self: ["CPUExecutionProvider"]})()
    monkeypatch.setattr(server, "_create_ort_session", lambda *a, **k: session)
    monkeypatch.setattr(server.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(Tokenizer, "from_file", staticmethod(lambda _p: _toy_tokenizer()))
    provider = cls(str(tmp_path))
    await provider.initialize()
    assert provider._tokenizer.truncation["max_length"] == expected_window
    assert provider.token_counter().window == expected_window
