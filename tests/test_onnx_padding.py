"""Padding-regression tests for the ONNX providers.

The providers must pad each batch to its LONGEST sequence, not to a fixed
ONNX_MAX_SEQ_LEN: fixed-length padding makes every forward pass cost the
full max length regardless of input size (measured ~90x slower for short
queries with the default 2048), while producing bit-identical embeddings —
pooling honors the attention mask, so the extra pad tokens never contribute.

These tests need the jina-v5-nano model assets on disk (they exercise the
real tokenizer and, for the parity test, the real ONNX session). CI runs
without model downloads, so they self-skip there; run them locally after
`python -m cembedding.download_model --model jina-v5-nano`.
"""

import os

import numpy as np
import pytest

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "models", "jina-embeddings-v5-text-nano",
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_DIR, "tokenizer.json")),
    reason="jina-v5-nano model assets not downloaded (local-only test)",
)

TEXTS = [
    "short query",
    "what did we decide about the deployment schedule last week?",
    "a medium document " + "with some repeated content " * 20,
]


def test_batch_pads_to_longest_not_max_seq_len():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    tok.enable_padding(pad_id=0, pad_token="<pad>")  # provider configuration
    tok.enable_truncation(max_length=2048)
    encodings = tok.encode_batch(TEXTS)
    lengths = {len(e.ids) for e in encodings}
    longest_real = max(sum(e.attention_mask) for e in encodings)
    assert lengths == {longest_real}, "batch must be padded to its longest row"
    assert longest_real < 2048, "test texts must not hit the truncation cap"


def test_embeddings_match_fixed_length_padding():
    """Pad-to-longest must reproduce the fixed-length-padding embeddings."""
    if not os.path.exists(os.path.join(MODEL_DIR, "model.onnx")):
        pytest.skip("jina-v5-nano ONNX weights not downloaded (local-only test)")
    import onnxruntime as ort
    from tokenizers import Tokenizer

    sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, "model.onnx"), providers=["CPUExecutionProvider"]
    )
    input_names = [i.name for i in sess.get_inputs()]

    def embed(fixed_len):
        tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        if fixed_len:
            tok.enable_padding(pad_id=0, pad_token="<pad>", length=fixed_len)
        else:
            tok.enable_padding(pad_id=0, pad_token="<pad>")
        tok.enable_truncation(max_length=2048)
        encs = tok.encode_batch(TEXTS)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        am = np.array([e.attention_mask for e in encs], dtype=np.int64)
        inputs = {"input_ids": ids, "attention_mask": am}
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.zeros_like(ids)
        out = sess.run(None, inputs)[0]
        last = np.maximum(am.sum(1) - 1, 0)
        vecs = out[np.arange(out.shape[0]), last].astype(np.float32)
        return vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)

    fixed, dynamic = embed(1024), embed(None)
    cos = (fixed * dynamic).sum(axis=1)
    assert np.all(cos > 1 - 1e-5), f"padding regime changed the embeddings: {cos}"
