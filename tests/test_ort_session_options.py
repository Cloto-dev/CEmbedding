"""ONNX Runtime session configuration.

Every ONNX provider builds its session through ``_create_ort_session`` so the
runtime is configured in exactly one place. These tests pin the mapping from
the environment knobs to ``SessionOptions``; the parity test at the end needs
the jina-v5-nano assets on disk and self-skips in CI, like the padding tests.
"""

import os

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")

from cembedding import server  # noqa: E402

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "models",
    "jina-embeddings-v5-text-nano",
)


def test_defaults_leave_threads_to_the_runtime_and_enable_all_optimizations(monkeypatch):
    monkeypatch.setattr(server, "ONNX_INTRA_OP_THREADS", 0)
    monkeypatch.setattr(server, "ONNX_GRAPH_OPT_LEVEL", "all")
    options = server._ort_session_options()
    assert options.intra_op_num_threads == 0  # 0 is the runtime's "decide yourself"
    assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("disable", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
        ("extended", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
        ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ],
)
def test_graph_optimization_level_maps_every_documented_value(monkeypatch, level, expected):
    monkeypatch.setattr(server, "ONNX_GRAPH_OPT_LEVEL", level)
    assert server._ort_session_options().graph_optimization_level == expected


def test_explicit_thread_count_reaches_the_session_options(monkeypatch):
    monkeypatch.setattr(server, "ONNX_INTRA_OP_THREADS", 3)
    assert server._ort_session_options().intra_op_num_threads == 3


def test_every_session_constructor_call_passes_the_options():
    """A provider that bypasses the shared options would silently drift."""
    import inspect

    source = inspect.getsource(server)
    calls = [line for line in source.splitlines() if "ort.InferenceSession(" in line]
    assert calls, "expected at least one session construction site"
    assert all("sess_options=options" in line for line in calls), calls


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_DIR, "model.onnx")),
    reason="jina-v5-nano model assets not downloaded (local-only test)",
)
def test_optimized_and_unoptimized_graphs_agree(monkeypatch):
    """Graph fusion may reorder float arithmetic; it must not change the embedding."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    tok.enable_padding(pad_id=0, pad_token="<pad>")
    enc = tok.encode_batch(["what did we decide about the deployment schedule last week?"])
    inputs = {
        "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
    }
    outputs = {}
    for level in ("disable", "all"):
        monkeypatch.setattr(server, "ONNX_GRAPH_OPT_LEVEL", level)
        session = server._create_ort_session(os.path.join(MODEL_DIR, "model.onnx"), ["CPUExecutionProvider"])
        assert (
            session.get_session_options().graph_optimization_level
            == server._ort_session_options().graph_optimization_level
        )
        outputs[level] = session.run(None, inputs)[0][0, -1]
    a, b = outputs["disable"], outputs["all"]
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cosine >= 0.9999, cosine
