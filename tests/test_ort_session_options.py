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


def _fresh_server_module(monkeypatch, value):
    """Import server.py as a new module, so the running server's own settings stand."""
    import importlib.util

    if value is None:
        monkeypatch.delenv("ONNX_ALLOW_SPINNING", raising=False)
    else:
        monkeypatch.setenv("ONNX_ALLOW_SPINNING", value)
    spec = importlib.util.spec_from_file_location(
        "cembedding._spinning_probe",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cembedding", "server.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spinning_is_off_unless_asked_for(monkeypatch):
    assert _fresh_server_module(monkeypatch, None).ONNX_ALLOW_SPINNING == "0"
    assert _fresh_server_module(monkeypatch, "1").ONNX_ALLOW_SPINNING == "1"


@pytest.mark.parametrize("value", ["yes", "true", "2", ""])
def test_spinning_rejects_anything_but_0_or_1(monkeypatch, value):
    with pytest.raises(ValueError, match="ONNX_ALLOW_SPINNING"):
        _fresh_server_module(monkeypatch, value)


@pytest.mark.parametrize("value", ["0", "1"])
def test_spinning_reaches_the_session_options(monkeypatch, value):
    monkeypatch.setattr(server, "ONNX_ALLOW_SPINNING", value)
    options = server._ort_session_options()
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == value


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


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_DIR, "model.onnx")),
    reason="jina-v5-nano model assets not downloaded (local-only test)",
)
def test_without_spinning_the_runtime_leaves_the_cpu_idle_after_a_run(monkeypatch):
    """The point of the default: once a run returns, the cores are the caller's again.

    The control run with spinning on shows the measurement can see the
    difference on this machine; where it cannot (one core), the test skips
    rather than pass on a measurement that could not have failed.
    """
    import time

    inputs = {
        "input_ids": np.ones((1, 24), dtype=np.int64),
        "attention_mask": np.ones((1, 24), dtype=np.int64),
    }

    def cpu_after_run(spinning):
        monkeypatch.setattr(server, "ONNX_ALLOW_SPINNING", spinning)
        session = server._create_ort_session(os.path.join(MODEL_DIR, "model.onnx"), ["CPUExecutionProvider"])
        session.run(None, inputs)
        time.sleep(1.0)
        out = session.run(None, inputs)[1]
        before = time.process_time()
        time.sleep(0.3)
        return time.process_time() - before, out

    spinning_cpu, spinning_out = cpu_after_run("1")
    if spinning_cpu < 0.15:
        pytest.skip(f"spinning not observable here ({spinning_cpu:.2f} CPU-s in 0.3 s)")
    idle_cpu, idle_out = cpu_after_run("0")
    assert idle_cpu < 0.05, idle_cpu
    np.testing.assert_array_equal(idle_out, spinning_out)  # threads wait differently; the arithmetic is the same
