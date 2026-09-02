"""jina-v5-nano precision selection.

Each precision is a separate .onnx / .onnx_data pair whose external-data file
name is recorded inside the .onnx, so the pair must keep its upstream stem on
disk. These tests pin that mapping and the rejection of unknown variants; they
need no model on disk.
"""

import pytest

from cembedding.download_model import JINA_VARIANT_STEMS, jina_v5_nano_files


def test_fp32_keeps_the_historical_file_names():
    files = jina_v5_nano_files("fp32")
    assert files == {
        "model.onnx": "onnx/model.onnx",
        "model.onnx_data": "onnx/model.onnx_data",
        "tokenizer.json": "tokenizer.json",
    }


@pytest.mark.parametrize("variant", sorted(JINA_VARIANT_STEMS))
def test_each_variant_pairs_the_graph_with_its_own_external_data(variant):
    stem = JINA_VARIANT_STEMS[variant]
    files = jina_v5_nano_files(variant)
    assert files[f"{stem}.onnx"] == f"onnx/{stem}.onnx"
    assert files[f"{stem}.onnx_data"] == f"onnx/{stem}.onnx_data"
    assert files["tokenizer.json"] == "tokenizer.json"


def test_unknown_variant_is_rejected_before_any_download():
    with pytest.raises(ValueError, match="q4"):
        jina_v5_nano_files("q4")


def test_server_rejects_an_unknown_variant_at_import(monkeypatch):
    import importlib

    import cembedding.server as server

    monkeypatch.setenv("EMBEDDING_MODEL_VARIANT", "q4")
    with pytest.raises(ValueError, match="EMBEDDING_MODEL_VARIANT"):
        importlib.reload(server)
    monkeypatch.delenv("EMBEDDING_MODEL_VARIANT")
    importlib.reload(server)  # restore the module for the rest of the session
    assert server.EMBEDDING_MODEL_VARIANT == "fp32"
