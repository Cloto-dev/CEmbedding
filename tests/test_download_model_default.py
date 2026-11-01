"""`cembedding-download-model` with no --model fetches bge-m3.

bge-m3 is the recommended model (MIT-licensed); the default used to be MiniLM. The
download functions are replaced, so no test touches the network or the disk.
"""

import sys

import pytest

from cembedding import download_model


@pytest.fixture
def fetched(monkeypatch):
    calls = []
    monkeypatch.setattr(download_model, "download", lambda: calls.append("miniml") or True)
    monkeypatch.setattr(
        download_model, "download_jina_v5_nano", lambda variant="fp32": calls.append("jina-v5-nano") or True
    )
    monkeypatch.setattr(download_model, "download_bge_m3", lambda model_dir="": calls.append("bge-m3") or True)
    return calls


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["cembedding-download-model", *args])
    with pytest.raises(SystemExit) as exit_info:
        download_model.main()
    return exit_info.value.code


def test_no_model_fetches_bge_m3(monkeypatch, fetched):
    assert _run(monkeypatch) == 0
    assert fetched == ["bge-m3"]


@pytest.mark.parametrize("name", ["miniml", "jina-v5-nano", "bge-m3"])
def test_a_named_model_is_still_the_one_fetched(monkeypatch, fetched, name):
    assert _run(monkeypatch, "--model", name) == 0
    assert fetched == [name]
