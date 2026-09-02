"""Coalescing of concurrent embedding requests into one model run.

A local provider owns one session, so requests are serialized either way; the
runner merges the ones that arrive while a run is in flight so they share the
next run instead of each paying for one. These tests pin the two things that
make it safe to do: the merged run must hand every caller exactly its own
rows, and a failing run must fail only its own callers without stopping the
requests queued behind it.

The parity test at the end needs the jina-v5-nano assets on disk and, like the
padding tests, self-skips where the model was never downloaded.
"""

import asyncio
import os
import threading

import numpy as np
import pytest

from cembedding.server import _BatchRunner

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "models",
    "jina-embeddings-v5-text-nano",
)


class FakeModel:
    """Stands in for ``_embed_sync``: records the batch of every run.

    ``block_first`` lets a test hold the first run open, which is the only way
    to make later requests arrive while a run is in flight.
    """

    def __init__(self, block_first=False):
        self.batches: list[list[str]] = []
        self._gate = threading.Event()
        self._block_first = block_first

    def __call__(self, texts: list[str]) -> list[list[float]]:
        first = not self.batches
        self.batches.append(list(texts))
        if first and self._block_first:
            self._gate.wait(timeout=5)
        # One row per text, carrying the text so callers can be checked
        # against what they asked for.
        return [[float(len(t)), float(hash(t) % 1000)] for t in texts]

    def release(self):
        self._gate.set()

    @property
    def runs(self) -> int:
        return len(self.batches)


async def _within(awaitable, seconds=10):
    """Await something that a broken runner might never resolve.

    A drain loop that died leaves its callers' futures pending forever, so
    without a bound the test would wedge the whole run instead of failing.
    """
    return await asyncio.wait_for(awaitable, timeout=seconds)


async def test_a_lone_request_runs_immediately_in_a_batch_of_one():
    model = FakeModel()
    runner = _BatchRunner(model, max_batch=64)

    result = await _within(runner.submit(["only text"]))

    assert model.batches == [["only text"]]
    assert result == [[float(len("only text")), float(hash("only text") % 1000)]]


async def test_requests_arriving_during_a_run_share_the_next_run():
    model = FakeModel(block_first=True)
    runner = _BatchRunner(model, max_batch=64)

    tasks = [asyncio.create_task(runner.submit([f"text {i}"])) for i in range(16)]
    # Let the first request reach the executor and block there, so the other
    # 15 pile up behind it.
    await asyncio.sleep(0.05)
    model.release()
    results = await _within(asyncio.gather(*tasks))

    assert model.runs < 16, f"no coalescing happened: {model.batches}"
    assert sum(len(b) for b in model.batches) == 16, "every text must be embedded exactly once"
    for i, result in enumerate(results):
        text = f"text {i}"
        assert result == [[float(len(text)), float(hash(text) % 1000)]], f"caller {i} got another caller's row"


async def test_a_failing_run_fails_only_its_own_callers():
    class FailingOnce:
        def __init__(self):
            self.batches = []

        def __call__(self, texts):
            self.batches.append(list(texts))
            if len(self.batches) == 1:
                raise RuntimeError("session blew up")
            return [[1.0] for _ in texts]

    model = FailingOnce()
    runner = _BatchRunner(model, max_batch=64)

    with pytest.raises(RuntimeError, match="session blew up"):
        await _within(runner.submit(["doomed"]))

    # The drain loop must have survived the failure.
    assert await _within(runner.submit(["later"])) == [[1.0]]
    assert model.batches == [["doomed"], ["later"]]


async def test_max_batch_zero_gives_every_request_its_own_run():
    model = FakeModel(block_first=True)
    runner = _BatchRunner(model, max_batch=0)

    tasks = [asyncio.create_task(runner.submit([f"text {i}"])) for i in range(8)]
    await asyncio.sleep(0.05)
    model.release()
    await _within(asyncio.gather(*tasks))

    assert model.runs == 8
    assert all(len(b) == 1 for b in model.batches)


async def test_a_request_larger_than_the_cap_still_runs_in_one_pass():
    model = FakeModel()
    runner = _BatchRunner(model, max_batch=64)

    texts = [f"text {i}" for i in range(100)]
    result = await _within(runner.submit(texts))

    assert model.batches == [texts], "one caller's texts must never be split across runs"
    assert len(result) == 100


async def test_the_cap_bounds_how_much_is_merged_into_a_run():
    model = FakeModel(block_first=True)
    runner = _BatchRunner(model, max_batch=4)

    tasks = [asyncio.create_task(runner.submit([f"text {i}", f"more {i}"])) for i in range(6)]
    await asyncio.sleep(0.05)
    model.release()
    await _within(asyncio.gather(*tasks))

    assert max(len(b) for b in model.batches) <= 4
    assert sum(len(b) for b in model.batches) == 12


async def test_an_empty_request_needs_no_run():
    model = FakeModel()
    runner = _BatchRunner(model, max_batch=64)

    assert await _within(runner.submit([])) == []
    assert model.runs == 0


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_DIR, "model.onnx")),
    reason="jina-v5-nano model assets not downloaded (local-only test)",
)
async def test_coalesced_requests_match_one_at_a_time_embeddings(monkeypatch):
    """Merging requests must not move a single bit of the output.

    The providers pad each batch to its own longest row and pool under the
    attention mask, so the batch a text is embedded in does not reach its
    vector — this is the end-to-end check of that through the real session.

    The session is pinned to the CPU execution provider because that is the
    part the server controls. An accelerator provider re-plans the graph per
    input shape and returns slightly different last bits for the same text
    depending on the batch it rode in (measured ~7e-7 on Apple CoreML), with
    or without coalescing: the same drift already showed up between a
    one-text request and a many-text one.
    """
    monkeypatch.setenv("ONNX_EP_PREFERENCE", "CPUExecutionProvider")
    from cembedding.server import OnnxJinaV5NanoProvider

    texts = [
        "short query",
        "what did we decide about the deployment schedule last week?",
        "a medium document " + "with some repeated content " * 20,
        "another query entirely",
    ] * 4

    provider = OnnxJinaV5NanoProvider(MODEL_DIR)
    await provider.initialize()
    try:
        alone = [(await _within(provider.embed([t])))[0] for t in texts]
        coalesced = await _within(asyncio.gather(*(asyncio.create_task(provider.embed([t])) for t in texts)), 60)
    finally:
        await provider.shutdown()

    np.testing.assert_array_equal(np.array(alone), np.array([rows[0] for rows in coalesced]))


@pytest.mark.parametrize(
    "provider_name",
    ["OnnxMiniLMProvider", "OnnxBgeM3Provider", "OnnxJinaV5NanoProvider", "MlxBgeM3Provider"],
)
def test_every_local_provider_runs_through_the_cap(provider_name, monkeypatch):
    """The knob has to reach each provider, not just the runner it is passed to."""
    from cembedding import server

    monkeypatch.setattr(server, "EMBEDDING_MAX_BATCH", 7)
    provider = getattr(server, provider_name)("some/model/dir")
    assert provider._runner._max_batch == 7
    assert provider._runner._run == provider._embed_sync


async def test_an_uninitialized_provider_still_refuses_to_embed():
    from cembedding.server import OnnxJinaV5NanoProvider

    with pytest.raises(RuntimeError, match="not initialized"):
        await OnnxJinaV5NanoProvider("some/model/dir").embed(["text"])


def test_a_negative_cap_is_rejected_at_import(monkeypatch):
    """Loaded as a fresh module so the running server's own settings stand."""
    import importlib.util

    monkeypatch.setenv("EMBEDDING_MAX_BATCH", "-1")
    spec = importlib.util.spec_from_file_location(
        "cembedding._batch_cap_probe",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cembedding", "server.py"),
    )
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(ValueError, match="EMBEDDING_MAX_BATCH"):
        spec.loader.exec_module(module)
