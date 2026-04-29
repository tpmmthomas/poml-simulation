"""Tests for the miner process — specifically the mid-proof interruption path.

These tests construct a `MinerProcess` but do NOT start it; they exercise
`_prove_interruptible` directly against a fake stand-in for the real
(expensive) `run_inference_and_prove`. Fake proof functions are monkeypatched
onto the `poml_sim.zkp` module in the parent process and — on Linux with the
default fork start method — inherited by the forked proof-worker subprocess.
"""

from __future__ import annotations

import multiprocessing
import sys
import time

import pytest

from poml_sim.miner import MinerProcess
from poml_sim.network import MessageType, NetworkMessage
from poml_sim.types import Block, BlockHeader


# These tests rely on `multiprocessing.get_context()` defaulting to "fork", so
# monkeypatches on `poml_sim.zkp.run_inference_and_prove` propagate from the
# test parent into the forked proof-worker child. On platforms where spawn is
# the default (macOS recent Pythons, Windows), the child would re-import the
# module fresh and bypass the monkeypatch.
_REQUIRES_FORK = pytest.mark.skipif(
    multiprocessing.get_start_method(allow_none=True) not in (None, "fork")
    or sys.platform not in ("linux",),
    reason="interruption tests require the fork start method",
)


def _make_miner(inbox=None, stop_event=None) -> MinerProcess:
    """Build a MinerProcess without starting it — we only drive methods directly."""
    return MinerProcess(
        miner_id=0,
        miner_pk=b"\x01" * 32,
        miner_sk=b"\x02" * 32,
        miner_vrf_vk=b"\x03" * 32,
        miner_vrf_sk=b"\x04" * 32,
        miner_enc_vrf_vk=b"\x05" * 32,
        miner_enc_vrf_sk=b"\x06" * 32,
        mempool=None,
        inbox=inbox if inbox is not None else multiprocessing.Queue(),
        coordinator_queue=multiprocessing.Queue(),
        stop_event=stop_event if stop_event is not None else multiprocessing.Event(),
        config={
            "ezkl_artifacts_dir": "model/",
            "difficulty_int": 0,
            "max_queries_per_block": 1,
            "input_shape": [1, 2, 8, 8],
            "diffusion_steps": 1,
        },
    )


def _make_fake_block(height: int) -> Block:
    header = BlockHeader(
        block_height=height,
        prev_hash=b"\x00" * 32,
        miner_pk=b"\x00" * 32,
        timestamp=0.0,
        difficulty=0,
        lottery_hash=b"\x00" * 32,
    )
    return Block(header=header)


# ---------------------------------------------------------------------------
# Module-level fakes. Defined at module scope (rather than inside tests) so
# they survive across fork cleanly and, in principle, also across spawn —
# spawn children can re-import this test module and find the same fakes.
# ---------------------------------------------------------------------------


def _slow_fake(conditioning, noise, artifacts_dir, input_shape):
    """Stand-in for a multi-minute ezkl proof; we want to abort long before it returns."""
    time.sleep(60)
    return [0.0], b"fake"


def _fast_fake(conditioning, noise, artifacts_dir, input_shape):
    """Stand-in for an instant proof — exercises the success path."""
    return [1.5, 2.5], b"proof_bytes"


def _raising_fake(conditioning, noise, artifacts_dir, input_shape):
    """Stand-in for a proof that fails — exercises the worker-error path."""
    raise RuntimeError("boom")


@_REQUIRES_FORK
@pytest.mark.timeout(30)
def test_prove_interruptible_aborts_on_new_block(monkeypatch):
    """A NEW_BLOCK arriving mid-proof aborts the subprocess within seconds.

    This is a regression test for the feature: without interruption, a miner
    would keep grinding through a ~60s proof even after losing the race.
    """
    from poml_sim import zkp as zkp_module

    monkeypatch.setattr(zkp_module, "run_inference_and_prove", _slow_fake)

    miner = _make_miner()

    # Pre-queue a new-block message so the first poll iteration inside
    # _prove_interruptible sees it and terminates the worker immediately.
    miner.inbox.put(
        NetworkMessage(
            msg_type=MessageType.NEW_BLOCK,
            sender_id=-1,
            payload=_make_fake_block(height=1),
        )
    )

    t0 = time.time()
    result = miner._prove_interruptible(
        conditioning=[0.0] * 64,
        noise=[0.0] * 64,
        artifacts_dir="model/",
        input_shape=[1, 2, 8, 8],
    )
    elapsed = time.time() - t0

    assert result is None, "expected None when aborting"
    # 5s is a generous bound: polling is 100ms and the fake proof is a
    # pure-Python sleep that dies instantly on SIGTERM.
    assert elapsed < 5.0, f"abort took too long: {elapsed:.2f}s"
    assert miner.local_height == 1, "local tip should have advanced to the new block"


@_REQUIRES_FORK
@pytest.mark.timeout(30)
def test_prove_interruptible_aborts_on_stop(monkeypatch):
    """A STOP message mid-proof tears down the subprocess and returns None."""
    from poml_sim import zkp as zkp_module

    monkeypatch.setattr(zkp_module, "run_inference_and_prove", _slow_fake)

    miner = _make_miner()

    miner.inbox.put(
        NetworkMessage(
            msg_type=MessageType.STOP,
            sender_id=-1,
            payload=None,
        )
    )

    t0 = time.time()
    result = miner._prove_interruptible(
        conditioning=[0.0] * 64,
        noise=[0.0] * 64,
        artifacts_dir="model/",
        input_shape=[1, 2, 8, 8],
    )
    elapsed = time.time() - t0

    assert result is None
    assert elapsed < 5.0, f"abort took too long: {elapsed:.2f}s"
    assert miner.stop_event.is_set(), "STOP should have set the stop_event"


@_REQUIRES_FORK
@pytest.mark.timeout(30)
def test_prove_interruptible_returns_proof_on_success(monkeypatch):
    """On the happy path, the helper returns exactly what the fake produced."""
    from poml_sim import zkp as zkp_module

    monkeypatch.setattr(zkp_module, "run_inference_and_prove", _fast_fake)

    miner = _make_miner()

    result = miner._prove_interruptible(
        conditioning=[0.0] * 64,
        noise=[0.0] * 64,
        artifacts_dir="model/",
        input_shape=[1, 2, 8, 8],
    )

    assert result is not None
    output_values, proof_bytes = result
    assert output_values == [1.5, 2.5]
    assert proof_bytes == b"proof_bytes"


@_REQUIRES_FORK
@pytest.mark.timeout(30)
def test_prove_interruptible_handles_worker_error(monkeypatch):
    """If the proof worker raises, the helper returns None cleanly."""
    from poml_sim import zkp as zkp_module

    monkeypatch.setattr(zkp_module, "run_inference_and_prove", _raising_fake)

    miner = _make_miner()

    result = miner._prove_interruptible(
        conditioning=[0.0] * 64,
        noise=[0.0] * 64,
        artifacts_dir="model/",
        input_shape=[1, 2, 8, 8],
    )

    assert result is None


@_REQUIRES_FORK
@pytest.mark.timeout(30)
def test_prove_interruptible_ignores_stale_block(monkeypatch):
    """A NEW_BLOCK whose height is <= our local height must NOT trigger abort.

    Covers the case where a late-arriving broadcast of an already-known block
    would otherwise spuriously cancel an in-flight proof.
    """
    from poml_sim import zkp as zkp_module

    monkeypatch.setattr(zkp_module, "run_inference_and_prove", _fast_fake)

    miner = _make_miner()
    # Pretend our local tip is already at height 5. A height-3 broadcast is stale.
    miner.local_height = 5

    miner.inbox.put(
        NetworkMessage(
            msg_type=MessageType.NEW_BLOCK,
            sender_id=-1,
            payload=_make_fake_block(height=3),
        )
    )

    result = miner._prove_interruptible(
        conditioning=[0.0] * 64,
        noise=[0.0] * 64,
        artifacts_dir="model/",
        input_shape=[1, 2, 8, 8],
    )

    assert result is not None, "stale block must not abort the proof"
    assert miner.local_height == 5, "local_height must not move backwards"
