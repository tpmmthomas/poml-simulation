"""Proof adapters must reject failed verification and altered artifacts."""

from types import SimpleNamespace
import json
import sys
import pytest
from poml_sim.backends import GPT2DeepProveBackend, EZKLDiffusionBackend, load_schedule
from poml_sim.cli import make_backend, parser
from poml_sim.crypto import sha256


def fake_gpt(tmp_path, verified=True, digest=None):
    backend = object.__new__(GPT2DeepProveBackend)
    backend.directory, backend.counter, backend.alpha = tmp_path, 0, 0.05
    backend.schedule, backend.weights = load_schedule(), None

    def run(request):
        path = tmp_path / request["request_id"]
        path.mkdir()
        (path / "proof.bin").write_bytes(b"proof")
        (path / "logits.i64").write_bytes(b"logits")
        return dict(
            verified=verified,
            proof_sha256=digest or sha256(b"proof").hex(),
            output_tokens=[10],
            output_length=1,
            logits_scale=8,
            logits_shape=[1, 10],
            inference_seconds=2,
            proof_seconds=3,
        )

    backend.prover = SimpleNamespace(ready={"embedding_std": 0.02}, run=run)
    return backend


@pytest.mark.parametrize("verified,digest", [(False, None), (True, "wrong")])
def test_gpt_rejects_failed_verification_or_changed_proof(tmp_path, verified, digest):
    backend = fake_gpt(tmp_path, verified, digest)
    with pytest.raises(RuntimeError, match="verification or digest"):
        backend.run((1, 2), (b"x" * 32,) * 3, 1)


def test_gpt_executes_again_and_preserves_each_fresh_proof(tmp_path):
    backend = fake_gpt(tmp_path)
    result = backend.run((1, 2), (b"x" * 32,) * 3, 1)
    backend.run((1, 2), (b"y" * 32,) * 3, 1)
    assert result.duration == 5 and result.output_length == 1
    assert (tmp_path / "pair-000000/proof.bin").exists()
    assert (tmp_path / "pair-000001/proof.bin").exists()
    with pytest.raises(ValueError, match="randomness"):
        backend.run((1, 2), (), 1)


def test_ezkl_rejects_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="setup missing"):
        EZKLDiffusionBackend(tmp_path)


def test_ezkl_adapter_rejects_unverified_proof(monkeypatch, tmp_path):
    backend = object.__new__(EZKLDiffusionBackend)
    backend.artifacts = tmp_path
    monkeypatch.setitem(
        sys.modules,
        "poml_sim.zkp",
        SimpleNamespace(
            run_inference_and_prove=lambda *args: ([1.0] * 64, b"proof"),
            verify_proof=lambda *args: False,
        ),
    )
    with pytest.raises(RuntimeError, match="verification failed"):
        backend.run((1.0,) * 64, (b"x" * 32,), 1)


def test_weights_from_different_operation_schedule_are_rejected(tmp_path):
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"schedule_sha256": "wrong", "weights": {}}))
    args = parser().parse_args(["--backend", "gpt2", "--weights", str(path)])
    with pytest.raises(ValueError, match="schedule"):
        make_backend(args)
