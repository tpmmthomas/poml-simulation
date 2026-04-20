"""Tests for ZKP wrapper.

These are integration tests that require EZKL artifacts to be set up.
Skip if artifacts are not present.
"""

import os

import pytest

ARTIFACTS_DIR = "model/"
REQUIRED_FILES = ["network.ezkl", "pk.key", "vk.key", "kzg.srs", "settings.json"]


def artifacts_available() -> bool:
    return all(os.path.exists(os.path.join(ARTIFACTS_DIR, f)) for f in REQUIRED_FILES)


@pytest.mark.skipif(not artifacts_available(), reason="EZKL artifacts not set up")
class TestZKP:
    def test_prove_and_verify(self):
        from poml_sim.zkp import run_inference_and_prove, verify_proof

        conditioning = [0.5] * 64
        noise = [0.1] * 64

        output, proof_bytes = run_inference_and_prove(
            conditioning=conditioning,
            noise=noise,
            artifacts_dir=ARTIFACTS_DIR,
            input_shape=[1, 2, 8, 8],
        )

        assert len(output) > 0
        assert len(proof_bytes) > 0

        # Verify the proof
        valid = verify_proof(proof_bytes, ARTIFACTS_DIR)
        assert valid is True

    def test_different_inputs_different_outputs(self):
        from poml_sim.zkp import run_inference_and_prove

        out1, _ = run_inference_and_prove(
            conditioning=[0.5] * 64,
            noise=[0.1] * 64,
            artifacts_dir=ARTIFACTS_DIR,
        )
        out2, _ = run_inference_and_prove(
            conditioning=[-0.5] * 64,
            noise=[-0.1] * 64,
            artifacts_dir=ARTIFACTS_DIR,
        )
        assert out1 != out2
