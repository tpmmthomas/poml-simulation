"""EZKL wrapper: prove() and verify() for PoML inference proofs.

Follows the public-output EZKL demo flow:
    gen_witness -> prove -> verify

The model is compiled with hashed input/output visibility, so the raw
float outputs are only available via the witness's `pretty_elements`
section; the top-level `outputs` field holds Poseidon-hashed field
elements used as public commitments.
"""

import json
import math
import os
import tempfile

import ezkl


def verify_proof(proof_bytes: bytes, artifacts_dir: str = "model/") -> bool:
    """Verify a ZK proof. Returns True iff valid."""
    vk_path = os.path.join(artifacts_dir, "vk.key")
    settings_path = os.path.join(artifacts_dir, "settings.json")
    # Pass srs_path explicitly so we use the SRS in the setup directory
    # instead of relying on ~/.ezkl/srs/ (which the tutorial populates via
    # ezkl.get_srs()).
    srs_path = os.path.join(artifacts_dir, "kzg.srs")

    with tempfile.TemporaryDirectory() as tmpdir:
        proof_path = os.path.join(tmpdir, "proof.json")
        with open(proof_path, "wb") as f:
            f.write(proof_bytes)
        return bool(ezkl.verify(proof_path, settings_path, vk_path, srs_path))


def run_inference_and_prove(
    conditioning: list[float],
    noise: list[float],
    artifacts_dir: str = "model/",
    input_shape: list[int] | None = None,
) -> tuple[list[float], bytes]:
    """Run model inference via EZKL witness generation and generate a proof.

    Returns (output_values, proof_bytes).
    """
    if input_shape is None:
        input_shape = [1, 2, 8, 8]

    # Pack noise (channel 0) and conditioning (channel 1) into the flat
    # [1, 2, 8, 8] tensor expected by the circuit.
    if input_shape != [1, 2, 8, 8] or len(noise) != 64 or len(conditioning) != 64:
        raise ValueError("tiny U-Net requires exactly two finite 8x8 channels")
    input_flat = list(noise) + list(conditioning)
    if not all(math.isfinite(value) for value in input_flat):
        raise ValueError("tiny U-Net inputs must be finite")

    compiled_path = os.path.join(artifacts_dir, "network.ezkl")
    pk_path = os.path.join(artifacts_dir, "pk.key")
    srs_path = os.path.join(artifacts_dir, "kzg.srs")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_json_path = os.path.join(tmpdir, "input.json")
        witness_path = os.path.join(tmpdir, "witness.json")
        proof_path = os.path.join(tmpdir, "proof.json")

        with open(input_json_path, "w") as f:
            json.dump({"input_data": [input_flat]}, f)

        ezkl.gen_witness(input_json_path, compiled_path, witness_path)

        with open(witness_path) as f:
            witness_data = json.load(f)
        # Under hashed output visibility, top-level "outputs" are Poseidon
        # field elements; real float outputs live in pretty_elements.
        rescaled = (witness_data.get("pretty_elements") or {}).get("rescaled_outputs")
        if not rescaled or len(rescaled[0]) != 64:
            raise RuntimeError("EZKL witness is missing its rescaled denoiser output")
        output_values = [float(x) for x in rescaled[0]]
        if not all(math.isfinite(value) for value in output_values):
            raise RuntimeError("EZKL returned nonfinite denoiser output")

        res = ezkl.prove(witness_path, compiled_path, pk_path, proof_path, srs_path)
        if not res:
            raise RuntimeError("ezkl.prove failed")

        with open(proof_path, "rb") as f:
            proof_bytes = f.read()

    return output_values, proof_bytes
