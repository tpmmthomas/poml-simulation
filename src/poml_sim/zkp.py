"""EZKL wrapper: prove() and verify() for PoML inference proofs.

Follows the public-output EZKL demo flow:
    gen_witness -> prove -> verify

The model is compiled with hashed input/output visibility, so the raw
float outputs are only available via the witness's `pretty_elements`
section; the top-level `outputs` field holds Poseidon-hashed field
elements used as public commitments.
"""

import json
import os
import tempfile

import ezkl


def verify_proof(proof_bytes: bytes, artifacts_dir: str = "model/") -> bool:
    """Verify a ZK proof. Returns True iff valid."""
    vk_path = os.path.join(artifacts_dir, "vk.key")
    settings_path = os.path.join(artifacts_dir, "settings.json")
    # Pass srs_path explicitly so we use the SRS shipped with the repo
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
    spatial = input_shape[2] * input_shape[3]
    assert len(noise) >= spatial, f"noise length {len(noise)} < spatial {spatial}"
    assert (
        len(conditioning) >= spatial
    ), f"conditioning length {len(conditioning)} < spatial {spatial}"
    input_flat = list(noise[:spatial]) + list(conditioning[:spatial])

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
        if rescaled:
            output_values = [float(x) for x in rescaled[0]]
        else:
            output_values = [float(x) for x in witness_data["outputs"][0]]

        res = ezkl.prove(witness_path, compiled_path, pk_path, proof_path, srs_path)
        assert res, "ezkl.prove failed"

        with open(proof_path, "rb") as f:
            proof_bytes = f.read()

    return output_values, proof_bytes
