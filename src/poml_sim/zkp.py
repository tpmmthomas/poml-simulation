"""EZKL wrapper: prove() and verify() for PoML inference proofs."""

import asyncio
import json
import os
import tempfile

import ezkl
import numpy as np


def generate_proof(
    input_data: list[float],
    artifacts_dir: str = "model/",
    input_shape: list[int] | None = None,
) -> bytes:
    """Generate a ZK proof for the given input data.

    Returns the proof bytes.
    """
    compiled_path = os.path.join(artifacts_dir, "network.ezkl")
    pk_path = os.path.join(artifacts_dir, "pk.key")
    srs_path = os.path.join(artifacts_dir, "kzg.srs")
    settings_path = os.path.join(artifacts_dir, "settings.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_json_path = os.path.join(tmpdir, "input.json")
        witness_path = os.path.join(tmpdir, "witness.json")
        proof_path = os.path.join(tmpdir, "proof.json")

        # Format input for EZKL
        data = {"input_data": [input_data]}
        with open(input_json_path, "w") as f:
            json.dump(data, f)

        # Generate witness
        asyncio.run(
            ezkl.gen_witness(input_json_path, compiled_path, witness_path)
        )

        # Generate proof
        res = ezkl.prove(
            witness_path,
            compiled_path,
            pk_path,
            proof_path,
            srs_path,
            "single",
        )
        assert res, "ezkl.prove failed"

        with open(proof_path, "rb") as f:
            proof_bytes = f.read()

    return proof_bytes


def verify_proof(
    proof_bytes: bytes,
    artifacts_dir: str = "model/",
) -> bool:
    """Verify a ZK proof. Returns True if valid."""
    vk_path = os.path.join(artifacts_dir, "vk.key")
    settings_path = os.path.join(artifacts_dir, "settings.json")
    srs_path = os.path.join(artifacts_dir, "kzg.srs")

    with tempfile.TemporaryDirectory() as tmpdir:
        proof_path = os.path.join(tmpdir, "proof.json")
        with open(proof_path, "wb") as f:
            f.write(proof_bytes)

        result = ezkl.verify(proof_path, settings_path, vk_path, srs_path)

    return bool(result)


def run_inference_and_prove(
    conditioning: list[float],
    noise: list[float],
    artifacts_dir: str = "model/",
    input_shape: list[int] | None = None,
) -> tuple[list[float], bytes]:
    """Run model inference via EZKL witness generation and generate proof.

    Returns (output_values, proof_bytes).
    """
    if input_shape is None:
        input_shape = [1, 2, 8, 8]

    # Interleave noise (channel 0) and conditioning (channel 1) into [1, 2, 8, 8]
    spatial = input_shape[2] * input_shape[3]  # 64
    assert len(noise) >= spatial, f"noise length {len(noise)} < spatial {spatial}"
    assert len(conditioning) >= spatial, f"conditioning length {len(conditioning)} < spatial {spatial}"

    input_flat: list[float] = []
    input_flat.extend(noise[:spatial])
    input_flat.extend(conditioning[:spatial])

    compiled_path = os.path.join(artifacts_dir, "network.ezkl")
    settings_path = os.path.join(artifacts_dir, "settings.json")
    pk_path = os.path.join(artifacts_dir, "pk.key")
    srs_path = os.path.join(artifacts_dir, "kzg.srs")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_json_path = os.path.join(tmpdir, "input.json")
        witness_path = os.path.join(tmpdir, "witness.json")
        proof_path = os.path.join(tmpdir, "proof.json")

        data = {"input_data": [input_flat]}
        with open(input_json_path, "w") as f:
            json.dump(data, f)

        # Generate witness (this runs the model internally)
        asyncio.run(
            ezkl.gen_witness(input_json_path, compiled_path, witness_path)
        )

        # Extract output from witness
        with open(witness_path) as f:
            witness_data = json.load(f)
        output_values = [float(x) for x in witness_data.get("outputs", [[]])[0]]

        # Generate proof
        res = ezkl.prove(
            witness_path,
            compiled_path,
            pk_path,
            proof_path,
            srs_path,
            "single",
        )
        assert res, "ezkl.prove failed"

        with open(proof_path, "rb") as f:
            proof_bytes = f.read()

    return output_values, proof_bytes
