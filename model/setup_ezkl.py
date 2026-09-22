"""One-time EZKL setup: gen-settings → calibrate → compile → get-srs → setup.

Produces all artifacts needed for proving and verification:
  - settings.json
  - network.ezkl (compiled circuit)
  - pk.key (proving key)
  - vk.key (verification key)
  - kzg.srs (structured reference string)
"""

import asyncio
import json
import os
import sys

import ezkl
import numpy as np


async def setup_ezkl(
    model_path: str = "model/network.onnx",
    artifacts_dir: str = "model/",
) -> None:
    settings_path = os.path.join(artifacts_dir, "settings.json")
    compiled_path = os.path.join(artifacts_dir, "network.ezkl")
    srs_path = os.path.join(artifacts_dir, "kzg.srs")
    vk_path = os.path.join(artifacts_dir, "vk.key")
    pk_path = os.path.join(artifacts_dir, "pk.key")
    calibration_path = os.path.join(artifacts_dir, "calibration.json")

    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Generate settings
    #   - input_visibility="hashed/public": inputs are committed via Poseidon hash
    #     (the hash is public, actual inputs are private to the prover).
    #     These are model commitments, not the complete PoML query commitment.
    #   - output_visibility="hashed/public": outputs are committed; encryption
    #     and its binding to this output remain trusted-host checks.
    #   - param_visibility="fixed": model weights are baked into the circuit,
    #     acting as an implicit commitment to θ (paper's c_θ).
    print("[EZKL] Generating settings...")
    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "hashed/public"
    py_run_args.output_visibility = "hashed/public"
    py_run_args.param_visibility = "fixed"
    res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
    assert res, "gen_settings failed"

    # 2. Generate calibration data
    print("[EZKL] Generating calibration data...")
    cal_data = {
        "input_data": [
            np.random.default_rng(42 + i).standard_normal((1, 2, 8, 8)).reshape(-1).tolist()
            for i in range(20)
        ]
    }
    with open(calibration_path, "w") as f:
        json.dump(cal_data, f)

    # 3. Calibrate settings
    print("[EZKL] Calibrating settings...")
    res = ezkl.calibrate_settings(calibration_path, model_path, settings_path, "resources")
    if asyncio.iscoroutine(res):
        res = await res

    # 4. Compile circuit
    print("[EZKL] Compiling circuit...")
    res = ezkl.compile_circuit(model_path, compiled_path, settings_path)
    assert res, "compile_circuit failed"

    # 5. Get SRS
    print("[EZKL] Fetching SRS...")
    res = ezkl.get_srs(settings_path=settings_path, srs_path=srs_path)
    if asyncio.iscoroutine(res) or asyncio.isfuture(res):
        res = await res

    # 6. Setup (generate proving & verification keys)
    print("[EZKL] Running setup (generating pk, vk)...")
    res = ezkl.setup(compiled_path, vk_path, pk_path, srs_path)
    if asyncio.iscoroutine(res) or asyncio.isfuture(res):
        res = await res
    assert res, "setup failed"

    print("[EZKL] Setup complete. Artifacts:")
    for name in [settings_path, compiled_path, srs_path, vk_path, pk_path]:
        size = os.path.getsize(name) if os.path.exists(name) else 0
        print(f"  {name}: {size / 1024:.1f} KB")


def main() -> None:
    model_path = sys.argv[1] if len(sys.argv) > 1 else "model/network.onnx"
    artifacts_dir = sys.argv[2] if len(sys.argv) > 2 else "model/"
    asyncio.run(setup_ezkl(model_path, artifacts_dir))


if __name__ == "__main__":
    main()
