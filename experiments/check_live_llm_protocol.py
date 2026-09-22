#!/usr/bin/env python3
"""Reverify a real archived composite proof and reject altered public bindings."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    """Check real proof files without running another GPT-2 inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(".scratch/deep-prove/target/release/poml-prover"),
    )
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.device

    def verify(directory):
        return subprocess.run(
            [
                str(args.binary.resolve()),
                "--verify",
                str(directory.resolve()),
                "--setup-directory",
                str(args.setup.resolve()),
            ],
            env=env,
            text=True,
            capture_output=True,
        )

    valid = verify(args.proof)
    if valid.returncode:
        raise RuntimeError(f"original proof failed: {valid.stderr}")
    report = {"original_verified": True, "rejected": {}}
    original = json.loads((args.proof / "request.json").read_text())
    changes = {
        "prompt": lambda r: r["prompt_tokens"].__setitem__(
            0, (r["prompt_tokens"][0] + 1) % 50257
        ),
        "noise": lambda r: r["noise"][0].__setitem__(0, r["noise"][0][0] + 0.1),
        "uniform_lower": lambda r: r["uniforms"].__setitem__(0, 0),
        "uniform_upper": lambda r: r["uniforms"].__setitem__(0, (1 << 53) - 1),
    }
    for name, change in changes.items():
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        for filename in ("proof.bin", "statement.bin.zst"):
            shutil.copy2(args.proof / filename, directory / filename)
        request = json.loads(json.dumps(original))
        change(request)
        (directory / "request.json").write_text(json.dumps(request))
        result = verify(directory)
        report["rejected"][name] = result.returncode != 0
        (directory / "verifier.log").write_text(result.stdout + result.stderr)
    if not report["rejected"]["prompt"] or not report["rejected"]["noise"]:
        raise RuntimeError("composite verifier accepted inconsistent prompt/noise")
    # Different uniforms can legitimately fall in the same token interval.
    # For this nondegenerate smoke proof, at least one endpoint must differ.
    if not any(report["rejected"][name] for name in ("uniform_lower", "uniform_upper")):
        raise RuntimeError("proof did not provide a nondegenerate sampling test")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
