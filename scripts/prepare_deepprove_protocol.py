"""Apply the pinned PoML graph/composite-verifier patch and build its CUDA worker."""

import argparse
import os
from pathlib import Path
import subprocess

from prepare_deepprove_work import ROOT, DEEP_REV, prepare, apply_once


def main():
    """Build the exact persistent prover used by the live experiment commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep-prove", type=Path, default=ROOT / ".scratch/deep-prove")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    checkout = args.deep_prove.resolve()
    prepare(checkout)
    apply_once(checkout, ROOT / "scripts/deepprove_protocol.patch", DEEP_REV)
    env = os.environ.copy()
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    subprocess.run(
        [
            "cargo",
            "check" if args.check_only else "build",
            "--release",
            "--features",
            "cuda",
            "--bin",
            "poml-prover",
        ],
        cwd=checkout,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
