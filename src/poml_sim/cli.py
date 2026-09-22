"""Run the standalone fresh-work PoML protocol and save an auditable report."""

import argparse
import json
from pathlib import Path
from .backends import GPT2DeepProveBackend, EZKLDiffusionBackend, SmokeBackend
from .lottery import LIMIT
from .simulation import run_chain
from .system import PoMLSystem


def parser():
    """Declare the supported simulator backends and explicit resource paths."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--backend", choices=("gpt2", "diffusion", "smoke"), default="smoke")
    result.add_argument("--blocks", type=int, default=3)
    result.add_argument("--miners", type=int, default=4)
    result.add_argument("--queries", type=int, default=64)
    result.add_argument("--max-output", type=int, default=16)
    result.add_argument("--difficulty", type=lambda value: int(value, 0), default=LIMIT // 100)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument(
        "--binary", type=Path, default=Path(".scratch/deep-prove/target/release/poml-prover")
    )
    result.add_argument("--setup-directory", type=Path)
    result.add_argument("--artifacts", type=Path, default=Path("model"))
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--weights", type=Path)
    result.add_argument("--prompts", type=Path, help="newline-separated prompt file")
    result.add_argument("--output", type=Path, default=Path("experiments/results/simulator"))
    return result


def make_backend(args):
    """Create a real prover backend or an explicitly named test double."""
    if args.backend == "smoke":
        return SmokeBackend()
    if args.backend == "diffusion":
        return EZKLDiffusionBackend(args.artifacts)
    weights = None
    if args.weights:
        from .backends import load_schedule

        fitted = json.loads(args.weights.read_text())
        if fitted.get("schedule_sha256") != load_schedule()["sha256"]:
            raise ValueError("fitted weights do not match the public operation schedule")
        weights = fitted["weights"]
    return GPT2DeepProveBackend(
        args.binary,
        args.output / "proofs",
        device=args.device,
        weights=weights,
        setup_directory=args.setup_directory,
    )


def main(argv=None):
    """Execute the chosen backend; incomplete chains return a nonzero status."""
    args = parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "run.json"
    if report_path.exists():
        raise ValueError("output already contains run.json; choose a fresh directory")
    backend = make_backend(args)
    try:
        system = PoMLSystem(backend, miners=args.miners, difficulty=args.difficulty, seed=args.seed)
        prompts = args.prompts.read_text().splitlines() if args.prompts else None
        rows = run_chain(
            system,
            blocks=args.blocks,
            pool_size=args.queries,
            seed=args.seed,
            prompts=prompts,
            max_output=args.max_output,
        )
        report = {
            "schema": "poml-sim-run-1",
            "backend": backend.name,
            "model": backend.identity,
            "seed": args.seed,
            "difficulty": str(args.difficulty),
            "miners": args.miners,
            "requested_blocks": args.blocks,
            "adopted_blocks": system.height,
            "blocks": rows,
            "burned": system.state.burned,
            "abstractions": [
                "trusted host composes model proof, commitment, VRF, complexity and encryption checks",
                "host sees witnesses and experiment keys; no output-privacy claim",
                "serial physical execution with independent virtual miners; zero network delay",
            ],
            "executions": system.executions,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "backend": backend.name,
                    "adopted_blocks": system.height,
                    "report": str(report_path),
                },
                indent=2,
            )
        )
        return 0 if system.height == args.blocks else 1
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
