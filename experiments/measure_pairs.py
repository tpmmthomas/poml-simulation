"""Build fresh proof-verified timing measurements for the two paper experiments."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from poml_sim.cli import parser, make_backend
from poml_sim.crypto import canonical, sha256
from poml_sim.system import PoMLSystem


def main(argv=None):
    """Measure online inference/proving only, excluding setup and verification."""
    argument_parser = parser()
    argument_parser.description = __doc__
    argument_parser.set_defaults(
        backend="gpt2", queries=32, output=Path("experiments/results/measurements")
    )
    argument_parser.add_argument("--replicates", type=int, default=2)
    args = argument_parser.parse_args(argv)
    if args.backend == "smoke":
        argument_parser.error("measurement banks require real GPT-2 or EZKL proofs")
    if min(args.queries, args.replicates) < 1:
        argument_parser.error("positive queries and replicates required")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "measurements.json"
    if destination.exists() or (args.output / "completed.jsonl").exists():
        argument_parser.error("choose a fresh output directory")
    backend = make_backend(args)
    try:
        system = PoMLSystem(backend, miners=1, seed=args.seed)
        if args.prompts:
            prompts = args.prompts.read_text().splitlines()
            if len(prompts) < args.queries:
                raise ValueError("prompt file has fewer prompts than requested measurements")
        elif args.backend == "gpt2":
            from poml_sim.benchmark_data import benchmark_examples

            examples = benchmark_examples(
                backend.tokenizer, "wikitext2", count=args.queries, seed=args.seed
            )
            prompts = [
                backend.tokenizer.decode(row["context"][: (8, 16, 24, 32)[i % 4]])
                for i, row in enumerate(examples)
            ]
        else:
            prompts = [f"diffusion conditioning {i}" for i in range(args.queries)]
        rows = []
        for i, prompt in enumerate(prompts[: args.queries]):
            query = system.submit_query(prompt, max_output=args.max_output)
            for replicate in range(args.replicates):
                response, duration = system.execute(
                    query, 0, sha256(canonical(["measure", args.seed, i, replicate]))
                )
                row = {
                    **system.executions[-1],
                    "query_id": query.qid.hex(),
                    "replicate": replicate,
                    "verified": True,
                    "fresh_inference": True,
                    "fresh_proof": True,
                    "inference_proof_seconds": duration,
                }
                rows.append(row)
                with (args.output / "completed.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(f"Completed pair {len(rows)}/{args.queries * args.replicates}", flush=True)
        destination.write_text(
            json.dumps(
                {
                    "schema": "poml-measurements-1",
                    "backend": backend.name,
                    "model": backend.identity,
                    "seed": args.seed,
                    "duration_scope": "online inference plus proving; excludes setup and proof verification",
                    "records": rows,
                },
                indent=2,
            )
            + "\n"
        )
        return 0
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
