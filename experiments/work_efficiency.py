"""Measure fresh GPT-2 responses and the time fraction producing useful work."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from poml_sim.benchmark_data import benchmark_examples
from poml_sim.cli import make_backend, parser
from poml_sim.crypto import canonical, sha256
from poml_sim.system import PoMLSystem
from poml_sim.work_efficiency import measure_response, summarize_records


def digest_file(path):
    """Hash a large file without retaining it in memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    """Save warmups, all fresh artifacts, raw timings, and reproducible summaries."""
    argument_parser = parser()
    argument_parser.description = __doc__
    argument_parser.set_defaults(
        backend="gpt2",
        queries=50,
        max_output=1,
        seed=20260925,
        output=Path("experiments/results/work-efficiency"),
        difficulty=(1 << 256) // 10**12,
    )
    argument_parser.add_argument("--warmup", type=int, default=2)
    argument_parser.add_argument("--bootstrap", type=int, default=10000)
    argument_parser.add_argument("--summarize", type=Path, help="reanalyze an existing campaign")
    args = argument_parser.parse_args(argv)
    if args.summarize:
        manifest = json.loads((args.summarize / "manifest.json").read_text())
        records = [
            json.loads(line)
            for line in (args.summarize / "measurements.jsonl").read_text().splitlines()
        ]
        if manifest.get("status") != "complete" or len(records) != manifest["queries"]:
            argument_parser.error("campaign is incomplete; no complete summary can be produced")
        report = summarize_records(records, bootstrap=args.bootstrap, seed=args.seed)
        (args.summarize / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["ratios"], indent=2))
        return 0
    if args.backend != "gpt2" or args.queries < 1 or args.warmup < 0 or args.bootstrap < 1:
        argument_parser.error("GPT-2, positive queries/bootstrap and nonnegative warmup required")
    if args.prompts:
        argument_parser.error("this campaign uses saved WikiText-2 token manifests")
    if args.output.exists() and any(args.output.iterdir()):
        argument_parser.error("choose a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "poml-work-efficiency-1",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "queries": args.queries,
        "warmup": args.warmup,
        "seed": args.seed,
        "max_output": args.max_output,
        "device": args.device,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "difficulty": str(args.difficulty),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv",
            ],
            text=True,
        ),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(p): digest_file(p) for p in sorted(Path("src/poml_sim").glob("*.py"))
        },
        "script_sha256": digest_file(__file__),
        "binary_sha256": digest_file(args.binary),
        "scope": "single-pair response production + one model verification + one protocol validation + one lottery",
        "useful": "worker inference (includes witness reconstruction) + proving + one model verification",
        "auxiliary": "all remaining online elapsed time, including serialization, artifact I/O and orchestration",
        "excluded": [
            "setup",
            "query submission",
            "network",
            "replicated validators",
            "duplicate queries",
        ],
        "abstractions": [
            "model-only proof; full private relation uses host receipt",
            "sign-then-hash VRF substitute",
            "single-ciphertext lottery prefix",
        ],
        "environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    backend = make_backend(args)
    try:
        manifest.update(
            model=backend.identity,
            prover=backend.prover.ready,
            initialization_seconds=time.perf_counter() - started,
        )
        examples = benchmark_examples(
            backend.tokenizer, "wikitext2", count=args.queries + args.warmup, seed=args.seed
        )
        prompts = []
        for i, example in enumerate(examples):
            # Warmups do not change the measured prompt-length distribution.
            index = i if i < args.warmup else i - args.warmup
            tokens = example["context"][: (8, 16, 24, 32)[index % 4]]
            prompts.append({**example, "tokens": tokens, "warmup": i < args.warmup})
        (args.output / "prompts.json").write_text(json.dumps(prompts, indent=2) + "\n")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        system = PoMLSystem(backend, miners=1, seed=args.seed, difficulty=args.difficulty)
        rows = []
        for i, prompt in enumerate(prompts):
            text = backend.tokenizer.decode(prompt["tokens"])
            query = system.submit_query(text, max_output=args.max_output)
            if list(query.inputs) != prompt["tokens"]:
                raise ValueError("tokenization roundtrip changed the saved prompt")
            binding = sha256(canonical(["work-efficiency", args.seed, i]))
            row = measure_response(system, query, binding)
            row.update(
                index=i,
                warmup=prompt["warmup"],
                source_id=prompt["id"],
                fresh_inference=True,
                fresh_proof=True,
            )
            filename = "warmup.jsonl" if row["warmup"] else "measurements.jsonl"
            with (args.output / filename).open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if not row["warmup"]:
                rows.append(row)
            print(
                f"{'Warmup' if row['warmup'] else 'Measured'} {i + 1}/{len(prompts)} "
                f"N={len(query.inputs)} K={row['output_length']} "
                f"elapsed={row['elapsed_seconds']:.2f}s useful={row['useful_seconds']:.2f}s",
                flush=True,
            )
        report = summarize_records(rows, bootstrap=args.bootstrap, seed=args.seed)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(report["ratios"], indent=2), flush=True)
        return 0
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
