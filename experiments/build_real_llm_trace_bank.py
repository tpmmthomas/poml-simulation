#!/usr/bin/env python3
"""Profile benchmark prompts and build held-out GPT-2/CUDA DeepProve traces.

Every evaluated prompt/replicate gets its own verified timing measurement.
Profiling lengths are frozen before evaluation and never enter the replay bank.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from poml_sim.llm_benchmark import (  # noqa: E402
    BENCHMARK_SCHEMA,
    challenge_seed,
    prediction_diagnostics,
    profile_pool,
    wikitext_prompts,
)
from poml_sim.llm_simulation import QuerySpec, complexity_for, load_schedule  # noqa: E402

DEFAULT_SCHEDULE = ROOT / "config/gpt2_reference_schedule.json"
DEFAULT_BINARY = ROOT / ".scratch/deep-prove/target/release/bench-llm"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cuda_environment(device: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    env.setdefault("RUST_LOG", "error")
    if device.startswith("cuda:"):
        index = int(device.split(":")[1])
        visible = env.get("CUDA_VISIBLE_DEVICES")
        env["CUDA_VISIBLE_DEVICES"] = (
            visible.split(",")[index] if visible else str(index)
        )
    return env


def _load_wikitext(cache_file: Path | None) -> tuple[object, dict]:
    """Use the previously downloaded Arrow dataset before any network access."""
    from datasets import Dataset, load_dataset
    from datasets.config import HF_DATASETS_CACHE

    if cache_file is None:
        matches = sorted(
            (Path(HF_DATASETS_CACHE) / "Salesforce___wikitext/wikitext-2-raw-v1").glob(
                "*/*/wikitext-test.arrow"
            )
        )
        if len(matches) > 1:
            raise ValueError(
                "multiple cached WikiText revisions; select --dataset-cache-file"
            )
        if matches:
            cache_file = matches[0]
    if cache_file is not None:
        rows = Dataset.from_file(str(cache_file))
    else:
        rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        cache_file = Path(rows.cache_files[0]["filename"])
    return rows, {
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "test",
        "cache_file": str(cache_file.resolve()),
        "cache_sha256": _digest(cache_file),
        "dataset_fingerprint": rows._fingerprint,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def _boundary_stop_tokens(tokenizer) -> list[int]:
    """Public token-level sentence/newline stop rule, identical for every query."""
    result = []
    for token in sorted(set(tokenizer.get_vocab().values())):
        text = tokenizer.decode([token])
        if "\n" in text or text.rstrip(" \t\"'”’)]").endswith((".", "!", "?")):
            result.append(token)
    return result


def _rollouts(
    args: argparse.Namespace, output: Path
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Checkpoint independent seeds and freeze profiles before evaluation."""
    import torch
    from experiments.gpt2_experiments import _load_model, generate_trace

    model, tokenizer, _ = _load_model(args.model, args.device)
    stop_tokens = (
        _boundary_stop_tokens(tokenizer) if args.stop_rule == "sentence" else []
    )
    _write_json(output / "stop_tokens.json", stop_tokens)
    prompt_path = output / "prompts.json"
    if prompt_path.exists():
        saved = json.loads(prompt_path.read_text())
        prompts, dataset = saved["prompts"], saved["dataset"]
    else:
        rows, dataset = _load_wikitext(args.dataset_cache_file)
        prompts = wikitext_prompts(
            rows,
            lambda s: tokenizer(s, add_special_tokens=False)["input_ids"],
            count=args.pool_size,
            lengths=tuple(args.prompt_lengths),
            max_output=args.max_output,
            seed=args.seed,
        )
        for prompt in prompts:
            prompt["prompt_text"] = tokenizer.decode(prompt["prompt_tokens"])
        _write_json(prompt_path, {"dataset": dataset, "prompts": prompts})
    phases = {}
    for phase, repetitions in (
        ("profile", args.profile_replicates),
        ("evaluation", args.evaluation_replicates),
    ):
        path = output / f"{phase}_rollouts.jsonl"
        rows = _read_jsonl(path)
        done = {r["challenge_seed"] for r in rows}
        expected = {
            challenge_seed(args.seed, phase, p["query_id"], i)
            for p in prompts
            for i in range(repetitions)
        }
        if len(done) != len(rows) or not done.issubset(expected):
            raise ValueError(f"invalid {phase} checkpoint challenges")
        with (
            path.open("a") as stream,
            tqdm(
                total=len(expected),
                initial=len(done),
                desc=f"GPT-2 {phase}",
                unit="rollout",
            ) as progress,
        ):
            for prompt in prompts:
                for replicate in range(repetitions):
                    seed = challenge_seed(
                        args.seed, phase, prompt["query_id"], replicate
                    )
                    if seed in done:
                        continue
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize(args.device)
                    started = time.perf_counter()
                    trace = generate_trace(
                        model,
                        tokenizer,
                        prompt["prompt_tokens"],
                        seed,
                        length=args.max_output,
                        device=args.device,
                        temperature=args.temperature,
                        stop_eos=True,
                        stop_token_ids=stop_tokens,
                    )
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize(args.device)
                    row = {
                        "query_id": prompt["query_id"],
                        "phase": phase,
                        "replicate": replicate,
                        "challenge_seed": seed,
                        "prompt_sha256": prompt["prompt_sha256"],
                        "prompt_length": prompt["prompt_length"],
                        "output_length": len(trace.tokens),
                        "output_tokens": trace.tokens,
                        "gpt2_duration": time.perf_counter() - started,
                        "stop_reason": "eos"
                        if trace.tokens[-1] == tokenizer.eos_token_id
                        else (
                            "boundary"
                            if trace.tokens[-1] in stop_tokens
                            else "length_cap"
                        ),
                    }
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    rows.append(row)
                    progress.update(1)
        phases[phase] = rows
        if phase == "profile":
            _write_json(output / "profile_pool.json", profile_pool(prompts, rows))
    model_revision = getattr(model.config, "_commit_hash", None)
    del model
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return (
        prompts,
        phases["profile"],
        phases["evaluation"],
        {"dataset": dataset, "model_revision": model_revision},
    )


def _parse_proofs(directory: Path, trials: list[dict]) -> list[dict]:
    """Align by trial position, never collapse different prompts sharing N,K."""
    ledger = _read_jsonl(directory / "deepprove_ledger.jsonl")
    with (directory / "deepprove_timings.csv").open() as stream:
        timings = list(csv.DictReader(stream))
    if len(ledger) != len(trials) or len(timings) != len(trials):
        raise ValueError("DeepProve ledger/timing trial counts do not match")
    results = []
    for index, (trial, audit, timing) in enumerate(zip(trials, ledger, timings), 1):
        meta = audit["meta"]
        n, k = trial["prompt_length"], trial["output_length"]
        if (
            audit.get("verified") is not True
            or meta["backend"] != "cuda"
            or meta["trial"] != index
            or meta["n"] != n
            or meta["k"] != k
            or meta["tokens"] != trial["prompt_tokens"]
        ):
            raise ValueError(
                f"DeepProve trial {index} provenance/verification mismatch"
            )
        if int(timing["min_user_len"]) != n or int(timing["max_context"]) != n + k:
            raise ValueError("DeepProve timing shape differs from ledger")
        inference, proof = float(timing["inference_time"]), float(timing["prove_full"])
        if inference <= 0 or proof <= 0:
            raise ValueError("nonpositive DeepProve time")
        results.append(
            {
                "inference_ms": inference,
                "proof_ms": proof,
                "verify_ms": float(timing["verify_full"]),
                "proof_size": int(timing["proof_size"]),
                "duration": (inference + proof) / 1000,
                "backend": "cuda",
                "proof_trial": index,
                "proof_directory": str(directory.resolve()),
                "proof_setup_max": meta["setup_max"],
            }
        )
    return results


def _prove_batch(
    args: argparse.Namespace, directory: Path, trials: list[dict]
) -> list[dict]:
    """Checkpoint a batch only after every exact-prompt proof is verified."""
    directory.mkdir(parents=True, exist_ok=True)
    inputs = directory / "trials.json"
    if inputs.exists() and json.loads(inputs.read_text()) != trials:
        raise ValueError("proof checkpoint input mismatch")
    _write_json(inputs, trials)
    if (directory / "COMPLETED").exists():
        return _parse_proofs(directory, trials)
    # Retrying a failed batch preserves all earlier completed batches.
    for filename in ("deepprove_ledger.jsonl", "deepprove_timings.csv"):
        (directory / filename).unlink(missing_ok=True)
    prompts = directory / "deepprove_prompts.json"
    _write_json(prompts, [t["prompt_tokens"] for t in trials])
    command = [
        str(args.bench_binary.resolve()),
        "--model",
        "gpt2",
        "--hf",
        args.model,
        "--pairs",
        ",".join(f"{t['prompt_length']}:{t['output_length']}" for t in trials),
        "--fixed-length",
        "--num-threads",
        str(args.threads),
        "--work-token-seed",
        str(args.seed),
        "--work-prompts",
        str(prompts),
        "--work-ledger",
        str(directory / "deepprove_ledger.jsonl"),
        "--bench",
        str(directory / "deepprove_timings.csv"),
    ]
    _write_json(directory / "command.json", command)
    with (
        (directory / "deepprove.log").open("w") as log,
        tqdm(total=len(trials), desc=directory.name, unit="proof") as progress,
    ):
        with subprocess.Popen(
            command,
            cwd=ROOT / ".scratch/deep-prove/zkml",
            env=_cuda_environment(args.device),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                if "POML_EVENT" in line and "stage=done" in line:
                    progress.update(1)
            if process.wait() != 0:
                raise RuntimeError(
                    f"DeepProve failed; see {directory / 'deepprove.log'}; resume to retry this batch"
                )
    result = _parse_proofs(directory, trials)
    (directory / "COMPLETED").write_text("verified\n")
    return result


def parser() -> argparse.ArgumentParser:
    """Build the benchmark campaign CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dataset-cache-file", type=Path)
    ap.add_argument("--pool-size", type=int, default=128)
    ap.add_argument("--prompt-lengths", type=int, nargs="+", default=[8, 16, 24, 32])
    ap.add_argument("--profile-replicates", type=int, default=16)
    ap.add_argument("--evaluation-replicates", type=int, default=4)
    ap.add_argument("--max-output", type=int, default=32)
    ap.add_argument("--stop-rule", choices=("sentence", "eos"), default="sentence")
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--proof-batch-size", type=int, default=32)
    ap.add_argument("--bench-binary", type=Path, default=DEFAULT_BINARY)
    ap.add_argument("--build-cuda", action="store_true")
    ap.add_argument(
        "--inference-only",
        action="store_true",
        help="profile/evaluate lengths without mining timings",
    )
    ap.add_argument("--resume", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Build frozen profiles and a held-out benchmark replay bank."""
    args = parser().parse_args(argv)
    schedule = load_schedule(args.schedule)
    if (
        args.pool_size < 1
        or min(args.profile_replicates, args.evaluation_replicates) < 2
    ):
        raise ValueError(
            "positive pool size and at least two replicates per phase required"
        )
    if (
        not args.prompt_lengths
        or min(args.prompt_lengths) < 2
        or args.max_output < 1
        or max(args.prompt_lengths) + args.max_output > schedule["setup_max"]
        or args.proof_batch_size < 1
        or args.temperature <= 0
    ):
        raise ValueError("invalid length, temperature, or batch setting")
    if not args.inference_only and not args.device.startswith("cuda"):
        raise ValueError("verified mining traces require --device cuda:N")
    output = args.output.resolve()
    config = {
        k: str(v.resolve()) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in {"resume", "inference_only", "build_cuda", "output"}
    }
    config.update(
        schema=BENCHMARK_SCHEMA,
        schedule_sha256=_digest(args.schedule),
        builder_sha256=_digest(Path(__file__)),
        benchmark_sha256=_digest(ROOT / "src/poml_sim/llm_benchmark.py"),
        sampler_sha256=_digest(ROOT / "experiments/gpt2_experiments.py"),
    )
    checkpoint = output / "run_config.json"
    if checkpoint.exists():
        if not args.resume or json.loads(checkpoint.read_text()) != config:
            raise ValueError(
                "existing campaign requires --resume with identical inputs and sources"
            )
    elif output.exists() and any(output.iterdir()):
        raise ValueError("refusing nonempty output without a benchmark checkpoint")
    _write_json(checkpoint, config)
    prompts, profile, evaluation, provenance = _rollouts(args, output)
    pool = profile_pool(prompts, profile)
    diagnostics = prediction_diagnostics(pool, profile, evaluation)
    _write_json(output / "prediction_diagnostics.json", diagnostics)
    if args.inference_only:
        print(
            json.dumps(
                {k: v for k, v in diagnostics.items() if k != "per_query"}, indent=2
            )
        )
        return 0
    if args.build_cuda:
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "bench-llm", "--features", "cuda"],
            cwd=ROOT / ".scratch/deep-prove",
            env=_cuda_environment(args.device),
            check=True,
        )
    if not args.bench_binary.is_file():
        raise FileNotFoundError(f"missing DeepProve binary {args.bench_binary}")
    prompt_by_id = {p["query_id"]: p for p in pool}
    trials = [
        {**r, "prompt_tokens": prompt_by_id[r["query_id"]]["prompt_tokens"]}
        for r in evaluation
    ]
    warmup = max(trials, key=lambda r: r["prompt_length"] + r["output_length"])
    measured = []
    for start in range(0, len(trials), args.proof_batch_size):
        chunk = trials[start : start + args.proof_batch_size]
        # Common warm-up keeps the setup size identical across proof batches.
        timings = _prove_batch(
            args, output / "proof_batches" / f"batch-{start:06d}", [warmup, *chunk]
        )[1:]
        measured.extend({**trial, **timing} for trial, timing in zip(chunk, timings))
    trace_path = output / "traces.jsonl"
    with trace_path.open("w") as stream:
        for row in measured:
            query = QuerySpec(row["query_id"], row["prompt_length"], args.max_output)
            row["complexity"] = complexity_for(query, row["output_length"], schedule)
            row["duration_scope"] = (
                "DeepProve CUDA inference plus proving for this prompt and fixed K"
            )
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    for prompt in pool:
        query = QuerySpec(prompt["query_id"], prompt["prompt_length"], args.max_output)
        prompt["fee_limit"] = complexity_for(query, args.max_output, schedule)
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = {
        **provenance,
        "schema": BENCHMARK_SCHEMA,
        "model": args.model,
        "device": args.device,
        "backend": "cuda",
        "pool_size": len(pool),
        "pool": pool,
        "profile_replicates": args.profile_replicates,
        "evaluation_replicates": args.evaluation_replicates,
        "seed": args.seed,
        "max_output": args.max_output,
        "temperature": args.temperature,
        "stop_rule": args.stop_rule,
        "stop_tokens_sha256": _digest(output / "stop_tokens.json"),
        "schedule_sha256": _digest(args.schedule),
        "trace_sha256": _digest(trace_path),
        "profile_sha256": _digest(output / "profile_rollouts.jsonl"),
        "evaluation_sha256": _digest(output / "evaluation_rollouts.jsonl"),
        "proof_measurements": len(measured),
        "proof_timing_reuse": False,
        "duration_scope": "DeepProve CUDA inference plus proving per evaluated benchmark prompt and K",
        "prover_limitation": "DeepProve uses its own fixed-length decoder; GPT-2 sampled output tokens and PoML embedding perturbation are not constrained by this benchmark proof",
        "python": platform.python_version(),
        "gpu_inventory": gpu.stdout.strip(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "datasets", "numpy")
        },
    }
    _write_json(trace_path.with_suffix(".metadata.json"), metadata)
    (output / "COMPLETED").write_text("verified benchmark evaluation bank\n")
    print(
        json.dumps(
            {
                "trace_file": str(trace_path),
                "queries": len(pool),
                "proofs": len(measured),
                "prediction_diagnostics": str(output / "prediction_diagnostics.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
