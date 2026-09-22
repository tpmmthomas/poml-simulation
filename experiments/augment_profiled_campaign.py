#!/usr/bin/env python3
"""Extend a completed profiled campaign with measured shorter-output pairs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from poml_sim.gpt2_work import reference_counts  # noqa: E402
from poml_sim.live_mining import FreshWork, append_json  # noqa: E402
from poml_sim.live_prover import LiveProver  # noqa: E402
from poml_sim.llm_simulation import load_schedule  # noqa: E402
from poml_sim.protocol_inputs import (  # noqa: E402
    canonical,
    digest,
    experimental_key,
    gaussian_noise,
    decoding_uniforms,
)
from experiments.run_live_llm_experiments import (  # noqa: E402
    SCHEMA,
    calibrate_difficulty,
    read_jsonl,
    summarize,
    write_json,
)


def hardlink_tree(source: Path, destination: Path) -> None:
    """Clone campaign metadata/artifacts without duplicating NAS storage."""
    def ignore_stages(directory, names):
        if Path(directory) == source:
            return {name for name in names if name in {"liveness", "cherry_pick", "wasted_work", "COMPLETED"}}
        return set()

    shutil.copytree(source, destination, copy_function=os.link, ignore=ignore_stages)
    # The proof/logit artifacts are immutable and may be hard-linked. These
    # ledgers are appended or rewritten during augmentation, so break their
    # links before any work starts.
    mutable = (
        "campaign.json",
        "preparation/pool.json",
        "preparation/profiles.jsonl",
        "preparation/timings.jsonl",
        "preparation/calibration.json",
        "preparation/bank_manifest.json",
        "preparation/COMPLETED",
        "preparation/prediction_diagnostics.json",
        "preparation/proofs/executed.jsonl",
    )
    for relative in mutable:
        target = destination / relative
        if target.exists():
            original = source / relative
            target.unlink()
            shutil.copy2(original, target)


def profile_variant(prover, directory, query, seed, replicate, alpha):
    challenge = canonical([seed, "augment-profile", query["query_id"], replicate])
    noise, vrfs = gaussian_noise(
        experimental_key(seed, "augment-profile-inf"),
        digest(challenge),
        query["prompt_length"],
        query["max_output_length"],
        alpha,
        prover.ready["embedding_std"],
    )
    identity = digest(challenge).hex()
    result = prover.run(
        {
            "request_id": identity,
            "prompt_tokens": query["prompt_tokens"],
            "noise": noise,
            "uniforms": decoding_uniforms(vrfs, query["prompt_length"]),
            "max_output": query["max_output_length"],
            "mode": "profile",
            "output_directory": str((directory / "profiles" / identity).resolve()),
        }
    )
    return {
        **result,
        "query_id": query["query_id"],
        "prompt_sha256": query["prompt_sha256"],
        "prompt_length": query["prompt_length"],
        "phase": "profile",
        "replicate": replicate,
        "challenge_seed": challenge.hex(),
        "gpt2_duration": result["inference_seconds"],
        "raw_complexity": sum(
            reference_counts(
                query["prompt_length"], result["output_length"], query["schedule"]
            )["combined"].values()
        ),
        "vrf_transcript": vrfs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--binary", type=Path, default=ROOT / ".scratch/deep-prove/target/release/poml-prover")
    parser.add_argument("--schedule", type=Path, default=ROOT / "config/gpt2_reference_schedule.json")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--base-prompts", type=int, default=8)
    parser.add_argument("--short-caps", type=int, nargs="+", default=[8, 16, 24])
    parser.add_argument("--profile-replicates", type=int, default=2)
    parser.add_argument("--proof-replicates", type=int, default=1)
    args = parser.parse_args(argv)
    source, destination = args.source.resolve(), args.destination.resolve()
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    if not (source / "preparation/COMPLETED").exists():
        raise ValueError("source campaign is not a completed profiled campaign")
    hardlink_tree(source, destination)
    for stage in ("liveness", "cherry_pick", "wasted_work", "COMPLETED"):
        target = destination / stage
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    schedule = load_schedule(args.schedule)
    old_pool = json.loads((destination / "preparation/pool.json").read_text())
    old_profiles = read_jsonl(destination / "preparation/profiles.jsonl")
    old_timings = read_jsonl(destination / "preparation/timings.jsonl")
    calibration = json.loads((destination / "preparation/calibration.json").read_text())
    selected = old_pool[: args.base_prompts]
    variants = []
    for base in selected:
        for cap in args.short_caps:
            if cap >= base["max_output_length"] or base["prompt_length"] + cap > 64:
                continue
            variants.append(
                {
                    **base,
                    "query_id": f"{base['query_id']}-cap-{cap}",
                    "request_id": f"{base['query_id']}-cap-{cap}",
                    "max_output_length": cap,
                    "profile_mean_output_length": float(cap),
                    "profile_stdev_output_length": 0.0,
                    "profile_trials": args.profile_replicates,
                    "profile_inference_seconds": 0.0,
                    "augmented_from": base["query_id"],
                }
            )
    if not variants:
        raise ValueError("no valid shorter variants were generated")
    existing_ids = {q["query_id"] for q in old_pool}
    if existing_ids.intersection(q["query_id"] for q in variants):
        raise ValueError("variant query identifiers already exist")

    preparation = destination / "preparation"
    new_profiles = []
    with LiveProver(
        args.binary,
        destination / "augment-prover",
        args.device,
        threads=8,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
    ) as prover:
        augment_ready = prover.ready
        for query in tqdm(variants, desc="Profile shorter-output variants"):
            query = {**query, "schedule": schedule}
            for replicate in range(args.profile_replicates):
                row = profile_variant(
                    prover,
                    preparation,
                    query,
                    args.seed,
                    replicate,
                    args.alpha,
                )
                row.pop("schedule", None)
                new_profiles.append(row)
        work = FreshWork(
            prover,
            schedule,
            calibration["scale"],
            preparation / "proofs",
            args.seed + 7,
            args.alpha,
        )
        existing = list(work.directory.glob("attempt-*"))
        work.executions = max(
            [int(p.name.split("-")[1]) for p in existing], default=-1
        ) + 1
        new_timings = []
        for query in tqdm(variants, desc="Prove shorter-output variants"):
            for replicate in range(args.proof_replicates):
                binding = digest(canonical(["augment-bank", args.seed, query["query_id"], replicate]))
                row = work.execute(
                    {**query, "request_id": f"augment:{query['query_id']}:{replicate}"},
                    0,
                    binding,
                    binding,
                    b"",
                    1,
                )
                row.pop("ciphertext_prefix")
                row["replicate"] = replicate
                row["service_seconds"] = row["duration"]
                row["duration"] = row["inference_proof_seconds"]
                append_json(preparation / "timings.jsonl", row)
                new_timings.append(row)

    for query in variants:
        rows = [r for r in new_profiles if r["query_id"] == query["query_id"]]
        query["profile_mean_output_length"] = statistics.fmean(r["output_length"] for r in rows)
        query["profile_stdev_output_length"] = (
            statistics.stdev(r["output_length"] for r in rows) if len(rows) > 1 else 0.0
        )
        query["profile_inference_seconds"] = sum(r["gpt2_duration"] for r in rows)
        query.pop("schedule", None)
    pool = old_pool + variants
    profiles = old_profiles + new_profiles
    timings = old_timings + new_timings
    write_json(preparation / "pool.json", pool)
    (preparation / "profiles.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True) for r in profiles) + "\n")
    bank_manifest = {
        "schema": "poml-measured-bank-1",
        "records": len(timings),
        "prompts": len(pool),
        "replicates_per_prompt": "mixed: original 2, augmented 1",
        "output_lengths": sorted({r["output_length"] for r in timings}),
        "prompt_lengths": sorted({r["prompt_length"] for r in timings}),
        "duration_statistics": summarize([r["duration"] for r in timings]),
        "fresh_inference_and_proof": all(r.get("fresh_inference") and r.get("fresh_proof") for r in timings),
        "bank_sha256": digest(canonical(timings)).hex(),
        "augmentation": {"base_prompts": args.base_prompts, "short_caps": args.short_caps, "new_pairs": len(new_timings)},
    }
    write_json(preparation / "bank_manifest.json", bank_manifest)
    updated = calibrate_difficulty(pool, timings, schedule, 4, 300, args.seed)
    updated.update(
        schema=SCHEMA,
        scale=calibration["scale"],
        prover=augment_ready,
        profile_rank_identifiable=True,
        profile_cap_fraction=statistics.fmean(r["stop_reason"] == "length_cap" for r in profiles),
        duration_statistics=bank_manifest["duration_statistics"],
        measured_bank=bank_manifest,
        profiling_inference_seconds=sum(r["gpt2_duration"] for r in profiles),
        calibration_physical_seconds=sum(r["host_seconds"] for r in timings),
        measurement_bank_physical_seconds=sum(r["gpt2_duration"] for r in profiles) + sum(r["host_seconds"] for r in timings),
        notes=[
            "Original verified proof records were hard-linked and reused.",
            "Augmented variants use real GPT-2/DeepProve runs capped at K=8,16,24.",
            "Experiments 1 and 2 replay measured inference+proof durations and recompute protocol lotteries.",
        ],
    )
    write_json(preparation / "calibration.json", updated)
    (preparation / "COMPLETED").write_text("complete\n")
    config = json.loads((destination / "campaign.json").read_text())
    config["output"] = str(destination)
    config["selection_blocks"] = 50
    config["sources"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT / "experiments/run_live_llm_experiments.py", *sorted((ROOT / "src/poml_sim").glob("*.py"))]
    }
    write_json(destination / "campaign.json", config)
    print(json.dumps({"destination": str(destination), "new_pairs": len(new_timings), "records": len(timings), "caps": args.short_caps}, indent=2))


if __name__ == "__main__":
    main()
