"""Benchmark prompt selection and leakage-free output-length profiling.

Dataset and model I/O stay in the experiment driver. This module operates on
plain records so provenance and profiling separation can be tested directly.
"""

from collections import defaultdict
import hashlib
import json
import math
import random
import statistics
from typing import Callable, Iterable


BENCHMARK_SCHEMA = "poml-benchmark-gpt2-deepprove-2"
SELECTION_POLICIES = ("uniform", "profiled-short", "profiled-long", "shortest-prompt")


def token_digest(tokens: list[int]) -> str:
    """Identify exact token sequences independently of dataset row identifiers."""
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


def wikitext_prompts(
    rows: Iterable[dict],
    encode: Callable[[str], list[int]],
    *,
    count: int,
    lengths: tuple[int, ...],
    max_output: int,
    seed: int,
) -> list[dict]:
    """Sample distinct token windows from distinct non-heading WikiText rows."""
    if count < 1 or not lengths or min(lengths) < 2 or max_output < 1:
        raise ValueError("positive count/output and prompt lengths >= 2 required")
    candidates = [
        (i, row["text"])
        for i, row in enumerate(rows)
        if row["text"].strip() and not row["text"].strip().startswith("=")
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    prompts, seen = [], set()
    for index, text in candidates:
        n = lengths[len(prompts) % len(lengths)]
        tokens = list(encode(text))
        if len(tokens) < n:
            continue
        offset = rng.randrange(len(tokens) - n + 1)
        window = tokens[offset : offset + n]
        digest = token_digest(window)
        if digest in seen:
            continue
        seen.add(digest)
        prompts.append(
            {
                "query_id": f"wikitext2-test-{index}-{offset}-{n}",
                "dataset": "Salesforce/wikitext",
                "dataset_config": "wikitext-2-raw-v1",
                "split": "test",
                "source_row": index,
                "token_offset": offset,
                "source_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "prompt_tokens": window,
                "prompt_sha256": digest,
                "prompt_length": n,
                "max_output_length": max_output,
            }
        )
        if len(prompts) == count:
            return prompts
    raise ValueError(
        f"only {len(prompts)} distinct eligible benchmark rows for {count} prompts"
    )


def challenge_seed(seed: int, phase: str, query_id: str, replicate: int) -> str:
    """Domain-separate profiling randomness from held-out evaluation randomness."""
    if phase not in {"profile", "evaluation"} or replicate < 0:
        raise ValueError("invalid rollout phase or replicate")
    return f"poml-benchmark:{seed}:{phase}:{query_id}:{replicate}"


def profile_pool(prompts: list[dict], profile: list[dict]) -> list[dict]:
    """Freeze per-prompt means using profiling observations only."""
    groups = defaultdict(list)
    challenges = set()
    for row in profile:
        if row["phase"] != "profile":
            raise ValueError("evaluation outcomes cannot be used for profiling")
        if row["challenge_seed"] in challenges:
            raise ValueError("duplicate profiling challenge")
        challenges.add(row["challenge_seed"])
        groups[row["query_id"]].append(row)
    if set(groups) != {p["query_id"] for p in prompts}:
        raise ValueError("profiling queries do not match prompt manifest")
    result = []
    for prompt in prompts:
        records = groups[prompt["query_id"]]
        values = [int(r["output_length"]) for r in records]
        if len(values) < 2 or any(
            k < 1 or k > prompt["max_output_length"] for k in values
        ):
            raise ValueError("at least two valid profiling lengths per prompt required")
        if any(r["prompt_sha256"] != prompt["prompt_sha256"] for r in records):
            raise ValueError("profiling prompt digest mismatch")
        result.append(
            {
                **prompt,
                "profile_mean_output_length": statistics.fmean(values),
                "profile_stdev_output_length": statistics.stdev(values),
                "profile_trials": len(values),
                "profile_inference_seconds": sum(r["gpt2_duration"] for r in records),
            }
        )
    return result


def _ranks(values: list[float]) -> list[float]:
    positions = defaultdict(list)
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        positions[values[index]].append(rank)
    return [statistics.fmean(positions[value]) for value in values]


def prediction_diagnostics(
    pool: list[dict], profile: list[dict], evaluation: list[dict]
) -> dict:
    """Measure held-out length prediction without changing the frozen ranking."""
    profile_seeds = {r["challenge_seed"] for r in profile}
    evaluation_seeds = [r["challenge_seed"] for r in evaluation]
    if profile_seeds.intersection(evaluation_seeds) or len(
        set(evaluation_seeds)
    ) != len(evaluation_seeds):
        raise ValueError("profiling and evaluation challenges must be distinct")
    groups = defaultdict(list)
    for row in evaluation:
        if row["phase"] != "evaluation":
            raise ValueError("evaluation bank contains a profiling outcome")
        groups[row["query_id"]].append(row)
    if set(groups) != {p["query_id"] for p in pool}:
        raise ValueError("evaluation queries do not match profiled pool")
    per_query = []
    for prompt in pool:
        records = groups[prompt["query_id"]]
        if len(records) < 2:
            raise ValueError("at least two held-out rollouts per query required")
        if any(r["prompt_sha256"] != prompt["prompt_sha256"] for r in records):
            raise ValueError("evaluation prompt digest mismatch")
        lengths = [r["output_length"] for r in records]
        per_query.append(
            {
                "query_id": prompt["query_id"],
                "prompt_length": prompt["prompt_length"],
                "profile_mean_k": prompt["profile_mean_output_length"],
                "evaluation_mean_k": statistics.fmean(lengths),
                "evaluation_sd_k": statistics.stdev(lengths),
                "evaluation_trials": len(lengths),
                "evaluation_cap_fraction": statistics.fmean(
                    r["stop_reason"] == "length_cap" for r in records
                ),
            }
        )
    predicted = [r["profile_mean_k"] for r in per_query]
    observed = [r["evaluation_mean_k"] for r in per_query]
    x, y = _ranks(predicted), _ranks(observed)
    rank_correlation = (
        statistics.correlation(x, y) if len(set(x)) > 1 and len(set(y)) > 1 else None
    )
    ordered = sorted(per_query, key=lambda r: (r["profile_mean_k"], r["query_id"]))
    quartile = max(1, math.ceil(len(ordered) / 4))
    return {
        "queries": len(pool),
        "profile_trials": len(profile),
        "evaluation_trials": len(evaluation),
        "mean_absolute_error_k": statistics.fmean(
            abs(a - b) for a, b in zip(predicted, observed)
        ),
        "spearman_rank_correlation": rank_correlation,
        "profile_mean_k_range": [min(predicted), max(predicted)],
        "evaluation_mean_k_range": [min(observed), max(observed)],
        "profile_shortest_quartile_evaluation_mean_k": statistics.fmean(
            r["evaluation_mean_k"] for r in ordered[:quartile]
        ),
        "profile_longest_quartile_evaluation_mean_k": statistics.fmean(
            r["evaluation_mean_k"] for r in ordered[-quartile:]
        ),
        "evaluation_cap_fraction": statistics.fmean(
            r["stop_reason"] == "length_cap" for r in evaluation
        ),
        "profiling_inference_seconds": sum(r["gpt2_duration"] for r in profile),
        "ranking_identifiable": len(set(predicted)) > 1 and len(set(observed)) > 1,
        "per_query": per_query,
    }
