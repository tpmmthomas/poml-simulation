#!/usr/bin/env python3
"""Experiment 5: uniform-fee query collisions in one-block PoML races.

The final grid uses accelerated discrete-event time. Attempt durations are
sampled from an empirical calibration produced by the repository's real EZKL
inference-and-proof path; lottery attempts retain the simulator's 256-bit
uniform-threshold semantics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR, localcontext
from pathlib import Path
from queue import Empty
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UINT256_SIZE = 1 << 256
MAX_DIFFICULTY = UINT256_SIZE - 1
SIMULATION_MODE = "empirical_discrete_event_v1"
SCHEMA_VERSION = 1
TIMING_HEARTBEAT_S = 10.0
TIMING_POLL_S = 1.0
DEFAULT_SEED_MANIFEST = PROJECT_ROOT / "experiments" / "exp5_seed_manifest.json"
MODEL_PROOF_FILES = (
    "network.onnx",
    "network.onnx.data",
    "network.ezkl",
    "settings.json",
    "pk.key",
    "vk.key",
    "kzg.srs",
)
ATTEMPT_FIELDS = (
    "run_id",
    "config_id",
    "run_seed",
    "miner_id",
    "attempt_index",
    "query_id",
    "fee",
    "status",
    "started_at_s",
    "scheduled_completion_s",
    "sampled_duration_s",
    "elapsed_s",
    "remaining_s",
    "coordinator_sequence",
    "completed_before_adoption",
    "partial",
    "collision",
    "lottery_draw_hex",
    "lottery_won",
    "cancellation_reason",
)


def canonical_json(value: Any) -> str:
    """Serialize experiment metadata for stable content identifiers."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_id(value: Any) -> str:
    """Return a stable SHA-256 identifier for canonical JSON data."""
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def derive_seed(root: str | int, namespace: str, *parts: object) -> int:
    """Derive a domain-separated 64-bit seed."""
    material = "\0".join([str(root), namespace, *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def analytical_difficulty(mean_attempt_seconds: float, miners: int, target_seconds: float) -> int:
    """Derive the lottery threshold from aggregate renewal-process throughput."""
    if mean_attempt_seconds <= 0 or miners <= 0 or target_seconds <= 0:
        raise ValueError("mean attempt time, miners, and target time must be positive")
    with localcontext() as context:
        context.prec = 100
        probability = Decimal(str(mean_attempt_seconds)) / (
            Decimal(miners) * Decimal(str(target_seconds))
        )
        threshold = int(
            (probability * Decimal(UINT256_SIZE)).to_integral_value(rounding=ROUND_FLOOR)
        )
    return max(1, min(threshold, MAX_DIFFICULTY))


class LazyPermutation:
    """Generate a uniformly shuffled prefix without allocating all Q entries."""

    def __init__(self, size: int, rng: random.Random) -> None:
        if size < 0:
            raise ValueError("permutation size cannot be negative")
        self.size = size
        self.rng = rng
        self.position = 0
        self.swaps: dict[int, int] = {}

    def draw(self) -> int | None:
        if self.position >= self.size:
            return None
        index = self.position
        swap_index = self.rng.randrange(index, self.size)
        index_value = self.swaps.get(index, index)
        swap_value = self.swaps.get(swap_index, swap_index)
        self.swaps[swap_index] = index_value
        self.swaps.pop(index, None)
        self.position += 1
        return swap_value


@dataclass(frozen=True)
class ActiveAttempt:
    miner_id: int
    attempt_index: int
    query_id: int
    started_at_s: float
    sampled_duration_s: float
    scheduled_completion_s: float
    tie_key: int
    insertion_sequence: int


def _attempt_record(
    attempt: ActiveAttempt,
    *,
    run_id: str,
    config_id: str,
    run_seed: int,
    fee: int,
    status: str,
    elapsed_s: float,
    remaining_s: float,
    coordinator_sequence: int | None,
    collision: bool | None,
    lottery_draw: int | None,
    lottery_won: bool | None,
    partial: bool,
    cancellation_reason: str = "",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "config_id": config_id,
        "run_seed": run_seed,
        "miner_id": attempt.miner_id,
        "attempt_index": attempt.attempt_index,
        "query_id": attempt.query_id,
        "fee": fee,
        "status": status,
        "started_at_s": attempt.started_at_s,
        "scheduled_completion_s": attempt.scheduled_completion_s,
        "sampled_duration_s": attempt.sampled_duration_s,
        "elapsed_s": elapsed_s,
        "remaining_s": remaining_s,
        "coordinator_sequence": coordinator_sequence,
        "completed_before_adoption": status == "completed",
        "partial": partial,
        "collision": collision,
        "lottery_draw_hex": f"0x{lottery_draw:064x}" if lottery_draw is not None else "",
        "lottery_won": lottery_won,
        "cancellation_reason": cancellation_reason,
    }


def simulate_race(
    *,
    miners: int,
    query_pool_size: int,
    difficulty: int,
    duration_samples_s: list[float],
    run_seed: int,
    config_id: str,
    run_id: str,
    fee: int = 1,
    retain_details: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Simulate one race and return its run-level and attempt-level records."""
    if miners <= 0 or query_pool_size <= 0:
        raise ValueError("miners and query pool size must be positive")
    if not 1 <= difficulty <= MAX_DIFFICULTY:
        raise ValueError("difficulty must be a positive 256-bit threshold")
    if fee < 0:
        raise ValueError("fee cannot be negative")
    if not duration_samples_s or any(
        not math.isfinite(sample) or sample <= 0 for sample in duration_samples_s
    ):
        raise ValueError("duration samples must be finite and positive")

    orderings = [
        LazyPermutation(
            query_pool_size,
            random.Random(derive_seed(run_seed, "query-order", miner_id)),
        )
        for miner_id in range(miners)
    ]
    duration_rngs = [
        random.Random(derive_seed(run_seed, "duration", miner_id))
        for miner_id in range(miners)
    ]
    lottery_rngs = [
        random.Random(derive_seed(run_seed, "lottery", miner_id))
        for miner_id in range(miners)
    ]
    tie_rngs = [
        random.Random(derive_seed(run_seed, "tie-order", miner_id))
        for miner_id in range(miners)
    ]

    event_heap: list[tuple[float, int, int, ActiveAttempt]] = []
    active: dict[int, ActiveAttempt] = {}
    attempts_started = [0] * miners
    insertion_sequence = 0

    def start_attempt(miner_id: int, started_at_s: float) -> bool:
        nonlocal insertion_sequence
        query_id = orderings[miner_id].draw()
        if query_id is None:
            return False
        attempts_started[miner_id] += 1
        duration = duration_rngs[miner_id].choice(duration_samples_s)
        insertion_sequence += 1
        attempt = ActiveAttempt(
            miner_id=miner_id,
            attempt_index=attempts_started[miner_id],
            query_id=query_id,
            started_at_s=started_at_s,
            sampled_duration_s=duration,
            scheduled_completion_s=started_at_s + duration,
            tie_key=tie_rngs[miner_id].getrandbits(64),
            insertion_sequence=insertion_sequence,
        )
        active[miner_id] = attempt
        heapq.heappush(
            event_heap,
            (
                attempt.scheduled_completion_s,
                attempt.tie_key,
                attempt.insertion_sequence,
                attempt,
            ),
        )
        return True

    for miner_id in range(miners):
        start_attempt(miner_id, 0.0)

    attempt_records: list[dict[str, Any]] = []
    completed_queries: list[list[int]] | None = (
        [[] for _ in range(miners)] if retain_details else None
    )
    completed_count = 0
    completion_counts: dict[int, int] = {}
    collision_count = 0
    coordinator_sequence = 0
    adoption_time_s: float | None = None
    winner_miner_id: int | None = None
    winner_query_id: int | None = None
    tied_completions_cancelled = 0

    while event_heap:
        _, _, _, attempt = heapq.heappop(event_heap)
        if active.get(attempt.miner_id) != attempt:
            continue
        del active[attempt.miner_id]
        coordinator_sequence += 1

        lottery_draw = lottery_rngs[attempt.miner_id].getrandbits(256)
        lottery_won = lottery_draw < difficulty
        collision = completion_counts.get(attempt.query_id, 0) > 0
        collision_count += int(collision)
        completion_counts[attempt.query_id] = completion_counts.get(attempt.query_id, 0) + 1
        completed_count += 1
        if completed_queries is not None:
            completed_queries[attempt.miner_id].append(attempt.query_id)
        if retain_details:
            attempt_records.append(
                _attempt_record(
                    attempt,
                    run_id=run_id,
                    config_id=config_id,
                    run_seed=run_seed,
                    fee=fee,
                    status="completed",
                    elapsed_s=attempt.sampled_duration_s,
                    remaining_s=0.0,
                    coordinator_sequence=coordinator_sequence,
                    collision=collision,
                    lottery_draw=lottery_draw,
                    lottery_won=lottery_won,
                    partial=False,
                )
            )

        if lottery_won:
            adoption_time_s = attempt.scheduled_completion_s
            winner_miner_id = attempt.miner_id
            winner_query_id = attempt.query_id
            break
        start_attempt(attempt.miner_id, attempt.scheduled_completion_s)

    partial_attempts_discarded = 0
    partial_elapsed_s = 0.0
    partial_remaining_s = 0.0
    if adoption_time_s is not None:
        for attempt in sorted(active.values(), key=lambda item: item.miner_id):
            elapsed = min(
                attempt.sampled_duration_s,
                max(0.0, adoption_time_s - attempt.started_at_s),
            )
            remaining = max(0.0, attempt.scheduled_completion_s - adoption_time_s)
            reason = (
                "adoption_preceded_tied_event"
                if attempt.scheduled_completion_s == adoption_time_s
                else "block_adopted"
            )
            is_tied_completion = reason == "adoption_preceded_tied_event"
            if is_tied_completion:
                tied_completions_cancelled += 1
            else:
                partial_attempts_discarded += 1
                partial_elapsed_s += elapsed
                partial_remaining_s += remaining
            if retain_details:
                attempt_records.append(
                    _attempt_record(
                        attempt,
                        run_id=run_id,
                        config_id=config_id,
                        run_seed=run_seed,
                        fee=fee,
                        status="cancelled",
                        elapsed_s=elapsed,
                        remaining_s=remaining,
                        coordinator_sequence=coordinator_sequence,
                        collision=None,
                        lottery_draw=None,
                        lottery_won=None,
                        partial=not is_tied_completion,
                        cancellation_reason=reason,
                    )
                )

    status = "adopted" if adoption_time_s is not None else "no_winner"
    run_record = {
        "schema_version": SCHEMA_VERSION,
        "simulation_mode": SIMULATION_MODE,
        "run_id": run_id,
        "config_id": config_id,
        "run_seed": run_seed,
        "status": status,
        "adoption_time_s": adoption_time_s,
        "adoption_coordinator_sequence": coordinator_sequence if adoption_time_s is not None else None,
        "winner_miner_id": winner_miner_id,
        "winner_query_id": winner_query_id,
        "completed_pairs": completed_count,
        "collision_count": collision_count,
        "collision_ratio": (
            collision_count / completed_count
            if adoption_time_s is not None and completed_count
            else None
        ),
        "partial_attempts_discarded": partial_attempts_discarded,
        "partial_elapsed_s": partial_elapsed_s,
        "partial_remaining_s": partial_remaining_s,
        "tied_completions_cancelled": tied_completions_cancelled,
        "unique_queries_completed": len(completion_counts),
        "per_miner_completed_query_ids": (
            {
                str(miner_id): query_ids
                for miner_id, query_ids in enumerate(completed_queries)
            }
            if completed_queries is not None
            else None
        ),
        "details_retained": retain_details,
        "nondeterminism_remaining": False,
    }
    return run_record, attempt_records


def summarize_adoption_times(values: list[float]) -> dict[str, float | int]:
    """Summarize calibration adoption times without collision outcomes."""
    if not values:
        return {"count": 0}
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    margin = 1.96 * stdev / math.sqrt(len(values))
    ordered = sorted(values)

    def percentile(probability: float) -> float:
        index = (len(ordered) - 1) * probability
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        fraction = index - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    return {
        "count": len(values),
        "mean_s": mean,
        "stdev_s": stdev,
        "min_s": ordered[0],
        "p05_s": percentile(0.05),
        "median_s": percentile(0.5),
        "p95_s": percentile(0.95),
        "max_s": ordered[-1],
        "mean_ci95_low_s": mean - margin,
        "mean_ci95_high_s": mean + margin,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _memory_total_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("ezkl", "numpy", "matplotlib", "pydantic", "cryptography"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def host_environment() -> dict[str, Any]:
    """Capture the hardware and software fields needed for reproduction."""
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": affinity,
        "memory_total_bytes": _memory_total_bytes(),
        "package_versions": _package_versions(),
    }


def model_proof_configuration(artifacts_dir: Path) -> dict[str, Any]:
    """Hash every model/proof artifact that can affect calibrated work."""
    resolved_artifacts_dir = artifacts_dir.resolve()
    try:
        recorded_artifacts_dir = resolved_artifacts_dir.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        recorded_artifacts_dir = resolved_artifacts_dir.as_posix()
    files: dict[str, dict[str, Any]] = {}
    for name in MODEL_PROOF_FILES:
        path = artifacts_dir / name
        if not path.exists():
            raise FileNotFoundError(f"required model/proof artifact is missing: {path}")
        files[name] = {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
    settings = json.loads((artifacts_dir / "settings.json").read_text())
    return {
        "artifacts_dir": recorded_artifacts_dir,
        "files": files,
        "settings": settings,
        "input_shape": [1, 2, 8, 8],
        "proof_system": "EZKL",
    }


def _artifact_identity(payload: dict[str, Any]) -> str:
    copy = dict(payload)
    copy.pop("artifact_id", None)
    return content_id(copy)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _summarize_samples(samples: list[float]) -> dict[str, float | int]:
    summary = summarize_adoption_times(samples)
    if summary["count"]:
        summary["mean_attempt_s"] = summary.pop("mean_s")
        summary["stdev_attempt_s"] = summary.pop("stdev_s")
    return summary


def _pin_calibration_resources(cpu_limit: int) -> list[int] | None:
    if cpu_limit <= 0:
        raise ValueError("cpu limit must be positive")
    selected: list[int] | None = None
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        available = sorted(os.sched_getaffinity(0))
        selected = available[: min(cpu_limit, len(available))]
        os.sched_setaffinity(0, selected)
    threads = str(len(selected) if selected else cpu_limit)
    for variable in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = threads
    return selected


def _timing_attempt_worker(
    conditioning: list[float],
    noise: list[float],
    artifacts_dir: str,
    input_shape: list[int],
    result_queue: Any,
) -> None:
    """Run one native EZKL attempt in a process the parent can terminate."""
    try:
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from poml_sim.zkp import run_inference_and_prove

        started = time.perf_counter()
        run_inference_and_prove(
            conditioning=conditioning,
            noise=noise,
            artifacts_dir=artifacts_dir,
            input_shape=input_shape,
        )
        result_queue.put(("ok", time.perf_counter() - started))
    except BaseException as error:
        result_queue.put(("error", repr(error)))


def _terminate_process(process: multiprocessing.Process) -> None:
    """Terminate a native-work child, escalating if it ignores SIGTERM."""
    if not process.is_alive():
        process.join(timeout=1.0)
        return
    process.terminate()
    process.join(timeout=5.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=2.0)


def _run_timing_attempt(
    *,
    attempt_number: int,
    total_attempts: int,
    conditioning: list[float],
    noise: list[float],
    artifacts_dir: Path,
) -> tuple[float | None, str | None]:
    """Run one timing attempt while keeping the parent responsive."""
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else None)
    result_queue = context.Queue()
    process = context.Process(
        target=_timing_attempt_worker,
        args=(conditioning, noise, str(artifacts_dir), [1, 2, 8, 8], result_queue),
        daemon=True,
    )
    process.start()
    started = time.perf_counter()
    next_heartbeat = started + TIMING_HEARTBEAT_S
    try:
        while True:
            try:
                result = result_queue.get(timeout=TIMING_POLL_S)
            except Empty:
                now = time.perf_counter()
                if now >= next_heartbeat:
                    print(
                        f"[timing] attempt {attempt_number}/{total_attempts} still running "
                        f"({now - started:.0f}s elapsed); press Ctrl+C to stop",
                        flush=True,
                    )
                    next_heartbeat = now + TIMING_HEARTBEAT_S
                if not process.is_alive():
                    return None, "timing worker exited without a result"
                continue
            if result and result[0] == "ok":
                return float(result[1]), None
            if result and result[0] == "error":
                return None, str(result[1])
            return None, f"unexpected timing worker result: {result!r}"
    except KeyboardInterrupt:
        print(
            f"[timing] stopping attempt {attempt_number}/{total_attempts}...",
            flush=True,
        )
        raise
    finally:
        _terminate_process(process)
        result_queue.close()
        result_queue.join_thread()


def _timing_partial_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".partial.json")


def _write_timing_checkpoint(
    path: Path,
    *,
    revision: dict[str, Any],
    attempts: int,
    samples: list[float],
    failures: list[dict[str, str | int]],
    model_configuration: dict[str, Any],
) -> None:
    _write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "ezkl_attempt_timing_checkpoint",
            "simulator_revision": revision,
            "samples_requested": attempts,
            "duration_samples_s": samples,
            "failed_attempts": failures,
            "model_proof_configuration": model_configuration,
            "updated_utc": _utc_now(),
        },
    )


def calibrate_attempt_timings(
    *,
    attempts: int,
    output_path: Path,
    artifacts_dir: Path,
    cpu_limit: int,
    resume: bool = False,
) -> dict[str, Any]:
    """Measure isolated real EZKL inference-plus-proof completion times."""
    if attempts <= 0:
        raise ValueError("attempt count must be positive")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite timing artifact: {output_path}")
    revision = _git_revision()
    if revision["dirty"]:
        raise RuntimeError("timing calibration requires a clean git worktree")
    partial_path = _timing_partial_path(output_path)
    selected_cpus = _pin_calibration_resources(cpu_limit)
    environment_before = host_environment()
    load_before = os.getloadavg() if hasattr(os, "getloadavg") else None
    model_configuration = model_proof_configuration(artifacts_dir)

    spatial = 8 * 8
    conditioning = [0.0] * spatial
    noise = [0.0] * spatial
    fixed_input_bytes = canonical_json(
        {"conditioning": conditioning, "noise": noise, "input_shape": [1, 2, 8, 8]}
    ).encode("ascii")
    samples: list[float] = []
    failures: list[dict[str, str | int]] = []
    if partial_path.exists():
        if not resume:
            raise FileExistsError(
                f"found interrupted calibration checkpoint: {partial_path}; "
                "rerun with --resume or use run-all"
            )
        checkpoint = _load_json(partial_path)
        if checkpoint.get("samples_requested") != attempts:
            raise ValueError("timing checkpoint was created for a different attempt count")
        if checkpoint.get("simulator_revision") != revision:
            raise ValueError("timing checkpoint revision does not match this checkout")
        if checkpoint.get("model_proof_configuration") != model_configuration:
            raise ValueError("timing checkpoint model/proof artifacts do not match")
        samples = [float(value) for value in checkpoint.get("duration_samples_s", [])]
        failures = list(checkpoint.get("failed_attempts", []))
        print(
            f"[timing] resuming checkpoint: {len(samples)}/{attempts} successful, "
            f"{len(failures)} failed attempts already recorded",
            flush=True,
        )

    attempt_number = len(samples) + len(failures)
    while len(samples) < attempts:
        attempt_number += 1
        print(
            f"[timing] starting attempt {attempt_number} "
            f"({len(samples)}/{attempts} successful)",
            flush=True,
        )
        try:
            duration, error = _run_timing_attempt(
                attempt_number=attempt_number,
                total_attempts=attempts,
                conditioning=conditioning,
                noise=noise,
                artifacts_dir=artifacts_dir,
            )
        except KeyboardInterrupt:
            _write_timing_checkpoint(
                partial_path,
                revision=revision,
                attempts=attempts,
                samples=samples,
                failures=failures,
                model_configuration=model_configuration,
            )
            print(
                f"[timing] interrupted; checkpoint saved to {partial_path}. "
                "Rerun with --resume to continue.",
                flush=True,
            )
            raise
        if error is not None:
            failures.append({"attempt_number": attempt_number, "error": error})
            print(f"[timing] attempt {attempt_number} failed: {error}", file=sys.stderr, flush=True)
        else:
            assert duration is not None
            samples.append(duration)
            print(
                f"[timing] completed {len(samples)}/{attempts} successful "
                f"({duration:.3f}s)",
                flush=True,
            )
        _write_timing_checkpoint(
            partial_path,
            revision=revision,
            attempts=attempts,
            samples=samples,
            failures=failures,
            model_configuration=model_configuration,
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "ezkl_attempt_timing_calibration",
        "created_utc": _utc_now(),
        "simulator_revision": revision,
        "samples_requested": attempts,
        "samples_successful": len(samples),
        "failed_attempts": failures,
        "duration_samples_s": samples,
        "statistics": _summarize_samples(samples),
        "resource_controls": {
            "requested_cpu_limit": cpu_limit,
            "selected_cpus": selected_cpus,
            "thread_environment": {
                variable: os.environ.get(variable)
                for variable in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "load_average_before": load_before,
            "load_average_after": os.getloadavg() if hasattr(os, "getloadavg") else None,
        },
        "environment_before": environment_before,
        "environment_after": host_environment(),
        "model_proof_configuration": model_configuration,
        "fixed_input": {
            "description": "64 zero conditioning values and 64 zero noise values",
            "sha256": hashlib.sha256(fixed_input_bytes).hexdigest(),
            "input_shape": [1, 2, 8, 8],
        },
        "nondeterminism_remaining": [
            "host background load and operating-system scheduling",
            "native EZKL backend execution",
            "wall-clock measurement noise",
        ],
    }
    payload["artifact_id"] = _artifact_identity(payload)
    _write_json(output_path, payload)
    partial_path.unlink(missing_ok=True)
    return payload


def _validate_timing_artifact(
    artifact: dict[str, Any], artifacts_dir: Path, *, strict_revision: bool
) -> None:
    if artifact.get("artifact_kind") != "ezkl_attempt_timing_calibration":
        raise ValueError("not an EZKL timing calibration artifact")
    if artifact.get("artifact_id") != _artifact_identity(artifact):
        raise ValueError("timing calibration artifact_id does not match its content")
    samples = artifact.get("duration_samples_s")
    if not isinstance(samples, list) or len(samples) < 1:
        raise ValueError("timing calibration has no duration samples")
    current_model = model_proof_configuration(artifacts_dir)
    if artifact.get("model_proof_configuration") != current_model:
        raise ValueError("timing calibration model/proof artifacts do not match this checkout")
    if strict_revision and artifact.get("simulator_revision") != _git_revision():
        raise ValueError("timing calibration simulator revision does not match this checkout")


def calibrate_difficulties(
    *,
    timing_path: Path,
    output_path: Path,
    seed_manifest_path: Path,
    artifacts_dir: Path,
    verification_runs: int,
) -> dict[str, Any]:
    """Calculate thresholds and verify adoption times without retuning them."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite difficulty artifact: {output_path}")
    timing = _load_json(timing_path)
    _validate_timing_artifact(timing, artifacts_dir, strict_revision=True)
    seed_manifest = _load_json(seed_manifest_path)
    miners_grid = seed_manifest["grid"]["miners"]
    target_grid = seed_manifest["grid"]["target_block_time_s"]
    calibration_root = seed_manifest["seed_roots"]["difficulty_calibration"]
    samples = [float(value) for value in timing["duration_samples_s"]]
    mean_attempt = statistics.fmean(samples)
    entries: list[dict[str, Any]] = []
    total_cells = len(miners_grid) * len(target_grid)
    cell_number = 0

    for miners in miners_grid:
        for target in target_grid:
            cell_number += 1
            difficulty = analytical_difficulty(mean_attempt, miners, target)
            adoption_times: list[float] = []
            failures = 0
            print(
                f"[difficulty] cell {cell_number}/{total_cells}: "
                f"M={miners}, target={target}s, running {verification_runs} races",
                flush=True,
            )
            for replicate in range(verification_runs):
                seed = derive_seed(
                    calibration_root,
                    "difficulty-calibration",
                    miners,
                    target,
                    replicate,
                )
                run, _ = simulate_race(
                    miners=miners,
                    query_pool_size=max(seed_manifest["grid"]["query_pool_size"]),
                    difficulty=difficulty,
                    duration_samples_s=samples,
                    run_seed=seed,
                    config_id=f"cal-M{miners}-T{target}",
                    run_id=f"cal-M{miners}-T{target}-R{replicate:04d}",
                    retain_details=False,
                )
                if run["status"] == "adopted":
                    adoption_times.append(float(run["adoption_time_s"]))
                else:
                    failures += 1
                progress_step = max(1, verification_runs // 10)
                if (replicate + 1) % progress_step == 0 or replicate + 1 == verification_runs:
                    print(
                        f"[difficulty] cell {cell_number}/{total_cells}: "
                        f"{replicate + 1}/{verification_runs} races complete",
                        flush=True,
                    )
            stats = summarize_adoption_times(adoption_times)
            entries.append(
                {
                    "miners": miners,
                    "target_block_time_s": target,
                    "difficulty_hex": f"0x{difficulty:064x}",
                    "per_attempt_win_probability": difficulty / UINT256_SIZE,
                    "derivation": "floor(2^256 * mean_attempt_s / (miners * target_s))",
                    "verification_runs": verification_runs,
                    "verification_failures": failures,
                    "realized_adoption_time": stats,
                    "relative_mean_error": (
                        (float(stats["mean_s"]) - target) / target if stats.get("count") else None
                    ),
                    "retuned_after_verification": False,
                }
            )
            print(
                f"[difficulty] M={miners} target={target}s "
                f"realized={stats.get('mean_s', float('nan')):.2f}s",
                flush=True,
            )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "analytical_difficulty_calibration",
        "created_utc": _utc_now(),
        "simulator_revision": _git_revision(),
        "timing_artifact_id": timing["artifact_id"],
        "timing_artifact_sha256": _sha256_file(timing_path),
        "seed_manifest_version": seed_manifest["version"],
        "seed_manifest_sha256": _sha256_file(seed_manifest_path),
        "verification_query_pool_size": max(seed_manifest["grid"]["query_pool_size"]),
        "verification_seed_root": calibration_root,
        "mean_attempt_s": mean_attempt,
        "entries": entries,
        "calibration_policy": "analytical threshold, verify adoption times, never retune",
        "collision_outcomes_inspected": False,
        "nondeterminism_remaining": False,
    }
    payload["artifact_id"] = _artifact_identity(payload)
    _write_json(output_path, payload)
    return payload


def _validate_difficulty_artifact(
    calibration: dict[str, Any],
    timing: dict[str, Any],
    timing_path: Path,
    seed_manifest: dict[str, Any],
    seed_manifest_path: Path,
) -> None:
    if calibration.get("artifact_kind") != "analytical_difficulty_calibration":
        raise ValueError("not an analytical difficulty calibration artifact")
    if calibration.get("artifact_id") != _artifact_identity(calibration):
        raise ValueError("difficulty artifact_id does not match its content")
    if calibration.get("simulator_revision") != _git_revision():
        raise ValueError("difficulty artifact simulator revision does not match this checkout")
    if calibration.get("timing_artifact_id") != timing.get("artifact_id"):
        raise ValueError("difficulty artifact references a different timing artifact")
    if calibration.get("timing_artifact_sha256") != _sha256_file(timing_path):
        raise ValueError("difficulty artifact timing digest does not match")
    if calibration.get("seed_manifest_version") != seed_manifest.get("version"):
        raise ValueError("difficulty artifact references a different seed manifest version")
    if calibration.get("seed_manifest_sha256") != _sha256_file(seed_manifest_path):
        raise ValueError("difficulty artifact seed manifest digest does not match")
    if calibration.get("verification_query_pool_size") != max(
        seed_manifest["grid"]["query_pool_size"]
    ):
        raise ValueError("difficulty artifact verification query pool does not match")


def _difficulty_lookup(calibration: dict[str, Any]) -> dict[tuple[int, float], int]:
    return {
        (int(entry["miners"]), float(entry["target_block_time_s"])): int(
            entry["difficulty_hex"], 16
        )
        for entry in calibration["entries"]
    }


def _validate_final_seed_manifest(seeds: dict[str, Any]) -> None:
    """Reject final campaigns that do not use the frozen experiment design."""
    expected_grid = {
        "miners": [10, 100, 1000],
        "target_block_time_s": [300, 600, 900],
        "query_pool_size": [1000, 5000, 10000],
    }
    if seeds.get("version") != "exp5-seeds-v1":
        raise ValueError("final campaign requires seed manifest version exp5-seeds-v1")
    if seeds.get("grid") != expected_grid:
        raise ValueError("final campaign requires the frozen 27-configuration grid")
    if seeds.get("uniform_fee") != 1:
        raise ValueError("final campaign requires uniform fee=1")
    if seeds.get("final_replicates_per_config") != 1000:
        raise ValueError("final campaign requires 1,000 replicates per configuration")
    if seeds.get("detailed_replicates_per_config") != 5:
        raise ValueError("final campaign requires five retained detailed replicates")
    if seeds.get("frozen_before_final_outcomes") is not True:
        raise ValueError("final seed manifest must declare frozen_before_final_outcomes=true")
    if seeds.get("final_seed_namespace") != "final-run":
        raise ValueError("final campaign requires the frozen final-run seed namespace")
    if seeds.get("final_seed_parts") != [
        "miners",
        "target_block_time_s",
        "query_pool_size",
        "zero_based_replicate",
    ]:
        raise ValueError("final campaign requires the frozen final seed part order")

    root = seeds["seed_roots"]["final"]
    derived = [
        derive_seed(root, "final-run", miners, target, query_pool_size, replicate)
        for miners in expected_grid["miners"]
        for target in expected_grid["target_block_time_s"]
        for query_pool_size in expected_grid["query_pool_size"]
        for replicate in range(1000)
    ]
    if len(set(derived)) != len(derived):
        raise ValueError("frozen final seed derivation contains duplicate seeds")


def _prepare_new_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _mean_stdev(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def run_final_campaign(
    *,
    timing_path: Path,
    calibration_path: Path,
    seed_manifest_path: Path,
    artifacts_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the frozen 27-configuration final campaign."""
    _prepare_new_output_dir(output_dir)
    timing = _load_json(timing_path)
    calibration = _load_json(calibration_path)
    seeds = _load_json(seed_manifest_path)
    _validate_final_seed_manifest(seeds)
    _validate_timing_artifact(timing, artifacts_dir, strict_revision=True)
    _validate_difficulty_artifact(
        calibration,
        timing,
        timing_path,
        seeds,
        seed_manifest_path,
    )

    durations = [float(value) for value in timing["duration_samples_s"]]
    difficulties = _difficulty_lookup(calibration)
    final_root = seeds["seed_roots"]["final"]
    replicates = int(seeds["final_replicates_per_config"])
    details_replicates = int(seeds["detailed_replicates_per_config"])
    timing_digest = _sha256_file(timing_path)
    calibration_digest = _sha256_file(calibration_path)
    revision = _git_revision()
    provenance = {
        "simulator_revision": revision,
        "environment": host_environment(),
        "model_proof_configuration": timing["model_proof_configuration"],
        "timing_artifact": {
            "path": str(timing_path),
            "artifact_id": timing["artifact_id"],
            "sha256": timing_digest,
        },
        "difficulty_artifact": {
            "path": str(calibration_path),
            "artifact_id": calibration["artifact_id"],
            "sha256": calibration_digest,
        },
        "seed_manifest": {
            "path": str(seed_manifest_path),
            "version": seeds["version"],
            "sha256": _sha256_file(seed_manifest_path),
        },
        "nondeterminism_remaining": False,
    }

    run_fields = (
        "run_id",
        "config_id",
        "replicate",
        "run_seed",
        "miners",
        "target_block_time_s",
        "query_pool_size",
        "fee",
        "difficulty_hex",
        "status",
        "adoption_time_s",
        "adoption_coordinator_sequence",
        "winner_miner_id",
        "winner_query_id",
        "completed_pairs",
        "collision_count",
        "collision_ratio",
        "partial_attempts_discarded",
        "partial_elapsed_s",
        "partial_remaining_s",
        "tied_completions_cancelled",
        "unique_queries_completed",
        "details_retained",
    )
    summary_fields = (
        "config_id",
        "miners",
        "target_block_time_s",
        "query_pool_size",
        "fee",
        "difficulty_hex",
        "runs_requested",
        "runs_adopted",
        "runs_failed",
        "mean_collision_ratio",
        "stdev_collision_ratio",
        "pooled_collision_ratio",
        "mean_adoption_time_s",
        "stdev_adoption_time_s",
        "mean_completed_pairs",
        "mean_partial_attempts_discarded",
        "mean_tied_completions_cancelled",
    )
    run_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    configurations: dict[str, dict[str, Any]] = {}
    any_failures = False

    for miners in seeds["grid"]["miners"]:
        for target in seeds["grid"]["target_block_time_s"]:
            difficulty = difficulties[(int(miners), float(target))]
            for query_pool_size in seeds["grid"]["query_pool_size"]:
                config_number = len(configurations) + 1
                total_configurations = (
                    len(seeds["grid"]["miners"])
                    * len(seeds["grid"]["target_block_time_s"])
                    * len(seeds["grid"]["query_pool_size"])
                )
                print(
                    f"[final] config {config_number}/{total_configurations}: "
                    f"M={miners}, target={target}s, Q={query_pool_size}; "
                    f"running {replicates} races",
                    flush=True,
                )
                config = {
                    "schema_version": SCHEMA_VERSION,
                    "simulation_mode": SIMULATION_MODE,
                    "miners": miners,
                    "target_block_time_s": target,
                    "query_pool_size": query_pool_size,
                    "uniform_fee": 1,
                    "outputs_per_query": 1,
                    "query_input_policy": "equivalent fixed input; identity cannot affect sampled cost",
                    "selection_policy": "independent uniform permutation without replacement per miner",
                    "cross_miner_selection": "independent; duplicate query selections allowed",
                    "lottery_policy": "run-seeded uniform 256-bit draw compared with fixed threshold",
                    "difficulty_hex": f"0x{difficulty:064x}",
                    "adoption_policy": "immediate coordinator adoption of first valid child",
                    "event_order_policy": "virtual time, then run-seeded tie key, then insertion sequence",
                    "partial_policy": "count only attempts started and unfinished at adoption",
                    "collision_policy": "completed query occurrences after the first coordinator-ordered completion",
                    "duration_sampling": "iid empirical draws with replacement",
                    "timing_artifact_sha256": timing_digest,
                    "difficulty_artifact_sha256": calibration_digest,
                    "seed_manifest_version": seeds["version"],
                    "replicates": replicates,
                    "detailed_replicates": details_replicates,
                }
                config_id = content_id(config)
                configurations[config_id] = config
                detail_dir = output_dir / "details" / config_id
                detail_runs: list[dict[str, Any]] = []
                detail_attempts: list[dict[str, Any]] = []
                config_runs: list[dict[str, Any]] = []

                for replicate in range(replicates):
                    run_seed = derive_seed(
                        final_root,
                        "final-run",
                        miners,
                        target,
                        query_pool_size,
                        replicate,
                    )
                    run_id = content_id(
                        {"config_id": config_id, "replicate": replicate, "seed": run_seed}
                    )
                    retain_details = replicate < details_replicates
                    run, attempts = simulate_race(
                        miners=int(miners),
                        query_pool_size=int(query_pool_size),
                        difficulty=difficulty,
                        duration_samples_s=durations,
                        run_seed=run_seed,
                        config_id=config_id,
                        run_id=run_id,
                        fee=1,
                        retain_details=retain_details,
                    )
                    run["replicate"] = replicate
                    run["miners"] = miners
                    run["target_block_time_s"] = target
                    run["query_pool_size"] = query_pool_size
                    run["fee"] = 1
                    run["difficulty_hex"] = f"0x{difficulty:064x}"
                    config_runs.append(run)
                    run_rows.append({field: run.get(field) for field in run_fields})
                    if retain_details:
                        detail_runs.append(run)
                        detail_attempts.extend(attempts)
                    if run["status"] != "adopted":
                        any_failures = True
                    progress_step = max(1, replicates // 10)
                    if (replicate + 1) % progress_step == 0 or replicate + 1 == replicates:
                        print(
                            f"[final] config {config_number}/{total_configurations}: "
                            f"{replicate + 1}/{replicates} races complete",
                            flush=True,
                        )

                detail_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    detail_dir / "runs.json",
                    {
                        "config": config,
                        "config_id": config_id,
                        "provenance": provenance,
                        "retention_policy": (
                            f"full run and attempt details retained for replicates "
                            f"0..{details_replicates - 1}; all {replicates} run summaries are in runs.csv"
                        ),
                        "runs": detail_runs,
                    },
                )
                with (detail_dir / "attempts.csv").open("w", newline="") as file_handle:
                    writer = csv.DictWriter(file_handle, fieldnames=ATTEMPT_FIELDS)
                    writer.writeheader()
                    writer.writerows(detail_attempts)

                adopted = [run for run in config_runs if run["status"] == "adopted"]
                ratios = [float(run["collision_ratio"]) for run in adopted]
                adoption_times = [float(run["adoption_time_s"]) for run in adopted]
                completed = [float(run["completed_pairs"]) for run in adopted]
                partials = [float(run["partial_attempts_discarded"]) for run in adopted]
                mean_ratio, stdev_ratio = _mean_stdev(ratios) if ratios else (float("nan"), float("nan"))
                mean_adoption, stdev_adoption = (
                    _mean_stdev(adoption_times) if adoption_times else (float("nan"), float("nan"))
                )
                total_completed = sum(int(run["completed_pairs"]) for run in adopted)
                summary_rows.append(
                    {
                        "config_id": config_id,
                        "miners": miners,
                        "target_block_time_s": target,
                        "query_pool_size": query_pool_size,
                        "fee": 1,
                        "difficulty_hex": f"0x{difficulty:064x}",
                        "runs_requested": replicates,
                        "runs_adopted": len(adopted),
                        "runs_failed": replicates - len(adopted),
                        "mean_collision_ratio": mean_ratio,
                        "stdev_collision_ratio": stdev_ratio,
                        "pooled_collision_ratio": (
                            sum(int(run["collision_count"]) for run in adopted) / total_completed
                            if total_completed
                            else float("nan")
                        ),
                        "mean_adoption_time_s": mean_adoption,
                        "stdev_adoption_time_s": stdev_adoption,
                        "mean_completed_pairs": statistics.fmean(completed) if completed else float("nan"),
                        "mean_partial_attempts_discarded": (
                            statistics.fmean(partials) if partials else float("nan")
                        ),
                        "mean_tied_completions_cancelled": (
                            statistics.fmean(
                                [float(run["tied_completions_cancelled"]) for run in adopted]
                            )
                            if adopted
                            else float("nan")
                        ),
                    }
                )
                print(
                    f"[final] M={miners} target={target}s Q={query_pool_size} "
                    f"mean_collision={mean_ratio:.6f}",
                    flush=True,
                )

    with (output_dir / "runs.csv").open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(run_rows)
    with (output_dir / "summary.csv").open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    campaign = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "uniform_fee_collision_campaign",
        "created_utc": _utc_now(),
        "campaign_status": "failed" if any_failures else "complete",
        "provenance": provenance,
        "configurations": configurations,
        "run_count": len(run_rows),
        "summary_count": len(summary_rows),
        "detail_retention": {
            "replicates_per_config": details_replicates,
            "reason": "user-selected storage policy for the 1000-replicate campaign",
        },
    }
    campaign["campaign_id"] = _artifact_identity(campaign)
    _write_json(output_dir / "campaign.json", campaign)
    generate_heatmaps(output_dir / "summary.csv", output_dir / "figures")
    if any_failures:
        raise RuntimeError("one or more frozen final runs exhausted all queries without adoption")
    return campaign


def generate_heatmaps(summary_path: Path, output_dir: Path) -> list[Path]:
    """Create one shared-scale collision heatmap per target block time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with summary_path.open(newline="") as file_handle:
        rows = list(csv.DictReader(file_handle))
    if not rows:
        raise ValueError(f"summary has no rows: {summary_path}")
    miners_values = sorted({int(row["miners"]) for row in rows})
    query_values = sorted({int(row["query_pool_size"]) for row in rows})
    target_values = sorted({float(row["target_block_time_s"]) for row in rows})
    run_counts = {int(row["runs_requested"]) for row in rows}
    run_count_label = (
        f" ({next(iter(run_counts)):,} runs)" if len(run_counts) == 1 else ""
    )
    values = [
        float(row["mean_collision_ratio"])
        for row in rows
        if math.isfinite(float(row["mean_collision_ratio"]))
    ]
    global_max = max(values) if values else 0.0
    color_max = global_max if global_max > 0 else 1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for target in target_values:
        matrix = np.full((len(miners_values), len(query_values)), np.nan)
        for row in rows:
            if float(row["target_block_time_s"]) != target:
                continue
            row_index = miners_values.index(int(row["miners"]))
            column_index = query_values.index(int(row["query_pool_size"]))
            matrix[row_index, column_index] = float(row["mean_collision_ratio"])

        figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=color_max, aspect="auto")
        axis.set_xticks(range(len(query_values)), [f"{value:,}" for value in query_values])
        axis.set_yticks(range(len(miners_values)), [str(value) for value in miners_values])
        axis.set_xlabel("Query pool size Q")
        axis.set_ylabel("Miner count M")
        axis.set_title(f"Uniform-fee collision ratio, target {target:g} s")
        for row_index in range(len(miners_values)):
            for column_index in range(len(query_values)):
                value = matrix[row_index, column_index]
                if not math.isnan(value):
                    text_color = "white" if value > color_max * 0.55 else "black"
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color=text_color,
                    )
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label(f"Mean collision ratio{run_count_label}")
        stem = f"collision_heatmap_target_{target:g}s"
        for extension in ("png", "pdf"):
            path = output_dir / f"{stem}.{extension}"
            figure.savefig(path, dpi=200)
            outputs.append(path)
        plt.close(figure)
    return outputs


def run_smoke(output_dir: Path) -> dict[str, Any]:
    """Run a synthetic deterministic smoke check without real EZKL timings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = [1.0, 1.25, 1.5]
    miners = 4
    target = 3.0
    query_pool_size = 20
    difficulty = analytical_difficulty(statistics.fmean(samples), miners, target)
    rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    smoke_root = "poml-exp5-smoke-v1"
    for replicate in range(5):
        seed = derive_seed(smoke_root, "smoke-run", replicate)
        run, run_attempts = simulate_race(
            miners=miners,
            query_pool_size=query_pool_size,
            difficulty=difficulty,
            duration_samples_s=samples,
            run_seed=seed,
            config_id="synthetic-smoke",
            run_id=f"smoke-{replicate}",
        )
        if run["status"] != "adopted":
            raise RuntimeError("synthetic smoke race did not adopt a block")
        if run["collision_count"] != run["completed_pairs"] - run["unique_queries_completed"]:
            raise AssertionError("collision metric invariant failed")
        rows.append(run)
        attempts.extend(run_attempts)
    payload = {
        "synthetic_fixture": True,
        "not_valid_for_final_results": True,
        "duration_samples_s": samples,
        "difficulty_hex": f"0x{difficulty:064x}",
        "runs": rows,
    }
    _write_json(output_dir / "smoke.json", payload)
    with (output_dir / "attempts.csv").open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=ATTEMPT_FIELDS)
        writer.writeheader()
        writer.writerows(attempts)
    return payload


def run_all(
    *,
    work_dir: Path,
    artifacts_dir: Path,
    attempts: int,
    cpu_limit: int,
    verification_runs: int,
    final_output_dir: Path,
) -> dict[str, Any]:
    """Run timing, difficulty, and final collection as one resumable workflow."""
    work_dir.mkdir(parents=True, exist_ok=True)
    timing_path = work_dir / "timing_calibration.json"
    difficulty_path = work_dir / "difficulty_calibration.json"

    print("[run-all] Stage 1/3: timing calibration", flush=True)
    if timing_path.exists():
        timing = _load_json(timing_path)
        _validate_timing_artifact(timing, artifacts_dir, strict_revision=True)
        print(
            f"[run-all] timing artifact already valid; reusing "
            f"{len(timing['duration_samples_s'])}/{timing['samples_requested']} samples",
            flush=True,
        )
    else:
        timing = calibrate_attempt_timings(
            attempts=attempts,
            output_path=timing_path,
            artifacts_dir=artifacts_dir,
            cpu_limit=cpu_limit,
            resume=True,
        )
    print("[run-all] Stage 1/3 complete", flush=True)

    print("[run-all] Stage 2/3: difficulty calibration", flush=True)
    if difficulty_path.exists():
        difficulty = _load_json(difficulty_path)
        seed_manifest = _load_json(DEFAULT_SEED_MANIFEST)
        _validate_difficulty_artifact(
            difficulty,
            timing,
            timing_path,
            seed_manifest,
            DEFAULT_SEED_MANIFEST,
        )
        print("[run-all] difficulty artifact already valid; reusing", flush=True)
    else:
        difficulty = calibrate_difficulties(
            timing_path=timing_path,
            output_path=difficulty_path,
            seed_manifest_path=DEFAULT_SEED_MANIFEST,
            artifacts_dir=artifacts_dir,
            verification_runs=verification_runs,
        )
    print("[run-all] Stage 2/3 complete", flush=True)

    print("[run-all] Stage 3/3: final collision campaign", flush=True)
    campaign = run_final_campaign(
        timing_path=timing_path,
        calibration_path=difficulty_path,
        seed_manifest_path=DEFAULT_SEED_MANIFEST,
        artifacts_dir=artifacts_dir,
        output_dir=final_output_dir,
    )
    print(
        f"[run-all] Stage 3/3 complete: {campaign['run_count']:,} runs, "
        f"{campaign['summary_count']} configurations",
        flush=True,
    )
    print(f"[run-all] outputs: {final_output_dir}", flush=True)
    return campaign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    timing = subparsers.add_parser("calibrate-timings", help="Measure 30 isolated real EZKL attempts")
    timing.add_argument("--attempts", type=int, default=30)
    timing.add_argument("--cpu-limit", type=int, default=16)
    timing.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "model")
    timing.add_argument("--output", type=Path, required=True)
    timing.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <output>.partial.json after Ctrl+C",
    )

    difficulty = subparsers.add_parser(
        "calibrate-difficulty", help="Derive and verify one threshold per (M, target)"
    )
    difficulty.add_argument("--timings", type=Path, required=True)
    difficulty.add_argument("--output", type=Path, required=True)
    difficulty.add_argument("--verification-runs", type=int, default=1000)
    difficulty.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEED_MANIFEST)
    difficulty.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "model")

    final = subparsers.add_parser("run-final", help="Run the frozen 27,000-race campaign")
    final.add_argument("--timings", type=Path, required=True)
    final.add_argument("--calibration", type=Path, required=True)
    final.add_argument("--output-dir", type=Path, required=True)
    final.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEED_MANIFEST)
    final.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "model")

    run_all_parser = subparsers.add_parser(
        "run-all",
        help="Run timing calibration, difficulty calibration, and final collection",
    )
    run_all_parser.add_argument(
        "--work-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "exp5_run_all",
        help="Directory for reusable timing and difficulty artifacts",
    )
    run_all_parser.add_argument("--attempts", type=int, default=30)
    run_all_parser.add_argument("--cpu-limit", type=int, default=16)
    run_all_parser.add_argument("--verification-runs", type=int, default=1000)
    run_all_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "exp5_final_2026_08_12",
        help="New directory for the final campaign outputs",
    )
    run_all_parser.add_argument(
        "--artifacts-dir", type=Path, default=PROJECT_ROOT / "model"
    )

    plot = subparsers.add_parser("plot", help="Regenerate heatmaps from a summary CSV")
    plot.add_argument("--summary", type=Path, required=True)
    plot.add_argument("--output-dir", type=Path, required=True)

    smoke = subparsers.add_parser("smoke", help="Run a synthetic no-EZKL smoke check")
    smoke.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "exp5_smoke",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "calibrate-timings":
        calibrate_attempt_timings(
            attempts=args.attempts,
            output_path=args.output,
            artifacts_dir=args.artifacts_dir,
            cpu_limit=args.cpu_limit,
            resume=args.resume,
        )
    elif args.command == "calibrate-difficulty":
        calibrate_difficulties(
            timing_path=args.timings,
            output_path=args.output,
            seed_manifest_path=args.seed_manifest,
            artifacts_dir=args.artifacts_dir,
            verification_runs=args.verification_runs,
        )
    elif args.command == "run-final":
        run_final_campaign(
            timing_path=args.timings,
            calibration_path=args.calibration,
            seed_manifest_path=args.seed_manifest,
            artifacts_dir=args.artifacts_dir,
            output_dir=args.output_dir,
        )
    elif args.command == "run-all":
        run_all(
            work_dir=args.work_dir,
            artifacts_dir=args.artifacts_dir,
            attempts=args.attempts,
            cpu_limit=args.cpu_limit,
            verification_runs=args.verification_runs,
            final_output_dir=args.output_dir,
        )
    elif args.command == "plot":
        generate_heatmaps(args.summary, args.output_dir)
    elif args.command == "smoke":
        run_smoke(args.output_dir)


if __name__ == "__main__":
    main()
