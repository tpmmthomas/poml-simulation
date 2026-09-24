"""Exclusive component timing and useful-work accounting for fresh PoML responses.

The complete online elapsed time is retained, including artifact I/O. Nested
timers partition time rather than charging a primitive and its caller twice.
"""

from collections import defaultdict
from contextlib import contextmanager, ExitStack
from functools import wraps
import math
import time
from unittest.mock import patch

from . import backends, lottery, system as protocol


class ComponentTimer:
    """Partition one sequential execution into exclusive nanosecond intervals."""

    def __init__(self, clock=time.perf_counter_ns):
        self.clock = clock
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.stack = []

    @contextmanager
    def section(self, name):
        """Charge nested work to its own category, including failed calls."""
        frame = [self.clock(), 0]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = self.clock() - frame[0]
            self.stack.pop()
            self.seconds[name] += (elapsed - frame[1]) / 1e9
            self.calls[name] += 1
            if self.stack:
                self.stack[-1][1] += elapsed

    def wrap(self, function, name):
        """Time an existing function without changing its return value."""

        @wraps(function)
        def measured(*args, **kwargs):
            with self.section(name):
                return function(*args, **kwargs)

        return measured


def measure_response(system, query, binding):
    """Time fresh production, one response validation and one single-pair lottery.

    Function patches are process-local: run one serialized worker, never threads.
    The underlying model proof is verified exactly once by the real backend.
    """
    timer = ComponentTimer()
    backend = system.backend
    targets = [
        (backend, "run", "adapter_other"),
        (backend.prover, "run", "worker"),
        (backend, "complexity", "complexity"),
        (protocol, "vrf_eval", "vrf_generation"),
        (protocol, "vrf_verify", "vrf_verification"),
        (protocol, "encrypt_output", "encryption"),
        (protocol, "sign", "signatures"),
        (protocol, "verify_signature", "signatures"),
        (protocol, "public_key", "key_derivation"),
        (backends, "gaussian_vector", "randomness_expansion"),
        (lottery, "complexity_threshold", "threshold"),
    ]
    for module in (protocol, backends, lottery):
        targets.extend(
            (module, name, category)
            for name, category in (("canonical", "serialization"), ("sha256", "hashing"))
        )
    with ExitStack() as stack:
        for owner, name, category in targets:
            stack.enter_context(
                patch.object(owner, name, timer.wrap(getattr(owner, name), category))
            )
        with timer.section("protocol_other"):
            response, _ = system.execute(query, 0, binding)
            system.validate_response(response, system.state, system.height + 1)
            value, won = lottery.evaluate_lottery(
                binding, (response.ciphertext,), system.difficulty, response.complexity
            )
    record = system.executions[-1].copy()
    if record.get("verified") is not True:
        raise ValueError("a fresh verified model proof is required")
    useful = sum(
        record[name] for name in ("inference_seconds", "proof_seconds", "verification_seconds")
    )
    if not math.isfinite(useful) or useful <= 0:
        raise ValueError("invalid prover timings")
    elapsed = sum(timer.seconds.values())
    worker_elapsed = timer.seconds.pop("worker")
    worker_extra = worker_elapsed - useful
    if worker_extra < 0:
        raise ValueError("worker elapsed time is less than its component times")
    timer.seconds["inference"] = record["inference_seconds"]
    timer.seconds["proof_generation"] = record["proof_seconds"]
    timer.seconds["proof_verification"] = record["verification_seconds"]
    timer.seconds["worker_serialization_io_other"] = worker_extra
    record.update(
        elapsed_seconds=elapsed,
        useful_seconds=useful,
        auxiliary_seconds=elapsed - useful,
        components_seconds=dict(timer.seconds),
        component_calls=dict(timer.calls),
        lottery_value=value.hex(),
        lottery_won=won,
        ciphertext_bytes=len(response.ciphertext),
        query_id=query.qid.hex(),
        binding=binding.hex(),
    )
    if not math.isclose(sum(timer.seconds.values()), elapsed, abs_tol=1e-8):
        raise ValueError("timing components do not sum to the elapsed interval")
    return record


def summarize_records(records, *, bootstrap=10000, seed=42):
    """Report ratios of sums and prompt-cluster bootstrap intervals by length.

    The worker's inference timer includes witness reconstruction. Its ratio is
    therefore an inference-and-witness sensitivity, not a bare-inference baseline.
    """
    import numpy as np

    if not records or bootstrap < 1:
        raise ValueError("records and positive bootstrap count required")
    fields = ("elapsed_seconds", "useful_seconds", "inference_seconds")
    values = np.array([[row[key] for key in fields] for row in records], dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("timings must be finite and positive")
    if (values[:, 0] < values[:, 1]).any() or (values[:, 1] < values[:, 2]).any():
        raise ValueError("require elapsed >= useful >= inference")
    if any(row.get("verified") is not True for row in records):
        raise ValueError("unverified attempt in summary")
    for row in records:
        times = row["components_seconds"]
        if any(not math.isfinite(value) or value < 0 for value in times.values()):
            raise ValueError("invalid component timing")
        if not math.isclose(sum(times.values()), row["elapsed_seconds"], abs_tol=1e-8):
            raise ValueError("components do not partition elapsed time")
        core = row["inference_seconds"] + row["proof_seconds"] + row["verification_seconds"]
        if not math.isclose(core, row["useful_seconds"], abs_tol=1e-8):
            raise ValueError("useful time disagrees with prover components")
        if not math.isclose(row["auxiliary_seconds"] + core, row["elapsed_seconds"], abs_tol=1e-8):
            raise ValueError("auxiliary time disagrees with elapsed time")

    def ratios(totals):
        elapsed, useful, inference = totals.T
        return {
            "alpha_cert": elapsed / useful,
            "useful_fraction": useful / elapsed,
            "auxiliary_fraction": 1 - useful / elapsed,
            "alpha_inference_and_witness": elapsed / inference,
            "inference_and_witness_fraction": inference / elapsed,
        }

    total = values.sum(axis=0)
    rng = np.random.default_rng(seed)
    resampled = np.zeros((bootstrap, 3))
    lengths = np.array([r["prompt_length"] for r in records])
    cluster_count = 0
    for length in sorted(set(lengths)):
        clusters = {}
        for index in np.flatnonzero(lengths == length):
            # Repeated prompts across workers are one sampling unit, not new inputs.
            key = records[index].get("prompt_sha256", str(index))
            clusters[key] = clusters.get(key, np.zeros(3)) + values[index]
        group = np.array(list(clusters.values()))
        cluster_count += len(group)
        indices = rng.integers(len(group), size=(bootstrap, len(group)))
        resampled += group[indices].sum(axis=1)
    confidence = {
        key: np.quantile(val, [0.025, 0.975]).tolist() for key, val in ratios(resampled).items()
    }

    def distribution(data):
        a = np.asarray(data, dtype=float)
        return dict(
            mean=float(a.mean()),
            median=float(np.median(a)),
            q25=float(np.quantile(a, 0.25)),
            q75=float(np.quantile(a, 0.75)),
            p95=float(np.quantile(a, 0.95)),
            total=float(a.sum()),
        )

    timing_fields = (*fields, "proof_seconds", "verification_seconds", "auxiliary_seconds")
    components = sorted(set().union(*(r["components_seconds"] for r in records)))
    return {
        "count": len(records),
        "ratios": {key: float(val) for key, val in ratios(total).items()},
        "ci95": confidence,
        "bootstrap": {
            "resamples": bootstrap,
            "seed": seed,
            "strata": "prompt_length",
            "unit": "prompt_sha256 cluster",
            "clusters": cluster_count,
        },
        "timings_seconds": {key: distribution([r[key] for r in records]) for key in timing_fields},
        "components_seconds": {
            key: distribution([r["components_seconds"].get(key, 0) for r in records])
            for key in components
        },
        "prompt_lengths": {str(n): int(sum(lengths == n)) for n in sorted(set(lengths))},
        "output_lengths": distribution([r["output_length"] for r in records]),
    }
