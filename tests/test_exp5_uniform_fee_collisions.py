from __future__ import annotations

import random
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp5_uniform_fee_collisions import (
    LazyPermutation,
    analytical_difficulty,
    derive_seed,
    simulate_race,
    _validate_final_seed_manifest,
    _run_timing_attempt,
)


def _race(**overrides):
    arguments = {
        "miners": 4,
        "query_pool_size": 20,
        "difficulty": 1 << 252,
        "duration_samples_s": [1.0, 1.25, 1.5],
        "run_seed": 42,
        "config_id": "config",
        "run_id": "run",
    }
    arguments.update(overrides)
    return simulate_race(**arguments)


def test_lazy_permutation_draws_every_value_once():
    permutation = LazyPermutation(100, random.Random(7))

    values = [permutation.draw() for _ in range(100)]

    assert sorted(values) == list(range(100))
    assert permutation.draw() is None


def test_analytical_difficulty_uses_aggregate_attempt_rate():
    assert analytical_difficulty(2.0, miners=2, target_seconds=4.0) == 1 << 254


def test_same_seed_reproduces_run_and_attempt_records():
    first = _race()
    second = _race()

    assert first == second


def test_collision_count_is_every_completion_after_first_per_query():
    run, _ = _race(miners=3, query_pool_size=4, difficulty=1, duration_samples_s=[1.0])

    assert run["status"] == "no_winner"
    assert run["completed_pairs"] == 12
    assert run["unique_queries_completed"] == 4
    assert run["collision_count"] == 8
    assert run["collision_ratio"] is None
    assert run["partial_attempts_discarded"] == 0


def test_equal_time_events_after_adoption_are_cancelled():
    run, attempts = _race(difficulty=(1 << 256) - 1, duration_samples_s=[1.0])
    cancelled = [attempt for attempt in attempts if attempt["status"] == "cancelled"]

    assert run["status"] == "adopted"
    assert run["completed_pairs"] == 1
    assert run["partial_attempts_discarded"] == 0
    assert run["tied_completions_cancelled"] == 3
    assert len(cancelled) == 3
    assert all(attempt["remaining_s"] == 0.0 for attempt in cancelled)
    assert all(not attempt["partial"] for attempt in cancelled)
    assert all(
        attempt["cancellation_reason"] == "adoption_preceded_tied_event"
        for attempt in cancelled
    )


def test_summary_mode_preserves_every_run_level_metric():
    detailed, attempts = _race()
    summary, summary_attempts = _race(retain_details=False)

    assert attempts
    assert summary_attempts == []
    assert summary["per_miner_completed_query_ids"] is None
    assert not summary["details_retained"]
    for key, value in detailed.items():
        if key not in {"per_miner_completed_query_ids", "details_retained"}:
            assert summary[key] == value


def test_domain_separation_keeps_random_streams_distinct():
    root = 1234

    assert derive_seed(root, "duration", 0) != derive_seed(root, "lottery", 0)
    assert derive_seed(root, "duration", 0) != derive_seed(root, "duration", 1)


def test_checked_in_manifest_freezes_all_final_seeds():
    manifest_path = PROJECT_ROOT / "experiments" / "exp5_seed_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    _validate_final_seed_manifest(manifest)


def test_interruptible_timing_attempt_returns_child_result(monkeypatch):
    import experiments.exp5_uniform_fee_collisions as experiment

    def fake_worker(conditioning, noise, artifacts_dir, input_shape, result_queue):
        del conditioning, noise, artifacts_dir, input_shape
        time.sleep(0.01)
        result_queue.put(("ok", 0.25))

    monkeypatch.setattr(experiment, "_timing_attempt_worker", fake_worker)

    duration, error = _run_timing_attempt(
        attempt_number=1,
        total_attempts=1,
        conditioning=[0.0] * 64,
        noise=[0.0] * 64,
        artifacts_dir=PROJECT_ROOT / "model",
    )

    assert duration == 0.25
    assert error is None