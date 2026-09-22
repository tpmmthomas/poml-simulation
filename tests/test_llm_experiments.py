"""Regression tests for response-aware Experiment 3 and archived-run replay."""

import argparse
import csv
import json
import hashlib
from pathlib import Path
import statistics
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import run_llm_experiments as experiment  # noqa: E402
from poml_sim.llm_simulation import QuerySpec, QueryTrace, simulate_race  # noqa: E402
from poml_sim.llm_benchmark import (  # noqa: E402
    BENCHMARK_SCHEMA,
    profile_pool,
    challenge_seed,
    token_digest,
)


@pytest.fixture
def benchmark_bank(tmp_path):
    """Small synthetic measurement fixture with the benchmark's strict schema."""
    root = tmp_path / "bank"
    root.mkdir()
    schedule = experiment._schedule(experiment.DEFAULT_SCHEDULE)
    prompts = [
        dict(
            query_id=f"b{i}",
            prompt_tokens=[i + 1] * n,
            prompt_sha256=token_digest([i + 1] * n),
            prompt_length=n,
            max_output_length=4,
        )
        for i, n in enumerate((2, 4))
    ]
    phases = {}
    for phase in ("profile", "evaluation"):
        rows = [
            dict(
                query_id=p["query_id"],
                prompt_length=p["prompt_length"],
                prompt_sha256=p["prompt_sha256"],
                phase=phase,
                replicate=j,
                challenge_seed=challenge_seed(1, phase, p["query_id"], j),
                output_length=k,
                gpt2_duration=0.1,
                stop_reason="eos",
            )
            for p, lengths in zip(prompts, ((1, 2), (3, 4)))
            for j, k in enumerate(lengths)
        ]
        phases[phase] = rows
        (root / f"{phase}_rollouts.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
    pool = profile_pool(prompts, phases["profile"])
    traces = []
    for i, row in enumerate(phases["evaluation"]):
        query = QuerySpec(row["query_id"], row["prompt_length"], 4)
        traces.append(
            {
                **row,
                "complexity": experiment.complexity_for(
                    query, row["output_length"], schedule
                ),
                "duration": 0.1 * row["output_length"],
                "proof_directory": "fixture",
                "proof_trial": i,
            }
        )
    (root / "traces.jsonl").write_text("\n".join(json.dumps(r) for r in traces) + "\n")
    (root / "prompts.json").write_text(json.dumps({"prompts": prompts}))
    metadata = dict(
        schema=BENCHMARK_SCHEMA,
        pool=pool,
        max_output=4,
        backend="cuda",
        duration_scope="DeepProve fixture",
        proof_timing_reuse=False,
        proof_measurements=4,
        schedule_sha256=hashlib.sha256(
            experiment.DEFAULT_SCHEDULE.read_bytes()
        ).hexdigest(),
    )
    for filename, field in (
        ("traces.jsonl", "trace_sha256"),
        ("profile_rollouts.jsonl", "profile_sha256"),
        ("evaluation_rollouts.jsonl", "evaluation_sha256"),
    ):
        metadata[field] = hashlib.sha256((root / filename).read_bytes()).hexdigest()
    (root / "traces.metadata.json").write_text(json.dumps(metadata))
    return root


def _benchmark_args(command, bank, output, *extra):
    return experiment.parser().parse_args(
        [
            command,
            "--pool-size",
            "2",
            "--miners",
            "2",
            "--target",
            "1",
            "--trace-file",
            str(bank / "traces.jsonl"),
            "--trace-pool-metadata",
            str(bank / "traces.metadata.json"),
            "--output",
            str(output),
            *extra,
        ]
    )


def test_default_paper_commands_reject_unmeasured_queries(tmp_path):
    args = experiment.parser().parse_args(
        ["calibrate", "--output", str(tmp_path / "calibration.json")]
    )
    with pytest.raises(ValueError, match="paper experiments require"):
        experiment.cmd_calibrate(args)


def test_benchmark_scaling_preserves_query_specific_length_and_runtime(
    benchmark_bank, tmp_path
):
    args = _benchmark_args("wasted-work", benchmark_bank, tmp_path / "waste")
    schedule = experiment._schedule(args.schedule)
    pool = experiment._make_pool(args, schedule, count=50)
    bank = experiment._load_trace_bank(args, pool)
    assert len({p.query_id for p in pool}) == 50
    assert {p.trace_query_id for p in pool} == {"b0", "b1"}
    result = simulate_race(
        pool,
        miner_count=2,
        seed=2,
        schedule=schedule,
        difficulty_probability=1e-12,
        trace_bank=bank,
        require_trace_bank=True,
        trace_duration=True,
    )
    by_id = {p.query_id: p for p in pool}
    assert all(
        a.trace in bank[by_id[a.query_id].trace_query_id] for a in result.attempts
    )


def test_profiled_selection_reports_time_yield_and_held_out_predictions(
    benchmark_bank, tmp_path, monkeypatch
):
    monkeypatch.setattr(experiment, "_plot_selection", lambda *a: None)
    monkeypatch.setattr(experiment, "_plot_predictions", lambda *a: None)
    output = tmp_path / "selection"
    args = _benchmark_args(
        "cherry-pick", benchmark_bank, output, "--races", "30", "--seeds", "2"
    )
    assert experiment.cmd_cherry_pick(args) == 0
    report = json.loads((output / "strategy_summary.json").read_text())
    assert set(report) == {
        "uniform",
        "profiled-short",
        "profiled-long",
        "shortest-prompt",
    }
    assert all(
        "attacker_blocks_per_hour_mean" in r and "block_rate_vs_uniform_ci95" in r
        for r in report.values()
    )
    assert (output / "query_predictions.csv").exists()
    assert (output / "COMPLETED").exists()


def test_tampered_profile_cannot_rank_using_evaluation_lengths(
    benchmark_bank, tmp_path
):
    args = _benchmark_args("cherry-pick", benchmark_bank, tmp_path / "selection")
    metadata = json.loads(args.trace_pool_metadata.read_text())
    metadata["pool"][0]["profile_mean_output_length"] = 4
    args.trace_pool_metadata.write_text(json.dumps(metadata))
    pool = experiment._make_pool(args, experiment._schedule(args.schedule))
    with pytest.raises(ValueError, match="profiling-only"):
        experiment._selection_profiles(args, pool)


def test_serial_throughput_uses_total_work_over_total_time():
    args = argparse.Namespace(trace_duration=True, work_rate=1)
    bank = {"q": [QueryTrace(1, 10, 1), QueryTrace(1, 10, 9)]}
    assert experiment._ticket_rate(args, bank) == 2


def test_cached_empirical_costs_preserve_the_complete_event_stream():
    schedule = experiment._schedule(experiment.DEFAULT_SCHEDULE)
    pool = [QuerySpec(f"q{i}", n, 4) for i, n in enumerate((2, 4, 8, 16))]
    traces = [QueryTrace(1, 1, 1.1), QueryTrace(2, 1, 2.3), QueryTrace(2, 1, 2.3)]
    prepared = experiment._empirical_trace_bank(pool, traces, schedule)
    assert all(len(values) == len(traces) for values in prepared.values())
    for seed in range(10):
        common = dict(
            pool=pool,
            miner_count=3,
            difficulty_probability=1e-12,
            seed=seed,
            schedule=schedule,
            trace_duration=True,
            response_inclusion_probability=1,
            max_events=None,
        )
        original = simulate_race(**common, trace_distribution=traces)
        cached = simulate_race(**common, trace_bank=prepared)
        assert cached == original


def test_wasted_work_saves_collision_only_per_seed_accounting(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "_plot_wasted_work", lambda *args: None)
    output = tmp_path / "collision_work"
    args = experiment.parser().parse_args(
        [
            "wasted-work",
            "--synthetic-smoke",
            "--miner-grid",
            "2",
            "--pool-grid",
            "4",
            "--targets",
            "100",
            "--seeds",
            "20",
            "--output",
            str(output),
        ]
    )
    assert experiment.cmd_wasted_work(args) == 0
    rows = list(csv.DictReader((output / "runs.csv").open()))
    assert len(rows) == 20
    for row in rows:
        assert row["termination_reason"] != "event_limit"
        if int(row["adopted"]):
            completed = int(row["total_completed_complexity"])
            canonical = int(row["winning_complexity"])
            responses = int(row["settled_complexity"])
            collision = int(row["collision_complexity"])
            assert completed == canonical + responses + collision
            assert int(row["unique_queries_completed"]) == int(
                row["settled_queries"]
            ) + int(row["canonical_queries"])
            assert float(row["wasted_work_pct"]) == pytest.approx(
                100 * collision / completed
            )
    assert any(int(row["settled_queries"]) > 0 for row in rows)
    assert (output / "COMPLETED").exists()
    assert json.loads((output / "manifest.json").read_text())["max_events"] is None
    with pytest.raises(FileExistsError):
        experiment.cmd_wasted_work(args)


def test_replay_verifies_the_original_no_response_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "_plot_wasted_work", lambda *args: None)
    archived = tmp_path / "archived"
    archived.mkdir()
    args = experiment.parser().parse_args(
        [
            "wasted-work",
            "--synthetic-smoke",
            "--miner-grid",
            "2",
            "--pool-grid",
            "4",
            "--targets",
            "100",
            "--seeds",
            "10",
            "--output",
            str(archived),
        ]
    )
    schedule = experiment._schedule(args.schedule)
    pool = experiment._make_pool(args, schedule, seed=args.seed + 4, count=4)
    local = argparse.Namespace(**vars(args))
    local.miners, local.target = 2, 100.0
    local.seed = args.seed + 2 * 100000 + 4 * 17 + 100
    local.races, local.progress = 10, "original fixture"
    local.difficulty_probability = experiment._difficulty(local, None)
    rows, _ = experiment._run_races(local, schedule, pool)
    old_waste = [
        100
        * (r["total_completed_complexity"] - r["successful_pair_complexity"])
        / r["total_completed_complexity"]
        for r in rows
        if r["adopted"]
    ]
    cell = dict(
        target_block_time=100.0,
        miners=2,
        pool_size=4,
        adopted_races=sum(r["adopted"] for r in rows),
        mean_completed_complexity=statistics.fmean(
            r["total_completed_complexity"] for r in rows
        ),
        mean_unfinished_complexity=statistics.fmean(
            r["unfinished_complexity"] for r in rows
        ),
        mean_wasted_work_pct=statistics.fmean(old_waste),
        stdev_wasted_work_pct=statistics.stdev(old_waste),
        difficulty_probability=local.difficulty_probability,
    )
    experiment._write_csv(archived / "cells.csv", [cell])
    experiment._write_json(
        archived / "manifest.json",
        {
            "schema": "poml-llm-wasted-work-1",
            "cli": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
        },
    )
    replay = experiment.parser().parse_args(
        [
            "wasted-work",
            "--replay-from",
            str(archived),
            "--output",
            str(tmp_path / "recovered"),
        ]
    )
    assert experiment.cmd_wasted_work(replay) == 0
    summary = json.loads((tmp_path / "recovered/summary.json").read_text())
    assert summary["archived_cells_verified"] == 1
    recovered = list(csv.DictReader((tmp_path / "recovered/cells.csv").open()))[0]
    assert float(recovered["mean_wasted_work_pct"]) < cell["mean_wasted_work_pct"]
    cell["mean_completed_complexity"] *= 2
    with pytest.raises(ValueError, match="Replay mismatch"):
        experiment._verify_replayed_cell(recovered, cell)


@pytest.mark.parametrize(
    "options",
    [
        ["--workers", "0"],
        ["--seeds", "0"],
        ["--targets", "0"],
        ["--miner-grid", "2", "2"],
        ["--miner-grid", "5", "--pool-grid", "2"],
    ],
)
def test_invalid_collision_grids_fail_before_running(options, tmp_path):
    args = experiment.parser().parse_args(
        ["wasted-work", "--output", str(tmp_path / "out"), *options]
    )
    with pytest.raises(ValueError):
        experiment.cmd_wasted_work(args)
