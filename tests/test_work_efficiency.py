"""Timing partitions and aggregate ratios must preserve measured work."""

import json
import pytest
from experiments.merge_work_efficiency import main as merge
from poml_sim.work_efficiency import ComponentTimer, summarize_records


def test_nested_timers_partition_elapsed_without_double_counting():
    ticks = iter([0, 10, 40, 100])
    timer = ComponentTimer(lambda: next(ticks))
    with timer.section("parent"):
        with timer.section("child"):
            pass
    assert timer.seconds == {"parent": 70 / 1e9, "child": 30 / 1e9}
    assert timer.stack == []


def test_failed_calls_preserve_timing_and_propagate_error():
    ticks = iter([0, 20])
    timer = ComponentTimer(lambda: next(ticks))
    with pytest.raises(ValueError):
        with timer.section("failed"):
            raise ValueError("bad proof")
    assert timer.seconds["failed"] == 20 / 1e9
    assert not timer.stack


def record(elapsed, useful, length=8):
    return dict(
        elapsed_seconds=elapsed,
        useful_seconds=useful,
        inference_seconds=useful / 2,
        proof_seconds=useful / 4,
        verification_seconds=useful / 4,
        auxiliary_seconds=elapsed - useful,
        prompt_length=length,
        output_length=2,
        verified=True,
        components_seconds={
            "inference": useful / 2,
            "proof_generation": useful / 4,
            "proof_verification": useful / 4,
            "hashing": elapsed - useful,
        },
        fresh_inference=True,
        fresh_proof=True,
    )


def test_ratio_of_sums_weights_slow_jobs_and_bootstrap_is_reproducible():
    rows = [record(2, 1), record(10, 10)]
    result = summarize_records(rows, bootstrap=100, seed=17)
    assert result["ratios"]["alpha_cert"] == pytest.approx(12 / 11)
    assert result["ratios"]["auxiliary_fraction"] == pytest.approx(1 / 12)
    assert result == summarize_records(rows, bootstrap=100, seed=17)
    assert result["ci95"]["alpha_cert"][0] <= 12 / 11 <= result["ci95"]["alpha_cert"][1]


@pytest.mark.parametrize(
    "rows", [[], [record(1, 2)], [record(float("nan"), 1)], [{**record(2, 1), "verified": False}]]
)
def test_invalid_or_unverified_timings_are_rejected(rows):
    with pytest.raises(ValueError):
        summarize_records(rows)


def test_repeated_prompts_are_resampled_together():
    rows = [{**record(2, 1), "prompt_sha256": "same"}, {**record(10, 10), "prompt_sha256": "same"}]
    result = summarize_records(rows, bootstrap=50)
    assert result["bootstrap"]["clusters"] == 1
    assert result["ci95"]["alpha_cert"] == pytest.approx([12 / 11, 12 / 11])


def test_component_accounting_errors_are_rejected():
    row = record(2, 1)
    row["components_seconds"]["hashing"] = 2
    with pytest.raises(ValueError, match="partition"):
        summarize_records([row])


def write_shard(path, elapsed, useful, **manifest):
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps({"status": "complete", "queries": 1, "model": "gpt2", **manifest})
    )
    (path / "measurements.jsonl").write_text(json.dumps(record(elapsed, useful)) + "\n")


def test_merge_recomputes_totals_and_preserves_each_source(tmp_path):
    first, second, output = [tmp_path / name for name in ("first", "second", "merged")]
    write_shard(first, 2, 1)
    write_shard(second, 10, 10)
    assert merge([str(first), str(second), "--output", str(output), "--bootstrap", "20"]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["count"] == 2
    assert summary["ratios"]["alpha_cert"] == pytest.approx(12 / 11)
    assert len(json.loads((output / "manifest.json").read_text())["source_manifests"]) == 2


@pytest.mark.parametrize("problem", ["duplicate", "incomplete", "incompatible"])
def test_merge_rejects_invalid_campaigns_without_writing_results(tmp_path, problem):
    first, second, output = [tmp_path / name for name in ("first", "second", "merged")]
    write_shard(first, 2, 1)
    write_shard(
        second,
        2,
        1,
        **(
            {"status": "running"}
            if problem == "incomplete"
            else {"model": "different"}
            if problem == "incompatible"
            else {}
        ),
    )
    with pytest.raises(SystemExit):
        merge(
            [str(first), str(first if problem == "duplicate" else second), "--output", str(output)]
        )
    assert not output.exists()
