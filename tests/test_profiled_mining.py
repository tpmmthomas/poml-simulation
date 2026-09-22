import hashlib
import json
from pathlib import Path
import pytest

from poml_sim.gpt2_work import reference_counts
from poml_sim.llm_simulation import load_schedule
from poml_sim.lottery import LIMIT, calibrate_scale, scaled_complexity
from poml_sim.profiled_mining import MeasuredBank, ProfiledWork
from poml_sim.protocol_inputs import canonical, digest, frame


@pytest.mark.parametrize("use_weights", [False, True])
def test_profiled_work_replays_duration_and_recomputes_current_artifacts(
    tmp_path, monkeypatch, use_weights
):
    schedule = load_schedule(Path("config/gpt2_reference_schedule.json"))
    source = tmp_path / "source"
    source.mkdir()
    query = {
        "query_id": "q0",
        "request_id": "calibration:q0:0",
        "prompt_tokens": [1, 2],
        "prompt_length": 2,
        "max_output_length": 3,
    }
    query["prompt_sha256"] = digest(canonical(query["prompt_tokens"])).hex()
    tokens = [3, 4]
    logits = b"\0" * (2 * 50257 * 8)
    (source / "logits.i64").write_bytes(logits)
    (source / "proof.bin").write_bytes(b"proof")
    (source / "request.json").write_text(
        json.dumps({"prompt_tokens": query["prompt_tokens"], "max_output": 3})
    )
    raw = sum(reference_counts(2, 2, schedule)["combined"].values())
    scale = calibrate_scale([raw], 10000)
    proof_sha = hashlib.sha256(b"proof").hexdigest()
    output = (
        frame(
            canonical(
                {
                    "qid": query["request_id"],
                    "tokens": tokens,
                    "logits_scale": 1,
                    "logits_shape": [2, 50257],
                }
            )
        )
        + logits
    )
    row = {
        **query,
        "qid": query["request_id"],
        "replicate": 0,
        "verified": True,
        "fresh_inference": True,
        "fresh_proof": True,
        "duration": 1.0,
        "inference_proof_seconds": 0.8,
        "inference_seconds": 0.2,
        "proof_seconds": 0.6,
        "proof_directory": str(source),
        "output_length": 2,
        "output_tokens": tokens,
        "stop_reason": "eos",
        "logits_scale": 1,
        "logits_shape": [2, 50257],
        "proof_sha256": proof_sha,
        "raw_complexity": raw,
        "complexity": scaled_complexity(raw, scale),
        "output_sha256": digest(output).hex(),
        "r": "00" * 32,
    }
    bank = MeasuredBank([query], [row], schedule, scale)
    weights = (
        {
            key: 3 if key == "integer_mac" else 0
            for key in reference_counts(2, 2, schedule)["combined"]
        }
        if use_weights
        else None
    )
    monkeypatch.setattr(
        "poml_sim.profiled_mining.evaluate_tickets",
        lambda *_args: {
            "tickets_evaluated": 1,
            "winning_ticket": 0,
            "winning_hash": "00" * 32,
            "prefix_sha256": "11" * 32,
        },
    )
    work = ProfiledWork(
        bank,
        {"embedding_std": 1.0, "setup_sha256": "test"},
        schedule,
        scale,
        tmp_path / "attempts",
        7,
        0.05,
        weights=weights,
    )
    result = work.execute(
        {**query, "request_id": "query-0"}, 0, b"0" * 32, b"1" * 32, b"", LIMIT
    )
    assert result["duration"] == 0.8
    assert result["fresh_proof"] is False
    assert result["proof_sha256"] == proof_sha
    assert result["source_output_sha256"] == row["output_sha256"]
    assert result["raw_complexity"] == raw
    weighted = (
        3 * reference_counts(2, 2, schedule)["combined"]["integer_mac"]
        if use_weights
        else raw
    )
    assert result["weighted_complexity"] == weighted
    assert result["complexity"] == scaled_complexity(weighted, scale)
    assert (
        row["complexity"] == 10000
    )  # Source measurements retain their original schedule.
    if use_weights:
        assert result["complexity_weights_sha256"] == digest(canonical(weights)).hex()
    assert (Path(result["attempt_directory"]) / "ciphertext.bin").exists()
