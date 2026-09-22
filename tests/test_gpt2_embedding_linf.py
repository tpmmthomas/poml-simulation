"""Regression tests for raw and quantized GPT-2 activation distances."""

import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.gpt2_embedding_linf import (  # noqa: E402
    appendix_rows,
    boundary_distances,
    clean_continuation,
    load_manifest,
    main,
    measure_prompt,
    summarize_records,
)


def test_distances_use_absolute_max_and_quantizer_bins():
    values = torch.tensor([[0.00049, -2.0], [0.00051, 3.0], [1.0, 2.0], [1.0, 2.0]])
    result = boundary_distances(values, 0.001, 2**31 - 1)
    assert result["raw_linf"].tolist() == [5.0, 0.0]
    assert result["quantized_linf_steps"].tolist() == [5000.0, 0.0]
    assert result["coordinates"] == 2
    # Very small raw differences can cross a rounding boundary.
    edge = boundary_distances(values[:2, :1], 0.001, 2**31 - 1)
    assert edge["raw_linf"].item() < 0.001
    assert edge["quantized_linf_steps"].item() == 1


def test_distances_detect_clipping_and_avoid_integer_overflow():
    result = boundary_distances(torch.tensor([[-3e9], [3e9]]), 1.0, 2**31 - 1)
    assert result["raw_linf"].item() == 6e9
    assert result["quantized_linf_steps"].item() == 2 * (2**31 - 1)
    assert result["clipped_coordinates"].item() == 2


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_distances_reject_invalid_quantization_scale(scale):
    with pytest.raises(ValueError, match="scale"):
        boundary_distances(torch.ones(2, 1), scale, 100)


def test_distances_reject_nonfinite_or_unpaired_activations():
    with pytest.raises(ValueError, match="non-finite"):
        boundary_distances(torch.tensor([[float("nan")], [0.0]]), 0.001, 100)
    with pytest.raises(ValueError, match="even batch"):
        boundary_distances(torch.zeros(3, 2), 0.001, 100)


@pytest.fixture
def small_gpt2():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            n_embd=8, n_head=2, n_layer=2, n_positions=32, vocab_size=32
        )
    ).eval()


def test_fixed_tokens_isolate_noise_and_capture_real_block_outputs(small_gpt2):
    prompt = {"prompt_id": "test:0", "stratum": "standard", "token_ids": [1, 2, 3]}
    rows = measure_prompt(
        small_gpt2,
        prompt,
        [4, 5],
        sigmas=[0.0, 0.01],
        pairs=2,
        steps=[0, 1, 2],
        seed=42,
        scale=0.001,
        clip=2**31 - 1,
    )
    baseline = [row for row in rows if row["sigma_abs"] == 0]
    assert all(row["raw_linf"] == [0.0, 0.0] for row in baseline)
    assert all(row["quantized_linf_steps"] == [0.0, 0.0] for row in baseline)
    noisy = {
        (row["boundary"], row["scope"], row["step"]): row
        for row in rows
        if row["sigma_abs"] > 0
    }
    assert min(noisy["embedding", "prompt", 0]["raw_linf"]) > 0
    assert noisy["embedding", "prompt", 0]["coordinates"] == 3 * 8
    assert noisy["block_0", "active_token", 2]["coordinates"] == 8
    # Generated embeddings and the first position-wise norm/QKV stay clean;
    # their first attention/block output can still depend on the noisy prefix.
    for boundary in (
        "embedding",
        "post_first_norm",
        "layer_0_q",
        "layer_0_k",
        "layer_0_v",
    ):
        assert noisy[boundary, "active_token", 2]["raw_linf"] == [0.0, 0.0]
    assert min(noisy["block_0", "active_token", 2]["raw_linf"]) > 0
    assert (
        noisy["block_1", "prompt", 0]["raw_linf"]
        != noisy["final_norm", "prompt", 0]["raw_linf"]
    )
    assert all(not module._forward_hooks for module in small_gpt2.modules())


def test_causal_teacher_forcing_matches_sequential_cached_logits(small_gpt2):
    prompt = [1, 2, 3]
    continuation = clean_continuation(small_gpt2, prompt, 3)
    assert len(continuation) == 3
    with torch.inference_mode():
        full = small_gpt2(torch.tensor([prompt + continuation]), use_cache=False)
        initial = small_gpt2(torch.tensor([prompt]), use_cache=True)
        cache = initial.past_key_values
        for index, token in enumerate(continuation):
            incremental = small_gpt2(
                torch.tensor([[token]]), past_key_values=cache, use_cache=True
            )
            cache = incremental.past_key_values
            torch.testing.assert_close(
                incremental.logits[:, -1],
                full.logits[:, len(prompt) + index],
                atol=1e-6,
                rtol=1e-5,
            )


def test_detailed_layers_capture_attention_feedforward_and_complete_context(small_gpt2):
    prompt = {"prompt_id": "test", "token_ids": [1, 2, 3]}
    rows = measure_prompt(
        small_gpt2,
        prompt,
        [4],
        sigmas=[0.01],
        pairs=1,
        steps=[0, 1],
        seed=42,
        scale=0.001,
        clip=2**31 - 1,
        noise_clip=None,
        noise_quantum=0.0,
        detailed_layers=True,
    )
    lookup = {(r["boundary"], r["scope"], r["step"]): r for r in rows}
    for name in [
        "attention_0",
        "feedforward_0",
        "attention_1",
        "feedforward_1",
        "final_norm",
    ]:
        assert lookup[name, "context", 1]["coordinates"] == 4 * 8
        assert min(lookup[name, "context", 1]["raw_linf"]) > 0
    assert lookup["embedding", "active_token", 1]["raw_linf"] == [0.0]
    assert (
        lookup["embedding", "context", 1]["raw_linf"]
        == lookup["embedding", "context", 0]["raw_linf"]
    )


def test_summary_pools_pair_values_instead_of_averaging_prompt_minima(tmp_path):
    records = []
    for prompt_id, raw in (("a", [1.0, 10.0]), ("b", [3.0, 4.0])):
        records.append(
            {
                "prompt_id": prompt_id,
                "stratum": "standard",
                "distances": [
                    {
                        "boundary": "block_0",
                        "sigma_abs": 0.01,
                        "scope": "prompt",
                        "step": 0,
                        "raw_linf": raw,
                        "quantized_linf_steps": [int(v * 1000) for v in raw],
                        "clipped_coordinates": [0, 0],
                        "coordinates": 8,
                    }
                ],
            }
        )
    row = summarize_records(records, 0.001)[0]
    assert row["raw_linf_min"] == 1.0
    assert row["raw_linf_median"] == 3.5
    assert row["raw_linf_mean"] == 4.5
    assert row["comparisons"] == 4
    assert row["dequantized_linf_min"] == 1.0
    (tmp_path / "metadata.json").write_text(json.dumps({"quantization_scale": 0.001}))
    (tmp_path / "linf_checkpoint.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records)
    )
    assert main(["--output-dir", str(tmp_path), "--report-only"]) == 0
    assert (tmp_path / "linf_table.md").exists()
    assert "raw_linf_min" in (tmp_path / "linf_summary.csv").read_text()


def test_manifest_rejects_duplicate_ids_and_empty_tokens(tmp_path):
    manifest = tmp_path / "prompts.jsonl"
    row = {"prompt_id": "test", "token_ids": [1]}
    manifest.write_text(json.dumps(row) + "\n" + json.dumps(row))
    with pytest.raises(ValueError, match="unique"):
        load_manifest(manifest)
    manifest.write_text(json.dumps({**row, "token_ids": []}))
    with pytest.raises(ValueError, match="token IDs"):
        load_manifest(manifest)


def test_appendix_minimum_ranges_over_all_blocks_steps_and_strata():
    records = []
    for stratum, block, step, distance in (
        ("standard", "block_0", 0, 10.0),
        ("standard", "block_1", 32, 2.0),
        ("copy", "block_0", 1, 0.5),
    ):
        records.append(
            {
                "stratum": stratum,
                "distances": [
                    {
                        "boundary": block,
                        "sigma_abs": 0.005,
                        "scope": "active_token",
                        "step": step,
                        "raw_linf": [distance],
                        "quantized_linf_steps": [distance * 1000],
                        "clipped_coordinates": [0],
                        "coordinates": 8,
                    }
                ],
            }
        )
    row = appendix_rows(records, 0.001)[0]
    assert row["active_blocks_raw_min"] == 0.5
    assert row["active_blocks_quantized_steps_min"] == 500
    assert row["prompt_blocks_raw_min"] is None
