"""Pure-function tests for the GPT-2 PoML experiment harness."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.gpt2_experiments import (  # noqa: E402
    Prompt,
    derive_vrf_private_key,
    derive_vrf_public_key,
    cache_workload_statistics,
    prefix_indicators,
    sample_token_from_logits,
    vrf_uniforms,
    run_collision_experiment,
    run_concentration_experiment,
    _resolve_devices,
    _cache_bytes,
    generate_trace,
)


def test_vrf_uniform_stream_is_reproducible_and_step_bound():
    first_uniforms, first_transcript = vrf_uniforms("seed-a", [10, 20, 30], 3)
    second_uniforms, second_transcript = vrf_uniforms("seed-a", [10, 20, 30], 3)
    changed_uniforms, changed_transcript = vrf_uniforms("seed-b", [10, 20, 30], 3)

    assert first_uniforms == second_uniforms
    assert first_transcript == second_transcript
    assert first_uniforms != changed_uniforms
    assert len({entry["input"] for entry in first_transcript}) == 3
    assert all(0 <= value < 1 for value in first_uniforms)
    assert derive_vrf_private_key("seed-a") == derive_vrf_private_key("seed-a")
    assert len(derive_vrf_public_key("seed-a")) == 32


def test_inverse_cdf_sampler_honours_temperature_and_top_k():
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0])
    token, token_probability, top_probability, entropy = sample_token_from_logits(
        logits, 0.01, 1.0, top_k=2
    )

    assert token == 0
    assert token_probability == top_probability
    assert entropy > 0

    # A high uniform value can only select from the retained top-k support.
    token, *_ = sample_token_from_logits(logits, 0.99, 1.0, top_k=2)
    assert token == 1


def test_inverse_cdf_sampler_rejects_invalid_uniform():
    try:
        sample_token_from_logits(torch.tensor([1.0, 0.0]), 1.0, 1.0)
    except ValueError as exc:
        assert "uniform" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected ValueError")


def test_prefix_indicators_censor_pairs_after_eos():
    assert prefix_indicators([1, 2, 3], [1, 2, 4], 4) == [1, 1, 1, 0, None]
    assert prefix_indicators([1], [1, 2], 3) == [1, 1, None, None]


def test_cache_workload_statistics_reports_exact_repeats_and_prefixes():
    prompts = [
        Prompt("a", "test", "a", (1, 2, 3, 4)),
        Prompt("b", "test", "b", (1, 2, 9, 4)),
    ]
    rows = cache_workload_statistics(prompts, repeat_probability=1.0, capacities=(1, 4), seed=3)
    exact = [row for row in rows if row["workload"] == "exact_repeat" and row["capacity"] == 4][0]
    shared = [row for row in rows if row["workload"] == "shared_prefix" and row["capacity"] == 4][0]
    assert exact["exact_hit_rate"] > 0
    assert shared["max_compatible_prefix_tokens"] >= 2


def test_device_resolver_honours_explicit_worker_list():
    assert _resolve_devices("cpu", None) == ["cpu"]
    assert _resolve_devices("cpu", "cuda:0, cuda:2") == ["cuda:0", "cuda:2"]


def test_cache_bytes_skips_uninitialized_dynamic_cache_slots():
    cache = SimpleNamespace(
        past_key_values=SimpleNamespace(
            key_cache=[None, torch.zeros(1, 2)],
            value_cache=[torch.ones(1, 3), None],
        )
    )
    assert _cache_bytes(cache) == (2 + 3) * torch.zeros((), dtype=torch.float32).element_size()


def test_cache_bytes_reads_transformers5_dynamic_cache_layers():
    layer = SimpleNamespace(keys=torch.zeros(1, 2), values=torch.ones(1, 3))
    cache = SimpleNamespace(past_key_values=SimpleNamespace(layers=[layer]))
    assert _cache_bytes(cache) == (2 + 3) * torch.zeros((), dtype=torch.float32).element_size()


class _ToyModel:
    def __call__(self, input_ids=None, past_key_values=None, use_cache=True):
        last = int(input_ids[0, -1])
        logits = torch.tensor([[[float((last + i) % 5) for i in range(5)]]])
        return SimpleNamespace(
            logits=logits,
            past_key_values=SimpleNamespace(key_cache=[], value_cache=[]),
        )


class _NoCallModel:
    def __call__(self, *args, **kwargs):  # pragma: no cover - failure proves resume
        raise AssertionError("checkpoint was not resumed")


class _ToyTokenizer:
    eos_token_id = None


def test_public_stop_token_ends_generation_without_forcing_a_length():
    model = _ToyModel()
    ordinary = generate_trace(model, _ToyTokenizer(), [1, 2], "stop-test", length=3)
    stopped = generate_trace(model, _ToyTokenizer(), [1, 2], "stop-test", length=3,
                             stop_token_ids=range(10))
    assert len(ordinary.tokens) == 3
    assert stopped.tokens == ordinary.tokens[:1]


def test_collision_checkpoint_resumes_without_model_calls(tmp_path, capsys):
    prompt = Prompt("p", "test", "toy", (1, 2))
    run_collision_experiment(
        _ToyModel(), _ToyTokenizer(), [prompt], pairs=1, temperatures=(1.0,),
        length=2, bootstrap_replicates=2, output_dir=tmp_path, progress_every=1,
    )
    assert "prompt=1/1" in capsys.readouterr().out
    resumed = run_collision_experiment(
        _NoCallModel(), _ToyTokenizer(), [prompt], pairs=1, temperatures=(1.0,),
        length=2, bootstrap_replicates=2, output_dir=tmp_path, progress_every=1,
    )
    assert resumed["summary"]


def test_concentration_checkpoint_resumes_without_model_calls(tmp_path):
    prompt = Prompt("p", "test", "toy", (1, 2))
    run_concentration_experiment(
        _ToyModel(), _ToyTokenizer(), [prompt], temperatures=(1.0,), length=2,
        output_dir=tmp_path, progress_every=1,
    )
    resumed = run_concentration_experiment(
        _NoCallModel(), _ToyTokenizer(), [prompt], temperatures=(1.0,), length=2,
        output_dir=tmp_path, progress_every=1,
    )
    assert resumed
