"""Relative scaling and unclipped Gaussian regression tests for both campaigns."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import gpt2_embedding_experiments as embedding  # noqa: E402
from experiments.gpt2_embedding_alpha import layer_minima, paper_boundaries  # noqa: E402


def test_relative_scales_use_population_std_of_whole_embedding_table():
    table = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    model = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(weight=table))
    std, sigmas = embedding.relative_noise_scales(model, [0.05, 0.1])
    assert std == pytest.approx(np.std(table.numpy(), ddof=0))
    assert sigmas == pytest.approx((0.05 * np.sqrt(5), 0.1 * np.sqrt(5)))


@pytest.mark.parametrize("alphas", [[], [0.0], [-1.0], [float("nan")], [float("inf")]])
def test_relative_scales_reject_invalid_alphas(alphas):
    with pytest.raises(ValueError, match="alphas"):
        embedding.relative_noise_scales(None, alphas)


def test_gaussian_sampler_retains_tails_and_exact_scale_factor(monkeypatch):
    # These uniforms yield z > 3; clipping would change the stated noise law.
    monkeypatch.setattr(embedding, "_stream_words", lambda *args: iter([0, 0]))
    noise = embedding.deterministic_embedding_noise(
        42, [1], (1, 1), 0.1, clip=None, quantum=0.0
    )
    twice = embedding.deterministic_embedding_noise(
        42, [1], (1, 1), 0.2, clip=None, quantum=0.0
    )
    assert noise.item() > 0.3
    assert twice.item() == 2 * noise.item()
    assert noise.item() / 1e-6 != round(noise.item() / 1e-6)


def test_utility_passes_unclipped_noise_configuration_to_all_perturbed_scores(
    monkeypatch,
):
    configs = []

    def score(*args, **kwargs):
        if kwargs["sigma_abs"]:
            configs.append(kwargs["config"])
        return {"nll": 1.0, "target_tokens": 1, "predicted_tokens": (3,)}

    monkeypatch.setattr(embedding, "score_continuation", score)
    example = embedding.UtilityExample("p", "piqa", "", (1, 2), (3,), ((3,), (4,)), 0)
    embedding._utility_record(
        None,
        [example],
        sigmas=[0.007],
        perturbation_seeds=1,
        seed=42,
        device="cpu",
        config=embedding.PerturbationConfig(0.0, clip=None, quantum=0.0),
    )
    assert len(configs) == 3
    assert all(config.clip is None and config.quantum == 0 for config in configs)


def test_layer_minima_use_whole_context_and_retain_each_sublayer():
    record = {"stratum": "standard", "distances": []}
    for boundary, label in paper_boundaries(1):
        for step in [0, 1]:
            record["distances"].append(
                {
                    "boundary": boundary,
                    "scope": "active_token" if boundary == "logits" else "context",
                    "step": step,
                    "sigma_abs": 0.01,
                    "raw_linf": [step + 0.1],
                    "quantized_linf_steps": [100 + 1000 * step],
                    "clipped_coordinates": [0],
                    "coordinates": 8,
                }
            )
    metadata = {
        "alphas": [0.1],
        "sigmas": [0.01],
        "layers": 1,
        "quantization_scale": 0.001,
    }
    rows = layer_minima([record], metadata)
    assert [row["boundary"] for row in rows] == [
        "embedding",
        "attention_0",
        "feedforward_0",
        "final_norm",
        "logits",
    ]
    assert all(row["raw_linf_min"] == 0.1 and row["comparisons"] == 2 for row in rows)
    with pytest.raises(ValueError, match="missing"):
        layer_minima([], metadata)
