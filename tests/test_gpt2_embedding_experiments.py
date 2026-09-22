"""Pure and toy-model tests for the GPT-2 embedding-perturbation driver."""

from __future__ import annotations

from types import SimpleNamespace
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.gpt2_embedding_experiments import (
    TracePrompt,
    UtilityExample,
    changed_coordinate_fraction,
    compare_quantized_boundaries,
    deterministic_embedding_noise,
    quantized_trace_commitment,
    replay_test_from_commitments,
    run_trace_separation_experiment,
    run_utility_experiment,
    select_operating_point,
    select_utility_scale,
    _sample_from_distribution,
)
import experiments.gpt2_embedding_experiments as embedding_experiments


class _Embedding(torch.nn.Module):
    def __init__(self, vocabulary: int = 32, dimension: int = 4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(vocabulary * dimension, dtype=torch.float32).reshape(vocabulary, dimension) / 10)

    def forward(self, input_ids):
        return self.weight[input_ids]


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = _Embedding()
        self.head = torch.nn.Linear(4, 32, bias=False)
        self.config = SimpleNamespace(n_embd=4, n_positions=64)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None, use_cache=True, output_hidden_states=False, past_key_values=None):
        values = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        logits = self.head(values)
        return SimpleNamespace(
            logits=logits,
            past_key_values=SimpleNamespace(),
            hidden_states=(values,) if output_hidden_states else None,
        )


class _Tokenizer:
    eos_token_id = None


def test_embedding_noise_is_domain_separated_and_quantized():
    first = deterministic_embedding_noise("a", [1, 2], (2, 4), 0.01)
    assert np_allclose(first, deterministic_embedding_noise("a", [1, 2], (2, 4), 0.01))
    assert not np_allclose(first, deterministic_embedding_noise("b", [1, 2], (2, 4), 0.01))
    assert np_allclose(first / 1e-6, (first / 1e-6).round())


def test_distribution_sampling_uses_probability_device():
    probabilities = torch.tensor([0.25, 0.75], dtype=torch.float64)
    assert _sample_from_distribution(probabilities, 0.5)[0] == 1
    if torch.cuda.is_available():
        assert _sample_from_distribution(probabilities.cuda(), 0.5)[0] == 1


def test_quantized_boundary_comparison_and_replay_binding():
    first = {"embedding": torch.tensor([1, 2]).numpy()}
    second = {"embedding": torch.tensor([1, 3]).numpy()}
    rows = compare_quantized_boundaries(first, second)
    assert rows[0]["collision"] is False
    assert changed_coordinate_fraction(first["embedding"], second["embedding"]) == 0.5
    old = quantized_trace_commitment(first)
    new = quantized_trace_commitment(second)
    assert replay_test_from_commitments([old], [new]) == {
        "old_under_old_accepted": 1,
        "old_under_new_rejected": 1,
        "fresh_under_new_accepted": 1,
    }
    assert quantized_trace_commitment(first, challenge="a") != quantized_trace_commitment(first, challenge="b")


def test_utility_and_separation_run_with_toy_model(tmp_path):
    model = _ToyModel().eval()
    examples = [UtilityExample("toy", "wikitext2", "toy", (1, 2), (3,))]
    utility = run_utility_experiment(model, examples, sigmas=(0.0, 0.01), perturbation_seeds=1, bootstrap_replicates=2, progress_every=1, output_dir=tmp_path / "utility")
    assert len(utility["summary"]) == 2
    assert select_utility_scale(utility["summary"]) == 0.0
    assert select_operating_point(
        utility["summary"],
        [{"sigma_abs": 0.01, "boundary": "embedding", "condition": "full_protocol", "changed_coordinate_fraction": 0.5, "collision_rate": 0.0}],
    ) == 0.01
    assert (tmp_path / "utility" / "utility_curve.png").exists()
    assert (tmp_path / "utility" / "utility_changes.png").exists()
    prompts = [TracePrompt("toy", "standard", "toy", (1, 2))]
    separation = run_trace_separation_experiment(model, _Tokenizer(), prompts, sigmas=(0.0,), pairs=1, length=2, progress_every=1, output_dir=tmp_path / "separation")
    assert separation["boundaries"]
    assert (tmp_path / "separation" / "replay_test.json").exists()
    for filename in ("trace_separation.png", "prefix_collisions.png", "token_agreement.png"):
        assert (tmp_path / "separation" / filename).exists()
    assert not (tmp_path / "separation" / "replay_test.png").exists()


def test_piqa_uses_script_free_fallback_when_datasets_rejects_loader(monkeypatch):
    calls = []

    def failing_loader(*args, **kwargs):
        raise RuntimeError("Dataset scripts are no longer supported, but found piqa.py")

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = failing_loader
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setattr(
        embedding_experiments,
        "_load_rows_from_dataset_service",
        lambda dataset, config, split: calls.append((dataset, config, split)) or [{"goal": "g"}],
    )

    rows = embedding_experiments._load_rows("piqa", "validation")
    assert rows == [{"goal": "g"}]
    assert calls == [("regisss/piqa", "default", "validation")]


def test_structural_prompts_use_script_free_dataset_fallback(monkeypatch):
    def failing_loader(*args, **kwargs):
        raise RuntimeError("Dataset scripts are no longer supported, but found bigbench.py")

    class Tokenizing:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2]}

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = failing_loader
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    requested = []

    def service_rows(dataset, config, split):
        requested.append((dataset, config, split))
        if dataset == "tasksource/bigbench":
            return [{"inputs": "repeat this"}, {"inputs": "repeat this too"}]
        return [{"prompt": "resist correction"}, {"prompt": "resist correction too"}]

    monkeypatch.setattr(embedding_experiments, "_load_rows_from_dataset_service", service_rows)
    prompts = embedding_experiments.load_trace_prompts(
        Tokenizing(), standard_examples=[], count_per_standard_task=1, max_structural_prompts=1
    )
    assert [prompt.text for prompt in prompts] == ["repeat this", "resist correction"]
    assert requested == [
        ("tasksource/bigbench", "repeat_copy_logic", "validation"),
        ("pminervini/inverse-scaling", "resisting-correction", "data"),
    ]


def test_mismatched_checkpoint_is_archived_for_default_rerun(tmp_path):
    output_dir = tmp_path / "checkpoints"
    path, _ = embedding_experiments._open_embedding_checkpoint(
        output_dir, filename="utility_checkpoint.jsonl", fingerprint="old", resume=True
    )
    path.write_text('{"key":"old","result":{}}\n')
    fresh_path, records = embedding_experiments._open_embedding_checkpoint(
        output_dir, filename="utility_checkpoint.jsonl", fingerprint="new", resume=True
    )
    assert fresh_path == output_dir / "utility_checkpoint.jsonl"
    assert records == {}
    assert list(output_dir.glob("utility_checkpoint.stale-*.jsonl"))


def np_allclose(first, second):
    import numpy as np

    return bool(np.allclose(first, second))
