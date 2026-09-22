"""Compatibility statistics on a local small GPT-2, without model downloads."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from poml_sim.gpt2_compatibility import (
    prompt_noise,
    forward,
    sample_token,
    utility,
    utility_summary,
    compare_trajectories,
)


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            n_embd=16,
            n_layer=2,
            n_head=2,
            n_positions=32,
            vocab_size=64,
            eos_token_id=63,
            attn_implementation="eager",
        )
    ).eval()


def test_zero_noise_matches_clean_token_forward_and_generated_embeddings_stay_clean(model):
    tokens = [1, 2, 3]
    noise = prompt_noise(model, tokens, 0, "first")
    with torch.inference_mode():
        expected = model(torch.tensor([tokens + [4]]), use_cache=False).logits
        actual = forward(model, tokens + [4], noise).logits
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(noise) == 0
    assert torch.equal(
        prompt_noise(model, tokens, 0.05, "same"), prompt_noise(model, tokens, 0.05, "same")
    )
    assert not torch.equal(
        prompt_noise(model, tokens, 0.05, "left"), prompt_noise(model, tokens, 0.05, "right")
    )


def test_inverse_cdf_respects_filtering_and_probability_edges():
    logits = torch.tensor([0.0, 2.0, 1.0])
    assert sample_token(logits, 0, top_k=1) == 1
    assert sample_token(logits, 0.99999, top_k=1) == 1
    assert sample_token(logits, 0, top_k=2, top_p=1) == 1
    assert sample_token(logits, 0.99999, top_k=2, top_p=1) == 2
    with pytest.raises(ValueError):
        sample_token(logits, 1)


def test_utility_clean_baseline_once_and_replicate_aggregation(model):
    example = dict(id="fixture", task="wikitext2", context=[1, 2], target=[3, 4], choices=[])
    rows = utility(model, [example], alphas=(0.05,), replicates=2)
    assert len(rows) == 3
    summary = utility_summary(rows)
    assert {r["replicates"] for r in summary} == {1, 2}
    assert all(r["value"] > 0 for r in summary)


def test_trace_captures_whole_prefix_boundaries_and_stops_on_eos(model, monkeypatch):
    example = dict(id="fixture", task="piqa", context=[1, 2])
    monkeypatch.setattr("poml_sim.gpt2_compatibility.sample_token", lambda *a, **k: 63)
    rows = compare_trajectories(model, example, steps=(0, 1, 4))
    assert {r["prefix"] for r in rows} == {0}
    assert {r["boundary"] for r in rows} == {"embedding", "1A", "1F", "2A", "2F", "norm", "logits"}
    assert all(r["linf"] > 0 for r in rows)


def test_independent_trajectories_sample_separately(model, monkeypatch):
    samples = iter([10, 20, 11, 21])
    seen = []
    original = model.get_input_embeddings().forward

    def record(ids):
        seen.append(ids.tolist()[0])
        return original(ids)

    monkeypatch.setattr(model.get_input_embeddings(), "forward", record)
    monkeypatch.setattr("poml_sim.gpt2_compatibility.sample_token", lambda *a, **k: next(samples))
    compare_trajectories(model, dict(id="fixture", context=[1, 2], task="piqa"), steps=(0, 1))
    assert [1, 2, 10] in seen and [1, 2, 20] in seen
