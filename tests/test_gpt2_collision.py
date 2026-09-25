"""Tests for the generated-prefix collision experiment."""

import csv

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from poml_sim.gpt2_collision import (  # noqa: E402
    Prompt,
    first_divergence,
    prefix_indicators,
    run_collision_experiment,
)


class _Tokenizer:
    eos_token_id = 15


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(43)
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            n_embd=16,
            n_layer=2,
            n_head=2,
            n_positions=16,
            vocab_size=16,
            eos_token_id=15,
            attn_implementation="eager",
        )
    ).eval()


def test_prefix_helpers_censor_eos_and_report_first_difference():
    assert prefix_indicators([1, 2], [1, 3], 3) == [1, 1, 0, None]
    assert first_divergence([1, 2], [1, 3]) == 2
    assert first_divergence([1], [1, 2]) == 2
    assert first_divergence([1, 2], [1, 2]) is None


def test_collision_campaign_writes_reproducible_outputs(model, tmp_path):
    prompts = [
        Prompt("wikitext2:0", "wikitext2", "one", (1, 2, 3)),
        Prompt("wikitext2:1", "wikitext2", "two", (4, 5, 6)),
    ]
    result = run_collision_experiment(
        model,
        _Tokenizer(),
        prompts,
        pairs=1,
        temperatures=(1.0,),
        length=3,
        seed=42,
        device="cpu",
        bootstrap_replicates=5,
        output_dir=tmp_path,
        resume=False,
    )
    assert result["aggregates"][0]["temperature"] == 1.0
    assert (tmp_path / "collision_curves.png").exists()
    assert (tmp_path / "collision_curves.pdf").exists()
    with (tmp_path / "collision_curves.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert rows[0]["prefix_length"] == "0"
    assert rows[0]["collision_probability"] == "1.0"
    assert (tmp_path / "first_divergence.csv").exists()
