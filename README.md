# PoML-Sim

Experiments for the Proof-of-ML-Inference protocol. Experiments 1–2 build a bank
of genuine perturbed GPT-2 inference and CUDA DeepProve proofs, then replay its
measured durations while computing fresh SHA-256 lotteries. Experiment 3
resamples recorded work and uses the equivalent winning probability by default.
Complexity uses the appendix operation counts and a frozen weighting and scale.

From the configured workspace, run inside tmux:

```bash
.venv/bin/python scripts/prepare_deepprove_protocol.py
.venv/bin/python experiments/run_live_llm_experiments.py all \
  --device cuda:1 --output /mnt/nas/thomas_work/poml-sim/experiments/results/llm_poml/live-protocol
```

The environment needs the pinned DeepProve checkout, CUDA, cached GPT-2 and
WikiText-2, and the Python dependencies from `pip install -e ".[dev,llm]"`.
Read the [live protocol guide](docs/features/llm_live_protocol.md) for stage
commands, resume behavior, runtime/storage requirements, metrics and assumptions.
The bank is the GPU stage; subsequent chain simulations run on the CPU.
To fit runtime weights and replay Experiment 2 using only an existing bank, see
the [runtime-weighting guide](docs/features/runtime_weighted_selection.md).
For the paper's full-grid uniform-fee query-pool experiment, run
`experiments/run_llm_uniform_fee_scaling.py` using the existing measurement bank;
see the [method and reproduction guide](docs/features/llm_uniform_fee_scaling.md).
Large artifacts and the Python environment live on NAS, with links preserving
workspace paths; see the [storage map](docs/features/nas_artifact_storage.md).

The composite verifier checks stochastic decoding against proved **public
logits**. These are honest execution experiments with documented omissions;
they do not implement the complete private PoML proof relation.

| Directory | Purpose and starting point |
| --- | --- |
| `src/poml_sim/` | Protocol bindings, live execution and event simulation; start with `live_mining.py` |
| `experiments/` | Runnable campaigns, qualification checks and reports; start with `run_live_llm_experiments.py` |
| `scripts/` | Environment preparation and pinned third-party patches; start with `prepare_deepprove_protocol.py` |
| `config/` | Public reference complexity schedule and experiment settings |
| `tests/` | Unit, regression and integration tests; run `.venv/bin/pytest -vv` |
| `docs/` | Methods, audits and historical experiment notes; start with `README.md` |
| `.scratch/` | NAS link: DeepProve checkouts/builds, paper draft and working artifacts |
| `experiments/results/` | NAS link: generated data, proofs, checkpoints and figures |
| `models/`, `model_cache/`, `.venv/` | NAS links: model downloads and installed Python environment |

The original diffusion/EZKL simulator and earlier measurement-bank drivers are
retained for historical reproduction; see [the experiment documentation](docs/experiments.md).
