# PoML Experiments

The current campaign uses distinct WikiText-2 prompts, Gaussian embedding noise,
stochastic decoding checked against proved public logits, a genuine CUDA
measurement bank, and calibrated complexity counts around 10,000. Experiments
1–2 replay the bank's measured inference+proof durations and recompute the
remaining protocol events, so the chain simulation does not invoke DeepProve for
every virtual attempt.

```bash
.venv/bin/python scripts/prepare_deepprove_protocol.py
.venv/bin/python experiments/run_live_llm_experiments.py all \
  --device cuda:1 --output /mnt/nas/thomas_work/poml-sim/experiments/results/llm_poml/live-protocol
```

Run in tmux and append `--resume` to continue. Read the
[live protocol guide](../docs/features/llm_live_protocol.md) for separate stage
commands, grids, honest execution assumptions, metrics, proof scope and runtime.
The workspace `results/` path is a link to NAS; existing relative output paths
also write there. See the [storage map](../docs/features/nas_artifact_storage.md).
The paper's uniform-fee query-pool subsection uses
[`run_llm_uniform_fee_scaling.py`](run_llm_uniform_fee_scaling.py): a separate
CPU replay of the original homogeneous design with complexity-weighted
duplicates and all grid cells, including Q ≤ M. See its
[method and commands](../docs/features/llm_uniform_fee_scaling.md).
The former measurement-bank-only campaign has been replaced by this explicit
profile-and-replay implementation; its historical results are not results of
the current campaign.

The following sections document the retained historical experiment drivers.

Drivers for the PoML simulator experiments (PoML vs. simplified PoW baseline)
and the Stable Diffusion appendix experiments that empirically validate the
Computational Independence of Activations (CIA) property underpinning the
PoML lottery.

## Files

### Main experiments (simulator)

| Path | What it does |
| --- | --- |
| [utils.py](utils.py) | Shared helpers: `run_poml`, log parsing, difficulty rescaling, statistics |
| [pow_sim.py](pow_sim.py) | Multi-process SHA-256 PoW simulator (lottery only — no blockchain) |
| [pow_calibrate.py](pow_calibrate.py) | Benchmarks local SHA-256 hashrate, writes a difficulty target for a chosen expected block time |
| [exp1_block_time_stability.py](exp1_block_time_stability.py) | 50-block PoML and PoW runs at 300 s target; emits mean/min/max/stdev/variance |
| [exp2_wasted_work.py](exp2_wasted_work.py) | PoML sweeps over expected block time (200/300/400 s) and miner count (2/4/8), 10 blocks each |
| [exp5_uniform_fee_collisions.py](exp5_uniform_fee_collisions.py) | One-block uniform-fee collision experiment using real EZKL timing calibration and accelerated discrete-event races |
| [exp5b_ratio_scatter.py](exp5b_ratio_scatter.py) | Exp 5 race engine over 1,000 stratified random `(Q, M)` pairs, with annotated wasted-work heatmaps across `Q/M` and miner-count bands |
| [exp5c_m_q_heatmap.py](exp5c_m_q_heatmap.py) | Exp 5 race engine on a regular log-spaced `(M, Q)` grid; one M-by-Q heatmap of percentage of wasted work per target block time. See the [Exp 5c report](../docs/experiment_5c.md) for the rationale, procedure, and findings. |
| [exp5_seed_manifest.json](exp5_seed_manifest.json) | Frozen, domain-separated seed roots and the 27-configuration Exp 5 grid |
| `results/` | CSVs and per-run logs (gitignored) |

### Appendix experiments (Stable Diffusion empirical validation)

| Path | What it does |
| --- | --- |
| [sd/hooks.py](sd/hooks.py) | `ActivationRecorder` — forward-hook helper for capturing intermediate UNet activations |
| [sd/utils.py](sd/utils.py) | Seeding, device selection, cosine/L2/L_inf/relative-error metrics, `MODELS_DIR` path |
| [exp3_sd_activation_divergence.py](exp3_sd_activation_divergence.py) | SD v1.4 — activation divergence under latent / prompt perturbation; produces `results/sd_activation_divergence.json` and `results/figures/sd_*.pdf` |
| [exp4_sd_inference_timing.py](exp4_sd_inference_timing.py) | SD v1.4 — 1 000 wall-clock inference passes with random prompts/seeds; produces `results/sd_inference_timing.json` |

### GPT-2 LLM experiments

| Path | What it does |
| --- | --- |
| [gpt2_experiments.py](gpt2_experiments.py) | Deterministic Ed25519-VRF GPT-2 sampling; estimates prefix collisions, token concentration, perplexity, and exact prompt/output KV-cache reuse |
| [complexity_experiment.py](complexity_experiment.py) | Fits DeepProve online cost (`inference_time + prove_full`) as a function of total sequence length |
| [run_complexity_pairs.py](run_complexity_pairs.py) | One-command shared-setup N,K sweep with fitted JSON and 3-D cost plot |
| [run_deepprove_work.py](run_deepprove_work.py) | Verified structural operation counts, five proof profiles, symbolic gas weights, and offline N,K calculator; see the [definition and report](../docs/features/deepprove_work_accounting.md) |
| [build_real_llm_trace_bank.py](build_real_llm_trace_bank.py) | Real GPT-2 challenge lengths plus verified CUDA DeepProve inference--proof timings for the trace-driven PoML campaigns |
| [gpt2_embedding_experiments.py](gpt2_embedding_experiments.py) | New-plan utility and challenge-conditioned quantized-trace separation experiments; includes target-token prompts, replay binding, and reusable-work timing |

The GPT-2 driver follows [the experimental plan](../docs/gpt2_plan.md). It uses
the public `gpt2` checkpoint, fixed token windows, inverse-CDF sampling from
the VRF transcript, EOS-aware prefixes, and prompt-level bootstrap intervals.
Install its optional dependencies with `pip install -e ".[dev,llm]"`; model
weights and benchmark datasets are downloaded by Hugging Face on first use.
Outputs are written below `experiments/results/gpt2/` (gitignored).

Run the deterministic DeepProve work validation with
`.venv/bin/python experiments/run_deepprove_work.py --prepare --output-dir experiments/results/deepprove_work/recheck`.
It uses CPU by default; `--cuda --device 0` selects a CUDA build. Add
`--compare-schedule experiments/results/deepprove_work/audited/schedule.json`
to validate against the established counts and stated componentwise tolerance
on another backend. After a
completed reference run, `.venv/bin/python experiments/run_deepprove_work.py --calculate 16:17`
computes the work vector without running GPT-2 or creating a proof.

SD weights are expected at `models/stable-diffusion/stable-diffusion-v1-4-fp16/`.
Run `python scripts/download_sd_model.py` once to fetch them.

## How PoML difficulty is calibrated

The 300 s target is derived from the reference log
[`logs/run_20260423_050539.log`](../logs/run_20260423_050539.log), which
produced **avg block time 214.28 s** at difficulty
`0x3fffff...ff` with 4 miners. Because per-block expected time is inversely
proportional to difficulty, we scale:

```
new_diff = old_diff * observed_time / target_time
```

For the 4-miner, 300 s case this gives roughly `0x2da6...`. The Exp 1 and
Exp 2 drivers recompute this at runtime so the calibration is self-documenting.

For the miner-count sweep we additionally rescale by `baseline_miners / N`
(aggregate proof rate ∝ N).

## How PoW difficulty is calibrated

`pow_calibrate.py` benchmarks SHA-256 on a single core for 5 s, scales by
`num_miners`, and picks `target_int = 2**256 // (hashrate * target_seconds)`.
Result is saved to `results/pow_calibration.json` and reused by Exp 1.

## Running

### Quick sanity check (no long waits)

```bash
python experiments/exp1_block_time_stability.py --smoke
python experiments/exp2_wasted_work.py --smoke
```

### Full experiments

```bash
# 1. Calibrate PoW difficulty (~10 s)
python experiments/pow_calibrate.py --target 300 --miners 4

# 2. Experiment 1: 50 blocks of PoML, then 50 blocks of PoW (~8+ hours total)
python experiments/exp1_block_time_stability.py --blocks 50 --target 300

# 3. Experiment 2: block-time sweep + miner-count sweep, 10 blocks each (~5 h)
python experiments/exp2_wasted_work.py --blocks 10

# 4. Experiment 5 smoke check
python experiments/exp5_uniform_fee_collisions.py smoke

# Complete Exp 5 workflow with progress and resumable timing calibration
python experiments/exp5_uniform_fee_collisions.py run-all

# Exp 5b: 1,000 shared random (Q, M) pairs at 300/600/900 s
python experiments/exp5b_ratio_scatter.py

# Regenerate Exp 5b heatmaps without rerunning the campaign
python experiments/exp5b_ratio_scatter.py --plot-runs experiments/results/exp5b_ratio_scatter/runs.csv

# Exp 5c: regular M-by-Q heatmap (3 block times * 75 cells * 100 seeds = 22,500 races)
python experiments/exp5c_m_q_heatmap.py

# Regenerate Exp 5c heatmaps without rerunning the campaign
python experiments/exp5c_m_q_heatmap.py --plot-runs experiments/results/exp5c_m_q_heatmap_100rep/runs.csv
```

### Appendix SD experiments

```bash
# One-off: fetch SD v1.4 weights into models/stable-diffusion/
python scripts/download_sd_model.py

# Exp 3: activation divergence (writes results + figures)
python experiments/exp3_sd_activation_divergence.py

# Exp 4: wall-clock inference timing
python experiments/exp4_sd_inference_timing.py
```

### GPT-2 LLM experiments

Start with a small wiring run (four prompts, four pairs, eight generated
tokens):

```bash
python experiments/gpt2_experiments.py collision --smoke
```

The default campaign is sized for roughly a few hours on a three-GPU host:
500 WikiText-2 prompts, four independent pairs per prompt, 32 generated tokens,
and the full five-temperature sweep:

```bash
python experiments/gpt2_experiments.py run-all \
  --device cuda --devices cuda:0,cuda:1,cuda:2 \
  --output-dir experiments/results/gpt2/full
```

With `--device cuda`, omitting `--devices` automatically assigns one worker
model to every visible GPU. The explicit list above is useful when pinning the
campaign to exactly three GPUs. Override `--prompts`, `--pairs`, `--length`, or
`--temperatures` for a larger plan-scale run; the default temperature list is
always `0.7,1.0,1.3,1.5,2.0`.
The cache timing diagnostic is intentionally fixed at `--temperature 1.0`.

Run the supporting measurements independently or together:

```bash
python experiments/gpt2_experiments.py concentration --smoke
python experiments/gpt2_experiments.py cache --prompts 1 --length 32
python experiments/gpt2_experiments.py perplexity --prompts 500
python experiments/gpt2_experiments.py run-all --smoke
```

The embedding-perturbation plan is a separate driver and does not modify the
original experiment:

```bash
# Both new experiments, practical defaults and resumable checkpoints
python experiments/gpt2_embedding_experiments.py run-all --device cuda

# Plan-scale runs can be launched independently
python experiments/gpt2_embedding_experiments.py utility --examples-per-task 1000
python experiments/gpt2_embedding_experiments.py separation --examples-per-task 100 --pairs 50

# Keep the campaign alive after disconnecting from a shell
scripts/run_gpt2_embedding_tmux.sh --device cuda --devices cuda:0,cuda:1
```

Use `--smoke` for a small wiring check.  Public structural strata can be
provided as newline-delimited prompt files with `--repeat-copy-file` and
`--resisting-correction-file`; smoke mode uses clearly labelled local
fallbacks when those datasets are not available.  Results are written under
`experiments/results/gpt2_embedding/` (or `--output-dir`) with utility and
trace-separation CSVs, replay counts, metadata, and checkpoints.

PIQA is loaded from the parquet-backed `regisss/piqa` mirror. If another
benchmark loader encounters a legacy `*.py` dataset script, the driver uses
the public Hugging Face Dataset Viewer fallback automatically. Structural
fallbacks use `tasksource/bigbench` and `pminervini/inverse-scaling`.

Pin a Hugging Face commit with `--revision <commit>` for a reproducible model
snapshot. `--prompt-file FILE` accepts one prompt source line per row and is a
useful offline fallback; each row must contain at least `--prompt-tokens`
GPT-2 tokens. Use `--device cuda` when a compatible GPU is available. Cache
timings are model-forward timings and do not imply reusable ZK-prover work.
The embedding utility and separation commands show live `tqdm` progress bars
and resume from their `*_checkpoint.jsonl` files when rerun with the same
settings. `--progress-every N` controls checkpoint-status refreshes;
incompatible checkpoints are archived as `.stale-*` before a new campaign
starts, and `--no-resume` remains available for a deliberate restart.

Separation work scales as prompts × sigmas × conditions × pairs. The default
`run-all` command tests 100 examples per standard task and caps each structural
stratum at 100; utility scoring uses 50 examples per task. Increase
`--examples-per-task` and `--max-structural-prompts` for the full released
strata. Sigma values, conditions, and pairs are unchanged.

### Outputs

- `results/exp1_poml_blocks.csv` — one row per block, time since previous.
- `results/exp1_pow_blocks.csv` — PoW equivalent with `miner_id`, `nonce`.
- `results/exp1_summary.csv` — one row per mechanism with mean/min/max/stdev/variance.
- `results/exp2_blocktime_sweep.csv`, `results/exp2_miner_sweep.csv` — one row per config
  with `completed_proofs`, `included_proofs`, `wasted_proofs`, `wasted_ratio`, and
  mean block time.
- `results/exp5_<campaign>/` — Exp 5 campaign provenance, 27,000 run summaries,
  27 configuration summaries, first-five detailed records, and three heatmaps.
- `results/exp5b_ratio_scatter/` — Exp 5b pair design, 3,000 run summaries,
  and one PNG/PDF wasted-work heatmap per target block time. Each populated
  cell reports the median percentage and sample count for one of six
  logarithmic `Q/M` bands and four miner-count bands; gray cells had no
  sampled feasible configurations.
- `results/exp5c_m_q_heatmap_100rep/` — Exp 5c grid definition, 22,500 run
  summaries (3 block times * 75 feasible cells * 100 attempted seeds), per-cell
  `mean ± std` wasted-work aggregates, and one PNG/PDF M-by-Q heatmap per target
  block time. M runs along the x-axis (10..10,000) and Q along the y-axis
  (20..100,000), so larger values are in the top-right corner; cells with
  `Q <= M` are masked gray. Each summary row reports both attempted
  `replicates` and adopted `adopted_runs`; no-winner races are excluded from
  wasted-work statistics.
- `results/logs/exp1_poml_*.log`, `results/logs/exp2_*.log` — the full PoML run
  log for each configuration (used by the driver to count completed proofs).
- `results/gpt2/` — GPT-2 collision/concentration/cache CSVs and plots,
  perplexity sanity result, metadata, and optional VRF transcript JSONL.
- `results/gpt2_embedding/` — utility summaries, the normalized utility graph
  and metric-change plots, quantized trace tables/heatmaps and
  prefix-collision tables/plots, token-agreement and replay results, prompt
  manifests, and optional inference reuse timing.

## Wasted-work metric definitions

- `completed_proofs` — every `Miner X: proof N/M done` line in the run's log,
  i.e. every proof any miner finished (WIN or miss), including those immediately
  invalidated when another miner's block landed.
- `included_proofs` — sum of `lottery_attempts` recorded on confirmed blocks
  (this is `len(block.results)`, the chain of cipher-texts in the winning block).
- `wasted_proofs = completed_proofs - included_proofs`
- `wasted_ratio = wasted_proofs / completed_proofs`

The expected pattern: wasted ratio rises with more miners (more parallel work
gets discarded each time someone wins) and falls with longer expected block
times (each block amortizes more useful work).
