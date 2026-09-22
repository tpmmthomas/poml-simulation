# Running the Experiments

The current paper evaluation uses the three commands documented in
[features/llm_poml_experiments.md](features/llm_poml_experiments.md). The older
EZKL/DDPM commands below remain available for implementation regression tests,
but they are no longer the main paper experiments.

This guide covers every step needed to reproduce the PoML vs. PoW simulator
experiments (Exp 1–2), the Stable Diffusion appendix experiments (Exp 3–4),
the uniform-fee collision experiment (Exp 5), and the GPT-2 LLM collision
experiments described in [gpt2_plan.md](gpt2_plan.md).
For a description of what each experiment measures, see
[experiments/README.md](../experiments/README.md).

---

## Prerequisites

Complete the simulation setup before running any experiment:

```bash
pip install -e ".[dev]"
bash scripts/setup_model.sh   # export ONNX model + generate EZKL artifacts (one-time)
```

The GPT-2 driver has separate optional dependencies and does not require EZKL:

```bash
pip install -e ".[dev,llm]"
```

---

## Simulator Experiments (Exp 1 & 2)

### 1. Calibrate PoW difficulty (~10 s)

Must be run before Exp 1 or 2. Benchmarks local SHA-256 throughput and writes
a difficulty target to `experiments/results/pow_calibration.json`.

```bash
python experiments/pow_calibrate.py --target 300 --miners 4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--target` | `300` | Expected PoW block time in seconds |
| `--miners` | `4` | Number of miner processes to simulate |

### 2. Experiment 1 — Block-time stability (~8 h for full run)

Runs 50-block PoML and 50-block PoW races at a 300 s block-time target and
reports mean / min / max / stdev / variance.

```bash
# Smoke test (fast, ~2 blocks each)
python experiments/exp1_block_time_stability.py --smoke

# Full run
python experiments/exp1_block_time_stability.py --blocks 50 --target 300
```

| Flag | Default | Description |
|------|---------|-------------|
| `--blocks` | `50` | Number of blocks per mechanism |
| `--target` | `300` | Expected block time in seconds |
| `--smoke` | off | Run 2 blocks each for a quick sanity check |

**Outputs**

| File | Contents |
|------|----------|
| `results/exp1_poml_blocks.csv` | One row per PoML block with elapsed time |
| `results/exp1_pow_blocks.csv` | One row per PoW block with `miner_id`, `nonce` |
| `results/exp1_summary.csv` | One row per mechanism: mean / min / max / stdev / variance |
| `results/logs/exp1_poml_*.log` | Full PoML simulation log |

### 3. Experiment 2 — Wasted work (~5 h for full run)

Sweeps over expected block times (200 / 300 / 400 s) and miner counts (2 / 4 / 8),
running 10 blocks per configuration. Reports `wasted_ratio` (orphaned proofs /
total proofs) for each setting.

```bash
# Smoke test
python experiments/exp2_wasted_work.py --smoke

# Full run
python experiments/exp2_wasted_work.py --blocks 10
```

| Flag | Default | Description |
|------|---------|-------------|
| `--blocks` | `10` | Blocks per configuration |
| `--smoke` | off | Run 2 blocks per config |

**Outputs**

| File | Contents |
|------|----------|
| `results/exp2_blocktime_sweep.csv` | Block-time sweep: one row per target time |
| `results/exp2_miner_sweep.csv` | Miner-count sweep: one row per miner count |
| `results/logs/exp2_*.log` | Full simulation logs |

Each CSV row contains `completed_proofs`, `included_proofs`, `wasted_proofs`,
`wasted_ratio`, and mean block time.

---

## GPT-2 LLM collision and cache experiments

`experiments/gpt2_experiments.py` implements the concrete protocol in
`docs/gpt2_plan.md`: fixed GPT-2 BPE prompt windows, the repository's
Ed25519 sign-then-hash VRF as a deterministic per-step random source,
inverse-CDF temperature sampling, EOS-aware prefix indicators, and
prompt-level bootstrap intervals. The default benchmark is the raw
WikiText-2 test split; LAMBADA is selected with `--dataset lambada`.

Run a smoke check first:

```bash
python experiments/gpt2_experiments.py collision --smoke
```

The default campaign is 500 prompts × 4 pairs × 32 generated tokens over all
five temperatures (`0.7,1.0,1.3,1.5,2.0`). On a three-GPU host, run all
measurements with:

```bash
python experiments/gpt2_experiments.py run-all \
  --device cuda --devices cuda:0,cuda:1,cuda:2 \
  --output-dir experiments/results/gpt2/full
```

`--device cuda` alone uses every visible GPU; `--devices` is optional and
pins the worker list explicitly. Each GPU receives its own model copy and
processes a round-robin share of prompt/temperature units. The runtime target
is hardware-dependent, so treat the 2–3 hour sizing as an estimate. The
temperature list applies to collision and concentration; cache timing stays at
its fixed `--temperature 1.0` diagnostic default.

Supporting commands are `concentration`, `perplexity`, and `cache`; use
`run-all --smoke` to exercise all four outputs. `--revision` pins a model
commit, and `--prompt-file` allows an offline newline-delimited prompt source.
Results include CSV data, PNG plots, `perplexity.json`, and `metadata.json`.
The cache command measures inference-state reuse only; it is not a claim that
ZK-prover witnesses or proofs can be reused. It also emits a bounded-LRU
workload model; adjust it with `--repeat-probability` and
`--cache-capacities 1,4,16,64`.

Collision and concentration runs print periodic prompt progress and write
append-only checkpoints under their output directory. Rerun the same command
after an interruption to resume; use `--progress-every 1` for every prompt or
`--no-resume` to discard the checkpoint and start over.

## GPT-2 embedding-perturbation experiments

The new plan is implemented by the separate
`experiments/gpt2_embedding_experiments.py` driver. It leaves the sampler/cache
driver above unchanged and provides two commands:

```bash
# Practical run-all defaults (checkpointed and multi-GPU aware)
python experiments/gpt2_embedding_experiments.py run-all --device cuda

# Individual plan experiments
python experiments/gpt2_embedding_experiments.py utility --examples-per-task 1000
python experiments/gpt2_embedding_experiments.py separation --examples-per-task 100 --pairs 50

# Faster separation pilot (caps the large structural stratum)
python experiments/gpt2_embedding_experiments.py separation \
  --max-structural-prompts 100 --sigmas 0,0.01,0.02 --pairs 2 --length 16 \
  --target-tokens 4

# Persistent shell for a long campaign
scripts/run_gpt2_embedding_tmux.sh --device cuda --devices cuda:0,cuda:1
```

`utility` reports clean versus one-time prefix-perturbed perplexity/accuracy
with paired bootstrap intervals over WikiText-2, LAMBADA, HellaSwag, PIQA, and
ARC-Easy. `separation` compares whole quantized boundaries under independent,
common-token-stream, and full-protocol challenges, reports changed-coordinate
fractions and generated-prefix collisions, and writes a challenge-bound trace
commitment replay check. Pass newline-delimited public prompt files with
`--repeat-copy-file` and `--resisting-correction-file`; `--smoke` uses labelled
local structural fallbacks and synthetic utility rows, so benchmark downloads
are not required for the wiring check. The optional
reuse table marks `rho_pf` as `not_integrated` until DeepProve proving is wired
in. See [the implementation note](features/gpt2_embedding_experiments.md) for
the output schema and scale/runtime guidance.

For the planned maximum-length sensitivity, repeat the collision command with
`--length 16`, `--length 32`, and `--length 64` (using separate output
directories). Add `--record-transcripts` when an independently verifiable
per-step VRF audit is required; omit it for the full campaign unless the large
JSONL artifact is needed.

---

## Experiment 5 — Uniform-Fee Query Collisions

Experiment 5 measures duplicate completed inference-proof work in one-block
races over a fixed pool of equivalent, uniform-fee queries.
For the complete protocol, calibration provenance, final results, and an
implementation-independent reproduction specification, see
[Experiment 5: Uniform-Fee Query Collisions](experiment_5.md).

The final grid is:

| Parameter | Values |
|------|---------|
| Miner count $M$ | 10, 100, 1000 |
| Target block time $\tau$ | 300, 600, 900 seconds |
| Query-pool size $Q$ | 1000, 5000, 10000 |
| Final replicates | 1000 per configuration (27,000 races) |

Running 1000 concurrent real EZKL workers is not feasible on the reference
host. Instead, the experiment first measures 30 isolated real
inference-plus-proof durations, then samples those empirical durations with
replacement in an accelerated discrete-event simulation. One simulated miner
therefore represents one independently provisioned, homogeneous node with the
same isolated resource allocation; it does not model 1000 miners contending
for one 16-core host.

Each miner consumes an independently and uniformly shuffled permutation of
query IDs $0,\ldots,Q-1$. A query appears at most once in one miner's prefix but
may be completed by many miners. Query identity cannot affect cost: all work
uses the fixed timing-calibration input and duration draws are independent of
query ID. Every query has fee 1 and one output. Account registration, fee
reservation, and fee debit are outside this accelerated experiment; presence
in the initialized pool defines eligibility.

### Event and metric semantics

Events are ordered by virtual completion time, then a run-seeded random tie
key, then insertion sequence. The first winning completion is immediately
adopted by the coordinator; post-adoption network propagation is irrelevant to
this one-block boundary. A completion at the same virtual time but ordered
after adoption is cancelled, has zero remaining time, and is not included in
the denominator. This preserves coordinator-emitted order without adding a
latency or consensus tie-breaking model.

For each adopted run:

$$
	ext{collision ratio}
=\frac{\sum_q \max(0, c_q-1)}{\sum_q c_q}
=\frac{\text{completed pairs}-\text{unique completed queries}}
{\text{completed pairs}},
$$

where $c_q$ counts full inference-plus-proof completions before adoption. The
winning pair is included. Every started but unfinished combined attempt is one
discarded partial; detailed records include elapsed and remaining virtual
time. An attempt scheduled at exactly the adoption time but ordered after the
adoption event is retained as `adoption_preceded_tied_event`; it has zero
remaining time and is reported separately, not counted as a partially
completed proof. A run where all miners exhaust their permutations without a
winner is recorded as failed, excluded from the mean ratio, never assigned a
replacement seed, and causes the final campaign command to exit nonzero.

### 1. Smoke test

The smoke command uses an explicitly synthetic timing fixture and cannot
produce final results:

```bash
python experiments/exp5_uniform_fee_collisions.py smoke \
    --output-dir experiments/results/exp5_smoke
```

### 2. Run the complete workflow

The three calibration/collection stages can be run with one command:

```bash
python experiments/exp5_uniform_fee_collisions.py run-all
```

This prints stage progress such as:

```text
[run-all] Stage 1/3: timing calibration
[timing] starting attempt 1 (0/30 successful)
[timing] attempt 1/30 still running (10s elapsed); press Ctrl+C to stop
[timing] completed 1/30 successful (86.421s)
[run-all] Stage 1/3 complete
[run-all] Stage 2/3: difficulty calibration
[difficulty] cell 1/9: M=10, target=300s, running 1000 races
[difficulty] cell 1/9: 100/1000 races complete
...
[run-all] Stage 3/3: final collision campaign
[final] config 1/27: M=10, target=300s, Q=1000; running 1000 races
[final] config 1/27: 100/1000 races complete
```

The default reusable artifacts are written to
`experiments/results/exp5_run_all/`; final output is written to
`experiments/results/exp5_final_2026_08_12/`. Existing valid timing and
difficulty artifacts are reused on subsequent `run-all` invocations. The
final output directory remains write-once and must be moved or renamed before
starting a new final campaign.

Use the options below to change scale or paths:

```bash
python experiments/exp5_uniform_fee_collisions.py run-all \
    --attempts 30 \
    --verification-runs 1000 \
    --cpu-limit 16 \
    --work-dir experiments/results/exp5_run_all \
    --output-dir experiments/results/exp5_final_2026_08_12
```

During timing calibration, each native EZKL attempt runs in a child process.
The parent prints a heartbeat every 10 seconds and remains responsive to
`Ctrl+C`. Interrupting an attempt terminates the child and writes
`timing_calibration.json.partial.json`. Re-run `run-all` to resume that
checkpoint, or run the timing command directly with `--resume`:

```bash
python experiments/exp5_uniform_fee_collisions.py calibrate-timings \
    --attempts 30 \
    --cpu-limit 16 \
    --output experiments/results/exp5_run_all/timing_calibration.json \
    --resume
```

### 3. Measure real EZKL attempt times

Run from a clean committed checkout. The command pins the process to 16 logical
CPUs, fixes Rayon/BLAS thread limits, uses one fixed valid input, records 30
successful full inference-plus-proof durations, and retains failures and host
metadata. It does not discard successful outliers.

```bash
python experiments/exp5_uniform_fee_collisions.py calibrate-timings \
    --attempts 30 \
    --cpu-limit 16 \
    --output experiments/results/exp5_timing_calibration.json
```

### 4. Calibrate fixed difficulties

For each $(M,\tau)$ pair, the threshold is calculated from the empirical
arithmetic mean $E[T]$:

$$
D=\left\lfloor 2^{256}\frac{E[T]}{M\tau}\right\rfloor.
$$

The command runs 1000 disjoint-seed verification races per pair and reports
realized adoption-time statistics and 95% normal confidence intervals. It
never retunes the threshold and never uses collision outcomes for calibration.
The same fixed difficulty is reused across all three $Q$ values for a given
$(M,\tau)$ pair.

```bash
python experiments/exp5_uniform_fee_collisions.py calibrate-difficulty \
    --timings experiments/results/exp5_timing_calibration.json \
    --verification-runs 1000 \
    --output experiments/results/exp5_difficulty_calibration.json
```

### 5. Run the frozen final campaign

The committed `experiments/exp5_seed_manifest.json` fixes separate public seed
roots for smoke, pilot, timing, difficulty calibration, and final runs. Each
random stream is domain-separated for query ordering, duration draws, lottery
draws, and equal-time ordering. Final seeds are SHA-256-derived from the frozen
root and $(M,\tau,Q,\text{replicate})$; all 27,000 derived seeds are unique.

```bash
python experiments/exp5_uniform_fee_collisions.py run-final \
    --timings experiments/results/exp5_timing_calibration.json \
    --calibration experiments/results/exp5_difficulty_calibration.json \
    --output-dir experiments/results/exp5_final_2026_08_12
```

Final collection requires timing and difficulty artifacts from the same clean
simulator commit and identical model/proof artifact digests. It refuses to
overwrite a non-empty campaign directory. Virtual races are deterministic;
remaining nondeterminism is confined to the real wall-clock timing calibration
and is recorded in that artifact.

**Outputs**

| File | Contents |
|------|----------|
| `campaign.json` | Commit, complete configuration map, environment, model/proof digests, seed and calibration provenance |
| `runs.csv` | All 27,000 run IDs, seeds, adoption times, collision metrics, partial-work metrics, and winners |
| `summary.csv` | One row per configuration; heatmaps use the arithmetic mean of adopted run ratios; pooled ratios, variation, and tied-cancellation counts are also retained |
| `details/<config_id>/runs.json` | Ordered per-miner completed query lists and full run records for frozen replicates 0–4 |
| `details/<config_id>/attempts.csv` | Every completed and cancelled attempt for frozen replicates 0–4 |
| `figures/collision_heatmap_target_{300,600,900}s.{png,pdf}` | Three separate annotated heatmaps using one shared 0-to-global-maximum color scale |

Full attempt-level and per-miner-list records are deliberately retained only
for the first five of 1000 replicates per configuration. This is the selected
storage policy; aggregate run records are retained for every replicate.

### Experiment 5b: random Q/M ratio sweep

Experiment 5b reuses the same race semantics and empirical timing calibration,
but replaces the 27-cell repeated grid with 1,000 unique random $(Q,M)$ pairs.
The default design places 20 pairs in each of 50 logarithmic $Q/M$ bins, samples
feasible miner counts log-uniformly, and enforces
$1\le M<Q\le100{,}000$. The same 1,000 pairs are run once at each target block
time so the three plots are directly comparable.

```bash
python experiments/exp5b_ratio_scatter.py
```

The command reads the existing Exp 5 timing artifact from
`experiments/results/exp5_run_all/timing_calibration.json`, derives a lottery
difficulty for each $(M,\tau)$ pair, and writes to
`experiments/results/exp5b_ratio_scatter/`. Use `--pairs`, `--ratio-bins`,
`--max-value`, `--targets`, `--seed-root`, `--timings`, or `--output-dir` to
override the design. Output directories are write-once.

| File | Contents |
|------|----------|
| `campaign.json` | Pair design, seed root, timing provenance, environment, and status counts |
| `runs.csv` | One run-level row per pair and target, including `Q/M`, collision ratio, and percentage of wasted work |
| `figures/wasted_work_scatter_target_{300,600,900}s.{png,pdf}` | Log-$x$ scatter plots colored by miner count; no-winner runs are reported but omitted from the points |

### Experiment 5c: regular M-by-Q heatmap

Experiment 5c keeps the same accelerated race engine and timing calibration as
Experiments 5 and 5b, but evaluates a regular log-spaced grid of exact $(M,Q)$
pairs. It is designed for a readable paper figure while retaining the
low-$Q/M$ dynamic regime and the near-zero high-$Q/M$ regime seen in Exp 5b.

| Parameter | Values |
|------|---------|
| Miner count $M$ | 10, 20, 50, 100, 200, 500, 1,000, 2,000, 5,000, 10,000 |
| Query-pool size $Q$ | 20, 50, 100, 200, 500, 1,000, 2,000, 5,000, 10,000, 20,000, 50,000, 100,000 |
| Feasible cells | 75, retaining only $Q>M$ |
| Target block time | 300, 600, 900 seconds |
| Attempts per cell | 100 |
| Total races | 22,500 |

The cell value is the mean percentage of wasted work across adopted races. The
annotation gives the mean and sample standard deviation as `mean% ± sd%`.
The completed campaign produced 22,413 adopted races and 87 `no_winner`
races; the latter are reported in `runs.csv` and excluded from the cell
statistics because they have no adopted block or collision ratio.

The heatmaps use miners $M$ on the horizontal axis, increasing to the right,
and query-pool size $Q$ on the vertical axis, increasing upward. Thus large
values occupy the top-right, while infeasible $Q\le M$ cells are gray. Green
denotes less wasted work and red denotes more wasted work, with black or white
annotations selected for contrast.

```bash
python experiments/exp5c_m_q_heatmap.py
```

The command writes to `experiments/results/exp5c_m_q_heatmap_100rep/` and
refuses to overwrite an existing campaign. To regenerate only the figures:

```bash
python experiments/exp5c_m_q_heatmap.py \
    --plot-runs experiments/results/exp5c_m_q_heatmap_100rep/runs.csv
```

| File | Contents |
|------|----------|
| `campaign.json` | Grid, replicate count, seed root, timing provenance, environment, and status counts |
| `runs.csv` | One row per attempted target/cell/seed race, including status and wasted-work percentage |
| `summary.csv` | One row per target/cell with attempted count, adopted count, mean, sample standard deviation, minimum, and maximum |
| `figures/wasted_work_heatmap_m_q_target_{300,600,900}s.{png,pdf}` | Annotated red-to-green M-by-Q heatmaps |

---

## Appendix Experiments — Stable Diffusion CIA Validation (Exp 3 & 4)

These experiments empirically validate the Computational Independence of
Activations (CIA) property using Stable Diffusion v1.4.

### One-time setup: download SD weights

```bash
python scripts/download_sd_model.py
```

Weights are saved to `models/stable-diffusion/stable-diffusion-v1-4-fp16/`.

### 4. Experiment 3 — Activation divergence

Measures how intermediate UNet activations diverge under small perturbations
to the latent or the prompt.

```bash
python experiments/exp3_sd_activation_divergence.py
```

**Outputs**

| File | Contents |
|------|----------|
| `results/sd_activation_divergence.json` | Per-layer cosine / L2 / L∞ / relative-error metrics |
| `results/figures/sd_*.pdf` | Publication-ready figures |

### 5. Experiment 4 — Inference timing

Runs 1 000 wall-clock inference passes with random prompts and seeds to
characterise timing variance.

```bash
python experiments/exp4_sd_inference_timing.py
```

**Output**: `results/sd_inference_timing.json` — per-run wall-clock times and
aggregate statistics.

---

## Recommended Run Order

```
scripts/setup_model.sh
└── experiments/pow_calibrate.py
    ├── experiments/exp1_block_time_stability.py
    └── experiments/exp2_wasted_work.py

scripts/download_sd_model.py
├── experiments/exp3_sd_activation_divergence.py
└── experiments/exp4_sd_inference_timing.py

experiments/exp5_uniform_fee_collisions.py
├── calibrate-timings
├── calibrate-difficulty
└── run-final

experiments/exp5b_ratio_scatter.py
└── reuse Exp 5 timings → 3,000 ratio-sweep races and scatter plots

experiments/exp5c_m_q_heatmap.py
└── reuse Exp 5 timings → 22,500 regular-grid races and M-by-Q heatmaps
```

Exp 1 and Exp 2 are independent of the SD experiments and can be run in
parallel on separate machines.

---

## Plotting Results

After Exp 1 and Exp 2 complete, generate figures with:

```bash
python experiments/plot_exp1_proof_times.py
python experiments/plot_exp2_wasted_work.py
```

Figures are written to `experiments/results/figures/`.
