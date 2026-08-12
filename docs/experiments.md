# Running the Experiments

This guide covers every step needed to reproduce the PoML vs. PoW simulator
experiments (Exp 1–2), the Stable Diffusion appendix experiments (Exp 3–4),
and the uniform-fee collision experiment (Exp 5).
For a description of what each experiment measures, see
[experiments/README.md](../experiments/README.md).

---

## Prerequisites

Complete the simulation setup before running any experiment:

```bash
pip install -e ".[dev]"
bash scripts/setup_model.sh   # export ONNX model + generate EZKL artifacts (one-time)
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

## Experiment 5 — Uniform-Fee Query Collisions

Experiment 5 measures duplicate completed inference-proof work in one-block
races over a fixed pool of equivalent, uniform-fee queries. The final grid is:

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

### 2. Measure real EZKL attempt times

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

### 3. Calibrate fixed difficulties

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

### 4. Run the frozen final campaign

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
