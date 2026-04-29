# Running the Experiments

This guide covers every step needed to reproduce the PoML vs. PoW simulator
experiments (Exp 1–2) and the Stable Diffusion appendix experiments (Exp 3–4).
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
