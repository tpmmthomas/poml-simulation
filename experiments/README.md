# PoML Experiments

Drivers for two empirical experiments on the PoML simulator, plus a simplified
Bitcoin-style PoW baseline.

## Files

| Path | What it does |
| --- | --- |
| [utils.py](utils.py) | Shared helpers: `run_poml`, log parsing, difficulty rescaling, statistics |
| [pow_sim.py](pow_sim.py) | Multi-process SHA-256 PoW simulator (lottery only — no blockchain) |
| [pow_calibrate.py](pow_calibrate.py) | Benchmarks local SHA-256 hashrate, writes a difficulty target for a chosen expected block time |
| [exp1_block_time_stability.py](exp1_block_time_stability.py) | 50-block PoML and PoW runs at 300 s target; emits mean/min/max/stdev/variance |
| [exp2_wasted_work.py](exp2_wasted_work.py) | PoML sweeps over expected block time (200/300/400 s) and miner count (2/4/8), 10 blocks each |
| `results/` | CSVs and per-run logs (gitignored) |

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
```

### Outputs

- `results/exp1_poml_blocks.csv` — one row per block, time since previous.
- `results/exp1_pow_blocks.csv` — PoW equivalent with `miner_id`, `nonce`.
- `results/exp1_summary.csv` — one row per mechanism with mean/min/max/stdev/variance.
- `results/exp2_blocktime_sweep.csv`, `results/exp2_miner_sweep.csv` — one row per config
  with `completed_proofs`, `included_proofs`, `wasted_proofs`, `wasted_ratio`, and
  mean block time.
- `results/logs/exp1_poml_*.log`, `results/logs/exp2_*.log` — the full PoML run
  log for each configuration (used by the driver to count completed proofs).

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
