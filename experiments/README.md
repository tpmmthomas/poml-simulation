# Experiments

The scripts in this directory implement the experiments named in the paper.
Start with the [experiment guide](../docs/experiments.md), which explains the
order in which a reader encounters the concepts and the meaning of each metric.

| Paper experiment | Entry point | Purpose |
| --- | --- | --- |
| Fresh inference–proof measurements | `measure_pairs.py` | Build the verified `(T,C)` bank used by the replay experiments |
| PoML Liveness and Block Generation Stability | `liveness.py` | Compare PoML block intervals with a PoW hash lottery or a Poisson control |
| Wasted Work Analysis | `wasted_work.py` | Measure duplicate completed complexity under a fixed query pool |
| DDPM Compatibility: Formal Statements and Experiments | `diffusion_compatibility.py` | Compare Stable Diffusion latent trajectories under perturbation |
| LLM Compatibility: Formal Statements and Experiments | `gpt2_compatibility.py` | Measure GPT-2 utility and layer-wise activation separation |
| A Reference Complexity Function for GPT-2 and DeepProve | `complexity_counts.py` | Count instrumented operations and validate the public schedule |
| A Reference Complexity Function for GPT-2 and DeepProve | `fit_complexity.py` | Fit nonnegative runtime weights with grouped validation |


## Quick checks

```bash
python experiments/liveness.py --smoke --blocks 3 --pow-mode poisson \
  --output experiments/results/liveness-smoke
python experiments/wasted_work.py --smoke --miners 2,4 --queries 4,20 \
  --targets 300 --repeats 3 --output experiments/results/waste-smoke
```

These commands use synthetic timing fixtures. They check wiring and output
schemas; they are not model measurements.
