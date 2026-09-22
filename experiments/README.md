# Paper experiments

Run these commands from the repository root after installing the relevant extras.
The [experiment guide](../docs/experiments.md) describes all options and caveats.

| Entry point | Purpose |
| --- | --- |
| `measure_pairs.py` | Fresh GPT-2/DeepProve or tiny-U-Net/EZKL inference/proof measurements |
| `liveness.py` | PoML block intervals, actual/Poisson PoW comparison, complexity/runtime plot |
| `wasted_work.py` | Completed duplicate-work grid with first-completion credit |
| `diffusion_compatibility.py` | Full SD v1-4 DDPM latent-trajectory experiment |
| `gpt2_compatibility.py` | Float32 utility and independent activation trajectories |
| `complexity_counts.py` | Instrumented DeepProve counts, validation and offline formula calculator |
| `fit_complexity.py` | Nonnegative operation-weight fitting with grouped validation |

Only experiments corresponding to the current paper are retained. Generated
measurements, proofs, ledgers, manifests and figures go under ignored `results/`.
