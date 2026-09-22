# Publication verification

Verified on 2026-09-22 with Python 3.11, Linux, the versions pinned in
`pyproject.toml`, RTX A6000 GPUs and the pinned DeepProve/CUDA worker.
The checks below are independent runs of this publication implementation.
Generated evidence is local under ignored `experiments/results/fc27/`.

| Check | Result / scope |
| --- | --- |
| Automated tests | 138 passed; protocol and rejection tests, arithmetic, cryptographic plumbing, fitting, CLI, real CPU PoW, local tiny GPT-2, DDPM scheduler comparison |
| Style | Ruff 0.15.6 check and format |
| Distribution | Wheel and source archive build; clean wheel install outside checkout ran three blocks with no Torch installed |
| Clean source installation | 129 passed, two optional model-test modules skipped; calculator, liveness and waste commands ran from the extracted source archive |
| GPT-2 simulator | Accepted blocks with fresh verified DeepProve proofs under uniform and fitted weights; context-64 setup reused |
| Fresh GPT-2 measurements | Eight verified proofs, four WikiText prompts × two repetitions; N=8/16/24/32, K cap=1 |
| Runtime fit | All eight measurements consumed; prompt-grouped nested validation and nonnegative integer weights exported; rank 4 |
| EZKL setup | New seed-42 tiny-U-Net ONNX export, calibration, circuit, SRS and proving/verification keys built from scratch |
| EZKL simulator and liveness | Fresh real proofs verified; accepted blocks using both existing and newly built setup artifacts |
| Prover preparation | All three patches apply cleanly and idempotently to pristine pinned checkouts; unrelated upstream demo LFS downloads skipped |
| Operation instrumentation | Fresh CUDA inference/proof at N=2,K=62; recorded inference and proof vectors exactly match the frozen reference |
| Offline count tools | 1,953 allowed N/K costs exported; campaign plan has 40 distinct pairs, 24 total lengths, 48 trials including controls |
| Liveness replay | 50 adopted blocks using the new eight-proof bank; Poisson comparison explicitly labelled |
| Actual PoW command | Two measured double-SHA-256 races via the liveness command, 0.02-second calibration target for command testing |
| Waste replay | 60 measured-bank races: M=10/100, Q=20/100, targets 300/600/900, five repeats; CSV and heatmaps generated |
| Stable Diffusion | Real SD v1-4, one sample, four DDPM steps, 256-equivalent latents, σ=.001 plus independent baseline |
| GPT-2 compatibility | Real pretrained GPT-2 utility and independent trace commands on explicit small token fixtures; all six real dataset loaders separately checked |

The default `pytest` suite does not download checkpoints or generate large ML
proofs. Tests requiring Torch/Transformers or Diffusers skip explicitly if those
extras are absent. Real model/prover checks are the explicit commands below.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pytest -q
ruff check src tests experiments scripts model
ruff format --check src tests experiments scripts model
uv build --out-dir experiments/results/dist
```

For a real proof smoke check, use the one-block commands in the
[README](../README.md). For a fresh instrumented counter check:

```bash
python experiments/complexity_counts.py --cuda --device 0 --pairs 2:62 \
  --compare-schedule src/poml_sim/data/gpt2_reference_schedule.json \
  --output-dir experiments/results/count-check
```

The real counter check rebuilt `bench-llm` and its setup, generated a proof and
verified it. The complete 40-pair instrumented campaign was **not** rerun here.
The full 1,000-input diffusion campaign, 100-example/task GPT-2 campaigns,
36,000-race waste grid and 50-block fresh-proof/hashing campaign were **not**
rerun here. The shipped commands expose those defaults; smoke runs validate
execution and semantics, not the manuscript's empirical claims.

The small fitting bank deliberately has very limited shape variation and was
collected with other host activity. Its errors are command-validation output,
not a replacement for the paper's calibration results. The liveness run also
retains its achieved mean instead of forcing the target: approximate calibration
and finite sampling can differ substantially from 300 seconds.

Resource observations: the new EZKL proving key is about 9 GiB; full GPT-2
context-64 proving can require most of a 48 GiB GPU. Setup is excluded from
online service time. Proof verification failures, missing artifacts, changed
proof digests and incompatible count/weight schedules raise errors.
