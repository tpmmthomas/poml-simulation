# Verification Record

This record describes what has been checked in the current implementation and
what a check means. It does not turn reduced runs into the paper's empirical
results.

The commands below were verified on 2026-09-25 with Python 3.11 on Linux, the
versions pinned in `pyproject.toml`, an RTX A6000 GPU, and the pinned
DeepProve/CUDA worker. Generated evidence is kept under the ignored
`experiments/results/fc27/` directory.

## Checks completed

| Check | Result / scope |
| --- | --- |
| Automated tests | 153 passed, including protocol rejection, arithmetic, cryptographic plumbing, fitting, CLI, CPU PoW, local GPT-2, DDPM scheduler checks, and prefix-collision accounting |
| Style | Ruff 0.15.6 check and format passed |
| Distribution | Wheel and source archive built; a clean wheel install ran three blocks outside the checkout without Torch |
| Clean source installation | 129 passed and two optional model-test modules skipped; calculator, liveness, and wasted-work commands ran from the extracted archive |
| GPT-2 simulator | Fresh verified DeepProve proofs accepted under uniform and fitted weights with the context-64 setup |
| Fresh measurements | Eight verified GPT-2 inference–proof pairs from four WikiText-2 prompts and two replicates |
| Runtime fit | Prompt-grouped nested validation exported nonnegative integer weights; feature rank four |
| Tiny diffusion simulator | A new seed-42 EZKL setup, circuit, SRS, proving key, and verification key were built from scratch; fresh proofs verified |
| Prover preparation | All protocol patches applied cleanly and idempotently to pristine pinned checkouts |
| Operation instrumentation | Fresh CUDA inference/proof at `N=2, K=62` matched the frozen reference vectors exactly |
| Offline complexity plan | 1,953 supported `N/K` costs exported; the plan contains 40 distinct pairs and 24 total lengths |
| PoML Liveness and Block Generation Stability | A 50-block measured replay completed; the Poisson comparison is explicitly labeled as a statistical control |
| Wasted Work Analysis | Measured-bank races generated CSVs and heatmaps, including exhaustion records |
| DDPM Compatibility: Formal Statements and Experiments | A real Stable Diffusion v1-4 run completed with one sample, four DDPM steps, and `sigma=.001` plus an independent baseline |
| LLM Compatibility: Formal Statements and Experiments | Real GPT-2 utility and independent-decoding trace commands completed on explicit small fixtures; all six dataset loaders were checked separately |
| GPT-2 generated-prefix collisions | Reduced local campaign verifies EOS-censored prefix counts, first-divergence accounting, CSV output, and PNG/PDF plot generation; the 500-prompt campaign remains a GPU run |

## Reproduce the basic checks

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=src .venv/bin/python -m pytest -q
uvx --from ruff==0.15.6 ruff check src tests experiments scripts model
uvx --from ruff==0.15.6 ruff format --check src tests experiments scripts model
uv build --out-dir experiments/results/dist
```

For a real proof smoke check, use the one-block commands in the
[README](../README.md). For a fresh instrumented complexity check:

```bash
python experiments/complexity_counts.py --cuda --device 0 --pairs 2:62 \
  --compare-schedule src/poml_sim/data/gpt2_reference_schedule.json \
  --output-dir experiments/results/count-check
```

## Scope of the evidence

The full 40-pair instrumented campaign, the 1,000-input DDPM Compatibility
experiment, the 100-example-per-task LLM Compatibility experiments, the
500-prompt GPT-2 prefix-collision campaign, the full Wasted Work Analysis grid,
and the 50-block fresh-proof and hashing campaign were not rerun during this
verification pass. Their commands expose the paper's defaults; reduced runs
validate execution and metric semantics only.

The small fitting bank has limited shape variation and was collected while the
host had other activity. Its errors validate the command, not the paper's
calibration results. Liveness retains the achieved mean rather than forcing
the 300-second target because calibration is approximate and finite sampling
can differ from the target.

## Resource and interpretation notes

- A fresh EZKL proving key is about 9 GiB.
- Full GPT-2 context-64 proving can use most of a 48 GiB GPU.
- Setup is excluded from online inference–proof service time.
- Proof-verification failures, changed proof digests, and incompatible
  complexity schedules raise errors rather than being silently substituted.
- The complete private PoML NP relation is abstracted by a trusted-host
  receipt. The simulator therefore does not claim the paper's distributed
  privacy or independent-verification guarantees.
