# PoML-Sim

Standalone Python implementation of the **Proof-of-ML-Inference (PoML)**
consensus protocol described in the companion paper. The repository contains a
protocol simulator, verified model-proof backends, and the experiments used to
study liveness, wasted work, DDPM compatibility, LLM compatibility, and the
GPT-2/DeepProve complexity function.

## Prerequisites

- Python 3.11 or newer on Linux.
- CUDA for the GPT-2/DeepProve backend and the larger model experiments.
- Extra model and prover dependencies only for the backend or experiment being
  run.

## Install and run a first check

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,experiments]'
poml-sim --backend smoke --blocks 3 --output experiments/results/demo
pytest -q
```

`smoke` is an explicit deterministic test double. It exercises the protocol
without running ML inference or a zero-knowledge proof. Use `gpt2` or
`diffusion` for a real inference–proof pair.

## Repository map

| Directory | Purpose / entry point |
| --- | --- |
| `src/poml_sim/` | Protocol, model backends, accounting, and analysis; start with `system.py` |
| `tests/` | Protocol, model, fitting, and command tests |
| `experiments/` | The paper's experiments; start with `README.md` |
| `scripts/` | Prover preparation, model downloads, and EZKL setup |
| `model/` | Tiny U-Net definition and EZKL export code |
| `docs/` | Experiment methods and verification records |

Downloaded models, prover checkouts, proofs, measurements, and figures belong
in the ignored `models/`, `.scratch/`, and `experiments/results/` directories.

## Run the simulator

### Tiny U-Net with EZKL

This backend uses a fixed-shape 8×8 denoising network and a genuine EZKL proof
and verification. It is the small diffusion instantiation used for protocol
checks; it is not Stable Diffusion.

```bash
pip install -e '.[ezkl]'
bash scripts/setup_model.sh experiments/results/ezkl-setup
poml-sim --backend diffusion --artifacts experiments/results/ezkl-setup \
  --miners 1 --queries 2 --blocks 1 \
  --difficulty 0x10000000000000000000000000000000000000000000000000000000000000000 \
  --output experiments/results/diffusion-demo
```

The difficulty is `2^256`, so every completed pair wins. Setup downloads an
SRS and creates about 9 GiB of proving-key data; setup and proving take minutes
on the tested hardware.

### GPT-2 with DeepProve

This backend uses GPT-2 small and the pinned DeepProve worker. It requires CUDA,
Rust, and the pinned nightly toolchain.

```bash
pip install -e '.[deepprove,experiments]'
rustup toolchain install nightly-2026-01-27
python scripts/prepare_deepprove_protocol.py
poml-sim --backend gpt2 --device cuda:0 --miners 1 --queries 2 --blocks 1 \
  --max-output 1 \
  --difficulty 0x10000000000000000000000000000000000000000000000000000000000000000 \
  --output experiments/results/gpt2-demo
```

The preparer builds the pinned worker in `.scratch/`. The first run downloads
GPT-2 and creates its setup. Prompts must tokenize to 2–63 tokens and the
prompt plus output must fit the 64-token setup. The default embedding-noise
scale is `alpha = 0.05`; decoding uses temperature 1, top-k 50, and top-p 0.95.

## Experiments

Read [the experiment guide](docs/experiments.md) for the paper's order,
commands, defaults, and metric definitions. The guide begins with fresh
inference–proof measurements and then introduces the following experiments:

- **PoML Liveness and Block Generation Stability**
- **Wasted Work Analysis**
- **DDPM Compatibility: Formal Statements and Experiments**
- **LLM Compatibility: Formal Statements and Experiments**
- **A Reference Complexity Function for GPT-2 and DeepProve**
- **Additional Wasted-Work Results**, produced as part of Wasted Work Analysis

## Tests

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=src .venv/bin/python -m pytest -q
uvx --from ruff==0.15.6 ruff check src tests experiments scripts model
uvx --from ruff==0.15.6 ruff format --check src tests experiments scripts model
```

## What is implemented, and what is abstracted?

| Paper claims | This Implementation |
| --- | --- |
| **Query submission.** A signed query contains `qid`, `taskID`, an input commitment, an expiry height, and a maximum fee. | `PoMLSystem.submit_query` creates and validates the same fields; raw inputs remain off-chain and are checked against the commitment. |
| **Miner registration.** Each miner registers an identity key, an inference VRF key, and an encryption VRF key. | `MinerKeys` holds three domain-separated Ed25519 keys and `Registration` activates their signed public-key triple in the ledger. |
| **Inference randomness.** An unbiasable VRF derives the ordered randomness collection `R` for each query. | A deterministic sign-then-hash Ed25519 VRF substitute derives and verifies each indexed value; it is not the paper's formally unbiasable VRF. |
| **PoML proof relation.** A NIZK proves model inference, the input commitment, complexity, encryption randomness, and ciphertext construction. | The model backend produces a genuine EZKL or DeepProve model proof. Verification of the proof however is not simulated. |
| **Block production.** Transactions are frozen first; the first seed uses `G(s, tx)`, later seeds use `H(π)`; a miner appends inference–proof pairs until it wins. | The simulator follows this order, checks the three key roles, verifies both VRF transcripts, and binds every pair to the preceding proof. |
| **Lottery.** A complexity-`C` pair wins with `tau_D(C) = 1 - (1 - D/2^256)^C`. | One SHA-256 hash of `G(s, tx)` and the complete ciphertext prefix is compared with the exact integer threshold `D_C`. |
| **Ledger and fees.** Valid blocks settle query fees as burn, solve, and include components; completed losing responses may be submitted later. | Account balances, expiries, duplicate-settlement checks, response transactions, longest-chain adoption, and first-received ties are implemented. |
| **Model instantiations.** The paper describes stochastic denoising models and perturbed autoregressive language models. | `diffusion` proves one tiny fixed-shape U-Net pass with `C=1`; `gpt2` runs fresh GPT-2 small inference and DeepProve proofs under the 64-token setup. |
| **Output privacy.** NIZK zero knowledge and randomized public-key encryption hide the model output from other parties. | X25519/HKDF/AES-GCM is implemented for output encryption |
| **Distributed execution.** The security argument assumes a synchronous network and independent miners. | Virtual miners are independent, but physical prover calls are serialized on one host with zero network delay. |

