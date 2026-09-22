# PoML-Sim

An independent, paper-aligned teaching implementation of **Proof of ML Inference
(PoML)**. It includes a runnable protocol simulator, the liveness and wasted-work
experiments, and the appendix compatibility and complexity tools. It does not
reproduce the paper's archived numerical results by construction.

Use Python 3.11+ on Linux. Start with the lightweight protocol demonstration:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,experiments]'
poml-sim --backend smoke --blocks 3 --output experiments/results/demo
pytest -q
```

`smoke` is an explicit deterministic test double: it runs neither ML nor ZK.
For real inference and verified model proofs, choose `gpt2` or `diffusion` below.
Start reading [the protocol implementation](src/poml_sim/system.py), then
[the paper mapping and abstraction audit](docs/features/publication.md).

| Directory | Purpose / entry point |
| --- | --- |
| `src/poml_sim/` | Protocol, backends, accounting and analysis; `system.py` |
| `tests/` | Protocol rejection tests, model math, fitting and command checks; `test_system.py` |
| `experiments/` | The two main experiments and appendix tools; [run guide](docs/experiments.md) |
| `scripts/` | Pinned prover preparation and model downloads; `prepare_deepprove_protocol.py` |
| `model/` | Tiny U-Net definition and EZKL setup; `tiny_unet.py` |
| `docs/` | Publication scope, experiment methods and verification; `README.md` |

Add domain code under `src/poml_sim/`, matching tests under `tests/`, and runnable
experiment commands under `experiments/`. Downloaded models, prover checkouts,
proofs and measurements belong in ignored `models/`, `.scratch/` and
`experiments/results/`; they are not distributed with this repository.

## Choose a real simulator backend

**Tiny U-Net / EZKL (CPU).** This is the agreed diffusion teaching model:
one 8×8 denoising-network pass, with a genuine EZKL proof and verification.
Its 11,401 weights are deterministically initialized, not trained; it does not
generate useful images or run Stable Diffusion.

```bash
pip install -e '.[ezkl]'
bash scripts/setup_model.sh experiments/results/ezkl-setup
poml-sim --backend diffusion --artifacts experiments/results/ezkl-setup \
  --miners 1 --queries 2 --blocks 1 \
  --difficulty 0x10000000000000000000000000000000000000000000000000000000000000000 \
  --output experiments/results/diffusion-demo
```

That difficulty is `2^256`: every completed pair wins, making this a bounded
integration check. Setup downloads an SRS and produces about 9 GiB of proving
key data with the tested EZKL version. Setup and proofs take minutes, depending
on hardware. Supply a fresh setup directory; the script refuses to replace keys.

**GPT-2 / DeepProve (CUDA).** Requires Rust/Cargo, the pinned
`nightly-2026-01-27` toolchain and CUDA development tools at `/usr/local/cuda`.
Tested on RTX A6000 GPUs; context-64 proving can require most of a 48 GiB GPU.
Check the upstream terms linked below before using DeepProve.

```bash
pip install -e '.[deepprove,experiments]'
rustup toolchain install nightly-2026-01-27
python scripts/prepare_deepprove_protocol.py
poml-sim --backend gpt2 --device cuda:0 --miners 1 --queries 2 --blocks 1 \
  --max-output 1 \
  --difficulty 0x10000000000000000000000000000000000000000000000000000000000000000 \
  --output experiments/results/gpt2-demo
```

The preparer clones pinned sources into `.scratch/`, applies the supplied work
counter and protocol patches, skips unrelated upstream demo LFS assets, then
builds `poml-prover`. The first run downloads
GPT-2 and builds its setup. Reuse that setup with
`--setup-directory experiments/results/gpt2-demo/proofs/setup`; every inference
and proof is still generated afresh. Prompts must tokenize to 2–63 tokens; the
prompt plus generated output must fit the 64-token setup. Use `--prompts` for a
newline-separated prompt file. The default α is 0.05, temperature 1, top-k 50,
top-p 0.95, with EOS or affordable output-cap termination.

Both real modes save `run.json`, including backend identity, per-execution
measurements, block accounting and cancellation diagnostics. DeepProve also
retains requests, model proofs and public logits. EZKL proofs are verified in
memory; their hashes and circuit identity are recorded. Runs fail rather than
substituting fake proofs. Choose a fresh output directory for every run.

## What is implemented, and what is abstracted?

| Paper mechanism | Implementation and limit |
| --- | --- |
| Query admission | Signed `(user, taskID)` query IDs, input commitments, expiry, maximum fee, parent-state balances and off-chain inputs |
| Mining | Frozen transaction lists, parent/transaction binding, previous-proof chaining, indexed inference randomness and separate encryption randomness |
| Lottery | **One SHA-256 hash** of the frozen binding and complete ciphertext prefix, tested against `floor(2^256 × [1 − (1 − D/2^256)^C])`; no enumeration of C hashes |
| Ledger | Registrations activate in subsequent blocks; duplicate settlement is rejected; burn/solve/include fees; no block reward; longest valid chain with first-received ties |
| Losing work | Completed losing responses can be settled in a later block's frozen transaction list |
| Full PoML NP relation | **Abstracted by trusted-host receipts.** Real model proofs do not jointly prove query commitment, VRFs, complexity, encryption and all PoML conditions inside one private circuit |
| Cryptography | Real Ed25519 signatures and X25519/HKDF/AES-GCM encryption; experimental sign-then-hash VRF and SHA-256 commitment stand-ins. Reproducible experiment keys, host-visible witnesses and DeepProve public logits mean **no protocol privacy/security claim** |
| GPT-2 | Real quantized DeepProve inference/proofs; host/composite verification of sampling. Float32 appendix evaluation is a separate experiment. Published reference operation counts omit added noise/sampling graph work |
| Diffusion | Real EZKL proof of one tiny random-weight U-Net pass, synthetic conditioning and `C=1`. Full SD v1-4 with 50 DDPM steps appears **only in the appendix compatibility experiment** |
| Distribution and timing | Independent virtual miners, zero network delay, serialized physical prover calls; measured inference/proof service time excludes setup and verification. Physically computed canceled jobs are reported separately. No peer network or Byzantine attack simulation |
| Large experiments | Measured `(time, C)` replay is explicit and does not create new proofs. Liveness also offers fresh proofs and an actual double-SHA-256 baseline; a Poisson baseline is optional |

Genesis funds demo users; difficulty and fee rates are fixed for a run. The
full validator supports forks, but the default honest zero-delay driver does
not produce them. See [the audit](docs/features/publication.md) for exact scope.

## Paper experiments and appendix

[The experiment guide](docs/experiments.md) gives full commands and reduced runs:

- Fresh timing bank → liveness (four miners, 256 queries, 50 blocks, 300 s target).
- Wasted completed work: first completion per query is useful, including work
  from losing miners; `M=10…10,000`, `Q=20…100,000`, 100 repeats at 300/600/900 s.
- Stable Diffusion v1-4 compatibility: 1,000 inputs, 50 **DDPM** steps, shared
  reverse-step noise for perturbations and an independent-noise baseline.
- GPT-2 compatibility: four utility tasks and six independent-decoding
  activation tasks, relative Gaussian embedding noise, saved token manifests.
- Instrumented operation counts and nonnegative runtime fitting with nested
  prompt-grouped validation. **Uniform operation weights are the default.**

The full campaigns are intentionally expensive. Smoke results and the
[verification record](docs/verification.md) do not establish the paper's reported
means, standard deviations or numerical tables.

## License and upstream components

Original PoML-Sim code is [MIT licensed](LICENSE). Third-party code, model weights,
datasets and upstream code represented in patch context retain their own terms;
MIT does not relicense them. In particular, use the license files at the pinned
[DeepProve revision](https://github.com/Lagrange-Labs/deep-prove/blob/9d1a53e2ef49ffa2c902b8689cd3c58057a4e662/LICENSE)
and [dp-crypto revision](https://github.com/Lagrange-Labs/dp-crypto/blob/22e8e93cd0f94a7638616a1ae22190feb0a6b275/LICENSE),
not just their Cargo metadata. Also see [EZKL](https://github.com/zkonduit/ezkl),
[GPT-2](https://huggingface.co/openai-community/gpt2) and
[Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4).
