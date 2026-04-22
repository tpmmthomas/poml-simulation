# PoML-Sim

Standalone Python simulation of the Proof-of-ML-Inference (PoML) blockchain consensus protocol described in the companion paper.

Multiple miner processes compete to produce blocks by running ML inference (tiny U-Net) and generating ZK proofs via EZKL. Miners pull queries from a shared mempool, derive per-query noise via a VRF, run inference, generate SNARK proofs, deterministically encrypt outputs, evaluate a hash-based lottery over the ciphertexts, and broadcast winning blocks over a simulated P2P network.

## Prerequisites

- Python 3.10+
- [EZKL](https://github.com/zkonduit/ezkl) Python bindings
- PyTorch, ONNX

## Setup

```bash
# Install the package and dev dependencies
pip install -e ".[dev]"
```

## Model & EZKL Setup (one-time)

Before running the simulation, you must export the model and generate all EZKL proving artifacts. This only needs to be done once (or repeated if you change the model):

```bash
bash scripts/setup_model.sh
```

This runs two steps:
1. **Export** the Tiny U-Net to ONNX (`model/network.onnx`)
2. **EZKL setup**: generates circuit settings, compiles the circuit, fetches the SRS, and generates proving/verification keys

After setup, `model/` will contain:
- `network.onnx` — ONNX model
- `network.ezkl` — compiled EZKL circuit
- `settings.json` — circuit settings
- `pk.key` / `vk.key` — proving and verification keys
- `kzg.srs` — structured reference string

## Running the Simulation

```bash
python scripts/run.py                # uses config.yaml
python scripts/run.py my_config.yaml # custom config
```

The simulation will:
1. Spawn N miner processes
2. Generate queries at the configured rate and push them to a shared mempool
3. Each miner fetches queries, runs inference + ZKP via EZKL, and evaluates the lottery
4. When a miner wins the lottery, their block is validated and broadcast to all others
5. Console logs show real-time progress; `metrics.json` is written at the end

## Configuration

Edit `config.yaml` to tune simulation parameters:

```yaml
# Paths
model_path: "model/network.onnx"       # Path to the ONNX model
ezkl_artifacts_dir: "model/"            # Directory with EZKL artifacts

# Simulation scale
num_miners: 4                           # Number of miner worker processes
num_queries: 20                         # Total inference queries to generate
query_rate: 2.0                         # Queries injected per second

# Blockchain parameters
difficulty: "0x00ffff..."               # Lottery threshold (hex, 256-bit)
                                        #   Higher = easier (more blocks)
                                        #   Lower = harder (fewer blocks)
block_reward: 50                        # Coins awarded per block to the miner
network_latency_ms: 100                 # Simulated P2P propagation delay (ms)
max_queries_per_block: 10               # Max queries a miner processes before
                                        #   giving up on the current block

# Model
input_shape: [1, 2, 8, 8]              # Model input shape [B, C, H, W]

# Reproducibility
seed: 42                                # RNG seed for query generation

# VRF noise schedule length (T in the paper). Only U_i[0] is fed to the
# single-pass U-Net; all T VRF outputs are still proven and validated.
diffusion_steps: 1
```

### Key parameters to tune

| Parameter | Effect |
|-----------|--------|
| `difficulty` | Controls how likely a proof chain wins the lottery. Raise the hex value (more leading `ff`s) to produce blocks faster; lower it to make mining harder. |
| `num_miners` | More miners = more parallel proving = faster block production, but also more resource usage. |
| `num_queries` | Total workload. The simulation ends when all queries have been processed (or timeout). |
| `query_rate` | How fast queries arrive. If set higher than proving throughput, the mempool backlog grows. |
| `max_queries_per_block` | Caps how many proofs a miner attempts per block. Higher values give more lottery attempts per block. |
| `network_latency_ms` | Simulates propagation delay. Higher values increase the chance of stale/orphan blocks. |

## Tests

```bash
pytest                          # run all tests
pytest tests/test_crypto.py     # crypto utilities only
pytest tests/test_blockchain.py # blockchain state only
pytest tests/test_zkp.py        # ZKP integration (requires EZKL artifacts)
pytest tests/test_e2e.py        # full e2e simulation (requires EZKL artifacts)
```

Unit tests for `crypto` and `blockchain` run without EZKL artifacts. The `zkp` and `e2e` tests are automatically skipped if the artifacts haven't been generated.

## Differences from the Paper

This simulation targets the revised PoML protocol (VRF-derived per-step noise, ciphertext-based lottery, deterministic PKE, no meta-proof). The following simplifications remain:

| Paper | Simulation | Rationale |
|-------|-----------|-----------|
| **Model M\_θ**: full DDPM with T denoising steps (reverse recurrence `x_{t-1} = f_θ(x_t, c, t) + σ_t z_t`) | **Tiny U-Net, single pass**. `diffusion_steps` (T) in the config controls the **VRF schedule length**; all T outputs are produced and validated, but only U\_i[0] is fed into the circuit as the noise channel. | A full DDPM loop inside ezkl is infeasible. The VRF side of the protocol is still exercised end-to-end for any T ≥ 1. |
| **Unbiasable VRF** (e.g. RFC 9381 ECVRF) | **Ed25519 sign-then-hash**: `y = SHA256(Ed25519.sign(sk, x))`, `π = sig`; verify the signature and check the hash. | Ed25519's RFC 8032 signatures are deterministic per message, giving uniqueness + pseudorandomness under ROM without an extra dependency. Not formally unbiasable. |
| **Deterministic PKE** `Enc(pk_u, y ‖ bind ‖ taskID)` with correctness + one-wayness | **Textbook RSA-2048** via `pow(m, e, n)` against `cryptography`'s RSA keys, with 245-byte chunking and a 4-byte length prefix. | Deterministic by construction, one-way under the RSA assumption; **not** IND-CPA. Not for production. |
| **L\_PoML ZKP** proving (a) inference correctness, (b) query commitment opening, (c) model commitment opening, and (d) `ct_i = Enc(pk_u, y ‖ bind ‖ taskID)` | **EZKL proof of inference + Poseidon input/output commitments**. Hashed visibility commits to the tensor fed into the circuit, but the circuit does not check that the noise channel came from `U_i[0]` nor that the encryption was applied honestly. | Validator-side VRF verification closes (a)'s "noise came from the VRF" question at the *statement* level; it does not enforce that the circuit actually consumed the VRF noise. Full circuit-level coverage would require RSA mod-exp inside the SNARK, which is out of scope. |
| **Per-step VRF transcript** `Π^VRF_i = ((z_{i,t}, π_{i,t}))_{t=1..T}` included in the block, verified separately from the ZKP | **Implemented** — produced by the miner per query, stored in `InferenceResult.vrf_transcript`, and verified by `blockchain.validate_block` against `BlockHeader.miner_vrf_vk`. | Matches the paper's revised block structure `B = (s, x, X, Y, Π, Π^VRF)`. |
| **Ciphertext-based lottery** `H_i = H(G(s,x), (ct_1, ..., ct_i)) < D` | **Implemented** — `crypto.evaluate_lottery(fingerprint, ciphertexts_so_far, D)`. The validator recomputes the hash and rejects if it disagrees with `BlockHeader.lottery_hash`. | Replaces the previous proof-based lottery. |
| **Ciphertext-based chain binding** `bind_1 = G(s,x)`, `bind_i = H(ct_{i-1})` for i ≥ 2 | **Implemented** — set by the miner before encryption and checked by the validator. | Matches the revised §4.4. |
| **Miner VRF key distribution** with an on-chain commitment `c_vk ← Commit(vk)` | **Self-declared** in `BlockHeader.miner_vrf_vk`. No on-chain registry. | Pragmatic at sim scale; a real deployment would want a one-time commitment analogous to the paper's miner-secret-key commitment. |
| **Meta-proof** π\_i^(2) for L\_det | **Removed**. | No longer required in the revised protocol; the VRF transcript plays the analogous role. |
| **Digital signatures** (Ed25519 or similar) | **HMAC-SHA256** stub for identity signatures. | Kept from the prior simulation; identity keys are independent of VRF keys. |
| **Commitment scheme** (Setup\_com, Commit, Open) for query inputs and model weights | **SHA-256 hash** as a binding commitment. | Simple and binding; not hiding. |
| **Distributed network** with real P2P gossip | **Simulated** via `multiprocessing.Queue` + `threading.Timer` delays. | All processes run locally. |

### What IS faithfully implemented

- **Block production loop** (Algorithm 1): retrieve queries → derive seed r\_i → evaluate VRF to produce U\_i and Π^VRF\_i → run inference → deterministically encrypt output → evaluate ciphertext lottery `H(G(s,x), (ct_1, ..., ct_i)) < D`.
- **Seed derivation**: r\_i = H(G(s,x) ‖ c\_{c,i} ‖ taskID\_i ‖ pk\_m ‖ i).
- **Block fingerprint**: G(s,x) = SHA-256(prev\_hash ‖ hash(txns)).
- **Separate miner keys**: each miner holds an identity keypair (pk\_m, sk\_m) *and* an independent Ed25519 VRF keypair (vk^VRF\_m, sk^VRF\_m).
- **Per-query VRF noise schedule**: `(z_{i,t}, π_{i,t}) ← VRF.Eval(sk^VRF, r_i ‖ t)` for t = 1..T, with all transcript entries verified by validators.
- **Deterministic ciphertext**: same (pk\_u, y, bind, taskID) always yields the same ct\_i.
- **Ciphertext-based chain binding and lottery**.
- **Block validity** (Definition 4.6, revised conditions 1–5): prev-hash and height, non-empty tuples, VRF verification, chain binding, and optional ZKP verification.
- **Longest-chain rule** consensus.
- **Proof-of-inference**: real EZKL SNARK proofs for the tiny U-Net.
- **Account-based coin system** with block rewards and query fees.
- **Concurrent mining** with multiple processes competing for blocks.

## Project Structure

```
poml-sim/
├── config.yaml                 # Simulation configuration
├── model/
│   ├── tiny_unet.py            # Tiny U-Net definition + ONNX export
│   └── setup_ezkl.py           # One-time EZKL artifact generation
├── src/poml_sim/
│   ├── config.py               # Pydantic config loader
│   ├── types.py                # Block, Query, Transaction, etc.
│   ├── crypto.py               # Hashing, seed derivation, ciphertext lottery, signing
│   ├── vrf.py                  # Ed25519 sign-then-hash VRF + noise schedule
│   ├── encryption.py           # Deterministic textbook RSA-2048
│   ├── blockchain.py           # Chain state + block validation (VRF + ciphertext checks)
│   ├── accounts.py             # Account balances + transfers
│   ├── mempool.py              # Process-safe query mempool
│   ├── network.py              # Simulated P2P with latency
│   ├── miner.py                # Miner worker process (Algorithm 1)
│   ├── query_generator.py      # Generates queries at configured rate
│   ├── zkp.py                  # EZKL prove/verify wrapper
│   ├── coordinator.py          # Main orchestrator (generates identity + VRF keys)
│   └── metrics.py              # JSON metrics output
├── scripts/
│   ├── setup_model.sh          # Model export + EZKL setup
│   └── run.py                  # Entry point (logs to logs/run_<ts>.log)
└── tests/
    ├── test_crypto.py
    ├── test_vrf.py
    ├── test_encryption.py
    ├── test_deterministic_pke.py
    ├── test_blockchain.py
    ├── test_zkp.py
    └── test_e2e.py
```
