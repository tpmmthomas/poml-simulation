# PoML-Sim

Standalone Python simulation of the Proof-of-ML-Inference (PoML) blockchain consensus protocol described in the companion paper.

Multiple miner processes compete to produce blocks by running ML inference (tiny U-Net) and generating ZK proofs via EZKL. Miners pull queries from a shared mempool, run inference, generate SNARK proofs, evaluate a hash-based lottery, and broadcast winning blocks over a simulated P2P network.

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

This simulation is a proof-of-concept and makes several simplifications compared to the full PoML protocol described in the paper (Section 4):

| Paper | Simulation | Rationale |
|-------|-----------|-----------|
| **Model M\_θ**: diffusion model (e.g. Stable Diffusion) with a full-scale U-Net | **Tiny U-Net** (~2-5K params, 8×8 spatial, channels 2→8→16→8→1) | EZKL circuit compilation and proving is infeasible for large models. The tiny U-Net preserves the architectural style (down-blocks, skip connections, up-blocks) while keeping proofs tractable. |
| **Meta-proof π\_i^(2)**: ZKP for L\_det proving that deterministic randomness ρ\_i = F(sk\_m, r\_i) was used | **Dummy string** (random 64-char hex); verification always returns `True` | Meta-proofs are the most expensive component—they prove a statement *about* the SNARK prover circuit itself. Out of scope for this PoC. |
| **Deterministic prover randomness** ρ\_i = F(sk\_m, r\_i) replacing the prover's random tape | **EZKL default randomness** | EZKL doesn't expose an API to inject custom prover randomness. The seed derivation r\_i is still computed per the paper for the noise input, but the prover's internal tape is not controlled. |
| **Proof chain binding** bind\_i = H(π\_{i-1}^(1)) for i ≥ 2 linking proofs sequentially | **Implemented** — bind\_1 = G(s,x) (fingerprint), bind\_i = SHA-256(π\_{i-1}) for i ≥ 2. Validated in `blockchain.py`. | Faithfully follows the paper's proof chain binding scheme. |
| **Output encryption** ct\_i = Enc(pk\_u, y\_i ‖ bind\_i ‖ taskID) | **Implemented** — X25519 key agreement + ChaCha20-Poly1305 AEAD. Each query carries the user's encryption public key; miners encrypt outputs before inclusion. | Uses modern ECIES-style encryption. The paper leaves the encryption scheme generic; this is a concrete instantiation. |
| **ZKP statement** L\_PoML proving inference correctness, commitment opening, model commitment opening, and encryption correctness | **EZKL proof of inference + Poseidon input/output commitments** | EZKL's `hashed/public` visibility mode wraps inputs and outputs in Poseidon hash commitments inside the circuit, approximating the paper's commitment openings. Model commitment and encryption correctness remain outside the circuit. |
| **Commitment scheme** (Setup\_com, Commit, Open) for query inputs and model weights | **SHA-256 hash** as a binding commitment | A simple hash commitment is sufficient for simulation purposes. Not hiding (inputs are public anyway in this PoC). |
| **Digital signatures** (Ed25519 or similar) | **HMAC-SHA256** stub | Simplified to avoid external key management. Signatures are checked optimistically. |
| **Optimistic variant** (Section 5b): VRF-based randomness binding + challenge-response protocol | **Not implemented** | Excluded from scope per user request. |
| **Miner secret-key commitment** c\_sk ← Commit(pp, sk\_m) registered on-chain | **Not implemented** | Only relevant for meta-proof verification, which is stubbed. |
| **Distributed network** with real P2P gossip | **Simulated network** via `multiprocessing.Queue` + `threading.Timer` delays | All processes run locally on one machine; latency is simulated. |

### What IS faithfully implemented

- **Block production loop** (Algorithm 1): miners retrieve queries → derive seed r\_i → run inference → generate ZKP → evaluate lottery H(G(s,x), Π\_i) < D
- **Seed derivation**: r\_i = H(G(s,x) ‖ c\_{c,i} ‖ taskID\_i ‖ pk\_m ‖ i) exactly per the paper
- **Block fingerprint**: G(s,x) = SHA-256(prev\_hash ‖ hash(txns))
- **Lottery mechanism**: H\_i = H(G(s,x), Π\_i) < D with configurable difficulty
- **Block validity** (Definition 4.6, conditions 1-2 and 5; condition 3 optionally; condition 4 stubbed)
- **Longest-chain rule** consensus
- **Proof-of-inference**: real EZKL SNARK proofs for the tiny U-Net
- **Account-based coin system** with block rewards and query fees
- **Concurrent mining** with multiple processes competing for blocks

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
│   ├── crypto.py               # Hashing, seed derivation, lottery, signing
│   ├── blockchain.py           # Chain state + block validation
│   ├── accounts.py             # Account balances + transfers
│   ├── mempool.py              # Process-safe query mempool
│   ├── network.py              # Simulated P2P with latency
│   ├── miner.py                # Miner worker process (Algorithm 1)
│   ├── query_generator.py      # Generates queries at configured rate
│   ├── zkp.py                  # EZKL prove/verify wrapper
│   ├── coordinator.py          # Main orchestrator
│   └── metrics.py              # JSON metrics output
├── scripts/
│   ├── setup_model.sh          # Model export + EZKL setup
│   └── run.py                  # Entry point
└── tests/
    ├── test_crypto.py
    ├── test_blockchain.py
    ├── test_zkp.py
    └── test_e2e.py
```
