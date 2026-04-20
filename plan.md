# Plan: PoML Protocol PoC Simulation

## TL;DR
Build a standalone Python simulation of the PoML blockchain consensus protocol where multiple miner processes compete to produce blocks by running ML inference (tiny U-Net) and generating ZK proofs via EZKL. Miners pull queries from a shared mempool, run inference, generate SNARK proofs, evaluate a hash-based lottery, and broadcast winning blocks over a simulated P2P network. Meta-proofs are dummy strings; verification of them always passes. Includes basic coin/transfer transactions and outputs console logs + JSON metrics.

---

## Language & Libraries

- **Python 3.10+** (matches existing project, EZKL has Python bindings, multiprocessing stdlib)
- **Core**: `ezkl` (ZKP), `torch` (model), `onnx` (export), `multiprocessing` (workers)
- **Support**: `pydantic` (config/data validation), `pyyaml` (config file), `hashlib` (SHA-256 hashing), `logging`, `json`

---

## Project Structure

```
poml-sim/
├── pyproject.toml
├── README.md
├── config.yaml                    # Central simulation config
├── model/
│   ├── tiny_unet.py               # Define & export tiny U-Net to ONNX
│   ├── setup_ezkl.py              # One-time EZKL setup (gen-settings → compile → setup)
│   ├── network.onnx               # (generated) ONNX model
│   ├── network.ezkl               # (generated) compiled circuit
│   ├── settings.json              # (generated) EZKL settings
│   ├── pk.key                     # (generated) proving key
│   ├── vk.key                     # (generated) verification key
│   └── kzg.srs                    # (generated) SRS
├── src/
│   └── poml_sim/
│       ├── __init__.py
│       ├── config.py              # Pydantic config loader
│       ├── types.py               # Block, Query, Transaction, Proof dataclasses
│       ├── crypto.py              # SHA-256 hashing, seed derivation, lottery eval
│       ├── mempool.py             # Process-safe mempool (multiprocessing.Queue)
│       ├── blockchain.py          # Chain state, validation, longest-chain rule
│       ├── network.py             # Simulated P2P message diffusion with latency
│       ├── miner.py               # Miner worker process
│       ├── query_generator.py     # Spawns queries at configured rate
│       ├── zkp.py                 # EZKL wrapper: prove() and verify()
│       ├── accounts.py            # Simple account-based balances + transfers
│       ├── coordinator.py         # Orchestrator: starts processes, relays messages
│       └── metrics.py             # Collect & dump metrics to JSON
├── scripts/
│   ├── setup_model.sh             # Runs model export + EZKL setup
│   └── run.py                     # Main entry point
└── tests/
    ├── test_crypto.py
    ├── test_blockchain.py
    ├── test_zkp.py
    └── test_e2e.py
```

---

## Steps

### Phase 1: Foundation (steps 1–3)

1. **Scaffold project** — Create `poml-sim/` with `pyproject.toml` (deps: ezkl, torch, onnx, pydantic, pyyaml), directory structure, and README.

2. **Config system** (`config.yaml` + `config.py`) — Pydantic model loading a YAML file. Parameters:
   - `model_path`: path to ONNX model
   - `ezkl_artifacts_dir`: path to compiled circuit, keys, SRS
   - `num_miners`: number of miner processes (e.g. 3–8)
   - `num_queries`: total queries to generate
   - `query_rate`: queries per second arrival rate
   - `difficulty`: lottery difficulty target D (uint256, controls expected block time)
   - `block_reward`: coins awarded to winning miner
   - `network_latency_ms`: simulated propagation delay
   - `max_queries_per_block`: cap on queries a miner processes before lottery check
   - `input_shape`: model input dimensions (e.g. [1, 2, 8, 8])
   - `seed`: simulation-wide random seed for reproducibility

3. **Data types** (`types.py`) — Pydantic/dataclass models:
   - `Query`: `(query_id, user_pk, task_id, conditioning_input, commitment, fee, signature)`
   - `Transaction`: `(sender, receiver, amount, signature)` — simple account transfer
   - `InferenceResult`: `(query_id, output, proof_bytes, meta_proof_dummy, seed_used)`
   - `BlockHeader`: `(block_height, prev_hash, miner_pk, timestamp, difficulty, nonce_proof_hash)`
   - `Block`: `(header, queries: list[Query], results: list[InferenceResult], transactions: list[Transaction], lottery_hash)`

### Phase 2: Model & ZKP (steps 4–6)

4. **Tiny U-Net model** (`model/tiny_unet.py`) — PyTorch U-Net:
   - Architecture: 2 down-blocks, bottleneck, 2 up-blocks with skip connections
   - Channels: 2 (input: noise + conditioning) → 8 → 16 → 8 → 1 (output)
   - Spatial: 8×8 (extremely small to keep EZKL circuit feasible)
   - Total: ~2–5K parameters
   - Export: `torch.onnx.export()` to `model/network.onnx`
   - The "conditioning" channel simulates the text-encoding input; the "noise" channel is seed-derived

5. **EZKL setup script** (`model/setup_ezkl.py`) — One-time setup, runs sequentially:
   1. `ezkl.gen_settings(model_path, settings_path)`
   2. `ezkl.calibrate_settings(settings_path, model_path, ...)` (optimize for resources)
   3. `ezkl.compile_circuit(model_path, compiled_path, settings_path)`
   4. `ezkl.get_srs(settings_path, srs_path)`
   5. `ezkl.setup(compiled_path, vk_path, pk_path, srs_path)`
   - All artifacts saved to `model/`

6. **ZKP wrapper** (`src/poml_sim/zkp.py`) — Two functions:
   - `generate_proof(input_data, model_artifacts_dir) → (proof_bytes, witness)`:
     1. Write input to temp `input.json`
     2. `ezkl.gen_witness(input_json, compiled_model, witness_path)`
     3. `ezkl.prove(witness_path, compiled_model, pk_path, proof_path, srs_path)`
     4. Read and return proof bytes
   - `verify_proof(proof_bytes, model_artifacts_dir) → bool`:
     1. Write proof to temp file
     2. `ezkl.verify(proof_path, vk_path, settings_path, srs_path)`
     3. Return boolean result

### Phase 3: Crypto & Blockchain (steps 7–9)

7. **Crypto utilities** (`crypto.py`):
   - `block_fingerprint(prev_hash, txns) → bytes`: G(s,x) = SHA-256(prev_hash || hash(txns))
   - `derive_seed(fingerprint, commitment, task_id, miner_pk, position) → bytes`: r_i per paper
   - `evaluate_lottery(fingerprint, proofs_so_far) → (hash_value, bool)`: H(G(s,x), Π) < D
   - `hash_block(block) → bytes`: block hash for chaining
   - `sign(sk, data) → sig` / `verify_sig(pk, data, sig) → bool`: Ed25519 or simple HMAC stub
   - `generate_dummy_meta_proof() → str`: random 64-char hex string

8. **Blockchain state** (`blockchain.py`):
   - `Blockchain` class:
     - `chain: list[Block]` (longest chain)
     - `add_block(block) → bool`: validate then append
     - `validate_block(block) → bool`:
       1. Check `prev_hash` matches tip
       2. Lottery hash < difficulty
       3. Non-empty queries + results
       4. All inference proofs verify via `zkp.verify_proof()`
       5. Meta-proof verification: always True (dummy)
       6. All transactions valid (balances, signatures)
     - `get_tip() → Block`
     - `get_height() → int`
     - Handle simple fork resolution: longest chain wins

9. **Account system** (`accounts.py`):
   - `AccountState`: dict mapping `pk → balance`
   - `apply_block_reward(miner_pk, reward)`
   - `apply_transaction(tx) → bool`: check balance, deduct sender, credit receiver
   - `validate_transaction(tx, state) → bool`

### Phase 4: Network & Workers (steps 10–13)

10. **Mempool** (`mempool.py`):
    - Wraps `multiprocessing.Queue`
    - `add_query(query)`: push query
    - `get_queries(max_n) → list[Query]`: pop up to N queries (non-blocking)
    - Queries not included in a valid block get re-added (coordinator handles this)

11. **Network simulator** (`network.py`):
    - `NetworkBus` class:
      - Holds per-miner `multiprocessing.Queue` for incoming messages
      - `broadcast(message, sender_id)`: push message to all other miners' queues after `network_latency_ms` delay (use `threading.Timer` in coordinator)
      - Message types: `NewBlock`, `NewQuery`
    - Simulates propagation delay per the config

12. **Miner process** (`miner.py`):
    - `MinerProcess(mp.Process)` — each miner runs this loop:
      1. Check inbox for new blocks → if received, validate & update local chain copy + discard in-progress work if block height advanced
      2. Fetch queries from mempool (up to `max_queries_per_block`)
      3. For each query i:
         a. Derive seed r_i from `crypto.derive_seed()`
         b. Prepare model input: noise from seed + conditioning from query
         c. Run PyTorch inference → output y_i
         d. Call `zkp.generate_proof(input_data)` → π_i
         e. Generate dummy meta-proof → π²_i
         f. Evaluate lottery: `crypto.evaluate_lottery(fingerprint, all_proofs_so_far)`
         g. If lottery won → assemble Block, send to coordinator via output queue, break
         h. If not won → continue to next query
      4. Between queries, check inbox again for new blocks (interrupt stale mining)

13. **Query generator** (`query_generator.py`):
    - Separate process or thread in coordinator
    - Generates `Query` objects at `query_rate` per second
    - Each query: random conditioning input, incrementing task_id per "user", random fee
    - Pushes to mempool via `mempool.add_query()`
    - Stops after `num_queries` total

### Phase 5: Orchestration & Metrics (steps 14–15)

14. **Coordinator** (`coordinator.py`):
    - Main orchestration process:
      1. Load config
      2. Initialize shared mempool, network bus, canonical blockchain
      3. Spawn N miner processes
      4. Start query generator
      5. Main loop:
         a. Listen on coordinator queue for miner-submitted blocks
         b. Validate block against canonical chain
         c. If valid: add to chain, broadcast `NewBlock` to all miners via network bus, apply rewards/transactions
         d. Log block event
         e. Return un-mined queries from losing miners back to mempool
      6. Termination: after target block height or all queries processed or timeout
      7. Dump metrics

15. **Metrics** (`metrics.py`):
    - Collect per-block: height, miner_id, timestamp, num_queries, proving_time, lottery_attempts, time_since_last_block
    - Aggregate: avg block time, orphan rate, total proofs generated, miner win distribution, avg queries per block
    - `dump_metrics(filepath)`: write JSON file
    - Console logging via Python `logging` module (structured format)

### Phase 6: Testing (step 16)

16. **Tests**:
    - `test_crypto.py`: seed derivation determinism, lottery threshold, hashing
    - `test_blockchain.py`: block validation (valid/invalid), chain fork resolution
    - `test_zkp.py`: prove → verify round-trip with tiny model (integration test)
    - `test_e2e.py`: run 2 miners, 5 queries, verify blocks produced and chain valid

---

## Relevant Files (existing, for reference)

- `src/poml/utils.py` — Can reference `seed_everything()`, `get_device()` patterns
- `src/poml/hooks.py` — Not directly needed for simulation
- `experiments/exp1_activation_divergence/run_sd.py` — Reference for SD inference patterns
- `paper/4_protocols.tex` — Canonical protocol specification (Algorithm 1, block structure)
- `paper/5_reduction.tex` — Lottery mechanism details
- `paper/macros.tex` — Symbol definitions for cross-referencing

---

## Verification

1. **Model export**: Run `model/tiny_unet.py` → confirm `network.onnx` is valid ONNX (`onnx.checker.check_model()`)
2. **EZKL setup**: Run `model/setup_ezkl.py` → confirm all 6 artifacts generated (settings.json, network.ezkl, pk.key, vk.key, kzg.srs)
3. **ZKP round-trip**: Call `generate_proof()` then `verify_proof()` → returns True
4. **Single miner**: Run simulation with 1 miner, 3 queries → blocks produced, chain valid
5. **Multi miner**: Run with 4 miners, 20 queries → verify only one block per height, metrics JSON written
6. **Lottery fairness**: Run with 8 miners, 100+ queries → check miner win distribution is roughly uniform
7. **pytest**: All unit and integration tests pass

---

## Decisions

- **Standalone project** (not inside existing poml/) per user request
- **Tiny U-Net** (~2-5K params, 8×8 spatial) as stand-in model to keep EZKL proof times feasible (seconds, not hours)
- **Account-based** coin system (simpler than UTXO for PoC)
- **Meta-proofs**: random hex strings, verification always returns True
- **No optimistic challenge protocol** — excluded from scope
- **No real encryption** of outputs — stub that wraps output bytes (real encryption is orthogonal to consensus simulation)
- **Signatures**: simple Ed25519 via `cryptography` lib, or HMAC stub if keeping deps minimal
- **multiprocessing.Queue** for all IPC — no external message broker needed
- **Single machine** — all processes local, network latency simulated via sleep/timer

## Further Considerations

1. **EZKL proof time**: If even the tiny 8×8 U-Net is too slow (>30s per proof), we can shrink to a 2-layer MLP while keeping the channel structure conceptually "U-Net-like". Test this early in Phase 2. **Recommendation**: start with 8×8 U-Net, have a fallback 4×4 variant.
2. **Process coordination on shutdown**: Miners need clean shutdown signals. Use `multiprocessing.Event` for graceful termination. Also need timeout guards to prevent zombie processes.
3. **Difficulty auto-tuning**: For the PoC, difficulty is static from config. A stretch goal would be Bitcoin-style difficulty adjustment every N blocks—but this is out of scope for initial implementation.
