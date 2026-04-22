# Interruptible mining: abort in-flight proofs on new blocks

## What changed

Miners now abort an in-flight ezkl proof immediately when a NEW_BLOCK lands
on the network (or a STOP signal is received), rather than finishing the
current ~60–90s proof and only then checking their inbox.

- `src/poml_sim/miner.py`:
  - New module-level worker `_prove_worker` — runs
    `run_inference_and_prove` in an isolated subprocess and reports the
    result (or any exception) back through a result queue.
  - New helper `_terminate_proc` — SIGTERM, wait 5s, SIGKILL as a fallback.
  - New method `MinerProcess._prove_interruptible` — spawns the worker,
    polls the network inbox at 100 ms while the proof runs, and tears down
    the worker the instant the chain advances (or stop is signalled).
  - The block-production loop in `run()` now calls `_prove_interruptible`
    instead of `run_inference_and_prove` directly. On abort it `break`s
    out of the inner per-query loop and restarts mining at the new tip.
  - `poml_sim.zkp` is eagerly imported once in `run()` so every forked
    proof worker inherits the already-loaded module instead of paying
    ezkl's import cost on every fork.
  - `MinerProcess` is now `daemon=False`. Python forbids daemon processes
    from spawning their own children; since the miner now spawns a
    grandchild proof worker, the miner itself must be non-daemon. The
    coordinator's `_shutdown()` already cleans up miners explicitly
    (stop_event → STOP → `join(timeout=10)` → `terminate()` fallback),
    so the OS-level auto-kill that daemon=True provides is redundant.

- `tests/test_miner.py` (new) — 5 tests covering:
  - NEW_BLOCK mid-proof → abort, local tip advances, returns None in <5s
  - STOP mid-proof → abort, `stop_event` set, returns None in <5s
  - successful proof → returns `(output_values, proof_bytes)` verbatim
  - worker exception → returns None cleanly
  - stale NEW_BLOCK (height ≤ local_height) is ignored; proof completes

## Why

Before this change, the miner only checked its inbox **between** queries
— i.e. between whole proof cycles. Because a single proof takes ~60–90s
and a network broadcast lands within `network_latency_ms` (default 100 ms),
a miner could spend up to 90 s grinding on a proof at height `h` while the
rest of the network had already accepted someone else's height-`h` block.
That wasted CPU and delayed the loser from re-joining the race at `h+1`.

The new behaviour matches how real PoW miners react to a tip change:
drop everything and switch to the new template as soon as you see it.

## How it works

`ezkl.prove` is a blocking native (Rust) call — there's no Python-level
hook to cancel it. The only way to interrupt it is at the OS level, so the
proof is now delegated to a short-lived child subprocess per proof call:

1. `MinerProcess` spawns `_prove_worker` via `multiprocessing.get_context()
   .Process`. On Linux (default `fork`) the child inherits the miner's
   already-loaded `poml_sim.zkp` module.
2. The parent loops, polling its inbox at `_INBOX_POLL_INTERVAL_S` (100 ms).
   - If `_process_inbox()` reports the chain advanced, or `stop_event` is
     set, the parent calls `_terminate_proc(proc)` (SIGTERM → 5 s → SIGKILL)
     and returns `None`. The caller in `run()` sees `None` and `break`s
     out of the inner per-query loop; the outer `while` iteration re-reads
     the (now-updated) tip and starts a fresh block attempt.
   - Otherwise the parent waits up to 100 ms for the worker's result on
     `result_q`. On success it returns `(output_values, proof_bytes)`;
     on worker exception or crash it returns `None`.
3. A `finally` block guarantees the child is reaped and the result queue's
   feeder thread is joined on every exit path.

Abort latency is dominated by SIGTERM teardown, not polling — on a fake
Python `time.sleep`-based proof the subprocess dies in milliseconds; on a
real ezkl proof, SIGTERM unblocks the Rust runtime promptly.

## Migration / impact

- No API or config changes. Existing `config.yaml` files and the
  `Coordinator` interface are unchanged.
- Observable behaviour change: when N miners race on the same height, the
  losers will log `proof N/M aborted after X.Xs — restarting at height H+1`
  and begin the next round immediately rather than continuing on the
  stale template. Total wall-clock throughput should improve slightly
  (fewer wasted proofs) and the inter-block time variance should drop.
- Resource: one extra short-lived subprocess per proof. The forked
  grandchild inherits the parent's CPU affinity mask and
  `RAYON/OMP/MKL_NUM_THREADS` env vars set by `scripts/run.py`, so the
  `cpu_limit` budget is still respected.
- Platform: `_prove_interruptible` uses the default multiprocessing
  context. Tests are marked to require the `fork` start method (Linux)
  because they monkeypatch `poml_sim.zkp.run_inference_and_prove` and
  rely on fork-inherited module state. The production code path works
  under spawn too (the worker re-imports `poml_sim.zkp` fresh), just
  untested here.
