"""Shared helpers for the experiments drivers.

Responsibilities:
- Build a `SimConfig` from per-experiment overrides and drive `Coordinator`
  programmatically (mirrors `scripts/run.py` but reroutes logging to a
  per-run file so each experiment run is isolated).
- Parse completed-run log files for calibration reference data and for
  counting proofs that were produced but never confirmed (wasted work).
- Rescale difficulty targets based on an observed/target block time pair.
- Compute summary statistics over a list of per-block wall-clock durations.
"""

from __future__ import annotations

import logging
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow `python experiments/foo.py` to find the poml_sim package under src/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from poml_sim.config import SimConfig  # noqa: E402
from poml_sim.coordinator import Coordinator  # noqa: E402


HEX_PREFIX = "0x"
# 64 hex chars = 256 bits. Pad up so ordering / arithmetic stays intuitive.
DIFFICULTY_HEX_WIDTH = 64

# Cap every experiment to this many CPU cores, matching the conditions of the
# reference calibration log (logs/run_20260423_050539.log). Keeping CPU budget
# constant across PoML/PoW and across miner-count sweeps makes per-block time
# comparisons meaningful. Override by passing cpu_limit explicitly in overrides.
DEFAULT_CPU_LIMIT = 16

logger = logging.getLogger(__name__)


@dataclass
class PomlRunResult:
    """Everything an experiment driver needs after a single PoML run."""

    block_times: list[float] = field(default_factory=list)  # seconds between blocks
    total_blocks: int = 0
    completed_proofs: int = 0   # "proof N/M done" lines across all miners
    included_proofs: int = 0    # sum of lottery_attempts over confirmed blocks
    wasted_proofs: int = 0
    wasted_ratio: float = 0.0
    log_path: Path = Path()
    difficulty_hex: str = ""
    config_snapshot: dict = field(default_factory=dict)


def scale_difficulty(current_hex: str, observed_time: float, target_time: float) -> str:
    """Scale a difficulty target so expected block time moves observed -> target.

    Block interval is inversely proportional to difficulty (success probability
    per attempt is difficulty / 2^256). To go from observed_time to target_time
    we multiply difficulty by observed_time / target_time.
    """
    if observed_time <= 0 or target_time <= 0:
        raise ValueError("times must be positive")
    current_int = int(current_hex, 16)
    new_int = int(current_int * observed_time / target_time)
    # Clamp to the valid 256-bit range.
    new_int = max(1, min(new_int, (1 << 256) - 1))
    return HEX_PREFIX + format(new_int, "x").rjust(DIFFICULTY_HEX_WIDTH, "0")


def parse_log_for_calibration(log_path: Path) -> tuple[str, float]:
    """Extract (difficulty_hex_full, avg_block_time_seconds) from a completed run.

    The coordinator log header truncates difficulty to the first 20 chars, so we
    reconstruct a usable hex by padding with 'f' (matches the existing
    0x3fff...ff style). If the exact hex is known, the caller should override.
    """
    text = log_path.read_text()

    # Header line: "Miners: N | Queries: N | Difficulty: 0xabc..."
    m = re.search(r"Difficulty:\s*(0x[0-9a-fA-F]+)", text)
    if not m:
        raise ValueError(f"Could not find Difficulty header in {log_path}")
    diff_prefix = m.group(1)
    # The header truncates to 20 chars; rehydrate by padding with 'f' if needed.
    raw = diff_prefix[2:]
    if len(raw) < DIFFICULTY_HEX_WIDTH:
        raw = raw + "f" * (DIFFICULTY_HEX_WIDTH - len(raw))
    difficulty = HEX_PREFIX + raw[:DIFFICULTY_HEX_WIDTH]

    # Footer line: "Avg block time: 214.28s"
    m = re.search(r"Avg block time:\s*([0-9.]+)s", text)
    if not m:
        raise ValueError(f"Could not find 'Avg block time' footer in {log_path}")
    avg_block_time = float(m.group(1))

    return difficulty, avg_block_time


# "Miner 0: proof 1/10 done (102.7s), lottery hash=0x... -> miss"
_PROOF_DONE_RE = re.compile(r"Miner \d+: proof \d+/\d+ done")


def count_completed_proofs(log_path: Path) -> int:
    """Count 'proof N/M done' lines in a run log — all proofs across all miners,
    whether the lottery was a WIN or a miss, whether they ended up in a block or not.
    """
    n = 0
    with open(log_path) as f:
        for line in f:
            if _PROOF_DONE_RE.search(line):
                n += 1
    return n


def summarize(values: list[float]) -> dict[str, float]:
    """Mean/min/max/stdev/variance over a list of floats. Empty-safe."""
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0,
                "variance": 0.0, "stdev": 0.0}
    n = len(values)
    mean = statistics.fmean(values)
    variance = statistics.pvariance(values) if n >= 1 else 0.0  # population variance
    stdev = variance ** 0.5
    return {
        "count": n,
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "variance": variance,
        "stdev": stdev,
    }


class _ProgressFilter(logging.Filter):
    """Select only high-signal events for the progress stream.

    Keeps per-block confirmations, per-proof completions, and the coordinator
    header/summary. Drops the high-volume query-generator and ezkl noise so
    the console stream stays readable during a long run.
    """

    _KEEP_PATTERNS = (
        "Block ",                # "Block N added" / ">>> Block N confirmed"
        "proof ",                # "Miner X: proof N/M done" / "starting proof"
        "PoML Simulation Starting",
        "Simulation Complete",
        "Avg block time",
        "Total blocks",
        "Total proofs",
        "Miner wins",
        "apply_cpu_limit",       # surface the CPU pin message so it's visible
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return any(p in msg for p in self._KEEP_PATTERNS)


def _install_run_log_handler(log_path: Path) -> list[logging.Handler]:
    """Attach a fresh file handler (full log) and a filtered stdout handler
    (progress stream) to the root logger, for the duration of one run.

    Each experiment run produces its own file log so parsing can be done
    per-run. The progress stream surfaces block- and proof-level events to
    the console so users tailing `run_all.sh` output can see progress
    without having to tail the per-run log file.
    """
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    file_h = logging.FileHandler(str(log_path), mode="w")
    file_h.setLevel(logging.INFO)
    file_h.setFormatter(fmt)

    console_h = logging.StreamHandler(sys.stdout)
    console_h.setLevel(logging.INFO)
    console_h.setFormatter(fmt)
    console_h.addFilter(_ProgressFilter())

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_h)
    root.addHandler(console_h)

    # Mute ezkl noise the same way scripts/run.py does.
    for name in ("ezkl.graph.model", "ezkl.circuit.table"):
        logging.getLogger(name).setLevel(logging.WARNING)

    class _DropPfsysNoise(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return "proof started" not in msg and "loaded proving key" not in msg

    logging.getLogger("ezkl.pfsys").addFilter(_DropPfsysNoise())
    return [file_h, console_h]


def apply_cpu_limit(cpu_limit: int | None, num_miners: int) -> None:
    """Pin this process (and inherited children) to `cpu_limit` cores and cap
    BLAS/rayon thread counts per miner.

    Mirrors scripts/run.py:_apply_cpu_limit, with logging so a reader can see
    the cap was actually applied. Safe to call multiple times.
    """
    if cpu_limit is None:
        logger.info("apply_cpu_limit: cpu_limit=None, no CPU cap applied")
        return

    if hasattr(os, "sched_setaffinity"):
        available = sorted(os.sched_getaffinity(0))
        if cpu_limit >= len(available):
            logger.info("apply_cpu_limit: cpu_limit=%d >= available=%d, no affinity change",
                        cpu_limit, len(available))
        else:
            chosen = set(available[:cpu_limit])
            os.sched_setaffinity(0, chosen)
            logger.info("apply_cpu_limit: pinned to %d CPUs: %s",
                        len(chosen), sorted(chosen))
    else:
        logger.warning("apply_cpu_limit: sched_setaffinity unavailable on this platform")

    per_miner = max(1, cpu_limit // max(1, num_miners))
    for var in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(per_miner)
    logger.info("apply_cpu_limit: thread caps %s=%d (per-miner budget)",
                "RAYON/OMP/MKL_NUM_THREADS", per_miner)


def run_poml(
    overrides: dict,
    log_path: Path,
    timeout: float,
    stop_after_blocks: int | None = None,
) -> PomlRunResult:
    """Drive a single PoML simulation with the given config overrides.

    Starts from `SimConfig` defaults, applies `overrides`, routes all logging
    to `log_path`, pins CPUs, and runs the `Coordinator` for up to `timeout`
    seconds. If `stop_after_blocks` is set, a watcher thread sets the
    coordinator's stop_event as soon as the chain reaches that height — lets
    us pre-fill the mempool with more queries than we plan to confirm (which
    avoids the deterministic-lottery stall when only 1-2 queries remain).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Always pin experiments to DEFAULT_CPU_LIMIT (16) cores to match the
    # calibration-log run conditions, unless the caller explicitly overrides.
    overrides = {"cpu_limit": DEFAULT_CPU_LIMIT, **overrides}
    config = SimConfig(**overrides)
    # Install handlers FIRST so apply_cpu_limit's log messages land in the
    # per-run log file + master stdout stream, not /dev/null.
    handlers = _install_run_log_handler(log_path)
    apply_cpu_limit(config.cpu_limit, config.num_miners)

    import threading
    import time as _time

    coordinator = Coordinator(config)
    watcher: threading.Thread | None = None
    if stop_after_blocks is not None:
        def _watch() -> None:
            while not coordinator.stop_event.is_set():
                if coordinator.blockchain.get_height() >= stop_after_blocks:
                    logger.info(
                        "run_poml: reached %d blocks, signalling stop_event",
                        stop_after_blocks,
                    )
                    coordinator.stop_event.set()
                    return
                _time.sleep(0.5)
        watcher = threading.Thread(target=_watch, daemon=True, name="block-count-watcher")
        watcher.start()

    try:
        coordinator.run(timeout=timeout)
    finally:
        if watcher is not None:
            coordinator.stop_event.set()
            watcher.join(timeout=5.0)
        # Detach handlers so follow-up runs don't duplicate log lines.
        root = logging.getLogger()
        for h in handlers:
            root.removeHandler(h)
            h.close()

    # Block times: delta between consecutive confirmed block timestamps.
    block_metrics = coordinator.metrics.blocks
    block_times: list[float] = []
    for i in range(1, len(block_metrics)):
        dt = block_metrics[i].timestamp - block_metrics[i - 1].timestamp
        if dt > 0:
            block_times.append(dt)

    included = sum(b.lottery_attempts for b in block_metrics)
    completed = count_completed_proofs(log_path)
    # A block's own winning proof should always be in `completed` too, but guard
    # against log rotation / format drift by clamping.
    completed = max(completed, included)
    wasted = completed - included
    ratio = (wasted / completed) if completed else 0.0

    return PomlRunResult(
        block_times=block_times,
        total_blocks=len(block_metrics),
        completed_proofs=completed,
        included_proofs=included,
        wasted_proofs=wasted,
        wasted_ratio=ratio,
        log_path=log_path,
        difficulty_hex=config.difficulty,
        config_snapshot=config.model_dump(),
    )
