#!/usr/bin/env python3
"""Main entry point for the PoML simulation."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from poml_sim.config import load_config
from poml_sim.coordinator import Coordinator


def main() -> None:
    # Route all logging to a timestamped file so the console stays clean.
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        filename=str(log_path),
        filemode="w",
    )

    # ezkl emits a burst of INFO lines per proof. Mute the pure-noise loggers,
    # and on ezkl.pfsys keep only the "proof took N.NNs" summary by filtering
    # out the "proof started" / "loaded proving key" pre-amble.
    for name in ("ezkl.graph.model", "ezkl.circuit.table"):
        logging.getLogger(name).setLevel(logging.WARNING)

    class _DropPfsysNoise(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return "proof started" not in msg and "loaded proving key" not in msg

    logging.getLogger("ezkl.pfsys").addFilter(_DropPfsysNoise())

    # One-line pointer on stderr so the user knows where output went.
    print(f"Logging to {log_path}", file=sys.stderr)

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    _apply_cpu_limit(config.cpu_limit, config.num_miners)

    coordinator = Coordinator(config)
    coordinator.run(timeout=config.simulation_timeout)


def _apply_cpu_limit(cpu_limit: int | None, num_miners: int) -> None:
    """Pin the simulation to `cpu_limit` cores (Linux only).

    Applied *before* miner processes are spawned so the affinity mask and the
    thread-count env vars are inherited. Without the env-var caps, rayon/BLAS
    inside each miner would default to spawning one thread per visible core
    (all 32 on this host) and the kernel would just thrash them across the
    pinned 16 — correct but slow. Dividing by num_miners gives each miner its
    fair share of the capped pool.
    """
    if cpu_limit is None:
        return

    logger = logging.getLogger("poml_sim.run")

    if hasattr(os, "sched_setaffinity"):
        available = sorted(os.sched_getaffinity(0))
        if cpu_limit >= len(available):
            logger.info(
                "cpu_limit=%d >= available cores (%d); not restricting",
                cpu_limit,
                len(available),
            )
        else:
            chosen = set(available[:cpu_limit])
            os.sched_setaffinity(0, chosen)
            logger.info(
                "Pinned simulation to %d CPUs: %s",
                len(chosen),
                sorted(chosen),
            )
    else:
        logger.warning(
            "os.sched_setaffinity unavailable on this platform; "
            "cpu_limit will only be enforced via thread-count env vars"
        )

    # Per-miner thread budget. Floor at 1 so tiny cpu_limit values still work.
    per_miner_threads = max(1, cpu_limit // max(1, num_miners))
    for var in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(per_miner_threads))
    logger.info(
        "Thread caps set: %s=%d (per-miner budget = cpu_limit // num_miners)",
        "RAYON_NUM_THREADS/OMP_NUM_THREADS/MKL_NUM_THREADS",
        per_miner_threads,
    )


if __name__ == "__main__":
    main()
