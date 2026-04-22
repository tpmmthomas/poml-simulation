#!/usr/bin/env python3
"""Main entry point for the PoML simulation."""

import logging
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

    # One-line pointer on stderr so the user knows where output went.
    print(f"Logging to {log_path}", file=sys.stderr)

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    coordinator = Coordinator(config)
    coordinator.run(timeout=600.0)


if __name__ == "__main__":
    main()
