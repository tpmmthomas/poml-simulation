#!/usr/bin/env python3
"""Main entry point for the PoML simulation."""

import logging
import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from poml_sim.config import load_config
from poml_sim.coordinator import Coordinator


def main() -> None:
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    coordinator = Coordinator(config)
    coordinator.run(timeout=600.0)


if __name__ == "__main__":
    main()
