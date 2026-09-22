#!/usr/bin/env python3
"""Run all current live PoML experiments through the established campaign entry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_live_llm_experiments import main as run_live  # noqa: E402


def main(argv=None):
    """Delegate to fresh-work mining; the old replay-only campaign is superseded."""
    return run_live(["all", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
