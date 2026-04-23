#!/usr/bin/env python3
"""Bar plots for Experiment 2: wasted-work ratio in PoML.

Reads the two sweep CSVs produced by exp2_wasted_work.py and renders a
two-panel figure:
  Left  – Block-time sweep (4 miners fixed): wasted ratio vs target block time
  Right – Miner-count sweep (300 s target): wasted ratio vs number of miners

Usage:
    python experiments/plot_exp2_wasted_work.py
    python experiments/plot_exp2_wasted_work.py --out path/to/plot.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_BLOCKTIME_CSV = _RESULTS_DIR / "exp2_blocktime_sweep.csv"
_MINER_CSV = _RESULTS_DIR / "exp2_miner_sweep.csv"

# Colour palette matching exp1 style
_BAR_COLOR = "#cfe2ff"
_EDGE_COLOR = "#1f4e79"
_ACCENT = "#c0392b"


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _wasted_pct(row: dict) -> float:
    return float(row["wasted_ratio"]) * 100


def render(out_path: Path) -> None:
    """Save the two-panel wasted-work figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 13})

    bt_rows = sorted(_read_csv(_BLOCKTIME_CSV), key=lambda r: float(r["target_block_time_s"]))
    mn_rows = sorted(_read_csv(_MINER_CSV), key=lambda r: int(r["num_miners"]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ── Left: block-time sweep ──────────────────────────────────────────────
    ax = axes[0]
    xs = [int(float(r["target_block_time_s"])) for r in bt_rows]
    ys = [_wasted_pct(r) for r in bt_rows]
    bars = ax.bar(
        range(len(xs)),
        ys,
        color=_BAR_COLOR,
        edgecolor=_EDGE_COLOR,
        linewidth=1.5,
        width=0.5,
        zorder=3,
    )
    for bar, val in zip(bars, ys):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            color=_EDGE_COLOR,
        )
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{x}s" for x in xs])
    ax.set_xlabel("Target block time (miners = 4)")
    ax.set_ylabel("Wasted proofs (%)")
    ax.set_title("Block-time sweep")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Right: miner-count sweep ────────────────────────────────────────────
    ax = axes[1]
    xs2 = [int(r["num_miners"]) for r in mn_rows]
    ys2 = [_wasted_pct(r) for r in mn_rows]
    bars2 = ax.bar(
        range(len(xs2)),
        ys2,
        color=_BAR_COLOR,
        edgecolor=_EDGE_COLOR,
        linewidth=1.5,
        width=0.5,
        zorder=3,
    )
    for bar, val in zip(bars2, ys2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            color=_EDGE_COLOR,
        )
    ax.set_xticks(range(len(xs2)))
    ax.set_xticklabels([f"{x}" for x in xs2])
    ax.set_xlabel("Number of miners (target = 300 s)")
    ax.set_ylabel("Wasted proofs (%)")
    ax.set_title("Miner-count sweep")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("PoML wasted-work ratio", fontsize=16)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_RESULTS_DIR / "exp2_wasted_work_plot.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()
    render(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
