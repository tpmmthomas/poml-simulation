#!/usr/bin/env python3
"""Box-and-whisker plot of PoML end-to-end per-miner proof generation times.

Parses a run_all master log (or any log that contains lines of the form
``Miner X: proof N/M done (Xs)``) and renders a boxplot of the per-proof
end-to-end durations from the Experiment 1 section.

Usage:
    python experiments/plot_exp1_proof_times.py \
        experiments/results/run_all_20260423_075657.log

    # explicit output path
    python experiments/plot_exp1_proof_times.py LOG --out path/to/plot.png
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

# Section markers emitted by run_all.sh to delimit Experiment 1 output.
_EXP1_START = "[run_all] [2/3] Experiment 1"
_EXP1_END = "[run_all] [3/3]"
_PROOF_RE = re.compile(r"proof \d+/\d+ done \(([\d.]+)s\)")


def extract_exp1_proof_times(log_path: Path) -> list[float]:
    """Return end-to-end proof durations (seconds) from the Exp-1 section.

    If the log has no section markers (e.g. a standalone exp1 log),
    the whole file is scanned.
    """
    text = log_path.read_text()
    start = text.find(_EXP1_START)
    if start == -1:
        # No run_all markers -- assume the whole file is one experiment.
        section = text
    else:
        end = text.find(_EXP1_END, start)
        section = text[start:end] if end != -1 else text[start:]

    return [float(m.group(1)) for m in _PROOF_RE.finditer(section)]


def render_boxplot(times: list[float], out_path: Path) -> None:
    """Save a single-box boxplot with mean/median annotations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.boxplot(
        times,
        vert=False,
        showmeans=True,
        meanline=True,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#cfe2ff", edgecolor="#1f4e79"),
        medianprops=dict(color="#1f4e79", linewidth=2),
        meanprops=dict(color="#c0392b", linewidth=2, linestyle="--"),
        whiskerprops=dict(color="#1f4e79"),
        capprops=dict(color="#1f4e79"),
    )
    ax.set_xlabel("Time (s)")
    ax.set_title(f"PoML inference+proof generation time (N={len(times)})")
    ax.set_yticks([1])
    ax.set_yticklabels(["PoML"])
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    median = statistics.median(times)
    mean = statistics.mean(times)
    ax.annotate(
        f"median = {median:.1f}s",
        xy=(median, 1.22),
        fontsize=12,
        color="#1f4e79",
        ha="center",
        va="bottom",
    )
    ax.annotate(
        f"mean = {mean:.1f}s",
        xy=(mean, 0.78),
        fontsize=12,
        color="#c0392b",
        ha="center",
        va="top",
    )

    legend_elems = [
        Line2D([0], [0], color="#1f4e79", lw=2, label="median"),
        Line2D([0], [0], color="#c0392b", lw=2, linestyle="--", label="mean"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=10, handlelength=1.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Path to run_all master log.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Defaults to "
            "<log_dir>/exp1_proof_time_boxplot.png"
        ),
    )
    args = parser.parse_args()

    if not args.log.is_file():
        print(f"error: log not found: {args.log}", file=sys.stderr)
        return 2

    times = extract_exp1_proof_times(args.log)
    if not times:
        print("error: no 'proof N/M done (Xs)' lines found in Exp-1 section.", file=sys.stderr)
        return 1

    out = args.out or (args.log.parent / "exp1_proof_time_boxplot.png")
    render_boxplot(times, out)

    print(f"Saved: {out}")
    print(
        f"N={len(times)}  "
        f"mean={statistics.mean(times):.2f}s  "
        f"median={statistics.median(times):.2f}s  "
        f"min={min(times):.2f}s  "
        f"max={max(times):.2f}s  "
        f"stdev={statistics.stdev(times):.2f}s  "
        f"variance={statistics.variance(times):.2f}s^2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
