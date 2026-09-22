#!/usr/bin/env python3
"""One-off analysis of exp5b results to inform the exp5c Q x M grid design."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).parent / "results" / "exp5b_ratio_scatter" / "runs.csv"

rows = []
with RUNS.open(newline="") as fh:
    for row in csv.DictReader(fh):
        if row["status"] == "adopted" and row["percentage_wasted_work"]:
            rows.append(
                (
                    int(row["miners"]),
                    int(row["query_pool_size"]),
                    float(row["q_over_m"]),
                    float(row["target_block_time_s"]),
                    float(row["percentage_wasted_work"]),
                )
            )

print(f"adopted rows: {len(rows)}")
ms = sorted({r[0] for r in rows})
qs = sorted({r[1] for r in rows})
ratios = sorted(r[2] for r in rows)
print(f"M range: {ms[0]} .. {ms[-1]}")
print(f"Q range: {qs[0]} .. {qs[-1]}")
print(f"Q/M range: {ratios[0]:.4f} .. {ratios[-1]:.1f}")

# Wasted work vs Q/M deciles (pooled across targets; check per-target too).
for target in (300.0, 600.0, 900.0):
    sel = [r for r in rows if r[3] == target]
    print(f"\n== target {target:g}s: {len(sel)} runs ==")
    buckets: dict[int, list[float]] = defaultdict(list)
    for m, q, ratio, _, wasted in sel:
        # log10 ratio bins of width 0.25
        import math

        b = math.floor(math.log10(ratio) / 0.25)
        buckets[b].append((ratio, wasted, m, q))
    for b in sorted(buckets):
        vals = buckets[b]
        wasted_vals = [v[1] for v in vals]
        lo = 10 ** (b * 0.25)
        hi = 10 ** ((b + 1) * 0.25)
        print(
            f"  Q/M [{lo:8.3f},{hi:9.3f}): n={len(vals):4d} "
            f"median={statistics.median(wasted_vals):6.2f}% "
            f"mean={statistics.fmean(wasted_vals):6.2f}% "
            f"min={min(wasted_vals):6.2f}% max={max(wasted_vals):6.2f}% "
            f"stdev={statistics.pstdev(wasted_vals):6.2f}"
        )

# For the low-ratio regime, how does wasted work vary with M at fixed ratio?
print("\n== low-ratio regime (Q/M < 10), target 600s, wasted vs M ==")
sel = [r for r in rows if r[3] == 600.0 and r[2] < 10.0]
import math

mbuckets: dict[int, list[float]] = defaultdict(list)
for m, q, ratio, _, wasted in sel:
    mbuckets[math.floor(math.log10(m))].append(wasted)
for b in sorted(mbuckets):
    vals = mbuckets[b]
    print(
        f"  M in [10^{b}, 10^{b+1}): n={len(vals):4d} median={statistics.median(vals):6.2f}% "
        f"min={min(vals):6.2f}% max={max(vals):6.2f}% stdev={statistics.pstdev(vals):6.2f}"
    )

print("\n== near-boundary regime (Q/M in [1,2)), wasted vs M, all targets ==")
sel = [r for r in rows if 1.0 <= r[2] < 2.0]
mbuckets = defaultdict(list)
for m, q, ratio, t, wasted in sel:
    mbuckets[math.floor(math.log10(m))].append(wasted)
for b in sorted(mbuckets):
    vals = mbuckets[b]
    print(
        f"  M in [10^{b}, 10^{b+1}): n={len(vals):4d} median={statistics.median(vals):6.2f}% "
        f"min={min(vals):6.2f}% max={max(vals):6.2f}% stdev={statistics.pstdev(vals):6.2f}"
    )

# Where is the knee? smallest ratio where median wasted < 5% and < 1%
print("\n== knee analysis (pooled) ==")
for thresh in (50.0, 25.0, 10.0, 5.0, 1.0):
    above = [r[2] for r in rows if r[4] < thresh]
    below = [r[2] for r in rows if r[4] >= thresh]
    if above and below:
        print(
            f"  wasted<{thresh:5.1f}%: min Q/M = {min(above):8.3f}; "
            f"wasted>={thresh:5.1f}%: max Q/M = {max(below):8.3f}"
        )
