"""Run a shared-setup DeepProve N,K sweep and render a 3-D cost plot."""
import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path
import re
from threading import Thread
from queue import Empty, Queue

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

def make_pairs(max_context, count, seed):
    rng = random.Random(seed)
    pairs = {(2, 1), (2, max_context - 2), (max_context // 2, 1)}
    while len(pairs) < count:
        n = rng.randint(2, max_context - 1)
        k = rng.randint(1, max_context - n)
        pairs.add((n, k))
    return sorted(pairs, key=lambda x: (x[0] + x[1], x[0]))


def make_design(max_context, unique_count, repeats, seed):
    """Include non-power-of-two boundaries and repeated anchors by construction."""
    boundaries = [7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64]
    anchors = []
    for s in boundaries:
        if s <= max_context:
            n = max(2, s // 2)
            if n + 1 <= s:
                anchors.append((n, s - n))
    base = set(anchors)
    # Fill the requested number of unique cells after preserving every anchor.
    rng = random.Random(seed)
    while len(base) < unique_count:
        n = rng.randint(2, max_context - 1)
        k = rng.randint(1, max_context - n)
        base.add((n, k))
    extras = [pair for pair in base if pair not in anchors]
    base = anchors + sorted(extras, key=lambda x: (x[0] + x[1], x[0]))[: unique_count - len(anchors)]
    if repeats >= len(anchors):
        repeated = anchors
    else:
        indices = np.linspace(0, len(anchors) - 1, repeats, dtype=int)
        repeated = [anchors[i] for i in sorted(set(indices))]
    return base + [pair for pair in repeated for _ in range(3)]


def plot(df, out):
    grouped = df.assign(online_ms=df.inference_time + df.prove_full).groupby(
        ["min_user_len", "max_context"], as_index=False
    ).online_ms.mean()
    n = grouped["min_user_len"].to_numpy(float)
    k = grouped["max_context"].to_numpy(float) - n
    cost = grouped.online_ms.to_numpy(float)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    points = ax.scatter(n, k, cost / 1000, c=cost / 1000, cmap="viridis", s=42)
    if len(df) >= 3:
        ax.plot_trisurf(n, k, cost / 1000, cmap="viridis", alpha=0.28, linewidth=0.2)
    ax.set_xlabel("N: prompt tokens")
    ax.set_ylabel("K: output tokens")
    ax.set_zlabel("online cost (s)")
    ax.set_title("DeepProve GPT-2 online inference + proving cost")
    fig.colorbar(points, ax=ax, pad=0.1, label="online cost (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fit_campaign(df):
    """Fit staged inference, proof, and combined models with a held-out split."""
    df = df.copy()
    df["online_ms"] = df.inference_time + df.prove_full
    df["P"] = 2 ** np.ceil(np.log2(df.max_context)).astype(int)
    # Repeated anchor trials are averaged before the split so they do not leak.
    cells = df.groupby(["min_user_len", "max_context"], as_index=False).mean(numeric_only=True)
    rng = np.random.default_rng(20260910)
    test = rng.random(len(cells)) < 0.2
    if test.all() or (~test).sum() < 5:
        test[:] = False
        test[::5] = True

    def regression(x, y):
        c, *_ = np.linalg.lstsq(x[~test], y[~test], rcond=None)
        pred = x[test] @ c
        return {"coefficients": c.tolist(), "rmse_ms": float(np.sqrt(np.mean((pred-y[test])**2))),
                "mae_ms": float(np.mean(np.abs(pred-y[test]))),
                "max_relative_error": float(np.max(np.abs(pred-y[test]) / np.maximum(y[test], 1e-9)))}

    s = cells.max_context.to_numpy(float)
    p = cells.P.to_numpy(float)
    inf = cells.inference_time.to_numpy(float)
    proof = cells.prove_full.to_numpy(float)
    online = cells.online_ms.to_numpy(float)
    models = {
        "inference_ab": regression(np.c_[s, s*s], inf),
        "proof_c0c1c2": regression(np.c_[np.ones_like(p), p, p*np.log2(p)], proof),
        "combined": regression(np.c_[np.ones_like(s), s, s*s, p, p*np.log2(p)], online),
    }
    for column in ("prove_claims", "prove_commitment_opening", "prove_full"):
        if column in cells:
            models[f"{column}_c0c1c2"] = regression(
                np.c_[np.ones_like(p), p, p*np.log2(p)], cells[column].to_numpy(float)
            )
    return {"n_trials": int(len(df)), "n_cells": int(len(cells)),
            "n_train_cells": int((~test).sum()), "n_test_cells": int(test.sum()),
            "models": models}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-context", type=int, default=64)
    ap.add_argument("--pairs", type=int, default=100, help="number of distinct pairs")
    ap.add_argument("--repeats", type=int, default=5, help="number of boundary anchors to repeat three times")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--output-dir", type=Path, default=Path("experiments/results/complexity"))
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--cuda", action="store_true", help="build and use DeepProve CUDA backend")
    ap.add_argument("--plan-only", action="store_true", help="print the design and estimate without running")
    args = ap.parse_args()
    if args.max_context < 4 or args.pairs < 3:
        raise SystemExit("max-context must be >=4 and pairs must be >=3")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"pairs_{stamp}.csv"
    png_path = args.output_dir / f"pairs_{stamp}.png"
    fit_path = args.output_dir / f"pairs_{stamp}.json"
    pair_list = make_design(args.max_context, args.pairs, args.repeats, args.seed)
    # Conservative estimate from the observed GPT-2 CPU proving curve
    # (roughly 40--75 s for total lengths 4--64), plus setup headroom.
    per_length = 0.9 if args.cuda else 0.65
    setup_seconds = 120 if args.cuda else 600
    estimated_seconds = setup_seconds + sum(40 + per_length * (n + k) for n, k in pair_list)
    print(f"Estimated wall time: {estimated_seconds / 3600:.1f} h "
          f"({len(pair_list)} trials, {args.pairs} distinct pairs, max context {args.max_context})")
    if args.plan_only:
        print("Pairs:", ",".join(f"{n}:{k}" for n, k in pair_list))
        return
    pair_arg = ",".join(f"{n}:{k}" for n, k in pair_list)
    root = Path(__file__).resolve().parents[1]
    dp = root / ".scratch/deep-prove"
    binary = dp / "target/release/bench-llm"
    run_env = dict(os.environ)
    if args.cuda:
        cuda_bin = Path("/usr/local/cuda/bin")
        if cuda_bin.is_dir():
            run_env["PATH"] = str(cuda_bin) + os.pathsep + run_env.get("PATH", "")
        run_env.setdefault("CUDA_HOME", "/usr/local/cuda")
    if not args.skip_build:
        build = ["cargo", "build", "--release"]
        if args.cuda:
            build += ["--features", "cuda"]
        build += ["--bin", "bench-llm"]
        subprocess.run(build, cwd=dp, env=run_env, check=True)
    command = [
        str(binary), "--model", "gpt2", "--hf", "openai-community/gpt2",
        "--pairs", pair_arg, "--bench", str(csv_path), "--num-threads", str(args.threads),
        "--fixed-length",
    ]
    # RNG_SEED fixes the generated token prefixes and quantization sample.
    run_env["RNG_SEED"] = str(args.seed)
    process = subprocess.Popen(command, cwd=dp / "zkml", env=run_env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    setup_progress = tqdm(total=1, desc="DeepProve setup", unit="setup")
    progress = tqdm(total=len(pair_list), desc="DeepProve trials", unit="trial")
    setup_done = False
    for line in process.stdout:
        if "POML_EVENT" in line:
            if "stage=setup_done" in line and not setup_done:
                setup_done = True
                setup_progress.update(1)
                setup_progress.close()
                print("Setup complete; proving trials are starting.")
            if "stage=done" in line:
                progress.update(1)
        if "panicked" in line.lower():
            print(line.rstrip())
    code = process.wait()
    progress.close()
    if not setup_done:
        setup_progress.close()
    if code:
        raise SystemExit(f"DeepProve exited with status {code}")
    # DeepProve writes the bench path relative to its working directory.
    actual_csv = dp / "zkml" / csv_path
    if not actual_csv.exists():
        actual_csv = csv_path
    df = pd.read_csv(actual_csv)
    result = fit_campaign(df)
    result.update({"seed": args.seed, "max_context": args.max_context,
                   "cuda": args.cuda,
                   "pairs": pair_list, "csv": str(actual_csv), "plot": str(png_path)})
    fit_path.write_text(json.dumps(result, indent=2) + "\n")
    plot(df, png_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
