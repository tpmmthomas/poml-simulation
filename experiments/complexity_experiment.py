"""Fit an online PoML complexity schedule from DeepProve benchmark CSV rows."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _fit_design(x, y, terms):
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    err = y - pred
    loo = []
    for i in range(len(y)):
        c, *_ = np.linalg.lstsq(np.delete(x, i, 0), np.delete(y, i), rcond=None)
        loo.append(abs(x[i] @ c - y[i]) / max(abs(y[i]), 1e-9))
    return {"terms": terms, "coefficients_ms": coef.tolist(),
            "rmse_ms": float(np.sqrt(np.mean(err**2))),
            "mae_ms": float(np.mean(np.abs(err))),
            "max_relative_error": float(np.max(np.abs(err) / np.maximum(abs(y), 1e-9))),
            "loo_max_relative_error": float(np.max(loo))}


def fit(df):
    n = df["min_user_len"].to_numpy(float)
    s = df["max_context"].to_numpy(float)
    k = s - n
    # DeepProve proves one causal pass over s tokens; its benchmark inference
    # is also a full pass.  Thus the identifiable online schedule is a
    # low-degree function of s=N+K.  Setup and verification are ignored.
    y = (df["inference_time"].to_numpy(float) + df["prove_full"].to_numpy(float))
    designs = {
        "total_length_quadratic": (np.c_[np.ones_like(s), s, s * s], ["constant", "S", "S2"]),
        "two_variable": (np.c_[np.ones_like(s), n, k, n * k, k * k],
                         ["constant", "N", "K", "NK", "K2"]),
    }
    out = {name: _fit_design(x, y, terms) for name, (x, terms) in designs.items()}
    out["n_rows"] = int(len(df))
    out["recommended"] = "two_variable" if out["two_variable"]["loo_max_relative_error"] < out["total_length_quadratic"]["loo_max_relative_error"] else "total_length_quadratic"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, nargs="+")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    df = pd.concat((pd.read_csv(path) for path in a.csv), ignore_index=True)
    required = {"min_user_len", "max_context", "inference_time", "prove_full"}
    missing = required - set(df)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")
    result = fit(df.dropna(subset=required))
    text = json.dumps(result, indent=2)
    output = a.out or a.csv[0].with_suffix(".complexity.json")
    output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
