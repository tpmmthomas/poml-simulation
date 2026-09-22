"""Run structural GPT-2 proof checks and compile a gas manifest with symbolic weights."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from poml_sim.gpt2_work import (  # noqa: E402
    MAX_COMPONENT_ERROR,
    SCHEDULE_VERSION,
    check_lengths,
    component_error,
    compile_schedule,
    ledger_counts,
    proof_axis_macs,
    reference_counts,
)


def validation_pairs() -> list[tuple[int, int]]:
    """Cover every padding boundary, split extremes, and fresh prompts at fixed lengths."""
    boundaries = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]
    off_boundaries = [
        (2, 1),
        (3, 4),
        (4, 5),
        (7, 8),
        (8, 9),
        (15, 16),
        (16, 17),
        (31, 32),
    ]
    different_splits = [
        (2, 5),
        (6, 1),
        (2, 15),
        (16, 1),
        (2, 31),
        (32, 1),
        (2, 62),
        (63, 1),
    ]
    fresh_tokens = [(3, 4), (8, 9), (16, 17), (32, 32), (2, 1), (2, 1), (2, 3), (2, 4)]
    unique = list(dict.fromkeys(boundaries + off_boundaries + different_splits + fresh_tokens))
    totals = (
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        12,
        14,
        15,
        16,
        17,
        20,
        24,
        28,
        30,
        31,
        32,
        33,
        40,
        48,
        63,
        64,
    )
    for total in totals:
        pair = (max(2, total // 2), total - max(2, total // 2))
        if pair not in unique:
            unique.append(pair)
    for total in totals:
        for n in range(2, total):
            if len(unique) >= 40:
                return unique + fresh_tokens
            if (n, total - n) not in unique:
                unique.append((n, total - n))
    raise RuntimeError("could not construct validation design")


def load_ledger(path: Path) -> list[dict]:
    """Load complete verified records; reject interrupted/truncated files."""
    rows = []
    with path.open() as stream:
        for index, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise ValueError(f"Incomplete ledger record {index}")
            row = json.loads(line)
            if row.get("verified") is not True:
                raise ValueError(f"Unverified record {index}")
            rows.append(row)
    return rows


def source_provenance(deep_prove: Path) -> dict:
    """Fingerprint all Rust sources and manifests used in the patched build."""
    crypto = deep_prove.parent / "dp-crypto-work"
    digest = hashlib.sha256()
    for label, root in (("deep-prove", deep_prove), ("dp-crypto", crypto)):
        files = (
            sorted(root.rglob("*.rs"))
            if label == "dp-crypto"
            else sorted((root / "zkml" / "src").rglob("*.rs"))
        )
        files += [root / "Cargo.toml", root / "Cargo.lock"]
        for path in files:
            if path.is_file() and "target" not in path.parts:
                digest.update(f"{label}/{path.relative_to(root)}\0".encode())
                digest.update(path.read_bytes())
    return {
        "source_sha256": digest.hexdigest(),
        "deepprove_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=deep_prove, text=True
        ).strip(),
        "crypto_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=crypto, text=True
        ).strip(),
    }


def plot_counts(rows: list[dict], output: Path) -> None:
    """Plot separate resource families without inventing relative gas weights."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = [r["n"] for r in rows]
    k = [r["k"] for r in rows]
    metrics = [
        ([r["integer_mac"] / 1e9 for r in rows], "Inference", "Dense MACs (billions)"),
        (
            [r["sumcheck_factor"] / 1e9 for r in rows],
            "Proof: reference sumcheck",
            "Factor units (billions)",
        ),
        (
            [
                sum(int(key[4:]) * value for key, value in r.items() if key.startswith("msm_"))
                / 1e6
                for r in rows
            ],
            "Proof: commitment sizes",
            "Scalar/base entries (millions)",
        ),
    ]
    fig = plt.figure(figsize=(17, 5.3), layout="constrained")
    for index, (values, title, label) in enumerate(metrics, 1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        ax.scatter(n, k, values, c=values, cmap="viridis", s=5, rasterized=True)
        ax.set(xlabel="Input tokens N", ylabel="Output tokens K", zlabel=label, title=title)
        ax.view_init(elev=26, azim=-125)
    fig.suptitle(
        "GPT-2 / DeepProve reference counts · max context 64 · weights remain symbolic",
        fontsize=14,
    )
    fig.savefig(output / "work_counts.png", dpi=170)
    fig.savefig(output / "work_counts.pdf")
    plt.close(fig)


def analyze(ledger: Path, output: Path, provenance: dict, comparison: Path | None = None) -> dict:
    """Compile the boundary plans, check every trial, and export all allowed pairs."""
    rows = load_ledger(ledger)
    provenance = {
        **provenance,
        "accounting_sha256": hashlib.sha256(
            (ROOT / "src/poml_sim/gpt2_work.py").read_bytes()
        ).hexdigest(),
    }
    schedule = (
        json.loads(comparison.read_text()) if comparison else compile_schedule(rows, provenance)
    )
    (output / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    comparisons = []
    for row in rows:
        n, k = row["meta"]["n"], row["meta"]["k"]
        actual = ledger_counts(row)
        expected = reference_counts(n, k, schedule)
        if row["meta"].get("schema") != SCHEDULE_VERSION or row["meta"]["setup_max"] != 64:
            raise ValueError("Observed configuration does not match the reference")
        if actual["inference"] != expected["inference"] or actual["proof"].get(
            "field_axis_mac"
        ) != proof_axis_macs(n + k):
            raise ValueError(f"Exact inference/axis formula mismatch at {(n, k)}")
        error = component_error(actual["proof"], expected["proof"])
        if error > MAX_COMPONENT_ERROR:
            raise ValueError(f"Proof component error {error:.2%} exceeds tolerance at {(n, k)}")
        comparisons.append(
            {
                "n": n,
                "k": k,
                "inference_equal": actual["inference"] == expected["inference"],
                "proof_equal": actual["proof"] == expected["proof"],
                "max_proof_component_relative_error": error,
                "differences": {
                    key: {
                        "observed": actual["proof"].get(key, 0),
                        "reference": expected["proof"].get(key, 0),
                    }
                    for key in actual["proof"].keys() | expected["proof"].keys()
                    if actual["proof"].get(key, 0) != expected["proof"].get(key, 0)
                },
            }
        )
    report = {
        "schedule_sha256": schedule["sha256"],
        "trials": len(rows),
        "all_exact": all(r["inference_equal"] and r["proof_equal"] for r in comparisons),
        "comparisons": comparisons,
        "within_tolerance": True,
        "max_proof_component_relative_error": max(
            r["max_proof_component_relative_error"] for r in comparisons
        ),
        "validation_tolerance": MAX_COMPONENT_ERROR,
        "observed_run": provenance,
    }
    if comparison is not None:
        report["reference_comparison"] = {
            "sha256": schedule["sha256"],
            "all_exact": report["all_exact"],
            "within_tolerance": True,
        }
    all_pairs = []
    for n in range(2, 64):
        for k in range(1, 65 - n):
            counts = reference_counts(n, k, schedule)["combined"]
            all_pairs.append({"n": n, "k": k, **counts})
    keys = sorted(set().union(*(row.keys() for row in all_pairs)) - {"n", "k"})
    with (output / "all_pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["n", "k"] + keys, restval=0)
        writer.writeheader()
        writer.writerows(all_pairs)
    report["calculated_pairs"] = len(all_pairs)
    plot_counts(all_pairs, output)
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run(args) -> None:
    """Build, prove, verify, and compare structural counts with visible progress."""
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    deep_prove = args.deep_prove.resolve()
    pairs = validation_pairs()
    if args.pairs:
        pairs = [tuple(map(int, pair.split(":"))) for pair in args.pairs.split(",")]
        for n, k in pairs:
            check_lengths(n, k)
        if max(n + k for n, k in pairs) != 64:
            raise ValueError("Custom pairs must include total length 64 to preserve setup capacity")
    if args.plan_only:
        print(
            json.dumps(
                {
                    "pairs": pairs,
                    "trials": len(pairs),
                    "setup_max": 64,
                    "weights": "symbolic",
                    "backend": "cuda" if args.cuda else "cpu",
                },
                indent=2,
            )
        )
        return
    if args.analyze:
        provenance_path = output / "run.json"
        if not provenance_path.exists():
            raise ValueError("Analysis requires the saved run.json provenance")
        saved = json.loads(provenance_path.read_text())
        print(json.dumps(analyze(args.analyze, output, saved, args.compare_schedule), indent=2))
        return
    if args.calculate:
        n, k = map(int, args.calculate.split(":"))
        from poml_sim.backends import load_schedule

        schedule = load_schedule(
            args.compare_schedule
            or (output / "schedule.json" if (output / "schedule.json").exists() else None)
        )
        print(json.dumps(reference_counts(n, k, schedule), indent=2))
        return
    ledger = output / "ledger.jsonl"
    if ledger.exists():
        raise FileExistsError(
            f"Refusing to mix setups in {ledger}; use --analyze or a new directory"
        )
    env = os.environ.copy()
    env["RNG_SEED"] = str(args.setup_seed)
    env["ZKML_BIT_LEN"] = "12"
    env["RUST_LOG"] = "error"
    if args.prepare:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/prepare_deepprove_work.py"),
                "--deep-prove",
                str(deep_prove),
            ],
            check=True,
        )
    if args.cuda:
        env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
        env.setdefault("CUDA_HOME", "/usr/local/cuda")
        if args.device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    if not args.skip_build:
        build = ["cargo", "build", "--release", "--bin", "bench-llm"]
        if args.cuda:
            build += ["--features", "cuda"]
        subprocess.run(build, cwd=deep_prove, env=env, check=True)
    binary = deep_prove / "target" / "release" / "bench-llm"
    provenance = source_provenance(deep_prove)
    provenance.update(
        {
            "setup_seed": args.setup_seed,
            "token_seed": args.token_seed,
            "quantization_bits": 12,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "cuda": args.cuda,
            "threads": args.threads,
            "pairs": pairs,
            "prompt_token_ids": (
                json.loads(args.work_prompts.read_text()) if args.work_prompts else None
            ),
        }
    )
    (output / "run.json").write_text(json.dumps(provenance, indent=2) + "\n")
    cmd = [
        str(binary),
        "--model",
        "gpt2",
        "--hf",
        "openai-community/gpt2",
        "--pairs",
        ",".join(f"{n}:{k}" for n, k in pairs),
        "--fixed-length",
        "--num-threads",
        str(args.threads),
        "--work-token-seed",
        str(args.token_seed),
        "--work-ledger",
        str(ledger),
        "--bench",
        str(output / "timings.csv"),
    ]
    if args.work_prompts:
        cmd += ["--work-prompts", str(args.work_prompts.resolve())]
    started = time.monotonic()
    with (
        (output / "run.log").open("w") as log,
        tqdm(total=1, desc="Shared setup", unit="setup") as setup,
        tqdm(total=len(pairs), desc="Verified work ledgers", unit="proof") as progress,
    ):
        process = subprocess.Popen(
            cmd,
            cwd=deep_prove / "zkml",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if "stage=setup_done" in line:
                    setup.update(1)
                    # Freeze the setup duration before the proof loop runs.
                    setup.close()
                elif "POML_EVENT trial=" in line and "stage=done" in line:
                    progress.update(1)
                elif "stage=start N=" in line:
                    progress.set_postfix_str(line.strip().split("stage=start ")[1])
            if process.wait() != 0:
                raise RuntimeError(f"DeepProve failed; see {output / 'run.log'}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    rows = load_ledger(ledger)
    if len(rows) != len(pairs):
        raise ValueError(f"Expected {len(pairs)} trials, got {len(rows)}")
    if [(r["meta"]["n"], r["meta"]["k"]) for r in rows] != pairs:
        raise ValueError("Ledger pairs do not match the requested campaign")
    if (
        provenance["prompt_token_ids"] is not None
        and [r["meta"]["tokens"] for r in rows] != provenance["prompt_token_ids"]
    ):
        raise ValueError("Ledger prompts do not match the requested token replay")
    expected_backend = "cuda" if args.cuda else "cpu"
    if any(r["meta"]["backend"] != expected_backend for r in rows):
        raise ValueError("Binary backend does not match --cuda; rebuild without --skip-build")
    report = analyze(ledger, output, provenance, args.compare_schedule)
    print(
        json.dumps(
            {
                "trials": report["trials"],
                "all_exact": report["all_exact"],
                "max_proof_component_relative_error": report["max_proof_component_relative_error"],
                "within_tolerance": report["within_tolerance"],
                "elapsed_seconds": round(time.monotonic() - started),
                "schedule": str(output / "schedule.json"),
            },
            indent=2,
        )
    )


def main() -> None:
    """Parse the campaign or offline calculator command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep-prove", type=Path, default=ROOT / ".scratch/deep-prove")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/results/complexity-counts",
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--device", type=int)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--setup-seed", type=int, default=20260910)
    parser.add_argument("--token-seed", type=int, default=20260910)
    parser.add_argument(
        "--work-prompts",
        type=Path,
        help="JSON arrays of exact prompt token IDs, one array per --pairs entry",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--pairs", help="optional N:K,N:K,... list; must include N+K=64")
    parser.add_argument(
        "--compare-schedule",
        type=Path,
        help="validate against an existing schedule, including independent devices or token seeds",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="apply pinned source patches before building",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--analyze", type=Path)
    modes.add_argument("--calculate", metavar="N:K")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.device is not None and not args.cuda:
        parser.error("--device requires --cuda")
    run(args)


if __name__ == "__main__":
    main()
