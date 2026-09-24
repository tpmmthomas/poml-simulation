"""Merge compatible useful-work timing shards and recompute one summary."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from poml_sim.work_efficiency import summarize_records


def main(argv=None):
    """Combine completed shard directories without creating new proofs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args(argv)
    if len(set(path.resolve() for path in args.shards)) != len(args.shards):
        parser.error("a shard directory may only be supplied once")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("choose a fresh output directory")
    records = []
    manifests = []
    for shard_number, directory in enumerate(args.shards):
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("status") != "complete":
            parser.error(f"incomplete shard: {directory}")
        rows = [
            json.loads(line) for line in (directory / "measurements.jsonl").read_text().splitlines()
        ]
        if len(rows) != manifest["queries"]:
            parser.error(f"measurement count does not match manifest: {directory}")
        if any(
            row.get(key) is not True
            for row in rows
            for key in ("verified", "fresh_inference", "fresh_proof")
        ):
            parser.error(f"unverified or non-fresh records: {directory}")
        if manifests:
            for key in ("model", "max_output", "binary_sha256", "scope", "useful", "auxiliary"):
                if manifest.get(key) != manifests[0].get(key):
                    parser.error(f"incompatible {key}: {directory}")
        for row in rows:
            row["shard"] = shard_number
        records.extend(rows)
        manifests.append(manifest)
    if not records:
        parser.error("no measurements found")
    summary = summarize_records(records, bootstrap=args.bootstrap, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "measurements.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    merged = {
        "schema": "poml-work-efficiency-merged-1",
        "status": "complete",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "shards": [str(path) for path in args.shards],
        "shard_count": len(args.shards),
        "measured_count": len(records),
        "queries": len(records),
        "bootstrap": {"resamples": args.bootstrap, "seed": args.seed},
        "substitutions": manifests[0].get("abstractions", []),
        "source_manifests": manifests,
        "by_shard": [
            summarize_records(
                [r for r in records if r["shard"] == i], bootstrap=args.bootstrap, seed=args.seed
            )
            for i in range(len(manifests))
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(merged, indent=2) + "\n")
    print(json.dumps(summary["ratios"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
