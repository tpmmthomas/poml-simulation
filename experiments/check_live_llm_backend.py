#!/usr/bin/env python3
"""Bounded real-prover checks for restart, EOS, and the full configured context."""

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poml_sim.live_prover import LiveProver  # noqa: E402
from poml_sim.protocol_inputs import (  # noqa: E402
    gaussian_noise,
    decoding_uniforms,
    experimental_key,
    digest,
)


def copy_setup(source, destination):
    """Copy an immutable NN setup; EOS testing registers a different public decoder."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("prover.bin.zst", "verifier.bin", "contract.json"):
        shutil.copy2(source / name, destination / name)


def eos_request(source, setup, output):
    """Choose a supplied uniform inside EOS's actual probability interval.

    This controlled edge case is a proof-system test, never benchmark data or
    a replacement for the VRF-derived uniforms used by miners.
    """
    import numpy as np

    contract = json.loads((setup / "contract.json").read_text())
    contract["decoding"] = {"temperature": 1.0, "top_k": 0, "top_p": 1.0}
    (setup / "contract.json").write_text(json.dumps(contract))
    request = json.loads((source / "request.json").read_text())
    logits = np.fromfile(source / "logits.i64", dtype="<i8")[:50257].tolist()
    order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
    maximum = logits[order[0]]
    weights = [
        round(math.exp((logits[i] - maximum) * contract["logits_scale"]) * 2**48)
        for i in order
    ]
    position = order.index(50256)
    total, before, mass = sum(weights), sum(weights[:position]), weights[position]
    if mass <= 0:
        raise ValueError("source prompt assigns no representable EOS probability")
    uniform = ((2 * before + mass) * 2**52) // total
    if not before <= uniform * total // 2**53 < before + mass:
        raise ValueError("EOS interval too narrow for a 53-bit uniform")
    request.update(
        request_id="controlled-eos",
        max_output=4,
        uniforms=[uniform, 0, 0, 0],
        mode="prove",
        output_directory=str(output),
    )
    return request


def main():
    """Execute one bounded backend qualification case, preserving its artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["restart", "eos", "max-context"])
    parser.add_argument(
        "--binary",
        type=Path,
        default=ROOT / ".scratch/deep-prove/target/release/poml-prover",
    )
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--source-proof", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("use a fresh validation output directory")
    setup = out / "prover"
    copy_setup(args.setup, setup)
    results, handshakes = [], []
    if args.case in {"restart", "eos"} and args.source_proof is None:
        raise ValueError("--source-proof is required for restart/EOS validation")
    request = None
    if args.case == "eos":
        request = eos_request(args.source_proof, setup, out / "proof")
    for repeat in range(2 if args.case == "restart" else 1):
        with LiveProver(
            args.binary,
            setup,
            args.device,
            top_k=0 if args.case == "eos" else 50,
            top_p=1.0 if args.case == "eos" else 0.95,
        ) as prover:
            handshakes.append(prover.ready)
            if args.case == "restart":
                request = json.loads((args.source_proof / "request.json").read_text())
                request.update(
                    request_id=f"restart-{repeat}",
                    mode="profile" if repeat == 0 else "prove",
                    output_directory=str(out / f"restart-{repeat}"),
                )
            elif args.case == "max-context":
                from transformers import AutoTokenizer
                from experiments.build_real_llm_trace_bank import _load_wikitext
                from poml_sim.llm_benchmark import wikitext_prompts

                tokenizer = AutoTokenizer.from_pretrained(
                    "openai-community/gpt2", local_files_only=True
                )
                rows, _ = _load_wikitext(None)
                prompts = wikitext_prompts(
                    rows,
                    tokenizer.encode,
                    count=8,
                    lengths=(32,),
                    max_output=32,
                    seed=73,
                )
                for i, q in enumerate(prompts):
                    noise, transcript = gaussian_noise(
                        experimental_key(73, "max-context"),
                        digest(str(i).encode()),
                        32,
                        32,
                        0.05,
                        prover.ready["embedding_std"],
                    )
                    request = dict(
                        request_id=f"length-check-{i}",
                        prompt_tokens=q["prompt_tokens"],
                        noise=noise,
                        uniforms=decoding_uniforms(transcript, 32),
                        max_output=32,
                        mode="profile",
                        output_directory=str(out / f"length-check-{i}"),
                    )
                    profile = prover.run(request)
                    if profile["output_length"] == 32:
                        request.update(
                            request_id="max-context-proof",
                            mode="prove",
                            output_directory=str(out / "proof"),
                        )
                        break
                else:
                    raise RuntimeError(
                        "all qualification prompts ended early; no context-64 test case"
                    )
            results.append(prover.run(request))
    if args.case == "restart":
        if (
            handshakes[0]["setup_sha256"] != handshakes[1]["setup_sha256"]
            or results[0]["output_tokens"] != results[1]["output_tokens"]
        ):
            raise RuntimeError("setup or deterministic output changed after restart")
    elif args.case == "eos" and results[0]["output_tokens"] != [50256]:
        raise RuntimeError("controlled EOS case failed")
    elif args.case == "max-context" and results[0]["context_length"] != 64:
        raise RuntimeError("context-64 case failed")
    report = {
        "case": args.case,
        "purpose": "qualification only; excluded from paper measurements",
        "handshakes": handshakes,
        "results": results,
    }
    (out / "check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
