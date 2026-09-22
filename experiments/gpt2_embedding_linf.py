"""Measure GPT-2 activation distances on fixed token continuations.

Reuse a saved prompt manifest, retain every pair's L-infinity distances, and
export appendix tables without conflating embedding noise with token changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpt2_embedding_experiments import (  # noqa: E402
    deterministic_embedding_noise,
    model_weight_sha256,
)
from gpt2_experiments import _load_model  # noqa: E402

SCHEMA_VERSION = 1


def boundary_distances(values: Any, scale: float, clip: int) -> dict[str, Any]:
    """Measure adjacent challenge pairs in raw and quantized activation units."""
    import torch

    if not np.isfinite(scale) or scale <= 0 or not 1 <= clip <= 2**31 - 1:
        raise ValueError("scale must be finite and positive; clip must fit int32")
    if values.ndim < 2 or values.shape[0] % 2 or values.numel() == 0:
        raise ValueError("expected a nonempty even batch of paired tensors")
    # Float64 subtraction preserves the exact difference of stored FP32 values.
    flat = values.detach().to(torch.float64).flatten(1)
    if not torch.isfinite(flat).all():
        raise ValueError("activation contains non-finite values")
    rounded = torch.round(flat / scale)
    quantized = rounded.clamp(-clip, clip)
    return {
        "raw_linf": (flat[0::2] - flat[1::2]).abs().amax(dim=1),
        "quantized_linf_steps": (quantized[0::2] - quantized[1::2]).abs().amax(dim=1),
        "clipped_coordinates": (rounded.abs() > clip)
        .sum(dim=1)
        .reshape(-1, 2)
        .sum(dim=1),
        "coordinates": flat.shape[1],
    }


def clean_continuation(model: Any, prompt_ids: Sequence[int], length: int) -> list[int]:
    """Generate exactly length greedy reference tokens once per clean prompt."""
    import torch

    device = next(model.parameters()).device
    tokens = torch.tensor([list(prompt_ids)], device=device)
    result: list[int] = []
    cache = None
    with torch.inference_mode():
        for _ in range(length):
            output = model(input_ids=tokens, past_key_values=cache, use_cache=True)
            tokens = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            result.append(int(tokens.item()))
            cache = output.past_key_values
    return result


def measure_prompt(
    model: Any,
    prompt: dict[str, Any],
    continuation: Sequence[int],
    *,
    sigmas: Sequence[float],
    pairs: int,
    steps: Sequence[int],
    seed: int,
    scale: float,
    clip: int,
    noise_clip: float | None = 3.0,
    noise_quantum: float = 1e-6,
    detailed_layers: bool = False,
) -> list[dict[str, Any]]:
    """Compare prompt tensors and active token rows at selected decode steps.

    A causal teacher-forced pass supplies all requested prefix evaluations.
    Hooks record actual block outputs, separately from final normalization.
    """
    import torch

    if pairs < 1 or not steps or min(steps) < 0 or max(steps) > len(continuation):
        raise ValueError("pairs must be positive and steps must fit the continuation")
    if not sigmas or any(not np.isfinite(s) or s < 0 for s in sigmas):
        raise ValueError("sigmas must be finite and nonnegative")
    prompt_ids = prompt["token_ids"]
    if not prompt_ids:
        raise ValueError("prompt must contain tokens")
    prefix_length = len(prompt_ids)
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(continuation)], device=device)
    positions = torch.arange(ids.shape[1], device=device).unsqueeze(0)
    if ids.shape[1] > model.config.n_positions:
        raise ValueError("prompt and continuation exceed the model context")
    indices = [prefix_length + step - 1 for step in steps]
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        clean_embeddings = model.get_input_embeddings()(ids)
        for sigma in sigmas:
            perturbed = clean_embeddings.repeat(2 * pairs, 1, 1)
            for pair in range(pairs):
                base = f"{seed}|{prompt['prompt_id']}|sigma={sigma:.12g}|pair={pair}|common_token_stream"
                for side, suffix in enumerate(("a", "b")):
                    # The seeds and expansion match the original campaign.
                    if sigma:
                        noise = deterministic_embedding_noise(
                            base + "|embed-" + suffix,
                            prompt_ids,
                            (prefix_length, model.config.n_embd),
                            sigma,
                            clip=noise_clip,
                            quantum=noise_quantum,
                        )
                        perturbed[2 * pair + side, :prefix_length] += torch.as_tensor(
                            noise,
                            device=device,
                            dtype=perturbed.dtype,
                        )
            pending: list[dict[str, Any]] = []

            def record(boundary: str, values: Any) -> None:
                scopes = [("prompt", 0, values[:, :prefix_length])]
                scopes.extend(
                    ("active_token", step, values[:, index])
                    for step, index in zip(steps, indices)
                )
                if detailed_layers:
                    scopes.extend(
                        ("context", step, values[:, : index + 1])
                        for step, index in zip(steps, indices)
                    )
                for scope, step, selected in scopes:
                    pending.append(
                        {
                            "sigma_abs": float(sigma),
                            "boundary": boundary,
                            "scope": scope,
                            "step": step,
                            **boundary_distances(selected, scale, clip),
                        }
                    )

            def make_hook(name: str):
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    record(name, output[0] if isinstance(output, tuple) else output)

                return hook

            def qkv_hook(_module: Any, _inputs: Any, output: Any) -> None:
                for name, value in zip(
                    ("layer_0_q", "layer_0_k", "layer_0_v"), output.chunk(3, dim=-1)
                ):
                    record(name, value)

            handles = []
            try:
                # The dropout input is the actual sum computed by GPT-2 after
                # perturbing token embeddings and then adding position vectors.
                handles.append(
                    model.transformer.drop.register_forward_hook(make_hook("embedding"))
                )
                handles.append(
                    model.transformer.h[0].ln_1.register_forward_hook(
                        make_hook("post_first_norm")
                    )
                )
                handles.append(
                    model.transformer.h[0].attn.c_attn.register_forward_hook(qkv_hook)
                )
                for index, block in enumerate(model.transformer.h):
                    handles.append(
                        block.register_forward_hook(make_hook(f"block_{index}"))
                    )
                    if detailed_layers:
                        # Projected sublayer outputs are distinct from the
                        # residual additions recorded by the block hook.
                        for name, module in (
                            ("attention", block.attn),
                            ("feedforward", block.mlp),
                            ("norm_attention", block.ln_1),
                            ("norm_feedforward", block.ln_2),
                        ):
                            handles.append(
                                module.register_forward_hook(
                                    make_hook(f"{name}_{index}")
                                )
                            )
                handles.append(
                    model.transformer.ln_f.register_forward_hook(
                        make_hook("final_norm")
                    )
                )
                output = model.transformer(
                    inputs_embeds=perturbed, position_ids=positions, use_cache=False
                )
                logits = model.lm_head(output.last_hidden_state[:, indices])
                for index, step in enumerate(steps):
                    pending.append(
                        {
                            "sigma_abs": float(sigma),
                            "boundary": "logits",
                            "scope": "active_token",
                            "step": step,
                            **boundary_distances(logits[:, index], scale, clip),
                        }
                    )
            finally:
                for handle in handles:
                    handle.remove()
            for row in pending:
                records.append(
                    {
                        key: value.cpu().tolist()
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in row.items()
                    }
                )
    return records


def summarize_records(
    records: Sequence[dict[str, Any]], scale: float
) -> list[dict[str, Any]]:
    """Pool exact per-pair distances, retaining minima and empirical quantiles."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for record in records:
        for row in record["distances"]:
            key = (
                record["stratum"],
                row["sigma_abs"],
                row["scope"],
                row["step"],
                row["boundary"],
            )
            groups.setdefault(key, []).append(row)
    result = []
    for (stratum, sigma, scope, step, boundary), rows in sorted(groups.items()):
        raw = np.asarray([v for row in rows for v in row["raw_linf"]])
        quantized = np.asarray([v for row in rows for v in row["quantized_linf_steps"]])
        result.append(
            {
                "stratum": stratum,
                "sigma_abs": sigma,
                "scope": scope,
                "step": step,
                "boundary": boundary,
                "prompts": len(rows),
                "comparisons": len(raw),
                "coordinates_min": min(row["coordinates"] for row in rows),
                "coordinates_max": max(row["coordinates"] for row in rows),
                "raw_linf_min": float(raw.min()),
                "raw_linf_p05": float(np.quantile(raw, 0.05)),
                "raw_linf_median": float(np.median(raw)),
                "raw_linf_mean": float(raw.mean()),
                "raw_linf_max": float(raw.max()),
                "quantized_linf_steps_min": int(quantized.min()),
                "quantized_linf_steps_median": float(np.median(quantized)),
                "dequantized_linf_min": float(quantized.min() * scale),
                "clipped_coordinates": sum(
                    sum(row["clipped_coordinates"]) for row in rows
                ),
            }
        )
    return result


def appendix_rows(
    records: Sequence[dict[str, Any]], scale: float
) -> list[dict[str, Any]]:
    """Take minima across strata, block outputs, and measured decoding steps."""
    summary = summarize_records(records, scale)
    result = []
    for sigma in sorted({row["sigma_abs"] for row in summary}):
        selected = [row for row in summary if row["sigma_abs"] == sigma]
        scopes = {
            "prompt_embedding": [
                row
                for row in selected
                if row["scope"] == "prompt" and row["boundary"] == "embedding"
            ],
            "prompt_blocks": [
                row
                for row in selected
                if row["scope"] == "prompt" and row["boundary"].startswith("block_")
            ],
            "active_blocks": [
                row
                for row in selected
                if row["scope"] == "active_token"
                and row["boundary"].startswith("block_")
            ],
            "logits": [row for row in selected if row["boundary"] == "logits"],
        }
        row = {"sigma_abs": sigma}
        for name, values in scopes.items():
            row[name + "_raw_min"] = min(
                (value["raw_linf_min"] for value in values), default=None
            )
            row[name + "_quantized_steps_min"] = min(
                (value["quantized_linf_steps_min"] for value in values), default=None
            )
        result.append(row)
    return result


def write_tables(
    records: Sequence[dict[str, Any]], output_dir: Path, scale: float
) -> None:
    """Export all summaries and a compact minimum-distance appendix table."""
    summary = summarize_records(records, scale)
    with (output_dir / "linf_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# GPT-2 embedding perturbation: measured minimum L-infinity distances",
        "",
        "Minima over prompt/challenge pairs; raw FP32 activations differenced in FP64.",
        f"Quantization bin width: {scale:g}. See linf_summary.csv for every boundary and step.",
        "",
        "| Stratum | sigma | Scope | Step | Embedding | First norm | First K | First V | Min over blocks | Final norm | Logits |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    groups: dict[tuple, dict[str, float]] = {}
    for row in summary:
        key = (row["stratum"], row["sigma_abs"], row["scope"], row["step"])
        groups.setdefault(key, {})[row["boundary"]] = row["raw_linf_min"]
    for (stratum, sigma, scope, step), values in sorted(groups.items()):
        blocks = [value for key, value in values.items() if key.startswith("block_")]
        cells = [
            values.get(name)
            for name in ("embedding", "post_first_norm", "layer_0_k", "layer_0_v")
        ]
        cells += [min(blocks), values.get("final_norm"), values.get("logits")]
        lines.append(
            f"| {stratum} | {sigma:g} | {scope} | {step} | "
            + " | ".join("—" if value is None else f"{value:.6g}" for value in cells)
            + " |"
        )
    (output_dir / "linf_table.md").write_text("\n".join(lines) + "\n")
    compact = appendix_rows(records, scale)
    with (output_dir / "linf_appendix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compact[0]))
        writer.writeheader()
        writer.writerows(compact)
    tex = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$\sigma_{\mathrm{abs}}$ & Prompt emb. & Prompt blocks & Active blocks & Logits \\",
        r"\midrule",
    ]
    for row in compact:
        values = [
            row[name]
            for name in (
                "sigma_abs",
                "prompt_embedding_raw_min",
                "prompt_blocks_raw_min",
                "active_blocks_raw_min",
                "logits_raw_min",
            )
        ]
        tex.append(
            " & ".join("---" if value is None else f"{value:.6g}" for value in values)
            + r" \\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "linf_appendix.tex").write_text("\n".join(tex) + "\n")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load and validate the saved, tokenized prompt set without dataset access."""
    prompts = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if not prompts or len({p["prompt_id"] for p in prompts}) != len(prompts):
        raise ValueError("manifest must contain unique prompt IDs")
    for prompt in prompts:
        if not prompt["token_ids"] or any(
            type(token) is not int or token < 0 for token in prompt["token_ids"]
        ):
            raise ValueError("each prompt must contain nonnegative integer token IDs")
    return prompts


def main(argv: Sequence[str] | None = None) -> int:
    """Run or resume an auditable fixed-continuation distance campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "experiments/results/gpt2_embedding/separation/prompt_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/gpt2_embedding_linf"),
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sigmas", default="0,0.005,0.01,0.02,0.04")
    parser.add_argument("--steps", default="0,1,4,8,16,32")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantization-scale", type=float, default=1e-3)
    parser.add_argument("--quantization-clip", type=int, default=2**31 - 1)
    parser.add_argument(
        "--limit", type=int, help="pilot only: first N manifest prompts"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="regenerate tables from recorded distances without loading GPT-2",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir
    metadata_path = output_dir / "metadata.json"
    checkpoint_path = output_dir / "linf_checkpoint.jsonl"
    if args.report_only:
        metadata = json.loads(metadata_path.read_text())
        records = [
            json.loads(line) for line in checkpoint_path.read_text().splitlines()
        ]
        write_tables(records, output_dir, metadata["quantization_scale"])
        return 0
    sigmas = sorted(set(float(s) for s in args.sigmas.split(",")))
    steps = sorted(set(int(s) for s in args.steps.split(",")))
    if (
        args.pairs < 1
        or min(steps) < 0
        or any(not np.isfinite(s) or s < 0 for s in sigmas)
    ):
        parser.error("pairs must be positive, steps and finite sigmas nonnegative")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    prompts = load_manifest(args.manifest)
    if args.limit:
        prompts = prompts[: args.limit]
    model, _tokenizer, torch = _load_model(args.model, args.device, args.revision)
    torch.set_num_threads(4)
    # Fix arithmetic policy for reproducibility and eliminate dropout.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.eval()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "revision": getattr(model.config, "_commit_hash", None),
        "model_weight_sha256": model_weight_sha256(model),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "prompt_ids": [p["prompt_id"] for p in prompts],
        "pairs": args.pairs,
        "sigmas": sigmas,
        "steps": steps,
        "seed": args.seed,
        "embedding_width": model.config.n_embd,
        "layers": model.config.n_layer,
        "quantization_scale": args.quantization_scale,
        "quantization_clip": args.quantization_clip,
        "noise": "original VRF/SHA256 Box-Muller, clipped at 3 sigma, quantum 1e-6",
        "continuation": "clean greedy, fixed length, EOS treated as a token, shared by both challenges",
        "evaluation": "batched causal teacher forcing; full prompt and active token rows; no DeepProve integration",
        "device": args.device,
        "gpu": torch.cuda.get_device_name(args.device)
        if args.device.startswith("cuda")
        else None,
        "torch": torch.__version__,
        "dtype": str(next(model.parameters()).dtype),
        "transformers": __import__("transformers").__version__,
        "platform": platform.platform(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        parser.error("incompatible saved metadata; choose another output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    records = (
        [json.loads(line) for line in checkpoint_path.read_text().splitlines()]
        if checkpoint_path.exists()
        else []
    )
    completed = {r["prompt_id"] for r in records}
    from tqdm import tqdm

    for prompt in tqdm(prompts, desc="GPT-2 L_inf", unit="prompt"):
        if prompt["prompt_id"] in completed:
            continue
        continuation = clean_continuation(model, prompt["token_ids"], max(steps))
        distances = measure_prompt(
            model,
            prompt,
            continuation,
            sigmas=sigmas,
            pairs=args.pairs,
            steps=steps,
            seed=args.seed,
            scale=args.quantization_scale,
            clip=args.quantization_clip,
        )
        record = {**prompt, "continuation": continuation, "distances": distances}
        with checkpoint_path.open("a") as handle:
            handle.write(
                json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
            )
        records.append(record)
    write_tables(records, output_dir, args.quantization_scale)
    print(
        f"Completed {len(records)} prompts; tables: {output_dir / 'linf_table.md'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
