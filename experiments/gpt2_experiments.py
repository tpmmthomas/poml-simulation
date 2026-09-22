"""GPT-2 collision, sampler-concentration, and KV-cache experiments.

The module intentionally keeps model/dataset imports lazy.  The pure sampling
and VRF helpers are therefore testable on a small machine, while the CLI loads
Hugging Face only when an experiment actually needs it.
"""

from __future__ import annotations

import argparse
import copy
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import random
import struct
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Allow ``python experiments/gpt2_experiments.py`` from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from poml_sim.vrf import vrf_eval  # noqa: E402


DEFAULT_TEMPERATURES = (0.7, 1.0, 1.3, 1.5, 2.0)
VRF_DOMAIN = b"poml-gpt2-sampler-v1\x00"


@dataclass(frozen=True)
class Prompt:
    """A deduplicated, fixed-length tokenized benchmark prompt."""

    prompt_id: str
    source: str
    text: str
    token_ids: tuple[int, ...]


@dataclass
class Trace:
    """One deterministic GPT-2 rollout and its sampler diagnostics."""

    tokens: list[int]
    token_probabilities: list[float]
    top_token_probabilities: list[float]
    entropies: list[float]
    eos_probabilities: list[float]
    vrf_inputs: list[str]
    vrf_outputs: list[str]
    vrf_proofs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_seed_bytes(seed: int | bytes | str) -> bytes:
    if isinstance(seed, bytes):
        return seed
    if isinstance(seed, int):
        if seed < 0:
            raise ValueError("seed must be non-negative")
        return str(seed).encode("ascii")
    return str(seed).encode("utf-8")


def derive_vrf_private_key(seed: int | bytes | str) -> bytes:
    """Derive a reproducible 32-byte Ed25519 private key from an experiment seed."""

    return hashlib.sha256(b"poml-gpt2-vrf-key-v1\x00" + _as_seed_bytes(seed)).digest()


def derive_vrf_public_key(seed: int | bytes | str) -> bytes:
    """Return the Ed25519 verification key corresponding to an experiment seed."""

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    return Ed25519PrivateKey.from_private_bytes(derive_vrf_private_key(seed)).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def token_digest(token_ids: Sequence[int]) -> bytes:
    """Hash token IDs with an unambiguous fixed-width representation."""

    payload = b"".join(struct.pack(">I", int(token)) for token in token_ids)
    return hashlib.sha256(b"poml-gpt2-prompt-v1\x00" + payload).digest()


def vrf_uniforms(
    seed: int | bytes | str,
    prompt_tokens: Sequence[int],
    length: int,
) -> tuple[list[float], list[dict[str, str]]]:
    """Return per-step uniform values and the exact VRF transcript.

    A seed-derived key is used only to make experiments reproducible.  Each
    step has a distinct message containing the prompt digest and step number;
    the existing repository VRF supplies the pseudorandom output and proof.
    """

    if length < 0:
        raise ValueError("length must be non-negative")
    sk = derive_vrf_private_key(seed)
    prompt_hash = token_digest(prompt_tokens)
    uniforms: list[float] = []
    transcript: list[dict[str, str]] = []
    for step in range(1, length + 1):
        message = VRF_DOMAIN + prompt_hash + struct.pack(">I", step)
        output, proof = vrf_eval(sk, message)
        # Midpoint mapping avoids the (theoretical) exact endpoint 1.0.
        integer = int.from_bytes(output[:8], "big")
        uniforms.append((integer + 0.5) / float(1 << 64))
        transcript.append(
            {
                "input": message.hex(),
                "output": output.hex(),
                "proof": proof.hex(),
            }
        )
    return uniforms, transcript


def _filtered_probabilities(
    logits: Any,
    temperature: float,
    *,
    top_k: int | None = None,
    top_p: float = 1.0,
) -> Any:
    """Apply the protocol's temperature/top-k/top-p policy to logits."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    import torch

    values = logits.detach().to(dtype=torch.float64, device="cpu") / temperature
    if values.ndim != 1:
        raise ValueError("logits must be a one-dimensional vocabulary vector")
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        top_k = min(top_k, values.numel())
        threshold = torch.topk(values, top_k).values[-1]
        values = values.masked_fill(values < threshold, -torch.inf)
    probabilities = torch.softmax(values, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative - sorted_probs >= top_p
        # Always retain the highest-probability token.
        remove[0] = False
        probabilities[sorted_indices[remove]] = 0.0
        probabilities /= probabilities.sum()
    return probabilities


def sample_token_from_logits(
    logits: Any,
    uniform: float,
    temperature: float,
    *,
    top_k: int | None = None,
    top_p: float = 1.0,
) -> tuple[int, float, float, float]:
    """Inverse-CDF sample and return ``(token, token_prob, top_prob, entropy)``."""

    if not 0 <= uniform < 1:
        raise ValueError("uniform must be in [0, 1)")
    probabilities = _filtered_probabilities(
        logits, temperature, top_k=top_k, top_p=top_p
    )
    import torch

    cdf = torch.cumsum(probabilities, dim=-1)
    index = int(torch.searchsorted(cdf, torch.tensor(uniform, dtype=cdf.dtype)).item())
    index = min(index, probabilities.numel() - 1)
    probability = float(probabilities[index].item())
    top_probability = float(probabilities.max().item())
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * torch.log(nonzero)).sum().item())
    return index, probability, top_probability, entropy


def prefix_indicators(first: Sequence[int], second: Sequence[int], length: int) -> list[int | None]:
    """Return pair indicators for ``ell=0..length``; ``None`` means EOS-censored."""

    result: list[int | None] = [1]
    for ell in range(1, length + 1):
        if len(first) < ell or len(second) < ell:
            result.append(None)
        else:
            result.append(int(tuple(first[:ell]) == tuple(second[:ell])))
    return result


def _load_model(model_name: str, device: str, revision: str | None = None):
    """Load a causal LM and tokenizer, with an actionable optional dependency error."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError(
            "GPT-2 experiments require transformers and torch; install with "
            "pip install -e '.[dev,llm]'"
        ) from exc
    load_kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval().to(torch.device(device))
    return model, tokenizer, torch


def load_prompts(
    tokenizer: Any,
    *,
    dataset: str = "wikitext2",
    count: int = 500,
    prompt_tokens: int = 32,
    seed: int = 42,
    prompt_file: Path | None = None,
) -> list[Prompt]:
    """Load and deterministically sample fixed-token windows from a benchmark."""

    if count < 1 or prompt_tokens < 1:
        raise ValueError("count and prompt_tokens must be positive")
    if prompt_file is not None:
        texts = [(str(i), line.rstrip("\n")) for i, line in enumerate(prompt_file.read_text().splitlines())]
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - CLI-only path
            raise RuntimeError(
                "Dataset loading requires the datasets package; install with "
                "pip install -e '.[dev,llm]' or pass --prompt-file"
            ) from exc
        if dataset == "wikitext2":
            rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
            texts = [(str(i), row.get("text", "")) for i, row in enumerate(rows)]
        elif dataset == "lambada":
            last_error: Exception | None = None
            rows = None
            for name, config in (("lambada", None), ("lambada_openai", None), ("EleutherAI/lambada_openai", None)):
                try:
                    rows = load_dataset(name, config, split="test") if config else load_dataset(name, split="test")
                    break
                except Exception as exc:  # dataset naming differs across datasets releases
                    last_error = exc
            if rows is None:
                raise RuntimeError("Could not load a LAMBADA test split") from last_error
            texts = [(str(i), row.get("text") or row.get("context") or row.get("passage") or "") for i, row in enumerate(rows)]
        else:
            raise ValueError(f"unknown dataset: {dataset}")

    candidates: list[Prompt] = []
    seen: set[tuple[int, ...]] = set()
    for row_id, text in texts:
        if not text or not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
        if len(ids) < prompt_tokens:
            continue
        # One canonical window per row keeps article-level sampling simple and
        # prevents long articles from dominating the prompt distribution.
        window = tuple(int(x) for x in ids[:prompt_tokens])
        if window in seen:
            continue
        seen.add(window)
        candidates.append(Prompt(f"{dataset}:{row_id}", dataset, text, window))
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} usable unique prompts, requested {count}")
    rng = random.Random(seed)
    selected = rng.sample(candidates, count)
    return selected


def generate_trace(
    model: Any,
    tokenizer: Any,
    prompt_tokens: Sequence[int],
    seed: int | bytes | str,
    *,
    temperature: float = 1.0,
    length: int = 32,
    device: str = "cpu",
    top_k: int | None = None,
    top_p: float = 1.0,
    stop_eos: bool = True,
    stop_token_ids: Sequence[int] = (),
) -> Trace:
    """Generate one cached autoregressive trace under the deterministic sampler."""

    if length < 0:
        raise ValueError("length must be non-negative")
    import torch

    max_positions = getattr(getattr(model, "config", None), "n_positions", None) or getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if max_positions is not None and len(prompt_tokens) + length > int(max_positions):
        raise ValueError(
            f"prompt ({len(prompt_tokens)}) + generation ({length}) exceeds model context ({max_positions})"
        )

    uniforms, transcript = vrf_uniforms(seed, prompt_tokens, length)
    input_ids = torch.tensor([list(prompt_tokens)], dtype=torch.long, device=device)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    stop_tokens = frozenset(stop_token_ids)
    tokens: list[int] = []
    token_probs: list[float] = []
    top_probs: list[float] = []
    entropies: list[float] = []
    eos_probs: list[float] = []
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True)
        for step in range(length):
            logits = output.logits[0, -1, :]
            probabilities = _filtered_probabilities(logits, temperature, top_k=top_k, top_p=top_p)
            eos_probs.append(float(probabilities[eos_id].item()) if eos_id is not None else 0.0)
            token, token_prob, top_prob, entropy = sample_token_from_logits(
                logits, uniforms[step], temperature, top_k=top_k, top_p=top_p
            )
            tokens.append(token)
            token_probs.append(token_prob)
            top_probs.append(top_prob)
            entropies.append(entropy)
            if token in stop_tokens or (stop_eos and eos_id is not None and token == eos_id):
                break
            next_token = torch.tensor([[token]], dtype=torch.long, device=device)
            output = model(input_ids=next_token, past_key_values=output.past_key_values, use_cache=True)
    return Trace(
        tokens=tokens,
        token_probabilities=token_probs,
        top_token_probabilities=top_probs,
        entropies=entropies,
        eos_probabilities=eos_probs,
        vrf_inputs=[entry["input"] for entry in transcript[: len(tokens)]],
        vrf_outputs=[entry["output"] for entry in transcript[: len(tokens)]],
        vrf_proofs=[entry["proof"] for entry in transcript[: len(tokens)]],
    )


def _bootstrap_prompt_curves(curves: np.ndarray, replicates: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Prompt-level percentile bootstrap for a matrix of per-prompt curves."""

    if curves.ndim != 2 or curves.shape[0] == 0:
        raise ValueError("curves must be a non-empty 2D array")
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, curves.shape[1]), dtype=float)
    for i in range(replicates):
        draws[i] = np.nanmean(curves[rng.integers(0, curves.shape[0], curves.shape[0])], axis=0)
    return np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_fingerprint(kind: str, **config: Any) -> str:
    """Return a stable fingerprint so a resume cannot mix incompatible runs."""

    payload = json.dumps({"kind": kind, **config}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _open_checkpoint(
    output_dir: Path | None,
    *,
    filename: str,
    fingerprint: str,
    resume: bool,
) -> tuple[Path | None, dict[str, dict[str, Any]]]:
    """Load an append-only checkpoint and validate its experiment identity."""

    if output_dir is None:
        return None, {}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    meta_path = path.with_suffix(".meta.json")
    if not resume:
        path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
    elif meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata.get("fingerprint") != fingerprint:
            raise ValueError(
                f"checkpoint {path} belongs to different settings; use a new output directory or --no-resume"
            )
    else:
        meta_path.write_text(json.dumps({"fingerprint": fingerprint}, indent=2) + "\n")
    records: dict[str, dict[str, Any]] = {}
    if resume and path.exists():
        with path.open() as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A killed process may leave a partial final line.
                    continue
                if "key" in record and "result" in record:
                    records[str(record["key"])] = record["result"]
    return path, records


def _append_checkpoint(path: Path | None, key: str, result: dict[str, Any]) -> None:
    if path is None:
        return
    with path.open("a") as handle:
        handle.write(json.dumps({"key": key, "result": result}, separators=(",", ":")) + "\n")
        handle.flush()


def _plot_collision(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot the early collision window alongside expected shared-prefix length."""

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], float(row["temperature"])), []).append(row)
    max_prefix_length = max(int(row["prefix_length"]) for row in rows)
    # The full 32-token tail remains in the CSV. The figure focuses on the
    # first eight positions because nearly every pair has already diverged.
    display_length = min(8, max_prefix_length)
    figure, (curve_axis, expected_axis) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.8),
        gridspec_kw={"width_ratios": (2.25, 1)},
    )
    expected_lengths: list[tuple[float, float, Any]] = []
    max_probability = 0.0
    for color_index, ((dataset, temperature), group) in enumerate(sorted(groups.items())):
        group.sort(key=lambda row: int(row["prefix_length"]))
        displayed = [row for row in group if 1 <= int(row["prefix_length"]) <= display_length]
        x = [int(row["prefix_length"]) for row in displayed]
        y = [float(row["collision_probability"]) for row in displayed]
        color = f"C{color_index}"
        curve_axis.plot(x, y, marker="o", linewidth=2, color=color, label=f"T={temperature:g}")
        max_probability = max(max_probability, *y)
        expected_lengths.append(
            (
                temperature,
                sum(float(row["collision_probability"]) for row in group if int(row["prefix_length"]) > 0),
                color,
            )
        )
    curve_axis.set_xlim(0.8, display_length + 0.2)
    curve_axis.set_xticks(range(1, display_length + 1))
    curve_axis.set_ylim(0, max(max_probability * 1.15, 0.01))
    curve_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    curve_axis.set_xlabel("Generated prefix length ℓ")
    curve_axis.set_ylabel("Two traces share prefix Cℓ")
    curve_axis.set_title("Early prefix-collision curve")
    curve_axis.grid(alpha=0.25)
    curve_axis.legend(title="Temperature", fontsize=8)

    temperatures = [temperature for temperature, _, _ in expected_lengths]
    values = [value for _, value, _ in expected_lengths]
    colors = [color for _, _, color in expected_lengths]
    bars = expected_axis.barh([f"T={temperature:g}" for temperature in temperatures], values, color=colors)
    expected_axis.invert_yaxis()
    expected_axis.set_xlim(0, max(max(values, default=0.0) * 1.2, 0.01))
    expected_axis.set_xlabel("Expected shared\nprefix (tokens)")
    expected_axis.set_title("E[shared prefix]")
    expected_axis.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        expected_axis.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.3f}", va="center", fontsize=9)

    final_rows = [row for row in rows if int(row["prefix_length"]) == max_prefix_length]
    tail_note = (
        f"No observed collisions at ℓ={max_prefix_length}."
        if final_rows and all(float(row["collision_probability"]) == 0 for row in final_rows)
        else f"Full curves through ℓ={max_prefix_length} are retained in the CSV."
    )
    figure.suptitle("GPT-2 prefix collisions (500 prompts × 4 pairs per temperature)", y=0.99)
    figure.text(
        0.5,
        0.01,
        f"The left panel zooms to ℓ≤{display_length}; {tail_note} C₀=100% by definition.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(path)
    plt.close(figure)


def _plot_divergence(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot the empirical CDF of the first divergent generated position."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(float(row["temperature"]), []).append(row)
    for temperature, group in sorted(grouped.items()):
        counts: dict[int, int] = {}
        for row in group:
            position = int(row["first_divergence"])
            counts[position] = counts.get(position, 0) + int(row["count"])
        total = sum(counts.values())
        if not total:
            continue
        running = 0
        xs, ys = [], []
        for position in sorted(counts):
            running += counts[position]
            xs.append(position)
            ys.append(running / total)
        plt.step(xs, ys, where="post", label=f"T={temperature:g}")
    plt.xlabel("First divergent generated position")
    plt.ylabel("Empirical CDF")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_cache_timings(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot fresh versus cached replay latency for the measured cache cuts."""

    import matplotlib.pyplot as plt

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        "prompt prefill" if int(row["output_prefix_tokens"]) == 0
        else f"output prefix {int(row['output_prefix_tokens'])}"
        for row in rows
    ]
    x = np.arange(len(rows))
    fresh = [float(row["fresh_seconds"]) for row in rows]
    cached = [float(row["cached_seconds"]) for row in rows]
    width = 0.38
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, fresh, width, label="fresh")
    plt.bar(x + width / 2, cached, width, label="cached")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Median seconds")
    plt.title("GPT-2 cache replay timing")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_cache_workload(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot LRU exact-hit and compatible-prefix rates by workload family."""

    import matplotlib.pyplot as plt

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    workloads = sorted({str(row["workload"]) for row in rows})
    capacities = sorted({int(row["capacity"]) for row in rows})
    capacity_x = {capacity: index for index, capacity in enumerate(capacities)}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for workload in workloads:
        selected = [row for row in rows if str(row["workload"]) == workload]
        selected.sort(key=lambda row: int(row["capacity"]))
        axes[0].plot(
            [capacity_x[int(row["capacity"])] for row in selected],
            [float(row["exact_hit_rate"]) for row in selected],
            marker="o",
            label=workload,
        )
        axes[1].plot(
            [capacity_x[int(row["capacity"])] for row in selected],
            [float(row["mean_compatible_prefix_tokens"]) for row in selected],
            marker="o",
            label=workload,
        )
    for axis in axes:
        axis.set_xticks(range(len(capacities)), [str(capacity) for capacity in capacities])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_title("Exact LRU hit rate")
    axes[0].set_xlabel("Cache capacity")
    axes[0].set_ylabel("Hit rate")
    axes[1].set_title("Compatible prompt-prefix length")
    axes[1].set_xlabel("Cache capacity")
    axes[1].set_ylabel("Mean tokens")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _compute_collision_record(
    model: Any,
    tokenizer: Any,
    prompt: Prompt,
    *,
    pairs: int,
    temperature: float,
    length: int,
    seed: int,
    device: str,
    record_transcripts: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute one prompt/temperature unit; safe to run in a GPU worker thread."""

    counts = np.zeros(length + 1, dtype=np.int64)
    comparisons = np.zeros(length + 1, dtype=np.int64)
    divergence_counts: dict[int, int] = {}
    sequence_probabilities: list[float] = []
    transcripts: list[dict[str, Any]] = []
    for pair_index in range(pairs):
        base = f"{seed}|{prompt.prompt_id}|{temperature:.12g}|{pair_index}"
        first = generate_trace(model, tokenizer, prompt.token_ids, base + "|a", temperature=temperature, length=length, device=device)
        second = generate_trace(model, tokenizer, prompt.token_ids, base + "|b", temperature=temperature, length=length, device=device)
        for trace in (first, second):
            sequence_probabilities.append(math.exp(sum(math.log(max(probability, 1e-300)) for probability in trace.token_probabilities)))
        if record_transcripts:
            for side, trace in (("a", first), ("b", second)):
                transcripts.append(
                    {
                        "dataset": prompt.source,
                        "prompt_id": prompt.prompt_id,
                        "temperature": temperature,
                        "pair_index": pair_index,
                        "side": side,
                        "seed": base + "|" + side,
                        "verification_key": derive_vrf_public_key(base + "|" + side).hex(),
                        "steps": [
                            {"input": item[0], "output": item[1], "proof": item[2]}
                            for item in zip(trace.vrf_inputs, trace.vrf_outputs, trace.vrf_proofs)
                        ],
                    }
                )
        indicators = prefix_indicators(first.tokens, second.tokens, length)
        for ell, indicator in enumerate(indicators):
            if indicator is not None:
                counts[ell] += indicator
                comparisons[ell] += 1
        min_len = min(len(first.tokens), len(second.tokens))
        mismatch = next((i + 1 for i in range(min_len) if first.tokens[i] != second.tokens[i]), None)
        divergence = mismatch if mismatch is not None else (min_len + 1 if len(first.tokens) != len(second.tokens) else None)
        if divergence is not None:
            divergence_counts[divergence] = divergence_counts.get(divergence, 0) + 1
    return (
        {
            "dataset": prompt.source,
            "prompt_id": prompt.prompt_id,
            "temperature": temperature,
            "counts": counts.tolist(),
            "comparisons": comparisons.tolist(),
            "divergence_counts": {str(key): value for key, value in divergence_counts.items()},
            "sequence_probabilities": sequence_probabilities,
        },
        transcripts,
    )


def _compute_concentration_rows(
    model: Any,
    tokenizer: Any,
    prompt: Prompt,
    *,
    temperature: float,
    length: int,
    samples_per_prompt: int,
    seed: int,
    device: str,
) -> list[dict[str, Any]]:
    """Compute concentration observations for one prompt/temperature unit."""

    traces = [
        generate_trace(
            model,
            tokenizer,
            prompt.token_ids,
            f"{seed}|{prompt.prompt_id}|{temperature:.12g}|{sample_index}",
            temperature=temperature,
            length=length,
            device=device,
        )
        for sample_index in range(samples_per_prompt)
    ]
    rows: list[dict[str, Any]] = []
    for position in range(length):
        for sample_index, trace in enumerate(traces):
            if position >= len(trace.tokens):
                continue
            rows.append(
                {
                    "dataset": prompt.source,
                    "prompt_id": prompt.prompt_id,
                    "temperature": temperature,
                    "sample": sample_index,
                    "position": position + 1,
                    "token": trace.tokens[position],
                    "token_probability": trace.token_probabilities[position],
                    "top_token_probability": trace.top_token_probabilities[position],
                    "entropy": trace.entropies[position],
                    "eos_probability": trace.eos_probabilities[position],
                }
            )
    return rows


def run_collision_experiment(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Prompt],
    *,
    pairs: int = 4,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    length: int = 32,
    seed: int = 42,
    device: str = "cpu",
    bootstrap_replicates: int = 1000,
    output_dir: Path | None = None,
    record_transcripts: bool = False,
    resume: bool = True,
    progress_every: int = 10,
    worker_models: Sequence[tuple[Any, Any, str]] | None = None,
) -> dict[str, Any]:
    """Estimate prefix collision curves and first-divergence statistics."""

    if pairs < 1:
        raise ValueError("pairs must be positive")
    if not prompts:
        raise ValueError("prompts must be non-empty")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    fingerprint = _checkpoint_fingerprint(
        "collision",
        prompt_ids=[prompt.prompt_id for prompt in prompts],
        pairs=pairs,
        temperatures=[float(value) for value in temperatures],
        length=length,
        seed=seed,
        device=device,
        record_transcripts=record_transcripts,
    )
    checkpoint_path, checkpoint_records = _open_checkpoint(
        output_dir,
        filename="collision_checkpoint.jsonl",
        fingerprint=fingerprint,
        resume=resume,
    )
    transcript_handle = None
    transcript_keys: set[tuple[str, float, int, str]] = set()
    if output_dir is not None and record_transcripts:
        transcript_path = output_dir / "vrf_transcripts.jsonl"
        if not resume:
            transcript_path.unlink(missing_ok=True)
        elif transcript_path.exists():
            with transcript_path.open() as existing:
                for line in existing:
                    try:
                        prior = json.loads(line)
                        transcript_keys.add((str(prior["prompt_id"]), float(prior["temperature"]), int(prior["pair_index"]), str(prior["side"])))
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        continue
        transcript_handle = transcript_path.open("a")
    started_at = time.perf_counter()
    workers = list(worker_models or [(model, tokenizer, device)])
    if not workers:
        raise ValueError("worker_models must be non-empty")
    executors = [ThreadPoolExecutor(max_workers=1) for _ in workers] if len(workers) > 1 else []
    summary_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    divergence_rows: list[dict[str, Any]] = []
    transcript_rows: list[dict[str, Any]] = []
    lp_rows: list[dict[str, Any]] = []
    for temperature in temperatures:
        print(f"[collision] temperature={temperature:g}: starting ({len(prompts)} prompts)", flush=True)
        prompt_curves: list[list[float]] = []
        pooled_counts = np.zeros(length + 1, dtype=np.int64)
        pooled_comparisons = np.zeros(length + 1, dtype=np.int64)
        pending = {}
        if executors:
            for prompt_index, pending_prompt in enumerate(prompts):
                pending_key = f"{temperature:.12g}|{pending_prompt.prompt_id}"
                if pending_key in checkpoint_records:
                    continue
                worker_model, worker_tokenizer, worker_device = workers[prompt_index % len(workers)]
                pending[pending_key] = executors[prompt_index % len(executors)].submit(
                    _compute_collision_record,
                    worker_model,
                    worker_tokenizer,
                    pending_prompt,
                    pairs=pairs,
                    temperature=temperature,
                    length=length,
                    seed=seed,
                    device=worker_device,
                    record_transcripts=record_transcripts,
                )
        for prompt_index, prompt in enumerate(prompts, start=1):
            key = f"{temperature:.12g}|{prompt.prompt_id}"
            record = checkpoint_records.get(key)
            was_resumed = record is not None
            if record is None:
                if key in pending:
                    record, generated_transcripts = pending[key].result()
                else:
                    worker_model, worker_tokenizer, worker_device = workers[0]
                    record, generated_transcripts = _compute_collision_record(
                        worker_model,
                        worker_tokenizer,
                        prompt,
                        pairs=pairs,
                        temperature=temperature,
                        length=length,
                        seed=seed,
                        device=worker_device,
                        record_transcripts=record_transcripts,
                    )
                for transcript in generated_transcripts:
                    transcript_key = (prompt.prompt_id, float(temperature), int(transcript["pair_index"]), str(transcript["side"]))
                    if transcript_key in transcript_keys:
                        continue
                    transcript_keys.add(transcript_key)
                    transcript_rows.append(transcript)
                    if transcript_handle is not None:
                        transcript_handle.write(json.dumps(transcript, separators=(",", ":")) + "\n")
                checkpoint_records[key] = record
                _append_checkpoint(checkpoint_path, key, record)
                if transcript_handle is not None:
                    transcript_handle.flush()
            counts = np.asarray(record["counts"], dtype=np.int64)
            comparisons = np.asarray(record["comparisons"], dtype=np.int64)
            divergence_counts = {int(key): int(value) for key, value in record["divergence_counts"].items()}
            sequence_probabilities = [float(value) for value in record["sequence_probabilities"]]
            curve = [float(counts[ell] / comparisons[ell]) if comparisons[ell] else np.nan for ell in range(length + 1)]
            prompt_curves.append(curve)
            for ell in range(length + 1):
                pooled_counts[ell] += counts[ell]
                pooled_comparisons[ell] += comparisons[ell]
            prompt_rows.extend(
                {
                    "dataset": prompt.source,
                    "prompt_id": prompt.prompt_id,
                    "temperature": temperature,
                    "prefix_length": ell,
                    "collisions": int(counts[ell]),
                    "comparisons": int(comparisons[ell]),
                    "collision_probability": curve[ell],
                }
                for ell in range(length + 1)
            )
            divergence_rows.extend(
                {"dataset": prompt.source, "prompt_id": prompt.prompt_id, "temperature": temperature, "first_divergence": ell, "count": count}
                for ell, count in sorted(divergence_counts.items())
            )
            lp_rows.append(
                {
                    "dataset": prompt.source,
                    "prompt_id": prompt.prompt_id,
                    "temperature": temperature,
                    "sequence_probability_mean": float(np.mean(sequence_probabilities)),
                    "sequence_probability_median": float(np.median(sequence_probabilities)),
                }
            )
            state = "resumed" if was_resumed else "completed"
            if prompt_index == 1 or prompt_index % progress_every == 0 or prompt_index == len(prompts):
                elapsed = time.perf_counter() - started_at
                rate = prompt_index / elapsed if elapsed else 0.0
                eta = (len(prompts) - prompt_index) / rate if rate else float("inf")
                eta_text = f"{eta:.0f}s" if math.isfinite(eta) else "unknown"
                print(f"[collision] temperature={temperature:g} prompt={prompt_index}/{len(prompts)} ({state}, ETA {eta_text})", flush=True)
        print(f"[collision] temperature={temperature:g}: complete", flush=True)
        curve_array = np.asarray(prompt_curves, dtype=float)
        ci_low, ci_high = _bootstrap_prompt_curves(curve_array, bootstrap_replicates, seed + int(temperature * 1000))
        for ell in range(length + 1):
            comparisons = int(pooled_comparisons[ell])
            collisions = int(pooled_counts[ell])
            probability = collisions / comparisons if comparisons else float("nan")
            # Rule-of-three upper bound when the tail has no observed collision.
            upper = 3.0 / comparisons if comparisons and collisions == 0 else None
            summary_rows.append(
                {
                    "dataset": prompts[0].source if prompts else "unknown",
                    "temperature": temperature,
                    "prefix_length": ell,
                    "collisions": collisions,
                    "comparisons": comparisons,
                    "collision_probability": probability,
                    "prompt_median": float(np.nanmedian(curve_array[:, ell])),
                    "prompt_p10": float(np.nanpercentile(curve_array[:, ell], 10)),
                    "prompt_p90": float(np.nanpercentile(curve_array[:, ell], 90)),
                    "prompt_fraction_with_collision": float(np.mean(curve_array[:, ell] > 0)),
                    "bootstrap_low": float(ci_low[ell]),
                    "bootstrap_high": float(ci_high[ell]),
                    "zero_collision_upper_95": upper,
                }
            )
    for executor in executors:
        executor.shutdown(wait=True)
    if transcript_handle is not None:
        transcript_handle.close()
    aggregate_rows = []
    for temperature in temperatures:
        rows_for_temperature = [row for row in summary_rows if row["temperature"] == temperature]
        aggregate_rows.append(
            {
                "dataset": prompts[0].source if prompts else "unknown",
                "temperature": temperature,
                "expected_common_prefix_length": float(
                    sum(
                        row["collision_probability"]
                        for row in rows_for_temperature
                        if row["prefix_length"] > 0 and math.isfinite(row["collision_probability"])
                    )
                ),
                "lp_sequence_probability_mean": float(np.mean([row["sequence_probability_mean"] for row in lp_rows if row["temperature"] == temperature])),
            }
        )
    result = {"summary": summary_rows, "prompt_curves": prompt_rows, "first_divergence": divergence_rows, "aggregates": aggregate_rows, "lp_estimates": lp_rows, "transcripts": transcript_rows}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(output_dir / "collision_curves.csv", summary_rows)
        _write_csv(output_dir / "prompt_curves.csv", prompt_rows)
        _write_csv(output_dir / "first_divergence.csv", divergence_rows)
        _write_csv(output_dir / "collision_aggregates.csv", aggregate_rows)
        _write_csv(output_dir / "lp_estimates.csv", lp_rows)
        _plot_collision(summary_rows, output_dir / "collision_curves.png")
        _plot_divergence(divergence_rows, output_dir / "first_divergence_cdf.png")
    return result


def run_concentration_experiment(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Prompt],
    *,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    length: int = 32,
    samples_per_prompt: int = 1,
    seed: int = 42,
    device: str = "cpu",
    output_dir: Path | None = None,
    resume: bool = True,
    progress_every: int = 10,
    worker_models: Sequence[tuple[Any, Any, str]] | None = None,
) -> list[dict[str, Any]]:
    """Record top-token probability, entropy, and EOS probability by position."""

    if not prompts:
        raise ValueError("prompts must be non-empty")
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be positive")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    fingerprint = _checkpoint_fingerprint(
        "concentration",
        prompt_ids=[prompt.prompt_id for prompt in prompts],
        temperatures=[float(value) for value in temperatures],
        length=length,
        samples_per_prompt=samples_per_prompt,
        seed=seed,
        device=device,
    )
    checkpoint_path, checkpoint_records = _open_checkpoint(
        output_dir,
        filename="concentration_checkpoint.jsonl",
        fingerprint=fingerprint,
        resume=resume,
    )
    started_at = time.perf_counter()
    workers = list(worker_models or [(model, tokenizer, device)])
    if not workers:
        raise ValueError("worker_models must be non-empty")
    executors = [ThreadPoolExecutor(max_workers=1) for _ in workers] if len(workers) > 1 else []
    rows: list[dict[str, Any]] = []
    for temperature in temperatures:
        print(f"[concentration] temperature={temperature:g}: starting ({len(prompts)} prompts)", flush=True)
        pending = {}
        if executors:
            for prompt_index, pending_prompt in enumerate(prompts):
                pending_key = f"{temperature:.12g}|{pending_prompt.prompt_id}"
                if pending_key in checkpoint_records:
                    continue
                worker_model, worker_tokenizer, worker_device = workers[prompt_index % len(workers)]
                pending[pending_key] = executors[prompt_index % len(executors)].submit(
                    _compute_concentration_rows,
                    worker_model,
                    worker_tokenizer,
                    pending_prompt,
                    temperature=temperature,
                    length=length,
                    samples_per_prompt=samples_per_prompt,
                    seed=seed,
                    device=worker_device,
                )
        for prompt_index, prompt in enumerate(prompts, start=1):
            key = f"{temperature:.12g}|{prompt.prompt_id}"
            record = checkpoint_records.get(key)
            was_resumed = record is not None
            if record is None:
                if key in pending:
                    prompt_rows = pending[key].result()
                else:
                    worker_model, worker_tokenizer, worker_device = workers[0]
                    prompt_rows = _compute_concentration_rows(
                        worker_model,
                        worker_tokenizer,
                        prompt,
                        temperature=temperature,
                        length=length,
                        samples_per_prompt=samples_per_prompt,
                        seed=seed,
                        device=worker_device,
                    )
                record = {"rows": prompt_rows}
                checkpoint_records[key] = record
                _append_checkpoint(checkpoint_path, key, record)
            prompt_rows = list(record["rows"])
            rows.extend(prompt_rows)
            if prompt_index == 1 or prompt_index % progress_every == 0 or prompt_index == len(prompts):
                elapsed = time.perf_counter() - started_at
                rate = prompt_index / elapsed if elapsed else 0.0
                eta = (len(prompts) - prompt_index) / rate if rate else float("inf")
                eta_text = f"{eta:.0f}s" if math.isfinite(eta) else "unknown"
                state = "resumed" if was_resumed else "completed"
                print(f"[concentration] temperature={temperature:g} prompt={prompt_index}/{len(prompts)} ({state}, ETA {eta_text})", flush=True)
        print(f"[concentration] temperature={temperature:g}: complete", flush=True)
    for executor in executors:
        executor.shutdown(wait=True)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(output_dir / "token_concentration.csv", rows)
        summary_rows: list[dict[str, Any]] = []
        for temperature in temperatures:
            for position in sorted({row["position"] for row in rows if row["temperature"] == temperature}):
                selected = [row for row in rows if row["temperature"] == temperature and row["position"] == position]
                summary_rows.append(
                    {
                        "dataset": prompts[0].source if prompts else "unknown",
                        "temperature": temperature,
                        "position": position,
                        "top_token_p10": float(np.percentile([row["top_token_probability"] for row in selected], 10)),
                        "top_token_median": float(np.median([row["top_token_probability"] for row in selected])),
                        "top_token_p90": float(np.percentile([row["top_token_probability"] for row in selected], 90)),
                        "entropy_median": float(np.median([row["entropy"] for row in selected])),
                        "eos_probability_median": float(np.median([row["eos_probability"] for row in selected])),
                    }
                )
        _write_csv(output_dir / "concentration_summary.csv", summary_rows)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        for temperature in temperatures:
            selected = [row for row in rows if row["temperature"] == temperature]
            positions = sorted({row["position"] for row in selected})
            medians = [float(np.median([row["top_token_probability"] for row in selected if row["position"] == p])) for p in positions]
            plt.plot(positions, medians, marker="o", label=f"T={temperature:g}")
        plt.xlabel("Generated position")
        plt.ylabel("Median top-token probability")
        plt.ylim(0, 1)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "top_token_probability.png")
        plt.close()
    return rows


def _sync(device: str, torch: Any) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _decode_tokens(model: Any, tokens: Sequence[int], past: Any, device: str, torch: Any) -> Any:
    """Replay known tokens through a KV cache, returning the final cache."""

    output = past
    for token in tokens:
        input_ids = torch.tensor([[int(token)]], dtype=torch.long, device=device)
        output = model(input_ids=input_ids, past_key_values=output.past_key_values, use_cache=True)
    return output


def _time_replay(model: Any, prompt: Sequence[int], continuation: Sequence[int], device: str, torch: Any) -> float:
    start = time.perf_counter()
    output = model(input_ids=torch.tensor([list(prompt)], dtype=torch.long, device=device), use_cache=True)
    _decode_tokens(model, continuation, output, device, torch)
    _sync(device, torch)
    return time.perf_counter() - start


def _cache_bytes(past: Any) -> int:
    values = getattr(past, "past_key_values", past)
    # Transformers >=4.45 may expose a DynamicCache rather than a tuple.
    if hasattr(values, "key_cache") and hasattr(values, "value_cache"):
        values = (values.key_cache, values.value_cache)
    elif hasattr(values, "layers"):
        # Transformers 5.x stores per-layer key/value tensors on DynamicLayer
        # objects rather than exposing the older key_cache/value_cache lists.
        values = [
            (getattr(layer, "keys", None), getattr(layer, "values", None))
            for layer in values.layers
        ]

    def tensor_bytes(value: Any) -> int:
        # DynamicCache can contain None for an uninitialized layer (and some
        # model implementations use nested lists), so account only for actual
        # tensor-like objects instead of assuming every slot is populated.
        if value is None:
            return 0
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            return int(value.numel()) * int(value.element_size())
        if isinstance(value, (list, tuple)):
            return sum(tensor_bytes(item) for item in value)
        return 0

    return tensor_bytes(values)


def run_cache_experiment(
    model: Any,
    tokenizer: Any,
    prompt: Prompt,
    *,
    length: int = 32,
    output_prefix_lengths: Sequence[int] = (0, 8, 16, 32),
    repeats: int = 3,
    seed: int = 42,
    device: str = "cpu",
    temperature: float = 1.0,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Measure fresh, prompt-prefill-cache, and exact output-prefix replay time."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    import torch

    print(f"[cache] prompt={prompt.prompt_id}: measuring fresh and cached paths", flush=True)
    baseline = generate_trace(model, tokenizer, prompt.token_ids, f"{seed}|cache", temperature=temperature, length=length, device=device)
    # Warm up kernels before collecting timings.
    _time_replay(model, prompt.token_ids, baseline.tokens[: min(2, len(baseline.tokens))], device, torch)
    fresh = float(np.median([_time_replay(model, prompt.token_ids, baseline.tokens, device, torch) for _ in range(repeats)]))
    with torch.inference_mode():
        prompt_output = model(input_ids=torch.tensor([list(prompt.token_ids)], dtype=torch.long, device=device), use_cache=True)
        prompt_cache_bytes = _cache_bytes(prompt_output)
        prompt_cache_times = []
        for _ in range(repeats):
            # DynamicCache objects are mutable in recent Transformers releases;
            # clone outside the timed region so each repeat starts at the same
            # prompt boundary.
            cached_prompt = copy.deepcopy(prompt_output)
            _sync(device, torch)
            start = time.perf_counter()
            _decode_tokens(model, baseline.tokens, cached_prompt, device, torch)
            _sync(device, torch)
            prompt_cache_times.append(time.perf_counter() - start)
    rows = [
        {
            "prompt_id": prompt.prompt_id,
            "temperature": temperature,
            "output_prefix_tokens": 0,
            "fresh_seconds": fresh,
            "cached_seconds": float(np.median(prompt_cache_times)),
            "saved_fraction": 1 - float(np.median(prompt_cache_times)) / fresh if fresh else 0.0,
            "cache_bytes": prompt_cache_bytes,
            "cache_kind": "prompt_prefill",
        }
    ]
    seen_prefix_lengths: set[int] = set()
    for prefix_length in output_prefix_lengths:
        prefix_length = min(max(0, int(prefix_length)), len(baseline.tokens))
        if prefix_length == 0 or prefix_length in seen_prefix_lengths:
            continue  # row above already reports prompt-prefill reuse
        seen_prefix_lengths.add(prefix_length)
        with torch.inference_mode():
            start = time.perf_counter()
            output = model(input_ids=torch.tensor([list(prompt.token_ids)], dtype=torch.long, device=device), use_cache=True)
            output = _decode_tokens(model, baseline.tokens[:prefix_length], output, device, torch)
            _sync(device, torch)
            cache_build = time.perf_counter() - start
            suffix_times = []
            for _ in range(repeats):
                cached_prefix = copy.deepcopy(output)
                start = time.perf_counter()
                _decode_tokens(model, baseline.tokens[prefix_length:], cached_prefix, device, torch)
                _sync(device, torch)
                suffix_times.append(time.perf_counter() - start)
            cached = cache_build + float(np.median(suffix_times))
            rows.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "temperature": temperature,
                    "output_prefix_tokens": prefix_length,
                    "fresh_seconds": fresh,
                    "cached_seconds": cached,
                    "saved_fraction": 1 - cached / fresh if fresh else 0.0,
                    "cache_bytes": _cache_bytes(output),
                    "cache_kind": "output_prefix",
                }
            )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(output_dir / "cache_timings.csv", rows)
        _plot_cache_timings(rows, output_dir / "cache_timings.png")
    print(f"[cache] prompt={prompt.prompt_id}: complete ({len(rows)} timing rows)", flush=True)
    return rows


def cache_workload_statistics(
    prompts: Sequence[Prompt],
    *,
    repeat_probability: float = 0.5,
    capacities: Sequence[int] = (1, 4, 16, 64),
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Estimate exact-hit and compatible-prefix rates for controlled prompt streams.

    This is a workload model, not a wall-clock benchmark. It makes the cache
    assumption explicit before the model timing in :func:`run_cache_experiment`:
    exact repeats, shared prompt prefixes, and one-token near matches are all
    represented, with a bounded LRU resident set.
    """

    if not prompts:
        raise ValueError("prompts must be non-empty")
    if not 0 <= repeat_probability <= 1:
        raise ValueError("repeat_probability must be in [0, 1]")
    if any(int(capacity) < 1 for capacity in capacities):
        raise ValueError("cache capacities must be positive")
    rng = random.Random(seed)
    families: dict[str, list[tuple[int, ...]]] = {
        "independent": [tuple(prompt.token_ids) for prompt in prompts],
        "exact_repeat": [],
        "shared_prefix": [],
        "near_match": [],
    }
    for index, prompt in enumerate(prompts):
        original = tuple(prompt.token_ids)
        families["exact_repeat"].append(original)
        if rng.random() < repeat_probability:
            families["exact_repeat"].append(original)
        other = tuple(prompts[(index + 1) % len(prompts)].token_ids)
        shared = min(16, len(original), len(other))
        families["shared_prefix"].extend((original, original[:shared] + other[shared:]))
        changed = list(original)
        if changed:
            changed[0] = (changed[0] + 1) % 50_257
        families["near_match"].extend((original, tuple(changed)))
    rows: list[dict[str, Any]] = []
    for family, stream in families.items():
        for capacity in capacities:
            resident: list[tuple[int, ...]] = []
            exact_hits = 0
            compatible_lengths: list[int] = []
            for query in stream:
                if query in resident:
                    exact_hits += 1
                    resident.remove(query)
                longest = max((_common_prefix_length(query, cached) for cached in resident), default=0)
                compatible_lengths.append(longest)
                resident.insert(0, query)
                del resident[int(capacity):]
            rows.append(
                {
                    "workload": family,
                    "capacity": int(capacity),
                    "requests": len(stream),
                    "exact_hit_rate": exact_hits / len(stream),
                    "mean_compatible_prefix_tokens": float(np.mean(compatible_lengths)) if compatible_lengths else 0.0,
                    "max_compatible_prefix_tokens": max(compatible_lengths, default=0),
                }
            )
    return rows


def _common_prefix_length(first: Sequence[int], second: Sequence[int]) -> int:
    length = 0
    for left, right in zip(first, second):
        if left != right:
            break
        length += 1
    return length


def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Prompt],
    *,
    device: str = "cpu",
    progress_every: int = 100,
) -> float:
    """Compute a simple next-token perplexity sanity check over prompt windows."""

    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    import torch

    total_nll = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for prompt_index, prompt in enumerate(prompts, start=1):
            ids = torch.tensor([list(prompt.token_ids)], dtype=torch.long, device=device)
            if ids.shape[1] < 2:
                continue
            logits = model(input_ids=ids).logits[:, :-1, :]
            targets = ids[:, 1:]
            nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
            total_nll += float(nll.item())
            total_tokens += int(targets.numel())
            if prompt_index == 1 or prompt_index % progress_every == 0 or prompt_index == len(prompts):
                print(f"[perplexity] prompt={prompt_index}/{len(prompts)}", flush=True)
    if total_tokens == 0:
        raise ValueError("no next-token targets available")
    return math.exp(total_nll / total_tokens)


def _parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of numbers")
    return values


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--revision", help="Hugging Face revision/commit to pin")
    parser.add_argument("--dataset", choices=("wikitext2", "lambada"), default="wikitext2")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--prompts", type=int, default=500)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--devices",
        help="comma-separated worker devices; with --device cuda, omitted means all visible GPUs",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/gpt2"))
    parser.add_argument("--smoke", action="store_true", help="small local run for wiring checks")
    parser.add_argument("--progress-every", type=int, default=10, metavar="N", help="print progress every N prompts (default: 10)")
    parser.add_argument("--no-resume", action="store_true", help="discard checkpoints and restart the experiment")


def _load_inputs(args: argparse.Namespace, *, model_device: str | None = None):
    model, tokenizer, torch = _load_model(args.model, model_device or args.device, args.revision)
    prompts_count = 4 if args.smoke else args.prompts
    prompts = load_prompts(tokenizer, dataset=args.dataset, count=prompts_count, prompt_tokens=args.prompt_tokens, seed=args.seed, prompt_file=args.prompt_file)
    return model, tokenizer, torch, prompts


def _resolve_devices(device: str, devices: str | None) -> list[str]:
    """Resolve the requested worker devices, expanding ``cuda`` to all GPUs."""

    if devices:
        resolved = [item.strip() for item in devices.split(",") if item.strip()]
        if not resolved:
            raise ValueError("--devices must contain at least one device")
        return resolved
    if device == "cuda":
        try:
            import torch

            count = int(torch.cuda.device_count())
        except ImportError:
            return [device]
        # Leave the generic ``cuda`` target intact when no device is visible so
        # PyTorch can report its actionable driver/runtime error.
        return [f"cuda:{index}" for index in range(count)] or [device]
    return [device]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collision = sub.add_parser("collision", help="estimate prefix collision curves")
    _common_parser(collision)
    collision.add_argument("--pairs", type=int, default=4)
    collision.add_argument("--temperatures", type=_parse_floats, default=DEFAULT_TEMPERATURES)
    collision.add_argument("--length", type=int, default=32)
    collision.add_argument("--bootstrap-replicates", type=int, default=1000)
    collision.add_argument("--record-transcripts", action="store_true", help="persist every per-step VRF transcript (large output)")
    concentration = sub.add_parser("concentration", help="measure token concentration")
    _common_parser(concentration)
    concentration.add_argument("--temperatures", type=_parse_floats, default=DEFAULT_TEMPERATURES)
    concentration.add_argument("--length", type=int, default=32)
    concentration.add_argument("--samples-per-prompt", type=int, default=1)
    cache = sub.add_parser("cache", help="measure exact prompt/output KV reuse")
    _common_parser(cache)
    cache.add_argument("--length", type=int, default=32)
    cache.add_argument("--temperature", type=float, default=1.0)
    cache.add_argument("--prefix-lengths", type=_parse_floats, default=(0, 8, 16, 32))
    cache.add_argument("--repeats", type=int, default=3)
    cache.add_argument("--repeat-probability", type=float, default=0.5)
    cache.add_argument("--cache-capacities", type=_parse_floats, default=(1, 4, 16, 64))
    perplexity = sub.add_parser("perplexity", help="run a next-token perplexity sanity check")
    _common_parser(perplexity)
    run_all = sub.add_parser("run-all", help="run collision, concentration, and cache experiments")
    _common_parser(run_all)
    run_all.add_argument("--pairs", type=int, default=4)
    run_all.add_argument("--temperatures", type=_parse_floats, default=DEFAULT_TEMPERATURES)
    run_all.add_argument("--length", type=int, default=32)
    run_all.add_argument("--bootstrap-replicates", type=int, default=1000)
    run_all.add_argument("--samples-per-prompt", type=int, default=1)
    run_all.add_argument("--repeats", type=int, default=3)
    run_all.add_argument("--repeat-probability", type=float, default=0.5)
    run_all.add_argument("--cache-capacities", type=_parse_floats, default=(1, 4, 16, 64))
    run_all.add_argument("--record-transcripts", action="store_true", help="persist every per-step VRF transcript (large output)")

    args = parser.parse_args(argv)
    if args.smoke:
        args.output_dir = args.output_dir / "smoke"
        if hasattr(args, "pairs"):
            args.pairs = min(args.pairs, 4)
        if hasattr(args, "length"):
            args.length = min(args.length, 8)
        if hasattr(args, "bootstrap_replicates"):
            args.bootstrap_replicates = min(args.bootstrap_replicates, 100)
    try:
        devices = _resolve_devices(args.device, args.devices)
        model, tokenizer, torch, prompts = _load_inputs(args, model_device=devices[0])
        worker_models: list[tuple[Any, Any, str]] = [(model, tokenizer, devices[0])]
        for worker_device in devices[1:]:
            worker_model, _, _ = _load_model(args.model, worker_device, args.revision)
            worker_models.append((worker_model, tokenizer, worker_device))
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model,
        "revision": args.revision,
        "model_config_sha256": hashlib.sha256(model.config.to_json_string().encode("utf-8")).hexdigest(),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "dataset": args.dataset,
        "prompt_count": len(prompts),
        "prompt_tokens": args.prompt_tokens,
        "seed": args.seed,
        "device": args.device,
        "devices": devices,
        "parallel_workers": len(worker_models),
        "torch_version": torch.__version__,
        "vrf": "repository Ed25519 sign-then-hash VRF; seed-derived key; per-step messages",
        "sampling": "inverse CDF, temperature, top_p=1, no top_k/repetition penalty by default",
        "eos": "stop at tokenizer EOS; no post-EOS tokens invented",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.command == "collision":
        # The plan calls for a next-token sanity check before collision data;
        # keeping it beside the curve makes accidental model/tokenizer changes
        # visible in every campaign directory.
        perplexity = evaluate_perplexity(model, tokenizer, prompts, device=devices[0], progress_every=args.progress_every)
        (args.output_dir / "perplexity.json").write_text(json.dumps({"perplexity": perplexity}, indent=2) + "\n")
        print(f"perplexity={perplexity:.4f}")
        run_collision_experiment(model, tokenizer, prompts, pairs=args.pairs, temperatures=args.temperatures, length=args.length, seed=args.seed, device=args.device, bootstrap_replicates=args.bootstrap_replicates, output_dir=args.output_dir, record_transcripts=args.record_transcripts, resume=not args.no_resume, progress_every=args.progress_every, worker_models=worker_models)
    elif args.command == "concentration":
        run_concentration_experiment(model, tokenizer, prompts, temperatures=args.temperatures, length=args.length, samples_per_prompt=args.samples_per_prompt, seed=args.seed, device=args.device, output_dir=args.output_dir, resume=not args.no_resume, progress_every=args.progress_every, worker_models=worker_models)
    elif args.command == "cache":
        run_cache_experiment(model, tokenizer, prompts[0], length=args.length, output_prefix_lengths=[int(x) for x in args.prefix_lengths], repeats=args.repeats, seed=args.seed, device=devices[0], temperature=args.temperature, output_dir=args.output_dir)
        workload_rows = cache_workload_statistics(prompts, repeat_probability=args.repeat_probability, capacities=[int(x) for x in args.cache_capacities], seed=args.seed)
        _write_csv(args.output_dir / "cache_workload.csv", workload_rows)
        _plot_cache_workload(workload_rows, args.output_dir / "cache_workload.png")
    elif args.command == "perplexity":
        value = evaluate_perplexity(model, tokenizer, prompts, device=devices[0], progress_every=args.progress_every)
        (args.output_dir / "perplexity.json").write_text(json.dumps({"perplexity": value}, indent=2) + "\n")
        print(f"perplexity={value:.4f}")
    elif args.command == "run-all":
        perplexity = evaluate_perplexity(model, tokenizer, prompts, device=devices[0], progress_every=args.progress_every)
        (args.output_dir / "perplexity.json").write_text(json.dumps({"perplexity": perplexity}, indent=2) + "\n")
        print(f"perplexity={perplexity:.4f}")
        run_collision_experiment(model, tokenizer, prompts, pairs=args.pairs, temperatures=args.temperatures, length=args.length, seed=args.seed, device=args.device, bootstrap_replicates=args.bootstrap_replicates, output_dir=args.output_dir / "collision", record_transcripts=args.record_transcripts, resume=not args.no_resume, progress_every=args.progress_every, worker_models=worker_models)
        run_concentration_experiment(model, tokenizer, prompts, temperatures=args.temperatures, length=args.length, samples_per_prompt=args.samples_per_prompt, seed=args.seed, device=args.device, output_dir=args.output_dir / "concentration", resume=not args.no_resume, progress_every=args.progress_every, worker_models=worker_models)
        run_cache_experiment(model, tokenizer, prompts[0], length=args.length, repeats=args.repeats, seed=args.seed, device=devices[0], output_dir=args.output_dir / "cache")
        workload_rows = cache_workload_statistics(prompts, repeat_probability=args.repeat_probability, capacities=[int(x) for x in args.cache_capacities], seed=args.seed)
        _write_csv(args.output_dir / "cache_workload.csv", workload_rows)
        _plot_cache_workload(workload_rows, args.output_dir / "cache_workload.png")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
