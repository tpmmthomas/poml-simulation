"""Independent GPT-2 sampling and generated-prefix collision measurements."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .crypto import sha256
from .vrf import vrf_eval

DEFAULT_TEMPERATURES = (0.7, 1.0, 1.3, 1.5, 2.0)
VRF_DOMAIN = b"poml-gpt2-sampler-v1\x00"


@dataclass(frozen=True)
class Prompt:
    """A deduplicated fixed-length tokenized prompt."""

    prompt_id: str
    source: str
    text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class Trace:
    """Generated tokens and their sampling probabilities."""

    tokens: tuple[int, ...]
    token_probabilities: tuple[float, ...]


def _seed_bytes(seed: int | bytes | str) -> bytes:
    if isinstance(seed, bytes):
        return seed
    if isinstance(seed, int):
        if seed < 0:
            raise ValueError("seed must be non-negative")
        return str(seed).encode("ascii")
    return str(seed).encode("utf-8")


def _vrf_private_key(seed: int | bytes | str) -> bytes:
    return hashlib.sha256(b"poml-gpt2-vrf-key-v1\x00" + _seed_bytes(seed)).digest()


def _token_digest(tokens: Sequence[int]) -> bytes:
    payload = b"".join(int(token).to_bytes(4, "big") for token in tokens)
    return sha256(b"poml-gpt2-prompt-v1\x00" + payload)


def _uniforms(seed: int | bytes | str, prompt_tokens: Sequence[int], length: int) -> list[float]:
    """Derive reproducible independent inverse-CDF uniforms from the repository VRF."""

    if length < 0:
        raise ValueError("length must be non-negative")
    secret = _vrf_private_key(seed)
    digest = _token_digest(prompt_tokens)
    values = []
    for step in range(1, length + 1):
        output, _ = vrf_eval(secret, VRF_DOMAIN + digest + step.to_bytes(4, "big"))
        integer = int.from_bytes(output[:8], "big")
        values.append((integer + 0.5) / float(1 << 64))
    return values


def load_prompts(
    tokenizer: Any,
    *,
    count: int = 500,
    prompt_tokens: int = 32,
    seed: int = 42,
) -> list[Prompt]:
    """Sample deduplicated fixed-token WikiText-2 test windows."""

    if count < 1 or prompt_tokens < 1:
        raise ValueError("count and prompt_tokens must be positive")
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - CLI-only dependency
        raise RuntimeError(
            "GPT-2 collision runs require datasets; install with pip install -e '.[llm]'"
        ) from exc
    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    candidates: list[Prompt] = []
    seen: set[tuple[int, ...]] = set()
    for row_id, row in enumerate(rows):
        text = row.get("text", "")
        if not text or not text.strip():
            continue
        token_ids = tuple(
            int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]
        )
        if len(token_ids) < prompt_tokens:
            continue
        window = token_ids[:prompt_tokens]
        if window in seen:
            continue
        seen.add(window)
        candidates.append(Prompt(f"wikitext2:{row_id}", "wikitext2", text, window))
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} usable prompts, requested {count}")
    return random.Random(seed).sample(candidates, count)


def _probabilities(logits: Any, temperature: float):
    import torch

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = logits.detach().to(dtype=torch.float64, device="cpu") / temperature
    return torch.softmax(values, dim=-1)


def sample_token(logits: Any, uniform: float, *, temperature: float = 1.0) -> tuple[int, float]:
    """Inverse-CDF sample from the full temperature-scaled vocabulary."""

    import torch

    if not 0 <= uniform < 1:
        raise ValueError("uniform must be in [0, 1)")
    probabilities = _probabilities(logits, temperature)
    index = int(
        torch.searchsorted(
            torch.cumsum(probabilities, dim=-1),
            torch.tensor(uniform, dtype=probabilities.dtype),
        ).item()
    )
    index = min(index, probabilities.numel() - 1)
    return index, float(probabilities[index].item())


def generate_trace(
    model: Any,
    tokenizer: Any,
    prompt_tokens: Sequence[int],
    seed: int | bytes | str,
    *,
    temperature: float,
    length: int,
    device: str,
) -> Trace:
    """Generate one EOS-aware cached autoregressive trace."""

    import torch

    max_positions = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", None
    )
    if max_positions is not None and len(prompt_tokens) + length > int(max_positions):
        raise ValueError("prompt plus generation exceeds the model context")
    uniforms = _uniforms(seed, prompt_tokens, length)
    input_ids = torch.tensor([list(prompt_tokens)], dtype=torch.long, device=device)
    eos_id = tokenizer.eos_token_id
    tokens: list[int] = []
    probabilities: list[float] = []
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True)
        for step, uniform in enumerate(uniforms):
            token, probability = sample_token(
                output.logits[0, -1], uniform, temperature=temperature
            )
            tokens.append(token)
            probabilities.append(probability)
            if eos_id is not None and token == eos_id:
                break
            next_token = torch.tensor([[token]], dtype=torch.long, device=device)
            output = model(
                input_ids=next_token,
                past_key_values=output.past_key_values,
                use_cache=True,
            )
    return Trace(tuple(tokens), tuple(probabilities))


def prefix_indicators(first: Sequence[int], second: Sequence[int], length: int) -> list[int | None]:
    """Return exact-prefix indicators; ``None`` marks EOS-censored pairs."""

    if length < 0:
        raise ValueError("length must be non-negative")
    return [
        1
        if ell == 0
        else None
        if len(first) < ell or len(second) < ell
        else int(first[:ell] == second[:ell])
        for ell in range(length + 1)
    ]


def first_divergence(first: Sequence[int], second: Sequence[int]) -> int | None:
    """Return the one-based first differing position, including EOS divergence."""

    for position, (left, right) in enumerate(zip(first, second), start=1):
        if left != right:
            return position
    if len(first) != len(second):
        return min(len(first), len(second)) + 1
    return None


def _bootstrap(curves: list[list[float]], replicates: int, seed: int):
    import numpy as np

    values = np.asarray(curves, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("curves must be a non-empty matrix")
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, values.shape[1]))
    for index in range(replicates):
        samples[index] = np.nanmean(values[rng.integers(0, len(values), len(values))], axis=0)
    return np.nanpercentile(samples, 2.5, axis=0), np.nanpercentile(samples, 97.5, axis=0)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_collision(rows: list[dict[str, Any]], path: Path, pairs: int, prompt_count: int) -> None:
    """Write the appendix graph with readable labels and no explanatory footer."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    temperatures = sorted({float(row["temperature"]) for row in rows})
    length = max(int(row["prefix_length"]) for row in rows)
    display_length = min(8, length)
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )
    figure, (curve_axis, expected_axis) = plt.subplots(
        1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": (2.25, 1)}
    )
    expected = []
    maximum = 0.0
    for color_index, temperature in enumerate(temperatures):
        selected = [
            row
            for row in rows
            if float(row["temperature"]) == temperature
            and 1 <= int(row["prefix_length"]) <= display_length
        ]
        selected.sort(key=lambda row: int(row["prefix_length"]))
        x = [int(row["prefix_length"]) for row in selected]
        y = [float(row["collision_probability"]) for row in selected]
        maximum = max(maximum, *y)
        curve_axis.plot(
            x,
            y,
            marker="o",
            linewidth=2.5,
            markersize=6,
            label=f"T={temperature:g}",
            color=f"C{color_index}",
        )
        expected.append(
            (
                temperature,
                sum(
                    float(row["collision_probability"])
                    for row in rows
                    if float(row["temperature"]) == temperature and int(row["prefix_length"]) > 0
                ),
                f"C{color_index}",
            )
        )
    curve_axis.set_xlim(0.8, display_length + 0.2)
    curve_axis.set_xticks(range(1, display_length + 1))
    curve_axis.set_ylim(0, max(maximum * 1.15, 0.01))
    curve_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    curve_axis.set_xlabel(r"Generated prefix length $\ell$")
    curve_axis.set_ylabel(r"Probability that two traces share prefix $C_\ell$")
    curve_axis.set_title("Early prefix-collision curve")
    curve_axis.grid(alpha=0.25)
    curve_axis.legend(title="Temperature")
    values = [value for _, value, _ in expected]
    bars = expected_axis.barh(
        [f"T={temperature:g}" for temperature, _, _ in expected],
        values,
        color=[color for _, _, color in expected],
    )
    expected_axis.invert_yaxis()
    expected_axis.set_xlim(0, max(max(values, default=0.0) * 1.2, 0.01))
    expected_axis.set_xlabel("Expected shared prefix (tokens)")
    expected_axis.set_title("Expected shared prefix")
    expected_axis.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        expected_axis.text(
            value, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=14
        )
    figure.suptitle(
        f"GPT-2 prefix collisions ({prompt_count} prompts × {pairs} pairs per temperature)",
        fontsize=21,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


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
    output_dir: Path,
    resume: bool = True,
    progress_every: int = 10,
) -> dict[str, Any]:
    """Run the collision campaign and write CSV, metadata, and plot outputs."""

    import numpy as np

    if not prompts or pairs < 1 or length < 1 or bootstrap_replicates < 1:
        raise ValueError("prompts, pairs, length, and bootstrap_replicates must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "collision_checkpoint.jsonl"
    metadata_path = output_dir / "collision_checkpoint.meta.json"
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "prompts": [p.prompt_id for p in prompts],
                "pairs": pairs,
                "temperatures": list(temperatures),
                "length": length,
                "seed": seed,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if not resume:
        checkpoint.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    if (
        metadata_path.exists()
        and json.loads(metadata_path.read_text()).get("fingerprint") != fingerprint
    ):
        raise ValueError("checkpoint settings differ; use a new output directory or resume=False")
    metadata_path.write_text(json.dumps({"fingerprint": fingerprint}, indent=2) + "\n")
    completed: dict[str, dict[str, Any]] = {}
    if resume and checkpoint.exists():
        with checkpoint.open() as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                    completed[record["key"]] = record["result"]
                except (json.JSONDecodeError, KeyError):
                    continue
    summary_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    divergence_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for temperature in temperatures:
        prompt_curves: list[list[float]] = []
        pooled_counts = np.zeros(length + 1, dtype=int)
        pooled_comparisons = np.zeros(length + 1, dtype=int)
        for index, prompt in enumerate(prompts, start=1):
            key = f"{temperature:.12g}|{prompt.prompt_id}"
            record = completed.get(key)
            if record is None:
                counts = np.zeros(length + 1, dtype=int)
                comparisons = np.zeros(length + 1, dtype=int)
                divergences: dict[str, int] = {}
                sequence_probabilities: list[float] = []
                for pair in range(pairs):
                    base = f"{seed}|{prompt.prompt_id}|{temperature:.12g}|{pair}"
                    first = generate_trace(
                        model,
                        tokenizer,
                        prompt.token_ids,
                        base + "|a",
                        temperature=temperature,
                        length=length,
                        device=device,
                    )
                    second = generate_trace(
                        model,
                        tokenizer,
                        prompt.token_ids,
                        base + "|b",
                        temperature=temperature,
                        length=length,
                        device=device,
                    )
                    sequence_probabilities.extend(
                        math.exp(sum(math.log(max(p, 1e-300)) for p in trace.token_probabilities))
                        for trace in (first, second)
                    )
                    for ell, indicator in enumerate(
                        prefix_indicators(first.tokens, second.tokens, length)
                    ):
                        if indicator is not None:
                            counts[ell] += indicator
                            comparisons[ell] += 1
                    divergence = first_divergence(first.tokens, second.tokens)
                    if divergence is not None:
                        divergences[str(divergence)] = divergences.get(str(divergence), 0) + 1
                record = {
                    "counts": counts.tolist(),
                    "comparisons": comparisons.tolist(),
                    "divergences": divergences,
                    "sequence_probability_mean": float(np.mean(sequence_probabilities)),
                }
                completed[key] = record
                with checkpoint.open("a") as stream:
                    stream.write(
                        json.dumps({"key": key, "result": record}, separators=(",", ":")) + "\n"
                    )
            counts = np.asarray(record["counts"], dtype=int)
            comparisons = np.asarray(record["comparisons"], dtype=int)
            curve = [
                counts[ell] / comparisons[ell] if comparisons[ell] else np.nan
                for ell in range(length + 1)
            ]
            prompt_curves.append(curve)
            pooled_counts += counts
            pooled_comparisons += comparisons
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
                {
                    "dataset": prompt.source,
                    "prompt_id": prompt.prompt_id,
                    "temperature": temperature,
                    "first_divergence": int(position),
                    "count": int(count),
                }
                for position, count in record["divergences"].items()
            )
            if index == 1 or index % progress_every == 0 or index == len(prompts):
                elapsed = time.perf_counter() - started
                print(
                    f"[collision] temperature={temperature:g} prompt={index}/{len(prompts)} ({elapsed:.0f}s)",
                    flush=True,
                )
        low, high = _bootstrap(prompt_curves, bootstrap_replicates, seed + int(temperature * 1000))
        for ell in range(length + 1):
            probability = (
                pooled_counts[ell] / pooled_comparisons[ell]
                if pooled_comparisons[ell]
                else float("nan")
            )
            summary_rows.append(
                {
                    "dataset": prompts[0].source,
                    "temperature": temperature,
                    "prefix_length": ell,
                    "collisions": int(pooled_counts[ell]),
                    "comparisons": int(pooled_comparisons[ell]),
                    "collision_probability": probability,
                    "prompt_median": float(np.nanmedian(np.asarray(prompt_curves)[:, ell])),
                    "prompt_p10": float(np.nanpercentile(np.asarray(prompt_curves)[:, ell], 10)),
                    "prompt_p90": float(np.nanpercentile(np.asarray(prompt_curves)[:, ell], 90)),
                    "prompt_fraction_with_collision": float(
                        np.mean(np.asarray(prompt_curves)[:, ell] > 0)
                    ),
                    "bootstrap_low": float(low[ell]),
                    "bootstrap_high": float(high[ell]),
                    "zero_collision_upper_95": (
                        3.0 / int(pooled_comparisons[ell])
                        if pooled_comparisons[ell] and pooled_counts[ell] == 0
                        else None
                    ),
                }
            )
        aggregate_rows.append(
            {
                "dataset": prompts[0].source,
                "temperature": temperature,
                "expected_common_prefix_length": float(
                    sum(
                        row["collision_probability"]
                        for row in summary_rows
                        if row["temperature"] == temperature and row["prefix_length"] > 0
                    )
                ),
                "lp_sequence_probability_mean": float(
                    np.mean(
                        [
                            completed[f"{temperature:.12g}|{prompt.prompt_id}"][
                                "sequence_probability_mean"
                            ]
                            for prompt in prompts
                        ]
                    )
                ),
            }
        )
    _write_csv(output_dir / "collision_curves.csv", summary_rows)
    _write_csv(output_dir / "prompt_curves.csv", prompt_rows)
    _write_csv(output_dir / "first_divergence.csv", divergence_rows)
    _write_csv(output_dir / "collision_aggregates.csv", aggregate_rows)
    _plot_collision(summary_rows, output_dir / "collision_curves.png", pairs, len(prompts))
    return {"summary": summary_rows, "aggregates": aggregate_rows}
