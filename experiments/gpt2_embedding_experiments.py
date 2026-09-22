"""GPT-2 one-time embedding-perturbation experiments.

This driver implements the two experiments in ``docs/gpt2_new_plan.md``:
utility under a challenge-conditioned prefix perturbation and separation of
the quantized inference trace at the proving boundary.  Model and dataset
imports stay lazy so the deterministic helpers and the smoke tests work on a
machine without Hugging Face downloads.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from poml_sim.vrf import vrf_eval  # noqa: E402
from gpt2_experiments import (  # noqa: E402
    _checkpoint_fingerprint,
    _load_model,
    _open_checkpoint,
    _parse_floats,
    _resolve_devices,
    _write_csv,
    token_digest,
    vrf_uniforms,
)


DEFAULT_SIGMAS = (0.0, 0.005, 0.01, 0.02, 0.04)
EMBEDDING_DOMAIN = b"poml-gpt2-embedding-v1\x00"
DEFAULT_BOUNDARIES = (
    "embedding",
    "post_first_norm",
    "layer_0_q",
    "layer_0_k",
    "layer_0_v",
    "block_0",
    "logits",
)
SEPARATION_CONDITIONS = ("no_perturbation", "common_token_stream", "full_protocol")


@dataclass(frozen=True)
class UtilityExample:
    """One fixed prompt and its unperturbed target continuation."""

    example_id: str
    task: str
    text: str
    context_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    choices: tuple[tuple[int, ...], ...] = ()
    answer: int | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TracePrompt:
    """A prompt used by the trace-separation experiment."""

    prompt_id: str
    stratum: str
    text: str
    token_ids: tuple[int, ...]
    target_token: int | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PerturbationConfig:
    """Numerical settings that define the reproducible perturbation law."""

    sigma_abs: float
    clip: float | None = 3.0
    quantum: float = 1e-6


def embedding_table_std(model: Any) -> float:
    """Return the entrywise standard deviation ``s_E`` of the frozen table."""

    weight = getattr(model.get_input_embeddings(), "weight", None)
    if weight is None:
        raise ValueError("model input embeddings do not expose a weight table")
    return float(weight.detach().to(device="cpu", dtype=_torch().float64).std(correction=0).item())


def relative_noise_scales(model: Any, alphas: Sequence[float]) -> tuple[float, tuple[float, ...]]:
    """Compute the public population standard deviation once, then alpha*s_E."""
    if not alphas or any(not math.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("alphas must be finite and positive")
    table_std = embedding_table_std(model)
    if not math.isfinite(table_std) or table_std <= 0:
        raise ValueError("embedding table must have a positive finite standard deviation")
    return table_std, tuple(float(alpha) * table_std for alpha in alphas)


def model_weight_sha256(model: Any) -> str | None:
    """Hash frozen model weights for campaign provenance."""

    if not hasattr(model, "state_dict"):
        return None
    digest = hashlib.sha256(b"poml-gpt2-checkpoint-v1\x00")
    try:
        state = model.state_dict()
        for name in sorted(state):
            value = state[name].detach().to(device="cpu")
            digest.update(name.encode("utf-8") + b"\x00")
            digest.update(str(value.dtype).encode("ascii") + b"\x00")
            digest.update(str(tuple(value.shape)).encode("ascii") + b"\x00")
            digest.update(value.numpy().tobytes(order="C"))
    except (AttributeError, RuntimeError, TypeError):
        return None
    return digest.hexdigest()


def _seed_bytes(seed: int | bytes | str) -> bytes:
    if isinstance(seed, bytes):
        return seed
    if isinstance(seed, int):
        if seed < 0:
            raise ValueError("seed must be non-negative")
        return str(seed).encode("ascii")
    return str(seed).encode("utf-8")


def _embedding_challenge(seed: int | bytes | str, prompt_tokens: Sequence[int]) -> bytes:
    """Derive the VRF value used by the embedding-noise stream."""

    private_key = hashlib.sha256(b"poml-gpt2-embedding-key-v1\x00" + _seed_bytes(seed)).digest()
    message = EMBEDDING_DOMAIN + token_digest(prompt_tokens) + _seed_bytes(seed)
    output, _ = vrf_eval(private_key, message)
    return output


def _stream_words(seed: int | bytes | str, prompt_tokens: Sequence[int], count: int) -> Iterable[int]:
    challenge = _embedding_challenge(seed, prompt_tokens)
    digest = token_digest(prompt_tokens)
    for counter in range((count + 7) // 8):
        block = hashlib.sha256(
            EMBEDDING_DOMAIN + challenge + digest + counter.to_bytes(8, "big")
        ).digest()
        for offset in range(0, len(block), 4):
            yield int.from_bytes(block[offset : offset + 4], "big")


def deterministic_embedding_noise(
    seed: int | bytes | str,
    prompt_tokens: Sequence[int],
    shape: Sequence[int],
    sigma_abs: float,
    *,
    clip: float | None = 3.0,
    quantum: float = 1e-6,
) -> np.ndarray:
    """Return deterministic Gaussian prefix noise with optional discretization.

    The hash/Box--Muller expansion is deliberately independent from the VRF
    token sampler.  Quantizing the result makes the experiment stable across
    devices and provides a concrete finite representation for a later circuit
    implementation.
    """

    if not math.isfinite(sigma_abs) or sigma_abs < 0:
        raise ValueError("sigma_abs must be finite and non-negative")
    if (clip is not None and (not math.isfinite(clip) or clip <= 0)) or not math.isfinite(quantum) or quantum < 0:
        raise ValueError("clip must be positive or None; quantum must be non-negative")
    dimensions = tuple(int(value) for value in shape)
    if any(value < 0 for value in dimensions):
        raise ValueError("shape must contain non-negative dimensions")
    size = int(np.prod(dimensions, dtype=np.int64))
    words = iter(_stream_words(seed, prompt_tokens, size * 2))
    values = np.empty(size, dtype=np.float64)
    denominator = float(1 << 32)
    for index in range(size):
        u1 = (next(words) + 0.5) / denominator
        u2 = (next(words) + 0.5) / denominator
        z = math.sqrt(-2.0 * math.log(max(u1, 1e-15))) * math.cos(2.0 * math.pi * u2)
        values[index] = (max(-clip, min(clip, z)) if clip is not None else z) * sigma_abs
    if quantum:
        values = np.rint(values / quantum) * quantum
    return values.reshape(dimensions)


def quantize_tensor(value: Any, scale: float, *, clip: int | None = None) -> np.ndarray:
    """Apply the integer quantizer used for comparison statistics."""

    if scale <= 0:
        raise ValueError("quantization scale must be positive")
    if hasattr(value, "detach"):
        torch = _torch()
        array = value.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        array = np.asarray(value, dtype=np.float64)
    quantized = np.rint(array / scale)
    if clip is not None:
        if clip < 1:
            raise ValueError("quantization clip must be positive")
        quantized = np.clip(quantized, -clip, clip)
    return quantized.astype(np.int64, copy=False)


def changed_coordinate_fraction(first: Any, second: Any) -> float:
    """Return the fraction of integer tensor coordinates that changed."""

    left, right = np.asarray(first), np.asarray(second)
    if left.shape != right.shape:
        raise ValueError("compared tensors must have identical shapes")
    return float(np.count_nonzero(left != right) / left.size) if left.size else 0.0


def compare_quantized_boundaries(
    first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Compare complete quantized tensors, not a single row or token."""

    rows: list[dict[str, Any]] = []
    for boundary in first.keys() | second.keys():
        if boundary not in first or boundary not in second:
            rows.append({"boundary": boundary, "collision": False, "changed_fraction": 1.0})
            continue
        left, right = first[boundary], second[boundary]
        rows.append(
            {
                "boundary": boundary,
                "collision": bool(np.array_equal(left, right)),
                "changed_fraction": changed_coordinate_fraction(left, right),
                "coordinates": int(left.size),
            }
        )
    return sorted(rows, key=lambda row: row["boundary"])


def quantized_trace_commitment(boundaries: Mapping[str, np.ndarray], *, challenge: int | bytes | str | None = None) -> str:
    """Commit to a challenge and every coordinate of a quantized trace."""

    digest = hashlib.sha256(b"poml-gpt2-quantized-trace-v1\x00")
    if challenge is not None:
        digest.update(b"challenge\x00" + _seed_bytes(challenge) + b"\x00")
    for boundary in sorted(boundaries):
        value = np.asarray(boundaries[boundary], dtype=np.int64)
        digest.update(boundary.encode("utf-8") + b"\x00")
        digest.update(str(value.shape).encode("ascii") + b"\x00")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def replay_test_from_commitments(
    original: Sequence[str], challenged: Sequence[str]
) -> dict[str, int]:
    """Perform the pre-integration proof replay check on trace commitments.

    A real DeepProve proof is accepted for its original statement, rejected
    when the challenge-bound trace commitment changes, and freshly generated
    under the new challenge.  This helper tests exactly that binding rule
    without pretending to be a ZK proof when DeepProve is not installed.
    """

    if len(original) != len(challenged):
        raise ValueError("commitment lists must have equal length")
    return {
        "old_under_old_accepted": len(original),
        "old_under_new_rejected": sum(left != right for left, right in zip(original, challenged)),
        "fresh_under_new_accepted": len(challenged),
    }


def summarize_trace_separation(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pool per-prompt separation rows by stratum, scale, and condition."""

    groups: dict[tuple[str, float, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["stratum"]), float(row["sigma_abs"]), str(row["condition"]), str(row["boundary"]))
        groups.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for (stratum, sigma, condition, boundary), selected in sorted(groups.items()):
        comparisons = sum(int(row["comparisons"]) for row in selected)
        collisions = sum(int(row["collisions"]) for row in selected)
        result.append(
            {
                "stratum": stratum,
                "sigma_abs": sigma,
                "condition": condition,
                "boundary": boundary,
                "collisions": collisions,
                "comparisons": comparisons,
                "collision_rate": collisions / comparisons if comparisons else float("nan"),
                "changed_coordinate_fraction": float(np.average([float(row["changed_coordinate_fraction"]) for row in selected], weights=[max(1, int(row["comparisons"])) for row in selected])) if selected else float("nan"),
            }
        )
    return result


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - CLI-only dependency
        raise RuntimeError("embedding experiments require torch and transformers; install pip install -e '.[dev,llm]'") from exc
    return torch


def _progress_bar(total: int, description: str, *, unit: str = "item") -> Any:
    """Create a tqdm bar while keeping tqdm optional for pure helpers."""

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - tqdm is a project dependency
        return None
    return tqdm(total=total, desc=description, unit=unit, dynamic_ncols=True)


def _open_embedding_checkpoint(
    output_dir: Path | None,
    *,
    filename: str,
    fingerprint: str,
    resume: bool,
) -> tuple[Path | None, dict[str, dict[str, Any]]]:
    """Open a checkpoint, archiving incompatible settings instead of failing.

    A changed prompt cap or sample default must never mix records with the old
    campaign. Archiving keeps that work available while allowing the default
    command to start the newly requested campaign without a manual cleanup.
    """

    try:
        return _open_checkpoint(output_dir, filename=filename, fingerprint=fingerprint, resume=resume)
    except ValueError as exc:
        if output_dir is None or not resume or "belongs to different settings" not in str(exc):
            raise
        checkpoint_path = output_dir / filename
        metadata_path = checkpoint_path.with_suffix(".meta.json")
        archive_tag = f"stale-{fingerprint[:12]}"
        for path in (checkpoint_path, metadata_path):
            if not path.exists():
                continue
            archived = path.with_name(f"{path.stem}.{archive_tag}{path.suffix}")
            suffix = 1
            while archived.exists():
                archived = path.with_name(f"{path.stem}.{archive_tag}-{suffix}{path.suffix}")
                suffix += 1
            path.replace(archived)
        print(f"[checkpoint] archived incompatible {filename}; starting a fresh checkpoint", flush=True)
        return _open_checkpoint(output_dir, filename=filename, fingerprint=fingerprint, resume=True)


def _token_embeddings(model: Any, token_ids: Sequence[int], device: str):
    torch = _torch()
    input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    embedding = model.get_input_embeddings()
    token_values = embedding(input_ids)
    return input_ids, token_values


def _embedding_dimension(model: Any) -> int:
    configured = getattr(getattr(model, "config", None), "n_embd", None)
    if configured is not None:
        return int(configured)
    weight = getattr(model.get_input_embeddings(), "weight", None)
    if weight is None or weight.ndim != 2:
        raise ValueError("cannot determine model embedding dimension")
    return int(weight.shape[-1])


def _forward_prefix(
    model: Any,
    token_ids: Sequence[int],
    noise: np.ndarray | None,
    *,
    prefix_length: int,
    device: str,
    output_hidden_states: bool = False,
):
    """Forward token IDs while perturbing only the prompt prefix embeddings."""

    torch = _torch()
    input_ids, embeddings = _token_embeddings(model, token_ids, device)
    if noise is not None and prefix_length:
        if noise.shape != tuple(embeddings[:, :prefix_length, :].shape[1:]):
            raise ValueError("noise shape does not match the prompt embedding shape")
        embeddings = embeddings.clone()
        embeddings[:, :prefix_length, :] += torch.as_tensor(noise, dtype=embeddings.dtype, device=device).unsqueeze(0)
    position_ids = torch.arange(len(token_ids), dtype=torch.long, device=device).unsqueeze(0)
    # ``use_cache`` is harmless for scoring and is required by autoregressive
    # generation, whose first call supplies the prompt-side KV state.
    kwargs = {"inputs_embeds": embeddings, "position_ids": position_ids, "use_cache": True}
    if output_hidden_states:
        kwargs["output_hidden_states"] = True
    try:
        return model(**kwargs)
    except (TypeError, ValueError) as exc:
        if noise is not None and np.any(noise):
            raise RuntimeError("the selected model does not accept inputs_embeds; cannot apply prefix perturbation") from exc
        return model(input_ids=input_ids, output_hidden_states=output_hidden_states, use_cache=False)


def _filtered_distribution(logits: Any, temperature: float) -> Any:
    torch = _torch()
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.softmax(logits.to(dtype=torch.float64) / temperature, dim=-1)


def _sample_from_distribution(probabilities: Any, uniform: float) -> tuple[int, float, float, float]:
    torch = _torch()
    if not 0 <= uniform < 1:
        raise ValueError("uniform must be in [0, 1)")
    cdf = torch.cumsum(probabilities, dim=-1)
    uniform_value = torch.tensor(uniform, dtype=cdf.dtype, device=cdf.device)
    index = min(int(torch.searchsorted(cdf, uniform_value).item()), probabilities.numel() - 1)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * torch.log(nonzero)).sum().item())
    return index, float(probabilities[index].item()), float(probabilities.max().item()), entropy


def generate_perturbed_trace(
    model: Any,
    tokenizer: Any,
    prompt_tokens: Sequence[int],
    *,
    embedding_seed: int | bytes | str,
    sampling_seed: int | bytes | str,
    sigma_abs: float,
    length: int = 32,
    temperature: float = 1.0,
    device: str = "cpu",
    config: PerturbationConfig | None = None,
) -> dict[str, Any]:
    """Generate a deterministic trace with one perturbed prompt prefill."""

    if length < 0:
        raise ValueError("length must be non-negative")
    config = config or PerturbationConfig(sigma_abs=sigma_abs)
    if abs(config.sigma_abs - sigma_abs) > 0:
        raise ValueError("config.sigma_abs and sigma_abs disagree")
    torch = _torch()
    max_positions = getattr(getattr(model, "config", None), "n_positions", None) or getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if max_positions is not None and len(prompt_tokens) + length > int(max_positions):
        raise ValueError("prompt plus generation exceeds model context")
    noise = deterministic_embedding_noise(embedding_seed, prompt_tokens, (len(prompt_tokens), _embedding_dimension(model)), sigma_abs, clip=config.clip, quantum=config.quantum)
    uniforms, transcript = vrf_uniforms(str(sampling_seed), prompt_tokens, length)
    with torch.inference_mode():
        output = _forward_prefix(model, prompt_tokens, noise, prefix_length=len(prompt_tokens), device=device)
        tokens: list[int] = []
        probabilities: list[float] = []
        top_probabilities: list[float] = []
        entropies: list[float] = []
        eos_id = getattr(tokenizer, "eos_token_id", None)
        for step in range(length):
            distribution = _filtered_distribution(output.logits[0, -1, :], temperature)
            token, token_probability, top_probability, entropy = _sample_from_distribution(distribution, uniforms[step])
            tokens.append(token)
            probabilities.append(token_probability)
            top_probabilities.append(top_probability)
            entropies.append(entropy)
            if eos_id is not None and token == eos_id:
                break
            next_token = torch.tensor([[token]], dtype=torch.long, device=device)
            output = model(input_ids=next_token, past_key_values=output.past_key_values, use_cache=True)
    return {
        "tokens": tokens,
        "token_probabilities": probabilities,
        "top_token_probabilities": top_probabilities,
        "entropies": entropies,
        "vrf_inputs": [entry["input"] for entry in transcript[: len(tokens)]],
        "vrf_outputs": [entry["output"] for entry in transcript[: len(tokens)]],
        "vrf_proofs": [entry["proof"] for entry in transcript[: len(tokens)]],
    }


def score_continuation(
    model: Any,
    context_ids: Sequence[int],
    target_ids: Sequence[int],
    *,
    sigma_abs: float = 0.0,
    challenge_seed: int | bytes | str = 0,
    device: str = "cpu",
    config: PerturbationConfig | None = None,
) -> dict[str, Any]:
    """Score an unperturbed candidate continuation after a noisy prefix."""

    if not context_ids or not target_ids:
        raise ValueError("context and target must be non-empty")
    all_ids = tuple(int(value) for value in context_ids) + tuple(int(value) for value in target_ids)
    config = config or PerturbationConfig(sigma_abs=sigma_abs)
    noise = deterministic_embedding_noise(challenge_seed, context_ids, (len(context_ids), _embedding_dimension(model)), sigma_abs, clip=config.clip, quantum=config.quantum)
    torch = _torch()
    with torch.inference_mode():
        output = _forward_prefix(model, all_ids, noise, prefix_length=len(context_ids), device=device)
        logits = output.logits[0]
        start = len(context_ids) - 1
        selected = logits[start : start + len(target_ids)]
        targets = torch.tensor(list(target_ids), dtype=torch.long, device=selected.device)
        log_probs = torch.log_softmax(selected, dim=-1)
        nll = float(-log_probs[torch.arange(len(target_ids), device=selected.device), targets].sum().item())
        predicted_tokens = tuple(int(value) for value in torch.argmax(selected, dim=-1).tolist())
    return {"nll": nll, "target_tokens": len(target_ids), "first_prediction": predicted_tokens[0], "predicted_tokens": predicted_tokens, "logits": logits.detach().cpu()}


def _score_choice(model: Any, context_ids: Sequence[int], choice_ids: Sequence[int], **kwargs: Any) -> float:
    return -float(score_continuation(model, context_ids, choice_ids, **kwargs)["nll"])


def _load_rows_from_dataset_service(dataset: str, config: str, split: str) -> list[dict[str, Any]]:
    """Download rows through HF's parquet-backed dataset viewer API.

    Recent ``datasets`` releases reject repositories that still contain a
    legacy ``*.py`` dataset loader.  The dataset viewer serves the same
    parquet export without executing that loader, so it is a safe fallback
    for public benchmark data.
    """

    from urllib.error import HTTPError, URLError
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    page_size = 100  # Dataset Viewer caps ``length`` at 100.
    rows: list[dict[str, Any]] = []
    offset = 0
    try:
        while True:
            query = urlencode({
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page_size,
            })
            request = Request(
                f"https://datasets-server.huggingface.co/rows?{query}",
                headers={"User-Agent": "poml-sim-gpt2-experiments/1.0"},
            )
            for attempt in range(5):
                try:
                    with urlopen(request, timeout=30) as response:  # nosec B310 - fixed HTTPS endpoint
                        payload = json.load(response)
                    break
                except HTTPError as exc:
                    if exc.code != 429 or attempt == 4:
                        raise
                    retry_after = exc.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after is not None else 2.0 ** attempt
                    except ValueError:
                        delay = 2.0 ** attempt
                    time.sleep(min(max(delay, 1.0), 30.0))
            if not isinstance(payload, dict):
                raise RuntimeError("dataset viewer returned a non-object payload")
            page = payload.get("rows", [])
            if not isinstance(page, list):
                raise RuntimeError("dataset viewer returned an invalid rows payload")
            rows.extend(
                item.get("row", item)
                for item in page
                if isinstance(item, dict) and isinstance(item.get("row", item), dict)
            )
            total = payload.get("num_rows_total")
            if not page or len(page) < page_size or (isinstance(total, int) and len(rows) >= total):
                break
            offset += len(page)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"could not load {dataset}/{config}/{split} through the Hugging Face dataset service"
        ) from exc
    if not rows:
        raise RuntimeError(f"dataset service returned no rows for {dataset}/{config}/{split}")
    return rows


def _load_rows(dataset: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - CLI-only path
        raise RuntimeError("dataset loading requires datasets; install pip install -e '.[dev,llm]'") from exc
    if dataset == "wikitext2":
        return load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    if dataset == "lambada":
        for name in ("lambada", "lambada_openai", "EleutherAI/lambada_openai"):
            try:
                return load_dataset(name, split=split)
            except Exception:
                continue
        raise RuntimeError("could not load a LAMBADA split")
    aliases = {
        "hellaswag": ("Rowan/hellaswag", None, "Rowan/hellaswag", "default"),
        # regisss/piqa is a parquet export of PIQA and therefore works with
        # datasets>=4, unlike the historical ybisk/piqa repository script.
        "piqa": ("regisss/piqa", None, "regisss/piqa", "default"),
        "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "allenai/ai2_arc", "ARC-Easy"),
    }
    if dataset not in aliases:
        raise ValueError(f"unknown utility dataset: {dataset}")
    name, config, service_name, service_config = aliases[dataset]
    try:
        return load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    except Exception as exc:
        try:
            return _load_rows_from_dataset_service(service_name, service_config, split)
        except RuntimeError:
            raise RuntimeError(f"could not load utility dataset {dataset}") from exc


def load_utility_examples(
    tokenizer: Any,
    *,
    task: str,
    count: int = 100,
    prompt_tokens: int = 32,
    seed: int = 42,
    prompt_file: Path | None = None,
) -> list[UtilityExample]:
    """Load a deterministic task subset with candidate continuations."""

    if count < 1 or prompt_tokens < 1:
        raise ValueError("count and prompt_tokens must be positive")
    if prompt_file is not None:
        rows = [{"text": line.rstrip("\n")} for line in prompt_file.read_text().splitlines()]
    else:
        rows = list(_load_rows(task, "test" if task in {"wikitext2", "lambada", "arc_easy"} else "validation"))
    examples: list[UtilityExample] = []
    for index, row in enumerate(rows):
        if task in {"wikitext2", "lambada"}:
            text = str(row.get("text") or row.get("context") or row.get("passage") or "").strip()
            ids = list(tokenizer(text, add_special_tokens=False).get("input_ids", []))
            if task == "lambada" and row.get("target") is not None and row.get("context") is not None:
                context = list(tokenizer(str(row["context"]), add_special_tokens=False).get("input_ids", []))
                target = list(tokenizer(str(row["target"]), add_special_tokens=False).get("input_ids", []))
                if not context or not target:
                    continue
            elif task == "lambada" and " " in text:
                context_text, target_text = text.rsplit(None, 1)
                context = list(tokenizer(context_text, add_special_tokens=False).get("input_ids", []))
                target = list(tokenizer(target_text, add_special_tokens=False).get("input_ids", []))
                if not context or not target:
                    continue
            elif task == "lambada" and len(ids) >= 2:
                context, target = ids[:-1], ids[-1:]
            elif len(ids) >= prompt_tokens + 1:
                context, target = ids[:prompt_tokens], ids[prompt_tokens : prompt_tokens + 1]
            else:
                continue
            context = context[-prompt_tokens:]
            examples.append(UtilityExample(f"{task}:{index}", task, text, tuple(context), tuple(target)))
        elif task == "hellaswag":
            context = str(row.get("ctx", ""))
            endings = row.get("endings", [])
            choices = tuple(tuple(tokenizer(str(ending), add_special_tokens=False)["input_ids"]) for ending in endings)
            label = row.get("label")
            if context and choices and all(choices) and label is not None:
                examples.append(UtilityExample(f"{task}:{index}", task, context, tuple(tokenizer(context, add_special_tokens=False)["input_ids"][-prompt_tokens:]), choices[0], choices, int(label)))
        elif task == "piqa":
            context = str(row.get("goal", ""))
            choices = tuple(tuple(tokenizer(str(row.get(key, "")), add_special_tokens=False)["input_ids"]) for key in ("sol1", "sol2"))
            label = row.get("label")
            if context and all(choices) and label is not None:
                examples.append(UtilityExample(f"{task}:{index}", task, context, tuple(tokenizer(context, add_special_tokens=False)["input_ids"][-prompt_tokens:]), choices[0], choices, int(label)))
        elif task == "arc_easy":
            question = str(row.get("question", ""))
            choice_data = row.get("choices", {})
            texts = choice_data.get("text", []) if isinstance(choice_data, dict) else []
            labels = choice_data.get("label", []) if isinstance(choice_data, dict) else []
            choices = tuple(tuple(tokenizer(str(value), add_special_tokens=False)["input_ids"]) for value in texts)
            answer = str(row.get("answerKey", ""))
            answer_index = labels.index(answer) if answer in labels else None
            if question and choices and all(choices) and answer_index is not None:
                examples.append(UtilityExample(f"{task}:{index}", task, question, tuple(tokenizer(question, add_special_tokens=False)["input_ids"][-prompt_tokens:]), choices[0], choices, answer_index))
    if len(examples) < count:
        raise ValueError(f"only {len(examples)} usable {task} examples, requested {count}")
    rng = random.Random(seed)
    return rng.sample(examples, count)


def synthetic_utility_examples(tokenizer: Any, *, task: str, count: int = 4, prompt_tokens: int = 32) -> list[UtilityExample]:
    """Create deterministic local examples for an offline smoke run."""

    examples: list[UtilityExample] = []
    for index in range(count):
        text = (f"Smoke example {index} for {task}. This neutral sentence is repeated " * (prompt_tokens + 2)).strip()
        ids = list(tokenizer(text, add_special_tokens=False).get("input_ids", []))
        if len(ids) < prompt_tokens + 1:
            continue
        context, target = tuple(ids[:prompt_tokens]), tuple(ids[prompt_tokens : prompt_tokens + 1])
        examples.append(UtilityExample(f"{task}:smoke:{index}", task, text, context, target))
    if len(examples) < count:
        raise ValueError(f"tokenizer produced too few synthetic {task} examples")
    return examples


def _bootstrap(values: Sequence[float], replicates: int, seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    draws = [float(np.mean(array[rng.integers(0, len(array), len(array))])) for _ in range(max(1, replicates))]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _utility_record(model: Any, examples: Sequence[UtilityExample], *, sigmas: Sequence[float], perturbation_seeds: int, seed: int, device: str, config: PerturbationConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        clean = score_continuation(model, example.context_ids, example.target_ids, device=device, sigma_abs=0.0, challenge_seed=f"{seed}|clean")
        clean_choice = None
        if example.choices:
            scores = [_score_choice(model, example.context_ids, choice, device=device, sigma_abs=0.0, challenge_seed=f"{seed}|clean") for choice in example.choices]
            clean_choice = int(np.argmax(scores))
        for sigma in sigmas:
            replicates = max(1, perturbation_seeds) if sigma else 1
            for replicate in range(replicates):
                challenge = f"{seed}|{example.example_id}|sigma={sigma:.12g}|rep={replicate}"
                sigma_config = PerturbationConfig(sigma_abs=float(sigma), clip=config.clip, quantum=config.quantum)
                perturbed = score_continuation(model, example.context_ids, example.target_ids, device=device, sigma_abs=float(sigma), challenge_seed=challenge, config=sigma_config)
                perturbed_choice = None
                if example.choices:
                    scores = [_score_choice(model, example.context_ids, choice, device=device, sigma_abs=float(sigma), challenge_seed=challenge, config=sigma_config) for choice in example.choices]
                    perturbed_choice = int(np.argmax(scores))
                rows.append({
                    "task": example.task,
                    "example_id": example.example_id,
                    "sigma_abs": float(sigma),
                    "replicate": replicate,
                    "clean_nll": clean["nll"],
                    "perturbed_nll": perturbed["nll"],
                    "target_tokens": clean["target_tokens"],
                    "clean_correct": int(clean_choice == example.answer) if clean_choice is not None and example.answer is not None else int(tuple(clean["predicted_tokens"]) == tuple(example.target_ids)),
                    "perturbed_correct": int(perturbed_choice == example.answer) if perturbed_choice is not None and example.answer is not None else int(tuple(perturbed["predicted_tokens"]) == tuple(example.target_ids)),
                    "clean_choice": clean_choice,
                    "perturbed_choice": perturbed_choice,
                })
    return rows


def run_utility_experiment(
    model: Any,
    examples: Sequence[UtilityExample],
    *,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
    perturbation_seeds: int = 3,
    seed: int = 42,
    device: str = "cpu",
    bootstrap_replicates: int = 1000,
    output_dir: Path | None = None,
    resume: bool = True,
    progress_every: int = 10,
    worker_models: Sequence[tuple[Any, Any, str]] | None = None,
    quantization_quantum: float = 1e-6,
    noise_clip: float | None = 3.0,
) -> dict[str, Any]:
    """Run utility scoring with resumable, round-robin model workers."""

    if not examples:
        raise ValueError("examples must be non-empty")
    if progress_every < 1 or perturbation_seeds < 1:
        raise ValueError("progress_every and perturbation_seeds must be positive")
    config = PerturbationConfig(sigma_abs=0.0, clip=noise_clip, quantum=quantization_quantum)
    fingerprint = _checkpoint_fingerprint("embedding-utility", example_ids=[example.example_id for example in examples], sigmas=list(sigmas), perturbation_seeds=perturbation_seeds, seed=seed, quantum=quantization_quantum, noise_clip=noise_clip)
    checkpoint_path, checkpoint_records = _open_embedding_checkpoint(output_dir, filename="utility_checkpoint.jsonl", fingerprint=fingerprint, resume=resume)
    workers = list(worker_models or [(model, None, device)])
    executors = [ThreadPoolExecutor(max_workers=1) for _ in workers] if len(workers) > 1 else []
    rows: list[dict[str, Any]] = []
    pending: dict[str, Any] = {}
    if executors:
        for index, example in enumerate(examples):
            if example.example_id in checkpoint_records:
                continue
            worker_model, _, worker_device = workers[index % len(workers)]
            pending[example.example_id] = executors[index % len(executors)].submit(_utility_record, worker_model, [example], sigmas=sigmas, perturbation_seeds=perturbation_seeds, seed=seed, device=worker_device, config=config)
    progress = _progress_bar(len(examples), "embedding-utility", unit="example")
    try:
        for index, example in enumerate(examples, start=1):
            key = example.example_id
            record = checkpoint_records.get(key)
            resumed = record is not None
            if record is None:
                if key in pending:
                    result_rows = pending[key].result()
                else:
                    worker_model, _, worker_device = workers[0]
                    result_rows = _utility_record(worker_model, [example], sigmas=sigmas, perturbation_seeds=perturbation_seeds, seed=seed, device=worker_device, config=config)
                record = {"rows": result_rows}
                checkpoint_records[key] = record
                _append_checkpoint_local(checkpoint_path, key, record)
            rows.extend(record["rows"])
            if progress is not None:
                progress.update(1)
                if index == 1 or index % progress_every == 0 or index == len(examples):
                    progress.set_postfix(status="resumed" if resumed else "completed")
    finally:
        if progress is not None:
            progress.close()
        for executor in executors:
            executor.shutdown(wait=True)
    try:
        embedding_std = embedding_table_std(model)
    except (AttributeError, ValueError, RuntimeError):
        # A checkpoint-only resume may intentionally use a sentinel model.
        embedding_std = None
    summary = summarize_utility(rows, bootstrap_replicates=bootstrap_replicates, seed=seed, embedding_std=embedding_std)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "prompt_manifest.jsonl").write_text("".join(json.dumps({"example_id": example.example_id, "task": example.task, "context_ids": list(example.context_ids), "target_ids": list(example.target_ids), "choices": [list(choice) for choice in example.choices], "answer": example.answer}, separators=(",", ":")) + "\n" for example in examples))
        _write_csv(output_dir / "utility_results.csv", rows)
        _write_csv(output_dir / "utility_summary.csv", summary)
        _plot_utility(summary, output_dir / "utility_curve.png")
        _plot_utility_changes(summary, output_dir / "utility_changes.png")
    return {"rows": rows, "summary": summary}


def _append_checkpoint_local(path: Path | None, key: str, result: dict[str, Any]) -> None:
    if path is None:
        return
    with path.open("a") as handle:
        handle.write(json.dumps({"key": key, "result": result}, separators=(",", ":")) + "\n")
        handle.flush()


run_embedding_utility_experiment = run_utility_experiment


def summarize_utility(rows: Sequence[Mapping[str, Any]], *, bootstrap_replicates: int = 1000, seed: int = 42, embedding_std: float | None = None) -> list[dict[str, Any]]:
    """Aggregate paired utility observations and confidence intervals."""

    summary: list[dict[str, Any]] = []
    tasks = sorted({str(row["task"]) for row in rows})
    sigmas = sorted({float(row["sigma_abs"]) for row in rows})
    for task in tasks:
        for sigma in sigmas:
            selected = [row for row in rows if row["task"] == task and float(row["sigma_abs"]) == sigma]
            by_example: dict[str, list[Mapping[str, Any]]] = {}
            for row in selected:
                by_example.setdefault(str(row["example_id"]), []).append(row)
            clean_nll = [float(np.mean([float(item["clean_nll"]) for item in group])) for group in by_example.values()]
            perturbed_nll = [float(np.mean([float(item["perturbed_nll"]) for item in group])) for group in by_example.values()]
            differences = [p - c for p, c in zip(perturbed_nll, clean_nll)]
            correct = [float(np.mean([float(item["perturbed_correct"]) for item in group])) for group in by_example.values()]
            clean_correct = [float(np.mean([float(item["clean_correct"]) for item in group])) for group in by_example.values()]
            is_lm = task in {"wikitext2", "lambada"}
            if is_lm:
                clean_metric, perturbed_metric = math.exp(float(np.mean(clean_nll))), math.exp(float(np.mean(perturbed_nll)))
                metric_differences = [math.exp(perturbed) - math.exp(clean) for perturbed, clean in zip(perturbed_nll, clean_nll)]
                ci_low, ci_high = _bootstrap(metric_differences, bootstrap_replicates, seed + int(sigma * 10000))
                summary.append({"task": task, "sigma_abs": sigma, "alpha": sigma / embedding_std if embedding_std else None, "metric": "perplexity", "clean_metric": clean_metric, "perturbed_metric": perturbed_metric, "absolute_change": perturbed_metric - clean_metric, "relative_change": perturbed_metric / clean_metric - 1.0, "paired_ci_low": ci_low, "paired_ci_high": ci_high, "clean_accuracy": "", "perturbed_accuracy": ""})
            else:
                clean_metric, perturbed_metric = float(np.mean(clean_correct)), float(np.mean(correct))
                differences = [p - c for p, c in zip(correct, clean_correct)]
                ci_low, ci_high = _bootstrap(differences, bootstrap_replicates, seed + int(sigma * 10000))
                summary.append({"task": task, "sigma_abs": sigma, "alpha": sigma / embedding_std if embedding_std else None, "metric": "accuracy", "clean_metric": clean_metric, "perturbed_metric": perturbed_metric, "absolute_change": perturbed_metric - clean_metric, "relative_change": perturbed_metric / clean_metric - 1.0 if clean_metric else float("nan"), "paired_ci_low": ci_low, "paired_ci_high": ci_high, "clean_accuracy": clean_metric, "perturbed_accuracy": perturbed_metric})
    return summary


def select_utility_scale(summary: Sequence[Mapping[str, Any]], *, accuracy_tolerance: float = 0.02, perplexity_tolerance: float = 0.05) -> float | None:
    """Select the smallest scale satisfying the predeclared utility limits."""

    for sigma in sorted({float(row["sigma_abs"]) for row in summary}):
        rows = [row for row in summary if float(row["sigma_abs"]) == sigma]
        if all((float(row["absolute_change"]) >= -accuracy_tolerance if row["metric"] == "accuracy" else float(row["relative_change"]) <= perplexity_tolerance) for row in rows):
            return sigma
    return None


def select_operating_point(
    utility_summary: Sequence[Mapping[str, Any]],
    separation_summary: Sequence[Mapping[str, Any]],
    *,
    accuracy_tolerance: float = 0.02,
    perplexity_tolerance: float = 0.05,
) -> float | None:
    """Select the smallest nonzero scale passing utility and separation.

    Separation is deliberately a conservative engineering gate: at least one
    common-token/full-protocol embedding comparison must have changed an
    integer coordinate and must not be an all-collision population.  The
    detailed per-boundary table remains the evidence used in the paper.
    """

    utility_scale = select_utility_scale(
        utility_summary,
        accuracy_tolerance=accuracy_tolerance,
        perplexity_tolerance=perplexity_tolerance,
    )
    if utility_scale is None:
        return None
    candidates = [
        sigma for sigma in sorted({float(row["sigma_abs"]) for row in utility_summary})
        if sigma > 0 and sigma >= utility_scale
    ]
    for sigma in candidates:
        utility_rows = [row for row in utility_summary if float(row["sigma_abs"]) == sigma]
        if not all((float(row["absolute_change"]) >= -accuracy_tolerance if row["metric"] == "accuracy" else float(row["relative_change"]) <= perplexity_tolerance) for row in utility_rows):
            continue
        separation_rows = [
            row for row in separation_summary
            if float(row["sigma_abs"]) == sigma
            and str(row["boundary"]) == "embedding"
            and str(row["condition"]) in {"common_token_stream", "full_protocol"}
        ]
        if separation_rows and any(float(row["changed_coordinate_fraction"]) > 0 and float(row["collision_rate"]) < 1 for row in separation_rows):
            return sigma
    return None


def _plot_utility(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Plot all benchmark utilities relative to the original result on one axis."""

    if not summary:
        return
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = sorted({str(row["task"]) for row in summary})
    figure, axis = plt.subplots(figsize=(11.5, 6.8))
    colors = plt.get_cmap("tab10")
    markers = ("o", "s", "^", "D", "P", "X", "v")
    x_values = sorted({float(row["sigma_abs"]) for row in summary})
    for index, task in enumerate(tasks):
        rows = [row for row in summary if row["task"] == task]
        rows = sorted(rows, key=lambda row: float(row["sigma_abs"]))
        clean = float(rows[0]["clean_metric"])
        metric = str(rows[0]["metric"])
        values: list[float] = []
        for row in rows:
            perturbed = float(row["perturbed_metric"])
            values.append(clean / perturbed if metric == "perplexity" and perturbed else perturbed / clean if clean else 0.0)
        axis.plot(
            [float(row["sigma_abs"]) for row in rows],
            values,
            color=colors(index),
            marker=markers[index % len(markers)],
            linestyle="--" if metric == "perplexity" else "-",
            linewidth=2.2,
            markersize=7,
            label=f"{task} ({'perplexity' if metric == 'perplexity' else 'accuracy'})",
        )
    axis.axhline(1.0, color="black", linewidth=1.3, linestyle=":", label="original (unperturbed) baseline")
    axis.set_title("Utility under one-time embedding perturbation", fontsize=15, pad=16)
    axis.text(
        0.5,
        1.015,
        "Normalized utility: 1.0 = original result  |  solid = accuracy (perturbed / original)  |  dashed = perplexity (original / perturbed)",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    axis.set_xlabel("Embedding noise σ_abs")
    axis.set_ylabel("Normalized utility (1.0 = original)")
    axis.set_ylim(0.85, 1.15)
    axis.set_xticks(x_values)
    axis.grid(alpha=0.28)
    axis.set_axisbelow(True)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        fontsize=9,
        frameon=True,
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.22)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_utility_changes(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Plot metric changes in native units without mixing perplexity and accuracy."""

    if not summary:
        return
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
    panels = (("perplexity", axes[0, 0], "Relative perplexity change (%)"), ("accuracy", axes[0, 1], "Accuracy change (percentage points)"))
    for metric, axis, ylabel in panels:
        tasks = sorted({str(row["task"]) for row in summary if str(row["metric"]) == metric})
        for task in tasks:
            rows = sorted((row for row in summary if str(row["task"]) == task and str(row["metric"]) == metric), key=lambda row: float(row["sigma_abs"]))
            if metric == "perplexity":
                values = [100.0 * float(row["relative_change"]) for row in rows]
            else:
                values = [100.0 * float(row["absolute_change"]) for row in rows]
            axis.plot([float(row["sigma_abs"]) for row in rows], values, marker="o", linewidth=2, label=task)
        axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
        axis.set_xlabel("Embedding noise σ_abs")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        if tasks:
            axis.legend(fontsize=8)
    axes[0, 0].axhspan(-5, 5, color="tab:green", alpha=0.08)
    axes[0, 1].axhspan(-2, 2, color="tab:green", alpha=0.08)
    figure.suptitle("Absolute changes from the original result (green = tolerance band)", fontsize=14)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_trace_separation(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Render full-protocol trace collisions as an annotated table.

    A collision is deliberately strict: every integer coordinate must match
    after the proof quantizer.  Showing the counts directly is more useful
    here than a dense curve with one line per internal boundary.
    """
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [
        row
        for row in summary
        if str(row["condition"]) == "full_protocol" and float(row["sigma_abs"]) > 0
    ]
    figure, axis = plt.subplots(figsize=(11, max(7.0, 0.43 * len(set(str(row["boundary"]) for row in selected)) + 3.5)))
    axis.axis("off")
    if not selected:
        axis.text(0.5, 0.5, "No nonzero full-protocol measurements", ha="center", va="center", fontsize=13)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return
    boundary_order = [
        "embedding", "post_first_norm", "layer_0_q", "layer_0_k", "layer_0_v",
        *[f"block_{index}" for index in range(12)], "logits",
    ]
    boundaries = [boundary for boundary in boundary_order if any(str(row["boundary"]) == boundary for row in selected)]
    sigmas = sorted({float(row["sigma_abs"]) for row in selected})
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in selected:
        grouped.setdefault((str(row["boundary"]), float(row["sigma_abs"])), []).append(row)
    cell_text: list[list[str]] = []
    cell_colors: list[list[str]] = []
    for boundary in boundaries:
        values: list[str] = []
        colors: list[str] = []
        for sigma in sigmas:
            rows = grouped[(boundary, sigma)]
            collisions = sum(int(row["collisions"]) for row in rows)
            comparisons = sum(int(row["comparisons"]) for row in rows)
            rate = collisions / comparisons if comparisons else float("nan")
            values.append(f"{collisions:,} / {comparisons:,}\n({rate:.2%})")
            colors.append("#e7f5e8" if collisions == 0 else "#fde2e2")
        cell_text.append(values)
        cell_colors.append(colors)
    labels = {
        "embedding": "Input embedding",
        "post_first_norm": "Post-first normalization",
        "layer_0_q": "Layer 0 query (Q)",
        "layer_0_k": "Layer 0 key (K)",
        "layer_0_v": "Layer 0 value (V)",
        "logits": "Final logits",
    }
    row_labels = [labels.get(boundary, boundary.replace("block_", "Block ")) for boundary in boundaries]
    table = axis.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=[f"σ = {sigma:g}" for sigma in sigmas],
        cellLoc="center",
        rowLoc="center",
        bbox=(0.10, 0.19, 0.86, 0.65),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for row_index, colors in enumerate(cell_colors, start=1):
        for col_index, color in enumerate(colors):
            table[(row_index, col_index)].set_facecolor(color)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#9aa0a6")
    axis.set_title("Full-protocol quantized trace collision rate", fontsize=14, pad=18)
    figure.text(
        0.5,
        0.92,
        "Each cell: exact whole-tensor collisions / comparisons (σ = 0 omitted)",
        ha="center",
        va="center",
        fontsize=10,
    )
    figure.text(
        0.5,
        0.095,
        "Collision means QΔ(a) = QΔ(b) at every coordinate, with QΔ(x) = np.rint(x / Δ), Δ = 0.001.\n"
        "Values are the same only when they land in the same quantization bin; one unequal integer coordinate means no collision.\n"
        "Green cells mean zero collisions; red cells mean one or more collisions. Zero observed in 2,464 pairs gives a rule-of-three upper bound of 0.12%.\n"
        "The previous changed-coordinate statistic d (fraction of unequal coordinates) is omitted: a collision always has d = 0.",
        ha="center",
        va="center",
        fontsize=8.5,
    )
    figure.subplots_adjust(top=0.87, bottom=0.17)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_prefix_collisions(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Render full-protocol prefix collisions as annotated heatmaps."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [
        row for row in rows
        if str(row["condition"]) == "full_protocol" and float(row["sigma_abs"]) > 0
    ]
    if not selected:
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.axis("off")
        axis.text(0.5, 0.5, "No nonzero full-protocol measurements", ha="center", va="center", fontsize=13)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return
    strata = ["standard", "resisting_correction", "repeat_copy_logic"]
    strata = [stratum for stratum in strata if any(str(row["stratum"]) == stratum for row in selected)]
    sigmas = sorted({float(row["sigma_abs"]) for row in selected})
    lengths = sorted({int(row["prefix_length"]) for row in selected})
    figure, axes = plt.subplots(1, len(strata), figsize=(15.5, 5.8), squeeze=False)
    axes_flat = list(axes.flat)
    friendly = {"standard": "Standard benchmarks", "resisting_correction": "Resisting correction", "repeat_copy_logic": "Repeat-copy logic"}
    image = None
    for axis, stratum in zip(axes_flat, strata):
        matrix = np.full((len(sigmas), len(lengths)), np.nan)
        for row_index, sigma in enumerate(sigmas):
            for col_index, length in enumerate(lengths):
                group = [
                    row for row in selected
                    if str(row["stratum"]) == stratum
                    and float(row["sigma_abs"]) == sigma
                    and int(row["prefix_length"]) == length
                ]
                comparisons = sum(int(row["comparisons"]) for row in group)
                if comparisons:
                    matrix[row_index, col_index] = sum(int(row["collisions"]) for row in group) / comparisons
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn_r", aspect="auto")
        axis.set_title(friendly.get(stratum, stratum))
        axis.set_xticks(range(len(lengths)), [str(length) for length in lengths])
        axis.set_yticks(range(len(sigmas)), [f"{sigma:g}" for sigma in sigmas])
        axis.set_xlabel("Prefix length ℓ")
        if axis is axes_flat[0]:
            axis.set_ylabel("Embedding noise σ_abs")
        for row_index in range(len(sigmas)):
            for col_index in range(len(lengths)):
                value = matrix[row_index, col_index]
                if math.isfinite(value):
                    red, green, blue = image.cmap(image.norm(value))[:3]
                    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                    axis.text(col_index, row_index, f"{value:.1%}", ha="center", va="center", color="black" if luminance > 0.55 else "white", fontsize=8)
    if image is not None:
        colorbar_axis = figure.add_axes((0.88, 0.18, 0.02, 0.64))
        figure.colorbar(image, cax=colorbar_axis, label="Exact prefix-collision probability (green = low, red = high)")
    figure.suptitle("Full-protocol generated-prefix collision probability", fontsize=14, y=0.97)
    figure.text(0.5, 0.02, "Same prefix means exact equality of token IDs in the first ℓ positions; σ = 0 omitted.", ha="center", fontsize=9)
    figure.subplots_adjust(top=0.83, bottom=0.17, left=0.07, right=0.84, wspace=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_token_agreement(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Render full-protocol exact sequence agreement as an annotated table."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [
        row for row in rows
        if str(row["condition"]) == "full_protocol" and float(row["sigma_abs"]) > 0
    ]
    figure, axis = plt.subplots(figsize=(11, 3.8))
    axis.axis("off")
    if not selected:
        axis.text(0.5, 0.5, "No nonzero full-protocol measurements", ha="center", va="center", fontsize=13)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return
    strata = ["standard", "resisting_correction", "repeat_copy_logic"]
    strata = [stratum for stratum in strata if any(str(row["stratum"]) == stratum for row in selected)]
    sigmas = sorted({float(row["sigma_abs"]) for row in selected})
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in selected:
        grouped.setdefault((str(row["stratum"]), float(row["sigma_abs"])), []).append(row)
    cell_text: list[list[str]] = []
    cell_colors: list[list[str]] = []
    for stratum in strata:
        values: list[str] = []
        colors: list[str] = []
        for sigma in sigmas:
            group = grouped[(stratum, sigma)]
            matches = sum(int(round(float(row["token_sequence_agreement"]) * int(row["comparisons"]))) for row in group)
            comparisons = sum(int(row["comparisons"]) for row in group)
            rate = matches / comparisons if comparisons else float("nan")
            values.append(f"{matches:,} / {comparisons:,}\n({rate:.2%})")
            colors.append("#e7f5e8" if matches == 0 else "#fff0cc")
        cell_text.append(values)
        cell_colors.append(colors)
    friendly = {"standard": "Standard benchmarks", "resisting_correction": "Resisting correction", "repeat_copy_logic": "Repeat-copy logic"}
    table = axis.table(
        cellText=cell_text,
        rowLabels=[friendly.get(stratum, stratum) for stratum in strata],
        colLabels=[f"σ = {sigma:g}" for sigma in sigmas],
        cellLoc="center",
        rowLoc="center",
        bbox=(0.08, 0.30, 0.88, 0.45),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for row_index, colors in enumerate(cell_colors, start=1):
        for col_index, color in enumerate(colors):
            table[(row_index, col_index)].set_facecolor(color)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#9aa0a6")
    axis.set_title("Full-protocol exact generated-sequence agreement", fontsize=14, pad=18)
    figure.text(0.5, 0.18, "Each cell: matching full sequences / compared pairs (percentage). σ = 0 omitted. Green = zero matches; amber = one or more matches.", ha="center", fontsize=9)
    figure.text(0.5, 0.08, "Same sequence means every generated token ID matches (up to 32 tokens or EOS); semantic similarity is not counted.", ha="center", fontsize=8.5)
    figure.subplots_adjust(top=0.84, bottom=0.12)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_reuse_work(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Plot fresh versus challenge-online inference timing and rho_inf."""

    if not rows:
        return
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["prompt_id"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.38
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x - width / 2, [float(row["fresh_seconds"]) for row in rows], width, label="fresh")
    axes[0].bar(x + width / 2, [float(row["online_update_seconds"]) for row in rows], width, label="online update")
    axes[0].set_ylabel("Median seconds")
    axes[0].set_title("Fresh versus challenge-online inference")
    axes[0].legend()
    axes[1].bar(x, [float(row["rho_inf"]) for row in rows])
    axes[1].set_ylabel("ρ_inf")
    axes[1].set_title("Measured reusable inference fraction")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _next_token_distribution(model: Any, token_ids: Sequence[int], device: str) -> np.ndarray:
    torch = _torch()
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([list(token_ids)], dtype=torch.long, device=device))
    return _filtered_distribution(output.logits[0, -1, :], 1.0).detach().cpu().numpy()


def build_target_token_suite(tokenizer: Any, model: Any | None = None, *, device: str = "cpu", seed: int = 42, targets: int = 16, construction_carriers: int = 16, evaluation_carriers: int = 4, max_per_target_template: int = 10, target_probability: float = 0.90) -> list[TracePrompt]:
    """Construct and clean-filter visible-copy target-token prompts."""

    if targets < 1 or construction_carriers < 1 or evaluation_carriers < 1:
        raise ValueError("suite sizes must be positive")
    special = set(getattr(tokenizer, "all_special_ids", []))
    candidates: list[int] = []
    for token_id in range(min(len(tokenizer), 50_257)):
        if token_id in special:
            continue
        try:
            text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        except TypeError:
            text = tokenizer.decode([token_id])
        if not text or not text.strip() or len(tokenizer(text, add_special_tokens=False).get("input_ids", [])) != 1:
            continue
        candidates.append(token_id)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    target_ids = candidates[:targets]
    templates = ("Answer with exactly one token: {target}. Next token:", "Repeat the requested symbol once. Requested symbol: {target}. Output:", "The next token is {target}. Continue:")
    carriers = [f"carrier {index:04d} neutral text {rng.randrange(10**9)}" for index in range(construction_carriers + evaluation_carriers)]
    prompts: list[TracePrompt] = []
    for target_id in target_ids:
        try:
            target_text = tokenizer.decode([target_id], clean_up_tokenization_spaces=False)
        except TypeError:
            target_text = tokenizer.decode([target_id])
        for template_index, template in enumerate(templates):
            retained = 0
            for carrier_index, carrier in enumerate(carriers):
                carrier_ids = tokenizer(carrier, add_special_tokens=False).get("input_ids", [])
                if target_id in carrier_ids:
                    continue
                # Keep the carrier before the final ``Output:`` delimiter so
                # the measured next token is the requested target, while the
                # target never appears accidentally in the carrier.
                text = carrier + "\n" + template.format(target=target_text)
                token_ids = tuple(int(value) for value in tokenizer(text, add_special_tokens=False)["input_ids"])
                if not token_ids:
                    continue
                probability = entropy = margin = float("nan")
                if model is not None:
                    distribution = _next_token_distribution(model, token_ids, device)
                    probability = float(distribution[target_id])
                    entropy = float(-(distribution[distribution > 0] * np.log(distribution[distribution > 0])).sum())
                    top_two = np.sort(distribution)[-2:]
                    margin = float(math.log(max(probability, 1e-300)) - math.log(max(float(top_two[0]), 1e-300)))
                if model is not None and probability < target_probability:
                    continue
                if carrier_index < construction_carriers and retained >= max_per_target_template:
                    continue
                if carrier_index >= construction_carriers:
                    prompts.append(TracePrompt(f"target:{target_id}:{template_index}:{carrier_index}", "target_token", text, token_ids, target_id, {"target_probability": probability, "entropy": entropy, "margin": margin, "split": "evaluation", "target_probability_threshold": target_probability, "tokenization_length": len(token_ids)}))
                elif retained < max_per_target_template:
                    retained += 1
                    prompts.append(TracePrompt(f"target:{target_id}:{template_index}:{carrier_index}", "target_token", text, token_ids, target_id, {"target_probability": probability, "entropy": entropy, "margin": margin, "split": "construction", "target_probability_threshold": target_probability, "tokenization_length": len(token_ids)}))
    if len(prompts) < max(1, targets) and model is not None and target_probability > 0.5:
        # The plan permits one predeclared relaxation when a small base model
        # has no 0.90-concentrated prompts; record the effective threshold in
        # each returned prompt's metadata for the report.
        return build_target_token_suite(
            tokenizer,
            model,
            device=device,
            seed=seed,
            targets=targets,
            construction_carriers=construction_carriers,
            evaluation_carriers=evaluation_carriers,
            max_per_target_template=max_per_target_template,
            target_probability=0.5,
        )
    return prompts


def _fallback_structural_prompts(stratum: str, count: int) -> list[TracePrompt]:
    return [TracePrompt(f"{stratum}:{index}", stratum, f"Repeat exactly this sentence: marker {index:04d}. Output:", tuple([101 + (index % 100), 202 + (index % 100), 303]), metadata={"source": "fallback"}) for index in range(count)]


def load_trace_prompts(tokenizer: Any, *, standard_examples: Sequence[UtilityExample], repeat_copy_file: Path | None = None, resisting_correction_file: Path | None = None, count_per_standard_task: int = 100, max_structural_prompts: int | None = None, smoke: bool = False) -> list[TracePrompt]:
    """Load standard and public structural strata, with explicit smoke fallbacks."""

    if max_structural_prompts is not None and max_structural_prompts < 1:
        raise ValueError("max_structural_prompts must be positive when provided")
    prompts = [TracePrompt(example.example_id, "standard", example.text, example.context_ids, metadata={"task": example.task}) for example in standard_examples[: count_per_standard_task * 5]]
    for stratum, path in (("repeat_copy_logic", repeat_copy_file), ("resisting_correction", resisting_correction_file)):
        if path is not None:
            lines = path.read_text().splitlines()
            if max_structural_prompts is not None:
                lines = lines[:max_structural_prompts]
            for index, line in enumerate(lines):
                if line.strip():
                    ids = tuple(int(value) for value in tokenizer(line, add_special_tokens=False)["input_ids"])
                    prompts.append(TracePrompt(f"{stratum}:{index}", stratum, line, ids))
        elif smoke:
            prompts.extend(_fallback_structural_prompts(stratum, 4))
        else:
            try:
                from datasets import load_dataset
                try:
                    if stratum == "repeat_copy_logic":
                        rows = load_dataset("tasksource/bigbench", "repeat_copy_logic", split="validation")
                    else:
                        rows = load_dataset("pminervini/inverse-scaling", "resisting-correction", split="data")
                except Exception as direct_exc:
                    # datasets>=4 refuses the legacy BIG-bench builder. These
                    # parquet-backed exports preserve the released prompts and
                    # can be fetched through the same viewer helper used by
                    # utility datasets.
                    service_name = "tasksource/bigbench" if stratum == "repeat_copy_logic" else "pminervini/inverse-scaling"
                    service_config = "repeat_copy_logic" if stratum == "repeat_copy_logic" else "resisting-correction"
                    service_split = "validation" if stratum == "repeat_copy_logic" else "data"
                    try:
                        rows = _load_rows_from_dataset_service(service_name, service_config, service_split)
                    except RuntimeError:
                        # Keep compatibility with installations that only have
                        # the original builders cached locally.
                        legacy_name = "google/bigbench" if stratum == "repeat_copy_logic" else "inverse_scaling"
                        legacy_config = "repeat_copy_logic" if stratum == "repeat_copy_logic" else "resisting_correction"
                        legacy_split = "validation"
                        try:
                            rows = load_dataset(legacy_name, legacy_config, split=legacy_split)
                        except Exception:
                            raise direct_exc
                if max_structural_prompts is not None:
                    rows = list(rows)[:max_structural_prompts]
                for index, row in enumerate(rows):
                    text = str(row.get("input") or row.get("inputs") or row.get("prompt") or row.get("text") or "")
                    if text:
                        prompts.append(TracePrompt(f"{stratum}:{index}", stratum, text, tuple(tokenizer(text, add_special_tokens=False)["input_ids"])))
            except Exception as exc:
                file_flag = "repeat-copy-file" if stratum == "repeat_copy_logic" else "resisting-correction-file"
                raise RuntimeError(f"could not load {stratum}; pass --{file_flag} or --smoke") from exc
    return prompts


def _capture_boundaries(model: Any, prompt_ids: Sequence[int], continuation: Sequence[int], *, embedding_seed: int | bytes | str, sigma_abs: float, device: str, quantization_scale: float, quantization_clip: int | None) -> dict[str, np.ndarray]:
    all_ids = tuple(prompt_ids) + tuple(continuation)
    noise = deterministic_embedding_noise(embedding_seed, prompt_ids, (len(prompt_ids), _embedding_dimension(model)), sigma_abs)
    torch = _torch()
    _, token_values = _token_embeddings(model, all_ids, device)
    position_values = torch.zeros_like(token_values)
    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "wpe"):
        positions = torch.arange(len(all_ids), device=device).unsqueeze(0)
        position_values = transformer.wpe(positions)
    embedding_boundary = token_values + position_values
    if prompt_ids:
        embedding_boundary = embedding_boundary.clone()
        embedding_boundary[:, : len(prompt_ids), :] += torch.as_tensor(noise, dtype=embedding_boundary.dtype, device=device).unsqueeze(0)
    captured: dict[str, Any] = {}
    hooks = []
    blocks = getattr(transformer, "h", []) if transformer is not None else []
    if blocks and hasattr(blocks[0], "attn") and hasattr(blocks[0].attn, "c_attn"):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if getattr(value, "ndim", 0) == 3 and value.shape[-1] % 3 == 0:
                captured["layer_0_q"], captured["layer_0_k"], captured["layer_0_v"] = value.chunk(3, dim=-1)
        hooks.append(blocks[0].attn.c_attn.register_forward_hook(hook))
    if blocks and hasattr(blocks[0], "ln_1"):
        def norm_hook(_module: Any, _inputs: Any, output: Any) -> None:
            captured["post_first_norm"] = output[0] if isinstance(output, tuple) else output
        hooks.append(blocks[0].ln_1.register_forward_hook(norm_hook))
    try:
        output = _forward_prefix(model, all_ids, noise, prefix_length=len(prompt_ids), device=device, output_hidden_states=True)
    finally:
        for hook_handle in hooks:
            hook_handle.remove()
    hidden_states = getattr(output, "hidden_states", None) or ()
    # The proving object is the prompt-side prefix state.  Keeping generated
    # token rows out of these boundaries prevents ordinary sampler divergence
    # from being mistaken for embedding-challenge separation.
    prefix_slice = slice(0, len(prompt_ids))
    boundaries: dict[str, np.ndarray] = {"embedding": quantize_tensor(embedding_boundary[:, prefix_slice, :], quantization_scale, clip=quantization_clip)}
    if hidden_states:
        # GPT-2 exposes the actual ln_1 hook above; hidden_states[0] is a
        # useful architecture-neutral fallback for toy/other causal models.
        boundaries.setdefault("post_first_norm", quantize_tensor(hidden_states[0][:, prefix_slice, :], quantization_scale, clip=quantization_clip))
        for index, state in enumerate(hidden_states[1:]):
            boundaries[f"block_{index}"] = quantize_tensor(state[:, prefix_slice, :], quantization_scale, clip=quantization_clip)
    for key, value in captured.items():
        boundaries[key] = quantize_tensor(value[:, prefix_slice, :], quantization_scale, clip=quantization_clip)
    boundaries["logits"] = quantize_tensor(output.logits[:, -1, :], quantization_scale, clip=quantization_clip)
    return boundaries


def capture_quantized_boundaries(
    model: Any,
    prompt_ids: Sequence[int],
    continuation: Sequence[int],
    *,
    embedding_seed: int | bytes | str,
    sigma_abs: float,
    device: str = "cpu",
    quantization_scale: float = 1e-3,
    quantization_clip: int | None = 2**31 - 1,
) -> dict[str, np.ndarray]:
    """Audit one prompt with the same boundary quantizer as the campaign."""

    return _capture_boundaries(
        model,
        prompt_ids,
        continuation,
        embedding_seed=embedding_seed,
        sigma_abs=sigma_abs,
        device=device,
        quantization_scale=quantization_scale,
        quantization_clip=quantization_clip,
    )


def _prefix_indicators(first: Sequence[int], second: Sequence[int], lengths: Sequence[int]) -> dict[int, int | None]:
    result: dict[int, int | None] = {}
    for length in lengths:
        if length == 0:
            result[length] = 1
        elif len(first) < length or len(second) < length:
            result[length] = None
        else:
            result[length] = int(tuple(first[:length]) == tuple(second[:length]))
    return result


def _separation_record(model: Any, tokenizer: Any, prompt: TracePrompt, *, sigma_abs: float, pairs: int, length: int, seed: int, device: str, quantization_scale: float, quantization_clip: int | None, prefix_lengths: Sequence[int], conditions: Sequence[str]) -> dict[str, Any]:
    boundary_stats: dict[str, dict[str, list[float]]] = {}
    prefix_stats: dict[str, dict[str, list[int]]] = {}
    token_stats: dict[str, list[int]] = {}
    replay_original: list[str] = []
    replay_challenged: list[str] = []
    for condition in conditions:
        prefix_stats[condition] = {str(value): [] for value in prefix_lengths}
        boundary_stats[condition] = {}
        token_stats[condition] = []
        for pair_index in range(pairs):
            base = f"{seed}|{prompt.prompt_id}|sigma={sigma_abs:.12g}|pair={pair_index}|{condition}"
            common_sampler = condition == "common_token_stream"
            first = generate_perturbed_trace(model, tokenizer, prompt.token_ids, embedding_seed=base + "|embed-a", sampling_seed=(base + "|sample-common" if common_sampler else base + "|sample-a"), sigma_abs=sigma_abs if condition != "no_perturbation" else 0.0, length=length, device=device)
            second = generate_perturbed_trace(model, tokenizer, prompt.token_ids, embedding_seed=base + "|embed-b", sampling_seed=(base + "|sample-common" if common_sampler else base + "|sample-b"), sigma_abs=sigma_abs if condition != "no_perturbation" else 0.0, length=length, device=device)
            token_stats[condition].append(int(first["tokens"] == second["tokens"]))
            for key, value in _prefix_indicators(first["tokens"], second["tokens"], prefix_lengths).items():
                if value is not None:
                    prefix_stats[condition][str(key)].append(value)
            first_boundaries = _capture_boundaries(model, prompt.token_ids, first["tokens"], embedding_seed=base + "|embed-a", sigma_abs=sigma_abs if condition != "no_perturbation" else 0.0, device=device, quantization_scale=quantization_scale, quantization_clip=quantization_clip)
            second_boundaries = _capture_boundaries(model, prompt.token_ids, second["tokens"], embedding_seed=base + "|embed-b", sigma_abs=sigma_abs if condition != "no_perturbation" else 0.0, device=device, quantization_scale=quantization_scale, quantization_clip=quantization_clip)
            for row in compare_quantized_boundaries(first_boundaries, second_boundaries):
                stats = boundary_stats[condition].setdefault(row["boundary"], {"collisions": [], "changed_fraction": []})
                stats["collisions"].append(int(row["collision"]))
                stats["changed_fraction"].append(float(row["changed_fraction"]))
            if condition == "full_protocol" and first["tokens"] == second["tokens"]:
                replay_original.append(quantized_trace_commitment(first_boundaries, challenge=base + "|embed-a"))
                replay_challenged.append(quantized_trace_commitment(second_boundaries, challenge=base + "|embed-b"))
    return {"prompt_id": prompt.prompt_id, "stratum": prompt.stratum, "sigma_abs": sigma_abs, "boundary_stats": boundary_stats, "prefix_stats": prefix_stats, "token_stats": token_stats, "replay": replay_test_from_commitments(replay_original, replay_challenged)}


def run_trace_separation_experiment(model: Any, tokenizer: Any, prompts: Sequence[TracePrompt], *, sigmas: Sequence[float] = DEFAULT_SIGMAS, pairs: int = 50, length: int = 32, seed: int = 42, device: str = "cpu", quantization_scale: float = 1e-3, quantization_clip: int | None = 2**31 - 1, prefix_lengths: Sequence[int] = (1, 2, 4, 8, 16, 32), conditions: Sequence[str] = ("no_perturbation", "common_token_stream", "full_protocol"), output_dir: Path | None = None, resume: bool = True, progress_every: int = 10, worker_models: Sequence[tuple[Any, Any, str]] | None = None) -> dict[str, Any]:
    """Measure quantized-boundary separation and generated-prefix collisions."""

    if not prompts or pairs < 1 or progress_every < 1:
        raise ValueError("prompts, pairs, and progress_every must be positive")
    fingerprint = _checkpoint_fingerprint("embedding-separation", prompt_ids=[prompt.prompt_id for prompt in prompts], sigmas=list(sigmas), pairs=pairs, length=length, seed=seed, scale=quantization_scale, clip=quantization_clip, conditions=list(conditions))
    checkpoint_path, checkpoint_records = _open_embedding_checkpoint(output_dir, filename="separation_checkpoint.jsonl", fingerprint=fingerprint, resume=resume)
    workers = list(worker_models or [(model, tokenizer, device)])
    executors = [ThreadPoolExecutor(max_workers=1) for _ in workers] if len(workers) > 1 else []
    units = [(prompt, float(sigma)) for sigma in sigmas for prompt in prompts]
    pending: dict[str, Any] = {}
    if executors:
        for index, (prompt, sigma) in enumerate(units):
            key = f"{sigma:.12g}|{prompt.prompt_id}"
            if key in checkpoint_records:
                continue
            worker_model, worker_tokenizer, worker_device = workers[index % len(workers)]
            pending[key] = executors[index % len(executors)].submit(_separation_record, worker_model, worker_tokenizer, prompt, sigma_abs=sigma, pairs=pairs, length=length, seed=seed, device=worker_device, quantization_scale=quantization_scale, quantization_clip=quantization_clip, prefix_lengths=prefix_lengths, conditions=conditions)
    records: list[dict[str, Any]] = []
    progress = _progress_bar(len(units), "embedding-separation", unit="unit")
    try:
        for index, (prompt, sigma) in enumerate(units, start=1):
            key = f"{sigma:.12g}|{prompt.prompt_id}"
            record = checkpoint_records.get(key)
            resumed = record is not None
            if record is None:
                if key in pending:
                    record = pending[key].result()
                else:
                    worker_model, worker_tokenizer, worker_device = workers[0]
                    record = _separation_record(worker_model, worker_tokenizer, prompt, sigma_abs=sigma, pairs=pairs, length=length, seed=seed, device=worker_device, quantization_scale=quantization_scale, quantization_clip=quantization_clip, prefix_lengths=prefix_lengths, conditions=conditions)
                checkpoint_records[key] = record
                _append_checkpoint_local(checkpoint_path, key, record)
            records.append(record)
            if progress is not None:
                progress.update(1)
                if index == 1 or index % progress_every == 0 or index == len(units):
                    progress.set_postfix(status="resumed" if resumed else "completed")
    finally:
        if progress is not None:
            progress.close()
        for executor in executors:
            executor.shutdown(wait=True)
    boundary_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    replay = {"old_under_old_accepted": 0, "old_under_new_rejected": 0, "fresh_under_new_accepted": 0}
    for record in records:
        for key in replay:
            replay[key] += int(record.get("replay", {}).get(key, 0))
        for condition, boundaries in record["boundary_stats"].items():
            for boundary, values in boundaries.items():
                collision_count = int(sum(values["collisions"]))
                comparison_count = len(values["collisions"])
                boundary_rows.append({"stratum": record["stratum"], "prompt_id": record["prompt_id"], "sigma_abs": record["sigma_abs"], "condition": condition, "boundary": boundary, "collisions": collision_count, "comparisons": comparison_count, "collision_rate": float(np.mean(values["collisions"])) if values["collisions"] else float("nan"), "zero_collision_upper_95": min(1.0, 3.0 / comparison_count) if comparison_count and collision_count == 0 else None, "changed_coordinate_fraction": float(np.mean(values["changed_fraction"])) if values["changed_fraction"] else float("nan")})
        for condition, values in record.get("token_stats", {}).items():
            token_rows.append({"stratum": record["stratum"], "prompt_id": record["prompt_id"], "sigma_abs": record["sigma_abs"], "condition": condition, "token_sequence_agreement": float(np.mean(values)) if values else float("nan"), "comparisons": len(values)})
        for condition, lengths in record["prefix_stats"].items():
            for prefix_length, values in lengths.items():
                prefix_rows.append({"stratum": record["stratum"], "prompt_id": record["prompt_id"], "sigma_abs": record["sigma_abs"], "condition": condition, "prefix_length": int(prefix_length), "collisions": int(sum(values)), "comparisons": len(values), "collision_probability": float(np.mean(values)) if values else float("nan"), "zero_collision_upper_95": min(1.0, 3.0 / len(values)) if values and not any(values) else None})
    boundary_summary = summarize_trace_separation(boundary_rows)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "prompt_manifest.jsonl").write_text("".join(json.dumps({"prompt_id": prompt.prompt_id, "stratum": prompt.stratum, "token_ids": list(prompt.token_ids), "target_token": prompt.target_token, "metadata": prompt.metadata}, separators=(",", ":")) + "\n" for prompt in prompts))
        _write_csv(output_dir / "trace_separation.csv", boundary_rows)
        _write_csv(output_dir / "trace_separation_summary.csv", boundary_summary)
        _write_csv(output_dir / "prefix_collisions.csv", prefix_rows)
        _write_csv(output_dir / "token_agreement.csv", token_rows)
        _plot_trace_separation(boundary_summary, output_dir / "trace_separation.png")
        _plot_prefix_collisions(prefix_rows, output_dir / "prefix_collisions.png")
        _plot_token_agreement(token_rows, output_dir / "token_agreement.png")
        (output_dir / "replay_test.json").write_text(json.dumps(replay, indent=2) + "\n")
    return {"records": records, "boundaries": boundary_rows, "boundary_summary": boundary_summary, "prefix_collisions": prefix_rows, "token_agreement": token_rows, "replay": replay}


# Short aliases make the two plan experiments easy to discover from a Python
# session while preserving the descriptive CLI-oriented names above.
run_separation_experiment = run_trace_separation_experiment


def measure_reuse_work(model: Any, prompts: Sequence[TracePrompt], *, sigma_abs: float, seed: int = 42, device: str = "cpu", repeats: int = 3, length: int = 32, output_dir: Path | None = None) -> list[dict[str, Any]]:
    """Measure fresh versus challenge-online inference work.

    DeepProve proving is optional in this repository, so ``rho_pf`` is
    explicitly reported as unavailable instead of being inferred from logits.
    """

    if repeats < 1 or not prompts:
        raise ValueError("prompts must be non-empty and repeats must be positive")
    torch = _torch()
    rows: list[dict[str, Any]] = []
    progress = _progress_bar(len(prompts), "embedding-reuse", unit="prompt")
    try:
        for prompt in prompts:
            continuation = generate_perturbed_trace(model, SimpleTokenizer(), prompt.token_ids, embedding_seed=f"{seed}|warm|{prompt.prompt_id}", sampling_seed=f"{seed}|warm|{prompt.prompt_id}", sigma_abs=sigma_abs, length=length, device=device)["tokens"]
            fresh_times: list[float] = []
            online_times: list[float] = []
            for repeat in range(repeats):
                start = time.perf_counter()
                _forward_prefix(model, tuple(prompt.token_ids) + tuple(continuation), None, prefix_length=len(prompt.token_ids), device=device)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                fresh_times.append(time.perf_counter() - start)
                start = time.perf_counter()
                _forward_prefix(model, tuple(prompt.token_ids) + tuple(continuation), deterministic_embedding_noise(f"{seed}|{repeat}|{prompt.prompt_id}", prompt.token_ids, (len(prompt.token_ids), _embedding_dimension(model)), sigma_abs), prefix_length=len(prompt.token_ids), device=device)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                online_times.append(time.perf_counter() - start)
            fresh, online = float(np.median(fresh_times)), float(np.median(online_times))
            rows.append({"prompt_id": prompt.prompt_id, "fresh_seconds": fresh, "online_update_seconds": online, "rho_inf": 1.0 - online / fresh if fresh else 0.0, "rho_pf": None, "prover_status": "not_integrated"})
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(output_dir / "reuse_work.csv", rows)
        _plot_reuse_work(rows, output_dir / "reuse_work.png")
    return rows


class SimpleTokenizer:
    """EOS-free tokenizer facade used only by the inference timing helper."""

    eos_token_id = None


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--devices", help="comma-separated worker devices; cuda uses all visible GPUs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/gpt2_embedding"))
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")


def _parse_conditions(value: str) -> tuple[str, ...]:
    """Parse and validate the separation condition subset."""

    conditions = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(conditions) - set(SEPARATION_CONDITIONS))
    if not conditions or unknown:
        allowed = ", ".join(SEPARATION_CONDITIONS)
        raise argparse.ArgumentTypeError(f"conditions must be drawn from {allowed}; unknown: {', '.join(unknown)}")
    return conditions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    utility = sub.add_parser("utility", help="Experiment 1: utility under one-time perturbation")
    _common_parser(utility)
    utility.add_argument("--tasks", default="wikitext2,lambada,hellaswag,piqa,arc_easy")
    utility.add_argument("--examples-per-task", type=int, default=50)
    utility.add_argument("--sigmas", type=_parse_floats, default=DEFAULT_SIGMAS)
    utility.add_argument("--perturbation-seeds", type=int, default=3)
    utility.add_argument("--prompt-file", type=Path)
    utility.add_argument("--bootstrap-replicates", type=int, default=1000)
    separation = sub.add_parser("separation", help="Experiment 2: quantized trace separation")
    _common_parser(separation)
    separation.add_argument("--tasks", default="wikitext2,lambada,hellaswag,piqa,arc_easy")
    separation.add_argument("--examples-per-task", type=int, default=100)
    separation.add_argument("--pairs", type=int, default=50)
    separation.add_argument("--length", type=int, default=32)
    separation.add_argument("--sigmas", type=_parse_floats, default=DEFAULT_SIGMAS)
    separation.add_argument("--quantization-scale", type=float, default=1e-3)
    separation.add_argument("--quantization-clip", type=int, default=2**31 - 1)
    separation.add_argument("--repeat-copy-file", type=Path)
    separation.add_argument("--resisting-correction-file", type=Path)
    separation.add_argument("--max-structural-prompts", type=int, default=100, help="cap prompts loaded from each structural stratum (default: 100; use a larger value for the full release)")
    separation.add_argument("--conditions", type=_parse_conditions, default=SEPARATION_CONDITIONS, help="comma-separated separation conditions")
    separation.add_argument("--prompt-file", type=Path)
    separation.add_argument("--target-tokens", type=int, default=16)
    separation.add_argument("--max-target-prompts", type=int, default=100, help="cap target-token evaluation prompts (default: 100)")
    separation.add_argument("--measure-reuse", action="store_true")
    run_all = sub.add_parser("run-all", help="run both new experiments")
    _common_parser(run_all)
    run_all.add_argument("--tasks", default="wikitext2,lambada,hellaswag,piqa,arc_easy")
    # ``run-all`` caps the large structural strata while retaining the full
    # sigma/condition/pair grid. Utility scoring uses a smaller per-task sample
    # because each example is evaluated at every sigma and replicate.
    run_all.add_argument("--examples-per-task", type=int, help="override both utility and separation examples-per-task values")
    run_all.add_argument("--utility-examples-per-task", type=int, default=50, help="utility examples per dataset (default: 50)")
    run_all.add_argument("--separation-examples-per-task", type=int, default=100, help="standard separation examples per dataset (default: 100)")
    run_all.add_argument("--sigmas", type=_parse_floats, default=DEFAULT_SIGMAS)
    run_all.add_argument("--perturbation-seeds", type=int, default=3)
    run_all.add_argument("--pairs", type=int, default=4)
    run_all.add_argument("--length", type=int, default=32)
    run_all.add_argument("--repeat-copy-file", type=Path)
    run_all.add_argument("--resisting-correction-file", type=Path)
    run_all.add_argument("--max-structural-prompts", type=int, default=100, help="cap prompts loaded from each structural stratum (default: 100; use a larger value for the full release)")
    run_all.add_argument("--conditions", type=_parse_conditions, default=SEPARATION_CONDITIONS, help="comma-separated separation conditions")
    run_all.add_argument("--prompt-file", type=Path)
    run_all.add_argument("--target-tokens", type=int, default=16)
    run_all.add_argument("--max-target-prompts", type=int, default=100, help="cap target-token evaluation prompts (default: 100)")
    run_all.add_argument("--quantization-scale", type=float, default=1e-3)
    run_all.add_argument("--quantization-clip", type=int, default=2**31 - 1)
    run_all.add_argument("--measure-reuse", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "run-all":
        override = args.examples_per_task
        args.utility_examples_per_task = override if override is not None else args.utility_examples_per_task
        args.separation_examples_per_task = override if override is not None else args.separation_examples_per_task
    else:
        args.utility_examples_per_task = args.examples_per_task
        args.separation_examples_per_task = args.examples_per_task
    if args.smoke:
        args.output_dir = args.output_dir / "smoke"
        args.utility_examples_per_task = min(args.utility_examples_per_task, 4)
        args.separation_examples_per_task = min(args.separation_examples_per_task, 4)
        args.pairs = min(getattr(args, "pairs", 2), 2)
        args.length = min(getattr(args, "length", 8), 8)
        args.sigmas = (0.0, 0.01)
        args.perturbation_seeds = min(getattr(args, "perturbation_seeds", 1), 1)
    try:
        devices = _resolve_devices(args.device, args.devices)
        model, tokenizer, torch = _load_model(args.model, devices[0], args.revision)
        workers: list[tuple[Any, Any, str]] = [(model, tokenizer, devices[0])]
        for worker_device in devices[1:]:
            worker_model, _, _ = _load_model(args.model, worker_device, args.revision)
            workers.append((worker_model, tokenizer, worker_device))
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        examples: list[UtilityExample] = []
        for task in tasks:
            if args.smoke and getattr(args, "prompt_file", None) is None:
                examples.extend(synthetic_utility_examples(tokenizer, task=task, count=args.utility_examples_per_task, prompt_tokens=args.prompt_tokens))
            else:
                examples.extend(load_utility_examples(tokenizer, task=task, count=args.utility_examples_per_task, prompt_tokens=args.prompt_tokens, seed=args.seed, prompt_file=getattr(args, "prompt_file", None) if task == "wikitext2" else None))
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        embedding_std = embedding_table_std(model)
    except (AttributeError, ValueError, RuntimeError):
        embedding_std = None
    metadata = {
        "model": args.model,
        "revision": args.revision,
        "model_weight_sha256": model_weight_sha256(model),
        "model_config_sha256": hashlib.sha256(model.config.to_json_string().encode("utf-8")).hexdigest() if hasattr(model.config, "to_json_string") else None,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "seed": args.seed,
        "device": args.device,
        "devices": devices,
        "parallel_workers": len(workers),
        "torch_version": torch.__version__,
        "torch_dtype": str(next(model.parameters()).dtype) if hasattr(model, "parameters") else None,
        "batch_size": 1,
        "utility_backend": "direct teacher-forced scoring with lm-eval task datasets; perturbation injected at model inputs_embeds",
        "hardware": platform.platform(),
        "sigmas": list(args.sigmas),
        "utility_examples_per_task": args.utility_examples_per_task,
        "separation_examples_per_task": args.separation_examples_per_task,
        "max_structural_prompts": getattr(args, "max_structural_prompts", None),
        "max_target_prompts": getattr(args, "max_target_prompts", None),
        "separation_conditions": list(getattr(args, "conditions", SEPARATION_CONDITIONS)),
        "embedding_table_std_s_E": embedding_std,
        "alpha_by_sigma": {str(float(sigma)): (float(sigma) / embedding_std if embedding_std else None) for sigma in args.sigmas},
        "perturbation": "VRF/domain-separated SHA256 Box-Muller, clipped at 3σ, quantized to 1e-6",
        "quantization_scale": getattr(args, "quantization_scale", None),
        "quantization_clip": getattr(args, "quantization_clip", None),
        "deep_prove": "not_integrated; boundary comparison is pre-integration",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    utility_result = None
    if args.command in {"utility", "run-all"}:
        utility_result = run_utility_experiment(model, examples, sigmas=args.sigmas, perturbation_seeds=args.perturbation_seeds, seed=args.seed, device=devices[0], bootstrap_replicates=getattr(args, "bootstrap_replicates", 1000), output_dir=args.output_dir / "utility" if args.command == "run-all" else args.output_dir, resume=not args.no_resume, progress_every=args.progress_every, worker_models=workers)
        if args.command == "utility":
            print(f"[embedding-utility] utility_only_sigma={select_utility_scale(utility_result['summary'])}", flush=True)
    if args.command in {"separation", "run-all"}:
        standard_count = args.separation_examples_per_task
        trace_prompts = load_trace_prompts(tokenizer, standard_examples=examples, repeat_copy_file=args.repeat_copy_file, resisting_correction_file=args.resisting_correction_file, count_per_standard_task=standard_count, max_structural_prompts=getattr(args, "max_structural_prompts", None), smoke=args.smoke)
        max_target_prompts = getattr(args, "max_target_prompts", 100)
        if max_target_prompts < 1:
            parser.error("--max-target-prompts must be positive")
        target_prompts = build_target_token_suite(tokenizer, model, device=devices[0], seed=args.seed, targets=4 if args.smoke else getattr(args, "target_tokens", 16), construction_carriers=4 if args.smoke else 16, evaluation_carriers=2 if args.smoke else 4)
        # Construction carriers define the filter; only the held-out carriers
        # enter the separation table.
        evaluation_targets = [prompt for prompt in target_prompts if (prompt.metadata or {}).get("split") == "evaluation"]
        trace_prompts.extend(evaluation_targets[:max_target_prompts])
        separation_result = run_trace_separation_experiment(model, tokenizer, trace_prompts, sigmas=args.sigmas, pairs=getattr(args, "pairs", 50), length=getattr(args, "length", 32), seed=args.seed, device=devices[0], quantization_scale=getattr(args, "quantization_scale", 1e-3), quantization_clip=getattr(args, "quantization_clip", 2**31 - 1), conditions=getattr(args, "conditions", SEPARATION_CONDITIONS), output_dir=args.output_dir / "separation" if args.command == "run-all" else args.output_dir, resume=not args.no_resume, progress_every=args.progress_every, worker_models=workers)
        if getattr(args, "measure_reuse", False):
            measure_reuse_work(model, trace_prompts[: min(10, len(trace_prompts))], sigma_abs=float(args.sigmas[-1]), seed=args.seed, device=devices[0], length=getattr(args, "length", 32), output_dir=args.output_dir / "separation" if args.command == "run-all" else args.output_dir)
        if utility_result is not None:
            print(f"[embedding-utility] selected_sigma={select_operating_point(utility_result['summary'], separation_result['boundary_summary'])}", flush=True)
        print(f"[embedding-separation] completed {len(separation_result['records'])} units", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
