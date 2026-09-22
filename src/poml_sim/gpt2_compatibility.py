"""Float32 GPT-2 utility and independent perturbed-trajectory comparisons."""

import math
from .crypto import canonical, sha256
from .protocol_inputs import experimental_key, gaussian_vector
from .vrf import vrf_eval

MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def load_model(device="cuda:0"):
    """Load the pinned public GPT-2 checkpoint in deterministic evaluation mode."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", revision=MODEL_REVISION)
    model = (
        AutoModelForCausalLM.from_pretrained(
            "openai-community/gpt2", revision=MODEL_REVISION, attn_implementation="eager"
        )
        .float()
        .eval()
        .to(device)
    )
    return model, tokenizer


def embedding_std(model):
    """Use population std of every entry in the frozen token embedding table."""
    return float(model.get_input_embeddings().weight.detach().double().std(correction=0))


def prompt_noise(model, tokens, alpha, seed):
    """Add independent Gaussian rows once, with sigma_abs = alpha * s_E."""
    import torch

    if not math.isfinite(alpha) or alpha < 0 or not tokens:
        raise ValueError("nonnegative finite alpha and nonempty prompt required")
    key = experimental_key(0, f"compatibility:{seed}")
    base = sha256(canonical(tokens))
    scale = alpha * embedding_std(model)
    rows = [
        gaussian_vector(vrf_eval(key, base + t.to_bytes(8, "big"))[0], model.config.n_embd)
        for t in range(1, len(tokens) + 1)
    ]
    return torch.tensor(rows, dtype=torch.float32, device=model.device) * scale


def forward(model, tokens, noise):
    """Evaluate a full causal prefix; newly generated token embeddings stay clean."""
    import torch

    ids = torch.tensor([tokens], device=model.device)
    embeddings = model.get_input_embeddings()(ids).clone()
    embeddings[:, : len(noise)] += noise
    return model(inputs_embeds=embeddings, use_cache=False)


def sample_token(logits, uniform, *, temperature=1.0, top_k=50, top_p=0.95):
    """Apply temperature, top-k/top-p filtering and inverse-CDF sampling."""
    import torch

    if not 0 <= uniform < 1 or temperature <= 0 or not 0 < top_p <= 1 or top_k < 1:
        raise ValueError("invalid decoding parameters")
    values, indices = torch.sort(logits.double() / temperature, descending=True, stable=True)
    values[top_k:] = -torch.inf
    probabilities = torch.softmax(values, dim=-1)
    excluded = probabilities.cumsum(-1) - probabilities >= top_p
    values[excluded] = -torch.inf
    probabilities = torch.softmax(values, dim=-1)
    # Use ascending token-id order for a fully specified inverse-CDF map.
    ordered = torch.zeros_like(probabilities).scatter(0, indices, probabilities)
    index = int(
        torch.searchsorted(
            ordered.cumsum(-1),
            torch.tensor(uniform, device=logits.device, dtype=torch.float64),
            right=True,
        )
    )
    return min(index, len(ordered) - 1)


def score_continuation(model, context, target, noise):
    """Return teacher-forced total NLL and target count after a noisy prompt."""
    import torch

    if not context or not target:
        raise ValueError("nonempty context and target required")
    with torch.inference_mode():
        logits = forward(model, context + target, noise).logits[
            0, len(context) - 1 : len(context) + len(target) - 1
        ]
        target_ids = torch.tensor(target, device=model.device)
        loss = torch.nn.functional.cross_entropy(logits.float(), target_ids, reduction="sum")
    return float(loss), len(target)


def utility(model, examples, *, alphas=(0.05, 0.1, 0.2, 0.4), replicates=3, seed=42):
    """Measure clean/noisy perplexity and multiple-choice accuracy per task."""
    if replicates < 1:
        raise ValueError("positive replicate count required")
    rows = []
    for example in examples:
        context = example["context"]
        for alpha in (0, *alphas):
            for replicate in range(1 if alpha == 0 else replicates):
                noise = prompt_noise(model, context, alpha, f"{seed}:{example['id']}:{replicate}")
                row = {
                    "id": example["id"],
                    "task": example["task"],
                    "alpha": alpha,
                    "replicate": replicate,
                }
                if example["choices"]:
                    scores = [
                        score_continuation(model, context, choice, noise)[0]
                        for choice in example["choices"]
                    ]
                    predicted = min(range(len(scores)), key=scores.__getitem__)
                    row["correct"] = int(predicted == example["answer"])
                else:
                    row["nll"], row["tokens"] = score_continuation(
                        model, context, example["target"], noise
                    )
                rows.append(row)
    return rows


def utility_summary(rows):
    """Pool NLL by token count and average perturbed replicate metrics."""
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["alpha"], row["replicate"]), []).append(row)
    values = {}
    for (task, alpha, _), selected in groups.items():
        metric = "accuracy" if "correct" in selected[0] else "perplexity"
        value = (
            sum(r["correct"] for r in selected) / len(selected)
            if metric == "accuracy"
            else math.exp(sum(r["nll"] for r in selected) / sum(r["tokens"] for r in selected))
        )
        values.setdefault((task, alpha, metric), []).append(value)
    return [
        {
            "task": task,
            "alpha": alpha,
            "metric": metric,
            "value": sum(v) / len(v),
            "replicates": len(v),
        }
        for (task, alpha, metric), v in values.items()
    ]


def capture(model, tokens, noise):
    """Capture the full context matrices at the paper's named boundaries."""
    import torch

    values, handles = {}, []
    modules = [("embedding", model.transformer.drop)]
    for index, block in enumerate(model.transformer.h, 1):
        modules.extend([(f"{index}A", block.attn), (f"{index}F", block.mlp)])
    modules.append(("norm", model.transformer.ln_f))

    def hook(name):
        def record(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            values[name] = tensor.detach().float().cpu()

        return record

    try:
        for name, module in modules:
            handles.append(module.register_forward_hook(hook(name)))
        with torch.inference_mode():
            output = forward(model, tokens, noise).logits[0, -1]
        values["logits"] = output.detach().float().cpu()
        return output, values
    finally:
        for handle in handles:
            handle.remove()


def compare_trajectories(
    model, example, *, alpha=0.05, pair=0, steps=(0, 1, 4, 8, 16, 32), seed=42
):
    """Independently decode both runs and compare only mutually executed prefixes."""
    import torch

    if not steps or min(steps) < 0:
        raise ValueError("nonnegative prefix lengths required")
    prompt = example["context"]
    contexts = [list(prompt), list(prompt)]
    noises = [
        prompt_noise(model, prompt, alpha, f"{seed}:{example['id']}:{pair}:{side}")
        for side in range(2)
    ]
    active = [True, True]
    rows = []
    for step in range(max(steps) + 1):
        if not all(active):
            break
        captures = []
        for side in range(2):
            with torch.inference_mode():
                if step in steps:
                    logits, values = capture(model, contexts[side], noises[side])
                    captures.append(values)
                else:
                    logits = forward(model, contexts[side], noises[side]).logits[0, -1]
            bits = sha256(canonical([seed, example["id"], pair, side, step, "decode"]))
            token = sample_token(logits, (int.from_bytes(bits[:8], "big") >> 11) / 2**53)
            contexts[side].append(token)
            active[side] = token != model.config.eos_token_id
        if step in steps:
            for boundary, first in captures[0].items():
                second = captures[1][boundary]
                if first.shape != second.shape:
                    raise ValueError("mutually executed prefix shapes differ")
                distance = float((first.double() - second.double()).abs().max())
                rows.append(
                    {
                        "id": example["id"],
                        "task": example["task"],
                        "alpha": alpha,
                        "pair": pair,
                        "prefix": step,
                        "boundary": boundary,
                        "linf": distance,
                    }
                )
    return rows
