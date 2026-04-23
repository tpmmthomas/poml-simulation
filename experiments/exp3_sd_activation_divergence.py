"""Appendix experiment — activation divergence under input perturbation (Stable Diffusion).

Runs the full denoising chain, recording UNet activations at every layer and
every denoising step.  Compares base vs. perturbed / independent chains to
empirically validate the CIA property underlying the PoML lottery.

This is the Stable Diffusion variant of what the paper labels the
``exp1_activation_divergence`` appendix experiment.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from tqdm import trange

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.sd.hooks import ActivationRecorder  # noqa: E402
from experiments.sd.utils import (  # noqa: E402
    MODELS_DIR,
    cosine_similarity_flat,
    get_device,
    l_infinity_flat,
    l2_distance_flat,
    relative_error,
    seed_everything,
)

SD_MODEL_PATH = MODELS_DIR / "stable-diffusion" / "stable-diffusion-v1-4-fp16"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N = 1_000
SIGMAS = [0.001, 0.01, 0.1, 0.5, 1.0]
NUM_INFERENCE_STEPS = 20
SEED = 0
PROMPTS = [
    "a photo of a cat",
    "a painting of a mountain landscape",
    "a sketch of a city skyline",
    "a photo of a dog playing in the park",
    "an oil painting of a sunset over the ocean",
]
OUT_DIR = Path(__file__).parent / "results"
FIG_DIR = OUT_DIR / "figures"


def get_unet_block_names(unet: torch.nn.Module) -> list[str]:
    """Return names of all significant blocks in forward-pass order."""
    names: list[str] = ["conv_in"]
    for name, _mod in unet.named_modules():
        if name.startswith("down_blocks.") and name.count(".") == 1:
            names.append(name)
    names.append("mid_block")
    for name, _mod in unet.named_modules():
        if name.startswith("up_blocks.") and name.count(".") == 1:
            names.append(name)
    names.append("conv_out")
    return names


def _unet_forward(
    unet: torch.nn.Module,
    layer_names: list[str],
    latent: torch.Tensor,
    t: torch.Tensor,
    prompt_embeds: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Single UNet forward pass; returns (activations_dict, noise_pred)."""
    recorder = ActivationRecorder(unet, layer_names)
    with recorder:
        out = unet(latent, t, encoder_hidden_states=prompt_embeds)
    acts = {k: v.cpu() for k, v in recorder.activations.items()}
    noise_pred = out.sample
    acts["__output__"] = noise_pred.cpu()
    return acts, noise_pred


def run() -> None:
    seed_everything(SEED)
    device = get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        str(SD_MODEL_PATH),
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to(device)
    unet = pipe.unet.eval()

    layer_names = get_unet_block_names(unet)
    measure_names = layer_names + ["__output__"]
    print(f"Hooking {len(layer_names)} blocks: {layer_names}")
    print(f"Full denoising chain: {NUM_INFERENCE_STEPS} steps")

    latent_shape = (1, 4, 64, 64)

    # Pre-encode all prompts
    with torch.no_grad():
        prompt_embeds_list = [
            pipe.encode_prompt(p, device, 1, do_classifier_free_guidance=False)[0]
            for p in PROMPTS
        ]

    # Determine the actual number of scheduler timesteps (PNDM adds priming
    # steps, so len(timesteps) may exceed NUM_INFERENCE_STEPS).
    _probe_sched = copy.deepcopy(pipe.scheduler)
    _probe_sched.set_timesteps(NUM_INFERENCE_STEPS, device=device)
    num_actual_steps = len(_probe_sched.timesteps)
    print(f"Scheduler produces {num_actual_steps} timesteps")

    # ------------------------------------------------------------------
    # Storage: (step_idx, layer_name) → list of N floats
    # ------------------------------------------------------------------
    step_layer_keys = [
        (s, n) for s in range(num_actual_steps) for n in measure_names
    ]

    cos_sigma: dict[float, dict[tuple, list[float]]] = {
        sig: {k: [] for k in step_layer_keys} for sig in SIGMAS
    }
    l2_sigma: dict[float, dict[tuple, list[float]]] = {
        sig: {k: [] for k in step_layer_keys} for sig in SIGMAS
    }
    abs_l2_sigma: dict[float, dict[tuple, list[float]]] = {
        sig: {k: [] for k in step_layer_keys} for sig in SIGMAS
    }
    l_inf_sigma: dict[float, dict[tuple, list[float]]] = {
        sig: {k: [] for k in step_layer_keys} for sig in SIGMAS
    }
    cos_ind_same: dict[tuple, list[float]] = {k: [] for k in step_layer_keys}
    l2_ind_same: dict[tuple, list[float]] = {k: [] for k in step_layer_keys}
    abs_l2_ind_same: dict[tuple, list[float]] = {k: [] for k in step_layer_keys}
    l_inf_ind_same: dict[tuple, list[float]] = {k: [] for k in step_layer_keys}
    cos_ind_diff: dict[tuple, list[float]] = {k: [] for k in step_layer_keys}
    l2_ind_diff: dict[tuple, list[float]] = {k: [] for k in step_layer_keys}
    abs_l2_ind_diff: dict[tuple, list[float]] = {k: [] for k in step_layer_keys}
    l_inf_ind_diff: dict[tuple, list[float]] = {k: [] for k in step_layer_keys}

    for _i in trange(N, desc="samples"):
        pidx = torch.randint(0, len(PROMPTS), (1,)).item()
        prompt_embeds = prompt_embeds_list[pidx]
        pidx2 = (pidx + torch.randint(1, len(PROMPTS), (1,)).item()) % len(PROMPTS)
        prompt_embeds2 = prompt_embeds_list[pidx2]

        z_init = torch.randn(latent_shape, device=device, dtype=torch.float16)

        # Prepare per-chain state: (latent, scheduler, prompt_embeds)
        def _make_sched():
            s = copy.deepcopy(pipe.scheduler)
            s.set_timesteps(NUM_INFERENCE_STEPS, device=device)
            return s

        base_sched = _make_sched()
        z_base = z_init.clone()

        sigma_chains = {}
        for sigma in SIGMAS:
            eta = torch.randn_like(z_init) * sigma
            sigma_chains[sigma] = {"z": z_init + eta, "sched": _make_sched()}

        z_ind_same = torch.randn(latent_shape, device=device, dtype=torch.float16)
        sched_ind_same = _make_sched()

        z_ind_diff = torch.randn(latent_shape, device=device, dtype=torch.float16)
        sched_ind_diff = _make_sched()

        with torch.no_grad():
            for step_idx, t in enumerate(base_sched.timesteps):
                # --- Base forward ---
                acts_base, pred_base = _unet_forward(
                    unet, layer_names, z_base, t, prompt_embeds
                )
                z_base = base_sched.step(pred_base, t, z_base).prev_sample

                # --- Sigma-perturbed (same prompt) ---
                for sigma in SIGMAS:
                    ch = sigma_chains[sigma]
                    acts_pert, pred_pert = _unet_forward(
                        unet, layer_names, ch["z"], t, prompt_embeds
                    )
                    ch["z"] = ch["sched"].step(pred_pert, t, ch["z"]).prev_sample
                    for n in measure_names:
                        cos_sigma[sigma][(step_idx, n)].append(
                            cosine_similarity_flat(acts_base[n], acts_pert[n])
                        )
                        l2_sigma[sigma][(step_idx, n)].append(
                            relative_error(acts_pert[n], acts_base[n])
                        )
                        abs_l2_sigma[sigma][(step_idx, n)].append(
                            l2_distance_flat(acts_pert[n], acts_base[n])
                        )
                        l_inf_sigma[sigma][(step_idx, n)].append(
                            l_infinity_flat(acts_pert[n], acts_base[n])
                        )

                # --- Independent baseline, same prompt ---
                acts_ind, pred_ind = _unet_forward(
                    unet, layer_names, z_ind_same, t, prompt_embeds
                )
                z_ind_same = sched_ind_same.step(pred_ind, t, z_ind_same).prev_sample
                for n in measure_names:
                    cos_ind_same[(step_idx, n)].append(
                        cosine_similarity_flat(acts_base[n], acts_ind[n])
                    )
                    l2_ind_same[(step_idx, n)].append(
                        relative_error(acts_ind[n], acts_base[n])
                    )
                    abs_l2_ind_same[(step_idx, n)].append(
                        l2_distance_flat(acts_ind[n], acts_base[n])
                    )
                    l_inf_ind_same[(step_idx, n)].append(
                        l_infinity_flat(acts_ind[n], acts_base[n])
                    )

                # --- Independent baseline, different prompt ---
                acts_ind2, pred_ind2 = _unet_forward(
                    unet, layer_names, z_ind_diff, t, prompt_embeds2
                )
                z_ind_diff = sched_ind_diff.step(
                    pred_ind2, t, z_ind_diff
                ).prev_sample
                for n in measure_names:
                    cos_ind_diff[(step_idx, n)].append(
                        cosine_similarity_flat(acts_base[n], acts_ind2[n])
                    )
                    l2_ind_diff[(step_idx, n)].append(
                        relative_error(acts_ind2[n], acts_base[n])
                    )
                    abs_l2_ind_diff[(step_idx, n)].append(
                        l2_distance_flat(acts_ind2[n], acts_base[n])
                    )
                    l_inf_ind_diff[(step_idx, n)].append(
                        l_infinity_flat(acts_ind2[n], acts_base[n])
                    )

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    def _stats(vals: list[float]) -> dict[str, float]:
        a = np.array(vals)
        return {"mean": float(np.mean(a)), "std": float(np.std(a)),
                "min": float(np.min(a)), "max": float(np.max(a))}

    def _build_per_step(data: dict[tuple, list[float]]) -> dict:
        out: dict = {}
        for step_idx in range(num_actual_steps):
            out[str(step_idx)] = {
                n: _stats(data[(step_idx, n)]) for n in measure_names
            }
        return out

    results: dict = {
        "layer_names": layer_names,
        "sigmas": SIGMAS,
        "N": N,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "num_actual_steps": num_actual_steps,
    }

    results["cosine"] = {
        "sigma_perturbed": {
            str(s): _build_per_step(cos_sigma[s]) for s in SIGMAS
        },
        "independent_same_prompt": _build_per_step(cos_ind_same),
        "independent_diff_prompt": _build_per_step(cos_ind_diff),
    }
    results["l2"] = {
        "sigma_perturbed": {
            str(s): _build_per_step(l2_sigma[s]) for s in SIGMAS
        },
        "independent_same_prompt": _build_per_step(l2_ind_same),
        "independent_diff_prompt": _build_per_step(l2_ind_diff),
    }
    results["abs_l2"] = {
        "sigma_perturbed": {
            str(s): _build_per_step(abs_l2_sigma[s]) for s in SIGMAS
        },
        "independent_same_prompt": _build_per_step(abs_l2_ind_same),
        "independent_diff_prompt": _build_per_step(abs_l2_ind_diff),
    }
    results["l_inf"] = {
        "sigma_perturbed": {
            str(s): _build_per_step(l_inf_sigma[s]) for s in SIGMAS
        },
        "independent_same_prompt": _build_per_step(l_inf_ind_same),
        "independent_diff_prompt": _build_per_step(l_inf_ind_diff),
    }

    out_path = OUT_DIR / "sd_activation_divergence.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    # ------------------------------------------------------------------
    # Plotting: similarity vs. denoising step (averaged across layers)
    # ------------------------------------------------------------------
    step_indices = list(range(num_actual_steps))

    def _step_avg(per_step_data: dict) -> tuple[list[float], list[float]]:
        """Return (means, stds) per step, averaged across layers."""
        means, stds = [], []
        for step_idx in step_indices:
            step_data = per_step_data[str(step_idx)]
            layer_means = [step_data[n]["mean"] for n in layer_names]
            layer_stds = [step_data[n]["std"] for n in layer_names]
            means.append(float(np.mean(layer_means)))
            # Propagate std: average of per-layer stds (conservative)
            stds.append(float(np.mean(layer_stds)))
        return means, stds

    def _plot(metric_key: str, ylabel: str, title: str, filename: str) -> None:
        data = results[metric_key]
        fig, ax = plt.subplots(figsize=(10, 5))

        for sigma in SIGMAS:
            m, s = _step_avg(data["sigma_perturbed"][str(sigma)])
            ma, sa = np.array(m), np.array(s)
            ax.plot(step_indices, ma, marker="o", markersize=3, label=f"σ={sigma}")
            ax.fill_between(step_indices, ma - sa, ma + sa, alpha=0.15)

        m_same, s_same = _step_avg(data["independent_same_prompt"])
        m_diff, s_diff = _step_avg(data["independent_diff_prompt"])
        ax.plot(step_indices, m_same, ls="--", color="grey",
                marker="s", markersize=3, label="indep. (same prompt)")
        ax.plot(step_indices, m_diff, ls=":", color="black",
                marker="^", markersize=3, label="indep. (diff prompt)")

        ax.set_xlabel("Denoising step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xticks(step_indices)
        fig.tight_layout()
        fig_path = FIG_DIR / filename
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Figure saved to {fig_path}")

    _plot(
        "cosine",
        "Mean cosine similarity (avg. across layers)",
        "Stable Diffusion — Cosine similarity vs. denoising step",
        "sd_cosine_sim_vs_step.pdf",
    )
    _plot(
        "l2",
        "Mean relative L2 error (avg. across layers)",
        "Stable Diffusion — Relative L2 error vs. denoising step",
        "sd_rel_l2_vs_step.pdf",
    )
    _plot(
        "abs_l2",
        "Mean absolute L2 distance (avg. across layers)",
        "Stable Diffusion — Absolute L2 distance vs. denoising step",
        "sd_abs_l2_vs_step.pdf",
    )
    _plot(
        "l_inf",
        "Mean L_inf distance (avg. across layers)",
        "Stable Diffusion - L_inf distance vs. denoising step",
        "sd_l_inf_vs_step.pdf",
    )


if __name__ == "__main__":
    run()
