"""Appendix experiment — wall-clock inference timing (Stable Diffusion).

Runs the full SD pipeline 1000 times with random prompts and random seeds,
measuring wall-clock time per inference.  Reports mean, min, max, and
standard deviation.  The timing distribution is a prerequisite for
estimating the PoML lottery difficulty parameter ``D``.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import torch
from diffusers import StableDiffusionPipeline

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.sd.utils import MODELS_DIR, get_device  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_RUNS = 1_000
NUM_INFERENCE_STEPS = 20
OUT_DIR = Path(__file__).parent / "results"
SD_MODEL_PATH = MODELS_DIR / "stable-diffusion" / "stable-diffusion-v1-4-fp16"

# A small pool of random-ish prompts; each run picks one at random so that
# no text-encoder output is accidentally cached across runs.
_PROMPT_POOL = [
    "a photo of a cat",
    "a painting of a mountain landscape",
    "a sketch of a city skyline",
    "a photo of a dog playing in the park",
    "an oil painting of a sunset over the ocean",
    "a digital art piece of a futuristic city",
    "a watercolor of a forest in autumn",
    "a cartoon drawing of a robot",
    "a photograph of a red rose",
    "a pencil sketch of a sailing ship",
]


def _random_prompt() -> str:
    """Pick a random prompt from the pool."""
    return random.choice(_PROMPT_POOL)


def run() -> None:
    device = get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        str(SD_MODEL_PATH),
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to(device)
    # Disable the NSFW safety checker to avoid its overhead skewing timing.
    pipe.safety_checker = None  # type: ignore[assignment]

    wall_times: list[float] = []

    print(f"Running {N_RUNS} inference passes …")
    for i in range(N_RUNS):
        seed = random.randint(0, 2**31 - 1)
        prompt = _random_prompt()
        generator = torch.Generator(device=device).manual_seed(seed)

        # Synchronise the GPU before starting the clock so device-side work
        # from the previous iteration does not bleed into this measurement.
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        _ = pipe(
            prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
            output_type="np",  # skip PIL conversion overhead
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        wall_times.append(elapsed)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{N_RUNS}]  last={elapsed:.3f}s  "
                  f"running_mean={mean(wall_times):.3f}s")

    mean_t = mean(wall_times)
    min_t = min(wall_times)
    max_t = max(wall_times)
    sd_t = stdev(wall_times)  # sample std-dev (N-1 denominator)

    print(f"\nResults over {N_RUNS} runs:")
    print(f"  Mean : {mean_t:.4f} s")
    print(f"  Min  : {min_t:.4f} s")
    print(f"  Max  : {max_t:.4f} s")
    print(f"  SD   : {sd_t:.4f} s")

    results = {
        "n_runs": N_RUNS,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "device": str(device),
        "mean_s": mean_t,
        "min_s": min_t,
        "max_s": max_t,
        "sd_s": sd_t,
        "wall_times_s": wall_times,
    }

    out_path = OUT_DIR / "sd_inference_timing.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    run()
