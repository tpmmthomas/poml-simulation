"""One-off script to download Stable Diffusion v1.4 (fp16) weights.

Usage:
    python scripts/download_sd_model.py

Downloads CompVis/stable-diffusion-v1-4 and re-saves it as the fp16 variant
under ``models/stable-diffusion/stable-diffusion-v1-4-fp16`` — the path
consumed by the SD appendix experiments
(``experiments/diffusion_compatibility.py``).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def download_stable_diffusion(dest: Path) -> None:
    """Download Stable Diffusion v1.4 (fp16 variant — smallest practical SD checkpoint)."""
    import torch
    from diffusers import StableDiffusionPipeline  # type: ignore

    print("Downloading Stable Diffusion v1.4 …")
    dest.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        cache_dir=str(dest),
        torch_dtype=torch.float16,
    )
    pipe.save_pretrained(dest / "stable-diffusion-v1-4-fp16")
    print(f"Stable Diffusion v1.4 saved to {dest / 'stable-diffusion-v1-4-fp16'}")
    del pipe


def main() -> None:
    print(f"Model root: {MODELS_DIR}\n")
    download_stable_diffusion(MODELS_DIR / "stable-diffusion")
    print("\nStable Diffusion weights downloaded successfully.")


if __name__ == "__main__":
    main()
