"""Shared helpers for the SD appendix experiments: metrics, seeding, paths."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


# poml-sim/experiments/sd/utils.py  ->  poml-sim/
ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors (flattened)."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ).item()


def l2_distance_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """L2 distance between two tensors (flattened)."""
    return torch.norm(a.flatten().float() - b.flatten().float()).item()


def l_infinity_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """L-infinity distance between two tensors."""
    return torch.max(torch.abs(a - b)).item()


def relative_error(approx: torch.Tensor, target: torch.Tensor) -> float:
    """||approx - target|| / ||target||."""
    return (
        torch.norm(approx.float() - target.float())
        / torch.norm(target.float()).clamp(min=1e-8)
    ).item()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
