"""Configuration loader for PoML simulation."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class SimConfig(BaseModel):
    model_path: str = "model/network.onnx"
    ezkl_artifacts_dir: str = "model/"

    num_miners: int = 4
    num_queries: int = 20
    query_rate: float = 2.0

    difficulty: str = "0x00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    block_reward: int = 50
    network_latency_ms: int = 100
    max_queries_per_block: int = 10

    input_shape: list[int] = [1, 2, 8, 8]
    seed: int = 42

    @property
    def difficulty_int(self) -> int:
        return int(self.difficulty, 16)


def load_config(path: str | Path = "config.yaml") -> SimConfig:
    path = Path(path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)
        return SimConfig(**data)
    return SimConfig()
