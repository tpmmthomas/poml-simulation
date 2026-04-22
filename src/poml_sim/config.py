"""Configuration loader for PoML simulation."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class SimConfig(BaseModel):
    model_path: str = "model/network.onnx"
    ezkl_artifacts_dir: str = "model/"

    num_miners: int = 4
    num_queries: int = 100
    # Query arrival pacing: an initial burst fills the mempool fast, then a
    # steady drip keeps load arriving over the course of the run.
    initial_burst: int = 10
    steady_interval_s: float = 20.0

    # Per-miner mempool fetch strategy. Cycled if shorter than num_miners.
    # Valid values: "sequential", "high_fee", "random". Heterogeneous by
    # default to model real-world miner behaviour where different miners
    # implement different fee/ordering policies.
    fetch_strategies: list[str] = ["sequential", "high_fee", "random"]

    difficulty: str = "0x00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    block_reward: int = 50
    network_latency_ms: int = 100
    max_queries_per_block: int = 10

    input_shape: list[int] = [1, 2, 8, 8]
    seed: int = 42

    # Wall-clock deadline for the coordinator's main loop, in seconds. The
    # simulation stops early once all queries are drained and at least one
    # block has been produced; otherwise it terminates when this elapses.
    simulation_timeout: float = 600.0

    # Cap total CPU cores used by the whole simulation (Linux only). When set,
    # the parent process is pinned to the first N cores via sched_setaffinity
    # (inherited by miner subprocesses), and rayon/BLAS thread-count env vars
    # are set so the per-miner proof parallelism divides evenly into the cap.
    # None = use all cores.
    cpu_limit: int | None = None

    # Length T of the VRF-derived noise schedule U_i = (z_{i,1}, ..., z_{i,T}).
    # The single-pass U-Net only consumes U_i[0] as its noise channel, but
    # all T transcript entries are produced and validated per the paper.
    diffusion_steps: int = 1

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
