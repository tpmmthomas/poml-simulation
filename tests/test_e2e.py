"""End-to-end test: run a small simulation with 2 miners and a few queries."""

import os
import time

import pytest

from poml_sim.config import SimConfig
from poml_sim.coordinator import Coordinator

ARTIFACTS_DIR = "model/"
REQUIRED_FILES = ["network.ezkl", "pk.key", "vk.key", "kzg.srs", "settings.json"]


def artifacts_available() -> bool:
    return all(os.path.exists(os.path.join(ARTIFACTS_DIR, f)) for f in REQUIRED_FILES)


@pytest.mark.skipif(not artifacts_available(), reason="EZKL artifacts not set up")
@pytest.mark.timeout(120)
class TestE2E:
    def test_small_simulation(self):
        config = SimConfig(
            model_path="model/network.onnx",
            ezkl_artifacts_dir="model/",
            num_miners=2,
            num_queries=3,
            query_rate=5.0,
            difficulty="0x00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            block_reward=50,
            network_latency_ms=50,
            max_queries_per_block=5,
            input_shape=[1, 2, 8, 8],
            seed=42,
        )

        coordinator = Coordinator(config)
        coordinator.run(timeout=90.0)

        # Verify blocks were produced
        assert coordinator.blockchain.get_height() >= 1

        # Verify chain is valid (all blocks chained correctly)
        chain = coordinator.blockchain.chain
        for i in range(1, len(chain)):
            block = chain[i]
            assert block.header.block_height == i
            assert len(block.queries) > 0
            assert len(block.results) > 0

        # Verify metrics were collected
        assert len(coordinator.metrics.blocks) >= 1

        # Cleanup
        if os.path.exists("metrics.json"):
            os.remove("metrics.json")
