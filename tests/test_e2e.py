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
@pytest.mark.timeout(300)
class TestE2E:
    def test_small_simulation(self):
        # Difficulty is set to the maximum 256-bit value so every proof wins
        # the lottery. Real EZKL proofs take ~60s each with this circuit, so
        # probabilistic mining is impractical for a smoke test; lottery math
        # is covered by unit tests.
        config = SimConfig(
            model_path="model/network.onnx",
            ezkl_artifacts_dir="model/",
            num_miners=2,
            num_queries=3,
            initial_burst=3,
            steady_interval_s=0.2,
            difficulty="0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            block_reward=50,
            network_latency_ms=50,
            max_queries_per_block=5,
            input_shape=[1, 2, 8, 8],
            seed=42,
        )

        coordinator = Coordinator(config)
        coordinator.run(timeout=180.0)

        # Verify blocks were produced
        assert coordinator.blockchain.get_height() >= 1

        # Verify chain is valid (all blocks chained correctly)
        chain = coordinator.blockchain.chain
        T = config.diffusion_steps
        for i in range(1, len(chain)):
            block = chain[i]
            assert block.header.block_height == i
            assert len(block.queries) > 0
            assert len(block.results) > 0
            # Revised protocol: miners must declare a VRF vk and attach a
            # per-query transcript of length T.
            assert len(block.header.miner_vrf_vk) == 32
            for result in block.results:
                assert len(result.vrf_transcript) == T
                assert len(result.ciphertext) > 0

        # Verify metrics were collected
        assert len(coordinator.metrics.blocks) >= 1

        # Cleanup
        if os.path.exists("metrics.json"):
            os.remove("metrics.json")
