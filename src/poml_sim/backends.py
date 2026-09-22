"""Fresh model execution using GPT-2/DeepProve or the small EZKL denoiser.

The protocol wraps these model proofs with an explicit trusted-host relation
check. Neither prover currently proves the entire private PoML NP relation.
"""

from dataclasses import dataclass, field
from importlib.resources import files
import json
from pathlib import Path
import time

from .crypto import canonical, sha256
from .gpt2_work import reference_counts, weighted_cost
from .protocol_inputs import gaussian_vector


@dataclass(frozen=True)
class Execution:
    """A fresh output and model proof, before the PoML statement is constructed."""

    output: bytes
    output_length: int
    complexity: int
    duration: float
    proof: bytes
    metadata: dict = field(default_factory=dict)


def load_schedule(path=None) -> dict:
    """Load the public, digest-checked appendix operation-count matrix."""
    resource = (
        Path(path) if path else files("poml_sim").joinpath("data/gpt2_reference_schedule.json")
    )
    schedule = json.loads(resource.read_text())
    reference_counts(2, 1, schedule)
    return schedule


class GPT2DeepProveBackend:
    """Use the pinned persistent worker, with fresh noisy inference and proofs."""

    name = "gpt2"
    proof_kind = "deepprove-public-logits"

    def __init__(
        self,
        binary,
        directory,
        *,
        device="cuda:0",
        context=64,
        alpha=0.05,
        weights=None,
        setup_directory=None,
    ):
        from transformers import AutoTokenizer
        from .live_prover import LiveProver

        if context != 64 or alpha <= 0:
            raise ValueError("the reference schedule requires context 64 and positive alpha")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "openai-community/gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e"
        )
        self.schedule = load_schedule()
        self.weights = weights
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prover = LiveProver(
            Path(binary),
            Path(setup_directory) if setup_directory else self.directory / "setup",
            device=device,
            context=context,
        )
        self.alpha = alpha
        self.identity = {
            "backend": self.name,
            "setup": self.prover.ready["setup_sha256"],
            "schedule": self.schedule["sha256"],
            "weights": weights,
            "alpha": alpha,
            "decoding": self.prover.ready["decoding"],
        }
        self.counter = 0

    def prepare_input(self, prompt: str) -> tuple:
        """Tokenize without silently truncating the user's query."""
        tokens = tuple(self.tokenizer.encode(prompt, add_special_tokens=False))
        if not 2 <= len(tokens) < 64:
            raise ValueError("GPT-2 prompt must tokenize to between 2 and 63 tokens")
        return tokens

    def max_output(self, inputs: tuple, requested: int) -> int:
        """Bound decoding by the registered setup's context limit."""
        return min(requested, 64 - len(inputs))

    def complexity(self, inputs: tuple, output_length: int) -> int:
        """Charge the public operation vector using the frozen weight schedule."""
        counts = reference_counts(len(inputs), output_length, self.schedule)["combined"]
        cost = sum(counts.values()) if self.weights is None else weighted_cost(counts, self.weights)
        if cost < 1:
            raise ValueError("complexity weights must charge every supported execution")
        return cost

    def randomness_count(self, inputs: tuple, cap: int) -> int:
        """Reserve N prompt-noise values and cap independent decoding values."""
        return len(inputs) + cap

    def run(self, inputs: tuple, randomness: tuple[bytes, ...], cap: int) -> Execution:
        """Prove the current challenge; reject absent or unverified proof artifacts."""
        n = len(inputs)
        if len(randomness) != n + cap:
            raise ValueError("wrong GPT-2 randomness schedule length")
        scale = self.alpha * self.prover.ready["embedding_std"]
        noise = [[scale * x for x in gaussian_vector(z, 768)] for z in randomness[:n]]
        uniforms = [int.from_bytes(z[:8], "big") >> 11 for z in randomness[n:]]
        location = self.directory / f"pair-{self.counter:06d}"
        self.counter += 1
        result = self.prover.run(
            {
                "request_id": location.name,
                "prompt_tokens": list(inputs),
                "noise": noise,
                "uniforms": uniforms,
                "max_output": cap,
                "mode": "prove",
                "output_directory": str(location.resolve()),
            }
        )
        proof = (location / "proof.bin").read_bytes()
        if result.get("verified") is not True or sha256(proof).hex() != result["proof_sha256"]:
            raise RuntimeError("DeepProve proof verification or digest failed")
        output = (
            canonical(
                {
                    "tokens": result["output_tokens"],
                    "logits_scale": result["logits_scale"],
                    "logits_shape": result["logits_shape"],
                }
            )
            + b"\n"
            + (location / "logits.i64").read_bytes()
        )
        return Execution(
            output,
            result["output_length"],
            self.complexity(inputs, result["output_length"]),
            result["inference_seconds"] + result["proof_seconds"],
            proof,
            {**result, "proof_directory": str(location), "proof_kind": self.proof_kind},
        )

    def close(self):
        """Release the persistent Rust worker and its GPU allocations."""
        self.prover.close()


class EZKLDiffusionBackend:
    """Prove one tiny 8x8 U-Net pass with EZKL; this is not full DDPM."""

    name = "diffusion"
    proof_kind = "ezkl-single-denoiser"

    def __init__(self, artifacts="model"):
        self.artifacts = Path(artifacts)
        required = ("network.ezkl", "settings.json", "vk.key", "pk.key", "kzg.srs")
        missing = [name for name in required if not (self.artifacts / name).is_file()]
        if missing:
            raise FileNotFoundError(f"EZKL setup missing {missing}; run scripts/setup_model.sh")
        self.identity = {
            "backend": self.name,
            "circuit": sha256((self.artifacts / "network.ezkl").read_bytes()).hex(),
            "verifier": sha256((self.artifacts / "vk.key").read_bytes()).hex(),
            "steps": 1,
            "complexity": 1,
        }

    def prepare_input(self, prompt: str) -> tuple:
        """Map demo text to fixed conditioning; this circuit has no text encoder."""
        return tuple(gaussian_vector(sha256(prompt.encode()), 64))

    def max_output(self, inputs: tuple, requested: int) -> int:
        """Each request produces one fixed-shaped denoiser output."""
        return 1

    def complexity(self, inputs: tuple, output_length: int) -> int:
        """The paper normalizes one fixed-shape diffusion pair to C=1."""
        if len(inputs) != 64 or output_length != 1:
            raise ValueError("invalid tiny-denoiser shape")
        return 1

    def randomness_count(self, inputs: tuple, cap: int) -> int:
        """The single-pass circuit consumes one independent initial noise vector."""
        return 1

    def run(self, inputs: tuple, randomness: tuple[bytes, ...], cap: int) -> Execution:
        """Create and independently verify a genuine EZKL inference proof."""
        from .zkp import run_inference_and_prove, verify_proof

        if len(randomness) != 1:
            raise ValueError("single-pass denoiser needs one randomness value")
        start = time.perf_counter()
        output, proof = run_inference_and_prove(
            list(inputs), gaussian_vector(randomness[0], 64), str(self.artifacts)
        )
        duration = time.perf_counter() - start
        if not verify_proof(proof, str(self.artifacts)):
            raise RuntimeError("EZKL proof verification failed")
        return Execution(
            canonical(output),
            1,
            1,
            duration,
            proof,
            {"verified": True, "proof_kind": self.proof_kind},
        )

    def close(self):
        """No persistent worker is held by the EZKL adapter."""


class SmokeBackend:
    """Explicit test double; never represents measured neural inference or ZK."""

    name = "smoke"
    proof_kind = "test-double"
    identity = {"backend": "smoke", "version": 1}

    def prepare_input(self, prompt):
        """Encode a fixture input without external models."""
        return tuple(prompt.encode())

    def max_output(self, inputs, requested):
        """Respect the fixture's requested output limit."""
        return requested

    def complexity(self, inputs, output_length):
        """Provide variable work for protocol tests."""
        return len(inputs) + output_length

    def randomness_count(self, inputs, cap):
        """Reserve one noise value and one value per possible output."""
        return 1 + cap

    def run(self, inputs, randomness, cap):
        """Return a deterministic fixture with a labelled virtual duration."""
        k = 1 + int.from_bytes(randomness[0][:4], "big") % cap
        output = sha256(canonical(inputs), canonical(randomness), canonical(k))
        complexity = self.complexity(inputs, k)
        return Execution(
            output,
            k,
            complexity,
            complexity / 10,
            sha256(b"smoke", output),
            {"verified": False, "proof_kind": self.proof_kind},
        )

    def close(self):
        """The test double owns no resources."""
