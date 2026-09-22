"""Persistent fresh GPT-2/DeepProve execution; proof artifacts are never replayed."""

import json
import os
from pathlib import Path
import selectors
import subprocess
import time

from tqdm import tqdm


class LiveProver:
    """Serve serialized actual executions on one CUDA prover and a fixed model."""

    def __init__(
        self,
        binary: Path,
        directory: Path,
        device: str = "cuda:0",
        context: int = 64,
        threads: int = 8,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        timeout: float = 1800.0,
    ):
        if not device.startswith("cuda:"):
            raise ValueError("the live campaign requires a CUDA prover")
        self.timeout = timeout
        directory.mkdir(parents=True, exist_ok=True)
        self.log = (directory / "prover.stderr.log").open("a")
        self.stdout_log = (directory / "prover.stdout.log").open("a")
        env = os.environ.copy()
        index = int(device.split(":")[1])
        if index < 0:
            raise ValueError("CUDA index must be nonnegative")
        visible = env.get("CUDA_VISIBLE_DEVICES")
        env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[index] if visible else str(index)
        env.setdefault("RUST_LOG", "error")
        # Fix representative-input quantization across worker restarts/miners.
        env["RNG_SEED"] = "20260915"
        env["POML_SETUP_DIRECTORY"] = str(directory.resolve())
        self.process = subprocess.Popen(
            [
                str(binary.resolve()),
                "--context",
                str(context),
                "--threads",
                str(threads),
                "--temperature",
                str(temperature),
                "--top-k",
                str(top_k),
                "--top-p",
                str(top_p),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            env=env,
        )
        try:
            self.ready = self._read("DeepProve setup")
            if self.ready.get("event") != "ready" or self.ready.get("backend") != "cuda":
                raise RuntimeError(f"invalid prover handshake: {self.ready}")
        except BaseException:
            self.close()
            raise

    def _read(self, description):
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        start = time.monotonic()
        with tqdm(desc=description, unit="s", leave=False, mininterval=5) as progress:
            try:
                while True:
                    if time.monotonic() - start > self.timeout:
                        raise TimeoutError(
                            f"DeepProve timed out during {description}; see {self.log.name}"
                        )
                    if not selector.select(timeout=1):
                        progress.update(int(time.monotonic() - start) - progress.n)
                        continue
                    line = self.process.stdout.readline()
                    if not line:
                        self.process.wait()
                        raise RuntimeError(
                            f"DeepProve exited {self.process.returncode}; see {self.log.name}"
                        )
                    self.stdout_log.write(line)
                    self.stdout_log.flush()
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict) and ("event" in value or "request_id" in value):
                        return value
            finally:
                selector.close()

    def run(self, request: dict) -> dict:
        """Run a new inference/proof even if a previous request used the same prompt."""
        path = Path(request["output_directory"])
        if path.exists() and any(path.iterdir()):
            raise ValueError(f"refusing to reuse an existing attempt directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
        (path / "request.json").write_text(json.dumps(request) + "\n")
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        result = self._read(f"{request['mode']} {request['request_id']}")
        (path / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if result.get("request_id") != request["request_id"]:
            raise RuntimeError("prover response identity mismatch")
        if request["mode"] == "prove" and not result.get("verified"):
            raise RuntimeError("fresh inference proof did not verify")
        return result

    def close(self):
        """Release the persistent worker even after a failed request."""
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        self.log.close()
        self.stdout_log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
