"""Run the GPT-2 generated-prefix collision experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from poml_sim.gpt2_compatibility import MODEL_REVISION, load_model  # noqa: E402
from poml_sim.gpt2_collision import (  # noqa: E402
    DEFAULT_TEMPERATURES,
    load_prompts,
    run_collision_experiment,
)


def _floats(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated temperatures")
    return values


def main(argv: list[str] | None = None) -> int:
    """Load the pinned GPT-2 checkpoint and run the collision campaign."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompts", type=int, default=500)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--temperatures", type=_floats, default=DEFAULT_TEMPERATURES)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--smoke", action="store_true", help="run four prompts and eight output tokens"
    )
    parser.add_argument("--no-resume", action="store_true", help="start a fresh checkpoint")
    args = parser.parse_args(argv)
    if (
        min(
            args.prompts,
            args.prompt_tokens,
            args.pairs,
            args.length,
            args.bootstrap_replicates,
            args.progress_every,
        )
        < 1
    ):
        parser.error("prompt, pair, length, bootstrap, and progress values must be positive")
    if args.smoke:
        args.prompts = min(args.prompts, 4)
        args.length = min(args.length, 8)
        args.bootstrap_replicates = min(args.bootstrap_replicates, 100)
    args.output.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.device)
    prompts = load_prompts(
        tokenizer,
        count=args.prompts,
        prompt_tokens=args.prompt_tokens,
        seed=args.seed,
    )
    (args.output / "prompts.json").write_text(
        json.dumps(
            [
                {
                    "id": prompt.prompt_id,
                    "source": prompt.source,
                    "text": prompt.text,
                    "token_ids": list(prompt.token_ids),
                }
                for prompt in prompts
            ],
            indent=2,
        )
        + "\n"
    )
    metadata = {
        "experiment": "gpt2_prefix_collision",
        "model": "openai-community/gpt2",
        "revision": MODEL_REVISION,
        "dataset": "wikitext2",
        "prompt_count": len(prompts),
        "prompt_tokens": args.prompt_tokens,
        "pairs_per_prompt": args.pairs,
        "temperatures": list(args.temperatures),
        "generation_length": args.length,
        "seed": args.seed,
        "device": args.device,
        "sampling": "inverse CDF over the full temperature-scaled vocabulary; no top-k or top-p filtering",
        "randomness": "independent repository sign-then-hash VRF uniforms for every trace and step",
        "eos": "stop at tokenizer EOS; pairs are censored after either trace stops",
        "model_config_sha256": hashlib.sha256(model.config.to_json_string().encode()).hexdigest(),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    run_collision_experiment(
        model,
        tokenizer,
        prompts,
        pairs=args.pairs,
        temperatures=args.temperatures,
        length=args.length,
        seed=args.seed,
        device=args.device,
        bootstrap_replicates=args.bootstrap_replicates,
        output_dir=args.output,
        resume=not args.no_resume,
        progress_every=args.progress_every,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
