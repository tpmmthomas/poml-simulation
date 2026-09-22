"""Apply the pinned DeepProve work-meter patches without modifying Cargo's cache."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEEP_REV = "9d1a53e2ef49ffa2c902b8689cd3c58057a4e662"
CRYPTO_REV = "22e8e93cd0f94a7638616a1ae22190feb0a6b275"


def git_environment() -> dict:
    """Skip unrelated upstream LFS demo assets; GPT-2 is fetched from HF."""
    # Upstream CNN/demo LFS objects are unnecessary for either GPT-2 worker.
    # Their availability must not prevent a clean source checkout from building.
    return {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a bounded Git operation with captured diagnostics."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=git_environment(),
    )


def apply_once(repo: Path, patch: Path, revision: str) -> None:
    """Apply only to the expected revision, accepting an already applied patch."""
    if git(repo, "rev-parse", "HEAD").stdout.strip() != revision:
        raise RuntimeError(f"{repo} must be at revision {revision}")
    if git(repo, "apply", "--reverse", "--check", str(patch), check=False).returncode == 0:
        return
    result = git(repo, "apply", "--check", str(patch), check=False)
    if result.returncode:
        raise RuntimeError(
            f"Patch does not cleanly apply to {repo}; local changes were preserved:\n{result.stderr}"
        )
    git(repo, "apply", str(patch))


def prepare(deep_prove: Path) -> None:
    """Create an isolated crypto checkout and apply both versioned patches."""
    crypto = deep_prove.parent / "dp-crypto-work"
    if not deep_prove.is_dir():
        deep_prove.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/Lagrange-Labs/deep-prove.git", str(deep_prove)],
            check=True,
            env=git_environment(),
        )
        git(deep_prove, "checkout", "--detach", DEEP_REV)
    if not crypto.exists():
        candidates = sorted((Path.home() / ".cargo/git/checkouts").glob("dp-crypto-*/22e8e93"))
        if candidates:
            shutil.copytree(candidates[0], crypto, ignore=shutil.ignore_patterns(".cargo-ok"))
        else:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/Lagrange-Labs/dp-crypto.git",
                    str(crypto),
                ],
                check=True,
                env=git_environment(),
            )
            git(crypto, "checkout", "--detach", CRYPTO_REV)
    apply_once(crypto, ROOT / "scripts/dp_crypto_work.patch", CRYPTO_REV)
    apply_once(deep_prove, ROOT / "scripts/deepprove_work.patch", DEEP_REV)


def main() -> None:
    """Prepare the default scratch checkout or an explicitly selected checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep-prove", type=Path, default=ROOT / ".scratch/deep-prove")
    args = parser.parse_args()
    prepare(args.deep_prove.resolve())
    print("DeepProve work instrumentation is ready")


if __name__ == "__main__":
    main()
