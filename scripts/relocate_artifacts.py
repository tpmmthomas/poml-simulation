"""Move this workspace's bulk artifacts with verified copies and stable symlinks.

Run only while experiment, build and package-install processes are stopped.
The destination journal permits recovery after an interrupted copy or cutover.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BULK_DIRECTORIES = ("experiments/results", "models", "model_cache", ".scratch", ".venv")


def write_journal(path: Path, journal: dict) -> None:
    """Publish a complete migration journal atomically."""
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(journal, indent=2) + "\n")
    temporary.replace(path)


def transfer(source: Path, destination: Path, verify: bool) -> None:
    """Copy metadata/hardlinks or checksum-verify every file without mutation."""
    options = ["rsync", "-aH", "--no-owner", "--no-group"]
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        paths = [str(source) + "/", str(destination) + "/"]
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        paths = [str(source), str(destination)]
    if verify:
        result = subprocess.run(
            options + ["--dry-run", "--delete", "--itemize-changes"] + paths,
            check=True,
            text=True,
            capture_output=True,
        )
        if result.stdout.strip():
            raise RuntimeError(
                f"Copy verification failed; original retained:\n{result.stdout}"
            )
        verify_contents(source, destination)
    else:
        subprocess.run(options + ["--partial", "--info=progress2"] + paths, check=True)


def verify_contents(source: Path, destination: Path) -> None:
    """Compare SHA-256 for all regular files with bounded parallel NAS reads."""

    def checksum(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.digest()

    def pairs():
        if source.is_file():
            yield source, destination
            return
        seen = set()
        for parent, _, names in os.walk(source, followlinks=False):
            for name in names:
                path = Path(parent) / name
                if path.is_symlink() or not path.is_file():
                    continue
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                yield path, destination / path.relative_to(source)

    def check(pair):
        original, copied = pair
        if checksum(original) != checksum(copied):
            raise RuntimeError(
                f"Copy verification failed; original retained: {original}"
            )

    # rsync above checks structure, link targets, hardlinks and metadata. Hash
    # independent files concurrently to avoid a single NAS read stream bottleneck.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(check, pairs()):
            pass


def relocate(root: Path, storage: Path, relative: str, journal: dict) -> None:
    """Verify the destination before replacing one local artifact with a link."""
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("artifact path must be relative to the repository")
    source = root / relative
    destination = storage / relative
    backup = source.with_name(f".{source.name}.nas-move-source")
    log = storage / "migration.json"
    entry = journal["artifacts"].get(relative)
    if source.is_symlink():
        if source.resolve() != destination.resolve() or not destination.exists():
            raise ValueError(f"unexpected or broken existing symlink: {source}")
        if backup.exists() and entry is not None and entry["state"] == "linked":
            # Cleanup may already have removed some files. Never copy this partial
            # rollback tree over the complete destination verified before cutover.
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
        if not backup.exists():
            if entry is None or entry["state"] not in {
                "verified",
                "linked",
                "complete",
            }:
                raise ValueError(
                    f"existing link has no verified migration record: {source}"
                )
            entry["state"] = "complete"
            write_journal(log, journal)
            print(f"Already on destination: {relative}", flush=True)
            return
    original = backup if backup.exists() else source
    if not original.exists():
        print(f"Absent, skipped: {relative}", flush=True)
        return
    if entry is None:
        if destination.exists():
            raise FileExistsError(
                f"unrelated destination already exists: {destination}"
            )
        entry = journal["artifacts"][relative] = {
            "source": str(source),
            "destination": str(destination),
            "state": "copying",
        }
        write_journal(log, journal)
    print(f"Copying {relative} to {destination}", flush=True)
    transfer(original, destination, verify=False)
    print(
        f"Checksum-verifying {relative}; local copy retained until verified", flush=True
    )
    transfer(original, destination, verify=True)
    entry["state"] = "verified"
    entry["verified_utc"] = datetime.now(timezone.utc).isoformat()
    write_journal(log, journal)
    # The adjacent rename preserves an intact rollback source until the link exists.
    if not backup.exists():
        source.rename(backup)
    elif source.exists() and not source.is_symlink():
        raise FileExistsError(f"both source and recovery copy exist: {source}")
    if not source.is_symlink():
        try:
            source.symlink_to(destination, target_is_directory=destination.is_dir())
        except BaseException:
            backup.rename(source)
            raise
    entry["state"] = "linked"
    write_journal(log, journal)
    if backup.is_dir():
        shutil.rmtree(backup)
    else:
        backup.unlink()
    entry["state"] = "complete"
    write_journal(log, journal)
    print(f"Moved and verified: {relative}", flush=True)


def main() -> None:
    """Relocate generated files only; keep tracked source files in the workspace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    storage = args.destination.resolve()
    if storage == ROOT or ROOT in storage.parents:
        raise ValueError("destination must be outside the repository")
    storage.mkdir(parents=True, exist_ok=True)
    journal_path = storage / "migration.json"
    journal = (
        json.loads(journal_path.read_text())
        if journal_path.exists()
        else {"repository": str(ROOT), "destination": str(storage), "artifacts": {}}
    )
    if journal["repository"] != str(ROOT) or journal["destination"] != str(storage):
        raise ValueError("migration journal belongs to another workspace")
    generated = sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "model").iterdir()
        if path.suffix in {".key", ".srs", ".ezkl", ".onnx", ".data", ".json"}
    )
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", *BULK_DIRECTORIES, *generated],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    if tracked:
        raise ValueError(f"refusing to relocate tracked files: {os.fsdecode(tracked)}")
    for relative in (*BULK_DIRECTORIES, *generated):
        relocate(ROOT, storage, relative, journal)
    print(f"All artifacts relocated. Journal: {journal_path}", flush=True)


if __name__ == "__main__":
    main()
