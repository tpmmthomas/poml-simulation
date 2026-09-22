"""Artifact relocation must preserve data and recover from interrupted cutovers."""

import importlib.util
import os
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "relocate_artifacts",
    Path(__file__).resolve().parents[1] / "scripts/relocate_artifacts.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def setup_paths(tmp_path):
    root, storage = tmp_path / "repo", tmp_path / "nas"
    root.mkdir()
    storage.mkdir()
    return root, storage, {"artifacts": {}}


def test_moves_directory_preserving_data_hardlinks_and_relative_symlinks(tmp_path):
    root, storage, journal = setup_paths(tmp_path)
    source = root / "results"
    source.mkdir()
    (source / "proof").write_bytes(b"actual proof\x00\xff")
    (source / "linked").hardlink_to(source / "proof")
    (source / "relative").symlink_to("proof")
    module.relocate(root, storage, "results", journal)
    module.relocate(root, storage, "results", journal)
    assert source.is_symlink()
    assert (source / "relative").read_bytes() == b"actual proof\x00\xff"
    assert (source / "proof").stat().st_ino == (source / "linked").stat().st_ino
    assert journal["artifacts"]["results"]["state"] == "complete"
    assert not (root / ".results.nas-move-source").exists()


def test_checksum_failure_preserves_original(tmp_path, monkeypatch):
    root, storage, journal = setup_paths(tmp_path)
    source = root / "proof.bin"
    source.write_bytes(b"original proof")
    transfer = module.transfer

    def corrupt_before_verification(src, dst, verify):
        if verify:
            dst.write_bytes(b"modified proof")
            # Same size and timestamp defeat rsync's ordinary metadata check.
            stat = src.stat()
            os.utime(dst, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        transfer(src, dst, verify)

    monkeypatch.setattr(module, "transfer", corrupt_before_verification)
    with pytest.raises(RuntimeError, match="verification failed"):
        module.relocate(root, storage, "proof.bin", journal)
    assert not source.is_symlink()
    assert source.read_bytes() == b"original proof"


def test_resumes_after_source_renamed_before_link_created(tmp_path):
    root, storage, journal = setup_paths(tmp_path)
    (root / ".proof.bin.nas-move-source").write_bytes(b"proof")
    (storage / "proof.bin").write_bytes(b"proof")
    journal["artifacts"]["proof.bin"] = {"state": "verified"}
    module.relocate(root, storage, "proof.bin", journal)
    assert (root / "proof.bin").is_symlink()
    assert (root / "proof.bin").read_bytes() == b"proof"


def test_refuses_unrelated_destination(tmp_path):
    root, storage, journal = setup_paths(tmp_path)
    (root / "proof.bin").write_bytes(b"source")
    (storage / "proof.bin").write_bytes(b"other")
    with pytest.raises(FileExistsError):
        module.relocate(root, storage, "proof.bin", journal)
    assert (storage / "proof.bin").read_bytes() == b"other"


def test_resumes_partial_local_cleanup_without_changing_destination(tmp_path):
    root, storage, journal = setup_paths(tmp_path)
    destination = storage / "results"
    destination.mkdir()
    (destination / "proof").write_bytes(b"complete proof")
    (destination / "log").write_text("complete log")
    backup = root / ".results.nas-move-source"
    backup.mkdir()
    (backup / "log").write_text("complete log")
    (root / "results").symlink_to(destination)
    journal["artifacts"]["results"] = {"state": "linked"}
    module.relocate(root, storage, "results", journal)
    assert (destination / "proof").read_bytes() == b"complete proof"
    assert not backup.exists()
