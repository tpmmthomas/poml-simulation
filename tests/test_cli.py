"""User-facing commands produce complete reports and preserve existing outputs."""

import json
from poml_sim.cli import main
from poml_sim.lottery import LIMIT
import pytest


def test_cli_smoke_writes_a_complete_protocol_report(tmp_path):
    assert (
        main(
            [
                "--backend",
                "smoke",
                "--blocks",
                "2",
                "--miners",
                "2",
                "--queries",
                "4",
                "--difficulty",
                str(LIMIT),
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["adopted_blocks"] == 2
    assert report["backend"] == "smoke"
    assert all(row["proof_kind"] == "test-double" for row in report["executions"])
    with pytest.raises(ValueError, match="already contains"):
        main(["--output", str(tmp_path)])


def test_cli_exhausted_race_is_reported_as_incomplete(tmp_path):
    assert (
        main(
            [
                "--backend",
                "smoke",
                "--blocks",
                "1",
                "--miners",
                "1",
                "--queries",
                "1",
                "--difficulty",
                "1",
                "--output",
                str(tmp_path),
            ]
        )
        == 1
    )
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["adopted_blocks"] == 0
    assert report["blocks"][0]["termination_reason"] == "query_pool_exhausted"
