import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_complexity_pairs import fit_campaign, make_design


def test_design_includes_non_power_two_boundaries_and_repeats():
    design = make_design(64, 100, 5, 123)
    assert 100 < len(design) <= 120
    totals = {n + k for n, k in design}
    assert {7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64} <= totals
    assert all(n >= 2 and k >= 1 and n + k <= 64 for n, k in design)


def test_staged_fit_reports_all_models_on_synthetic_rows():
    rows = []
    for n in range(2, 10):
        for k in range(1, 4):
            s = n + k
            rows.append({
                "min_user_len": n,
                "max_context": s,
                "inference_time": 10 + 2 * s + s * s,
                "prove_full": 100 + 3 * (2 ** ((s - 1).bit_length()))
                + 4 * (2 ** ((s - 1).bit_length())) * ((s - 1).bit_length()),
            })
    result = fit_campaign(pd.DataFrame(rows))
    assert result["n_cells"] == len(rows)
    assert {"inference_ab", "proof_c0c1c2", "combined"} <= set(result["models"])
    assert "prove_full_c0c1c2" in result["models"]
