"""Specifications for the reference accounting rules, independent of timing."""

import pytest

from poml_sim.gpt2_work import (
    ATTENTION_MACS,
    LINEAR_MACS,
    check_lengths,
    component_error,
    event_counts,
    inference_counts,
    inference_macs,
    ledger_counts,
    matmul_macs,
    next_power_of_two,
    proof_axis_macs,
    reference_counts,
    schedule_digest,
    weighted_cost,
)


@pytest.mark.parametrize("n,k", [(2, 1), (3, 4), (16, 17), (32, 32), (63, 1)])
def test_inference_formula_matches_explicit_pass_enumeration(n, k):
    # Independently enumerate prefill, cached calls, and final full trace.
    passes = [(n, n)] + [(1, n + j) for j in range(1, k + 1)] + [(n + k, n + k)]
    expected = sum(q * LINEAR_MACS + q * keys * ATTENTION_MACS for q, keys in passes)
    assert inference_macs(n, k) == expected


def test_dense_attention_counts_masked_entries():
    assert matmul_macs([12, 7, 64], [12, 64, 7]) == 12 * 7 * 64 * 7


def test_broadcast_matmul_and_incompatible_shapes():
    assert matmul_macs([1, 2, 3], [4, 3, 5]) == 4 * 2 * 3 * 5
    with pytest.raises(ValueError):
        matmul_macs([2, 2, 3], [4, 3, 5])


def test_sumcheck_charges_geometric_reductions_and_degree():
    result = event_counts(
        {
            "op": "sumcheck",
            "variables": 3,
            "degree": 2,
            "mles": 2,
            "terms": [2],
            "mle_variables": [3, 3],
            "mle_supports": [3, 3],
            "monomials": [[3, 3, 2]],
        }
    )
    assert result == {
        "sumcheck_fold": 14,
        "sumcheck_term": 21,
        "sumcheck_factor": 42,
        "sumcheck_round": 3,
    }


def test_batched_sumcheck_uses_each_polynomials_own_domain():
    result = event_counts(
        {
            "op": "sumcheck",
            "variables": 3,
            "mle_variables": [1],
            "mle_supports": [1],
            "monomials": [[1, 1, 2]],
        }
    )
    # Two inactive rounds sum two entries each; the one active round has
    # three evaluation points. The MLE is folded only once.
    assert result == {
        "sumcheck_fold": 1,
        "sumcheck_term": 7,
        "sumcheck_factor": 14,
        "sumcheck_round": 3,
    }


def test_legacy_sumcheck_ledger_is_rejected_instead_of_charging_largest_domain():
    with pytest.raises(ValueError, match="individual polynomial domains"):
        event_counts({"op": "sumcheck", "variables": 3, "mles": 2, "terms": [2]})


def test_implicit_zero_padding_only_scales_the_surviving_coefficient():
    result = event_counts(
        {
            "op": "sumcheck",
            "variables": 20,
            "mle_variables": [20],
            "mle_supports": [2],
            "monomials": [[20, 2, 2]],
        }
    )
    assert result == {
        "sumcheck_fold": 21,
        "sumcheck_term": 63,
        "sumcheck_factor": 126,
        "sumcheck_round": 20,
    }


def test_pcs_open_does_not_double_count_msm():
    result = event_counts({"op": "pcs_open", "length": 8, "variables": 3})
    assert result == {"pcs_fold": 6, "pcs_horner": 42, "pcs_division": 21}


def test_sumcheck_rounds_do_not_make_mle_fold_charge_quadratic():
    assert event_counts({"op": "mle_fix", "length": 16, "variables": 4}) == {"mle_fold": 15}


def test_missing_weights_and_negative_weights_are_rejected():
    with pytest.raises(ValueError, match="Missing"):
        weighted_cost({"integer_mac": 5}, {})
    with pytest.raises(ValueError, match="nonnegative"):
        weighted_cost({"integer_mac": 5}, {"integer_mac": -1})
    assert weighted_cost({"integer_mac": 5}, {"integer_mac": 3}) == 15


@pytest.mark.parametrize("n,k", [(1, 2), (2, 0), (64, 1), (-1, 1)])
def test_invalid_lengths_fail(n, k):
    with pytest.raises(ValueError):
        check_lengths(n, k)


def test_lengths_must_be_exact_integers():
    with pytest.raises(TypeError):
        check_lengths(2.0, 2)


def test_padding_boundaries():
    assert [next_power_of_two(n) for n in (7, 8, 9, 31, 32, 33, 64)] == [
        8,
        8,
        16,
        32,
        32,
        64,
        64,
    ]


def test_same_total_can_have_different_inference_work():
    assert inference_counts(2, 31) != inference_counts(32, 1)


def test_unverified_and_excluded_events_fail():
    with pytest.raises(ValueError, match="Unverified"):
        ledger_counts({"verified": False, "events": []})
    with pytest.raises(ValueError, match="phase"):
        ledger_counts({"verified": True, "events": [{"phase": "setup", "op": "msm", "length": 4}]})


def test_unknown_operations_fail_instead_of_being_silently_omitted():
    with pytest.raises(ValueError, match="Unknown"):
        event_counts({"op": "unknown"})


def test_schedule_digest_is_order_independent():
    assert schedule_digest({"a": 1, "b": 2}) == schedule_digest(
        {"b": 2, "a": 1, "sha256": "ignored"}
    )


def test_axis_reduction_formula_matches_equation_by_equation_count():
    s, d, h, f, layers, vocab = 17, 768, 12, 3072, 12, 50257
    qkv = 3 * (s * d + d * d + d * h)
    attention = 2 * s * d + h * s * s + s * d
    output = s * d + d * d
    ffn = s * d + d * f + s * f + f * d
    logits = s * d + d * vocab
    assert proof_axis_macs(s) == layers * (qkv + attention + output + ffn) + logits


def test_calculator_selects_padding_bucket_without_interpolating():
    schedule = {
        "version": "gpt2-deepprove-work-3",
        "setup_max": 64,
        "proof_profiles": {"16": {"msm_16": 2}, "32": {"msm_32": 3}},
    }
    schedule["sha256"] = schedule_digest(schedule)
    assert reference_counts(8, 8, schedule)["proof"]["msm_16"] == 2
    assert reference_counts(8, 9, schedule)["proof"]["msm_32"] == 3
    with pytest.raises(ValueError, match="No audited"):
        reference_counts(32, 32, schedule)


def test_calculator_rejects_changed_manifest():
    schedule = {
        "version": "gpt2-deepprove-work-3",
        "setup_max": 64,
        "proof_profiles": {"8": {"msm_8": 2}},
    }
    schedule["sha256"] = schedule_digest(schedule)
    schedule["proof_profiles"]["8"]["msm_8"] = 3
    with pytest.raises(ValueError, match="digest"):
        reference_counts(4, 4, schedule)


def test_point_encoding_length_is_diagnostic_not_gas():
    assert event_counts({"op": "transcript_append", "bytes": 16}) == {}
    assert event_counts({"op": "transcript_append", "bytes": 158}) == {}
    assert event_counts({"op": "transcript_points", "count": 1}) == {"transcript_points": 1}


def test_compiler_rejects_mixed_setup_and_unknown_schema():
    from poml_sim.gpt2_work import compile_schedule

    with pytest.raises(ValueError, match="schema"):
        compile_schedule([{"meta": {"schema": "unknown"}}], {})
    rows = [
        {
            "meta": {
                "schema": "gpt2-deepprove-work-3",
                "setup_max": 64,
                "setup_digest": digest,
            }
        }
        for digest in ("setup-a", "setup-b")
    ]
    with pytest.raises(ValueError, match="shared setup"):
        compile_schedule(rows, {})


def test_compiler_fails_on_held_out_mismatch_instead_of_fitting(monkeypatch):
    from poml_sim import gpt2_work

    # Isolate compilation from event expansion: five matching boundary plans
    # and one held-out plan with an unexpected extra commitment.
    rows = [
        {
            "meta": {
                "schema": gpt2_work.SCHEDULE_VERSION,
                "setup_max": 64,
                "setup_digest": "fixed",
                "n": 2,
                "k": s - 2,
            },
            "unexpected": s == 7,
        }
        for s in (4, 8, 16, 32, 64, 7)
    ]

    def expanded(row):
        n, k = row["meta"]["n"], row["meta"]["k"]
        return {
            "inference": gpt2_work.inference_counts(n, k),
            "proof": {
                "field_axis_mac": gpt2_work.proof_axis_macs(n + k),
                "msm_16": 1 + row["unexpected"],
            },
        }

    monkeypatch.setattr(gpt2_work, "ledger_counts", expanded)
    with pytest.raises(ValueError, match="padding rule mismatch"):
        gpt2_work.compile_schedule(rows, {})


def test_component_error_bounds_every_nonnegative_weighting():
    observed = {"integer_mac": 1000, "msm_4096": 388}
    predicted = {"integer_mac": 1000, "msm_4096": 387}
    bound = component_error(observed, predicted)
    assert bound == 1 / 388
    for weights in (
        {"integer_mac": 1, "msm_4096": 100000},
        {"integer_mac": 0, "msm_4096": 1},
        {"integer_mac": 17, "msm_4096": 0},
    ):
        a, p = weighted_cost(observed, weights), weighted_cost(predicted, weights)
        assert abs(a - p) / a <= bound


def test_absent_observed_component_cannot_hide_unbounded_weighted_error():
    assert component_error({}, {"msm_4096": 1}) == float("inf")
    assert component_error({}, {}) == 0


@pytest.mark.parametrize("reference, observed", [(1000, 1001), (374, 386)])
def test_small_adaptive_variation_does_not_rewrite_canonical_profile(
    monkeypatch, reference, observed
):
    from poml_sim import gpt2_work

    rows = [
        {
            "meta": {
                "schema": gpt2_work.SCHEDULE_VERSION,
                "setup_max": 64,
                "setup_digest": "fixed",
                "n": 2,
                "k": s - 2,
            }
        }
        for s in (4, 8, 16, 32, 64, 7)
    ]

    def expanded(row):
        n, k = row["meta"]["n"], row["meta"]["k"]
        return {
            "inference": gpt2_work.inference_counts(n, k),
            "proof": {
                "field_axis_mac": gpt2_work.proof_axis_macs(n + k),
                "msm_16": observed if n + k == 7 else reference,
            },
        }

    monkeypatch.setattr(gpt2_work, "ledger_counts", expanded)
    schedule = gpt2_work.compile_schedule(rows, {})
    assert schedule["proof_profiles"]["8"]["msm_16"] == reference
