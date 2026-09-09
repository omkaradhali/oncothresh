"""Tests for ThresholdEvaluator.evaluate() and bootstrap_ci()."""

import numpy as np
import pytest
from pydantic import ValidationError

from oncothresh import ThresholdEvaluator, compare_models
from oncothresh._results import (
    BiasAnalysisResult,
    BootstrapResult,
    BoundaryCalibrationResult,
    CompareModelsResult,
    DecisionCurveResult,
    MultiThresholdReport,
    NNTResult,
    SubgroupResult,
    ThresholdResult,
    ThresholdSensitivityResult,
)


# Fixtures
def _perfect_evaluator(n: int = 100, threshold: float = 0.20) -> ThresholdEvaluator:
    """Model that perfectly replicates ground truth."""
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 1, n)
    return ThresholdEvaluator(y_true=y_true, y_pred=y_true.copy())


def _known_evaluator() -> ThresholdEvaluator:
    """
    Handcrafted case at threshold=0.5:
      y_true: [0.8, 0.9, 0.1, 0.2, 0.7, 0.3]   positives: idx 0,1,4  negatives: idx 2,3,5
      y_pred: [0.8, 0.2, 0.1, 0.9, 0.7, 0.3]   positives: idx 0,3,4  negatives: idx 1,2,5

    Confusion at threshold=0.5:
      TP=2 (idx 0,4), FP=1 (idx 3), FN=1 (idx 1), TN=2 (idx 2,5)
    """
    y_true = [0.8, 0.9, 0.1, 0.2, 0.7, 0.3]
    y_pred = [0.8, 0.2, 0.1, 0.9, 0.7, 0.3]
    return ThresholdEvaluator(y_true=y_true, y_pred=y_pred)


# Constructor validation
def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        ThresholdEvaluator(y_true=[0.1, 0.2], y_pred=[0.1])


def test_2d_array_raises():
    with pytest.raises(ValueError, match="1-D"):
        ThresholdEvaluator(y_true=[[0.1, 0.2]], y_pred=[[0.1, 0.2]])


def test_too_few_samples_raises():
    with pytest.raises(ValueError, match="At least 2"):
        ThresholdEvaluator(y_true=[0.5], y_pred=[0.5])


def test_list_inputs_converted_to_ndarray():
    ev = ThresholdEvaluator(y_true=[0.1, 0.9], y_pred=[0.2, 0.8])
    assert isinstance(ev.y_true, np.ndarray)
    assert isinstance(ev.y_pred, np.ndarray)


def test_nan_in_y_true_raises():
    with pytest.raises(ValueError, match="finite"):
        ThresholdEvaluator(y_true=[0.1, np.nan], y_pred=[0.1, 0.2])


def test_inf_in_y_pred_raises():
    with pytest.raises(ValueError, match="finite"):
        ThresholdEvaluator(y_true=[0.1, 0.2], y_pred=[0.1, np.inf])


# evaluate(): perfect model
def test_perfect_model_sensitivity_and_specificity():
    ev = _perfect_evaluator()
    result = ev.evaluate(threshold=0.20)
    assert result.sensitivity == pytest.approx(1.0)
    assert result.specificity == pytest.approx(1.0)


def test_perfect_model_f1_and_mcc():
    ev = _perfect_evaluator()
    result = ev.evaluate(threshold=0.20)
    assert result.f1 == pytest.approx(1.0)
    assert result.mcc == pytest.approx(1.0)


def test_perfect_model_accuracy():
    ev = _perfect_evaluator()
    result = ev.evaluate(threshold=0.20)
    assert result.accuracy == pytest.approx(1.0)


# evaluate(): known case
def test_known_case_sensitivity():
    # TP=2, FN=1 → sensitivity=2/3
    result = _known_evaluator().evaluate(threshold=0.5)
    assert result.sensitivity == pytest.approx(2 / 3, abs=1e-9)


def test_known_case_specificity():
    # TN=2, FP=1 → specificity=2/3
    result = _known_evaluator().evaluate(threshold=0.5)
    assert result.specificity == pytest.approx(2 / 3, abs=1e-9)


def test_known_case_ppv():
    # TP=2, FP=1 → PPV=2/3
    result = _known_evaluator().evaluate(threshold=0.5)
    assert result.ppv == pytest.approx(2 / 3, abs=1e-9)


def test_known_case_npv():
    # TN=2, FN=1 → NPV=2/3
    result = _known_evaluator().evaluate(threshold=0.5)
    assert result.npv == pytest.approx(2 / 3, abs=1e-9)


def test_known_case_accuracy():
    # TP+TN=4, total=6 → accuracy=4/6=2/3
    result = _known_evaluator().evaluate(threshold=0.5)
    assert result.accuracy == pytest.approx(2 / 3, abs=1e-9)


def test_known_case_counts():
    result = _known_evaluator().evaluate(threshold=0.5)
    assert result.n_total == 6
    assert result.n_positive == 3  # tp + fn
    assert result.n_negative == 3  # tn + fp


def test_result_is_frozen():
    result = _known_evaluator().evaluate(threshold=0.5)
    with pytest.raises(ValidationError):
        result.sensitivity = 1.0  # type: ignore[misc]


def test_result_str_contains_threshold():
    result = _known_evaluator().evaluate(threshold=0.5)
    assert "0.50" in str(result)


# evaluate(): edge cases
def test_all_predicted_negative_sensitivity_is_zero():
    """Model always predicts 0, sensitivity must be 0, specificity 1."""
    y_true = [0.1, 0.9, 0.8, 0.2]
    y_pred = [0.0, 0.0, 0.0, 0.0]
    ev = ThresholdEvaluator(y_true=y_true, y_pred=y_pred)
    result = ev.evaluate(threshold=0.5)
    assert result.sensitivity == pytest.approx(0.0)
    assert result.specificity == pytest.approx(1.0)


def test_all_predicted_positive_specificity_is_zero():
    y_true = [0.1, 0.9, 0.8, 0.2]
    y_pred = [1.0, 1.0, 1.0, 1.0]
    ev = ThresholdEvaluator(y_true=y_true, y_pred=y_pred)
    result = ev.evaluate(threshold=0.5)
    assert result.specificity == pytest.approx(0.0)
    assert result.sensitivity == pytest.approx(1.0)


def test_zero_division_ppv_returns_zero():
    """If no samples predicted positive, PPV denominator is zero → return 0."""
    ev = ThresholdEvaluator(y_true=[0.1, 0.9], y_pred=[0.0, 0.0])
    result = ev.evaluate(threshold=0.5)
    assert result.ppv == pytest.approx(0.0)


def test_threshold_at_boundary():
    """Samples exactly at threshold count as positive (>=)."""
    ev = ThresholdEvaluator(y_true=[0.5, 0.5], y_pred=[0.5, 0.5])
    result = ev.evaluate(threshold=0.5)
    assert result.accuracy == pytest.approx(1.0)


def test_result_type():
    result = _known_evaluator().evaluate(threshold=0.5)
    assert isinstance(result, ThresholdResult)


# bootstrap_ci()
def test_bootstrap_result_type():
    ev = _perfect_evaluator()
    ci = ev.bootstrap_ci(threshold=0.20, n_bootstrap=50, random_state=42)
    assert isinstance(ci, BootstrapResult)


def test_bootstrap_ci_contains_point_estimate():
    """Point estimate must lie within the CI (trivially true for a perfect model)."""
    ev = _perfect_evaluator()
    ci = ev.bootstrap_ci(threshold=0.20, n_bootstrap=200, random_state=0)
    assert ci.sensitivity.lower <= ci.sensitivity.estimate <= ci.sensitivity.upper


def test_bootstrap_ci_lower_lte_upper():
    ev = _known_evaluator()
    ci = ev.bootstrap_ci(threshold=0.5, n_bootstrap=200, random_state=7)
    for attr in ("sensitivity", "specificity", "ppv", "npv", "f1", "accuracy"):
        interval = getattr(ci, attr)
        assert interval.lower <= interval.upper, f"{attr}: lower > upper"


def test_bootstrap_reproducible():
    ev = _known_evaluator()
    ci1 = ev.bootstrap_ci(threshold=0.5, n_bootstrap=100, random_state=42)
    ci2 = ev.bootstrap_ci(threshold=0.5, n_bootstrap=100, random_state=42)
    assert ci1.sensitivity.lower == ci2.sensitivity.lower
    assert ci1.sensitivity.upper == ci2.sensitivity.upper


def test_bootstrap_confidence_stored():
    ev = _perfect_evaluator()
    ci = ev.bootstrap_ci(threshold=0.20, n_bootstrap=50, confidence=0.90, random_state=1)
    assert ci.confidence == pytest.approx(0.90)
    assert ci.sensitivity.confidence == pytest.approx(0.90)


def test_bootstrap_n_stored():
    ev = _perfect_evaluator()
    ci = ev.bootstrap_ci(threshold=0.20, n_bootstrap=77, random_state=1)
    assert ci.n_bootstrap == 77


def test_bootstrap_ci_str():
    ev = _perfect_evaluator()
    ci = ev.bootstrap_ci(threshold=0.20, n_bootstrap=50, random_state=1)
    s = str(ci.sensitivity)
    assert "95% CI" in s


def test_bootstrap_default_method_is_bca():
    ev = _known_evaluator()
    ci = ev.bootstrap_ci(threshold=0.5, n_bootstrap=200, random_state=3)
    assert ci.method == "bca"


def test_bootstrap_percentile_method_selectable():
    ev = _known_evaluator()
    ci = ev.bootstrap_ci(threshold=0.5, n_bootstrap=200, random_state=3, method="percentile")
    assert ci.method == "percentile"
    for attr in ("sensitivity", "specificity", "ppv", "npv", "f1", "accuracy"):
        interval = getattr(ci, attr)
        assert interval.lower <= interval.upper


def test_bootstrap_rejects_unknown_method():
    ev = _known_evaluator()
    with pytest.raises(ValueError, match="method must be"):
        ev.bootstrap_ci(threshold=0.5, method="jackknife")


def test_bootstrap_perfect_model_degenerate_ci_collapses_to_estimate():
    """BCa is undefined when a metric never varies (perfect model). The interval then
    collapses to the point estimate rather than producing nan bounds."""
    ev = _perfect_evaluator()
    ci = ev.bootstrap_ci(threshold=0.20, n_bootstrap=200, random_state=0)
    assert ci.sensitivity.lower == pytest.approx(1.0)
    assert ci.sensitivity.upper == pytest.approx(1.0)
    assert ci.sensitivity.estimate == pytest.approx(1.0)


def test_bootstrap_result_str_contains_method():
    ev = _known_evaluator()
    ci = ev.bootstrap_ci(threshold=0.5, n_bootstrap=100, random_state=1)
    s = str(ci)
    assert "BootstrapResult" in s
    assert "method=bca" in s


# multi_threshold_report()
def test_multi_threshold_report_returns_correct_type():
    ev = _perfect_evaluator()
    report = ev.multi_threshold_report(thresholds=[0.20, 0.50])
    assert isinstance(report, MultiThresholdReport)


def test_multi_threshold_report_count():
    ev = _perfect_evaluator()
    report = ev.multi_threshold_report(thresholds=[0.20, 0.50])
    assert len(report.results) == 2


def test_multi_threshold_report_thresholds_match():
    ev = _perfect_evaluator()
    report = ev.multi_threshold_report(thresholds=[0.20, 0.50])
    assert report.thresholds == [0.20, 0.50]


def test_multi_threshold_empty_raises():
    ev = _perfect_evaluator()
    with pytest.raises(ValueError, match="empty"):
        ev.multi_threshold_report(thresholds=[])


# decision_curve()
#
# DCA semantics (Vickers 2006):
#   - clinical_threshold defines the disease label, fixed across the sweep:
#       y_true_bin = (y_true >= clinical_threshold)
#   - y_pred is a calibrated probability P(y_true >= clinical_threshold), in [0, 1].
#   - The pt sweep moves the clinician's intervention threshold:
#       y_pred_bin(pt) = (y_pred >= pt)
#
# Known case used throughout:
#   clinical_threshold = 0.5
#   y_true = [0.1, 0.2, 0.8, 0.9]  → y_true_bin = [0, 0, 1, 1], prevalence = 0.5, N = 4
#   y_pred = [0.1, 0.2, 0.8, 0.9]  (perfectly calibrated probabilities)
#
# At pt=0.50:  y_pred_bin=[0,0,1,1]
#   TP=2, FP=0, harm_weight=1.0
#   NB_model = 2/4 - 0/4 * 1.0  = 0.5
#   NB_all   = 0.5 - 0.5 * 1.0  = 0.0
#
# At pt=0.20:  y_pred_bin=[0,1,1,1]  (0.2 >= 0.2 is True)
#   TP=2, FP=1, harm_weight=0.25
#   NB_model = 2/4 - 1/4 * 0.25 = 0.4375
#   NB_all   = 0.5 - 0.5 * 0.25 = 0.375


def _dca_evaluator() -> ThresholdEvaluator:
    """Perfect calibration on 4 samples, gives clean, hand-verifiable NB values."""
    y = [0.1, 0.2, 0.8, 0.9]
    return ThresholdEvaluator(y_true=y, y_pred=y)


def _dca_evaluator_with_fp() -> ThresholdEvaluator:
    """Model flags index 1 (y_true=0.2, y_pred=0.6) as positive at pt=0.50.

    Under clinical_threshold=0.5: y_true_bin=[0,0,1,1], prevalence=0.5, N=4.
    At pt=0.50: y_pred_bin=[0,1,1,1] → TP=2, FP=1.
    """
    y_true = [0.1, 0.2, 0.8, 0.9]
    y_pred = [0.1, 0.6, 0.8, 0.9]
    return ThresholdEvaluator(y_true=y_true, y_pred=y_pred)


def test_decision_curve_returns_correct_type():
    result = _dca_evaluator().decision_curve(clinical_threshold=0.5, thresholds=[0.20, 0.50])
    assert isinstance(result, DecisionCurveResult)


def test_decision_curve_output_lengths_match():
    """thresholds, nb_model, and nb_all must all have the same length."""
    result = _dca_evaluator().decision_curve(clinical_threshold=0.5, thresholds=[0.20, 0.50, 0.80])
    assert len(result.thresholds) == 3
    assert len(result.net_benefit_model) == 3
    assert len(result.net_benefit_all) == 3


def test_decision_curve_default_thresholds_length():
    """Default sweep is 99 points (np.linspace(0.01, 0.99, 99))."""
    result = _dca_evaluator().decision_curve(clinical_threshold=0.5)
    assert len(result.thresholds) == 99


def test_decision_curve_clinical_threshold_stored():
    """The result is self-describing: clinical_threshold and prevalence are persisted."""
    result = _dca_evaluator().decision_curve(clinical_threshold=0.5, thresholds=[0.50])
    assert result.clinical_threshold == pytest.approx(0.5)
    assert result.prevalence == pytest.approx(0.5)


def test_decision_curve_prevalence_fixed_across_sweep():
    """prevalence is derived once from clinical_threshold, does not move with pt."""
    result = _dca_evaluator().decision_curve(
        clinical_threshold=0.5, thresholds=[0.10, 0.20, 0.50, 0.80]
    )
    # Sanity: under clinical_threshold=0.5, two of four samples are positive.
    assert result.prevalence == pytest.approx(0.5)


def test_decision_curve_nb_model_perfect_at_pt_050():
    # Perfect probs, clinical_threshold=0.5, pt=0.50: TP=2, FP=0 → NB = 2/4 = 0.5
    result = _dca_evaluator().decision_curve(clinical_threshold=0.5, thresholds=[0.50])
    assert result.net_benefit_model[0] == pytest.approx(0.5, abs=1e-9)


def test_decision_curve_nb_all_at_pt_050():
    # prevalence=0.5, harm_weight=1.0 → NB_all = 0.5 - 0.5*1.0 = 0.0
    result = _dca_evaluator().decision_curve(clinical_threshold=0.5, thresholds=[0.50])
    assert result.net_benefit_all[0] == pytest.approx(0.0, abs=1e-9)


def test_decision_curve_nb_model_at_pt_020():
    # clinical_threshold=0.5, pt=0.20: y_pred_bin=[0,1,1,1] → TP=2, FP=1
    # NB_model = 2/4 - 1/4 * 0.25 = 0.4375
    result = _dca_evaluator().decision_curve(clinical_threshold=0.5, thresholds=[0.20])
    assert result.net_benefit_model[0] == pytest.approx(0.4375, abs=1e-9)


def test_decision_curve_nb_all_at_pt_020():
    # prevalence=0.5 (fixed), harm_weight=0.25 → NB_all = 0.5 - 0.5*0.25 = 0.375
    result = _dca_evaluator().decision_curve(clinical_threshold=0.5, thresholds=[0.20])
    assert result.net_benefit_all[0] == pytest.approx(0.375, abs=1e-9)


def test_decision_curve_fp_reduces_nb_model():
    # Adding one FP at pt=0.50: TP=2, FP=1, harm=1.0 → NB = 0.5 - 0.25 = 0.25
    result = _dca_evaluator_with_fp().decision_curve(clinical_threshold=0.5, thresholds=[0.50])
    assert result.net_benefit_model[0] == pytest.approx(0.25, abs=1e-9)


def test_decision_curve_nb_none_is_always_zero():
    """net_benefit_none is a convenience property that always returns zeros."""
    result = _dca_evaluator().decision_curve(clinical_threshold=0.5, thresholds=[0.20, 0.50, 0.80])
    assert result.net_benefit_none == [0.0, 0.0, 0.0]


def test_decision_curve_nb_all_can_be_negative():
    """
    When prevalence under clinical_threshold is 0, treat-all causes pure harm.

    y_true = [0.1, 0.2, 0.3, 0.4], clinical_threshold=0.5 → y_true_bin all 0,
    prevalence=0. At pt=0.50: NB_all = 0 - 1.0 * 1.0 = -1.0.
    """
    ev = ThresholdEvaluator(y_true=[0.1, 0.2, 0.3, 0.4], y_pred=[0.1, 0.2, 0.3, 0.4])
    result = ev.decision_curve(clinical_threshold=0.5, thresholds=[0.50])
    assert result.net_benefit_all[0] == pytest.approx(-1.0, abs=1e-9)
    assert result.prevalence == pytest.approx(0.0)


def test_decision_curve_rejects_pt_at_or_above_one():
    """pt >= 1.0 makes pt/(1-pt) diverge, so it is rejected rather than returning NaN."""
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\)"):
        _dca_evaluator().decision_curve(clinical_threshold=0.5, thresholds=[0.50, 1.0])


def test_decision_curve_rejects_negative_pt():
    """A negative probability threshold is invalid."""
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\)"):
        _dca_evaluator().decision_curve(clinical_threshold=0.5, thresholds=[-0.1, 0.5])


def test_decision_curve_perfect_model_beats_treat_all():
    """A perfectly-calibrated model should have NB >= treat-all at every pt."""
    result = _dca_evaluator().decision_curve(
        clinical_threshold=0.5, thresholds=[0.10, 0.20, 0.50, 0.80]
    )
    for nb_m, nb_a in zip(result.net_benefit_model, result.net_benefit_all, strict=True):
        if np.isnan(nb_m) or np.isnan(nb_a):
            continue
        assert nb_m >= nb_a - 1e-9


def test_decision_curve_result_is_frozen():
    result = _dca_evaluator().decision_curve(clinical_threshold=0.5, thresholds=[0.50])
    with pytest.raises(ValidationError):
        result.thresholds = [0.99]  # type: ignore[misc]


def test_decision_curve_str_describes_inputs():
    result = _dca_evaluator().decision_curve(clinical_threshold=0.5, thresholds=[0.20, 0.50])
    text = str(result)
    assert "n_points=2" in text
    assert "clinical_threshold=0.50" in text
    assert "prevalence=" in text


def test_decision_curve_rejects_clinical_threshold_out_of_range():
    """clinical_threshold must be in [0, 1]."""
    ev = _dca_evaluator()
    with pytest.raises(ValueError, match="clinical_threshold"):
        ev.decision_curve(clinical_threshold=1.5)
    with pytest.raises(ValueError, match="clinical_threshold"):
        ev.decision_curve(clinical_threshold=-0.1)


def test_decision_curve_rejects_y_pred_outside_unit_interval():
    """y_pred must be a probability (DCA harm-weight assumes it). Raise loud."""
    ev = ThresholdEvaluator(y_true=[0.1, 0.2, 0.8, 0.9], y_pred=[0.0, 0.5, 1.2, 0.9])
    with pytest.raises(ValueError, match="y_pred must lie in"):
        ev.decision_curve(clinical_threshold=0.5)


# nnt()
#
# Reuses _known_evaluator (TP=2, FP=1, FN=1, TN=2 at threshold=0.5 → PPV=NPV=2/3).
# nnt_positive = 1/PPV = 1.5. nnt_negative = 1/(1-NPV) = 3.0.


def test_nnt_returns_correct_type():
    result = _known_evaluator().nnt(threshold=0.5)
    assert isinstance(result, NNTResult)


def test_nnt_positive_known_case():
    result = _known_evaluator().nnt(threshold=0.5)
    assert result.nnt_positive == pytest.approx(1.5, abs=1e-9)


def test_nnt_negative_known_case():
    result = _known_evaluator().nnt(threshold=0.5)
    assert result.nnt_negative == pytest.approx(3.0, abs=1e-9)


def test_nnt_ppv_npv_match_evaluate():
    """The stored PPV/NPV must match what evaluate() reports at the same threshold."""
    ev = _known_evaluator()
    point = ev.evaluate(threshold=0.5)
    nnt = ev.nnt(threshold=0.5)
    assert nnt.ppv == pytest.approx(point.ppv)
    assert nnt.npv == pytest.approx(point.npv)


def test_nnt_perfect_model_returns_inf_negative():
    """NPV=1 means no clearance ever hides a missed positive → nnt_negative = inf."""
    ev = ThresholdEvaluator(y_true=[0.1, 0.2, 0.8, 0.9], y_pred=[0.1, 0.2, 0.8, 0.9])
    result = ev.nnt(threshold=0.5)
    assert result.nnt_positive == pytest.approx(1.0)
    assert result.nnt_negative == float("inf")


def test_nnt_zero_ppv_returns_inf_positive():
    """PPV=0 means every flag is wrong → nnt_positive = inf (clinical meaning is clear)."""
    # All flagged samples are true negatives. No true positives flagged.
    ev = ThresholdEvaluator(
        y_true=[0.1, 0.2, 0.8, 0.9],
        y_pred=[0.9, 0.8, 0.2, 0.1],  # inverted predictions
    )
    result = ev.nnt(threshold=0.5)
    assert result.nnt_positive == float("inf")


def test_nnt_counts_match_evaluate():
    result = _known_evaluator().nnt(threshold=0.5)
    assert result.n_total == 6
    assert result.n_positive == 3
    assert result.n_negative == 3


def test_nnt_result_is_frozen():
    result = _known_evaluator().nnt(threshold=0.5)
    with pytest.raises(ValidationError):
        result.nnt_positive = 2.0  # type: ignore[misc]


def test_nnt_str_handles_inf():
    """__str__ must render inf as a clean symbol, not 'inf'."""
    ev = ThresholdEvaluator(y_true=[0.1, 0.2, 0.8, 0.9], y_pred=[0.1, 0.2, 0.8, 0.9])
    result = ev.nnt(threshold=0.5)
    assert "∞" in str(result)


# threshold_sensitivity()
#
# These tests exist primarily to lock down the linspace-clamp bug that previously
# allowed the requested step size to be silently compressed when the window hit a
# [0, 1] boundary. The key invariant: consecutive thresholds differ by exactly
# `step`, regardless of clamping.


def test_threshold_sensitivity_returns_correct_type():
    result = _known_evaluator().threshold_sensitivity(threshold=0.5)
    assert isinstance(result, ThresholdSensitivityResult)


def test_threshold_sensitivity_step_size_preserved_when_not_clamped():
    """Standard case (no boundary clip): 11 points at step 0.01 over [0.45, 0.55]."""
    result = _known_evaluator().threshold_sensitivity(threshold=0.5, delta=0.05, step=0.01)
    assert len(result.thresholds) == 11
    diffs = np.diff(result.thresholds)
    assert np.allclose(diffs, 0.01, atol=1e-9)


def test_threshold_sensitivity_step_size_preserved_when_low_clamped():
    """
    Regression test for the linspace-clamp bug.

    threshold=0.02, delta=0.05, step=0.01. Without the fix linspace ran over
    [0, 0.07] with 11 points (step 0.007). With the fix the negative offsets are
    dropped and the surviving points retain step 0.01.
    """
    result = _known_evaluator().threshold_sensitivity(threshold=0.02, delta=0.05, step=0.01)
    diffs = np.diff(result.thresholds)
    assert np.allclose(diffs, 0.01, atol=1e-9)
    # All points must lie in [0, 1].
    assert all(0.0 <= t <= 1.0 for t in result.thresholds)


def test_threshold_sensitivity_step_size_preserved_when_high_clamped():
    """Same bug, mirrored at the upper boundary."""
    result = _known_evaluator().threshold_sensitivity(threshold=0.98, delta=0.05, step=0.01)
    diffs = np.diff(result.thresholds)
    assert np.allclose(diffs, 0.01, atol=1e-9)
    assert all(0.0 <= t <= 1.0 for t in result.thresholds)


def test_threshold_sensitivity_nominal_index_points_at_threshold():
    result = _known_evaluator().threshold_sensitivity(threshold=0.5, delta=0.05, step=0.01)
    assert result.thresholds[result.nominal_index] == pytest.approx(0.5, abs=1e-9)


def test_threshold_sensitivity_shifts_are_signed_offsets():
    result = _known_evaluator().threshold_sensitivity(threshold=0.5, delta=0.05, step=0.01)
    assert result.shifts[0] == pytest.approx(-0.05, abs=1e-9)
    assert result.shifts[-1] == pytest.approx(0.05, abs=1e-9)
    assert result.shifts[result.nominal_index] == pytest.approx(0.0, abs=1e-9)


def test_threshold_sensitivity_arrays_aligned():
    """thresholds, shifts, sensitivities, specificities must all have the same length."""
    result = _known_evaluator().threshold_sensitivity(threshold=0.5, delta=0.05, step=0.01)
    n = len(result.thresholds)
    assert len(result.shifts) == n
    assert len(result.sensitivities) == n
    assert len(result.specificities) == n


def test_threshold_sensitivity_rejects_invalid_params():
    ev = _known_evaluator()
    with pytest.raises(ValueError, match="delta"):
        ev.threshold_sensitivity(threshold=0.5, delta=0.0)
    with pytest.raises(ValueError, match="step"):
        ev.threshold_sensitivity(threshold=0.5, delta=0.05, step=0.0)


def test_threshold_sensitivity_result_is_frozen():
    result = _known_evaluator().threshold_sensitivity(threshold=0.5)
    with pytest.raises(ValidationError):
        result.thresholds = [0.5]  # type: ignore[misc]


def test_threshold_sensitivity_str_summarizes_sweep():
    result = _known_evaluator().threshold_sensitivity(threshold=0.5, delta=0.05, step=0.01)
    s = str(result)
    assert "ThresholdSensitivityResult" in s
    assert "nominal=0.50" in s


# boundary_calibration()
#
# Boundary-weighted ECE inside [threshold - window, threshold + window]. With a
# perfectly-calibrated model (y_pred == y_true) inside the window, ECE = 0.0.


def _boundary_eval_perfect() -> ThresholdEvaluator:
    """20 samples evenly spread across [0.10, 0.30] with perfectly-calibrated predictions."""
    y = np.linspace(0.10, 0.30, 20)
    return ThresholdEvaluator(y_true=y, y_pred=y.copy())


def test_boundary_calibration_returns_correct_type():
    result = _boundary_eval_perfect().boundary_calibration(threshold=0.20)
    assert isinstance(result, BoundaryCalibrationResult)


def test_boundary_calibration_perfect_model_ece_is_zero():
    """y_pred == y_true inside the boundary zone → mean(pred) == mean(true) per bin → ECE = 0."""
    result = _boundary_eval_perfect().boundary_calibration(threshold=0.20, window=0.10, n_bins=5)
    assert result.ece == pytest.approx(0.0, abs=1e-9)


def test_boundary_calibration_biased_model_ece_positive():
    """Systematic upward bias inside the zone → ECE > 0."""
    y_true = np.linspace(0.10, 0.30, 20)
    y_pred = np.clip(y_true + 0.05, 0.0, 1.0)
    ev = ThresholdEvaluator(y_true=y_true, y_pred=y_pred)
    result = ev.boundary_calibration(threshold=0.20, window=0.10, n_bins=5)
    assert result.ece > 0.0
    assert result.ece == pytest.approx(0.05, abs=1e-9)


def test_boundary_calibration_excludes_predictions_outside_window():
    """Samples outside [threshold ± window] must not contribute to ECE."""
    # 10 samples inside [0.10, 0.30] perfectly calibrated. 10 samples outside the
    # window with large bias, the ECE should still come out as 0.0 because
    # boundary_calibration filters by y_pred ∈ [threshold ± window].
    y_in = np.linspace(0.12, 0.28, 10)
    y_out = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.05, 0.04, 0.03, 0.02, 0.01])
    y_true = np.concatenate([y_in, y_out])
    y_pred = np.concatenate([y_in.copy(), np.zeros(10)])  # outside samples grossly miscalibrated
    ev = ThresholdEvaluator(y_true=y_true, y_pred=y_pred)
    result = ev.boundary_calibration(threshold=0.20, window=0.10, n_bins=5)
    assert result.ece == pytest.approx(0.0, abs=1e-9)


def test_boundary_calibration_empty_zone_returns_nan():
    """If no predictions fall inside the boundary zone, ECE is NaN, not a crash."""
    y_true = np.array([0.80, 0.85, 0.90, 0.95])
    y_pred = np.array([0.80, 0.85, 0.90, 0.95])  # all far from threshold=0.20
    ev = ThresholdEvaluator(y_true=y_true, y_pred=y_pred)
    result = ev.boundary_calibration(threshold=0.20, window=0.05, n_bins=5)
    assert np.isnan(result.ece)
    assert result.n_samples == 0


def test_boundary_calibration_rejects_invalid_params():
    ev = _boundary_eval_perfect()
    with pytest.raises(ValueError, match="window"):
        ev.boundary_calibration(threshold=0.20, window=0.0)
    with pytest.raises(ValueError, match="n_bins"):
        ev.boundary_calibration(threshold=0.20, n_bins=0)


def test_boundary_calibration_result_is_frozen():
    result = _boundary_eval_perfect().boundary_calibration(threshold=0.20)
    with pytest.raises(ValidationError):
        result.ece = 0.5  # type: ignore[misc]


def test_boundary_calibration_is_reliable_true_when_well_populated():
    """100 samples over 5 bins (average 20 per bin) clears the >= 5 per bin rule."""
    y = np.linspace(0.10, 0.30, 100)
    ev = ThresholdEvaluator(y_true=y, y_pred=y.copy())
    result = ev.boundary_calibration(threshold=0.20, window=0.10, n_bins=5)
    assert result.n_samples == 100
    assert result.is_reliable is True
    assert "sparse" not in str(result)


def test_boundary_calibration_is_reliable_false_when_sparse():
    """8 samples over 5 bins falls below the >= 5 per bin rule, so is_reliable is False."""
    y = np.linspace(0.12, 0.28, 8)
    ev = ThresholdEvaluator(y_true=y, y_pred=y.copy())
    result = ev.boundary_calibration(threshold=0.20, window=0.10, n_bins=5)
    assert result.is_reliable is False
    assert "sparse boundary zone" in str(result)


def test_boundary_calibration_empty_zone_str():
    y = np.array([0.80, 0.85, 0.90, 0.95])
    ev = ThresholdEvaluator(y_true=y, y_pred=y.copy())
    result = ev.boundary_calibration(threshold=0.20, window=0.05, n_bins=5)
    assert "no boundary samples" in str(result)


# compare_models()
#
# compare_models requires every evaluator to share one test set (equal-length y_true),
# so both fixtures below are built from the same 6-sample cohort:
#   perfect: y_pred == y_true, all metrics = 1.0 at threshold=0.5
#   known:   TP=2, FP=1, FN=1, TN=2 at threshold=0.5 -> sensitivity=specificity=ppv=npv=2/3


def _compare_pair() -> tuple[ThresholdEvaluator, ThresholdEvaluator]:
    """A perfect model and the known 2/3 model on the same 6-sample cohort."""
    y_true = [0.8, 0.9, 0.1, 0.2, 0.7, 0.3]
    y_pred_known = [0.8, 0.2, 0.1, 0.9, 0.7, 0.3]
    perfect = ThresholdEvaluator(y_true=y_true, y_pred=list(y_true))
    known = ThresholdEvaluator(y_true=y_true, y_pred=y_pred_known)
    return perfect, known


def test_compare_models_returns_correct_type():
    result = compare_models(list(_compare_pair()), threshold=0.5)
    assert isinstance(result, CompareModelsResult)


def test_compare_models_result_count_matches_evaluators():
    result = compare_models(list(_compare_pair()), threshold=0.5)
    assert len(result.results) == 2
    assert len(result.model_names) == 2


def test_compare_models_threshold_stored():
    result = compare_models(list(_compare_pair()), threshold=0.5)
    assert result.threshold == pytest.approx(0.5)


def test_compare_models_default_names():
    result = compare_models(list(_compare_pair()), threshold=0.5)
    assert result.model_names == ["Model 1", "Model 2"]


def test_compare_models_custom_names():
    result = compare_models(
        list(_compare_pair()),
        threshold=0.5,
        model_names=["UNI", "CONCH"],
    )
    assert result.model_names == ["UNI", "CONCH"]


def test_compare_models_metrics_match_individual_evaluate():
    """Each model's metrics in CompareModelsResult must equal evaluate() called directly."""
    ev1, ev2 = _compare_pair()
    direct1 = ev1.evaluate(threshold=0.5)
    direct2 = ev2.evaluate(threshold=0.5)

    compared = compare_models([ev1, ev2], threshold=0.5)

    for metric in ("sensitivity", "specificity", "ppv", "npv", "f1", "mcc", "accuracy"):
        assert getattr(compared.results[0], metric) == pytest.approx(
            getattr(direct1, metric), abs=1e-9
        ), f"Model 1 {metric} mismatch"
        assert getattr(compared.results[1], metric) == pytest.approx(
            getattr(direct2, metric), abs=1e-9
        ), f"Model 2 {metric} mismatch"


def test_compare_models_three_models():
    rng = np.random.default_rng(99)
    y_true = rng.uniform(0, 1, 50)
    ev1 = ThresholdEvaluator(y_true=y_true, y_pred=y_true.copy())
    ev2 = ThresholdEvaluator(y_true=y_true, y_pred=rng.uniform(0, 1, 50))
    ev3 = ThresholdEvaluator(y_true=y_true, y_pred=rng.uniform(0, 1, 50))

    result = compare_models([ev1, ev2, ev3], threshold=0.5, model_names=["A", "B", "C"])
    assert len(result.results) == 3
    assert result.model_names == ["A", "B", "C"]


def test_compare_models_results_order_preserved():
    """Results must appear in the same order as the evaluators list."""
    ev_perfect, ev_known = _compare_pair()
    result = compare_models([ev_perfect, ev_known], threshold=0.5)

    # Perfect model: sensitivity=1.0. Known model: sensitivity=2/3.
    assert result.results[0].sensitivity == pytest.approx(1.0)
    assert result.results[1].sensitivity == pytest.approx(2 / 3, abs=1e-9)


def test_compare_models_too_few_evaluators_raises():
    with pytest.raises(ValueError, match="at least 2"):
        compare_models([_perfect_evaluator()], threshold=0.5)


def test_compare_models_empty_evaluators_raises():
    with pytest.raises(ValueError, match="at least 2"):
        compare_models([], threshold=0.5)


def test_compare_models_rejects_mismatched_cohorts():
    """Evaluators scored on different-size test sets cannot be compared head-to-head."""
    ev_a = _perfect_evaluator(n=100)
    ev_b = _known_evaluator()  # n=6
    with pytest.raises(ValueError, match="same test set"):
        compare_models([ev_a, ev_b], threshold=0.5)


def test_compare_models_name_length_mismatch_raises():
    with pytest.raises(ValueError, match="model_names length"):
        compare_models(
            list(_compare_pair()),
            threshold=0.5,
            model_names=["Only One"],
        )


def test_compare_models_result_is_frozen():
    result = compare_models(list(_compare_pair()), threshold=0.5)
    with pytest.raises(ValidationError):
        result.threshold = 0.99  # type: ignore[misc]


def test_compare_models_str_contains_model_names():
    result = compare_models(
        list(_compare_pair()),
        threshold=0.5,
        model_names=["UNI", "CONCH"],
    )
    s = str(result)
    assert "UNI" in s
    assert "CONCH" in s


def test_compare_models_str_contains_threshold():
    result = compare_models(list(_compare_pair()), threshold=0.5)
    assert "0.50" in str(result)


def test_compare_models_str_contains_n_models():
    result = compare_models(list(_compare_pair()), threshold=0.5)
    assert "n_models=2" in str(result)


# bias_analysis()
def _scanner_metadata() -> dict[str, list[str]]:
    """Aligned with _known_evaluator(): idx 0-2 -> Scanner A, idx 3-5 -> Scanner B.

    _known_evaluator() confusion at threshold=0.5: TP=2 (idx 0,4), FP=1 (idx 3),
    FN=1 (idx 1), TN=2 (idx 2,5). Split by scanner:
      Scanner A (idx 0,1,2): TP=1 (idx0), FN=1 (idx1), TN=1 (idx2)
        -> sensitivity=0.5, specificity=1.0
      Scanner B (idx 3,4,5): FP=1 (idx3), TP=1 (idx4), TN=1 (idx5)
        -> sensitivity=1.0, specificity=0.5
    """
    return {
        "scanner": ["Scanner A", "Scanner A", "Scanner A", "Scanner B", "Scanner B", "Scanner B"]
    }


def test_bias_analysis_returns_correct_type():
    result = _known_evaluator().bias_analysis(_scanner_metadata(), threshold=0.5)
    assert isinstance(result, BiasAnalysisResult)
    assert all(isinstance(g, SubgroupResult) for g in result.by_column["scanner"])


def test_bias_analysis_overall_matches_evaluate():
    ev = _known_evaluator()
    result = ev.bias_analysis(_scanner_metadata(), threshold=0.5)
    assert result.overall == ev.evaluate(threshold=0.5)


def test_bias_analysis_group_labels_sorted():
    result = _known_evaluator().bias_analysis(_scanner_metadata(), threshold=0.5)
    labels = [g.group for g in result.by_column["scanner"]]
    assert labels == ["Scanner A", "Scanner B"]


def test_bias_analysis_group_sensitivity_and_specificity():
    result = _known_evaluator().bias_analysis(_scanner_metadata(), threshold=0.5)
    by_group = {g.group: g for g in result.by_column["scanner"]}

    assert by_group["Scanner A"].sensitivity == pytest.approx(0.5)
    assert by_group["Scanner A"].specificity == pytest.approx(1.0)
    assert by_group["Scanner B"].sensitivity == pytest.approx(1.0)
    assert by_group["Scanner B"].specificity == pytest.approx(0.5)


def test_bias_analysis_false_rates_are_complements():
    result = _known_evaluator().bias_analysis(_scanner_metadata(), threshold=0.5)
    for group in result.by_column["scanner"]:
        assert group.false_negative_rate == pytest.approx(1.0 - group.sensitivity)
        assert group.false_positive_rate == pytest.approx(1.0 - group.specificity)


def test_bias_analysis_group_counts():
    result = _known_evaluator().bias_analysis(_scanner_metadata(), threshold=0.5)
    for group in result.by_column["scanner"]:
        assert group.n_total == 3
        assert group.n_positive + group.n_negative == group.n_total


def test_bias_analysis_is_reliable_respects_min_group_size():
    result = _known_evaluator().bias_analysis(_scanner_metadata(), threshold=0.5, min_group_size=3)
    assert all(g.is_reliable for g in result.by_column["scanner"])

    strict = _known_evaluator().bias_analysis(_scanner_metadata(), threshold=0.5, min_group_size=4)
    assert all(not g.is_reliable for g in strict.by_column["scanner"])


def test_bias_analysis_multiple_columns():
    metadata = _scanner_metadata()
    metadata["batch"] = ["1", "1", "2", "2", "3", "3"]
    result = _known_evaluator().bias_analysis(metadata, threshold=0.5)
    assert set(result.by_column) == {"scanner", "batch"}
    assert len(result.by_column["batch"]) == 3


def test_bias_analysis_empty_metadata_raises():
    with pytest.raises(ValueError, match="at least one column"):
        _known_evaluator().bias_analysis({}, threshold=0.5)


def test_bias_analysis_mismatched_column_length_raises():
    with pytest.raises(ValueError, match="scanner"):
        _known_evaluator().bias_analysis({"scanner": ["A", "B"]}, threshold=0.5)


def test_bias_analysis_rejects_non_positive_min_group_size():
    with pytest.raises(ValueError, match="min_group_size"):
        _known_evaluator().bias_analysis(_scanner_metadata(), threshold=0.5, min_group_size=0)


def test_bias_analysis_stores_threshold_and_min_group_size():
    result = _known_evaluator().bias_analysis(_scanner_metadata(), threshold=0.5, min_group_size=2)
    assert result.threshold == 0.5
    assert result.min_group_size == 2


def test_bias_analysis_result_is_frozen():
    result = _known_evaluator().bias_analysis(_scanner_metadata(), threshold=0.5)
    with pytest.raises(ValidationError):
        result.threshold = 0.99  # type: ignore[misc]


def test_subgroup_result_is_frozen():
    result = _known_evaluator().bias_analysis(_scanner_metadata(), threshold=0.5)
    group = result.by_column["scanner"][0]
    with pytest.raises(ValidationError):
        group.n_total = 999  # type: ignore[misc]


def test_bias_analysis_str_contains_threshold():
    result = _known_evaluator().bias_analysis(_scanner_metadata(), threshold=0.5)
    assert "0.50" in str(result)


def test_bias_analysis_str_contains_group_labels():
    result = _known_evaluator().bias_analysis(_scanner_metadata(), threshold=0.5)
    s = str(result)
    assert "Scanner A" in s
    assert "Scanner B" in s


def test_subgroup_result_str_flags_unreliable_group():
    result = _known_evaluator().bias_analysis(_scanner_metadata(), threshold=0.5, min_group_size=10)
    group = result.by_column["scanner"][0]
    assert "unreliable" in str(group)


def _homogeneous_class_metadata() -> dict[str, list[str]]:
    """Aligned with _known_evaluator(): "R1" is the 3 ground-truth positives (idx 0,1,4),
    "R2" is the 3 ground-truth negatives (idx 2,3,5). Each group has zero of the other class.
    """
    return {"region": ["R1", "R1", "R2", "R2", "R1", "R2"]}


def test_bias_analysis_zero_positive_or_negative_group_is_not_reliable():
    result = _known_evaluator().bias_analysis(
        _homogeneous_class_metadata(), threshold=0.5, min_group_size=1
    )
    by_group = {g.group: g for g in result.by_column["region"]}

    assert by_group["R1"].n_positive == 3
    assert by_group["R1"].n_negative == 0
    assert not by_group["R1"].is_reliable

    assert by_group["R2"].n_positive == 0
    assert by_group["R2"].n_negative == 3
    assert not by_group["R2"].is_reliable


def test_bias_analysis_zero_class_group_does_not_raise():
    # Should not crash even though one class is entirely absent from each group.
    result = _known_evaluator().bias_analysis(_homogeneous_class_metadata(), threshold=0.5)
    assert len(result.by_column["region"]) == 2


def test_bias_analysis_rejects_string_metadata_column():
    with pytest.raises(ValueError, match="single string"):
        _known_evaluator().bias_analysis({"scanner": "Scanner A"}, threshold=0.5)


def test_bias_analysis_rejects_non_1d_metadata():
    metadata = {"scanner": [["A"], ["A"], ["A"], ["B"], ["B"], ["B"]]}
    with pytest.raises(ValueError, match="1-D"):
        _known_evaluator().bias_analysis(metadata, threshold=0.5)


def test_bias_analysis_rejects_nan_metadata():
    metadata = {"scanner": ["A", "A", "A", "B", "B", float("nan")]}
    with pytest.raises(ValueError, match="NaN"):
        _known_evaluator().bias_analysis(metadata, threshold=0.5)


def test_bias_analysis_rejects_unhashable_labels():
    metadata = {"scanner": [{"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}]}
    with pytest.raises(ValueError, match="unhashable"):
        _known_evaluator().bias_analysis(metadata, threshold=0.5)


def test_bias_analysis_rejects_numpy_float32_nan_metadata():
    metadata = {"scanner": ["A", "A", "A", "B", "B", np.float32("nan")]}
    with pytest.raises(ValueError, match="NaN"):
        _known_evaluator().bias_analysis(metadata, threshold=0.5)


def test_bias_analysis_rejects_inf_metadata():
    metadata = {"scanner": ["A", "A", "A", "B", "B", float("inf")]}
    with pytest.raises(ValueError, match="NaN or inf"):
        _known_evaluator().bias_analysis(metadata, threshold=0.5)
