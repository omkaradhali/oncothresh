"""Tests for ThresholdEvaluator.evaluate() and bootstrap_ci()."""

import numpy as np
import pytest
from pydantic import ValidationError

from oncothresh import ThresholdEvaluator
from oncothresh._results import (
    BootstrapResult,
    DecisionCurveResult,
    MultiThresholdReport,
    ThresholdResult,
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


# evaluate() — perfect model
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


# evaluate() — known case
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


# evaluate() — edge cases
def test_all_predicted_negative_sensitivity_is_zero():
    """Model always predicts 0 — sensitivity must be 0, specificity 1."""
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


@pytest.mark.filterwarnings("ignore::UserWarning")
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
# Known case used throughout:
#   y_true = [0.8, 0.9, 0.1, 0.2]
#   y_pred = [0.8, 0.9, 0.1, 0.2]  ← perfect model
#
# At pt=0.50:  y_true_bin=[1,1,0,0], y_pred_bin=[1,1,0,0]
#   TP=2, FP=0, N=4, harm_weight=1.0, prevalence=0.5
#   NB_model  = 2/4 - 0/4 * 1.0 = 0.50
#   NB_all    = 0.5  - 0.5  * 1.0 = 0.00
#
# At pt=0.20:  y_true_bin=[1,1,0,1], y_pred_bin=[1,1,0,1]  (0.2 >= 0.2 is True)
#   TP=3, FP=0, N=4, harm_weight=0.25, prevalence=0.75
#   NB_model  = 3/4 - 0   = 0.75
#   NB_all    = 0.75 - 0.25*0.25 = 0.6875

def _dca_evaluator() -> ThresholdEvaluator:
    """Perfect model on 4 samples — gives clean, hand-verifiable NB values."""
    y = [0.8, 0.9, 0.1, 0.2]
    return ThresholdEvaluator(y_true=y, y_pred=y)


def _dca_evaluator_with_fp() -> ThresholdEvaluator:
    """Model with one false positive at pt=0.50 (index 2: true=0.1, pred=0.6)."""
    y_true = [0.8, 0.9, 0.1, 0.2]
    y_pred = [0.8, 0.9, 0.6, 0.2]
    return ThresholdEvaluator(y_true=y_true, y_pred=y_pred)


def test_decision_curve_returns_correct_type():
    result = _dca_evaluator().decision_curve(thresholds=[0.20, 0.50])
    assert isinstance(result, DecisionCurveResult)


def test_decision_curve_output_lengths_match():
    """thresholds, nb_model, and nb_all must all have the same length."""
    result = _dca_evaluator().decision_curve(thresholds=[0.20, 0.50, 0.80])
    assert len(result.thresholds) == 3
    assert len(result.net_benefit_model) == 3
    assert len(result.net_benefit_all) == 3


def test_decision_curve_default_thresholds_length():
    """Default sweep is 99 points (np.linspace(0.01, 0.99, 99))."""
    result = _dca_evaluator().decision_curve()
    assert len(result.thresholds) == 99


def test_decision_curve_nb_model_perfect_at_pt_050():
    # Perfect model at pt=0.50: TP=2, FP=0, N=4, harm_weight=1.0 → NB=0.5
    result = _dca_evaluator().decision_curve(thresholds=[0.50])
    assert result.net_benefit_model[0] == pytest.approx(0.5, abs=1e-9)


def test_decision_curve_nb_all_at_pt_050():
    # Treat-all at pt=0.50: prevalence=0.5, harm_weight=1.0 → NB=0.0
    result = _dca_evaluator().decision_curve(thresholds=[0.50])
    assert result.net_benefit_all[0] == pytest.approx(0.0, abs=1e-9)


def test_decision_curve_nb_model_perfect_at_pt_020():
    # Perfect model at pt=0.20: TP=3, FP=0, N=4, harm_weight=0.25 → NB=0.75
    result = _dca_evaluator().decision_curve(thresholds=[0.20])
    assert result.net_benefit_model[0] == pytest.approx(0.75, abs=1e-9)


def test_decision_curve_nb_all_at_pt_020():
    # Treat-all at pt=0.20: prevalence=0.75, harm_weight=0.25 → NB=0.75-0.25*0.25=0.6875
    result = _dca_evaluator().decision_curve(thresholds=[0.20])
    assert result.net_benefit_all[0] == pytest.approx(0.6875, abs=1e-9)


def test_decision_curve_fp_reduces_nb_model():
    # Adding a false positive at pt=0.50 reduces NB: TP=2, FP=1, N=4 → NB=0.5-0.25=0.25
    result = _dca_evaluator_with_fp().decision_curve(thresholds=[0.50])
    assert result.net_benefit_model[0] == pytest.approx(0.25, abs=1e-9)


def test_decision_curve_nb_none_is_always_zero():
    """net_benefit_none is a convenience property that always returns zeros."""
    result = _dca_evaluator().decision_curve(thresholds=[0.20, 0.50, 0.80])
    assert result.net_benefit_none == [0.0, 0.0, 0.0]


def test_decision_curve_nb_all_can_be_negative():
    """
    At high pt where prevalence is low, treat-all NB goes negative.
    This is correct — do not clip. A negative NB means the strategy causes net harm.

    y_true = [0.1, 0.2, 0.3, 0.4], all below 0.50 → prevalence=0 at pt=0.50.
    NB_all = 0 - 1.0 * 1.0 = -1.0
    """
    ev = ThresholdEvaluator(y_true=[0.1, 0.2, 0.3, 0.4], y_pred=[0.1, 0.2, 0.3, 0.4])
    result = ev.decision_curve(thresholds=[0.50])
    assert result.net_benefit_all[0] == pytest.approx(-1.0, abs=1e-9)


def test_decision_curve_pt_at_or_above_one_returns_nan():
    """pt >= 1.0 makes pt/(1-pt) undefined — must return NaN, not crash."""
    result = _dca_evaluator().decision_curve(thresholds=[0.50, 1.0])
    assert not np.isnan(result.net_benefit_model[0])
    assert np.isnan(result.net_benefit_model[1])
    assert np.isnan(result.net_benefit_all[1])


def test_decision_curve_perfect_model_beats_treat_all():
    """A perfect model should have higher NB than treat-all at clinical thresholds."""
    result = _dca_evaluator().decision_curve(thresholds=[0.20, 0.50])
    for nb_m, nb_a in zip(result.net_benefit_model, result.net_benefit_all, strict=True):
        assert nb_m >= nb_a


def test_decision_curve_result_is_frozen():
    result = _dca_evaluator().decision_curve(thresholds=[0.50])
    with pytest.raises(ValidationError):
        result.thresholds = [0.99]  # type: ignore[misc]


def test_decision_curve_str_contains_n_points():
    result = _dca_evaluator().decision_curve(thresholds=[0.20, 0.50])
    assert "n_points=2" in str(result)
