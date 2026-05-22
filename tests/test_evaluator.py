"""Tests for ThresholdEvaluator.evaluate() and bootstrap_ci()."""

import numpy as np
import pytest
from pydantic import ValidationError

from oncothresh import ThresholdEvaluator, compare_models
from oncothresh._results import (
    BootstrapResult,
    CompareModelsResult,
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
    """Perfect calibration on 4 samples — gives clean, hand-verifiable NB values."""
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
    """prevalence is derived once from clinical_threshold — does not move with pt."""
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


def test_decision_curve_pt_at_or_above_one_returns_nan():
    """pt >= 1.0 makes pt/(1-pt) undefined — must return NaN, not crash."""
    result = _dca_evaluator().decision_curve(clinical_threshold=0.5, thresholds=[0.50, 1.0])
    assert not np.isnan(result.net_benefit_model[0])
    assert np.isnan(result.net_benefit_model[1])
    assert np.isnan(result.net_benefit_all[1])


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


# compare_models()
#
# Two evaluators used throughout:
#   ev_perfect: y_pred == y_true → all metrics = 1.0 at threshold=0.5
#   ev_known:   known confusion matrix at threshold=0.5:
#               TP=2, FP=1, FN=1, TN=2 → sensitivity=specificity=ppv=npv=2/3


def test_compare_models_returns_correct_type():
    result = compare_models([_perfect_evaluator(), _known_evaluator()], threshold=0.5)
    assert isinstance(result, CompareModelsResult)


def test_compare_models_result_count_matches_evaluators():
    result = compare_models([_perfect_evaluator(), _known_evaluator()], threshold=0.5)
    assert len(result.results) == 2
    assert len(result.model_names) == 2


def test_compare_models_threshold_stored():
    result = compare_models([_perfect_evaluator(), _known_evaluator()], threshold=0.5)
    assert result.threshold == pytest.approx(0.5)


def test_compare_models_default_names():
    result = compare_models([_perfect_evaluator(), _known_evaluator()], threshold=0.5)
    assert result.model_names == ["Model 1", "Model 2"]


def test_compare_models_custom_names():
    result = compare_models(
        [_perfect_evaluator(), _known_evaluator()],
        threshold=0.5,
        model_names=["UNI", "CONCH"],
    )
    assert result.model_names == ["UNI", "CONCH"]


def test_compare_models_metrics_match_individual_evaluate():
    """Each model's metrics in CompareModelsResult must equal evaluate() called directly."""
    ev1 = _perfect_evaluator()
    ev2 = _known_evaluator()
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
    ev_perfect = _perfect_evaluator()
    ev_known = _known_evaluator()
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


def test_compare_models_name_length_mismatch_raises():
    with pytest.raises(ValueError, match="model_names length"):
        compare_models(
            [_perfect_evaluator(), _known_evaluator()],
            threshold=0.5,
            model_names=["Only One"],
        )


def test_compare_models_result_is_frozen():
    result = compare_models([_perfect_evaluator(), _known_evaluator()], threshold=0.5)
    with pytest.raises(ValidationError):
        result.threshold = 0.99  # type: ignore[misc]


def test_compare_models_str_contains_model_names():
    result = compare_models(
        [_perfect_evaluator(), _known_evaluator()],
        threshold=0.5,
        model_names=["UNI", "CONCH"],
    )
    s = str(result)
    assert "UNI" in s
    assert "CONCH" in s


def test_compare_models_str_contains_threshold():
    result = compare_models([_perfect_evaluator(), _known_evaluator()], threshold=0.5)
    assert "0.50" in str(result)


def test_compare_models_str_contains_n_models():
    result = compare_models([_perfect_evaluator(), _known_evaluator()], threshold=0.5)
    assert "n_models=2" in str(result)
