"""Tests for ThresholdEvaluator.evaluate() and bootstrap_ci()."""

import numpy as np
import pytest
from pydantic import ValidationError

from oncothresh import ThresholdEvaluator
from oncothresh._results import BootstrapResult, MultiThresholdReport, ThresholdResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# evaluate() — perfect model
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# evaluate() — known case
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# evaluate() — edge cases
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# bootstrap_ci()
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# multi_threshold_report()
# ---------------------------------------------------------------------------


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
