"""
oncothresh: Clinical threshold evaluation for oncology AI models.

Quick start::

    from oncothresh import ThresholdEvaluator

    ev = ThresholdEvaluator(y_true=tc_scores, y_pred=model_predictions)

    # Single threshold
    result = ev.evaluate(threshold=0.20)

    # With confidence intervals
    ci = ev.bootstrap_ci(threshold=0.20, n_bootstrap=1000, random_state=42)

    # Side-by-side at clinical cutoffs
    report = ev.multi_threshold_report(thresholds=[0.20, 0.50])
"""

from oncothresh._evaluator import ThresholdEvaluator, compare_models
from oncothresh._results import (
    BootstrapResult,
    BoundaryCalibrationResult,
    CompareModelsResult,
    ConfidenceInterval,
    DecisionCurveResult,
    MultiThresholdReport,
    NNTResult,
    ThresholdResult,
    ThresholdSensitivityResult,
)

__version__ = "0.1.0"
__all__ = [
    "ThresholdEvaluator",
    "compare_models",
    "ThresholdResult",
    "BootstrapResult",
    "ConfidenceInterval",
    "MultiThresholdReport",
    "DecisionCurveResult",
    "NNTResult",
    "ThresholdSensitivityResult",
    "BoundaryCalibrationResult",
    "CompareModelsResult",
]
