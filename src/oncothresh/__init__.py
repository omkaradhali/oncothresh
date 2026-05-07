"""
oncothresh — Clinical threshold evaluation for oncology AI models.

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

from oncothresh._evaluator import ThresholdEvaluator
from oncothresh._results import (
    BootstrapResult,
    ConfidenceInterval,
    MultiThresholdReport,
    ThresholdResult,
)

__version__ = "0.1.0"
__all__ = [
    "ThresholdEvaluator",
    "ThresholdResult",
    "BootstrapResult",
    "ConfidenceInterval",
    "MultiThresholdReport",
]
