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

from importlib.metadata import PackageNotFoundError, version

from oncothresh._evaluator import ThresholdEvaluator, compare_models
from oncothresh._results import (
    BiasAnalysisResult,
    BootstrapResult,
    BoundaryCalibrationResult,
    CompareModelsResult,
    ConfidenceInterval,
    DecisionCurveResult,
    MultiThresholdReport,
    NNTResult,
    SubgroupResult,
    ThresholdResult,
    ThresholdSensitivityResult,
)

# Single source of truth: read the version from installed distribution metadata (which
# comes from pyproject.toml) so __version__ can never drift from the packaged version.
try:
    __version__ = version("oncothresh")
except PackageNotFoundError:  # imported from a source tree without an installed distribution
    __version__ = "0.0.0+unknown"
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
    "SubgroupResult",
    "BiasAnalysisResult",
]
