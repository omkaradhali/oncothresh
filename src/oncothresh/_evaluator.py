from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from oncothresh._results import (
    BootstrapResult,
    ConfidenceInterval,
    MultiThresholdReport,
    ThresholdResult,
)


class ThresholdEvaluator:
    """
    Evaluate a continuous oncology AI model at predefined clinical decision thresholds.

    Parameters
    ----------
    y_true : array-like of float
        Ground-truth continuous scores (e.g. pathologist TC scores, 0.0–1.0).
    y_pred : array-like of float
        Model-predicted continuous scores, same scale as y_true.
    """

    def __init__(
        self,
        y_true: np.ndarray | list[float],
        y_pred: np.ndarray | list[float],
    ) -> None:
        self.y_true = np.asarray(y_true, dtype=float)
        self.y_pred = np.asarray(y_pred, dtype=float)

        if self.y_true.shape != self.y_pred.shape:
            raise ValueError(
                f"y_true and y_pred must have the same shape, "
                f"got {self.y_true.shape} and {self.y_pred.shape}"
            )
        if self.y_true.ndim != 1:
            raise ValueError("y_true and y_pred must be 1-D arrays")
        if len(self.y_true) < 2:
            raise ValueError("At least 2 samples are required")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, threshold: float) -> ThresholdResult:
        """
        Compute classification metrics at a single clinical decision threshold.

        The threshold binarizes both y_true and y_pred: a sample is "positive"
        (above threshold) if its score >= threshold.

        Parameters
        ----------
        threshold : float
            Clinical cutoff value (e.g. 0.20 for 20% TC NGS eligibility).

        Returns
        -------
        ThresholdResult
        """
        y_true_bin, y_pred_bin = self._binarize(threshold)
        return self._compute_metrics(threshold, y_true_bin, y_pred_bin)

    def bootstrap_ci(
        self,
        threshold: float,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        random_state: int | None = None,
    ) -> BootstrapResult:
        """
        Estimate 95% confidence intervals for all metrics via non-parametric bootstrapping.

        Parameters
        ----------
        threshold : float
            Clinical cutoff value.
        n_bootstrap : int
            Number of bootstrap resamples. 1000 is standard; use 2000 for publication.
        confidence : float
            Confidence level, default 0.95 (95% CI).
        random_state : int | None
            Seed for reproducibility.

        Returns
        -------
        BootstrapResult
        """
        rng = np.random.default_rng(random_state)
        n = len(self.y_true)

        metrics: dict[str, list[float]] = {
            k: [] for k in ("sensitivity", "specificity", "ppv", "npv", "f1", "mcc", "accuracy")
        }

        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            result = self._compute_metrics(
                threshold,
                (self.y_true[idx] >= threshold).astype(int),
                (self.y_pred[idx] >= threshold).astype(int),
            )
            for key in metrics:
                metrics[key].append(getattr(result, key))

        point = self.evaluate(threshold)
        alpha = (1.0 - confidence) / 2.0

        def _ci(key: str) -> ConfidenceInterval:
            samples = np.array(metrics[key])
            return ConfidenceInterval(
                estimate=getattr(point, key),
                lower=float(np.quantile(samples, alpha)),
                upper=float(np.quantile(samples, 1.0 - alpha)),
                confidence=confidence,
            )

        return BootstrapResult(
            threshold=threshold,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            sensitivity=_ci("sensitivity"),
            specificity=_ci("specificity"),
            ppv=_ci("ppv"),
            npv=_ci("npv"),
            f1=_ci("f1"),
            mcc=_ci("mcc"),
            accuracy=_ci("accuracy"),
        )

    def multi_threshold_report(self, thresholds: list[float]) -> MultiThresholdReport:
        """
        Evaluate at multiple clinical cutoffs and return a side-by-side report.

        Parameters
        ----------
        thresholds : list[float]
            Ordered list of clinical cutoffs (e.g. [0.20, 0.50] for TC).
        """
        if not thresholds:
            raise ValueError("thresholds must not be empty")
        results = [self.evaluate(t) for t in thresholds]
        return MultiThresholdReport(results=results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _binarize(self, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        return (
            (self.y_true >= threshold).astype(int),
            (self.y_pred >= threshold).astype(int),
        )

    def _compute_metrics(
        self,
        threshold: float,
        y_true_bin: np.ndarray,
        y_pred_bin: np.ndarray,
    ) -> ThresholdResult:
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()

        sensitivity = self._safe_divide(tp, tp + fn)
        specificity = self._safe_divide(tn, tn + fp)
        ppv = self._safe_divide(tp, tp + fp)
        npv = self._safe_divide(tn, tn + fn)
        f1 = self._safe_divide(2 * ppv * sensitivity, ppv + sensitivity)
        mcc = float(matthews_corrcoef(y_true_bin, y_pred_bin))
        accuracy = self._safe_divide(tp + tn, tp + tn + fp + fn)

        n_total = int(tp + tn + fp + fn)
        n_positive = int(tp + fn)
        n_negative = int(tn + fp)

        return ThresholdResult(
            threshold=threshold,
            sensitivity=sensitivity,
            specificity=specificity,
            ppv=ppv,
            npv=npv,
            f1=f1,
            mcc=mcc,
            accuracy=accuracy,
            n_positive=n_positive,
            n_negative=n_negative,
            n_total=n_total,
        )

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        """Return 0.0 when denominator is zero rather than raising."""
        return float(numerator / denominator) if denominator != 0 else 0.0
