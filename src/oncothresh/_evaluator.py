from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from oncothresh._results import (
    BootstrapResult,
    ConfidenceInterval,
    DecisionCurveResult,
    MultiThresholdReport,
    ThresholdResult,
)


class ThresholdEvaluator:
    """
    Evaluate a continuous oncology AI model at predefined clinical decision thresholds.

    Most oncology AI models output a continuous score, a number between 0 and 1 representing
    the model's confidence (e.g. predicted tumor cellularity, Ki-67 proliferation index, or
    PD-L1 expression level). Standard ML metrics like accuracy are computed on binary labels,
    not continuous scores. In clinical practice, a specific threshold converts that continuous
    score into a binary decision: positive (at or above threshold) or negative (below threshold).

    ThresholdEvaluator bridges that gap. You provide the ground-truth scores and model
    predictions once at construction, then evaluate at any threshold or multiple thresholds
    simultaneously to get the full suite of classification metrics relevant to clinical
    decision making: sensitivity, specificity, PPV, NPV, F1, MCC, and accuracy.

    Typical workflow::

        ev = ThresholdEvaluator(y_true=pathologist_scores, y_pred=model_scores)

        # Evaluate at a single clinical cutoff
        result = ev.evaluate(threshold=0.20)

        # Estimate how reliable that result is via bootstrapped confidence intervals
        ci = ev.bootstrap_ci(threshold=0.20, n_bootstrap=1000, random_state=42)

        # Compare performance side-by-side at multiple cutoffs
        report = ev.multi_threshold_report(thresholds=[0.20, 0.50])

    Parameters
    ----------
    y_true : array-like of float
        Ground-truth continuous scores (e.g. pathologist TC scores, 0.0-1.0).
        Must be a 1-D array or list with at least 2 samples.
    y_pred : array-like of float
        Model-predicted continuous scores on the same scale as y_true.
        Must be the same length as y_true.

    Raises
    ------
    ValueError
        If y_true and y_pred have different shapes, are not 1-D, or contain fewer than
        2 samples.
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

        # Classification metrics require binary labels; threshold converts continuous scores to 0/1.
        y_true_bin, y_pred_bin = self._binarize(threshold)

        evaluation_result = self._compute_metrics(threshold, y_true_bin, y_pred_bin)

        return evaluation_result

    def bootstrap_ci(
        self,
        threshold: float,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        random_state: int | None = None,
    ) -> BootstrapResult:
        """
        Estimate confidence intervals (CIs) for all metrics via non-parametric bootstrapping.

        A confidence interval answers "how much can I trust this metric?" A single call to
        evaluate() gives one number (e.g. sensitivity = 0.857), but that number came from one
        specific dataset. Bootstrap CI estimates the realistic range that metric could fall in
        if the dataset were slightly different.

        How it works:
          1. Draw n_bootstrap resamples from the dataset, each the same size as the original
             but sampled with replacement (some patients appear multiple times, some not at all).
          2. Compute all metrics on each resample, collecting n_bootstrap values per metric.
          3. Sort those values and cut the extreme tails: for a 95% CI, discard the bottom
             2.5% and top 2.5%, leaving the middle 95% as the interval [lower, upper].
          4. Report the point estimate from the full original dataset alongside the interval.

        Interpreting the result:
          - Tight CI (e.g. 0.857, 95% CI: 0.831-0.881) → stable, reliable estimate.
          - Wide CI  (e.g. 0.857, 95% CI: 0.42-0.99)   → dataset too small to trust the number.
          - Reviewers and clinicians use the CI width to judge whether results are real or a
            fluke of sample size. Always report CIs in publications.

        Parameters
        ----------
        threshold : float
            Clinical cutoff value.
        n_bootstrap : int
            Number of bootstrap resamples. 1000 is standard; use 2000 for publication.
        confidence : float
            Confidence level, default 0.95 (95% CI). Use 0.99 for a stricter 99% CI.
        random_state : int | None
            Seed for reproducibility. Set this to a fixed integer (e.g. 42) to get the
            same CI bounds across runs — required for reproducible published results.

        Returns
        -------
        BootstrapResult
            Point estimate and CI bounds for every metric at the given threshold.
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

        Runs evaluate() at each threshold and bundles results into a single
        MultiThresholdReport. Purely orchestration — no computation happens here.
        For TC scoring the standard cutoffs are [0.20, 0.50] (NGS eligibility
        and treatment response).

        Parameters
        ----------
        thresholds : list[float]
            Ordered list of clinical cutoffs. Results are returned in the same order.

        Returns
        -------
        MultiThresholdReport
            One ThresholdResult per threshold, accessible via .results and .thresholds.

        Raises
        ------
        ValueError
            If thresholds is an empty list.
        """
        if not thresholds:
            raise ValueError("thresholds must not be empty")
        results = [self.evaluate(t) for t in thresholds]
        return MultiThresholdReport(results=results)


    def decision_curve(
        self,
        thresholds: np.ndarray | list[float] | None = None,
    ) -> DecisionCurveResult:
        """
        Compute Decision Curve Analysis (DCA) across a range of clinical decision thresholds.

        Why DCA? Standard metrics (sensitivity, specificity, AUC) measure discrimination —
        how well the model separates positives from negatives. They cannot answer: "Is using
        this model to guide clinical decisions actually better than a simpler policy?" DCA
        answers that question by computing net benefit at every possible harm trade-off.

        The key concept is the threshold probability ``pt``: the probability of disease at
        which a clinician would decide to intervene. A clinician who says "I'll order NGS
        if there's a ≥20% chance the patient's TC is ≥20%" has pt=0.20. Their implicit
        harm ratio is pt/(1-pt) = 0.25: they consider 4 unnecessary procedures equivalent
        to 1 missed case.

        At each pt, three strategies are compared::

            Model:      NB = TP/N - FP/N x pt/(1-pt)
            Treat all:  NB = prevalence - (1-prevalence) x pt/(1-pt)
            Treat none: NB = 0  (no interventions → no harm, no benefit)

        The model adds clinical value wherever its net benefit exceeds both treat-all and
        zero. Negative net benefit means the strategy causes net harm — this is valid and
        should not be clipped.

        Interpreting the curves:

        - Find your ``pt`` on the x-axis (where your clinical harm tolerance sits).
        - If ``net_benefit_model > net_benefit_all`` at that pt → model beats "treat everyone".
        - If ``net_benefit_model > 0`` at that pt → model beats "treat nobody".
        - The width of the pt range where the model wins is the clinical utility window.

        Parameters
        ----------
        thresholds : array-like of float or None
            The pt values to sweep. Each value must be in [0, 1); pt=1.0 is excluded
            because pt/(1-pt) is undefined there (any model with FPs would have NB=−∞).
            Values outside [0, 1) produce NaN entries in the output.

            For TC analysis the range [0.05, 0.50] covers both clinical cutoffs (0.20
            and 0.50) with context on either side. Defaults to np.linspace(0.01, 0.99, 99)
            when not provided.

        Returns
        -------
        DecisionCurveResult
            ``thresholds``, ``net_benefit_model``, and ``net_benefit_all`` arrays of equal
            length. Access ``net_benefit_none`` (always 0) via the result's property.

        References
        ----------
        Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating
        prediction models. Med Decis Making. 2006;26(6):565-574.
        """
        if thresholds is None:
            # 99 evenly-spaced points from 1% to 99% — fine-grained enough for smooth plots,
            # excludes 0 (trivial: everyone is positive) and 1 (undefined: pt/(1−pt) → ∞).
            thresholds = np.linspace(0.01, 0.99, 99)

        pts = np.asarray(thresholds, dtype=float)
        n = len(self.y_true)

        nb_model: list[float] = []
        nb_all: list[float] = []

        for pt in pts:
            # pt/(1−pt) → ∞ at pt≥1.0: NB collapses to −∞ for any model with FPs.
            # Return NaN so callers can detect and skip these points when plotting.
            if pt >= 1.0:
                nb_model.append(float("nan"))
                nb_all.append(float("nan"))
                continue

            # Binarize at this threshold — mirrors evaluate(): >= is "positive".
            y_true_bin = self.y_true >= pt
            y_pred_bin = self.y_pred >= pt

            # TP: model says positive AND truly positive.
            # FP: model says positive BUT truly negative.
            tp = int(np.sum(y_pred_bin & y_true_bin))
            fp = int(np.sum(y_pred_bin & ~y_true_bin))

            # harm_weight converts FPs to the same "currency" as TPs.
            # pt=0.20 → harm_weight=0.25: 4 unnecessary procedures ≈ 1 missed case.
            # pt=0.50 → harm_weight=1.00: FP and FN are equally harmful.
            harm_weight = pt / (1.0 - pt)

            # Net benefit for the model: TPs rewarded, FPs penalised by harm_weight.
            nb_model.append(tp / n - fp / n * harm_weight)

            # Prevalence at this specific pt: fraction of y_true scored >= pt.
            # Recomputed per pt because more patients are "positive" at lower cutoffs.
            prevalence = float(y_true_bin.mean())

            # Net benefit for treat-all: derived from the same formula with TP=all positives,
            # FP=all negatives. Rearranges neatly to: prevalence − (1−prevalence) × harm_weight.
            nb_all.append(prevalence - (1.0 - prevalence) * harm_weight)

        return DecisionCurveResult(
            thresholds=pts.tolist(),
            net_benefit_model=nb_model,
            net_benefit_all=nb_all,
        )

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
        """
        Core calculation step. Expects pre-binarized 0/1 arrays from _binarize().
        Builds the confusion matrix (tn, fp, fn, tp) and derives all metrics from
        those four counts. _safe_divide handles zero denominators on degenerate inputs
        (e.g. all-positive bootstrap resamples) without crashing.
        """
        # Build 2x2 confusion matrix and unpack into tn/fp/fn/tp. labels=[0,1] forces
        # a full 2x2 even when a bootstrap resample accidentally contains only positives
        # or only negatives — without it sklearn returns a 1x1 and the unpacking crashes.
        # Ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
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
