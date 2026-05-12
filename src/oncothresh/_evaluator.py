from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from oncothresh._results import (
    BootstrapResult,
    BoundaryCalibrationResult,
    ConfidenceInterval,
    DecisionCurveResult,
    MultiThresholdReport,
    NNTResult,
    ThresholdResult,
    ThresholdSensitivityResult,
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


    def nnt(self, threshold: float) -> NNTResult:
        """
        Compute the Number Needed to Test (NNT) at a clinical decision threshold.

        NNT answers the question clinicians ask in practice: "Given the model's behaviour
        at this threshold, how many patients do I need to act on to find one true case —
        and how many cleared patients might be hiding a missed case?"

        Two values are returned:

        **nnt_positive** — efficiency of a positive call
            Derived from PPV (positive predictive value).
            Formula: 1 / PPV
            Interpretation: on average, ``nnt_positive`` model-positive patients need to
            be tested before one true positive is found. A value of 2.0 means every
            second flagged patient is a true case; a value of 10.0 means 9 out of 10
            flagged patients are unnecessary referrals.

        **nnt_negative** — safety of a negative call
            Derived from NPV (negative predictive value).
            Formula: 1 / (1 - NPV)
            Interpretation: on average, ``nnt_negative`` model-negative patients are
            cleared before one missed true positive is encountered. A value of 100 means
            1 in 100 cleared patients is actually a missed case — low risk. A value of 5
            means 1 in 5 clearances conceals a true positive — high risk.

        Infinite values are valid and carry clinical meaning:
            - ``nnt_positive = inf`` when PPV = 0: the model never flags a true positive,
              so no amount of testing will find one via this model.
            - ``nnt_negative = inf`` when NPV = 1: the model never misses a true positive,
              so there are no missed cases among cleared patients.

        This method calls ``evaluate()`` internally — no extra computation is performed
        beyond what ``evaluate()`` already does.

        Parameters
        ----------
        threshold : float
            Clinical cutoff value (e.g. 0.20 for 20% TC NGS eligibility threshold).

        Returns
        -------
        NNTResult
            Contains nnt_positive, nnt_negative, and the underlying PPV/NPV values that
            produced them, along with sample counts for context.

        Examples
        --------
        >>> ev = ThresholdEvaluator(y_true=[0.1, 0.3, 0.6, 0.8], y_pred=[0.1, 0.25, 0.55, 0.85])
        >>> result = ev.nnt(threshold=0.20)
        >>> print(result.nnt_positive)  # how many flags per true positive
        >>> print(result.nnt_negative)  # how many clearances per missed case
        """
        result = self.evaluate(threshold)

        # PPV = 0 means every flag is a false alarm: 1/0 is mathematically undefined,
        # but the clinical interpretation is clear — you will never find a true positive
        # by acting on this model's positive calls. inf is the correct answer.
        nnt_positive = (1.0 / result.ppv) if result.ppv > 0 else float("inf")

        # 1 − NPV is the rate of missed cases among cleared patients.
        # NPV = 1 means no cleared patient is a missed case: 1/(1−1) = 1/0 → inf,
        # meaning you could clear infinitely many patients without missing a single true positive.
        false_omission_rate = 1.0 - result.npv
        nnt_negative = (1.0 / false_omission_rate) if false_omission_rate > 0 else float("inf")

        return NNTResult(
            threshold=threshold,
            nnt_positive=nnt_positive,
            nnt_negative=nnt_negative,
            ppv=result.ppv,
            npv=result.npv,
            n_positive=result.n_positive,
            n_negative=result.n_negative,
            n_total=result.n_total,
        )

    def threshold_sensitivity(
        self,
        threshold: float,
        delta: float = 0.05,
        step: float = 0.01,
    ) -> ThresholdSensitivityResult:
        """
        Analyse how sensitivity and specificity change as the clinical threshold shifts.

        A brief note on naming: "threshold sensitivity analysis" is a term from statistics
        and engineering meaning "how sensitive is this result to a change in the threshold
        parameter?" It does not refer to clinical sensitivity (the true positive rate).
        Both meanings appear here — the method name describes the analysis technique,
        while the ``sensitivities`` field in the result holds the clinical metric.

        What this method does:
            Sweeps the decision threshold across [threshold - delta, threshold + delta],
            computes clinical sensitivity and specificity at each point, and returns the
            full curves. The range is clamped to [0, 1] because scores outside that range
            are not clinically meaningful.

        Why this matters clinically:
            A published threshold (e.g. 20% TC for NGS eligibility) is almost never used
            verbatim across all labs. Staining protocols differ, scanners differ, and
            pathologist conventions drift. A threshold sensitivity analysis answers: "If
            our lab uses 18% instead of 20%, how much does the model's sensitivity change?
            Are we putting patients at risk?"

            If the curves are steep near the nominal threshold, the model is fragile — it
            was tuned specifically to that cutoff and will not transfer safely. If the
            curves are flat, the model is robust across reasonable clinical variation.

            Typical interpretation for TC scoring:
                - ≤3% sensitivity drop over ±5% shift: robust, safe to deploy
                - 5-10% drop: moderate fragility, document the exact cutoff used
                - >10% drop: the model is threshold-brittle — report results across
                  a range of thresholds rather than a single point

        Parameters
        ----------
        threshold : float
            The nominal clinical cutoff to analyse (e.g. 0.20 for 20% TC).
        delta : float
            Half-width of the sweep. The analysis covers
            [threshold - delta, threshold + delta], clamped to [0, 1].
            Default 0.05 covers ±5%, which spans the typical lab-to-lab variation
            for TC thresholds.
        step : float
            Resolution of the sweep — distance between consecutive threshold values.
            Default 0.01 (1%) gives 11 evaluation points over the default ±5% range,
            which is granular enough for plotting and clinical reporting.
            Use 0.005 (0.5%) for publication-quality figures.

        Returns
        -------
        ThresholdSensitivityResult
            Three parallel arrays (``thresholds``, ``sensitivities``, ``specificities``)
            plus ``shifts`` (signed distance from the nominal cutoff) and
            ``nominal_index`` (which array position corresponds to the reference threshold).

        Raises
        ------
        ValueError
            If delta <= 0 or step <= 0.
        """
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}")

        # Build the sweep grid using linspace rather than arange to avoid floating-point
        # drift. arange with a float step accumulates rounding error over many steps
        # (e.g. 0.01 + 0.01 + ... != exact multiples). linspace guarantees the exact
        # number of points and places them uniformly regardless of float representation.
        n_steps = int(round(2 * delta / step)) + 1
        lo = max(0.0, threshold - delta)
        hi = min(1.0, threshold + delta)
        pts = np.linspace(lo, hi, n_steps)

        sensitivities: list[float] = []
        specificities: list[float] = []

        for pt in pts:
            r = self.evaluate(pt)
            sensitivities.append(r.sensitivity)
            specificities.append(r.specificity)

        # shifts[i] = how far pts[i] is from the nominal threshold.
        # Round to 10 decimal places to suppress floating-point noise in the display
        # (e.g. 0.19999999999998 instead of 0.20 after linspace arithmetic).
        shifts = [round(float(pt) - threshold, 10) for pt in pts]

        # Identify which index in the sweep corresponds to the nominal threshold.
        # We find the closest point rather than assuming an exact match, because clamping
        # and linspace rounding may shift the nominal value slightly from its ideal position.
        nominal_index = int(np.argmin(np.abs(pts - threshold)))

        return ThresholdSensitivityResult(
            nominal_threshold=threshold,
            delta=delta,
            thresholds=[round(float(pt), 10) for pt in pts],
            shifts=shifts,
            sensitivities=sensitivities,
            specificities=specificities,
            nominal_index=nominal_index,
        )

    def boundary_calibration(
        self,
        threshold: float,
        window: float = 0.10,
        n_bins: int = 10,
    ) -> BoundaryCalibrationResult:
        """
        Compute boundary-weighted calibration error near a clinical decision threshold.

        Global calibration metrics (e.g. overall ECE across all predictions) can mask the
        most clinically dangerous miscalibration. A model that is well-calibrated on average
        may still be systematically biased near the exact cutoff that determines whether a
        patient receives NGS testing, chemotherapy, or an immunotherapy agent.

        This method focuses exclusively on the *boundary zone* — predictions that fall
        within ``window`` of ``threshold``. Those are the close-call samples where a
        calibration error most directly changes a treatment decision.

        How calibration is measured for a regression model:
            Regression calibration asks: "when the model predicts a score of X, is the
            true value actually around X?" This is different from binary classifier
            calibration (where you check whether P̂(positive) matches the observed
            positive rate). Here, both y_pred and y_true are continuous scores and we
            check their agreement within the boundary zone by binning.

        Algorithm:
            1. Select boundary samples: keep only samples where y_pred ∈ [threshold ± window].
            2. Divide the boundary zone into ``n_bins`` equal-width bins.
            3. For each bin, compute mean(y_pred) and mean(y_true).
            4. ECE = Σ (n_bin / N_boundary) x |mean_pred_bin - mean_true_bin|
               Empty bins are skipped (contribute 0 weight to ECE).

        Choosing ``window``:
            The window should span the clinical uncertainty range — the zone where a
            real prediction could plausibly be on either side of the threshold.
            - TC 20% threshold: window=0.10 captures predictions from 10% to 30%, which
              includes cases a pathologist might read as borderline.
            - Tighter window (0.05): focuses on the sharpest close-calls; requires more
              data to populate bins reliably.
            - Wider window (0.15): more samples, smoother bins, but includes predictions
              that are not truly on the boundary.

        Choosing ``n_bins``:
            Fewer bins (5) are more stable with small datasets — each bin has more samples,
            so mean values are reliable. More bins (10-20) give a finer-grained reliability
            diagram but need proportionally more boundary samples to avoid empty bins.
            Rule of thumb: aim for at least 5 samples per bin on average
            (n_samples / n_bins ≥ 5).

        Parameters
        ----------
        threshold : float
            The clinical decision cutoff (e.g. 0.20 for 20% TC).
        window : float
            Half-width of the boundary zone. Samples with y_pred in
            [threshold - window, threshold + window] are included.
            Clamped to [0, 1]. Default 0.10 (±10%).
        n_bins : int
            Number of equal-width bins within the boundary zone.
            Default 10. Use fewer bins (5) for small datasets.

        Returns
        -------
        BoundaryCalibrationResult
            Contains the scalar ``ece``, bin-level calibration data
            (``bin_mean_predicted``, ``bin_mean_actual``, ``bin_counts``), and metadata.
            When ``n_samples == 0``, ``ece`` is ``float("nan")`` and bin value arrays
            contain only ``float("nan")`` entries.

        Raises
        ------
        ValueError
            If window <= 0 or n_bins < 1.
        """
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        if n_bins < 1:
            raise ValueError(f"n_bins must be at least 1, got {n_bins}")

        # Boundary zone: clamp to [0, 1] since scores live in that range.
        lo = max(0.0, threshold - window)
        hi = min(1.0, threshold + window)

        # Select only predictions that fall in the boundary zone.
        # We use the predicted score (y_pred) as the filter criterion — not y_true —
        # because the model's decision is based on what it predicts, not the true label.
        # A sample with y_true=0.35 but y_pred=0.45 is not a boundary case for the
        # 20% threshold; the model confidently called it positive.
        boundary_mask = (self.y_pred >= lo) & (self.y_pred <= hi)
        n_boundary = int(boundary_mask.sum())

        # Bin grid: n_bins equal-width bins spanning the full boundary zone.
        # Using linspace avoids the floating-point accumulation that plagues np.arange
        # with float steps (see threshold_sensitivity for the same rationale).
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = [(edges[i] + edges[i + 1]) / 2.0 for i in range(n_bins)]

        y_pred_b = self.y_pred[boundary_mask]
        y_true_b = self.y_true[boundary_mask]

        bin_mean_pred: list[float] = []
        bin_mean_true: list[float] = []
        bin_counts: list[int] = []

        for i in range(n_bins):
            # Half-open [lo, hi) for all bins except the last, which is [lo, hi] to ensure
            # samples exactly at the upper edge are not excluded.
            if i < n_bins - 1:
                in_bin = (y_pred_b >= edges[i]) & (y_pred_b < edges[i + 1])
            else:
                in_bin = (y_pred_b >= edges[i]) & (y_pred_b <= edges[i + 1])

            count = int(in_bin.sum())
            bin_counts.append(count)

            if count > 0:
                bin_mean_pred.append(float(y_pred_b[in_bin].mean()))
                bin_mean_true.append(float(y_true_b[in_bin].mean()))
            else:
                # Empty bins contribute nothing to ECE; store nan so callers can
                # distinguish "empty bin" from "perfectly calibrated bin" in plots.
                bin_mean_pred.append(float("nan"))
                bin_mean_true.append(float("nan"))

        # Compute ECE: weighted average of |mean_pred - mean_true| across non-empty bins.
        if n_boundary == 0:
            # No predictions fell near this threshold — ECE is undefined, not zero.
            ece = float("nan")
        else:
            ece = sum(
                (count / n_boundary) * abs(mp - mt)
                for count, mp, mt in zip(bin_counts, bin_mean_pred, bin_mean_true, strict=True)
                if count > 0  # empty bins have nan values; skip rather than propagate nan
            )

        return BoundaryCalibrationResult(
            threshold=threshold,
            window=window,
            n_samples=n_boundary,
            ece=ece,
            bin_edges=[round(float(e), 10) for e in edges],
            bin_centers=[round(float(c), 10) for c in centers],
            bin_mean_predicted=bin_mean_pred,
            bin_mean_actual=bin_mean_true,
            bin_counts=bin_counts,
        )

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
