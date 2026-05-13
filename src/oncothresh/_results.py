from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ThresholdResult(BaseModel):
    """Evaluation metrics at a single clinical decision threshold."""

    model_config = ConfigDict(frozen=True)

    threshold: float = Field(
        description=(
            "The cutoff value used to split predictions into positive/negative "
            "(e.g. 0.20 means scores ≥ 20% TC are treated as positive)."
        )
    )
    sensitivity: float = Field(
        description=(
            "Of all true positives, the fraction the model correctly caught. "
            "Also called recall or true positive rate. High sensitivity = few missed positives."
        )
    )
    specificity: float = Field(
        description=(
            "Of all true negatives, the fraction the model correctly ruled out. "
            "Also called true negative rate. High specificity = few false alarms."
        )
    )
    ppv: float = Field(
        description=(
            "Positive Predictive Value — of all samples the model called positive, "
            "the fraction that actually were. Also called precision."
        )
    )
    npv: float = Field(
        description=(
            "Negative Predictive Value — of all samples the model called negative, "
            "the fraction that actually were."
        )
    )
    f1: float = Field(
        description=(
            "Harmonic mean of PPV and sensitivity. A single number balancing precision "
            "and recall. Useful when positive and negative classes are imbalanced."
        )
    )
    mcc: float = Field(
        description=(
            "Matthews Correlation Coefficient. Uses all four cells of the confusion matrix. "
            "Ranges from -1 (perfectly wrong) through 0 (random) to +1 (perfect). "
            "Most robust single metric for imbalanced classes."
        )
    )
    accuracy: float = Field(
        description=(
            "Fraction of all samples the model classified correctly. "
            "Can be misleading on imbalanced datasets, use MCC or F1 alongside it."
        )
    )
    n_positive: int = Field(
        description="Number of ground-truth positive samples (at or above the threshold in y_true)."
    )
    n_negative: int = Field(
        description="Number of ground-truth negative samples (below the threshold in y_true)."
    )
    n_total: int = Field(description="Total number of samples evaluated (n_positive + n_negative).")

    def __str__(self) -> str:
        return (
            f"ThresholdResult(threshold={self.threshold:.2f}, "
            f"sensitivity={self.sensitivity:.3f}, specificity={self.specificity:.3f}, "
            f"ppv={self.ppv:.3f}, npv={self.npv:.3f}, f1={self.f1:.3f}, "
            f"mcc={self.mcc:.3f}, accuracy={self.accuracy:.3f}, "
            f"n={self.n_total} [{self.n_positive}+/{self.n_negative}-])"
        )


class ConfidenceInterval(BaseModel):
    """A single metric's point estimate plus its bootstrapped confidence interval bounds."""

    model_config = ConfigDict(frozen=True)

    estimate: float = Field(
        description="The metric value computed on the full dataset (not a bootstrap average)."
    )
    lower: float = Field(description="Lower bound of the confidence interval.")
    upper: float = Field(description="Upper bound of the confidence interval.")
    confidence: float = Field(
        description=(
            "Confidence level used (e.g. 0.95 means 95% of bootstrap intervals "
            "contain the true value)."
        )
    )

    def __str__(self) -> str:
        pct = int(self.confidence * 100)
        return f"{self.estimate:.3f} ({pct}% CI: {self.lower:.3f}-{self.upper:.3f})"


class BootstrapResult(BaseModel):
    """Bootstrap confidence intervals for all metrics at a single threshold."""

    model_config = ConfigDict(frozen=True)

    threshold: float = Field(description="The clinical cutoff value used for this bootstrap run.")
    n_bootstrap: int = Field(
        description="Number of bootstrap resamples. 1000 is standard; use 2000 for publication."
    )
    confidence: float = Field(
        description="Confidence level used for all intervals (e.g. 0.95 for 95% CI)."
    )
    sensitivity: ConfidenceInterval = Field(description="Bootstrap CI for sensitivity.")
    specificity: ConfidenceInterval = Field(description="Bootstrap CI for specificity.")
    ppv: ConfidenceInterval = Field(
        description="Bootstrap CI for positive predictive value (precision)."
    )
    npv: ConfidenceInterval = Field(description="Bootstrap CI for negative predictive value.")
    f1: ConfidenceInterval = Field(description="Bootstrap CI for F1 score.")
    mcc: ConfidenceInterval = Field(
        description="Bootstrap CI for Matthews Correlation Coefficient."
    )
    accuracy: ConfidenceInterval = Field(description="Bootstrap CI for accuracy.")

    def __str__(self) -> str:
        pct = int(self.confidence * 100)
        header = f"BootstrapResult(threshold={self.threshold:.2f}, n={self.n_bootstrap}, {pct}% CI)"
        rows = (
            f"  sensitivity : {self.sensitivity}",
            f"  specificity : {self.specificity}",
            f"  ppv         : {self.ppv}",
            f"  npv         : {self.npv}",
            f"  f1          : {self.f1}",
            f"  mcc         : {self.mcc}",
            f"  accuracy    : {self.accuracy}",
        )
        return "\n".join([header, *rows])


class MultiThresholdReport(BaseModel):
    """Side-by-side results at multiple clinical thresholds."""

    model_config = ConfigDict(frozen=True)

    results: list[ThresholdResult] = Field(
        default_factory=list,
        description="Ordered list of ThresholdResult objects, one per threshold evaluated.",
    )

    @property
    def thresholds(self) -> list[float]:
        return [r.threshold for r in self.results]


class DecisionCurveResult(BaseModel):
    """
    Net benefit curves from Decision Curve Analysis (DCA) across a sweep of threshold
    probabilities.

    DCA compares three strategies at each threshold probability ``pt``:

    - **Model**: use the model's predicted scores to decide who to treat.
    - **Treat all**: treat every patient regardless of model output.
    - **Treat none**: treat nobody (net benefit = 0 by definition).

    The model adds clinical value wherever ``net_benefit_model`` exceeds both
    ``net_benefit_all`` and zero.

    ``net_benefit_none`` is not stored because it is always 0 — access it via the
    ``net_benefit_none`` property, which returns a list of zeros the same length as
    ``thresholds``.
    """

    model_config = ConfigDict(frozen=True)

    thresholds: list[float] = Field(
        description=(
            "The threshold probability (pt) values that were swept. Each pt is the "
            "probability of disease at which a clinician would decide to intervene."
        )
    )
    net_benefit_model: list[float] = Field(
        description=(
            "Net benefit of the model at each threshold probability. "
            "Computed as TP/N - FP/N x pt/(1-pt). May be negative, indicating net harm."
        )
    )
    net_benefit_all: list[float] = Field(
        description=(
            "Net benefit of treating every patient at each threshold probability. "
            "Computed as prevalence - (1-prevalence) x pt/(1-pt). "
            "This is the baseline the model must beat to be clinically useful."
        )
    )

    @property
    def net_benefit_none(self) -> list[float]:
        """Net benefit of treating nobody — always 0.0 at every threshold."""
        return [0.0] * len(self.thresholds)

    def __str__(self) -> str:
        n = len(self.thresholds)
        lo, hi = self.thresholds[0], self.thresholds[-1]
        return (
            f"DecisionCurveResult(thresholds=[{lo:.2f}–{hi:.2f}], "
            f"n_points={n}, net_benefit_none=0.0)"
        )


class NNTResult(BaseModel):
    """
    Number Needed to Test (NNT) at a clinical decision threshold.

    NNT adapts the clinical epidemiology concept of "Number Needed to Treat" to the
    diagnostic AI context. Where NNT-to-treat asks "how many patients must receive
    treatment to benefit one?", NNT-to-test asks two questions about the model's
    decision boundary:

    1. **nnt_positive** ("how efficient is a positive call?")
       Of all patients the model flags as positive (predicted score >= threshold),
       how many does it take to find one true positive?
       Formula: 1 / PPV
       Example: nnt_positive=1.14 means 114 flags yield ~100 true positives — very efficient.
       Example: nnt_positive=5.0 means 5 flags for 1 true positive — 80% are false alarms.

    2. **nnt_negative** ("how dangerous is a negative call?")
       Of all patients the model clears as negative (predicted score < threshold),
       how many cleared patients conceal one missed true positive?
       Formula: 1 / (1 - NPV)
       Example: nnt_negative=50 means 1 in 50 cleared patients is actually a missed case.
       Example: nnt_negative=inf means the model never misses a true positive (perfect NPV=1).

    Both values are stored as raw floats. In clinical writing they are typically rounded
    to the nearest integer or one decimal place (e.g. "NNT=5" or "NNT=4.8"). The result
    stores the exact float so callers can format as needed.

    Infinite values (float("inf")) arise naturally when a metric is perfect:
    - nnt_positive is inf when PPV=0 (every flag is wrong; you will never find a true positive)
    - nnt_negative is inf when NPV=1 (every clearance is correct; no missed cases exist)
    """

    model_config = ConfigDict(frozen=True)

    threshold: float = Field(
        description="The clinical cutoff used to binarize predictions for this NNT calculation."
    )
    nnt_positive: float = Field(
        description=(
            "Number of model-positive calls needed to find one true positive. "
            "Equal to 1/PPV. Lower is better. "
            "float('inf') when PPV=0 (model never correctly flags a true positive)."
        )
    )
    nnt_negative: float = Field(
        description=(
            "Number of model-negative calls needed to encounter one missed true positive. "
            "Equal to 1/(1-NPV). Higher is safer. "
            "float('inf') when NPV=1 (model never misses a true positive)."
        )
    )
    ppv: float = Field(
        description="Positive predictive value used to derive nnt_positive (from evaluate())."
    )
    npv: float = Field(
        description="Negative predictive value used to derive nnt_negative (from evaluate())."
    )
    n_positive: int = Field(
        description="Ground-truth positive sample count at this threshold."
    )
    n_negative: int = Field(
        description="Ground-truth negative sample count at this threshold."
    )
    n_total: int = Field(description="Total sample count (n_positive + n_negative).")

    def __str__(self) -> str:
        # float("inf") is not printable as a clean number, so we substitute "∞".
        def _fmt(v: float) -> str:
            return "∞" if v == float("inf") else f"{v:.2f}"

        return (
            f"NNTResult(threshold={self.threshold:.2f}, "
            f"nnt_positive={_fmt(self.nnt_positive)}, "
            f"nnt_negative={_fmt(self.nnt_negative)}, "
            f"ppv={self.ppv:.3f}, npv={self.npv:.3f}, "
            f"n={self.n_total} [{self.n_positive}+/{self.n_negative}-])"
        )


class ThresholdSensitivityResult(BaseModel):
    """
    Sensitivity and specificity curves across a sweep of thresholds around a clinical cutoff.

    A note on terminology: "threshold sensitivity analysis" is a statistics/engineering term
    meaning "how sensitive are the results to a change in this parameter?" It does NOT refer
    to the clinical metric called sensitivity (true positive rate). Both concepts appear in
    this class — ``sensitivities`` is the clinical metric, ``ThresholdSensitivityResult``
    describes the analysis. Context always disambiguates, but it is worth knowing upfront.

    This result stores three parallel arrays of equal length, indexed together:

    - ``thresholds[i]`` — a threshold value within [nominal_threshold ± delta]
    - ``sensitivities[i]`` — the model's clinical sensitivity (TPR) at that threshold
    - ``specificities[i]`` — the model's clinical specificity (TNR) at that threshold
    - ``shifts[i]`` — signed distance from the nominal threshold (negative = lower cutoff)

    The arrays are ordered from lowest threshold to highest. The nominal threshold appears
    as one of the entries; its index is ``nominal_index``.

    Interpreting the curves:
        - Flat curves: the model is robust — performance is stable across threshold variation.
          Threshold choice doesn't matter much; the model generalises to lab-specific cutoffs.
        - Steep curves: the model is fragile — small threshold shifts cause large performance
          drops. The published cutoff was likely tuned to the training set and may not transfer.
        - In TC scoring: if sensitivity drops more than ~5% when the threshold shifts by ±2%,
          consider reporting results at a range of thresholds rather than a single point.
    """

    model_config = ConfigDict(frozen=True)

    nominal_threshold: float = Field(
        description=(
            "The reference clinical cutoff around which the sweep is centred "
            "(e.g. 0.20 for the NGS eligibility threshold)."
        )
    )
    delta: float = Field(
        description=(
            "Half-width of the sweep range. The analysis covers "
            "[nominal_threshold - delta, nominal_threshold + delta], "
            "clamped to [0, 1]."
        )
    )
    thresholds: list[float] = Field(
        description="Threshold values swept, ordered from lowest to highest."
    )
    shifts: list[float] = Field(
        description=(
            "Signed distance of each threshold from the nominal cutoff "
            "(thresholds[i] - nominal_threshold). Negative = lower cutoff, positive = higher."
        )
    )
    sensitivities: list[float] = Field(
        description=(
            "Clinical sensitivity (true positive rate) at each swept threshold. "
            "Paired index-for-index with ``thresholds``."
        )
    )
    specificities: list[float] = Field(
        description=(
            "Clinical specificity (true negative rate) at each swept threshold. "
            "Paired index-for-index with ``thresholds``."
        )
    )
    nominal_index: int = Field(
        description=(
            "Index into ``thresholds`` (and the paired metric arrays) where "
            "the nominal threshold sits. Use this to identify the reference "
            "point when plotting."
        )
    )

    def __str__(self) -> str:
        lo = self.thresholds[0]
        hi = self.thresholds[-1]
        n = len(self.thresholds)
        # Show sensitivity and specificity at the endpoints and at the nominal threshold
        # so a quick print gives a summary of how much the metrics move across the sweep.
        nom_sens = self.sensitivities[self.nominal_index]
        nom_spec = self.specificities[self.nominal_index]
        return (
            f"ThresholdSensitivityResult("
            f"nominal={self.nominal_threshold:.2f}, "
            f"range=[{lo:.2f}–{hi:.2f}], n_points={n}, "
            f"sensitivity@nominal={nom_sens:.3f}, "
            f"specificity@nominal={nom_spec:.3f})"
        )


class BoundaryCalibrationResult(BaseModel):
    """
    Boundary-weighted calibration analysis near a clinical decision threshold.

    Standard global calibration (e.g. Expected Calibration Error over all predictions)
    can mask the most clinically dangerous miscalibration. A model may be well-calibrated
    on average yet systematically off near the exact cutoff where treatment decisions are
    made. This result focuses exclusively on that boundary zone.

    What "boundary zone" means:
        Only samples whose *predicted* score falls within [threshold ± window] are included.
        These are the close-call predictions — the ones the model is least certain about and
        that are most likely to flip the binary decision if the score is even slightly wrong.
        Predictions far from the threshold do not affect the decision (0.05 is negative at
        both 20% and 15% thresholds), so they are excluded.

    What calibration means for a regression model:
        Unlike classifier calibration (where you compare P̂(positive) to the observed
        positive rate), regression calibration checks whether the model's *continuous score*
        tracks the true continuous score on average. Within the boundary zone:
            - Bin the predicted scores into ``n_bins`` equal-width bins.
            - For each bin, compute mean(y_pred) and mean(y_true).
            - Perfect calibration: mean_pred ≈ mean_true in every bin (points on the diagonal
              of a reliability diagram).
            - Systematic offset: mean_pred > mean_true in the upper half of the zone means
              the model over-scores near the threshold — potentially over-referring patients.

    The ECE formula (applied locally to the boundary zone):
        ECE = Σ_bins [ (n_bin / N_boundary) x |mean_pred_bin x mean_true_bin| ]

        This is a weighted average of absolute calibration error per bin, where empty bins
        are skipped. Result is in the same units as the scores (e.g. 0.03 means 3 percentage
        points of average error in the boundary zone).

    Interpreting ``ece``:
        - ECE < 0.02 (2 pp): well-calibrated at the boundary — acceptable for clinical use.
        - ECE 0.02-0.05 (2-5 pp): moderate boundary miscalibration — document and monitor.
        - ECE > 0.05 (5 pp): the model's predictions near the threshold are systematically
          biased. A score of 0.21 may reflect a true TC of 0.26, for example. This should
          raise concern about deploying the threshold without recalibration.

    Empty boundary zone:
        If no predictions fall within [threshold ± window], ``n_samples`` will be 0 and
        ``ece`` will be ``float("nan")``. This usually means the model almost never predicts
        scores near this threshold — worth investigating separately (the model may be
        over-confident and never produce uncertain predictions).
    """

    model_config = ConfigDict(frozen=True)

    threshold: float = Field(
        description="The clinical cutoff around which the boundary zone is centred."
    )
    window: float = Field(
        description=(
            "Half-width of the boundary zone. The zone covers "
            "[threshold - window, threshold + window], clamped to [0, 1]."
        )
    )
    n_samples: int = Field(
        description=(
            "Number of samples whose predicted score fell within the boundary zone. "
            "If 0, ece is float('nan') and all bin arrays are empty."
        )
    )
    ece: float = Field(
        description=(
            "Boundary Expected Calibration Error: weighted average of |mean_pred - mean_true| "
            "across bins within the boundary zone. float('nan') when n_samples=0. "
            "In the same units as the scores (0.03 = 3 percentage points of error)."
        )
    )
    bin_edges: list[float] = Field(
        description=(
            "n_bins + 1 values defining the bin boundaries within the boundary zone. "
            "Bins are equal-width. Use these as x-axis tick marks on a reliability diagram."
        )
    )
    bin_centers: list[float] = Field(
        description="Midpoint of each bin. Convenience field: (bin_edges[i] + bin_edges[i+1]) / 2."
    )
    bin_mean_predicted: list[float] = Field(
        description=(
            "Mean predicted score within each bin. float('nan') for empty bins. "
            "On a reliability diagram, this is the x-coordinate of each point."
        )
    )
    bin_mean_actual: list[float] = Field(
        description=(
            "Mean true score within each bin. float('nan') for empty bins. "
            "On a reliability diagram, this is the y-coordinate of each point. "
            "Perfect calibration: bin_mean_actual[i] ≈ bin_mean_predicted[i]."
        )
    )
    bin_counts: list[int] = Field(
        description="Number of samples in each bin. Used to weight the ECE computation."
    )

    def __str__(self) -> str:
        lo = self.bin_edges[0] if self.bin_edges else float("nan")
        hi = self.bin_edges[-1] if self.bin_edges else float("nan")
        ece_str = "nan (no boundary samples)" if self.n_samples == 0 else f"{self.ece:.4f}"
        return (
            f"BoundaryCalibrationResult("
            f"threshold={self.threshold:.2f}, "
            f"window={self.window:.2f}, "
            f"zone=[{lo:.2f}–{hi:.2f}], "
            f"n_samples={self.n_samples}, "
            f"ece={ece_str})"
        )


class CompareModelsResult(BaseModel):
    """
    Side-by-side comparison of two or more models at the same clinical decision threshold.

    Each model is represented by one ThresholdResult — the metrics produced by running
    evaluate() at the shared threshold. The comparison shows all standard classification
    metrics (sensitivity, specificity, PPV, NPV, F1, MCC, accuracy) for each model so
    clinicians and researchers can judge which model performs better at a given cutoff.

    Typical use case: comparing UNI vs CONCH feature extractors at the 20% TC threshold,
    or comparing a new model version against a published baseline.

    Attributes
    ----------
    threshold : float
        The clinical cutoff at which all models were evaluated.
    model_names : list[str]
        Display names for each model, in the same order as ``results``.
    results : list[ThresholdResult]
        One ThresholdResult per model, ordered to match ``model_names``.
    """

    model_config = ConfigDict(frozen=True)

    threshold: float = Field(
        description="The clinical cutoff at which all models were evaluated."
    )
    model_names: list[str] = Field(
        description="Display name for each model, ordered to match results."
    )
    results: list[ThresholdResult] = Field(
        description="One ThresholdResult per model, ordered to match model_names."
    )

    _METRICS: tuple[str, ...] = (
        "sensitivity", "specificity", "ppv", "npv", "f1", "mcc", "accuracy"
    )

    def __str__(self) -> str:
        col = 12
        label_col = 14
        header = f"{'Metric':<{label_col}}" + "".join(f"{name:>{col}}" for name in self.model_names)
        sep = "-" * (label_col + col * len(self.model_names))
        rows = [
            f"CompareModelsResult(threshold={self.threshold:.2f}, n_models={len(self.results)})",
            header,
            sep,
        ]
        for metric in self._METRICS:
            row = f"{metric:<{label_col}}" + "".join(
                f"{getattr(r, metric):>{col}.3f}" for r in self.results
            )
            rows.append(row)
        return "\n".join(rows)
