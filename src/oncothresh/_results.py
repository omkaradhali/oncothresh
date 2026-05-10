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
