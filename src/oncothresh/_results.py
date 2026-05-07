from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ThresholdResult(BaseModel):
    """Evaluation metrics at a single clinical decision threshold."""

    model_config = ConfigDict(frozen=True)

    threshold: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1: float
    mcc: float
    accuracy: float
    n_positive: int
    n_negative: int
    n_total: int

    def __str__(self) -> str:
        return (
            f"ThresholdResult(threshold={self.threshold:.2f}, "
            f"sensitivity={self.sensitivity:.3f}, specificity={self.specificity:.3f}, "
            f"ppv={self.ppv:.3f}, npv={self.npv:.3f}, f1={self.f1:.3f}, "
            f"mcc={self.mcc:.3f}, accuracy={self.accuracy:.3f}, "
            f"n={self.n_total} [{self.n_positive}+/{self.n_negative}-])"
        )


class ConfidenceInterval(BaseModel):
    model_config = ConfigDict(frozen=True)

    estimate: float
    lower: float
    upper: float
    confidence: float

    def __str__(self) -> str:
        pct = int(self.confidence * 100)
        return f"{self.estimate:.3f} ({pct}% CI: {self.lower:.3f}–{self.upper:.3f})"


class BootstrapResult(BaseModel):
    """Bootstrap confidence intervals for all metrics at a single threshold."""

    model_config = ConfigDict(frozen=True)

    threshold: float
    n_bootstrap: int
    confidence: float
    sensitivity: ConfidenceInterval
    specificity: ConfidenceInterval
    ppv: ConfidenceInterval
    npv: ConfidenceInterval
    f1: ConfidenceInterval
    mcc: ConfidenceInterval
    accuracy: ConfidenceInterval


class MultiThresholdReport(BaseModel):
    """Side-by-side results at multiple clinical thresholds."""

    results: list[ThresholdResult] = Field(default_factory=list)

    @property
    def thresholds(self) -> list[float]:
        return [r.threshold for r in self.results]
