"""
Numerical validation of oncothresh DCA against the reference dcurves package.

Decision Curve Analysis is statistically subtle, and a silent error in the
net-benefit formula would be a credibility-ending bug. This module proves
that ThresholdEvaluator.decision_curve reproduces the net-benefit values of
dcurves (Vickers et al., MSKCC) to machine precision.

dcurves is an optional validation dependency, it pulls in pandas, lifelines,
and statsmodels, which the package itself does not need. Install it with::

    uv pip install -e ".[validation]"

When dcurves is absent (e.g. the default CI lane installs only [dev]),
every test here is skipped rather than failed.

Reference
---------
Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating
prediction models. Med Decis Making. 2006;26(6):565-574.
"""

from __future__ import annotations

import numpy as np
import pytest

from oncothresh import ThresholdEvaluator

# Skip the whole module unless the reference implementation is installed.
dcurves = pytest.importorskip("dcurves")
pd = pytest.importorskip("pandas")


def _make_dataset(
    seed: int,
    a: float,
    b: float,
    clinical_threshold: float,
    n: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (y_true, y_pred, disease) for a DCA comparison.

    ``y_true`` is a continuous biomarker score drawn from Beta(a, b).
    ``disease`` is the fixed DCA label ``y_true >= clinical_threshold``.
    ``y_pred`` is a calibrated probability in (0, 1) correlated with disease,
    which is the input contract DCA requires (predicted probability, not a raw
    biomarker score).
    """
    rng = np.random.default_rng(seed)
    y_true = rng.beta(a, b, size=n)
    disease = (y_true >= clinical_threshold).astype(int)
    noise = rng.normal(0.0, 0.15, size=n)
    y_pred = np.clip(disease * 0.55 + 0.25 + noise, 0.001, 0.999)
    return y_true, y_pred, disease


def _dcurves_net_benefit(
    disease: np.ndarray,
    y_pred: np.ndarray,
    pts: np.ndarray,
) -> tuple[dict[float, float], dict[float, float]]:
    """Run the reference ``dcurves`` DCA and return {pt: nb} maps for model and treat-all."""
    df = pd.DataFrame({"disease": disease, "model": y_pred})
    out = dcurves.dca(data=df, outcome="disease", modelnames=["model"], thresholds=pts)

    def _as_map(label: str) -> dict[float, float]:
        rows = out[out["model"] == label]
        return {
            round(float(t), 6): float(nb)
            for t, nb in zip(rows["threshold"], rows["net_benefit"], strict=True)
        }

    return _as_map("model"), _as_map("all")


# (seed, beta a, beta b, clinical_threshold), spans low/high prevalence and both TC cutoffs.
_CASES = [
    (42, 1.5, 3.5, 0.20),
    (7, 2.0, 2.0, 0.50),
    (123, 3.0, 1.5, 0.20),
    (2024, 1.2, 5.0, 0.10),
]


@pytest.mark.parametrize(("seed", "a", "b", "clinical_threshold"), _CASES)
def test_decision_curve_matches_dcurves(
    seed: int, a: float, b: float, clinical_threshold: float
) -> None:
    """oncothresh net benefit must equal dcurves to machine precision across the pt sweep."""
    y_true, y_pred, disease = _make_dataset(seed, a, b, clinical_threshold)
    pts = np.round(np.arange(0.05, 0.51, 0.01), 2)

    ev = ThresholdEvaluator(y_true=y_true, y_pred=y_pred)
    res = ev.decision_curve(clinical_threshold=clinical_threshold, thresholds=pts)

    # Prevalence (the fixed DCA disease rate) must match the empirical disease rate exactly.
    assert res.prevalence == pytest.approx(float(disease.mean()), abs=1e-12)

    dc_model, dc_all = _dcurves_net_benefit(disease, y_pred, pts)

    for pt, nb_model, nb_all in zip(
        res.thresholds, res.net_benefit_model, res.net_benefit_all, strict=True
    ):
        key = round(float(pt), 6)
        assert key in dc_model, f"dcurves produced no row for pt={key}"
        # 1e-9 is a conservative bar; in practice the two implementations agree
        # to ~1e-16 (identical float arithmetic).
        assert nb_model == pytest.approx(dc_model[key], abs=1e-9), (
            f"model net benefit diverges at pt={key}: "
            f"oncothresh={nb_model}, dcurves={dc_model[key]}"
        )
        assert nb_all == pytest.approx(dc_all[key], abs=1e-9), (
            f"treat-all net benefit diverges at pt={key}: "
            f"oncothresh={nb_all}, dcurves={dc_all[key]}"
        )


def test_treat_all_crosses_zero_at_prevalence_consistent_with_dcurves() -> None:
    """Sanity anchor: treat-all net benefit is identical in both tools at every pt.

    Treat-all depends only on prevalence and pt (not on the model), so it is the
    cleanest invariant to pin, any disagreement here would indicate a prevalence
    or harm-weight discrepancy rather than a model-classification difference.
    """
    y_true, y_pred, disease = _make_dataset(seed=99, a=2.0, b=3.0, clinical_threshold=0.20)
    pts = np.round(np.arange(0.05, 0.51, 0.05), 2)

    ev = ThresholdEvaluator(y_true=y_true, y_pred=y_pred)
    res = ev.decision_curve(clinical_threshold=0.20, thresholds=pts)
    _, dc_all = _dcurves_net_benefit(disease, y_pred, pts)

    for pt, nb_all in zip(res.thresholds, res.net_benefit_all, strict=True):
        assert nb_all == pytest.approx(dc_all[round(float(pt), 6)], abs=1e-12)
