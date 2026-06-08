# oncothresh

**Clinical threshold evaluation for oncology AI models.**

[![CI](https://github.com/omkaradhali/oncothresh/actions/workflows/ci.yml/badge.svg)](https://github.com/omkaradhali/oncothresh/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Continuous oncology AI models (tumor cellularity (TC), Ki-67, TMB, PD-L1) are deployed at specific clinical thresholds that decide patient treatment. Global metrics such as ICC, MAE, and AUC measure overall agreement. They do not answer the question that matters at the bedside: *how reliable is this model at the exact cutoff that governs the decision?*

`oncothresh` is built around that question. It evaluates classification metrics, bootstrap confidence intervals, threshold sensitivity, boundary-weighted calibration, decision-curve net benefit, and Number-Needed-to-Test at one or more clinical decision thresholds.

---

## Why oncothresh

- **The gap.** Patho-Bench, PathBench, and PathBench-MIL benchmark pathology foundation models globally. None evaluates performance at predefined clinical thresholds with uncertainty quantification. The 2025 *Lancet Digital Health* commentary on prediction thresholds in oncology AI explicitly flagged this as an unmet methodological need.
- **The audience.** Two readers consume the same output: the ML engineer who wants metric values plus CIs, and the clinician who needs to know what those numbers mean for patients. Every result object is designed to be read at both levels. Printable summaries are concise, and the underlying fields carry the full numbers a paper or audit needs.

---

## Installation

```bash
pip install oncothresh
```

Requires Python 3.10+. Core dependencies: `numpy`, `scikit-learn`, `pydantic`. Plotting helpers (forthcoming in v0.1.1) live behind an optional extra: `pip install oncothresh[plotting]`.

---

## Quickstart

```python
from oncothresh import ThresholdEvaluator

ev = ThresholdEvaluator(y_true=tc_scores, y_pred=model_predictions)

# Single-threshold metrics
result = ev.evaluate(threshold=0.20)

# Bootstrap CIs
ci = ev.bootstrap_ci(threshold=0.20, n_bootstrap=1000, random_state=42)

# Side-by-side at multiple clinical cutoffs
report = ev.multi_threshold_report(thresholds=[0.20, 0.50])
```

---

## Metrics glossary

The metrics below appear throughout the output. Each result object reports the exact number. This table is the plain-language reading for clinical interpretation.

| Metric | What it measures | Plain reading |
|---|---|---|
| Sensitivity | True positive rate (also called recall) | Of patients who truly meet the threshold, the fraction the model catches. High sensitivity means few missed positives. |
| Specificity | True negative rate | Of patients who truly fall below the threshold, the fraction the model correctly clears. High specificity means few false alarms. |
| PPV | Positive predictive value (also called precision) | When the model flags a sample as positive, how often it is right. |
| NPV | Negative predictive value | When the model clears a sample as negative, how often it is right. |
| F1 | Harmonic mean of PPV and sensitivity | One number balancing precision and recall. Useful when the classes are imbalanced. |
| MCC | Matthews correlation coefficient | A correlation between predicted and true labels using all four cells of the confusion matrix. Runs from -1 (always wrong) through 0 (random) to +1 (perfect). The most reliable single number on imbalanced data. |
| Accuracy | Fraction of all samples classified correctly | Easy to read, but can mislead on imbalanced data. Read it alongside MCC or F1. |

---

## Worked example

The following walkthrough uses synthetic data shaped like a real TC distribution: 500 patches drawn from a right-skewed Beta(1.5, 3.5) (mean about 0.30), with a noisy model that adds a small upward bias. Every output block below is verbatim, produced by running the example against the current release.

### Setup

```python
import numpy as np
from oncothresh import ThresholdEvaluator, compare_models

rng = np.random.default_rng(seed=42)
n = 500
y_true = rng.beta(a=1.5, b=3.5, size=n)            # ground-truth TC, mean ~0.30
noise = rng.normal(loc=0.02, scale=0.08, size=n)   # slight upward model bias
y_pred = np.clip(y_true + noise, 0.0, 1.0)

ev = ThresholdEvaluator(y_true=y_true, y_pred=y_pred)
```

### 1. `evaluate()`: metrics at a single clinical cutoff

```python
print(ev.evaluate(threshold=0.20))
```

```
ThresholdResult(threshold=0.20, sensitivity=0.948, specificity=0.750,
                ppv=0.875, npv=0.886, f1=0.910, mcc=0.728, accuracy=0.878,
                n=500 [324+/176-])
```

Returns sensitivity, specificity, PPV, NPV, F1, MCC, accuracy, and sample counts. Use it as the building block for everything else. Every other method calls `evaluate()` internally.

### 2. `bootstrap_ci()`: confidence intervals on every metric

```python
print(ev.bootstrap_ci(threshold=0.20, n_bootstrap=1000, random_state=42))
```

```
BootstrapResult(threshold=0.20, n=1000, 95% CI)
  sensitivity : 0.948 (95% CI: 0.923-0.972)
  specificity : 0.750 (95% CI: 0.682-0.812)
  ppv         : 0.875 (95% CI: 0.838-0.908)
  npv         : 0.886 (95% CI: 0.832-0.934)
  f1          : 0.910 (95% CI: 0.886-0.931)
  mcc         : 0.728 (95% CI: 0.663-0.789)
  accuracy    : 0.878 (95% CI: 0.850-0.906)
```

Non-parametric bootstrap with `random_state` for reproducible CI bounds. Use `n_bootstrap=2000` for publication.

### 3. `multi_threshold_report()`: side-by-side at multiple cutoffs

```python
report = ev.multi_threshold_report(thresholds=[0.20, 0.50])
for r in report.results:
    print(r)
```

```
ThresholdResult(threshold=0.20, sensitivity=0.948, specificity=0.750, ...)
ThresholdResult(threshold=0.50, sensitivity=0.894, specificity=0.933, ...)
```

Useful when the same model is deployed across multiple decisions (e.g. TC at 20% for NGS eligibility *and* 50% for chemo response assessment).

### 4. `nnt()`: Number Needed to Test

```python
print(ev.nnt(threshold=0.20))
```

```
NNTResult(threshold=0.20, nnt_positive=1.14, nnt_negative=8.76,
          ppv=0.875, npv=0.886, n=500 [324+/176-])
```

- `nnt_positive = 1.14` means roughly 114 model-positive flags yield 100 true positives.
- `nnt_negative = 8.76` means roughly 1 in 9 cleared patients hides a missed case.

Clinicians quote NNT in meetings far more often than raw PPV/NPV.

### 5. `threshold_sensitivity()`: fragility around the cutoff

```python
print(ev.threshold_sensitivity(threshold=0.20, delta=0.05, step=0.01))
```

```
ThresholdSensitivityResult(nominal=0.20, range=[0.15-0.25], n_points=11,
                           sensitivity@nominal=0.948, specificity@nominal=0.750)
```

Sweeps the threshold across plus or minus `delta` and returns three parallel arrays (`thresholds`, `sensitivities`, `specificities`) plus `shifts` (signed distance from nominal) and `nominal_index` for plotting. Flat curves mean the model is robust to lab-to-lab variation in the cutoff. Steep curves mean it is threshold-brittle.

### 6. `boundary_calibration()`: ECE within the clinical decision zone

```python
print(ev.boundary_calibration(threshold=0.20, window=0.10, n_bins=10))
```

```
BoundaryCalibrationResult(threshold=0.20, window=0.10, zone=[0.10-0.30],
                          n_samples=189, ece=0.0197)
```

Global ECE can hide miscalibration at the cutoff that actually matters. `boundary_calibration()` restricts the reliability analysis to the boundary zone (predictions within plus or minus `window` of the threshold) and reports the weighted ECE there, plus the per-bin reliability data needed to draw a localised reliability diagram.

Rules of thumb (in the same units as the scores):

- ECE below 0.02: well calibrated near the boundary.
- 0.02 to 0.05: moderate boundary miscalibration, document and monitor.
- Above 0.05: systematically biased predictions at the cutoff, consider recalibration.

### 7. `decision_curve()`: net benefit across threshold probabilities

`decision_curve` requires you to name the clinical decision under analysis. Pass `clinical_threshold` (e.g. 0.20 for NGS-eligibility TC) and the disease label is fixed once as `y_true >= clinical_threshold`. Only the clinician's intervention threshold `pt` is swept. `y_pred` must be in `[0, 1]` and is interpreted as the model's predicted probability that `y_true >= clinical_threshold`. If your model emits a raw biomarker score, calibrate (Platt or isotonic) first.

**Calibrating a raw biomarker score to a probability.** A continuous TC or Ki-67 score is not a probability. Passing it to `decision_curve` directly raises `ValueError`. Convert it with scikit-learn's `CalibratedClassifierCV` (or `IsotonicRegression`) against the clinical-threshold label:

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

# Fit on a held-out calibration set. Features are the raw model scores reshaped to
# (n, 1). Labels are 1 where y_true clears the clinical threshold, 0 otherwise.
cal = CalibratedClassifierCV(
    estimator=LogisticRegression(),
    method="isotonic",  # use "sigmoid" for Platt scaling
    cv=5,
)
cal.fit(raw_scores_calib.reshape(-1, 1), (y_true_calib >= 0.20).astype(int))

# Predict P(y_true >= 0.20) for the evaluation set, then call decision_curve.
y_pred_prob = cal.predict_proba(raw_scores_eval.reshape(-1, 1))[:, 1]
ev = ThresholdEvaluator(y_true=y_true_eval, y_pred=y_pred_prob)
dca = ev.decision_curve(clinical_threshold=0.20)
```

Best practice: fit the calibrator on a separate calibration split (not the data you used to train the model), and report calibration quality alongside DCA. `ev.boundary_calibration(threshold=0.20)` is the natural companion.

```python
dca = ev.decision_curve(
    clinical_threshold=0.20,
    thresholds=np.linspace(0.05, 0.50, 10),
)
print(dca)
for pt, nb_m, nb_a in zip(dca.thresholds, dca.net_benefit_model, dca.net_benefit_all):
    print(f"pt={pt:.3f}  nb_model={nb_m:+.4f}  nb_all={nb_a:+.4f}")
```

```
DecisionCurveResult(clinical_threshold=0.20, prevalence=0.648, pt=[0.05-0.50], n_points=10)
pt=0.050  nb_model=+0.6313  nb_all=+0.6295
pt=0.100  nb_model=+0.6200  nb_all=+0.6089
pt=0.150  nb_model=+0.6082  nb_all=+0.5859
pt=0.200  nb_model=+0.5920  nb_all=+0.5600
pt=0.250  nb_model=+0.5560  nb_all=+0.5307
pt=0.300  nb_model=+0.4849  nb_all=+0.4971
pt=0.350  nb_model=+0.4160  nb_all=+0.4585
pt=0.400  nb_model=+0.3540  nb_all=+0.4133
pt=0.450  nb_model=+0.2620  nb_all=+0.3600
pt=0.500  nb_model=+0.2080  nb_all=+0.2960
```

The model adds clinical value wherever `net_benefit_model` exceeds *both* `net_benefit_all` and zero. In this example the curves cross between pt=0.25 and pt=0.30. At pt below the crossover the model beats treat-all. At pt above it the false positives outweigh its true positives under the clinician's harm tolerance, and treat-all becomes the safer policy. `prevalence` is fixed (derived once from `clinical_threshold=0.20`) and is reported on the result so the artefact is self-describing.

Coming from R? The equivalent is `dcurves::dca()`. `oncothresh` produces the same net benefit curves with a Python/sklearn-style interface rather than R's tidy data frame style.

### 8. `compare_models()`: head-to-head at the same cutoff

```python
y_pred_v2 = np.clip(y_true + rng.normal(loc=0.0, scale=0.15, size=n), 0.0, 1.0)
ev_v2 = ThresholdEvaluator(y_true=y_true, y_pred=y_pred_v2)

print(compare_models([ev, ev_v2], threshold=0.20, model_names=["v1", "v2"]))
```

```
CompareModelsResult(threshold=0.20, n_models=2)
Metric                  v1          v2
--------------------------------------
sensitivity          0.948       0.846
specificity          0.750       0.682
ppv                  0.875       0.830
npv                  0.886       0.706
f1                   0.910       0.838
mcc                  0.728       0.532
accuracy             0.878       0.788
```

All evaluators must share the same test set (equal-length `y_true`) so the comparison is on a common denominator. Typical use cases: comparing UNI vs. CONCH features at the 20% TC cutoff, or comparing a new model version against a published baseline.

---

## API at a glance

All methods live on `ThresholdEvaluator(y_true, y_pred)` unless noted.

| Method | Returns | Use when |
|---|---|---|
| `evaluate(threshold)` | `ThresholdResult` | You need sensitivity / specificity / PPV / NPV / F1 / MCC / accuracy at one cutoff |
| `bootstrap_ci(threshold, n_bootstrap=1000, confidence=0.95, random_state=None)` | `BootstrapResult` | You need confidence intervals on every metric |
| `multi_threshold_report(thresholds)` | `MultiThresholdReport` | The same model is deployed at more than one cutoff |
| `nnt(threshold)` | `NNTResult` | You want a clinician-friendly framing: flags per true positive, clearances per missed case |
| `threshold_sensitivity(threshold, delta=0.05, step=0.01)` | `ThresholdSensitivityResult` | You want to know how fragile performance is to small threshold shifts |
| `boundary_calibration(threshold, window=0.10, n_bins=10)` | `BoundaryCalibrationResult` | You want calibration error *at the cutoff*, not globally |
| `decision_curve(clinical_threshold, thresholds=None)` | `DecisionCurveResult` | You want net benefit across a sweep of harm tolerances for a fixed clinical decision. Requires `y_pred` to be a calibrated probability in `[0, 1]` |
| `compare_models(evaluators, threshold, model_names=None)` *(module-level)* | `CompareModelsResult` | You want side-by-side metrics for two or more models at the same cutoff |

Result objects are immutable Pydantic models with `.model_dump_json()` and informative `__str__` output for quick inspection.

---

## Clinical thresholds reference

| Biomarker | Threshold | Clinical decision |
|---|---|---|
| Tumor Cellularity (TC) | 20% | NGS eligibility (CAP/ASCO guidelines) |
| Tumor Cellularity (TC) | 50% | Chemo response assessment |
| Ki-67 | 20% | Breast cancer treatment planning (luminal B vs. A) |
| TMB | 10 mut/Mb | Pembrolizumab eligibility (FDA-approved cutoff) |
| PD-L1 CPS | 1, 10, 20 | Anti-PD-1 therapy selection |

---

## Versioning and stability

`v0.1` is an early release. The eight methods documented above are the locked surface for v0.1. Signatures and result fields will be preserved through v0.2 patches. Additional methods (`shap_threshold_analysis`, `bias_analysis`, `full_report`) are planned for v0.2.

---

## Background and references

The basic diagnostic metrics (sensitivity, specificity, PPV, NPV, F1, accuracy) are standard classification and diagnostic-test measures with no single originating paper. The clinical methods and the two metrics that do have a clear origin are referenced below. Full citations with DOIs are in the JOSS paper bibliography (`paper/paper.bib`).

- **The clinical-threshold gap and the case for decision-analytic evaluation:** Van Calster et al., *The Lancet Digital Health*, 2025.
- **Decision Curve Analysis (net benefit):** Vickers and Elkin, *Medical Decision Making*, 2006.
- **Matthews correlation coefficient (MCC):** Matthews, *Biochimica et Biophysica Acta*, 1975.
- **Number Needed to Treat (the basis for NNT):** Laupacis, Sackett, and Roberts, *New England Journal of Medicine*, 1988.
- **Pathology foundation models in the first application:** UNI (Chen et al., *Nature Medicine*, 2024) and CONCH (Lu et al., *Nature Medicine*, 2024).
- **Reference DCA implementation used for validation:** `dcurves` (Sjoberg).

---

## Citing oncothresh

A JOSS paper is in preparation (target submission: 2026-06-08). DOI and BibTeX entry will be added here on acceptance. Until then, please cite the GitHub repository.

---

## Development

```bash
git clone https://github.com/omkaradhali/oncothresh.git
cd oncothresh
uv sync --extra dev

uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ --cov=src/oncothresh
```

CI matrix runs on Python 3.10, 3.11, 3.12, and 3.13 with a 70% coverage gate.

---

## License

MIT. See [LICENSE](LICENSE).
