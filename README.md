# oncothresh

**Clinical threshold evaluation for oncology AI models.**

`oncothresh` evaluates continuous oncology AI models at predefined clinical decision thresholds — the exact cutoffs that govern patient treatment decisions.

## Installation

```bash
pip install oncothresh
```

## Quick Start

```python
from oncothresh import ThresholdEvaluator

ev = ThresholdEvaluator(y_true=tc_scores, y_pred=model_predictions)

# Evaluate at the 20% TC threshold (NGS eligibility cutoff)
result = ev.evaluate(threshold=0.20)
print(result.sensitivity, result.specificity)

# With 95% confidence intervals
ci = ev.bootstrap_ci(threshold=0.20, n_bootstrap=1000, random_state=42)
print(ci.sensitivity)  # 0.923 (95% CI: 0.871–0.962)

# Side-by-side at multiple clinical cutoffs
report = ev.multi_threshold_report(thresholds=[0.20, 0.50])
```

## Decision Curve Analysis

`decision_curve()` computes net benefit across a sweep of threshold probabilities,
answering whether using the model to guide clinical decisions is better than
treating everyone or no one.

```python
import numpy as np

result = ev.decision_curve(thresholds=np.linspace(0.05, 0.50, 46))

# result.thresholds         — the pt values swept
# result.net_benefit_model  — net benefit of the model at each pt
# result.net_benefit_all    — net benefit of treating everyone at each pt
# result.net_benefit_none   — always 0.0 (treat nobody)
```

The model adds clinical value wherever `net_benefit_model` exceeds both
`net_benefit_all` and zero.

**Coming from R?** The equivalent in R is `dcurves::dca()` from the
[dcurves](https://cran.r-project.org/package=dcurves) package (Sjoberg et al.).
`oncothresh` produces the same net benefit curves; the interface follows
Python/sklearn conventions rather than R's tidy data frame style.

## Clinical Thresholds Reference

| Biomarker | Threshold | Clinical Decision |
|---|---|---|
| Tumor Cellularity (TC) | 20% | NGS eligibility (CAP/ASCO guidelines) |
| Tumor Cellularity (TC) | 50% | Chemo response assessment |
| Ki-67 | 20% | Breast cancer treatment planning |
| TMB | 10 mut/Mb | Pembrolizumab eligibility |
| PD-L1 CPS | 1, 10, 20 | Anti-PD-1 therapy selection |

## License

MIT
