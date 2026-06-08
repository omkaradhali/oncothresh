---
title: 'oncothresh: Clinical-threshold evaluation for oncology AI models'
tags:
  - Python
  - oncology
  - computational pathology
  - clinical decision support
  - model evaluation
  - decision curve analysis
  - calibration
authors:
  - name: Omkar Adhali
    orcid: 0009-0002-9955-5163
    affiliation: 1
affiliations:
  - name: Independent Researcher, New York, NY, USA
    index: 1
date: 8 June 2026
bibliography: paper.bib
---

# Summary

Continuous oncology AI models predict quantities such as tumor cellularity (TC),
Ki-67 proliferation index, tumor mutational burden (TMB), and PD-L1 expression.
In clinical practice these continuous scores are not used directly. A fixed
threshold converts each score into a binary decision that governs patient care,
for example a 20% TC cutoff for next-generation sequencing (NGS) eligibility, or
10 mut/Mb TMB for pembrolizumab eligibility. The clinically relevant question is
therefore not "how well does the model agree overall?" but "how reliable is the
model *at the exact cutoff that decides treatment?*"

`oncothresh` is a small, focused Python package that answers that question. Given
ground-truth scores and model predictions, a stateful `ThresholdEvaluator` works
at one or more clinical thresholds and returns:

- standard classification metrics (sensitivity, specificity, PPV, NPV, F1, MCC,
  accuracy) with bias-corrected and accelerated (BCa) bootstrap confidence
  intervals [@efron1987],
- Number-Needed-to-Test (NNT),
- threshold-sensitivity curves that quantify how fragile performance is to small
  shifts in the cutoff,
- boundary-weighted calibration error restricted to the decision zone,
- Decision Curve Analysis (DCA) net benefit [@vickers2006],
- and side-by-side model comparison.

Results are returned as immutable, serialisable objects with concise printed
summaries, so the same output serves an ML engineer who needs exact values for a
methods section and a pathologist who needs to know what those values mean for
patients.

# Statement of need

Tools for benchmarking pathology foundation models (Patho-Bench, PathBench,
PathBench-MIL) primarily evaluate models globally, across the whole score range.
Performance at predefined clinical thresholds, with the uncertainty quantification
needed to trust a decision at that cutoff, is typically left to ad-hoc,
per-project implementations, even though the threshold is where the clinical
decision is actually made. Recent guidance on evaluating clinical AI prediction
models emphasises that operating points chosen by statistical optimisation can be
inconsistent with decision theory, and argues for decision-analytic measures such
as net benefit at the thresholds where decisions are actually made
[@vancalster2025]. In day-to-day work that gap is bridged by repeated boilerplate:
re-binarising scores, hand-rolling confidence intervals, and reimplementing net
benefit, a recurring source of subtle statistical error in exactly the analyses
that inform clinical decisions.

The adjacent tools do not fill the gap. `scikit-learn` [@scikit-learn] provides
the underlying metrics but no clinical-threshold framing, no NNT, and no DCA.
`dcurves` [@dcurves] provides gold-standard decision curve analysis (including
survival endpoints) but is focused on net benefit alone and is rooted in the R and
tidy-dataframe idiom. `oncothresh` occupies the space between them: a
threshold-first evaluation layer with a scikit-learn-style API, aimed at the
people building and validating oncology models in Python.

The package targets three audiences. **ML engineers and biostatisticians** in
computational pathology get bootstrap CIs and comparison tables ready for a
manuscript. **Regulatory and QA teams** get threshold-sensitivity output that
documents device stability when staining or scanner protocols shift the effective
cutoff. **Clinicians** get plain, printable summaries, for example NNT (derived
from PPV and NPV) framed as "flags per true positive" and "clearances per missed
case", rather than raw matrices. This dual-audience design is deliberate: it keeps
outputs mathematically rigorous for ML engineering while remaining accessible for
clinical interpretation.

Two design choices follow directly from the requirement to be safe for clinical
reporting. First, `decision_curve()` enforces correct DCA semantics: the disease
label is fixed once as `y_true >= clinical_threshold` and only the clinician's
intervention threshold is swept, and `y_pred` must be a calibrated probability in
`[0, 1]` (the package raises rather than silently producing an uninterpretable
curve from a raw biomarker score). Second, because a statistical bug in a tool
used for clinical reporting would be far more damaging than a missing feature, the
DCA implementation is cross-validated against `dcurves` [@dcurves] in the test
suite, which enforces agreement within a strict floating-point tolerance (below
1e-9, and in practice the two implementations agree to about 1e-16). Calibration
is similarly treated as a first-class, threshold-local concern via
`boundary_calibration()`, which restricts reliability analysis to the decision
zone where miscalibration changes a treatment decision, information that a single
global calibration number hides.

`oncothresh` is intended both as practical infrastructure and as a citable
methodological artefact. Its first application is a benchmark of pathology
foundation models (UNI [@uni2024], CONCH [@conch2024]) for automated tumor
cellularity assessment, where it provides the clinical-threshold analysis at the
20% and 50% TC cutoffs. It is built on `numpy` [@numpy], `scipy` [@scipy], and
`scikit-learn` [@scikit-learn], with `pydantic` for validated, serialisable result
objects, and supports Python 3.10 to 3.13 to remain compatible with hospital IT
environments.

# Acknowledgements

We thank the maintainers of `dcurves` for a clear reference implementation of
decision curve analysis against which `oncothresh` is validated.

# References
