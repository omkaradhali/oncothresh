# Contributing to oncothresh

Thanks for your interest. `oncothresh` is a small, focused library — contributions that sharpen the clinical-threshold use case are very welcome.

## Scope

`oncothresh` evaluates oncology AI models at predefined clinical thresholds on **binary outcomes derived from continuous scores** (e.g. TC, Ki-67, TMB, PD-L1). Pull requests inside this scope — new metrics, plotting, calibration utilities, better docs — are in scope and welcome.

Out of scope for now:

- **Time-to-event / survival endpoints.** Use [`dcurves`](https://github.com/ddsjoberg/dcurves) for net benefit on survival outcomes; oncothresh intentionally does not duplicate that work.
- **Multi-class classification.** The library is built for the single-threshold binary case.

If you have a clinical use case that does not fit, open an issue first to discuss before sending a PR.

## Development setup

`oncothresh` uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/omkaradhali/oncothresh
cd oncothresh
uv sync --extra dev
```

Optional extras:

```bash
uv sync --extra dev --extra plotting   # adds matplotlib for the (future) plotting submodule
```

## Running checks locally

The same checks CI runs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ --cov=src/oncothresh --cov-fail-under=70
```

Tests must pass on Python 3.10, 3.11, 3.12, and 3.13. The CI matrix covers all four; locally, picking one is fine.

## Adding a new method

If you are adding a new `ThresholdEvaluator` method:

1. **Add the public method** to `src/oncothresh/_evaluator.py` with a clinically-framed docstring (motivation, formula, parameter ranges, how to interpret the result).
2. **Add a frozen result dataclass** to `src/oncothresh/_results.py`. Results are Pydantic models with `model_config = ConfigDict(frozen=True)` and a custom `__str__` that prints concisely for clinical readers.
3. **Cover the public API** with tests in `tests/test_evaluator.py`. At minimum: type check, hand-verifiable known case, edge cases (empty input, all-positive, all-negative), result-is-frozen.
4. **Update the README** "API at a glance" table and add a worked example under the existing section structure if the method is user-facing.

## Statistical correctness

This library is used to inform clinical decisions and to support publications. Statistical bugs matter. Two rules:

- Any new statistical method must include a reference (paper or canonical implementation) in the docstring.
- When implementing something that already exists in R or Python (e.g. `dcurves`, `scikit-learn`), include a regression test in `tests/validation/` that reproduces published numbers or matches the reference implementation to ≥4 decimal places.

## Reporting bugs

Open a GitHub issue with:

- The version of `oncothresh` (`oncothresh.__version__`).
- A minimal reproducer (~10 lines) using synthetic data.
- Expected vs observed output.
- Why you believe the observed output is wrong — citation to a paper or reference implementation is most useful.

Statistical bug reports get priority.

## Commit + PR style

- Conventional commit subjects: `fix:`, `feat:`, `test:`, `docs:`, `refactor:`, `ci:`. Keep subjects single-line.
- One logical change per PR. Mixing a refactor with a bug fix makes the diff hard to review.
- Pre-merge: `ruff check`, `ruff format --check`, and `pytest` must all be green locally and in CI.

## License

By contributing, you agree that your contributions will be licensed under the MIT License (see `LICENSE`).
