# Waste2Energy

Waste2Energy is a layered research repository for a JOCP-oriented mixed organic waste planning study. The current repository centers on a California regional case, surrogate-assisted pathway evaluation, explicit multi-objective planning, scenario robustness, and manuscript-facing evidence qualification.

## Current Scientific Boundary

The main-paper code path is:

`data-process -> model_ready tables -> surrogate evaluation -> planning optimization -> scenario robustness`

Current planning compares four pathway families in one shared optimization-ready table:

- `baseline`
- `ad`
- `pyrolysis`
- `htc`

Current manuscript-safe boundary:

- the planning layer now uses trained surrogate artifacts when a pathway has model support
- HTC and pyrolysis are surrogate-enabled pathways in the current codebase
- baseline and AD currently use documented static fallbacks rather than trained pathway surrogates
- manuscript claims are intended to be supported by planning, benchmark, scenario, and audit artifacts rather than by appendix-only controller comparisons

## Environment Setup

### Option 1: `uv` (preferred)

```powershell
uv venv
.\.venv\Scripts\activate
uv sync --dev
```

### Option 2: local `.venv` with `pip`

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

The repository should not rely on packages already present in system Python. If imports such as `pandas`, `python-dateutil`, `pyomo`, `pymoo`, or `pytest` fail, recreate the environment and reinstall from `pyproject.toml`.

## Primary CLI Entry Points

Install the package first, then use:

```powershell
waste2energy-train
waste2energy-plan
waste2energy-benchmark
waste2energy-scenario
waste2energy-audit
```

Equivalent module entry points also work:

```powershell
.\.venv\Scripts\python.exe -m waste2energy.planning.cli
.\.venv\Scripts\python.exe -m waste2energy.benchmark_cli
.\.venv\Scripts\python.exe -m waste2energy.scenarios.cli
.\.venv\Scripts\python.exe -m waste2energy.audit
```

## Surrogate Layer

The training package supports multiple traditional ML models:

- `xgboost`
- `catboost`
- `lightgbm`
- `rf`
- `extra_trees`
- `elastic_net`
- `gradient_boosting`

Example single-run training:

```powershell
waste2energy-train --model extra_trees --dataset paper1_htc_scope --target energy_recovery_pct --split-strategy strict_group
```

Example full traditional-ML suite:

```powershell
python scripts/run_traditional_ml_suite.py
```

Surrogate artifacts now conceptually live under `outputs/surrogates/`. The repository still supports the historical `outputs/xgboost/` root for backward compatibility, and current code will read either location automatically.

Key surrogate artifacts:

- `traditional_ml_suite_summary.csv`
- `traditional_ml_suite_summary_strict_group.csv`
- `traditional_ml_suite_summary_leave_study_out.csv`
- per-model `metrics.json`
- per-model `predictions.csv`
- per-model `feature_importance.csv`
- per-model `run_config.json`

Current surrogate predictions exported into the planning layer also include model-side uncertainty diagnostics:

- target-level prediction intervals (`*_ci_lower`, `*_ci_upper`)
- interval-width proxies (`*_prediction_std`, `*_uncertainty_ratio`)
- interval provenance fields (`*_uncertainty_method`, `*_uncertainty_calibration_count`)

## Planning Layer

Refresh the multi-pathway planning dataset:

```powershell
python scripts/data-process/11_build_planning_mult_pathway_dataset.py
```

Run the baseline planning workflow:

```powershell
waste2energy-plan
```

Or with explicit controls:

```powershell
waste2energy-plan `
  --objective-weight-preset balanced_cleaner_production `
  --robustness-factor 0.35 `
  --uncertainty-penalty-mode prefer_interval_mean `
  --carbon-budget-factor 1.0 `
  --optimization-method pyomo `
  --pyomo-solver appsi_highs `
  --pareto-point-count 12
```

What the planning layer now does:

- loads optimization-ready candidate rows
- calls the surrogate evaluator for supported pathways
- attaches prediction uncertainty and applies a robust penalty
- formulates explicit objective and constraint inputs
- solves a constrained allocation problem with Pyomo when an external solver is available
- exports recommendation tables and Pareto-front tables

Key planning artifacts under `outputs/planning/`:

- `scored_cases.csv`
- `scenario_recommendations.csv`
- `portfolio_allocations.csv`
- `portfolio_summary.csv`
- `scenario_summary.csv`
- `pathway_summary.csv`
- `pareto_candidates.csv`
- `pareto_front.csv`
- `scenario_constraints.csv`
- `surrogate_predictions.csv`
- `optimization_diagnostics.csv`
- `run_config.json`

Interpretation rule:

- `scenario_recommendations.csv` is the ranked case-level view
- `portfolio_allocations.csv` is the constrained optimized allocation view
- `pareto_front.csv` is the portfolio-level trade-off surface for cost vs environment vs energy
- `surrogate_predictions.csv` records whether each pathway row came from trained surrogate inference or documented fallback mode
- `optimization_diagnostics.csv` records the solver backend and status by scenario
- `uncertainty_penalty_mode` controls whether robustness penalties use the mean interval-derived ratio, the maximum interval-derived ratio, or only an explicit combined ratio

Current solver rule:

- default professional path: `Pyomo + appsi_highs`
- secondary Pyomo fallback: `highs`, then `glpk`
- last-resort fallback: `scipy_milp`

## Ablation And Benchmark Layer

Run the planning ablation and benchmark suite:

```powershell
waste2energy-benchmark
```

Key benchmark outputs under `outputs/benchmark/baseline/`:

- `benchmark_summary.csv`
- `benchmark_shift_summary.csv`
- `benchmark_allocations.csv`
- `benchmark_scenario_summary.csv`
- `benchmark_diagnostics.csv`
- `benchmark_bootstrap_shift_samples.csv`
- `benchmark_statistical_summary.csv`
- `run_config.json`

The benchmark suite compares the current evidence-aware planning baseline against targeted counterfactual variants, including:

- classic multi-objective optimizer without evidence-aware or robustness-aware adjustments
- no robustness penalty
- no evidence penalty
- no carbon constraint
- ranking-dominated unconstrained portfolio
- greedy weighted-score heuristic baseline

Use `benchmark_shift_summary.csv` to inspect which pathways or selected case sets change when evidence-aware penalties or deployment constraints are relaxed.
Use `benchmark_statistical_summary.csv` when you need bootstrap-backed uncertainty intervals or shift rates for benchmark deltas. You can enable repeated benchmark runs with:

```powershell
waste2energy-benchmark --bootstrap-replicates 30 --bootstrap-random-seed 42
```

## Scenario And Robustness Layer

Run the baseline robustness workflow:

```powershell
waste2energy-scenario
```

The scenario layer reuses the same planning entry point, then aggregates:

- stress-test summaries
- within-scenario decision stability
- cross-scenario persistence
- uncertainty envelopes
- recommendation-confidence summaries that combine baseline selection, stress persistence, and evidence maturity
- refreshed manuscript-facing planning tables based on the latest scenario evidence
- refreshed audit outputs when the linked planning outputs are available

Key outputs under `outputs/scenarios/`:

- `stress_registry.csv`
- `stress_test_summary.csv`
- `decision_stability.csv`
- `cross_scenario_stability.csv`
- `uncertainty_summary.csv`
- `run_config.json`

## Sensitivity Analysis

The planning package now includes a weight-sensitivity utility that perturbs the shared objective-weight system by small increments and tracks changes in:

- top-ranked case IDs
- selected portfolio case IDs
- allocation and recommendation stability summaries

It is exposed in Python for scripted experiments:

```powershell
.\.venv\Scripts\python.exe -c "from waste2energy.planning.sensitivity import analyze_weight_sensitivity; a,b = analyze_weight_sensitivity(); print(a.head())"
```

## Confirmatory Audit

Regenerate the manuscript-facing audit bundle:

```powershell
waste2energy-audit
```

Audit outputs under `outputs/audit/`:

- `ml_split_coverage_summary.csv`
- `ml_best_result_summary.csv`
- `ml_claim_flag_table.csv`
- `planning_claim_flag_table.csv`
- `planning_recommendation_confidence_summary.csv`
- `planning_transferability_risk_summary.csv`
- `benchmark_claim_summary.csv`
- `benchmark_manuscript_sentences.csv`
- `artifact_inventory.csv`
- `audit_manifest.json`

`planning_claim_flag_table.csv` is the compact manuscript-facing inventory for pathway claims after planning/scenario refresh. It is the safest audit-side table to consult before writing pathway recommendations into the paper.
`planning_recommendation_confidence_summary.csv` provides a narrower pathway-level confidence grade so recommendation language can reflect both evidence maturity and robustness persistence.
`planning_transferability_risk_summary.csv` summarizes how much of each selected scenario portfolio is backed by conditionally transferable versus auxiliary-only pathway evidence.
`benchmark_claim_summary.csv` translates ablation outputs into scenario-level evidence about which design elements materially shape the exported recommendation.
`benchmark_manuscript_sentences.csv` gives manuscript-safe aggregate wording for benchmark evidence so the paper can describe the most influential design choices without overclaiming.

Run manuscript sync when you want benchmark-ready LaTeX macros and figure-table exports:

```powershell
.\.venv\Scripts\python.exe scripts\run_manuscript_sync.py
```

This now refreshes benchmark-facing manuscript artifacts such as:

- `data/processed/figures_tables/paper1_benchmark_results_table.csv`
- `data/processed/figures_tables/paper1_benchmark_claim_summary.csv`
- `data/processed/figures_tables/paper1_benchmark_narrative.json`
- `data/processed/figures_tables/paper1_benchmark_section_templates.json`
- `data/processed/figures_tables/paper1_benchmark_section_templates.md`
- `data/processed/figures_tables/paper1_benchmark_section_templates.tex`
- benchmark-related macros in `waste2energy-paper/sections/99-auto-macros.tex`

## Minimal Reviewer Workflow

For a fresh reviewer-oriented reproduction pass, the shortest reliable path is:

```powershell
waste2energy-plan
waste2energy-scenario
waste2energy-audit
.\.venv\Scripts\python.exe -m pytest -q
```

This sequence ensures:

- planning outputs are regenerated from the current code
- scenario outputs and manuscript-facing planning tables are refreshed from the same evidence base
- audit tables are rebuilt from the refreshed outputs
- repository-owned regression tests confirm the core workflow contracts

## Smoke Tests

Run the repository smoke tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Use the explicit `tests/` target for reviewer reproduction. A root-level `pytest` may also collect vendored reference-repository suites under `Reference/`, which are outside the Waste2Energy smoke-test contract.

The current smoke suite covers:

- planning baseline
- scenario robustness baseline
- shared weight-system alignment
- surrogate evaluator interface and fallback behavior

## Notes For Reviewers

- The planning layer is no longer only a static weighted ranking scaffold; it now couples surrogate outputs, uncertainty-aware robustification, and explicit optimization constraints.
- Pyomo, pymoo, and highspy are installed in the current environment. The repository now prefers `Pyomo + appsi_highs` for the planning solver path and falls back only when a stronger backend is unavailable.
- Baseline and AD remain documented fallback pathways in the current surrogate layer. This is intentional and should be described as a staged evidence boundary rather than hidden as if all pathways had equal surrogate maturity.
- controller-level appendix experiments may exist in the repository, but they should not be presented as the main-paper center of gravity.
