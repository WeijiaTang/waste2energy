# Waste2Energy

This repository builds submission-oriented data assets and machine-learning baselines for Waste2Energy research.

## Current ML-ready Datasets

The first-round XGBoost baselines use the processed matrix datasets under `data/processed/model_ready`:

- `ml_training_matrix_htc_direct.csv`
- `ml_training_matrix_pyrolysis_direct.csv`
- `paper1_ml_matrix_htc_scope.csv`

These matrix datasets are the direct training interface because they already exclude the explanatory object columns that remain in the more human-readable dataset exports.

## Train One Baseline

Run a single dataset-target combination:

```powershell
python scripts/train_xgboost_baseline.py --dataset htc_direct --target product_char_yield_pct
```

You can also use the installed CLI:

```powershell
waste2energy-train --dataset pyrolysis_direct --target energy_recovery_pct
```

Train a non-XGBoost tree baseline:

```powershell
waste2energy-train --model rf --dataset htc_direct --target carbon_retention_pct
waste2energy-train --model extra_trees --dataset pyrolysis_direct --target energy_recovery_pct
```

## Run the Full Baseline Suite

Run all supported datasets against all four target columns:

```powershell
python scripts/run_xgboost_baseline_suite.py
```

Run the full traditional-ML comparison suite across XGBoost, Random Forest, Extra Trees, ElasticNet, and Gradient Boosting:

```powershell
python scripts/run_traditional_ml_suite.py
```

This writes per-run artifacts under:

- `outputs/xgboost/<dataset>/<target>/` for XGBoost
- `outputs/xgboost/<model>/<dataset>/<target>/` for RF, Extra Trees, ElasticNet, and Gradient Boosting

- `model.json` or `model.joblib`
- `metrics.json`
- `predictions.csv`
- `feature_importance.csv`
- `run_config.json`

The suite also writes:

- `outputs/xgboost/baseline_suite_summary.csv`
- `outputs/xgboost/baseline_suite_summary.json`
- `outputs/xgboost/traditional_ml_suite_summary.csv`
- `outputs/xgboost/traditional_ml_suite_summary.json`

## Validation Tiers

The training package now supports multiple validation strategies through `--split-strategy`:

- `recommended`: legacy row-level split for quick baseline comparison
- `strict_group`: feature-group-aware split that blocks duplicate-feature leakage across splits
- `leave_source_repo_out`: source-repository holdout when at least two real source groups exist
- `leave_study_out`: study-level holdout when citation or study metadata are available

Examples:

```powershell
waste2energy-train --model xgboost --dataset htc_direct --target product_char_yield_pct --split-strategy strict_group
waste2energy-train --model extra_trees --dataset pyrolysis_direct --target product_char_yield_pct --split-strategy leave_study_out
```

Current result files worth comparing are:

- `outputs/xgboost/traditional_ml_suite_summary.csv`
- `outputs/xgboost/traditional_ml_suite_summary_strict_group.csv`
- `outputs/xgboost/leave_study_out/traditional_ml_suite_summary_leave_study_out.csv`
- `outputs/xgboost/leave_study_out_htc_paper1/traditional_ml_suite_summary_leave_study_out.csv`

For Paper 1 writing, prefer `strict_group` as the main benchmark table and use `leave_study_out` as the stronger generalization stress test.

## Planning Layer Baseline

Run the current planning-layer baseline:

```powershell
python scripts/run_planning_baseline.py
```

Or use the installed CLI when the package is available:

```powershell
waste2energy-plan --top-k-per-scenario 5 --max-portfolio-candidates 3 --deployable-capacity-fraction 0.85
```

The planning CLI now does two things in one run:

- scores all optimization-ready cases with the current energy, environment, and cost objectives
- builds a constraint-aware scenario portfolio using capped candidate share, subtype diversification, and deployable-capacity limits

Key planning artifacts are written under `outputs/planning/baseline/`:

- `scenario_constraints.csv`
- `scenario_recommendations.csv`
- `pareto_candidates.csv`
- `portfolio_allocations.csv`
- `portfolio_summary.csv`
- `scenario_summary.csv`
- `run_config.json`

Interpretation rule for the current planning layer:

- `scenario_recommendations.csv` is the ranked single-case view
- `portfolio_allocations.csv` is the planning-layer allocation view
- `scenario_summary.csv` is the manuscript-facing scenario digest
- `run_config.json` records whether each objective is still proxy-based or already publication-ready

## Scenario And Robustness Layer

Run the current scenario and robustness baseline:

```powershell
python scripts/run_scenario_robustness.py
```

Or use the installed CLI:

```powershell
waste2energy-scenario --deployable-capacity-fraction 0.85 --max-portfolio-candidates 3
```

This layer reuses the planning engine, applies a bounded registry of parameter perturbations, and exports reviewer-facing robustness evidence under `outputs/scenarios/baseline/`.

Key artifacts are:

- `stress_registry.csv`
- `stress_test_summary.csv`
- `decision_stability.csv`
- `cross_scenario_stability.csv`
- `uncertainty_summary.csv`
- `run_config.json`

Interpretation rule for the current scenario layer:

- `stress_test_summary.csv` shows how portfolio outcomes change under parameter perturbations
- `decision_stability.csv` tracks which candidate designs survive repeatedly within each real scenario
- `cross_scenario_stability.csv` shows which baseline portfolio members persist across the actual California scenario set
- `uncertainty_summary.csv` is the compact manuscript-facing envelope for outcome variation and top-case switching

Current boundary:

- the robustness layer is built on the same HTC mixed-feed planning dataset used by the planning layer
- robustness conclusions should therefore be written as planning-parameter stress evidence, not as full four-pathway uncertainty proof

## Operation Appendix Environment

Build the current planning-derived appendix environment:

```powershell
python scripts/run_operation_env_baseline.py
```

Or use the installed CLI:

```powershell
waste2energy-operation --horizon-steps 12
```

This layer reads the stable planning/scenario outputs and derives one environment per real scenario from:

- the dominant candidate
- candidate-capacity and scenario-budget boundaries
- uncertainty-derived disturbance amplitudes

Key artifacts under `outputs/operation/baseline/` are:

- `operation_environment_specs.csv`
- `baseline_rollout_steps.csv`
- `baseline_rollout_summary.csv`
- `run_config.json`

Current environment definition:

- state: throughput level, candidate-share coverage, severity offset, disturbance multipliers, and boundary pressure
- action: throughput adjustment and severity adjustment, each in `{-1, 0, +1}`
- reward: weighted energy gain plus environmental gain minus cost and boundary-violation penalties

Current appendix boundary:

- the environment and simple baseline policies are ready
- DRL training is the next layer, not part of the current main-paper logic

## Gymnasium And SB3 Appendix

The appendix environment is now also exposed as a `gymnasium`-style environment and supports `stable-baselines3` training with `SAC` and `TD3`.

Run a short RL smoke test:

```powershell
python scripts/run_operation_rl.py --algorithm sac --total-timesteps 256 --evaluation-episodes 5
python scripts/run_operation_rl.py --algorithm td3 --total-timesteps 256 --evaluation-episodes 5
```

Or use the CLI directly:

```powershell
waste2energy-operation --mode rl --algorithm sac --total-timesteps 256 --evaluation-episodes 5
```

RL appendix artifacts are written under:

- `outputs/operation/rl/sac/`
- `outputs/operation/rl/td3/`

Key files are:

- `operation_environment_specs.csv`
- `training_summary.csv`
- `evaluation_rollouts.csv`
- `run_config.json`

Current DRL appendix rule:

- treat the current SAC/TD3 runs as a runnable appendix code path and smoke-tested benchmark scaffold
- do not over-interpret short training runs as final scientific evidence
- scale `total_timesteps`, add repeated seeds, and compare against the deterministic baselines before writing substantive appendix conclusions

## Formal Appendix Comparison

Run the current multi-seed RL-vs-baseline comparison:

```powershell
python scripts/run_operation_rl.py --mode compare --seeds 42,43 --total-timesteps 256 --evaluation-episodes 5
```

This writes a formal comparison package under `outputs/operation/comparison/`:

- `baseline_policy_summary.csv`
- `sac_training_summary.csv`
- `td3_training_summary.csv`
- `sac_seed_aggregate_summary.csv`
- `td3_seed_aggregate_summary.csv`
- `rl_vs_baseline_comparison.csv`
- `run_config.json`

Interpretation rule:

- `baseline_policy_summary.csv` is the deterministic control reference table
- `*_training_summary.csv` stores per-seed RL runs
- `*_seed_aggregate_summary.csv` is the manuscript-facing RL aggregate per scenario
- `rl_vs_baseline_comparison.csv` is the appendix comparison table that ranks SAC, TD3, and the deterministic baselines within each scenario

Current evidence rule:

- use the comparison table to discuss relative behavior of RL and heuristic control under the current HTC-derived appendix environment
- do not write strong RL superiority claims until the comparison is expanded with longer training, more seeds, and sensitivity checks
