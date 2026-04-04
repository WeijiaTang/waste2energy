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
