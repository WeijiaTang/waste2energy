from __future__ import annotations

import pandas as pd

from waste2energy.surrogates.artifacts import build_ranked_suite_summary_frame, build_selected_models_manifest
from waste2energy.surrogates.train import freeze_selected_models_from_results


def test_ranked_suite_summary_prefers_validation_metrics_over_test_metrics():
    rows = [
        {
            "model_key": "xgboost",
            "dataset_key": "demo",
            "target_column": "target_a",
            "split_strategy": "strict_group",
            "feature_count": 4,
            "train_rows": 10,
            "validation_rows": 4,
            "test_rows": 4,
            "metrics_path": "m1",
            "predictions_path": "p1",
            "feature_importance_path": "f1",
            "model_path": "model1",
            "run_config_path": "cfg1",
            "train_r2": 0.95,
            "validation_r2": 0.70,
            "validation_rmse": 2.0,
            "validation_mae": 1.0,
            "test_r2": 0.90,
            "test_rmse": 1.5,
            "test_mae": 0.9,
        },
        {
            "model_key": "rf",
            "dataset_key": "demo",
            "target_column": "target_a",
            "split_strategy": "strict_group",
            "feature_count": 4,
            "train_rows": 10,
            "validation_rows": 4,
            "test_rows": 4,
            "metrics_path": "m2",
            "predictions_path": "p2",
            "feature_importance_path": "f2",
            "model_path": "model2",
            "run_config_path": "cfg2",
            "train_r2": 0.92,
            "validation_r2": 0.82,
            "validation_rmse": 2.2,
            "validation_mae": 1.1,
            "test_r2": 0.75,
            "test_rmse": 2.5,
            "test_mae": 1.8,
        },
    ]

    ranked = build_ranked_suite_summary_frame(rows)

    assert ranked.loc[0, "model_key"] == "rf"
    assert bool(ranked.loc[0, "is_selected_model"])
    assert ranked.loc[0, "selection_metric_name"] == "validation_r2"
    assert ranked.loc[0, "selection_metric_value"] == ranked.loc[0, "validation_r2"]


def test_selected_models_manifest_contains_one_selected_row_per_dataset_target():
    ranked = build_ranked_suite_summary_frame(
        [
            {
                "model_key": "xgboost",
                "dataset_key": "demo",
                "target_column": "target_a",
                "split_strategy": "strict_group",
                "feature_count": 4,
                "train_rows": 10,
                "validation_rows": 4,
                "test_rows": 4,
                "metrics_path": "m1",
                "predictions_path": "p1",
                "feature_importance_path": "f1",
                "model_path": "model1",
                "run_config_path": "cfg1",
                "validation_r2": 0.70,
                "validation_rmse": 2.0,
                "validation_mae": 1.0,
                "test_r2": 0.90,
                "test_rmse": 1.5,
                "test_mae": 0.9,
            },
            {
                "model_key": "rf",
                "dataset_key": "demo",
                "target_column": "target_a",
                "split_strategy": "strict_group",
                "feature_count": 4,
                "train_rows": 10,
                "validation_rows": 4,
                "test_rows": 4,
                "metrics_path": "m2",
                "predictions_path": "p2",
                "feature_importance_path": "f2",
                "model_path": "model2",
                "run_config_path": "cfg2",
                "validation_r2": 0.82,
                "validation_rmse": 2.2,
                "validation_mae": 1.1,
                "test_r2": 0.75,
                "test_rmse": 2.5,
                "test_mae": 1.8,
            },
        ]
    )

    manifest = build_selected_models_manifest(ranked)

    assert len(manifest) == 1
    assert manifest.loc[0, "selected_model_key"] == "rf"
    assert manifest.loc[0, "selection_status"] == "selected_on_validation"


def test_freeze_selected_models_from_results_writes_refit_manifest_and_artifacts(tmp_path):
    results = [
        {
            "model_key": "rf",
            "dataset_key": "pyrolysis_direct",
            "target_column": "product_char_yield_pct",
            "split_strategy": "strict_group",
            "feature_count": 11,
            "row_counts": {"train": 10, "validation": 4, "test": 3},
            "metrics": {
                "train": {"r2": 0.9, "rmse": 1.0, "mae": 0.8},
                "validation": {"r2": 0.82, "rmse": 1.4, "mae": 1.1},
                "test": {"r2": 0.55, "rmse": 1.8, "mae": 1.5},
            },
            "outputs": {
                "metrics": "benchmark_metrics_rf.json",
                "predictions": "benchmark_predictions_rf.csv",
                "feature_importance": "benchmark_feature_importance_rf.csv",
                "model": "benchmark_model_rf.joblib",
                "run_config": "benchmark_run_config_rf.json",
            },
        },
        {
            "model_key": "xgboost",
            "dataset_key": "pyrolysis_direct",
            "target_column": "product_char_yield_pct",
            "split_strategy": "strict_group",
            "feature_count": 11,
            "row_counts": {"train": 10, "validation": 4, "test": 3},
            "metrics": {
                "train": {"r2": 0.95, "rmse": 0.9, "mae": 0.7},
                "validation": {"r2": 0.60, "rmse": 1.8, "mae": 1.4},
                "test": {"r2": 0.88, "rmse": 1.1, "mae": 0.9},
            },
            "outputs": {
                "metrics": "benchmark_metrics_xgb.json",
                "predictions": "benchmark_predictions_xgb.csv",
                "feature_importance": "benchmark_feature_importance_xgb.csv",
                "model": "benchmark_model_xgb.json",
                "run_config": "benchmark_run_config_xgb.json",
            },
        },
    ]

    manifest_path = freeze_selected_models_from_results(
        results=results,
        output_root=str(tmp_path),
        split_strategy="strict_group",
    )

    manifest = pd.read_csv(manifest_path)
    row = manifest.iloc[0]
    assert row["selected_model_key"] == "rf"
    assert row["artifact_role"] == "selected_model_refit"
    assert row["training_scope"] == "train_plus_validation"
    assert "selected_models\\strict_group\\pyrolysis_direct\\product_char_yield_pct" in row["model_path"]
