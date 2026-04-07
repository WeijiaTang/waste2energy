# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from pathlib import Path

import pandas as pd

from waste2energy.planning.inputs import load_planning_input_bundle
from waste2energy.planning.surrogate_evaluator import SURROGATE_TARGETS, SurrogateEvaluator, build_surrogate_predictions


def test_surrogate_evaluator_outputs_predictions_and_fallbacks():
    bundle = load_planning_input_bundle()
    subset = (
        bundle.frame[bundle.frame["pathway"].isin(["htc", "pyrolysis", "baseline", "ad"])]
        .head(12)
        .reset_index(drop=True)
    )
    predictions = build_surrogate_predictions(subset)

    assert len(predictions) == len(subset)
    assert "combined_uncertainty_ratio" in predictions.columns
    assert predictions["combined_uncertainty_ratio"].ge(0.0).all()
    assert predictions["predicted_product_char_yield_pct"].notna().all()

    merged = subset[["optimization_case_id", "pathway"]].merge(
        predictions[["optimization_case_id", "pathway", "surrogate_mode"]],
        on=["optimization_case_id", "pathway"],
        how="left",
    )
    fallback_rows = merged[merged["pathway"].isin(["baseline", "ad"])]
    assert not fallback_rows.empty
    assert (fallback_rows["surrogate_mode"] == "documented_static_fallback").all()


def test_surrogate_evaluator_uses_documented_fallback_when_required_feature_is_missing(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "optimization_case_id": "case-1",
                "pathway": "pyrolysis",
                "feedstock_hhv_mj_per_kg": pd.NA,
                "feedstock_moisture_pct": 72.0,
                "product_char_yield_pct": 30.0,
                "product_char_hhv_mj_per_kg": 20.0,
                "energy_recovery_pct": 55.0,
                "carbon_retention_pct": 40.0,
                "source_dataset_kind": "planning_candidate",
                "sample_id": "planning::001",
            }
        ]
    )

    evaluator = SurrogateEvaluator()

    artifact = type(
        "Artifact",
        (),
        {
            "model_key": "rf",
            "dataset_key": "pyrolysis_direct",
            "split_strategy": "strict_group",
            "feature_columns": ("feedstock_hhv_mj_per_kg", "feedstock_moisture_pct"),
            "model_path": None,
            "metrics_path": None,
        },
    )()

    class DummyModel:
        def predict(self, feature_frame):
            raise AssertionError("trained model should not be called when required features are missing")

    monkeypatch.setattr(evaluator, "_resolve_artifact", lambda **kwargs: artifact)
    monkeypatch.setattr(evaluator, "_load_model", lambda _artifact: DummyModel())
    monkeypatch.setattr(evaluator, "_estimate_prediction_std", lambda _artifact: 1.0)

    predictions = evaluator.evaluate(frame)

    assert predictions.loc[0, "surrogate_feature_imputation_flag"]
    assert predictions.loc[0, "surrogate_prediction_status"] == "documented_fallback_missing_required_feature"
    assert "feedstock_hhv_mj_per_kg" in predictions.loc[0, "surrogate_missing_feature_columns"]
    assert "missing_required_feature" in predictions.loc[0, "surrogate_fallback_reason"]
    for target in SURROGATE_TARGETS:
        assert "missing_required_feature" in predictions.loc[0, f"{target}_prediction_source"]
    assert predictions.loc[0, "predicted_product_char_yield_pct"] == 30.0


def test_surrogate_evaluator_missing_target_value_keeps_uncertainty_fields_missing(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "optimization_case_id": "case-1",
                "pathway": "pyrolysis",
                "feedstock_hhv_mj_per_kg": pd.NA,
                "feedstock_moisture_pct": 72.0,
                "product_char_yield_pct": pd.NA,
                "product_char_hhv_mj_per_kg": 20.0,
                "energy_recovery_pct": 55.0,
                "carbon_retention_pct": 40.0,
                "source_dataset_kind": "planning_candidate",
                "sample_id": "planning::001",
            }
        ]
    )

    evaluator = SurrogateEvaluator()

    artifact = type(
        "Artifact",
        (),
        {
            "model_key": "rf",
            "dataset_key": "pyrolysis_direct",
            "split_strategy": "strict_group",
            "feature_columns": ("feedstock_hhv_mj_per_kg", "feedstock_moisture_pct"),
            "model_path": None,
            "metrics_path": None,
        },
    )()

    monkeypatch.setattr(evaluator, "_resolve_artifact", lambda **kwargs: artifact)

    predictions = evaluator.evaluate(frame)

    assert predictions.loc[0, "surrogate_prediction_status"] == "documented_fallback_missing_target_value"
    assert pd.isna(predictions.loc[0, "product_char_yield_pct_prediction_std"])
    assert pd.isna(predictions.loc[0, "product_char_yield_pct_ci_lower"])
    assert pd.isna(predictions.loc[0, "product_char_yield_pct_ci_upper"])


def test_surrogate_evaluator_prefers_selected_manifest_over_raw_test_ranking(tmp_path, monkeypatch):
    outputs_root = Path(tmp_path)
    manifest = pd.DataFrame(
        [
            {
                "dataset_key": "pyrolysis_direct",
                "target_column": "product_char_yield_pct",
                "split_strategy": "strict_group",
                "selected_model_key": "rf",
                "selection_status": "selected_on_validation",
                "selection_metric_name": "validation_r2",
                "selection_metric_value": 0.8,
                "selected_validation_r2": 0.8,
                "selected_validation_rmse": 2.0,
                "selected_validation_mae": 1.0,
                "selected_test_r2": 0.5,
                "selected_test_rmse": 3.0,
                "selected_test_mae": 2.0,
                "model_path": "dummy",
                "metrics_path": "dummy",
                "predictions_path": "dummy",
                "feature_importance_path": "dummy",
                "run_config_path": "dummy",
            }
        ]
    )
    manifest.to_csv(outputs_root / "selected_models_manifest_strict_group.csv", index=False)

    legacy = pd.DataFrame(
        [
            {
                "dataset_key": "pyrolysis_direct",
                "target_column": "product_char_yield_pct",
                "split_strategy": "strict_group",
                "model_key": "xgboost",
                "test_r2": 0.95,
                "validation_r2": 0.1,
            }
        ]
    )
    legacy.to_csv(outputs_root / "traditional_ml_suite_summary_strict_group.csv", index=False)

    evaluator = SurrogateEvaluator(outputs_root=outputs_root)

    monkeypatch.setattr(
        evaluator,
        "_build_artifact_from_selected_manifest",
        lambda row: row["selected_model_key"],
    )
    monkeypatch.setattr(
        evaluator,
        "_build_artifact_from_summary",
        lambda row: row["model_key"],
    )

    selected = evaluator._resolve_artifact(pathway="pyrolysis", target_column="product_char_yield_pct")

    assert selected == "rf"


def test_surrogate_evaluator_reads_refit_artifact_paths_from_selected_manifest(tmp_path):
    outputs_root = Path(tmp_path)
    artifact_dir = outputs_root / "selected_models" / "strict_group" / "pyrolysis_direct" / "product_char_yield_pct"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = artifact_dir / "run_config.json"
    run_config_path.write_text(
        '{"feature_columns": ["feedstock_hhv_mj_per_kg", "feedstock_moisture_pct"]}',
        encoding="utf-8",
    )
    model_path = artifact_dir / "model.joblib"
    model_path.write_text("placeholder", encoding="utf-8")
    metrics_path = artifact_dir / "metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")

    manifest = pd.DataFrame(
        [
            {
                "dataset_key": "pyrolysis_direct",
                "target_column": "product_char_yield_pct",
                "split_strategy": "strict_group",
                "selected_model_key": "rf",
                "selection_status": "selected_on_validation_refit_train_plus_validation",
                "artifact_role": "selected_model_refit",
                "training_scope": "train_plus_validation",
                "selection_metric_name": "validation_r2",
                "selection_metric_value": 0.8,
                "selected_validation_r2": 0.8,
                "selected_validation_rmse": 2.0,
                "selected_validation_mae": 1.0,
                "selected_test_r2": 0.5,
                "selected_test_rmse": 3.0,
                "selected_test_mae": 2.0,
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
                "predictions_path": str(artifact_dir / "predictions.csv"),
                "feature_importance_path": str(artifact_dir / "feature_importance.csv"),
                "run_config_path": str(run_config_path),
            }
        ]
    )
    manifest.to_csv(outputs_root / "selected_models_manifest_strict_group.csv", index=False)

    evaluator = SurrogateEvaluator(outputs_root=outputs_root)
    artifact = evaluator._resolve_artifact(pathway="pyrolysis", target_column="product_char_yield_pct")

    assert artifact is not None
    assert artifact.model_key == "rf"
    assert artifact.model_path == model_path
    assert artifact.run_config_path == run_config_path
    assert artifact.feature_columns == ("feedstock_hhv_mj_per_kg", "feedstock_moisture_pct")
