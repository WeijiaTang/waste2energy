# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

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
