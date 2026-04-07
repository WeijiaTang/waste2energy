# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import pandas as pd

from waste2energy.planning.inputs import load_planning_input_bundle
from waste2energy.planning.surrogate_evaluator import build_surrogate_predictions


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

