from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from waste2energy.planning.surrogate_evaluator import SurrogateEvaluator


def test_htc_resolution_prioritizes_catboost_from_leave_study_out_benchmark(monkeypatch, tmp_path):
    outputs_root = tmp_path / "surrogates"
    benchmark_root = tmp_path / "benchmark" / "htc_model_compare_lso"
    outputs_root.mkdir(parents=True)
    benchmark_root.mkdir(parents=True)
    _write_summary_row(
        benchmark_root,
        model_key="catboost",
        target_column="product_char_hhv_mj_per_kg",
        validation_r2=0.48,
        test_r2=0.28,
    )
    _write_summary_row(
        benchmark_root,
        model_key="lightgbm",
        target_column="product_char_hhv_mj_per_kg",
        validation_r2=0.53,
        test_r2=0.21,
    )

    monkeypatch.setattr("waste2energy.planning.surrogate_evaluator.BENCHMARK_OUTPUTS_DIR", tmp_path / "benchmark")
    monkeypatch.setattr(SurrogateEvaluator, "_model_runtime_available", lambda self, model_key: True)

    evaluator = SurrogateEvaluator(outputs_root=outputs_root)
    artifact = evaluator._resolve_artifact(pathway="htc", target_column="product_char_hhv_mj_per_kg")

    assert artifact is not None
    assert artifact.model_key == "catboost"
    assert "htc_model_compare_lso" in str(artifact.model_path)


def test_htc_resolution_falls_back_to_available_stacking_when_boosters_are_unavailable(monkeypatch, tmp_path):
    outputs_root = tmp_path / "surrogates"
    benchmark_root = tmp_path / "benchmark" / "htc_model_compare_lso"
    outputs_root.mkdir(parents=True)
    benchmark_root.mkdir(parents=True)
    _write_summary_row(
        benchmark_root,
        model_key="catboost",
        target_column="product_char_yield_pct",
        validation_r2=0.44,
        test_r2=0.41,
    )
    _write_summary_row(
        benchmark_root,
        model_key="lightgbm",
        target_column="product_char_yield_pct",
        validation_r2=0.39,
        test_r2=0.19,
    )
    _write_summary_row(
        benchmark_root,
        model_key="stacking",
        target_column="product_char_yield_pct",
        validation_r2=0.17,
        test_r2=0.16,
        model_suffix="joblib",
    )
    _write_summary_row(
        outputs_root / "strict_group",
        model_key="extra_trees",
        target_column="product_char_yield_pct",
        validation_r2=0.81,
        test_r2=0.79,
        model_suffix="joblib",
    )
    pd.DataFrame(
        [
            {
                "dataset_key": "paper1_htc_scope",
                "target_column": "product_char_yield_pct",
                "model_key": "extra_trees",
                "validation_r2": 0.81,
                "test_r2": 0.79,
                "split_strategy": "strict_group",
            }
        ]
    ).to_csv(outputs_root / "traditional_ml_suite_summary_strict_group.csv", index=False)

    monkeypatch.setattr("waste2energy.planning.surrogate_evaluator.BENCHMARK_OUTPUTS_DIR", tmp_path / "benchmark")
    monkeypatch.setattr(
        SurrogateEvaluator,
        "_model_runtime_available",
        lambda self, model_key: model_key in {"stacking", "extra_trees"},
    )

    evaluator = SurrogateEvaluator(outputs_root=outputs_root)
    artifact = evaluator._resolve_artifact(pathway="htc", target_column="product_char_yield_pct")

    assert artifact is not None
    assert artifact.model_key == "stacking"
    assert "htc_model_compare_lso" in str(artifact.model_path)


def _write_summary_row(
    root: Path,
    *,
    model_key: str,
    target_column: str,
    validation_r2: float,
    test_r2: float,
    model_suffix: str | None = None,
) -> None:
    dataset_key = "htc_direct"
    artifact_dir = root / model_key / dataset_key / target_column
    artifact_dir.mkdir(parents=True, exist_ok=True)
    suffix = model_suffix or ("cbm" if model_key == "catboost" else "txt" if model_key == "lightgbm" else "joblib")
    (artifact_dir / f"model.{suffix}").write_text("stub", encoding="utf-8")
    (artifact_dir / "run_config.json").write_text(json.dumps({"feature_columns": ["feedstock_carbon_pct"]}), encoding="utf-8")
    (artifact_dir / "metrics.json").write_text(json.dumps({"test": {"r2": test_r2}}), encoding="utf-8")
    summary_path = root / "traditional_ml_suite_summary_leave_study_out.csv"
    row = pd.DataFrame(
        [
            {
                "dataset_key": dataset_key,
                "target_column": target_column,
                "model_key": model_key,
                "validation_r2": validation_r2,
                "test_r2": test_r2,
                "split_strategy": "leave_study_out",
            }
        ]
    )
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(summary_path, index=False)
