from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common import build_run_manifest, write_json
from ..config import OUTPUTS_DIR


def write_training_outputs(
    *,
    model_key: str,
    dataset_key: str,
    target_column: str,
    output_dir: str | None,
    model,
    feature_importance: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics_payload: dict[str, dict[str, float]],
    bundle,
    model_config,
    model_family: str,
    model_file_name: str,
    save_model_fn,
) -> dict[str, str]:
    split_root = OUTPUTS_DIR if bundle.split_strategy == "recommended" else OUTPUTS_DIR / bundle.split_strategy

    if output_dir:
        target_dir = Path(output_dir)
    elif model_key == "xgboost":
        target_dir = split_root / dataset_key / target_column
    else:
        target_dir = split_root / model_key / dataset_key / target_column
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / model_file_name
    metrics_path = target_dir / "metrics.json"
    predictions_path = target_dir / "predictions.csv"
    importance_path = target_dir / "feature_importance.csv"
    config_path = target_dir / "run_config.json"

    save_model_fn(model, model_path)
    predictions.to_csv(predictions_path, index=False)
    feature_importance.to_csv(importance_path, index=False)
    write_json(metrics_path, metrics_payload)

    run_config = build_run_manifest(
        model_key=model_key,
        dataset_key=dataset_key,
        dataset_file=bundle.spec.file_name,
        dataset_description=bundle.spec.description,
        target_column=target_column,
        split_strategy=bundle.split_strategy,
        feature_columns=bundle.feature_columns,
        training_splits=list(bundle.spec.training_splits),
        weight_column=bundle.spec.weight_column,
        row_counts={key: int(len(value)) for key, value in bundle.split_frames.items()},
        model_family=model_family,
        model_config=model_config,
    )
    write_json(config_path, run_config)

    return {
        "model": str(model_path),
        "metrics": str(metrics_path),
        "predictions": str(predictions_path),
        "feature_importance": str(importance_path),
        "run_config": str(config_path),
    }


def build_suite_summary_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        row = {
            "model_key": result["model_key"],
            "dataset_key": result["dataset_key"],
            "target_column": result["target_column"],
            "split_strategy": result["split_strategy"],
            "feature_count": result["feature_count"],
            "train_rows": result["row_counts"].get("train", 0),
            "validation_rows": result["row_counts"].get("validation", 0),
            "test_rows": result["row_counts"].get("test", 0),
            "metrics_path": result["outputs"]["metrics"],
            "predictions_path": result["outputs"]["predictions"],
            "feature_importance_path": result["outputs"]["feature_importance"],
            "model_path": result["outputs"]["model"],
        }
        for split_name in ("train", "validation", "test"):
            split_metrics = result["metrics"].get(split_name, {})
            for metric_name in ("r2", "rmse", "mae"):
                row[f"{split_name}_{metric_name}"] = split_metrics.get(metric_name)
        rows.append(row)

    rows.sort(
        key=lambda item: (
            float("-inf") if item.get("test_r2") is None else float(item["test_r2"]),
            item["model_key"],
            item["dataset_key"],
            item["target_column"],
        ),
        reverse=True,
    )
    return rows


def write_suite_summary(
    *,
    results: list[dict[str, object]],
    output_root: str | None,
    split_strategy: str,
) -> tuple[str, str]:
    summary_frame = pd.DataFrame(build_suite_summary_rows(results))
    suite_dir = Path(output_root) if output_root else OUTPUTS_DIR
    suite_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if split_strategy == "recommended" else f"_{split_strategy}"
    summary_csv = suite_dir / f"traditional_ml_suite_summary{suffix}.csv"
    summary_json = suite_dir / f"traditional_ml_suite_summary{suffix}.json"
    summary_frame.to_csv(summary_csv, index=False)
    write_json(summary_json, results)
    return str(summary_csv), str(summary_json)
