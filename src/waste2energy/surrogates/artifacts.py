# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from pathlib import Path

import hashlib
import pandas as pd

from ..common import build_run_manifest, write_json
from ..config import resolve_surrogate_outputs_dir


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
    row_counts_override: dict[str, int] | None = None,
    training_splits_override: list[str] | None = None,
    additional_run_manifest_fields: dict[str, object] | None = None,
) -> dict[str, str]:
    base_outputs_dir = resolve_surrogate_outputs_dir()
    split_root = base_outputs_dir if bundle.split_strategy == "recommended" else base_outputs_dir / bundle.split_strategy

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
        dataset_version_label=_dataset_version_label(bundle),
        dataset_fingerprint=_dataset_fingerprint(bundle.frame),
        split_frame_fingerprints=_split_frame_fingerprints(bundle.split_frames),
        target_column=target_column,
        split_strategy=bundle.split_strategy,
        feature_columns=bundle.feature_columns,
        training_splits=training_splits_override or list(bundle.spec.training_splits),
        weight_column=bundle.spec.weight_column,
        row_counts=row_counts_override or {key: int(len(value)) for key, value in bundle.split_frames.items()},
        model_family=model_family,
        model_config=model_config,
        **(additional_run_manifest_fields or {}),
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
            "run_config_path": result["outputs"]["run_config"],
        }
        for split_name in ("train", "validation", "test"):
            split_metrics = result["metrics"].get(split_name, {})
            for metric_name in ("r2", "rmse", "mae"):
                row[f"{split_name}_{metric_name}"] = split_metrics.get(metric_name)
        rows.append(row)

    ranked_frame = build_ranked_suite_summary_frame(rows)
    return ranked_frame.to_dict(orient="records")


def build_ranked_suite_summary_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["selection_metric_name"] = "validation_r2"
    frame["selection_metric_value"] = pd.to_numeric(frame.get("validation_r2"), errors="coerce")
    frame["reporting_test_r2"] = pd.to_numeric(frame.get("test_r2"), errors="coerce")
    frame["reporting_test_rmse"] = pd.to_numeric(frame.get("test_rmse"), errors="coerce")
    frame["reporting_test_mae"] = pd.to_numeric(frame.get("test_mae"), errors="coerce")

    ranked_groups: list[pd.DataFrame] = []
    for _, subset in frame.groupby(["dataset_key", "target_column", "split_strategy"], dropna=False, sort=False):
        working = subset.copy()
        working["_validation_r2_sort"] = pd.to_numeric(working.get("validation_r2"), errors="coerce").fillna(float("-inf"))
        working["_validation_rmse_sort"] = pd.to_numeric(working.get("validation_rmse"), errors="coerce").fillna(float("inf"))
        working["_validation_mae_sort"] = pd.to_numeric(working.get("validation_mae"), errors="coerce").fillna(float("inf"))
        working = working.sort_values(
            ["_validation_r2_sort", "_validation_rmse_sort", "_validation_mae_sort", "model_key"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)
        working["selection_rank_within_dataset_target"] = range(1, len(working) + 1)
        working["is_selected_model"] = working["selection_rank_within_dataset_target"].eq(1)
        ranked_groups.append(
            working.drop(columns=["_validation_r2_sort", "_validation_rmse_sort", "_validation_mae_sort"])
        )

    ranked = pd.concat(ranked_groups, ignore_index=True)
    return ranked.sort_values(
        ["dataset_key", "target_column", "split_strategy", "selection_rank_within_dataset_target", "model_key"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)


def build_selected_models_manifest(summary_frame: pd.DataFrame) -> pd.DataFrame:
    if summary_frame.empty:
        return pd.DataFrame()

    selected = summary_frame[summary_frame["is_selected_model"].astype(bool)].copy()
    if selected.empty:
        return pd.DataFrame()

    selected["selection_status"] = selected["selection_metric_value"].notna().map(
        {True: "selected_on_validation", False: "selection_unavailable"}
    )
    selected["artifact_role"] = "benchmark_selected_candidate"
    selected["training_scope"] = "train_only"
    selected["selection_evidence_source"] = "benchmark_validation_metrics"
    selected = selected.rename(
        columns={
            "model_key": "selected_model_key",
            "selection_metric_name": "selection_metric_name",
            "selection_metric_value": "selection_metric_value",
            "validation_r2": "selected_validation_r2",
            "validation_rmse": "selected_validation_rmse",
            "validation_mae": "selected_validation_mae",
            "test_r2": "selected_test_r2",
            "test_rmse": "selected_test_rmse",
            "test_mae": "selected_test_mae",
        }
    )
    selected["selection_trace_id"] = (
        selected["dataset_key"].astype(str)
        + "::"
        + selected["target_column"].astype(str)
        + "::"
        + selected["split_strategy"].astype(str)
        + "::"
        + selected["selected_model_key"].astype(str)
    )
    selected["selection_data_version"] = pd.NA
    selected["selection_data_fingerprint"] = pd.NA
    selected["selection_random_state"] = pd.NA
    selected["benchmark_model_path"] = selected["model_path"]
    selected["benchmark_metrics_path"] = selected["metrics_path"]
    selected["benchmark_predictions_path"] = selected["predictions_path"]
    selected["benchmark_feature_importance_path"] = selected["feature_importance_path"]
    selected["benchmark_run_config_path"] = selected["run_config_path"]
    selected["benchmark_validation_r2"] = selected["selected_validation_r2"]
    selected["benchmark_validation_rmse"] = selected["selected_validation_rmse"]
    selected["benchmark_validation_mae"] = selected["selected_validation_mae"]
    selected["benchmark_test_r2"] = selected["selected_test_r2"]
    selected["benchmark_test_rmse"] = selected["selected_test_rmse"]
    selected["benchmark_test_mae"] = selected["selected_test_mae"]
    selected["benchmark_train_rows"] = selected["train_rows"]
    selected["benchmark_validation_rows"] = selected["validation_rows"]
    selected["benchmark_test_rows"] = selected["test_rows"]
    selected["benchmark_data_version"] = pd.NA
    selected["benchmark_data_fingerprint"] = pd.NA
    selected["benchmark_random_state"] = pd.NA
    ordered_columns = [
        "dataset_key",
        "target_column",
        "split_strategy",
        "selected_model_key",
        "selection_status",
        "artifact_role",
        "training_scope",
        "selection_trace_id",
        "selection_evidence_source",
        "selection_metric_name",
        "selection_metric_value",
        "selection_data_version",
        "selection_data_fingerprint",
        "selection_random_state",
        "selected_validation_r2",
        "selected_validation_rmse",
        "selected_validation_mae",
        "selected_test_r2",
        "selected_test_rmse",
        "selected_test_mae",
        "feature_count",
        "train_rows",
        "validation_rows",
        "test_rows",
        "model_path",
        "metrics_path",
        "predictions_path",
        "feature_importance_path",
        "run_config_path",
        "benchmark_validation_r2",
        "benchmark_validation_rmse",
        "benchmark_validation_mae",
        "benchmark_test_r2",
        "benchmark_test_rmse",
        "benchmark_test_mae",
        "benchmark_train_rows",
        "benchmark_validation_rows",
        "benchmark_test_rows",
        "benchmark_data_version",
        "benchmark_data_fingerprint",
        "benchmark_random_state",
        "benchmark_model_path",
        "benchmark_metrics_path",
        "benchmark_predictions_path",
        "benchmark_feature_importance_path",
        "benchmark_run_config_path",
    ]
    available_columns = [column for column in ordered_columns if column in selected.columns]
    return selected[available_columns].sort_values(
        ["dataset_key", "target_column", "split_strategy"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def write_suite_summary(
    *,
    results: list[dict[str, object]],
    output_root: str | None,
    split_strategy: str,
) -> tuple[str, str, str]:
    summary_frame = pd.DataFrame(build_suite_summary_rows(results))
    selected_manifest = build_selected_models_manifest(summary_frame)
    suite_dir = Path(output_root) if output_root else resolve_surrogate_outputs_dir()
    suite_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if split_strategy == "recommended" else f"_{split_strategy}"
    summary_csv = suite_dir / f"traditional_ml_suite_summary{suffix}.csv"
    summary_json = suite_dir / f"traditional_ml_suite_summary{suffix}.json"
    selected_manifest_csv = suite_dir / f"selected_models_manifest_benchmark{suffix}.csv"
    if not selected_manifest.empty:
        selected_manifest = selected_manifest.copy()
        selected_manifest["selection_benchmark_manifest_path"] = str(selected_manifest_csv)
    summary_frame.to_csv(summary_csv, index=False)
    selected_manifest.to_csv(selected_manifest_csv, index=False)
    write_json(summary_json, results)
    return str(summary_csv), str(summary_json), str(selected_manifest_csv)


def _dataset_version_label(bundle) -> str:
    return f"{bundle.spec.key}:{bundle.spec.file_name}"


def _dataset_fingerprint(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "empty_frame"
    normalized = frame.sort_values("sample_id").reset_index(drop=True)
    csv_payload = normalized.to_csv(index=False)
    return hashlib.sha256(csv_payload.encode("utf-8")).hexdigest()


def _split_frame_fingerprints(split_frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    return {
        split_name: _dataset_fingerprint(split_frame)
        for split_name, split_frame in split_frames.items()
    }
