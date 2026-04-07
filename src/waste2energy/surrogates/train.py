from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data import DATASET_KEYS, TARGET_COLUMNS, frame_to_xy, load_dataset_bundle
from ..models import MODEL_KEYS, get_model_ops
from ..models.xgboost_regressor import XGBoostConfig
from .artifacts import (
    build_ranked_suite_summary_frame,
    build_selected_models_manifest,
    write_suite_summary,
    write_training_outputs,
)
from .evaluate import regression_metrics


def run_xgboost_baseline(
    dataset_key: str,
    target_column: str,
    output_dir: str | None = None,
    model_config: XGBoostConfig | None = None,
    split_strategy: str = "recommended",
) -> dict[str, object]:
    return run_regression_baseline(
        model_key="xgboost",
        dataset_key=dataset_key,
        target_column=target_column,
        output_dir=output_dir,
        model_config=model_config,
        split_strategy=split_strategy,
    )


def run_regression_baseline(
    model_key: str,
    dataset_key: str,
    target_column: str,
    output_dir: str | None = None,
    model_config=None,
    split_strategy: str = "recommended",
) -> dict[str, object]:
    if target_column not in TARGET_COLUMNS:
        allowed = ", ".join(TARGET_COLUMNS)
        raise ValueError(f"Unsupported target '{target_column}'. Choose from: {allowed}")

    bundle = load_dataset_bundle(dataset_key, split_strategy=split_strategy)
    model_ops = get_model_ops(model_key)
    active_config = model_config or model_ops["default_config"]

    train_frame = bundle.split_frames["train"]
    x_train, y_train, train_weight = frame_to_xy(
        train_frame,
        bundle.feature_columns,
        target_column,
        bundle.spec.weight_column,
    )
    model = model_ops["train_model"](
        x_train,
        y_train,
        sample_weight=train_weight,
        config=active_config,
    )

    metrics_payload: dict[str, dict[str, float]] = {}
    prediction_frames: list[pd.DataFrame] = []
    for split_name, split_frame in bundle.split_frames.items():
        if split_frame.empty:
            continue
        x_split, y_split, _ = frame_to_xy(split_frame, bundle.feature_columns, target_column)
        predictions = model.predict(x_split)
        metrics_payload[split_name] = regression_metrics(y_split, predictions)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "sample_id": split_frame["sample_id"],
                    "split": split_name,
                    "model_key": model_key,
                    "dataset_key": dataset_key,
                    "target_column": target_column,
                    "split_strategy": split_strategy,
                    "y_true": y_split,
                    "y_pred": predictions,
                }
            )
        )

    feature_importance = model_ops["build_feature_importance"](model, bundle.feature_columns)
    outputs = write_training_outputs(
        model_key=model_key,
        dataset_key=dataset_key,
        target_column=target_column,
        output_dir=output_dir,
        model=model,
        feature_importance=feature_importance,
        predictions=pd.concat(prediction_frames, ignore_index=True),
        metrics_payload=metrics_payload,
        bundle=bundle,
        model_config=active_config,
        model_family=model_ops["model_family"],
        model_file_name=model_ops["model_file_name"],
        save_model_fn=model_ops["save_model"],
    )
    return {
        "model_key": model_key,
        "dataset_key": dataset_key,
        "target_column": target_column,
        "split_strategy": split_strategy,
        "metrics": metrics_payload,
        "feature_count": len(bundle.feature_columns),
        "row_counts": {key: int(len(value)) for key, value in bundle.split_frames.items()},
        "outputs": outputs,
    }


def run_xgboost_baseline_suite(
    dataset_keys: list[str] | None = None,
    target_columns: list[str] | None = None,
    output_root: str | None = None,
    split_strategy: str = "recommended",
) -> dict[str, object]:
    return run_regression_baseline_suite(
        model_keys=["xgboost"],
        dataset_keys=dataset_keys,
        target_columns=target_columns,
        output_root=output_root,
        split_strategy=split_strategy,
    )


def run_regression_baseline_suite(
    model_keys: list[str] | None = None,
    dataset_keys: list[str] | None = None,
    target_columns: list[str] | None = None,
    output_root: str | None = None,
    split_strategy: str = "recommended",
) -> dict[str, object]:
    selected_models = model_keys or list(MODEL_KEYS)
    selected_datasets = dataset_keys or list(DATASET_KEYS)
    selected_targets = target_columns or list(TARGET_COLUMNS)
    results: list[dict[str, object]] = []

    for model_key in selected_models:
        for dataset_key in selected_datasets:
            for target_column in selected_targets:
                run_output_dir = None
                if output_root:
                    run_output_dir = str(Path(output_root) / model_key / dataset_key / target_column)
                result = run_regression_baseline(
                    model_key=model_key,
                    dataset_key=dataset_key,
                    target_column=target_column,
                    output_dir=run_output_dir,
                    split_strategy=split_strategy,
                )
                results.append(result)

    summary_csv, summary_json, selected_manifest_csv = write_suite_summary(
        results=results,
        output_root=output_root,
        split_strategy=split_strategy,
    )
    final_selected_manifest_csv = freeze_selected_models_from_results(
        results=results,
        output_root=output_root,
        split_strategy=split_strategy,
    )

    return {
        "run_count": len(results),
        "models": selected_models,
        "datasets": selected_datasets,
        "targets": selected_targets,
        "split_strategy": split_strategy,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "selected_manifest_csv": selected_manifest_csv,
        "final_selected_manifest_csv": final_selected_manifest_csv,
        "results": results,
    }


def freeze_selected_models_from_results(
    *,
    results: list[dict[str, object]],
    output_root: str | None,
    split_strategy: str,
) -> str:
    summary_frame = pd.DataFrame(results)
    if summary_frame.empty:
        return ""

    ranked_rows = pd.DataFrame(
        [
            {
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
                **{
                    f"{split_name}_{metric_name}": result["metrics"].get(split_name, {}).get(metric_name)
                    for split_name in ("train", "validation", "test")
                    for metric_name in ("r2", "rmse", "mae")
                },
            }
            for result in results
        ]
    )
    selected_manifest = build_selected_models_manifest(build_ranked_suite_summary_frame(ranked_rows.to_dict("records")))
    if selected_manifest.empty:
        return ""

    base_outputs_root = Path(output_root) if output_root else _resolve_output_root_for_split(split_strategy)
    selected_models_root = base_outputs_root / "selected_models" / split_strategy
    selected_models_root.mkdir(parents=True, exist_ok=True)

    frozen_rows: list[dict[str, object]] = []
    for _, selected_row in selected_manifest.iterrows():
        frozen_rows.append(
            _freeze_selected_model(
                selected_row=selected_row,
                selected_models_root=selected_models_root,
            )
        )

    final_manifest = pd.DataFrame(frozen_rows).sort_values(
        ["dataset_key", "target_column", "split_strategy"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    suffix = "" if split_strategy == "recommended" else f"_{split_strategy}"
    final_manifest_path = base_outputs_root / f"selected_models_manifest{suffix}.csv"
    final_manifest.to_csv(final_manifest_path, index=False)
    return str(final_manifest_path)


def _freeze_selected_model(
    *,
    selected_row: pd.Series,
    selected_models_root: Path,
) -> dict[str, object]:
    dataset_key = str(selected_row["dataset_key"])
    target_column = str(selected_row["target_column"])
    model_key = str(selected_row["selected_model_key"])
    split_strategy = str(selected_row["split_strategy"])
    bundle = load_dataset_bundle(dataset_key, split_strategy=split_strategy)
    model_ops = get_model_ops(model_key)

    refit_frame = pd.concat(
        [
            bundle.split_frames.get("train", pd.DataFrame()),
            bundle.split_frames.get("validation", pd.DataFrame()),
        ],
        ignore_index=True,
    )
    if refit_frame.empty:
        refit_frame = bundle.split_frames.get("train", pd.DataFrame()).copy()

    x_refit, y_refit, refit_weight = frame_to_xy(
        refit_frame,
        bundle.feature_columns,
        target_column,
        bundle.spec.weight_column,
    )
    model = model_ops["train_model"](
        x_refit,
        y_refit,
        sample_weight=refit_weight,
        config=model_ops["default_config"],
    )

    predictions_payload, metrics_payload = _build_refit_reporting_payload(
        model=model,
        bundle=bundle,
        model_key=model_key,
        dataset_key=dataset_key,
        target_column=target_column,
        split_strategy=split_strategy,
    )
    feature_importance = model_ops["build_feature_importance"](model, bundle.feature_columns)
    target_dir = selected_models_root / dataset_key / target_column
    outputs = write_training_outputs(
        model_key=model_key,
        dataset_key=dataset_key,
        target_column=target_column,
        output_dir=str(target_dir),
        model=model,
        feature_importance=feature_importance,
        predictions=predictions_payload,
        metrics_payload=metrics_payload,
        bundle=bundle,
        model_config=model_ops["default_config"],
        model_family=model_ops["model_family"],
        model_file_name=model_ops["model_file_name"],
        save_model_fn=model_ops["save_model"],
        row_counts_override={
            "refit_train": int(len(refit_frame)),
            "benchmark_train": int(len(bundle.split_frames.get("train", pd.DataFrame()))),
            "benchmark_validation": int(len(bundle.split_frames.get("validation", pd.DataFrame()))),
            "benchmark_test": int(len(bundle.split_frames.get("test", pd.DataFrame()))),
        },
        training_splits_override=["train", "validation"],
        additional_run_manifest_fields={
            "artifact_role": "selected_model_refit",
            "training_scope": "train_plus_validation",
            "selection_metric_name": selected_row.get("selection_metric_name", "validation_r2"),
            "selection_metric_value": selected_row.get("selection_metric_value"),
            "selected_model_key": model_key,
            "benchmark_model_path": selected_row.get("benchmark_model_path") or selected_row.get("model_path"),
            "benchmark_metrics_path": selected_row.get("benchmark_metrics_path") or selected_row.get("metrics_path"),
            "benchmark_predictions_path": selected_row.get("benchmark_predictions_path") or selected_row.get("predictions_path"),
            "benchmark_feature_importance_path": selected_row.get("benchmark_feature_importance_path") or selected_row.get("feature_importance_path"),
            "benchmark_run_config_path": selected_row.get("benchmark_run_config_path") or selected_row.get("run_config_path"),
        },
    )

    frozen = selected_row.to_dict()
    frozen.update(
        {
            "artifact_role": "selected_model_refit",
            "training_scope": "train_plus_validation",
            "selection_status": "selected_on_validation_refit_train_plus_validation",
            "model_path": outputs["model"],
            "metrics_path": outputs["metrics"],
            "predictions_path": outputs["predictions"],
            "feature_importance_path": outputs["feature_importance"],
            "run_config_path": outputs["run_config"],
            "refit_train_rows": int(len(refit_frame)),
            "refit_validation_rows": 0,
            "refit_test_rows": int(len(bundle.split_frames.get("test", pd.DataFrame()))),
        }
    )
    if "selected_test_r2" in metrics_payload.get("test", {}):
        frozen["selected_test_r2"] = metrics_payload["test"]["r2"]
    else:
        frozen["selected_test_r2"] = metrics_payload.get("test", {}).get("r2")
    frozen["selected_test_rmse"] = metrics_payload.get("test", {}).get("rmse")
    frozen["selected_test_mae"] = metrics_payload.get("test", {}).get("mae")
    frozen["refit_test_r2"] = metrics_payload.get("test", {}).get("r2")
    frozen["refit_test_rmse"] = metrics_payload.get("test", {}).get("rmse")
    frozen["refit_test_mae"] = metrics_payload.get("test", {}).get("mae")
    return frozen


def _build_refit_reporting_payload(
    *,
    model,
    bundle,
    model_key: str,
    dataset_key: str,
    target_column: str,
    split_strategy: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    prediction_frames: list[pd.DataFrame] = []
    metrics_payload: dict[str, dict[str, float]] = {}
    refit_frame = pd.concat(
        [
            bundle.split_frames.get("train", pd.DataFrame()),
            bundle.split_frames.get("validation", pd.DataFrame()),
        ],
        ignore_index=True,
    )
    reporting_splits = {
        "refit_train": refit_frame,
        "test": bundle.split_frames.get("test", pd.DataFrame()),
    }
    for split_name, split_frame in reporting_splits.items():
        if split_frame.empty:
            continue
        x_split, y_split, _ = frame_to_xy(split_frame, bundle.feature_columns, target_column)
        predictions = model.predict(x_split)
        metrics_payload[split_name] = regression_metrics(y_split, predictions)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "sample_id": split_frame["sample_id"],
                    "split": split_name,
                    "model_key": model_key,
                    "dataset_key": dataset_key,
                    "target_column": target_column,
                    "split_strategy": split_strategy,
                    "y_true": y_split,
                    "y_pred": predictions,
                }
            )
        )
    return pd.concat(prediction_frames, ignore_index=True), metrics_payload


def _resolve_output_root_for_split(split_strategy: str) -> Path:
    from ..config import resolve_surrogate_outputs_dir

    return resolve_surrogate_outputs_dir()
