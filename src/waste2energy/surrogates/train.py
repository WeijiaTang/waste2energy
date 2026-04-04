from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data import DATASET_KEYS, TARGET_COLUMNS, frame_to_xy, load_dataset_bundle
from ..models import MODEL_KEYS, get_model_ops
from ..models.xgboost_regressor import XGBoostConfig
from .artifacts import write_suite_summary, write_training_outputs
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

    summary_csv, summary_json = write_suite_summary(
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
        "results": results,
    }
