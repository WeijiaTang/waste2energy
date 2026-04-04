from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import MODEL_READY_DIR
from .specs import DEFAULT_EXCLUDED_COLUMNS, TARGET_COLUMNS, DatasetSpec, get_dataset_spec
from .splits import (
    NON_FEATURE_METADATA_COLUMNS,
    attach_split_metadata,
    build_leave_group_out_splits,
    build_recommended_splits,
    build_strict_group_splits,
)


@dataclass(frozen=True)
class DatasetBundle:
    spec: DatasetSpec
    frame: pd.DataFrame
    feature_columns: list[str]
    split_frames: dict[str, pd.DataFrame]
    split_strategy: str


def load_dataset_bundle(dataset_key: str, split_strategy: str = "recommended") -> DatasetBundle:
    spec = get_dataset_spec(dataset_key)
    path = MODEL_READY_DIR / spec.file_name
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    frame = pd.read_csv(path)
    validate_required_columns(frame, spec)
    frame = attach_split_metadata(frame, spec, split_strategy)

    feature_columns = infer_feature_columns(frame)
    numeric_columns = feature_columns + TARGET_COLUMNS
    if spec.weight_column:
        numeric_columns.append(spec.weight_column)

    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    validate_numeric_completeness(frame, feature_columns, spec)

    if split_strategy == "recommended":
        split_frames = build_recommended_splits(frame, spec)
    elif split_strategy == "strict_group":
        split_frames = build_strict_group_splits(frame, feature_columns, spec)
    elif split_strategy == "leave_source_repo_out":
        split_frames = build_leave_group_out_splits(frame, spec, group_column="source_repo")
    elif split_strategy == "leave_study_out":
        split_frames = build_leave_group_out_splits(frame, spec, group_column="study_group")
    else:
        raise ValueError(
            "Unsupported split_strategy "
            f"'{split_strategy}'. Choose from: recommended, strict_group, "
            "leave_source_repo_out, leave_study_out"
        )

    if split_frames["train"].empty:
        raise ValueError(f"Training split is empty for dataset '{dataset_key}'.")

    return DatasetBundle(
        spec=spec,
        frame=frame,
        feature_columns=feature_columns,
        split_frames=split_frames,
        split_strategy=split_strategy,
    )


def frame_to_xy(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    weight_column: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    if target_column not in TARGET_COLUMNS:
        allowed = ", ".join(TARGET_COLUMNS)
        raise ValueError(f"Unsupported target '{target_column}'. Choose from: {allowed}")

    x_frame = frame[feature_columns].copy()
    y_series = frame[target_column].copy()
    sample_weight = frame[weight_column].copy() if weight_column else None
    return x_frame, y_series, sample_weight


def validate_required_columns(frame: pd.DataFrame, spec: DatasetSpec) -> None:
    required = {"sample_id", "recommended_split", *TARGET_COLUMNS}
    if spec.weight_column:
        required.add(spec.weight_column)

    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Dataset '{spec.key}' is missing required columns: {', '.join(sorted(missing))}"
        )


def infer_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = set(DEFAULT_EXCLUDED_COLUMNS + TARGET_COLUMNS + NON_FEATURE_METADATA_COLUMNS)
    feature_columns = [column for column in frame.columns if column not in excluded]
    if not feature_columns:
        raise ValueError("No feature columns were inferred from the matrix dataset.")
    return feature_columns


def validate_numeric_completeness(
    frame: pd.DataFrame,
    feature_columns: list[str],
    spec: DatasetSpec,
) -> None:
    non_numeric = [
        column for column in feature_columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(
            f"Dataset '{spec.key}' contains non-numeric feature columns: {', '.join(non_numeric)}"
        )

    check_columns = feature_columns + TARGET_COLUMNS
    if spec.weight_column:
        check_columns.append(spec.weight_column)

    missing_counts = frame[check_columns].isna().sum()
    failing = missing_counts[missing_counts > 0]
    if not failing.empty:
        details = ", ".join(f"{column}={count}" for column, count in failing.items())
        raise ValueError(f"Dataset '{spec.key}' still contains missing numeric values: {details}")
