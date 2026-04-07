from __future__ import annotations

import hashlib

import pandas as pd

from ..config import MODEL_READY_DIR
from .specs import DatasetSpec


NON_FEATURE_METADATA_COLUMNS = [
    "source_repo",
    "source_file",
    "source_dataset_kind",
    "reference_label",
    "study_group",
]


def build_recommended_splits(
    frame: pd.DataFrame,
    spec: DatasetSpec,
) -> dict[str, pd.DataFrame]:
    return {
        "train": frame[frame["recommended_split"].isin(spec.training_splits)].copy(),
        "validation": frame[frame["recommended_split"] == "validation"].copy(),
        "test": frame[frame["recommended_split"] == "test"].copy(),
    }


def build_strict_group_splits(
    frame: pd.DataFrame,
    feature_columns: list[str],
    spec: DatasetSpec,
) -> dict[str, pd.DataFrame]:
    observed_mask = frame["recommended_split"] != "augmentation"
    observed_frame = frame.loc[observed_mask].copy()
    augmentation_frame = frame.loc[~observed_mask].copy()

    observed_frame["_feature_group_id"] = feature_group_signature(observed_frame, feature_columns)
    observed_frame["_strict_split"] = observed_frame["_feature_group_id"].map(stable_group_split)

    split_frames = {
        "train": observed_frame[observed_frame["_strict_split"] == "train"].drop(
            columns=["_feature_group_id", "_strict_split"]
        ),
        "validation": observed_frame[observed_frame["_strict_split"] == "validation"].drop(
            columns=["_feature_group_id", "_strict_split"]
        ),
        "test": observed_frame[observed_frame["_strict_split"] == "test"].drop(
            columns=["_feature_group_id", "_strict_split"]
        ),
    }

    if (
        "augmentation" in spec.training_splits
        and "strict_group" in spec.augmentation_training_splits
        and not augmentation_frame.empty
    ):
        split_frames["train"] = pd.concat(
            [split_frames["train"], augmentation_frame],
            ignore_index=True,
        )

    return split_frames


def build_leave_group_out_splits(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    group_column: str,
) -> dict[str, pd.DataFrame]:
    observed_mask = frame["recommended_split"] != "augmentation"
    observed_frame = frame.loc[observed_mask].copy()
    augmentation_frame = frame.loc[~observed_mask].copy()

    if group_column not in observed_frame.columns:
        raise ValueError(
            f"Split strategy requires metadata column '{group_column}', but it is unavailable "
            f"for dataset '{spec.key}'."
        )

    group_values = observed_frame[group_column].fillna("").astype(str).str.strip()
    if group_column == "study_group":
        group_values = group_values.replace({"": "__missing_study__", "NA": "__missing_study__"})
    else:
        group_values = group_values.replace({"": "__missing_group__", "NA": "__missing_group__"})
    observed_frame["_holdout_group"] = group_values

    group_counts = observed_frame["_holdout_group"].value_counts().to_dict()
    unique_groups = sorted(group_counts)
    if len(unique_groups) < 2:
        raise ValueError(
            f"Split strategy '{group_column}' requires at least 2 distinct groups, but "
            f"dataset '{spec.key}' only has {len(unique_groups)}."
        )

    group_split_map = assign_holdout_group_splits(group_counts)
    observed_frame["_holdout_split"] = observed_frame["_holdout_group"].map(group_split_map)

    split_frames = {
        "train": observed_frame[observed_frame["_holdout_split"] == "train"].drop(
            columns=["_holdout_group", "_holdout_split"]
        ),
        "validation": observed_frame[observed_frame["_holdout_split"] == "validation"].drop(
            columns=["_holdout_group", "_holdout_split"]
        ),
        "test": observed_frame[observed_frame["_holdout_split"] == "test"].drop(
            columns=["_holdout_group", "_holdout_split"]
        ),
    }

    if (
        "augmentation" in spec.training_splits
        and group_column_to_split_strategy(group_column) in spec.augmentation_training_splits
        and not augmentation_frame.empty
    ):
        split_frames["train"] = pd.concat(
            [split_frames["train"], augmentation_frame],
            ignore_index=True,
        )

    return split_frames


def attach_split_metadata(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    split_strategy: str,
) -> pd.DataFrame:
    if split_strategy not in {"leave_source_repo_out", "leave_study_out"}:
        return frame

    metadata = load_split_metadata(spec, frame["sample_id"])
    return frame.merge(metadata, on="sample_id", how="left", validate="one_to_one")


def feature_group_signature(frame: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    rounded = frame[feature_columns].round(8)
    return rounded.astype(str).agg("|".join, axis=1)


def stable_group_split(group_id: str) -> str:
    digest = hashlib.sha256(group_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10
    if bucket <= 6:
        return "train"
    if bucket == 7:
        return "validation"
    return "test"


def load_split_metadata(spec: DatasetSpec, sample_ids: pd.Series) -> pd.DataFrame:
    candidate_paths = [companion_dataset_path(spec), MODEL_READY_DIR / "ml_training_dataset.csv"]
    base = pd.DataFrame({"sample_id": pd.Index(sample_ids).drop_duplicates()})
    merge_columns = ["sample_id", "source_repo", "source_file", "source_dataset_kind", "reference_label"]

    for candidate in candidate_paths:
        if not candidate.exists():
            continue

        available_columns = pd.read_csv(candidate, nrows=0).columns.tolist()
        use_columns = [column for column in merge_columns if column in available_columns]
        if "sample_id" not in use_columns:
            continue

        part = pd.read_csv(candidate, usecols=use_columns)
        part = part[part["sample_id"].isin(base["sample_id"])].drop_duplicates(subset="sample_id")
        base = base.merge(part, on="sample_id", how="left", suffixes=("", "__new"))
        for column in use_columns:
            if column == "sample_id":
                continue
            new_column = f"{column}__new"
            if new_column not in base.columns:
                continue
            if column in base.columns:
                base[column] = base[column].combine_first(base[new_column])
            else:
                base[column] = base[new_column]
            base = base.drop(columns=new_column)

    base["study_group"] = derive_study_group(base)
    return base


def companion_dataset_path(spec: DatasetSpec):
    if "matrix" in spec.file_name:
        return MODEL_READY_DIR / spec.file_name.replace("matrix", "dataset")
    return MODEL_READY_DIR / spec.file_name


def derive_study_group(metadata: pd.DataFrame) -> pd.Series:
    source_repo = metadata.get("source_repo", pd.Series(index=metadata.index, dtype="object")).fillna(
        "unknown_repo"
    )
    source_file = metadata.get("source_file", pd.Series(index=metadata.index, dtype="object")).fillna(
        "unknown_source_file"
    )
    reference_label = metadata.get(
        "reference_label",
        pd.Series(index=metadata.index, dtype="object"),
    ).fillna("")

    cleaned_reference = (
        reference_label.astype(str).str.strip().replace({"": pd.NA, "NA": pd.NA, "nan": pd.NA})
    )
    fallback = source_repo.astype(str).str.strip() + "::" + source_file.astype(str).str.strip()
    return cleaned_reference.fillna(fallback)


def assign_holdout_group_splits(group_counts: dict[str, int]) -> dict[str, str]:
    ordered_groups = sorted(
        group_counts.items(),
        key=lambda item: (-item[1], stable_group_order_key(item[0])),
    )
    if len(ordered_groups) == 2:
        return {
            ordered_groups[0][0]: "train",
            ordered_groups[1][0]: "test",
        }

    assignments: dict[str, str] = {}
    seeded_splits = ["train", "validation", "test"]
    split_rows = {name: 0 for name in seeded_splits}
    total_rows = sum(group_counts.values())
    target_rows = {
        "train": total_rows * 0.70,
        "validation": total_rows * 0.15,
        "test": total_rows * 0.15,
    }

    for split_name, (group_id, row_count) in zip(seeded_splits, ordered_groups[:3], strict=False):
        assignments[group_id] = split_name
        split_rows[split_name] += row_count

    for group_id, row_count in ordered_groups[3:]:
        split_name = max(
            target_rows,
            key=lambda candidate: (
                target_rows[candidate] - split_rows[candidate],
                candidate == "train",
            ),
        )
        assignments[group_id] = split_name
        split_rows[split_name] += row_count

    return assignments


def stable_group_order_key(group_id: str) -> str:
    return hashlib.sha256(group_id.encode("utf-8")).hexdigest()


def group_column_to_split_strategy(group_column: str) -> str:
    mapping = {
        "source_repo": "leave_source_repo_out",
        "study_group": "leave_study_out",
    }
    return mapping.get(group_column, group_column)
