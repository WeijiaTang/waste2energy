from __future__ import annotations

from dataclasses import dataclass


TARGET_COLUMNS = [
    "product_char_yield_pct",
    "product_char_hhv_mj_per_kg",
    "energy_recovery_pct",
    "carbon_retention_pct",
]

DEFAULT_EXCLUDED_COLUMNS = [
    "sample_id",
    "recommended_split",
    "recommended_sample_weight",
]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    file_name: str
    description: str
    training_splits: tuple[str, ...] = ("train",)
    weight_column: str | None = None


DATASET_SPECS = {
    "htc_direct": DatasetSpec(
        key="htc_direct",
        file_name="ml_training_matrix_htc_direct.csv",
        description="Observed HTC literature rows encoded for direct supervised learning.",
    ),
    "pyrolysis_direct": DatasetSpec(
        key="pyrolysis_direct",
        file_name="ml_training_matrix_pyrolysis_direct.csv",
        description="Observed pyrolysis literature rows encoded for direct supervised learning.",
    ),
    "paper1_htc_scope": DatasetSpec(
        key="paper1_htc_scope",
        file_name="paper1_ml_matrix_htc_scope.csv",
        description=(
            "Paper 1 HTC scope with observed manure and food-waste rows plus augmentation rows."
        ),
        training_splits=("train", "augmentation"),
        weight_column="recommended_sample_weight",
    ),
}

DATASET_KEYS = tuple(DATASET_SPECS.keys())


def get_dataset_spec(dataset_key: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[dataset_key]
    except KeyError as exc:
        allowed = ", ".join(DATASET_KEYS)
        raise ValueError(f"Unsupported dataset '{dataset_key}'. Choose from: {allowed}") from exc
