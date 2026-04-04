from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
UNIFIED_DIR = ROOT / "data" / "processed" / "unified_features"
MODEL_READY_DIR = ROOT / "data" / "processed" / "model_ready"

OBSERVED_INPUT = UNIFIED_DIR / "wet_waste_biomass_opt_combined_standardized.csv"
MIXED_INPUT = UNIFIED_DIR / "paper1_mixed_waste_feature_prototypes.csv"

HTC_FEATURES = [
    "feedstock_carbon_pct",
    "feedstock_hydrogen_pct",
    "feedstock_nitrogen_pct",
    "feedstock_oxygen_pct",
    "feedstock_moisture_pct",
    "feedstock_volatile_matter_pct",
    "feedstock_fixed_carbon_pct",
    "feedstock_ash_pct",
    "process_temperature_c",
    "residence_time_min",
]

PYROLYSIS_FEATURES = [
    "feedstock_carbon_pct",
    "feedstock_hydrogen_pct",
    "feedstock_nitrogen_pct",
    "feedstock_oxygen_pct",
    "feedstock_volatile_matter_pct",
    "feedstock_fixed_carbon_pct",
    "feedstock_ash_pct",
    "feedstock_hhv_mj_per_kg",
    "process_temperature_c",
    "residence_time_min",
    "heating_rate_c_per_min",
]

PAPER1_HTC_FEATURES = [
    "blend_manure_ratio",
    "blend_wet_waste_ratio",
    "feedstock_carbon_pct",
    "feedstock_hydrogen_pct",
    "feedstock_nitrogen_pct",
    "feedstock_oxygen_pct",
    "feedstock_moisture_pct",
    "feedstock_volatile_matter_pct",
    "feedstock_fixed_carbon_pct",
    "feedstock_ash_pct",
    "process_temperature_c",
    "residence_time_min",
]

TARGET_COLUMNS = [
    "product_char_yield_pct",
    "product_char_hhv_mj_per_kg",
    "energy_recovery_pct",
    "carbon_retention_pct",
]


def stable_split(sample_id: str) -> str:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def ensure_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def build_observed_htc_dataset(observed: pd.DataFrame) -> pd.DataFrame:
    htc = observed[observed["pathway"] == "htc"].copy()
    htc = ensure_numeric(htc, HTC_FEATURES + TARGET_COLUMNS)
    htc = htc.dropna(subset=HTC_FEATURES + TARGET_COLUMNS).copy()
    htc["dataset_scope"] = "observed_htc_direct_ml"
    htc["data_tier"] = "observed_literature"
    htc["recommended_split"] = htc["sample_id"].map(stable_split)
    htc["paper1_scope_flag"] = htc["feedstock_group"].isin(["food_waste", "manure"])
    columns = [
        "sample_id",
        "dataset_scope",
        "data_tier",
        "recommended_split",
        "source_repo",
        "source_file",
        "source_dataset_kind",
        "reference_label",
        "pathway",
        "feedstock_name",
        "feedstock_group",
    ] + HTC_FEATURES + TARGET_COLUMNS
    return htc[columns].reset_index(drop=True)


def build_observed_pyrolysis_dataset(observed: pd.DataFrame) -> pd.DataFrame:
    pyrolysis = observed[observed["pathway"] == "pyrolysis"].copy()
    pyrolysis = ensure_numeric(pyrolysis, PYROLYSIS_FEATURES + TARGET_COLUMNS)
    pyrolysis = pyrolysis.dropna(subset=PYROLYSIS_FEATURES + TARGET_COLUMNS).copy()
    pyrolysis["dataset_scope"] = "observed_pyrolysis_direct_ml"
    pyrolysis["data_tier"] = "observed_literature"
    pyrolysis["recommended_split"] = pyrolysis["sample_id"].map(stable_split)
    columns = [
        "sample_id",
        "dataset_scope",
        "data_tier",
        "recommended_split",
        "source_repo",
        "source_file",
        "source_dataset_kind",
        "reference_label",
        "pathway",
    ] + PYROLYSIS_FEATURES + TARGET_COLUMNS
    return pyrolysis[columns].reset_index(drop=True)


def build_paper1_scope_dataset(observed: pd.DataFrame, mixed: pd.DataFrame) -> pd.DataFrame:
    observed_scope = observed[
        (observed["pathway"] == "htc") & (observed["feedstock_group"].isin(["food_waste", "manure"]))
    ].copy()
    observed_scope = ensure_numeric(observed_scope, PAPER1_HTC_FEATURES[2:] + TARGET_COLUMNS)
    observed_scope["blend_manure_ratio"] = observed_scope["feedstock_group"].map(
        {"manure": 1.0, "food_waste": 0.0}
    )
    observed_scope["blend_wet_waste_ratio"] = observed_scope["feedstock_group"].map(
        {"manure": 0.0, "food_waste": 1.0}
    )
    observed_scope = observed_scope.dropna(subset=PAPER1_HTC_FEATURES + TARGET_COLUMNS).copy()
    observed_scope["dataset_scope"] = "paper1_htc_scope"
    observed_scope["data_tier"] = "observed_single_feed"
    observed_scope["row_origin"] = "observed"
    observed_scope["recommended_split"] = observed_scope["sample_id"].map(stable_split)
    observed_scope["recommended_sample_weight"] = 1.0
    observed_scope["recommended_ml_use"] = "benchmark_trainable"
    observed_scope["manure_subtype"] = pd.NA

    mixed_scope = mixed.copy()
    mixed_scope = ensure_numeric(mixed_scope, PAPER1_HTC_FEATURES + TARGET_COLUMNS)
    mixed_scope = mixed_scope.dropna(subset=PAPER1_HTC_FEATURES + TARGET_COLUMNS).copy()
    mixed_scope["dataset_scope"] = "paper1_htc_scope"
    mixed_scope["data_tier"] = "synthetic_blended_reference"
    mixed_scope["row_origin"] = "synthetic"
    mixed_scope["recommended_split"] = "augmentation"
    mixed_scope["recommended_sample_weight"] = 0.35
    mixed_scope["recommended_ml_use"] = "augmentation_or_scenario_screening"
    mixed_scope["source_file"] = mixed_scope["source_file"].fillna("mixed_feature_generation")
    mixed_scope["source_dataset_kind"] = mixed_scope["source_dataset_kind"].fillna(
        "synthetic_blended_reference"
    )
    if "reference_label" not in mixed_scope.columns:
        mixed_scope["reference_label"] = pd.NA

    columns = [
        "sample_id",
        "dataset_scope",
        "data_tier",
        "row_origin",
        "recommended_split",
        "recommended_sample_weight",
        "recommended_ml_use",
        "source_repo",
        "source_file",
        "source_dataset_kind",
        "reference_label",
        "pathway",
        "feedstock_name",
        "feedstock_group",
        "manure_subtype",
        "blend_manure_ratio",
        "blend_wet_waste_ratio",
    ] + PAPER1_HTC_FEATURES[2:] + TARGET_COLUMNS
    combined = pd.concat(
        [observed_scope[columns], mixed_scope[columns]],
        ignore_index=True,
    )
    return combined.reset_index(drop=True)


def build_matrix(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    passthrough_columns: list[str],
) -> pd.DataFrame:
    matrix = dataset[passthrough_columns + feature_columns + categorical_columns + TARGET_COLUMNS].copy()
    if categorical_columns:
        encoded = pd.get_dummies(matrix[categorical_columns], prefix=categorical_columns, dtype=int)
    else:
        encoded = pd.DataFrame(index=matrix.index)
    out = pd.concat(
        [
            matrix[passthrough_columns + feature_columns + TARGET_COLUMNS].reset_index(drop=True),
            encoded.reset_index(drop=True),
        ],
        axis=1,
    )
    return out


def write_manifest(
    htc_dataset: pd.DataFrame,
    pyrolysis_dataset: pd.DataFrame,
    paper1_dataset: pd.DataFrame,
    outputs: list[Path],
) -> None:
    payload = {
        "dataset_family": "ml_ready_direct_use",
        "source_files": [
            str(OBSERVED_INPUT.relative_to(ROOT)),
            str(MIXED_INPUT.relative_to(ROOT)),
        ],
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
        "dataset_summaries": {
            "ml_training_dataset_htc_direct": {
                "row_count": int(len(htc_dataset)),
                "feature_columns": HTC_FEATURES,
                "categorical_columns": ["feedstock_group"],
                "target_columns": TARGET_COLUMNS,
                "notes": [
                    "Observed HTC rows only.",
                    "Suitable for direct supervised learning without pathway-wide missing columns.",
                ],
            },
            "ml_training_dataset_pyrolysis_direct": {
                "row_count": int(len(pyrolysis_dataset)),
                "feature_columns": PYROLYSIS_FEATURES,
                "categorical_columns": [],
                "target_columns": TARGET_COLUMNS,
                "notes": [
                    "Observed pyrolysis rows only.",
                    "Feedstock-group labels are omitted because the copied pyrolysis rows do not preserve them consistently.",
                ],
            },
            "paper1_ml_dataset_htc_scope": {
                "row_count": int(len(paper1_dataset)),
                "observed_rows": int((paper1_dataset["row_origin"] == "observed").sum()),
                "synthetic_rows": int((paper1_dataset["row_origin"] == "synthetic").sum()),
                "feature_columns": PAPER1_HTC_FEATURES,
                "categorical_columns": ["feedstock_group", "manure_subtype", "row_origin"],
                "target_columns": TARGET_COLUMNS,
                "notes": [
                    "Paper 1 scoped HTC dataset built from observed manure and food-waste rows plus synthetic mixed-feed prototypes.",
                    "Synthetic rows should be treated as augmentation or scenario-screening support, not as a blind substitute for observed validation data.",
                ],
            },
        },
    }
    out_path = MODEL_READY_DIR / "ml_ready_dataset_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)

    observed = pd.read_csv(OBSERVED_INPUT)
    mixed = pd.read_csv(MIXED_INPUT)

    htc_dataset = build_observed_htc_dataset(observed)
    pyrolysis_dataset = build_observed_pyrolysis_dataset(observed)
    paper1_dataset = build_paper1_scope_dataset(observed, mixed)

    htc_matrix = build_matrix(
        htc_dataset,
        feature_columns=HTC_FEATURES,
        categorical_columns=["feedstock_group"],
        passthrough_columns=["sample_id", "recommended_split"],
    )
    pyrolysis_matrix = build_matrix(
        pyrolysis_dataset,
        feature_columns=PYROLYSIS_FEATURES,
        categorical_columns=[],
        passthrough_columns=["sample_id", "recommended_split"],
    )
    paper1_matrix = build_matrix(
        paper1_dataset,
        feature_columns=PAPER1_HTC_FEATURES,
        categorical_columns=["feedstock_group", "manure_subtype", "row_origin"],
        passthrough_columns=[
            "sample_id",
            "recommended_split",
            "recommended_sample_weight",
        ],
    )

    outputs = [
        MODEL_READY_DIR / "ml_training_dataset_htc_direct.csv",
        MODEL_READY_DIR / "ml_training_matrix_htc_direct.csv",
        MODEL_READY_DIR / "ml_training_dataset_pyrolysis_direct.csv",
        MODEL_READY_DIR / "ml_training_matrix_pyrolysis_direct.csv",
        MODEL_READY_DIR / "paper1_ml_dataset_htc_scope.csv",
        MODEL_READY_DIR / "paper1_ml_matrix_htc_scope.csv",
    ]

    htc_dataset.to_csv(outputs[0], index=False)
    htc_matrix.to_csv(outputs[1], index=False)
    pyrolysis_dataset.to_csv(outputs[2], index=False)
    pyrolysis_matrix.to_csv(outputs[3], index=False)
    paper1_dataset.to_csv(outputs[4], index=False)
    paper1_matrix.to_csv(outputs[5], index=False)

    write_manifest(htc_dataset, pyrolysis_dataset, paper1_dataset, outputs)

    for output in outputs:
        print(f"Wrote {output}")
    print(f"Wrote {MODEL_READY_DIR / 'ml_ready_dataset_manifest.json'}")


if __name__ == "__main__":
    main()
