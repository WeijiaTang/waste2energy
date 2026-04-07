# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..common import (
    METRIC_TON_TO_SHORT_TON,
    SHORT_TON_TO_METRIC_TON,
    emission_factor_to_metric_ton,
    normalize_emission_factor_unit,
)
from ..config import MODEL_READY_DIR


DEFAULT_PLANNING_DATASET = MODEL_READY_DIR / "optimization_input_dataset.csv"
DEFAULT_SCENARIO_METRIC_ADJUSTMENT_TABLE = MODEL_READY_DIR / "scenario_metric_adjustment_calibration.csv"

REQUIRED_PLANNING_COLUMNS = [
    "optimization_case_id",
    "sample_id",
    "scenario_name",
    "pathway",
    "blend_manure_ratio",
    "blend_wet_waste_ratio",
    "feedstock_carbon_pct",
    "feedstock_moisture_pct",
    "feedstock_hhv_mj_per_kg",
    "process_temperature_c",
    "residence_time_min",
    "product_char_yield_pct",
    "product_char_hhv_mj_per_kg",
    "energy_recovery_pct",
    "carbon_retention_pct",
    "baseline_waste_treatment_emission_factor_kgco2e_per_short_ton_reference",
    "scenario_wet_waste_feed_allocation_ton_per_year_proxy",
    "scenario_baseline_waste_treatment_emission_factor_kgco2e_per_short_ton",
    "scenario_grid_electricity_emission_factor_kgco2e_per_kwh",
    "energy_price_multiplier",
    "policy_multiplier",
    "scenario_total_mixed_feed_ton_per_year_proxy",
    "baseline_waste_treatment_factor_unit_reference",
    "net_system_cost_usd_per_year",
    "unit_net_system_cost_usd_per_ton",
    "cost_model_basis",
    "cost_model_source_trace",
]

SURROGATE_FEATURE_COLUMNS = [
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
    "feedstock_hhv_mj_per_kg",
    "process_temperature_c",
    "residence_time_min",
    "heating_rate_c_per_min",
    "feedstock_group",
    "manure_subtype",
    "source_dataset_kind",
]

REAL_COST_CANDIDATE_COLUMNS = [
    "total_system_cost_usd_per_year",
    "unit_treatment_cost_usd_per_ton",
    "net_system_cost_usd_per_year",
    "unit_net_system_cost_usd_per_ton",
]

REQUIRED_SCENARIO_METRIC_ADJUSTMENT_COLUMNS = [
    "scenario_name",
    "pathway",
    "energy_multiplier",
    "environment_multiplier",
    "cost_multiplier",
    "carbon_load_multiplier",
    "adjustment_source",
    "adjustment_reference",
    "adjustment_rationale",
]


@dataclass(frozen=True)
class PlanningInputBundle:
    frame: pd.DataFrame
    dataset_path: Path
    scenario_names: tuple[str, ...]
    pathways: tuple[str, ...]
    real_cost_columns: tuple[str, ...]
    surrogate_feature_columns: tuple[str, ...]
    unit_registry: dict[str, object]


def load_planning_input_bundle(dataset_path: str | Path | None = None) -> PlanningInputBundle:
    path = Path(dataset_path) if dataset_path else DEFAULT_PLANNING_DATASET
    if not path.exists():
        raise FileNotFoundError(f"Planning dataset not found: {path}")

    frame = pd.read_csv(path)
    validate_planning_frame(frame, path)
    frame = normalize_planning_units(frame)
    real_cost_columns = tuple(column for column in REAL_COST_CANDIDATE_COLUMNS if column in frame.columns)
    scenario_names = tuple(sorted(frame["scenario_name"].dropna().astype(str).unique().tolist()))
    pathways = tuple(sorted(frame["pathway"].dropna().astype(str).unique().tolist()))
    surrogate_feature_columns = tuple(column for column in SURROGATE_FEATURE_COLUMNS if column in frame.columns)
    return PlanningInputBundle(
        frame=frame,
        dataset_path=path,
        scenario_names=scenario_names,
        pathways=pathways,
        real_cost_columns=real_cost_columns,
        surrogate_feature_columns=surrogate_feature_columns,
        unit_registry={
            "planning_mass_unit_basis": "metric_ton",
            "baseline_emission_factor_internal_unit": "kgco2e_per_metric_ton",
            "baseline_emission_factor_source_unit_column": "baseline_waste_treatment_factor_unit_reference",
            "short_ton_to_metric_ton_factor": SHORT_TON_TO_METRIC_TON,
            "metric_ton_to_short_ton_factor": METRIC_TON_TO_SHORT_TON,
        },
    )


def load_scenario_metric_adjustment_table(
    table_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    path = Path(table_path) if table_path else DEFAULT_SCENARIO_METRIC_ADJUSTMENT_TABLE
    if not path.exists():
        raise FileNotFoundError(f"Scenario metric adjustment calibration table not found: {path}")

    frame = pd.read_csv(path)
    validate_scenario_metric_adjustment_table(frame, path)
    normalized = frame.copy()
    normalized["scenario_name"] = normalized["scenario_name"].astype(str).str.strip()
    normalized["pathway"] = normalized["pathway"].astype(str).str.strip()
    normalized["adjustment_source"] = normalized["adjustment_source"].astype(str).str.strip()
    normalized["adjustment_reference"] = normalized["adjustment_reference"].astype(str).str.strip()
    normalized["adjustment_rationale"] = normalized["adjustment_rationale"].astype(str).str.strip()
    return normalized, path


def validate_planning_frame(frame: pd.DataFrame, dataset_path: Path) -> None:
    missing = [column for column in REQUIRED_PLANNING_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Planning dataset '{dataset_path}' is missing required columns: {', '.join(missing)}"
        )

    if frame.empty:
        raise ValueError(f"Planning dataset '{dataset_path}' is empty.")

    normalized_units = (
        frame["baseline_waste_treatment_factor_unit_reference"].dropna().astype(str).map(normalize_emission_factor_unit)
    )
    if normalized_units.empty:
        raise ValueError(
            "Planning dataset must define at least one baseline_waste_treatment_factor_unit_reference value."
        )

    required_numeric_columns = [
        "baseline_waste_treatment_emission_factor_kgco2e_per_short_ton_reference",
        "scenario_baseline_waste_treatment_emission_factor_kgco2e_per_short_ton",
    ]
    for column in required_numeric_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any():
            invalid_rows = numeric[numeric.isna()].index.tolist()
            preview = ", ".join(str(index) for index in invalid_rows[:5])
            raise ValueError(
                f"Planning dataset '{dataset_path}' contains missing or non-numeric values in required column "
                f"'{column}' at row(s): {preview}."
            )


def validate_scenario_metric_adjustment_table(frame: pd.DataFrame, table_path: Path) -> None:
    missing = [column for column in REQUIRED_SCENARIO_METRIC_ADJUSTMENT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Scenario metric adjustment table '{table_path}' is missing required columns: {', '.join(missing)}"
        )

    if frame.empty:
        raise ValueError(f"Scenario metric adjustment table '{table_path}' is empty.")

    frame = frame.copy()
    frame["scenario_name"] = frame["scenario_name"].astype(str).str.strip()
    frame["pathway"] = frame["pathway"].astype(str).str.strip()
    if (frame["scenario_name"] == "").any() or (frame["pathway"] == "").any():
        raise ValueError(
            f"Scenario metric adjustment table '{table_path}' contains blank scenario_name/pathway values."
        )

    duplicate_mask = frame.duplicated(subset=["scenario_name", "pathway"], keep=False)
    if duplicate_mask.any():
        duplicates = frame.loc[duplicate_mask, ["scenario_name", "pathway"]].drop_duplicates()
        preview = ", ".join(
            f"{row.scenario_name}/{row.pathway}" for row in duplicates.itertuples(index=False)
        )
        raise ValueError(
            f"Scenario metric adjustment table '{table_path}' contains duplicate scenario/pathway rows: {preview}"
        )

    numeric_columns = [
        "energy_multiplier",
        "environment_multiplier",
        "cost_multiplier",
        "carbon_load_multiplier",
    ]
    for column in numeric_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any():
            invalid_rows = numeric[numeric.isna()].index.tolist()
            preview = ", ".join(str(index) for index in invalid_rows[:5])
            raise ValueError(
                f"Scenario metric adjustment table '{table_path}' contains missing or non-numeric values in "
                f"'{column}' at row(s): {preview}."
            )
        if (numeric <= 0.0).any():
            invalid_rows = numeric[numeric <= 0.0].index.tolist()
            preview = ", ".join(str(index) for index in invalid_rows[:5])
            raise ValueError(
                f"Scenario metric adjustment table '{table_path}' must keep '{column}' strictly positive; "
                f"violations at row(s): {preview}."
            )

    text_columns = [
        "adjustment_source",
        "adjustment_reference",
        "adjustment_rationale",
    ]
    for column in text_columns:
        values = frame[column].astype(str).str.strip()
        if (values == "").any():
            invalid_rows = values[values == ""].index.tolist()
            preview = ", ".join(str(index) for index in invalid_rows[:5])
            raise ValueError(
                f"Scenario metric adjustment table '{table_path}' contains blank values in '{column}' at row(s): {preview}."
            )


def normalize_planning_units(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    source_units = normalized["baseline_waste_treatment_factor_unit_reference"].astype(str).map(
        normalize_emission_factor_unit
    )
    normalized["baseline_emission_factor_source_unit"] = source_units
    normalized["planning_mass_unit_basis"] = "metric_ton"
    normalized["short_ton_to_metric_ton_factor"] = SHORT_TON_TO_METRIC_TON
    normalized["metric_ton_to_short_ton_factor"] = METRIC_TON_TO_SHORT_TON
    normalized["baseline_emission_factor_internal_unit"] = "kgco2e_per_metric_ton"

    normalized["baseline_waste_treatment_emission_factor_kgco2e_per_metric_ton_reference"] = [
        emission_factor_to_metric_ton(value, unit)
        for value, unit in zip(
            pd.to_numeric(
                normalized["baseline_waste_treatment_emission_factor_kgco2e_per_short_ton_reference"],
                errors="coerce",
            ),
            source_units,
            strict=False,
        )
    ]
    normalized["scenario_baseline_waste_treatment_emission_factor_kgco2e_per_metric_ton"] = [
        emission_factor_to_metric_ton(value, unit)
        for value, unit in zip(
            pd.to_numeric(
                normalized["scenario_baseline_waste_treatment_emission_factor_kgco2e_per_short_ton"],
                errors="coerce",
            ),
            source_units,
            strict=False,
        )
    ]

    if "pathway_emission_factor_kgco2e_per_short_ton_reference" in normalized.columns:
        normalized["pathway_emission_factor_kgco2e_per_metric_ton_reference"] = _convert_optional_emission_factor_series(
            pd.to_numeric(
                normalized["pathway_emission_factor_kgco2e_per_short_ton_reference"],
                errors="coerce",
            ),
            source_units,
        )
    if "pathway_emission_factor_kgco2e_per_short_ton_scenario_proxy" in normalized.columns:
        normalized["pathway_emission_factor_kgco2e_per_metric_ton_scenario_proxy"] = _convert_optional_emission_factor_series(
            pd.to_numeric(
                normalized["pathway_emission_factor_kgco2e_per_short_ton_scenario_proxy"],
                errors="coerce",
            ),
            source_units,
        )

    return normalized


def _convert_optional_emission_factor_series(
    values: pd.Series,
    source_units: pd.Series,
) -> pd.Series:
    converted: list[float] = []
    for value, unit in zip(values, source_units, strict=False):
        if pd.isna(value):
            converted.append(float("nan"))
            continue
        converted.append(emission_factor_to_metric_ton(float(value), unit))
    return pd.Series(converted, index=values.index, dtype=float)
