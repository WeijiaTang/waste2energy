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
            ).fillna(0.0),
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
            ).fillna(0.0),
            source_units,
            strict=False,
        )
    ]

    if "pathway_emission_factor_kgco2e_per_short_ton_reference" in normalized.columns:
        normalized["pathway_emission_factor_kgco2e_per_metric_ton_reference"] = [
            emission_factor_to_metric_ton(value, unit)
            for value, unit in zip(
                pd.to_numeric(
                    normalized["pathway_emission_factor_kgco2e_per_short_ton_reference"],
                    errors="coerce",
                ).fillna(0.0),
                source_units,
                strict=False,
            )
        ]
    if "pathway_emission_factor_kgco2e_per_short_ton_scenario_proxy" in normalized.columns:
        normalized["pathway_emission_factor_kgco2e_per_metric_ton_scenario_proxy"] = [
            emission_factor_to_metric_ton(value, unit)
            for value, unit in zip(
                pd.to_numeric(
                    normalized["pathway_emission_factor_kgco2e_per_short_ton_scenario_proxy"],
                    errors="coerce",
                ).fillna(0.0),
                source_units,
                strict=False,
            )
        ]

    return normalized
