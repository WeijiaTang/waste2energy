from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
    "net_system_cost_usd_per_year",
    "unit_net_system_cost_usd_per_ton",
    "cost_model_basis",
    "cost_model_source_trace",
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


def load_planning_input_bundle(dataset_path: str | Path | None = None) -> PlanningInputBundle:
    path = Path(dataset_path) if dataset_path else DEFAULT_PLANNING_DATASET
    if not path.exists():
        raise FileNotFoundError(f"Planning dataset not found: {path}")

    frame = pd.read_csv(path)
    validate_planning_frame(frame, path)
    real_cost_columns = tuple(column for column in REAL_COST_CANDIDATE_COLUMNS if column in frame.columns)
    scenario_names = tuple(sorted(frame["scenario_name"].dropna().astype(str).unique().tolist()))
    pathways = tuple(sorted(frame["pathway"].dropna().astype(str).unique().tolist()))
    return PlanningInputBundle(
        frame=frame,
        dataset_path=path,
        scenario_names=scenario_names,
        pathways=pathways,
        real_cost_columns=real_cost_columns,
    )


def validate_planning_frame(frame: pd.DataFrame, dataset_path: Path) -> None:
    missing = [column for column in REQUIRED_PLANNING_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Planning dataset '{dataset_path}' is missing required columns: {', '.join(missing)}"
        )

    if frame.empty:
        raise ValueError(f"Planning dataset '{dataset_path}' is empty.")
