from __future__ import annotations

import numpy as np
import pandas as pd

from .inputs import PlanningInputBundle


def enrich_with_objectives(bundle: PlanningInputBundle) -> tuple[pd.DataFrame, dict[str, str]]:
    frame = bundle.frame.copy()

    allocation_ton = pd.to_numeric(
        frame["scenario_wet_waste_feed_allocation_ton_per_year_proxy"], errors="coerce"
    ).fillna(0.0)
    allocation_ton_safe = allocation_ton.replace(0.0, np.nan)
    char_yield = pd.to_numeric(frame["product_char_yield_pct"], errors="coerce").fillna(0.0) / 100.0
    char_hhv = pd.to_numeric(frame["product_char_hhv_mj_per_kg"], errors="coerce").fillna(0.0)
    carbon_fraction = pd.to_numeric(frame["feedstock_carbon_pct"], errors="coerce").fillna(0.0) / 100.0
    carbon_retention = pd.to_numeric(frame["carbon_retention_pct"], errors="coerce").fillna(0.0) / 100.0
    baseline_emission = pd.to_numeric(
        frame["scenario_baseline_waste_treatment_emission_factor_kgco2e_per_short_ton"],
        errors="coerce",
    ).fillna(0.0)
    total_mixed_feed_ton = pd.to_numeric(
        frame.get("scenario_total_mixed_feed_ton_per_year_proxy"), errors="coerce"
    ).fillna(allocation_ton)
    process_temperature = pd.to_numeric(frame["process_temperature_c"], errors="coerce").fillna(0.0)
    residence_time = pd.to_numeric(frame["residence_time_min"], errors="coerce").fillna(0.0)
    moisture = pd.to_numeric(frame["feedstock_moisture_pct"], errors="coerce").fillna(0.0)
    policy_multiplier = pd.to_numeric(frame["policy_multiplier"], errors="coerce").fillna(1.0)
    explicit_energy_intensity = _optional_numeric_column(frame, "pathway_energy_intensity_mj_per_ton")
    energy_intensity = explicit_energy_intensity.fillna(1000.0 * char_yield * char_hhv)

    explicit_environment_benefit = _optional_numeric_column(
        frame, "pathway_environment_benefit_kgco2e_per_ton"
    )
    environment_intensity = explicit_environment_benefit.fillna(baseline_emission * carbon_retention)
    environment_intensity = environment_intensity * policy_multiplier

    frame["recoverable_energy_proxy_mj_per_year"] = allocation_ton * energy_intensity
    frame["stored_carbon_proxy_ton_per_year"] = allocation_ton * 1000.0 * carbon_fraction * carbon_retention / 1000.0
    frame["avoided_baseline_emissions_proxy_kgco2e_per_year"] = allocation_ton * environment_intensity

    cost_intensity, annual_cost, cost_status = _build_cost_terms(
        frame=frame,
        real_cost_columns=bundle.real_cost_columns,
        allocation_ton=allocation_ton,
        total_mixed_feed_ton=total_mixed_feed_ton,
    )

    frame["planning_energy_intensity_mj_per_ton"] = energy_intensity.fillna(0.0)
    frame["planning_environment_intensity_kgco2e_per_ton"] = environment_intensity.fillna(0.0)
    frame["planning_cost_intensity_proxy_or_real_per_ton"] = cost_intensity.fillna(0.0)

    frame["planning_energy_objective"] = frame["recoverable_energy_proxy_mj_per_year"]
    frame["planning_environment_objective"] = frame["avoided_baseline_emissions_proxy_kgco2e_per_year"]
    frame["total_cost_proxy_or_real"] = annual_cost.fillna(0.0)
    frame["planning_cost_objective"] = frame["total_cost_proxy_or_real"]

    readiness = {
        "energy_objective_status": (
            "mixed_explicit_pathway_energy_and_char_fallback"
            if explicit_energy_intensity.notna().any()
            else "semi_physical_proxy_from_char_yield_and_char_hhv"
        ),
        "environment_objective_status": (
            "mixed_explicit_pathway_benefit_and_carbon_retention_fallback"
            if explicit_environment_benefit.notna().any()
            else "proxy_from_baseline_emission_and_carbon_retention"
        ),
        "cost_objective_status": cost_status,
    }
    return frame, readiness


def _normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return pd.Series(0.0, index=values.index)
    return (values - minimum) / (maximum - minimum)


def _build_cost_terms(
    *,
    frame: pd.DataFrame,
    real_cost_columns: tuple[str, ...],
    allocation_ton: pd.Series,
    total_mixed_feed_ton: pd.Series,
) -> tuple[pd.Series, pd.Series, str]:
    annual_cost = _optional_numeric_column(frame, "net_system_cost_usd_per_year")
    if annual_cost.notna().all():
        intensity = (annual_cost / allocation_ton.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        return intensity.fillna(0.0), annual_cost.fillna(0.0), "real_net_annual_system_cost"

    unit_net_cost = _optional_numeric_column(frame, "unit_net_system_cost_usd_per_ton")
    if unit_net_cost.notna().all():
        annual = (unit_net_cost * allocation_ton).fillna(0.0)
        return unit_net_cost.fillna(0.0), annual, "real_unit_net_system_cost"

    unit_total_feed_cost = _optional_numeric_column(frame, "unit_net_system_cost_usd_per_total_mixed_ton")
    if unit_total_feed_cost.notna().all():
        annual = (unit_total_feed_cost * total_mixed_feed_ton).fillna(0.0)
        intensity = (annual / allocation_ton.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        return intensity.fillna(0.0), annual, "real_total_feed_scaled_net_system_cost"

    raise ValueError(
        "Planning cost objective now requires real net cost columns. "
        "Regenerate optimization_input_dataset.csv with net_system_cost_usd_per_year or unit_net_system_cost_usd_per_ton."
    )


def _optional_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")
