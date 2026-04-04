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
    process_temperature = pd.to_numeric(frame["process_temperature_c"], errors="coerce").fillna(0.0)
    residence_time = pd.to_numeric(frame["residence_time_min"], errors="coerce").fillna(0.0)
    moisture = pd.to_numeric(frame["feedstock_moisture_pct"], errors="coerce").fillna(0.0)
    energy_multiplier = pd.to_numeric(frame["energy_price_multiplier"], errors="coerce").fillna(1.0)
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

    severity = (
        0.45 * _normalize(process_temperature)
        + 0.35 * _normalize(residence_time)
        + 0.20 * _normalize(moisture)
    )
    throughput_scale = _normalize(allocation_ton)
    support_relief = 1.0 + 0.60 * _normalize(energy_multiplier) + 0.40 * _normalize(policy_multiplier)

    cost_intensity, annual_cost, cost_status = _build_cost_terms(
        frame=frame,
        real_cost_columns=bundle.real_cost_columns,
        allocation_ton=allocation_ton,
        allocation_ton_safe=allocation_ton_safe,
        severity=severity,
        throughput_scale=throughput_scale,
        support_relief=support_relief,
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
    allocation_ton_safe: pd.Series,
    severity: pd.Series,
    throughput_scale: pd.Series,
    support_relief: pd.Series,
) -> tuple[pd.Series, pd.Series, str]:
    base_factor = _optional_numeric_column(frame, "pathway_cost_proxy_base_factor").fillna(1.0)
    proxy_intensity = (base_factor * ((0.70 * severity + 0.30 * throughput_scale) / support_relief)).fillna(0.0)

    if "unit_treatment_cost_usd_per_ton" in real_cost_columns:
        explicit = pd.to_numeric(frame["unit_treatment_cost_usd_per_ton"], errors="coerce")
        intensity = explicit.where(explicit.notna(), proxy_intensity).fillna(0.0)
        annual = intensity * allocation_ton
        status = (
            "mixed_real_unit_and_proxy_from_unit_treatment_cost_usd_per_ton"
            if explicit.notna().any() and explicit.isna().any()
            else "real_unit_from_unit_treatment_cost_usd_per_ton"
        )
        return intensity, annual, status

    for annual_column in ["net_system_cost_usd_per_year", "total_system_cost_usd_per_year"]:
        if annual_column in real_cost_columns:
            explicit_annual = pd.to_numeric(frame[annual_column], errors="coerce")
            explicit_intensity = (explicit_annual / allocation_ton_safe).replace([np.inf, -np.inf], np.nan)
            intensity = explicit_intensity.where(explicit_intensity.notna(), proxy_intensity).fillna(0.0)
            annual = explicit_annual.where(explicit_annual.notna(), intensity * allocation_ton).fillna(0.0)
            status = (
                f"mixed_real_annual_and_proxy_from_{annual_column}"
                if explicit_annual.notna().any() and explicit_annual.isna().any()
                else f"real_annual_from_{annual_column}"
            )
            return intensity, annual, status

    annual = proxy_intensity * allocation_ton
    return proxy_intensity, annual, "proxy_only_process_and_support_index"


def _optional_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")
