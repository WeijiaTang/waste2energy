# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import numpy as np
import pandas as pd


SURROGATE_TARGET_COLUMNS = (
    "product_char_yield_pct",
    "product_char_hhv_mj_per_kg",
    "energy_recovery_pct",
    "carbon_retention_pct",
)


def assemble_objective_frame(
    *,
    base_frame: pd.DataFrame,
    surrogate_predictions: pd.DataFrame,
    robustness_factor: float,
    real_cost_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, str]]:
    frame = base_frame.copy()
    merged = frame.merge(
        surrogate_predictions,
        on=["optimization_case_id", "pathway"],
        how="left",
        validate="one_to_one",
    )

    for target in SURROGATE_TARGET_COLUMNS:
        predicted = pd.to_numeric(
            merged.get(f"predicted_{target}", merged.get(target, 0.0)),
            errors="coerce",
        )
        direct = pd.to_numeric(merged.get(target, 0.0), errors="coerce")
        merged[target] = predicted.fillna(direct).fillna(0.0)

    allocation_ton = pd.to_numeric(
        merged["scenario_wet_waste_feed_allocation_ton_per_year_proxy"],
        errors="coerce",
    ).fillna(0.0)
    allocation_ton_safe = allocation_ton.replace(0.0, np.nan)
    total_mixed_feed_ton = pd.to_numeric(
        merged.get("scenario_total_mixed_feed_ton_per_year_proxy"),
        errors="coerce",
    ).fillna(allocation_ton)
    total_mixed_feed_ton_safe = total_mixed_feed_ton.replace(0.0, np.nan)

    feedstock_hhv = pd.to_numeric(merged["feedstock_hhv_mj_per_kg"], errors="coerce").fillna(0.0)
    char_yield = pd.to_numeric(merged["product_char_yield_pct"], errors="coerce").fillna(0.0) / 100.0
    char_hhv = pd.to_numeric(merged["product_char_hhv_mj_per_kg"], errors="coerce").fillna(0.0)
    energy_recovery = pd.to_numeric(merged["energy_recovery_pct"], errors="coerce").fillna(0.0) / 100.0
    carbon_retention = pd.to_numeric(merged["carbon_retention_pct"], errors="coerce").fillna(0.0) / 100.0
    carbon_fraction = pd.to_numeric(merged["feedstock_carbon_pct"], errors="coerce").fillna(0.0) / 100.0
    baseline_emission = pd.to_numeric(
        merged["scenario_baseline_waste_treatment_emission_factor_kgco2e_per_metric_ton"],
        errors="coerce",
    ).fillna(0.0)
    policy_multiplier = pd.to_numeric(merged["policy_multiplier"], errors="coerce").fillna(1.0)

    explicit_energy_intensity = _optional_numeric_column(merged, "pathway_energy_intensity_mj_per_ton")
    surrogate_energy_intensity = (feedstock_hhv * 1000.0 * energy_recovery).replace([np.inf, -np.inf], np.nan)
    char_energy_intensity = (1000.0 * char_yield * char_hhv).replace([np.inf, -np.inf], np.nan)
    energy_intensity = surrogate_energy_intensity.where(
        surrogate_energy_intensity > 0.0,
        char_energy_intensity,
    )
    energy_intensity = energy_intensity.where(energy_intensity > 0.0, explicit_energy_intensity)
    energy_intensity = energy_intensity.fillna(char_energy_intensity).fillna(explicit_energy_intensity).fillna(0.0)

    explicit_environment_benefit = _optional_numeric_column(
        merged,
        "pathway_environment_benefit_kgco2e_per_ton",
    )
    environment_intensity = explicit_environment_benefit.where(
        explicit_environment_benefit.notna(),
        baseline_emission * carbon_retention,
    ).fillna(0.0)
    environment_intensity = environment_intensity * policy_multiplier

    uncertainty_ratio = pd.to_numeric(
        merged.get("combined_uncertainty_ratio", 0.0),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    robust_multiplier = (1.0 - robustness_factor * uncertainty_ratio).clip(lower=0.50, upper=1.00)
    robust_cost_multiplier = (1.0 + robustness_factor * uncertainty_ratio).clip(lower=1.00, upper=1.50)

    merged["recoverable_energy_proxy_mj_per_year"] = total_mixed_feed_ton * energy_intensity
    merged["stored_carbon_proxy_ton_per_year"] = (
        total_mixed_feed_ton * 1000.0 * carbon_fraction * carbon_retention / 1000.0
    )
    merged["avoided_baseline_emissions_proxy_kgco2e_per_year"] = total_mixed_feed_ton * environment_intensity

    cost_intensity, annual_cost, cost_status = _build_cost_terms(
        frame=merged,
        real_cost_columns=real_cost_columns,
        allocation_ton=allocation_ton,
        total_mixed_feed_ton=total_mixed_feed_ton,
    )

    merged["planning_energy_intensity_mj_per_ton"] = (energy_intensity * robust_multiplier).fillna(0.0)
    merged["planning_environment_intensity_kgco2e_per_ton"] = (
        environment_intensity * robust_multiplier
    ).fillna(0.0)
    merged["planning_cost_intensity_proxy_or_real_per_ton"] = (
        cost_intensity * robust_cost_multiplier
    ).fillna(0.0)

    merged["planning_energy_objective"] = (
        total_mixed_feed_ton * merged["planning_energy_intensity_mj_per_ton"]
    ).fillna(0.0)
    merged["planning_environment_objective"] = (
        total_mixed_feed_ton * merged["planning_environment_intensity_kgco2e_per_ton"]
    ).fillna(0.0)
    merged["total_cost_proxy_or_real"] = annual_cost.fillna(0.0)
    merged["planning_cost_objective"] = (
        merged["planning_cost_intensity_proxy_or_real_per_ton"] * allocation_ton_safe
    ).replace([np.inf, -np.inf], np.nan).fillna(merged["total_cost_proxy_or_real"])

    merged["robustness_penalty"] = (
        robustness_factor * uncertainty_ratio * np.maximum(merged["planning_energy_objective"], 0.0)
    )
    merged["combined_uncertainty_ratio"] = uncertainty_ratio
    merged["planning_carbon_load_kgco2e_per_ton"] = _build_carbon_load(merged, baseline_emission)
    merged["planning_energy_balance_mj_per_year"] = (
        total_mixed_feed_ton * merged["planning_energy_intensity_mj_per_ton"]
    )
    merged["planning_mass_balance_feed_ton_per_year"] = total_mixed_feed_ton
    merged["planning_cost_balance_usd_per_year"] = merged["planning_cost_objective"]
    merged["planning_carbon_unit_basis"] = merged.get(
        "baseline_emission_factor_internal_unit",
        "kgco2e_per_metric_ton",
    )

    readiness = {
        "energy_objective_status": "surrogate_or_direct_energy_intensity_with_robust_adjustment",
        "environment_objective_status": "surrogate_or_direct_environment_benefit_with_robust_adjustment",
        "cost_objective_status": cost_status,
        "robustness_status": "prediction_interval_penalty_applied_to_energy_environment_and_cost",
    }
    return merged, readiness


def _build_cost_terms(
    *,
    frame: pd.DataFrame,
    real_cost_columns: tuple[str, ...],
    allocation_ton: pd.Series,
    total_mixed_feed_ton: pd.Series,
) -> tuple[pd.Series, pd.Series, str]:
    annual_cost = _optional_numeric_column(frame, "net_system_cost_usd_per_year")
    if annual_cost.notna().any():
        intensity = (annual_cost / allocation_ton.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        return intensity.fillna(0.0), annual_cost.fillna(0.0), "real_net_annual_system_cost"

    unit_net_cost = _optional_numeric_column(frame, "unit_net_system_cost_usd_per_ton")
    if unit_net_cost.notna().any():
        annual = (unit_net_cost * allocation_ton).fillna(0.0)
        return unit_net_cost.fillna(0.0), annual, "real_unit_net_system_cost"

    unit_total_feed_cost = _optional_numeric_column(frame, "unit_net_system_cost_usd_per_total_mixed_ton")
    if unit_total_feed_cost.notna().any():
        annual = (unit_total_feed_cost * total_mixed_feed_ton).fillna(0.0)
        intensity = (annual / allocation_ton.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        return intensity.fillna(0.0), annual, "real_total_feed_scaled_net_system_cost"

    available = ", ".join(real_cost_columns) if real_cost_columns else "none"
    raise ValueError(
        "Planning cost objective requires real net cost columns. "
        f"Available detected columns: {available or 'none'}."
    )


def _build_carbon_load(frame: pd.DataFrame, baseline_emission: pd.Series) -> pd.Series:
    explicit = _optional_numeric_column(frame, "pathway_emission_factor_kgco2e_per_metric_ton_scenario_proxy")
    if explicit.notna().any():
        return explicit.fillna((baseline_emission - frame["planning_environment_intensity_kgco2e_per_ton"]).clip(lower=0.0))
    return (baseline_emission - frame["planning_environment_intensity_kgco2e_per_ton"]).clip(lower=0.0)


def _optional_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")
