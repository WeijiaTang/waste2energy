# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import numpy as np
import pandas as pd

from .surrogate_evaluator import SUPPORTED_SURROGATE_PATHWAYS


SURROGATE_TARGET_COLUMNS = (
    "product_char_yield_pct",
    "product_char_hhv_mj_per_kg",
    "energy_recovery_pct",
    "carbon_retention_pct",
)

CRITICAL_COLUMNS = {
    "scenario_wet_waste_feed_allocation_ton_per_year_proxy": "allocation mass basis is required for annualized planning objectives",
    "scenario_baseline_waste_treatment_emission_factor_kgco2e_per_metric_ton": "baseline treatment emission factor is required for carbon accounting",
}

NONCRITICAL_IMPUTATION_COLUMNS = {
    "combined_uncertainty_ratio": 0.0,
    "policy_multiplier": 1.0,
}


def assemble_objective_frame(
    *,
    base_frame: pd.DataFrame,
    surrogate_predictions: pd.DataFrame,
    robustness_factor: float,
    real_cost_columns: tuple[str, ...],
    config=None,
) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame, pd.DataFrame]:
    if config is None:
        from .solve import PlanningConfig

        config = PlanningConfig()
    frame = base_frame.copy()
    merged = frame.merge(
        surrogate_predictions,
        on=["optimization_case_id", "pathway"],
        how="left",
        validate="one_to_one",
    )
    merged["planning_imputation_notes"] = ""
    merged["planning_exclusion_reason"] = ""
    merged["planning_data_quality_status"] = "pending"
    merged["planning_energy_intensity_source"] = "missing"
    merged["planning_environment_intensity_source"] = "missing"
    merged["planning_cost_intensity_source"] = "missing"
    merged["planning_carbon_load_source"] = "missing"
    merged["scenario_total_mixed_feed_source"] = "missing"
    merged["combined_uncertainty_ratio_source"] = "missing"
    merged["policy_multiplier_source"] = "missing"
    merged["scenario_external_evidence_source"] = "missing"

    for column, note in CRITICAL_COLUMNS.items():
        series = _required_numeric_column(merged, column)
        missing_mask = series.isna()
        if missing_mask.any():
            merged.loc[missing_mask, "planning_exclusion_reason"] = _append_message(
                merged.loc[missing_mask, "planning_exclusion_reason"],
                f"critical_missing:{column}",
            )
            merged.loc[missing_mask, "planning_imputation_notes"] = _append_message(
                merged.loc[missing_mask, "planning_imputation_notes"],
                f"{column}:{note}",
            )
        merged[column] = series

    for target in SURROGATE_TARGET_COLUMNS:
        predicted = _optional_numeric_column(merged, f"predicted_{target}")
        direct = _optional_numeric_column(merged, target)
        merged[target], merged = _coalesce_with_flag(
            frame=merged,
            target_column=target,
            preferred=predicted,
            fallback=direct,
            fallback_reason="direct_measurement",
        )

    merged["combined_uncertainty_ratio"], merged = _impute_with_flag(
        frame=merged,
        column="combined_uncertainty_ratio",
        default_value=NONCRITICAL_IMPUTATION_COLUMNS["combined_uncertainty_ratio"],
        flag_reason="default_uncertainty_zero",
        source_column="combined_uncertainty_ratio_source",
        explicit_source_label="explicit_uncertainty_column",
        default_source_label="default_uncertainty_zero",
    )
    merged["policy_multiplier"], merged = _impute_with_flag(
        frame=merged,
        column="policy_multiplier",
        default_value=NONCRITICAL_IMPUTATION_COLUMNS["policy_multiplier"],
        flag_reason="default_policy_multiplier_one",
        source_column="policy_multiplier_source",
        explicit_source_label="explicit_policy_multiplier_column",
        default_source_label="default_policy_multiplier_one",
    )
    merged["feedstock_scale_factor"], merged = _impute_with_flag(
        frame=merged,
        column="feedstock_scale_factor",
        default_value=1.0,
        flag_reason="default_feedstock_scale_factor_one",
        source_column="scenario_external_evidence_source",
        explicit_source_label="scenario_external_evidence_table",
        default_source_label="default_feedstock_scale_factor_one",
    )
    merged["feedstock_cost_elasticity"], merged = _impute_with_flag(
        frame=merged,
        column="feedstock_cost_elasticity",
        default_value=0.0,
        flag_reason="default_feedstock_cost_elasticity_zero",
        source_column=None,
    )
    merged["carbon_tax_usd_per_ton_co2e"], merged = _impute_with_flag(
        frame=merged,
        column="carbon_tax_usd_per_ton_co2e",
        default_value=0.0,
        flag_reason="default_carbon_tax_zero",
        source_column=None,
    )

    allocation_ton = merged["scenario_wet_waste_feed_allocation_ton_per_year_proxy"]
    allocation_ton_safe = allocation_ton.replace(0.0, np.nan)
    total_mixed_feed_ton, merged = _coalesce_with_flag(
        frame=merged,
        target_column="scenario_total_mixed_feed_ton_per_year_proxy",
        preferred=_optional_numeric_column(merged, "scenario_total_mixed_feed_ton_per_year_proxy"),
        fallback=allocation_ton,
        fallback_reason="fallback_to_scenario_wet_waste_feed_allocation",
        source_column="scenario_total_mixed_feed_source",
        preferred_source_label="explicit_total_mixed_feed_column",
        fallback_source_label="fallback_to_wet_waste_feed_allocation",
    )
    total_mixed_feed_ton_safe = total_mixed_feed_ton.replace(0.0, np.nan)
    baseline_emission = merged["scenario_baseline_waste_treatment_emission_factor_kgco2e_per_metric_ton"]
    policy_multiplier = merged["policy_multiplier"]

    explicit_energy_intensity = _optional_numeric_column(merged, "pathway_energy_intensity_mj_per_ton")
    explicit_environment_benefit = _optional_numeric_column(merged, "pathway_environment_benefit_kgco2e_per_ton")
    merged = _attach_surrogate_support_columns(merged, config)

    feedstock_hhv = _optional_numeric_column(merged, "feedstock_hhv_mj_per_kg")
    char_yield = _optional_numeric_column(merged, "product_char_yield_pct") / 100.0
    char_hhv = _optional_numeric_column(merged, "product_char_hhv_mj_per_kg")
    energy_recovery = _optional_numeric_column(merged, "energy_recovery_pct") / 100.0
    carbon_retention = _optional_numeric_column(merged, "carbon_retention_pct") / 100.0
    carbon_fraction = _optional_numeric_column(merged, "feedstock_carbon_pct") / 100.0

    surrogate_energy_intensity = (feedstock_hhv * 1000.0 * energy_recovery).replace([np.inf, -np.inf], np.nan)
    char_energy_intensity = (1000.0 * char_yield * char_hhv).replace([np.inf, -np.inf], np.nan)
    energy_intensity = pd.Series(np.nan, index=merged.index, dtype=float)
    energy_intensity, merged = _adopt_series(
        frame=merged,
        target=energy_intensity,
        candidate=surrogate_energy_intensity.where(surrogate_energy_intensity > 0.0),
        source_column="planning_energy_intensity_source",
        source_label="surrogate_energy_formula",
    )
    energy_intensity, merged = _adopt_series(
        frame=merged,
        target=energy_intensity,
        candidate=char_energy_intensity.where(char_energy_intensity > 0.0),
        source_column="planning_energy_intensity_source",
        source_label="char_energy_formula",
    )
    energy_intensity, merged = _adopt_series(
        frame=merged,
        target=energy_intensity,
        candidate=explicit_energy_intensity.where(explicit_energy_intensity > 0.0),
        source_column="planning_energy_intensity_source",
        source_label="explicit_energy_column",
    )

    missing_energy_mask = energy_intensity.isna()
    if missing_energy_mask.any():
        merged.loc[missing_energy_mask, "planning_exclusion_reason"] = _append_message(
            merged.loc[missing_energy_mask, "planning_exclusion_reason"],
            "critical_missing:planning_energy_intensity_mj_per_ton",
        )
        merged.loc[missing_energy_mask, "planning_imputation_notes"] = _append_message(
            merged.loc[missing_energy_mask, "planning_imputation_notes"],
            "planning_energy_intensity_mj_per_ton:no_physical_or_explicit_source_available",
        )

    environment_intensity = explicit_environment_benefit.where(explicit_environment_benefit.notna())
    merged.loc[environment_intensity.notna(), "planning_environment_intensity_source"] = "explicit_environment_column"
    environment_intensity, merged = _adopt_series(
        frame=merged,
        target=environment_intensity,
        candidate=baseline_emission * carbon_retention,
        source_column="planning_environment_intensity_source",
        source_label="baseline_times_carbon_retention",
    )
    environment_intensity = environment_intensity * policy_multiplier
    missing_environment_mask = environment_intensity.isna()
    if missing_environment_mask.any():
        merged.loc[missing_environment_mask, "planning_exclusion_reason"] = _append_message(
            merged.loc[missing_environment_mask, "planning_exclusion_reason"],
            "critical_missing:planning_environment_intensity_kgco2e_per_ton",
        )
        merged.loc[missing_environment_mask, "planning_imputation_notes"] = _append_message(
            merged.loc[missing_environment_mask, "planning_imputation_notes"],
            "planning_environment_intensity_kgco2e_per_ton:no_explicit_or_fallback_source_available",
        )

    uncertainty_ratio = merged["combined_uncertainty_ratio"].clip(lower=0.0)
    effective_uncertainty_ratio = (
        uncertainty_ratio * pd.to_numeric(merged["evidence_uncertainty_multiplier"], errors="coerce").fillna(1.0)
    ).clip(lower=0.0)
    robust_multiplier = (1.0 - robustness_factor * effective_uncertainty_ratio).clip(lower=0.40, upper=1.00)
    robust_cost_multiplier = (1.0 + robustness_factor * effective_uncertainty_ratio).clip(lower=1.00, upper=1.90)

    merged["recoverable_energy_proxy_mj_per_year"] = total_mixed_feed_ton * energy_intensity
    merged["stored_carbon_proxy_ton_per_year"] = (
        total_mixed_feed_ton * 1000.0 * carbon_fraction * carbon_retention / 1000.0
    )
    merged["avoided_baseline_emissions_proxy_kgco2e_per_year"] = total_mixed_feed_ton * environment_intensity

    merged["planning_energy_intensity_mj_per_ton"] = energy_intensity * robust_multiplier
    merged["planning_environment_intensity_kgco2e_per_ton"] = environment_intensity * robust_multiplier
    merged["planning_carbon_load_kgco2e_per_ton"] = _build_carbon_load(merged, baseline_emission)

    cost_intensity, annual_cost, cost_status, merged = _build_cost_terms(
        frame=merged,
        real_cost_columns=real_cost_columns,
        allocation_ton=allocation_ton,
        total_mixed_feed_ton=total_mixed_feed_ton,
        carbon_load=merged["planning_carbon_load_kgco2e_per_ton"],
    )

    merged["planning_cost_intensity_proxy_or_real_per_ton"] = cost_intensity * robust_cost_multiplier

    merged["planning_energy_objective"] = total_mixed_feed_ton * merged["planning_energy_intensity_mj_per_ton"]
    merged["planning_environment_objective"] = (
        total_mixed_feed_ton * merged["planning_environment_intensity_kgco2e_per_ton"]
    )
    merged["total_cost_proxy_or_real"] = annual_cost
    merged["planning_cost_objective"] = (
        merged["planning_cost_intensity_proxy_or_real_per_ton"] * allocation_ton_safe
    ).replace([np.inf, -np.inf], np.nan).fillna(merged["total_cost_proxy_or_real"])

    merged["robustness_penalty"] = (
        robustness_factor * effective_uncertainty_ratio * np.maximum(merged["planning_energy_objective"], 0.0)
    )
    merged["combined_uncertainty_ratio"] = uncertainty_ratio
    merged["effective_uncertainty_ratio"] = effective_uncertainty_ratio
    merged["planning_energy_balance_mj_per_year"] = (
        total_mixed_feed_ton * merged["planning_energy_intensity_mj_per_ton"]
    )
    merged["planning_mass_balance_feed_ton_per_year"] = total_mixed_feed_ton
    merged["planning_cost_balance_usd_per_year"] = merged["planning_cost_objective"]
    merged["planning_carbon_unit_basis"] = merged.get(
        "baseline_emission_factor_internal_unit",
        "kgco2e_per_metric_ton",
    )
    merged["planning_imputation_flag"] = merged["planning_imputation_notes"].astype(str).str.len().gt(0)

    exclusions = merged[merged["planning_exclusion_reason"].astype(str).str.len() > 0].copy()
    exclusions["planning_data_quality_status"] = "excluded"
    retained = merged[merged["planning_exclusion_reason"].astype(str).str.len() == 0].copy()
    retained["planning_data_quality_status"] = np.where(
        retained["planning_imputation_flag"],
        "imputed_noncritical",
        "complete",
    )

    readiness = {
        "energy_objective_status": "surrogate_or_direct_energy_intensity_with_explicit_missing_data_guardrails",
        "environment_objective_status": "surrogate_or_direct_environment_benefit_with_explicit_missing_data_guardrails",
        "cost_objective_status": cost_status,
        "robustness_status": "prediction_interval_penalty_applied_with_evidence_weighted_uncertainty",
        "data_quality_status": "critical-missing candidates excluded; noncritical imputations flagged in outputs",
    }

    data_quality_summary = _build_data_quality_summary(
        evaluated_frame=merged,
        retained_frame=retained,
        excluded_frame=exclusions,
    )
    return retained.reset_index(drop=True), readiness, data_quality_summary, exclusions.reset_index(drop=True)


def _build_cost_terms(
    *,
    frame: pd.DataFrame,
    real_cost_columns: tuple[str, ...],
    allocation_ton: pd.Series,
    total_mixed_feed_ton: pd.Series,
    carbon_load: pd.Series,
) -> tuple[pd.Series, pd.Series, str, pd.DataFrame]:
    annual_cost = _optional_numeric_column(frame, "net_system_cost_usd_per_year")
    unit_net_cost = _optional_numeric_column(frame, "unit_net_system_cost_usd_per_ton")
    unit_total_feed_cost = _optional_numeric_column(frame, "unit_net_system_cost_usd_per_total_mixed_ton")
    allocation_ton_safe = allocation_ton.replace(0.0, np.nan)

    if not (annual_cost.notna().any() or unit_net_cost.notna().any() or unit_total_feed_cost.notna().any()):
        available = ", ".join(real_cost_columns) if real_cost_columns else "none"
        raise ValueError(
            "Planning cost objective requires real net cost columns. "
            f"Available detected columns: {available or 'none'}."
        )

    intensity = pd.Series(np.nan, index=frame.index, dtype=float)
    annual = pd.Series(np.nan, index=frame.index, dtype=float)

    annual_mask = annual_cost.notna() & allocation_ton_safe.notna()
    if annual_mask.any():
        annual.loc[annual_mask] = annual_cost.loc[annual_mask]
        intensity.loc[annual_mask] = (
            annual_cost.loc[annual_mask] / allocation_ton.loc[annual_mask].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        frame.loc[annual_mask, "planning_cost_intensity_source"] = "real_net_annual_system_cost"

    unit_mask = intensity.isna() & unit_net_cost.notna() & allocation_ton.notna()
    if unit_mask.any():
        intensity.loc[unit_mask] = unit_net_cost.loc[unit_mask]
        annual.loc[unit_mask] = unit_net_cost.loc[unit_mask] * allocation_ton.loc[unit_mask]
        frame.loc[unit_mask, "planning_cost_intensity_source"] = "real_unit_net_system_cost"

    total_feed_mask = intensity.isna() & unit_total_feed_cost.notna() & total_mixed_feed_ton.notna() & allocation_ton_safe.notna()
    if total_feed_mask.any():
        annual.loc[total_feed_mask] = unit_total_feed_cost.loc[total_feed_mask] * total_mixed_feed_ton.loc[total_feed_mask]
        intensity.loc[total_feed_mask] = (
            annual.loc[total_feed_mask] / allocation_ton.loc[total_feed_mask].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        frame.loc[total_feed_mask, "planning_cost_intensity_source"] = "real_total_feed_scaled_net_system_cost"

    missing_mask = intensity.isna() | annual.isna()
    if missing_mask.any():
        frame.loc[missing_mask, "planning_exclusion_reason"] = _append_message(
            frame.loc[missing_mask, "planning_exclusion_reason"],
            "critical_missing:planning_cost_intensity_proxy_or_real_per_ton",
        )
        frame.loc[missing_mask, "planning_imputation_notes"] = _append_message(
            frame.loc[missing_mask, "planning_imputation_notes"],
            "planning_cost_intensity_proxy_or_real_per_ton:no_valid_real_cost_source_available",
        )

    feedstock_scale_factor = _optional_numeric_column(frame, "feedstock_scale_factor").clip(lower=1e-6).fillna(1.0)
    feedstock_cost_elasticity = _optional_numeric_column(frame, "feedstock_cost_elasticity").clip(lower=0.0).fillna(0.0)
    scenario_scale_multiplier = np.power(feedstock_scale_factor, -feedstock_cost_elasticity).clip(0.70, 1.05)
    carbon_tax = _optional_numeric_column(frame, "carbon_tax_usd_per_ton_co2e").clip(lower=0.0).fillna(0.0)
    carbon_tax_cost = carbon_tax * carbon_load.clip(lower=0.0).fillna(0.0) / 1000.0
    information_premium = _optional_numeric_column(frame, "information_deficit_premium_usd_per_ton").clip(lower=0.0).fillna(0.0)

    frame["scenario_scale_cost_multiplier"] = scenario_scale_multiplier
    frame["carbon_tax_cost_intensity_usd_per_ton"] = carbon_tax_cost
    frame["information_deficit_premium_usd_per_ton"] = information_premium

    adjusted_intensity = intensity * scenario_scale_multiplier + carbon_tax_cost + information_premium
    adjusted_annual = (
        adjusted_intensity * allocation_ton_safe
    ).replace([np.inf, -np.inf], np.nan).fillna(annual * scenario_scale_multiplier)

    return adjusted_intensity, adjusted_annual, "rowwise_real_cost_source_with_external_evidence_and_information_premium", frame


def _build_carbon_load(frame: pd.DataFrame, baseline_emission: pd.Series) -> pd.Series:
    explicit = _optional_numeric_column(frame, "pathway_emission_factor_kgco2e_per_metric_ton_scenario_proxy")
    fallback = baseline_emission - frame["planning_environment_intensity_kgco2e_per_ton"]
    is_surrogate_supported = (
        pd.to_numeric(frame.get("is_surrogate_supported", pd.Series([0.0] * len(frame), index=frame.index)), errors="coerce")
        .fillna(0.0)
        .astype(bool)
    )
    frame.loc[explicit.notna(), "planning_carbon_load_source"] = "explicit_pathway_emission_factor"
    fallback_mask = explicit.isna() & fallback.notna()
    supported_mask = fallback_mask & is_surrogate_supported
    limited_mask = fallback_mask & ~is_surrogate_supported
    if supported_mask.any():
        frame.loc[supported_mask, "planning_carbon_load_source"] = "baseline_minus_environment_benefit_surrogate_supported"
    if limited_mask.any():
        frame.loc[limited_mask, "planning_carbon_load_source"] = "baseline_minus_environment_benefit_evidence_limited"
    carbon_load = explicit.where(explicit.notna(), fallback)
    # Carbon load is defined as gross residual emissions only. Any carbon credit
    # or offset should be represented in a separate incentive or benefit term.
    return carbon_load.clip(lower=0.0)


def _build_data_quality_summary(
    *,
    evaluated_frame: pd.DataFrame,
    retained_frame: pd.DataFrame,
    excluded_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = evaluated_frame.groupby("scenario_name", dropna=False)
    for scenario_name, scenario_frame in grouped:
        retained = retained_frame[retained_frame["scenario_name"] == scenario_name]
        excluded = excluded_frame[excluded_frame["scenario_name"] == scenario_name]
        rows.append(
            {
                "scenario_name": scenario_name,
                "evaluated_candidate_count": int(len(scenario_frame)),
                "retained_candidate_count": int(len(retained)),
                "excluded_candidate_count": int(len(excluded)),
                "noncritical_imputation_candidate_count": int(
                    retained["planning_imputation_flag"].sum() if not retained.empty else 0
                ),
                "complete_candidate_count": int(
                    (~retained["planning_imputation_flag"]).sum() if not retained.empty else 0
                ),
                "candidate_retention_ratio": float(len(retained) / len(scenario_frame)) if len(scenario_frame) else 0.0,
                "excluded_candidate_ratio": float(len(excluded) / len(scenario_frame)) if len(scenario_frame) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)


def _required_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _optional_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _append_message(series: pd.Series, message: str) -> pd.Series:
    return series.astype(str).replace("nan", "").apply(
        lambda current: message if current == "" else f"{current};{message}"
    )


def _coalesce_with_flag(
    *,
    frame: pd.DataFrame,
    target_column: str,
    preferred: pd.Series,
    fallback: pd.Series,
    fallback_reason: str,
    source_column: str | None = None,
    preferred_source_label: str | None = None,
    fallback_source_label: str | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    result = preferred.where(preferred.notna(), fallback)
    if source_column:
        preferred_mask = preferred.notna()
        fallback_mask = preferred.isna() & fallback.notna()
        if preferred_source_label and preferred_mask.any():
            frame.loc[preferred_mask, source_column] = preferred_source_label
        if fallback_source_label and fallback_mask.any():
            frame.loc[fallback_mask, source_column] = fallback_source_label
    imputed_mask = preferred.isna() & fallback.notna()
    if imputed_mask.any():
        frame.loc[imputed_mask, "planning_imputation_notes"] = _append_message(
            frame.loc[imputed_mask, "planning_imputation_notes"],
            f"{target_column}:{fallback_reason}",
        )
    return result, frame


def _impute_with_flag(
    *,
    frame: pd.DataFrame,
    column: str,
    default_value: float,
    flag_reason: str,
    source_column: str | None = None,
    explicit_source_label: str | None = None,
    default_source_label: str | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    values = _optional_numeric_column(frame, column)
    missing_mask = values.isna()
    if source_column:
        explicit_mask = values.notna()
        if explicit_source_label and explicit_mask.any():
            frame.loc[explicit_mask, source_column] = explicit_source_label
        if default_source_label and missing_mask.any():
            frame.loc[missing_mask, source_column] = default_source_label
    if missing_mask.any():
        frame.loc[missing_mask, "planning_imputation_notes"] = _append_message(
            frame.loc[missing_mask, "planning_imputation_notes"],
            f"{column}:{flag_reason}",
        )
    return values.fillna(default_value), frame


def _adopt_series(
    *,
    frame: pd.DataFrame,
    target: pd.Series,
    candidate: pd.Series,
    source_column: str,
    source_label: str,
) -> tuple[pd.Series, pd.DataFrame]:
    adopt_mask = target.isna() & candidate.notna()
    if adopt_mask.any():
        frame.loc[adopt_mask, source_column] = source_label
    updated = target.where(target.notna(), candidate)
    return updated, frame


def _attach_surrogate_support_columns(frame: pd.DataFrame, config) -> pd.DataFrame:
    pathway = frame["pathway"].astype(str).str.strip().str.lower()
    surrogate_mode = frame.get("surrogate_mode", pd.Series([""] * len(frame), index=frame.index)).astype(str)
    support_level = np.select(
        [
            ~pathway.isin(tuple(SUPPORTED_SURROGATE_PATHWAYS)),
            surrogate_mode.eq("trained_surrogate"),
            surrogate_mode.eq("trained_surrogate_with_documented_fallback"),
            surrogate_mode.eq("documented_static_fallback"),
        ],
        [
            "unsupported_pathway",
            "surrogate_supported",
            "trained_surrogate_with_documented_fallback",
            "documented_static_fallback",
        ],
        default="documented_static_fallback",
    )
    weight_map = {
        "surrogate_supported": 1.0,
        "trained_surrogate_with_documented_fallback": float(config.partial_surrogate_weight),
        "documented_static_fallback": float(config.static_fallback_weight),
        "unsupported_pathway": float(config.unsupported_pathway_weight),
    }
    uncertainty_multiplier_map = {
        "surrogate_supported": 1.0,
        "trained_surrogate_with_documented_fallback": float(config.partial_surrogate_uncertainty_multiplier),
        "documented_static_fallback": float(config.static_fallback_uncertainty_multiplier),
        "unsupported_pathway": float(config.unsupported_pathway_uncertainty_multiplier),
    }
    premium_map = {
        "surrogate_supported": 0.0,
        "trained_surrogate_with_documented_fallback": float(config.partial_surrogate_information_premium_usd_per_ton),
        "documented_static_fallback": float(config.static_fallback_information_premium_usd_per_ton),
        "unsupported_pathway": float(config.unsupported_pathway_information_premium_usd_per_ton),
    }
    frame["surrogate_support_level"] = support_level
    frame["is_surrogate_supported"] = frame["surrogate_support_level"].eq("surrogate_supported")
    frame["evidence_based_weight"] = frame["surrogate_support_level"].map(weight_map).astype(float)
    frame["evidence_uncertainty_multiplier"] = frame["surrogate_support_level"].map(uncertainty_multiplier_map).astype(float)
    frame["information_deficit_premium_usd_per_ton"] = frame["surrogate_support_level"].map(premium_map).astype(float)
    return frame
