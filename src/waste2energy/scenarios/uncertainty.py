from __future__ import annotations

import pandas as pd


def build_uncertainty_summary(
    *,
    stress_test_summary: pd.DataFrame,
    decision_stability: pd.DataFrame,
) -> pd.DataFrame:
    if stress_test_summary.empty:
        return pd.DataFrame(columns=["scenario_name", "stress_test_count"])

    objective_summary = (
        stress_test_summary.groupby("scenario_name", dropna=False)
        .agg(
            stress_test_count=("stress_test_name", "nunique"),
            top_case_switch_count=("top_portfolio_case_id", "nunique"),
            energy_min=("portfolio_energy_objective", "min"),
            energy_mean=("portfolio_energy_objective", "mean"),
            energy_max=("portfolio_energy_objective", "max"),
            environment_min=("portfolio_environment_objective", "min"),
            environment_mean=("portfolio_environment_objective", "mean"),
            environment_max=("portfolio_environment_objective", "max"),
            cost_min=("portfolio_cost_objective", "min"),
            cost_mean=("portfolio_cost_objective", "mean"),
            cost_max=("portfolio_cost_objective", "max"),
            coverage_min=("scenario_feed_coverage_ratio", "min"),
            coverage_mean=("scenario_feed_coverage_ratio", "mean"),
            coverage_max=("scenario_feed_coverage_ratio", "max"),
            unmet_feed_max=("remaining_unmet_feed_ton_per_year", "max"),
        )
        .reset_index()
    )

    for prefix in ["energy", "environment", "cost", "coverage"]:
        objective_summary[f"{prefix}_range"] = (
            objective_summary[f"{prefix}_max"] - objective_summary[f"{prefix}_min"]
        )
        objective_summary[f"{prefix}_range_ratio"] = _safe_ratio_series(
            objective_summary[f"{prefix}_range"],
            objective_summary[f"{prefix}_mean"],
        )

    if decision_stability.empty:
        objective_summary["stable_candidate_count"] = 0
        objective_summary["dominant_sample_id"] = ""
        objective_summary["dominant_selection_rate"] = pd.NA
        return objective_summary.sort_values("scenario_name").reset_index(drop=True)

    stable_counts = (
        decision_stability.groupby("scenario_name", dropna=False)["stable_under_majority_rule"]
        .sum()
        .rename("stable_candidate_count")
        .reset_index()
    )
    dominant = (
        decision_stability.sort_values(
            ["scenario_name", "selection_rate", "avg_portfolio_rank"],
            ascending=[True, False, True],
        )
        .groupby("scenario_name", dropna=False)
        .head(1)[["scenario_name", "sample_id", "selection_rate"]]
        .rename(
            columns={
                "sample_id": "dominant_sample_id",
                "selection_rate": "dominant_selection_rate",
            }
        )
    )
    return (
        objective_summary.merge(stable_counts, on="scenario_name", how="left")
        .merge(dominant, on="scenario_name", how="left")
        .fillna({"stable_candidate_count": 0, "dominant_sample_id": ""})
        .sort_values("scenario_name")
        .reset_index(drop=True)
    )


def _safe_ratio_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = denominator.replace(0.0, pd.NA)
    return numerator / safe_denominator
