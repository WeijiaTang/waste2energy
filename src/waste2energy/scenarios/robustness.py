from __future__ import annotations

import pandas as pd


def build_stress_test_summary(
    *,
    stress_registry: pd.DataFrame,
    scenario_constraints: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
) -> pd.DataFrame:
    constraint_columns = [
        "scenario_name",
        "stress_test_name",
        "stress_test_description",
        "capacity_constraint_binding",
        "capacity_binding_reason",
        "scenario_feed_budget_ton_per_year",
        "effective_processing_budget_ton_per_year",
        "candidate_share_cap_ton_per_year",
        "subtype_share_cap_ton_per_year",
        "unmet_feed_before_portfolio_ton_per_year",
        "pre_portfolio_feed_coverage_ratio",
    ]
    merged = portfolio_summary.merge(
        scenario_constraints,
        on=["scenario_name", "stress_test_name", "stress_test_description"],
        how="left",
        suffixes=("", "_constraint"),
    )
    selected = merged[
        [
            "scenario_name",
            "stress_test_name",
            "stress_test_description",
            "selected_candidate_count",
            "distinct_manure_subtypes",
            "allocated_feed_ton_per_year",
            "scenario_feed_coverage_ratio",
            "remaining_unmet_feed_ton_per_year",
            "portfolio_energy_objective",
            "portfolio_environment_objective",
            "portfolio_cost_objective",
            "portfolio_score_mass",
            "top_portfolio_case_id",
            "top_portfolio_manure_subtype",
            "top_portfolio_temperature_c",
            "top_portfolio_residence_time_min",
            *[column for column in constraint_columns if column not in {"scenario_name", "stress_test_name", "stress_test_description"}],
        ]
    ]
    registry_columns = [
        "stress_test_name",
        "energy_weight",
        "environment_weight",
        "cost_weight",
        "top_k_per_scenario",
        "max_portfolio_candidates",
        "max_candidate_share",
        "max_subtype_share",
        "min_distinct_subtypes",
        "deployable_capacity_fraction",
    ]
    return selected.merge(
        stress_registry[registry_columns],
        on="stress_test_name",
        how="left",
    ).sort_values(["scenario_name", "stress_test_name"]).reset_index(drop=True)


def build_decision_stability(portfolio_allocations: pd.DataFrame) -> pd.DataFrame:
    if portfolio_allocations.empty:
        return pd.DataFrame(
            columns=[
                "scenario_name",
                "sample_id",
                "selection_rate",
            ]
        )

    total_runs = (
        portfolio_allocations[["scenario_name", "stress_test_name"]]
        .drop_duplicates()
        .groupby("scenario_name")
        .size()
        .rename("total_stress_runs")
        .reset_index()
    )

    grouped = (
        portfolio_allocations.groupby(["scenario_name", "sample_id"], dropna=False)
        .agg(
            optimization_case_count=("optimization_case_id", "nunique"),
            stress_run_count=("stress_test_name", "nunique"),
            avg_portfolio_rank=("portfolio_rank", "mean"),
            best_portfolio_rank=("portfolio_rank", "min"),
            worst_portfolio_rank=("portfolio_rank", "max"),
            avg_allocated_feed_share=("allocated_feed_share", "mean"),
            max_allocated_feed_share=("allocated_feed_share", "max"),
            avg_planning_score=("planning_score", "mean"),
            top_rank_count=("portfolio_rank", lambda series: int((series == 1).sum())),
            manure_subtype=("manure_subtype", lambda series: _first_non_empty(series)),
            representative_case_id=("optimization_case_id", lambda series: _first_non_empty(series)),
            stress_tests_selected=("stress_test_name", lambda series: "|".join(sorted(set(series.astype(str))))),
        )
        .reset_index()
    )

    merged = grouped.merge(total_runs, on="scenario_name", how="left")
    merged["selection_rate"] = merged["stress_run_count"] / merged["total_stress_runs"]
    merged["stable_under_majority_rule"] = merged["selection_rate"] >= 0.5
    merged["stable_under_consensus_rule"] = merged["selection_rate"] >= 0.999999
    return merged.sort_values(
        ["scenario_name", "selection_rate", "avg_portfolio_rank"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_cross_scenario_stability(portfolio_allocations: pd.DataFrame) -> pd.DataFrame:
    baseline = portfolio_allocations[portfolio_allocations["stress_test_name"] == "baseline"].copy()
    if baseline.empty:
        return pd.DataFrame(columns=["sample_id", "selected_scenario_count"])

    total_scenarios = baseline["scenario_name"].nunique()
    summary = (
        baseline.groupby("sample_id", dropna=False)
        .agg(
            selected_scenario_count=("scenario_name", "nunique"),
            scenario_names=("scenario_name", lambda series: "|".join(sorted(set(series.astype(str))))),
            avg_allocated_feed_share=("allocated_feed_share", "mean"),
            avg_portfolio_rank=("portfolio_rank", "mean"),
            manure_subtype=("manure_subtype", lambda series: _first_non_empty(series)),
            representative_case_id=("optimization_case_id", lambda series: _first_non_empty(series)),
        )
        .reset_index()
    )
    summary["total_scenarios"] = total_scenarios
    summary["cross_scenario_selection_rate"] = summary["selected_scenario_count"] / total_scenarios
    summary["selected_in_all_scenarios"] = summary["selected_scenario_count"] == total_scenarios
    return summary.sort_values(
        ["cross_scenario_selection_rate", "avg_portfolio_rank"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _first_non_empty(series: pd.Series) -> str:
    values = [str(value) for value in series if str(value).strip() and str(value).strip().lower() != "nan"]
    return values[0] if values else ""
