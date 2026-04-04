from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .solve import PlanningConfig


def build_scenario_constraints(frame: pd.DataFrame, config: "PlanningConfig") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario_name, scenario_frame in frame.groupby("scenario_name", dropna=False):
        anchor = scenario_frame.iloc[0]
        feed_budget = _value(anchor, "scenario_wet_waste_feed_allocation_ton_per_year_proxy")
        available_capacity = _value(anchor, "facility_total_available_capacity_ton_per_year_reference")
        permitted_capacity = _value(anchor, "facility_total_permitted_capacity_ton_per_year_reference")
        required_new_capacity = _value(anchor, "organic_waste_recycling_capacity_needed_ton_per_year_reference")

        deployable_new_capacity = required_new_capacity * config.deployable_capacity_fraction
        positive_bounds = [value for value in [feed_budget, deployable_new_capacity, available_capacity] if value > 0.0]
        effective_budget = min(positive_bounds) if positive_bounds else 0.0
        binding_reason = _binding_reason(
            feed_budget=feed_budget,
            effective_budget=effective_budget,
            deployable_new_capacity=deployable_new_capacity,
            available_capacity=available_capacity,
        )

        rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_feed_budget_ton_per_year": feed_budget,
                "facility_total_available_capacity_ton_per_year": available_capacity,
                "facility_total_permitted_capacity_ton_per_year": permitted_capacity,
                "required_new_capacity_ton_per_year": required_new_capacity,
                "deployable_new_capacity_ton_per_year": deployable_new_capacity,
                "effective_processing_budget_ton_per_year": effective_budget,
                "unmet_feed_before_portfolio_ton_per_year": max(feed_budget - effective_budget, 0.0),
                "pre_portfolio_feed_coverage_ratio": _safe_ratio(effective_budget, feed_budget),
                "capacity_constraint_binding": effective_budget + 1e-9 < feed_budget,
                "capacity_binding_reason": binding_reason,
                "candidate_share_cap_ton_per_year": effective_budget * config.max_candidate_share,
                "subtype_share_cap_ton_per_year": effective_budget * config.max_subtype_share,
                "max_portfolio_candidates": int(config.max_portfolio_candidates),
                "min_distinct_subtypes": int(min(config.min_distinct_subtypes, config.max_portfolio_candidates)),
                "deployable_capacity_fraction": float(config.deployable_capacity_fraction),
            }
        )

    return pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)


def _binding_reason(
    *,
    feed_budget: float,
    effective_budget: float,
    deployable_new_capacity: float,
    available_capacity: float,
) -> str:
    if feed_budget <= 0.0:
        return "no_feed_budget"
    if effective_budget + 1e-9 >= feed_budget:
        return "feed_budget"
    if deployable_new_capacity > 0.0 and abs(effective_budget - deployable_new_capacity) <= 1e-6:
        return "deployable_new_capacity"
    if available_capacity > 0.0 and abs(effective_budget - available_capacity) <= 1e-6:
        return "available_facility_capacity"
    return "unknown_constraint"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _value(row: pd.Series, column: str) -> float:
    if column not in row.index:
        return 0.0
    return float(pd.to_numeric(pd.Series([row[column]]), errors="coerce").fillna(0.0).iloc[0])
