from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .artifacts import write_planning_outputs
from .constraints import build_scenario_constraints
from .inputs import load_planning_input_bundle
from .objectives import enrich_with_objectives


@dataclass(frozen=True)
class PlanningConfig:
    energy_weight: float = 0.40
    environment_weight: float = 0.35
    cost_weight: float = 0.25
    top_k_per_scenario: int = 5
    max_portfolio_candidates: int = 3
    max_candidate_share: float = 0.45
    max_subtype_share: float = 0.60
    min_distinct_subtypes: int = 2
    deployable_capacity_fraction: float = 0.85


def run_planning_baseline(
    dataset_path: str | None = None,
    output_dir: str | None = None,
    config: PlanningConfig | None = None,
) -> dict[str, object]:
    active_config = config or PlanningConfig()
    _validate_config(active_config)

    bundle = load_planning_input_bundle(dataset_path=dataset_path)
    execution = execute_planning_pipeline(bundle=bundle, config=active_config)
    outputs = write_planning_outputs(
        scored=execution["scored"],
        scenario_recommendations=execution["scenario_recommendations"],
        pareto_candidates=execution["pareto_candidates"],
        scenario_constraints=execution["scenario_constraints"],
        portfolio_allocations=execution["portfolio_allocations"],
        portfolio_summary=execution["portfolio_summary"],
        scenario_summary=execution["scenario_summary"],
        pathway_summary=execution["pathway_summary"],
        output_dir=output_dir,
        config=active_config,
        bundle=bundle,
        readiness=execution["objective_readiness"],
    )

    return {
        "dataset_path": str(bundle.dataset_path),
        "scenario_names": list(bundle.scenario_names),
        "pathways": list(bundle.pathways),
        "objective_readiness": execution["objective_readiness"],
        "planner_variant": "constraint_aware_weighted_portfolio",
        "row_count": int(len(execution["scored"])),
        "pareto_candidate_count": int(len(execution["pareto_candidates"])),
        "recommendation_count": int(len(execution["scenario_recommendations"])),
        "portfolio_allocation_count": int(len(execution["portfolio_allocations"])),
        "outputs": outputs,
    }


def execute_planning_pipeline(bundle, config: PlanningConfig) -> dict[str, object]:
    objective_frame, readiness = enrich_with_objectives(bundle)
    scenario_constraints = build_scenario_constraints(objective_frame, config)
    scored = score_cases(objective_frame, config)
    scenario_recommendations = build_scenario_recommendations(scored, config.top_k_per_scenario)
    pareto_candidates = build_pareto_candidates(scored)
    portfolio_allocations = build_scenario_portfolios(scored, scenario_constraints, config)
    portfolio_summary = build_portfolio_summary(portfolio_allocations, scenario_constraints)
    scenario_summary = build_scenario_summary(
        scored=scored,
        recommendations=scenario_recommendations,
        portfolio_summary=portfolio_summary,
    )
    pathway_summary = build_pathway_summary(
        scored=scored,
        portfolio_allocations=portfolio_allocations,
    )
    return {
        "objective_frame": objective_frame,
        "objective_readiness": readiness,
        "scenario_constraints": scenario_constraints,
        "scored": scored,
        "scenario_recommendations": scenario_recommendations,
        "pareto_candidates": pareto_candidates,
        "portfolio_allocations": portfolio_allocations,
        "portfolio_summary": portfolio_summary,
        "scenario_summary": scenario_summary,
        "pathway_summary": pathway_summary,
    }


def score_cases(frame: pd.DataFrame, config: PlanningConfig) -> pd.DataFrame:
    scored = frame.copy()
    scored["energy_utility"] = _normalize(scored["planning_energy_objective"])
    scored["environment_utility"] = _normalize(scored["planning_environment_objective"])
    scored["cost_utility"] = 1.0 - _normalize(scored["planning_cost_objective"])
    scored["planning_score"] = (
        config.energy_weight * scored["energy_utility"]
        + config.environment_weight * scored["environment_utility"]
        + config.cost_weight * scored["cost_utility"]
    )
    return scored.sort_values(
        ["scenario_name", "planning_score", "planning_energy_objective"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_scenario_recommendations(scored: pd.DataFrame, top_k: int) -> pd.DataFrame:
    ranked = scored.copy()
    ranked["scenario_rank"] = ranked.groupby("scenario_name")["planning_score"].rank(
        method="first", ascending=False
    )
    return ranked[ranked["scenario_rank"] <= top_k].reset_index(drop=True)


def build_pareto_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario_name, scenario_frame in scored.groupby("scenario_name", dropna=False):
        mask = _pareto_efficient_mask(
            scenario_frame["planning_energy_objective"],
            scenario_frame["planning_environment_objective"],
            scenario_frame["planning_cost_objective"],
        )
        subset = scenario_frame.loc[mask].copy()
        subset["pareto_scope"] = "scenario"
        subset["pareto_scenario_name"] = scenario_name
        rows.extend(subset.to_dict("records"))

    global_mask = _pareto_efficient_mask(
        scored["planning_energy_objective"],
        scored["planning_environment_objective"],
        scored["planning_cost_objective"],
    )
    global_subset = scored.loc[global_mask].copy()
    global_subset["pareto_scope"] = "global"
    global_subset["pareto_scenario_name"] = "all"
    rows.extend(global_subset.to_dict("records"))
    return pd.DataFrame(rows).drop_duplicates(subset=["optimization_case_id", "pareto_scope"]).reset_index(
        drop=True
    )


def build_scenario_portfolios(
    scored: pd.DataFrame,
    scenario_constraints: pd.DataFrame,
    config: PlanningConfig,
) -> pd.DataFrame:
    constraint_map = scenario_constraints.set_index("scenario_name").to_dict("index")
    rows: list[dict[str, object]] = []

    for scenario_name, scenario_frame in scored.groupby("scenario_name", dropna=False):
        constraint = constraint_map.get(scenario_name, {})
        selected = _select_portfolio_candidates(scenario_frame, config)
        if selected.empty:
            continue

        allocated = _allocate_budget_across_selected(selected, constraint, config)
        if allocated.empty:
            continue

        rows.extend(allocated.to_dict("records"))

    if not rows:
        return pd.DataFrame(
            columns=[
                "scenario_name",
                "optimization_case_id",
                "allocated_feed_ton_per_year",
                "allocated_feed_share",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["scenario_name", "portfolio_rank", "planning_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_portfolio_summary(
    portfolio_allocations: pd.DataFrame,
    scenario_constraints: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    allocation_map = {
        name: frame.copy()
        for name, frame in portfolio_allocations.groupby("scenario_name", dropna=False)
    }

    for _, constraint in scenario_constraints.iterrows():
        scenario_name = constraint["scenario_name"]
        allocation_frame = allocation_map.get(scenario_name, pd.DataFrame())
        allocated_feed = _sum_column(allocation_frame, "allocated_feed_ton_per_year")
        effective_budget = float(constraint["effective_processing_budget_ton_per_year"])
        feed_budget = float(constraint["scenario_feed_budget_ton_per_year"])

        rows.append(
            {
                "scenario_name": scenario_name,
                "selected_candidate_count": int(len(allocation_frame)),
                "distinct_manure_subtypes": int(
                    allocation_frame["manure_subtype"].dropna().astype(str).nunique()
                    if "manure_subtype" in allocation_frame.columns
                    else 0
                ),
                "allocated_feed_ton_per_year": allocated_feed,
                "effective_processing_budget_ton_per_year": effective_budget,
                "scenario_feed_budget_ton_per_year": feed_budget,
                "portfolio_fill_ratio": _safe_ratio(allocated_feed, effective_budget),
                "scenario_feed_coverage_ratio": _safe_ratio(allocated_feed, feed_budget),
                "unallocated_processing_budget_ton_per_year": max(effective_budget - allocated_feed, 0.0),
                "remaining_unmet_feed_ton_per_year": max(feed_budget - allocated_feed, 0.0),
                "portfolio_energy_objective": _sum_column(allocation_frame, "allocated_energy_objective"),
                "portfolio_environment_objective": _sum_column(
                    allocation_frame, "allocated_environment_objective"
                ),
                "portfolio_cost_objective": _sum_column(allocation_frame, "allocated_cost_objective"),
                "portfolio_score_mass": _sum_column(allocation_frame, "allocated_score_mass"),
                "top_portfolio_case_id": _first_value(allocation_frame, "optimization_case_id"),
                "top_portfolio_manure_subtype": _first_value(allocation_frame, "manure_subtype"),
                "top_portfolio_temperature_c": _first_value(allocation_frame, "process_temperature_c"),
                "top_portfolio_residence_time_min": _first_value(allocation_frame, "residence_time_min"),
            }
        )

    return pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)


def build_scenario_summary(
    *,
    scored: pd.DataFrame,
    recommendations: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
) -> pd.DataFrame:
    top1 = recommendations[recommendations["scenario_rank"] == 1].copy()
    top1_columns = [
        "scenario_name",
        "optimization_case_id",
        "planning_score",
        "planning_energy_objective",
        "planning_environment_objective",
        "planning_cost_objective",
        "blend_manure_ratio",
        "blend_wet_waste_ratio",
        "process_temperature_c",
        "residence_time_min",
        "manure_subtype",
    ]
    top1 = top1[[column for column in top1_columns if column in top1.columns]].rename(
        columns={
            "optimization_case_id": "top_ranked_case_id",
            "planning_score": "top_ranked_case_score",
            "planning_energy_objective": "top_ranked_energy_objective",
            "planning_environment_objective": "top_ranked_environment_objective",
            "planning_cost_objective": "top_ranked_cost_objective",
            "process_temperature_c": "top_ranked_temperature_c",
            "residence_time_min": "top_ranked_residence_time_min",
            "manure_subtype": "top_ranked_manure_subtype",
        }
    )
    counts = scored.groupby("scenario_name").size().reset_index(name="candidate_count")
    return (
        counts.merge(top1, on="scenario_name", how="left")
        .merge(portfolio_summary, on="scenario_name", how="left")
        .sort_values("scenario_name")
        .reset_index(drop=True)
    )


def build_pathway_summary(
    *,
    scored: pd.DataFrame,
    portfolio_allocations: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    allocation_map = {
        (scenario_name, pathway): frame.copy()
        for (scenario_name, pathway), frame in portfolio_allocations.groupby(
            ["scenario_name", "pathway"], dropna=False
        )
    }

    for (scenario_name, pathway), scenario_pathway_frame in scored.groupby(
        ["scenario_name", "pathway"], dropna=False
    ):
        best_case = scenario_pathway_frame.sort_values(
            ["planning_score", "planning_energy_objective"],
            ascending=[False, False],
        ).iloc[0]
        allocation_frame = allocation_map.get((scenario_name, pathway), pd.DataFrame())
        rows.append(
            {
                "scenario_name": scenario_name,
                "pathway": pathway,
                "candidate_count": int(len(scenario_pathway_frame)),
                "best_case_id": best_case["optimization_case_id"],
                "best_case_score": float(best_case["planning_score"]),
                "best_case_energy_objective": float(best_case["planning_energy_objective"]),
                "best_case_environment_objective": float(best_case["planning_environment_objective"]),
                "best_case_cost_objective": float(best_case["planning_cost_objective"]),
                "best_case_manure_subtype": best_case.get("manure_subtype"),
                "best_case_blend_manure_ratio": best_case.get("blend_manure_ratio"),
                "best_case_blend_wet_waste_ratio": best_case.get("blend_wet_waste_ratio"),
                "portfolio_selected_count": int(len(allocation_frame)),
                "portfolio_allocated_feed_ton_per_year": _sum_column(
                    allocation_frame, "allocated_feed_ton_per_year"
                ),
                "portfolio_allocated_feed_share": _sum_column(allocation_frame, "allocated_feed_share"),
                "portfolio_energy_objective": _sum_column(allocation_frame, "allocated_energy_objective"),
                "portfolio_environment_objective": _sum_column(
                    allocation_frame, "allocated_environment_objective"
                ),
                "portfolio_cost_objective": _sum_column(allocation_frame, "allocated_cost_objective"),
                "portfolio_top_case_id": _first_value(allocation_frame, "optimization_case_id"),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["scenario_name", "best_case_score"], ascending=[True, False]
    ).reset_index(drop=True)


def _select_portfolio_candidates(scenario_frame: pd.DataFrame, config: PlanningConfig) -> pd.DataFrame:
    ordered = scenario_frame.sort_values(
        ["planning_score", "planning_energy_objective"],
        ascending=[False, False],
    ).copy()

    selected_indices: list[int] = []
    selection_stage: dict[int, str] = {}
    subtype_keys: set[str] = set()
    target_diversity = min(config.min_distinct_subtypes, config.max_portfolio_candidates)

    for idx, row in ordered.iterrows():
        subtype_key = _candidate_subtype_key(row)
        if len(subtype_keys) >= target_diversity:
            break
        if subtype_key in subtype_keys:
            continue
        selected_indices.append(idx)
        selection_stage[idx] = "diversity_pass"
        subtype_keys.add(subtype_key)

    for idx, _ in ordered.iterrows():
        if len(selected_indices) >= config.max_portfolio_candidates:
            break
        if idx in selection_stage:
            continue
        selected_indices.append(idx)
        selection_stage[idx] = "score_pass"

    selected = ordered.loc[selected_indices].copy()
    selected["portfolio_selection_stage"] = [selection_stage[idx] for idx in selected.index]
    return selected.reset_index(drop=True)


def _allocate_budget_across_selected(
    selected: pd.DataFrame,
    constraint: dict[str, object],
    config: PlanningConfig,
) -> pd.DataFrame:
    total_budget = float(constraint.get("effective_processing_budget_ton_per_year", 0.0) or 0.0)
    if total_budget <= 0.0:
        return pd.DataFrame()

    candidate_cap = total_budget * config.max_candidate_share
    subtype_cap = total_budget * config.max_subtype_share
    allocation = pd.Series(0.0, index=selected.index, dtype=float)
    subtype_allocation: dict[str, float] = {}
    weights = _coerce_positive_weights(selected["planning_score"])

    for _ in range(max(1, len(selected) * 12)):
        remaining = total_budget - float(allocation.sum())
        if remaining <= 1e-6:
            break

        active_indices: list[int] = []
        for idx, row in selected.iterrows():
            subtype_key = _candidate_subtype_key(row)
            if allocation.loc[idx] + 1e-9 >= candidate_cap:
                continue
            if subtype_allocation.get(subtype_key, 0.0) + 1e-9 >= subtype_cap:
                continue
            active_indices.append(idx)

        if not active_indices:
            break

        active_weights = _coerce_positive_weights(weights.loc[active_indices])
        weight_sum = float(active_weights.sum())
        progress = 0.0

        for idx in active_indices:
            row = selected.loc[idx]
            subtype_key = _candidate_subtype_key(row)
            desired = remaining * float(active_weights.loc[idx] / weight_sum)
            remaining_candidate_cap = candidate_cap - float(allocation.loc[idx])
            remaining_subtype_cap = subtype_cap - float(subtype_allocation.get(subtype_key, 0.0))
            delta = min(desired, remaining_candidate_cap, remaining_subtype_cap)
            if delta <= 1e-9:
                continue
            allocation.loc[idx] += delta
            subtype_allocation[subtype_key] = float(subtype_allocation.get(subtype_key, 0.0) + delta)
            progress += delta

        if progress <= 1e-9:
            break

    result = selected.copy()
    result["allocated_feed_ton_per_year"] = allocation
    result["allocated_feed_share"] = result["allocated_feed_ton_per_year"] / total_budget
    result["candidate_capacity_cap_ton_per_year"] = candidate_cap
    result["subtype_capacity_cap_ton_per_year"] = subtype_cap
    result["scenario_feed_budget_ton_per_year"] = float(constraint.get("scenario_feed_budget_ton_per_year", 0.0))
    result["effective_processing_budget_ton_per_year"] = total_budget
    result["allocated_energy_objective"] = (
        result["allocated_feed_ton_per_year"] * result["planning_energy_intensity_mj_per_ton"]
    )
    result["allocated_environment_objective"] = (
        result["allocated_feed_ton_per_year"] * result["planning_environment_intensity_kgco2e_per_ton"]
    )
    result["allocated_cost_objective"] = (
        result["allocated_feed_ton_per_year"] * result["planning_cost_intensity_proxy_or_real_per_ton"]
    )
    result["allocated_score_mass"] = result["allocated_feed_ton_per_year"] * result["planning_score"]
    result = result[result["allocated_feed_ton_per_year"] > 1e-6].copy()
    result = result.sort_values(
        ["allocated_feed_ton_per_year", "planning_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    result["portfolio_rank"] = range(1, len(result) + 1)
    return result


def _coerce_positive_weights(series: pd.Series) -> pd.Series:
    weights = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    if float(weights.sum()) > 0.0:
        return weights
    return pd.Series(1.0, index=series.index, dtype=float)


def _candidate_subtype_key(row: pd.Series) -> str:
    manure_subtype = str(row.get("manure_subtype", "") or "").strip()
    if manure_subtype:
        return manure_subtype
    return str(row.get("sample_id", "unknown_candidate"))


def _sum_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _first_value(frame: pd.DataFrame, column: str) -> object:
    if frame.empty or column not in frame.columns:
        return None
    return frame.iloc[0][column]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return pd.Series(0.0, index=values.index)
    return (values - minimum) / (maximum - minimum)


def _pareto_efficient_mask(
    energy: pd.Series,
    environment: pd.Series,
    cost: pd.Series,
) -> pd.Series:
    frame = pd.DataFrame(
        {
            "energy": pd.to_numeric(energy, errors="coerce").fillna(0.0),
            "environment": pd.to_numeric(environment, errors="coerce").fillna(0.0),
            "cost": pd.to_numeric(cost, errors="coerce").fillna(0.0),
        }
    ).reset_index(drop=True)

    efficient = []
    for _, row in frame.iterrows():
        dominated = (
            (frame["energy"] >= row["energy"])
            & (frame["environment"] >= row["environment"])
            & (frame["cost"] <= row["cost"])
            & (
                (frame["energy"] > row["energy"])
                | (frame["environment"] > row["environment"])
                | (frame["cost"] < row["cost"])
            )
        )
        efficient.append(not bool(dominated.any()))
    return pd.Series(efficient, index=energy.index)


def _validate_config(config: PlanningConfig) -> None:
    weights = [config.energy_weight, config.environment_weight, config.cost_weight]
    if any(weight < 0.0 for weight in weights):
        raise ValueError("Planning weights must be non-negative.")
    if sum(weights) <= 0.0:
        raise ValueError("At least one planning weight must be positive.")
    if config.top_k_per_scenario <= 0:
        raise ValueError("top_k_per_scenario must be positive.")
    if config.max_portfolio_candidates <= 0:
        raise ValueError("max_portfolio_candidates must be positive.")
    if not 0.0 < config.max_candidate_share <= 1.0:
        raise ValueError("max_candidate_share must be within (0, 1].")
    if not 0.0 < config.max_subtype_share <= 1.0:
        raise ValueError("max_subtype_share must be within (0, 1].")
    if config.max_subtype_share + 1e-9 < config.max_candidate_share:
        raise ValueError("max_subtype_share must be greater than or equal to max_candidate_share.")
    if config.min_distinct_subtypes < 1:
        raise ValueError("min_distinct_subtypes must be at least 1.")
    if not 0.0 < config.deployable_capacity_fraction <= 1.0:
        raise ValueError("deployable_capacity_fraction must be within (0, 1].")
