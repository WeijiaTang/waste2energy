# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
except Exception:  # pragma: no cover - dependency fallback
    Bounds = None
    LinearConstraint = None
    milp = None

try:  # pragma: no cover - optional dependency
    import pyomo.environ as pyo
except Exception:  # pragma: no cover - dependency fallback
    pyo = None

if TYPE_CHECKING:
    from .solve import PlanningConfig


@dataclass(frozen=True)
class ScenarioOptimizationResult:
    scenario_name: str
    allocations: pd.DataFrame
    diagnostics: dict[str, object]


def build_candidate_score_frame(
    scenario_frame: pd.DataFrame,
    config: "PlanningConfig",
) -> pd.DataFrame:
    scored = _apply_scenario_external_evidence(scenario_frame.copy(), config)
    if "effective_uncertainty_ratio" not in scored.columns:
        scored["effective_uncertainty_ratio"] = pd.to_numeric(
            scored.get("combined_uncertainty_ratio", pd.Series([0.0] * len(scored), index=scored.index)),
            errors="coerce",
        ).fillna(0.0)
    if "evidence_based_weight" not in scored.columns:
        scored["evidence_based_weight"] = 1.0
    _require_numeric_columns(
        scored,
        columns=(
            "planning_energy_intensity_mj_per_ton",
            "planning_environment_intensity_kgco2e_per_ton",
            "planning_cost_intensity_proxy_or_real_per_ton",
            "planning_carbon_load_kgco2e_per_ton",
            "combined_uncertainty_ratio",
            "effective_uncertainty_ratio",
            "evidence_based_weight",
        ),
        context="candidate score construction",
    )
    scored["energy_utility"] = _normalize(scored["planning_energy_intensity_mj_per_ton"])
    scored["environment_utility"] = _normalize(scored["planning_environment_intensity_kgco2e_per_ton"])
    scored["cost_utility"] = 1.0 - _normalize(scored["planning_cost_intensity_proxy_or_real_per_ton"])
    scored["robustness_utility"] = 1.0 - _normalize(scored["effective_uncertainty_ratio"])
    scored["evidence_utility"] = _normalize(scored["evidence_based_weight"])
    scored["weighted_score_per_ton"] = (
        config.energy_weight * scored["energy_utility"]
        + config.environment_weight * scored["environment_utility"]
        + config.cost_weight * scored["cost_utility"]
        + config.robustness_factor * scored["robustness_utility"]
        + 0.15 * scored["evidence_utility"]
    )
    scored["planning_score"] = scored["weighted_score_per_ton"] * scored["evidence_based_weight"]
    scored["planning_score_scope"] = "scenario_local_optimizer"
    return scored.sort_values(
        ["planning_score", "planning_energy_intensity_mj_per_ton"],
        ascending=[False, False],
    ).reset_index(drop=True)


def solve_scenario_optimization(
    scenario_frame: pd.DataFrame,
    scenario_constraint: dict[str, object],
    config: "PlanningConfig",
) -> ScenarioOptimizationResult:
    scored = build_candidate_score_frame(scenario_frame, config)
    if scored.empty:
        return ScenarioOptimizationResult(
            scenario_name=str(scenario_constraint.get("scenario_name", "")),
            allocations=pd.DataFrame(),
            diagnostics={
                "solver_status": "empty_candidate_set",
                **_constraint_diagnostics(scored, scenario_constraint, config, pd.DataFrame()),
            },
        )

    pyomo_attempt_diagnostics: dict[str, object] = {}
    if config.optimization_method in {"auto", "pyomo"}:
        pyomo_result = _solve_with_pyomo_if_available(scored, scenario_constraint, config)
        if pyomo_result is not None and not pyomo_result.allocations.empty:
            return pyomo_result
        if pyomo_result is not None:
            pyomo_attempt_diagnostics = _prefix_diagnostics(pyomo_result.diagnostics, "pyomo_attempt_")

    scipy_result = _solve_with_scipy_milp(scored, scenario_constraint, config)
    if not pyomo_attempt_diagnostics:
        return scipy_result
    return ScenarioOptimizationResult(
        scenario_name=scipy_result.scenario_name,
        allocations=scipy_result.allocations,
        diagnostics={**pyomo_attempt_diagnostics, **scipy_result.diagnostics},
    )


def generate_pareto_front(
    scenario_frame: pd.DataFrame,
    scenario_constraint: dict[str, object],
    config: "PlanningConfig",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    weight_grid = _build_weight_grid(point_count=max(config.pareto_point_count, 6))

    for energy_weight, environment_weight, cost_weight in weight_grid:
        variant = config.copy_with_weights(
            energy_weight=energy_weight,
            environment_weight=environment_weight,
            cost_weight=cost_weight,
        )
        result = solve_scenario_optimization(scenario_frame, scenario_constraint, variant)
        allocations = result.allocations.copy()
        if allocations.empty:
            continue
        rows.append(
            {
                "scenario_name": str(scenario_constraint.get("scenario_name", "")),
                "pareto_energy_weight": energy_weight,
                "pareto_environment_weight": environment_weight,
                "pareto_cost_weight": cost_weight,
                "pareto_scope": "portfolio",
                "selected_case_ids": "|".join(allocations["optimization_case_id"].astype(str).tolist()),
                "portfolio_energy_objective": float(allocations["allocated_energy_objective"].sum()),
                "portfolio_environment_objective": float(allocations["allocated_environment_objective"].sum()),
                "portfolio_cost_objective": float(allocations["allocated_cost_objective"].sum()),
                "portfolio_score_mass": float(allocations["allocated_score_mass"].sum()),
                "portfolio_carbon_load_kgco2e": float(
                    allocations["allocated_carbon_load_kgco2e"].sum()
                    if "allocated_carbon_load_kgco2e" in allocations.columns
                    else 0.0
                ),
                "solver_status": result.diagnostics.get("solver_status", "unknown"),
            }
        )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    mask = _pareto_efficient_mask(
        frame["portfolio_energy_objective"],
        frame["portfolio_environment_objective"],
        frame["portfolio_cost_objective"],
    )
    return frame.loc[mask].drop_duplicates(subset=["scenario_name", "selected_case_ids"]).reset_index(drop=True)


def _solve_with_pyomo_if_available(
    scored: pd.DataFrame,
    scenario_constraint: dict[str, object],
    config: "PlanningConfig",
) -> ScenarioOptimizationResult | None:
    if pyo is None:
        return ScenarioOptimizationResult(
            scenario_name=str(scenario_constraint.get("scenario_name", "")),
            allocations=pd.DataFrame(),
            diagnostics={
                "solver_status": "pyomo_not_installed",
                "solver_backend": "pyomo",
            },
        )

    try:  # pragma: no cover - pyomo path depends on optional solver
        model = build_pyomo_model(scored, scenario_constraint, config)
        if config.pyomo_solver_preference == "auto":
            solver_candidates = ("appsi_highs", "highs", "glpk", "cbc")
        else:
            solver_candidates = (config.pyomo_solver_preference,)
        for solver_name in solver_candidates:
            solver = pyo.SolverFactory(solver_name)
            if solver is None or not solver.available(False):
                continue
            solved = solver.solve(model)
            status = str(solved.solver.termination_condition)
            solver_status = str(solved.solver.status)
            if status.lower() not in {"optimal", "feasible"}:
                continue
            allocations = _extract_pyomo_allocations(scored, model, scenario_constraint)
            if allocations.empty:
                continue
            return ScenarioOptimizationResult(
                scenario_name=str(scenario_constraint.get("scenario_name", "")),
                allocations=allocations,
                diagnostics={
                    "solver_status": status,
                    "solver_backend": f"pyomo:{solver_name}",
                    "solver_native_status": solver_status,
                    **_constraint_diagnostics(scored, scenario_constraint, config, allocations),
                },
            )
    except Exception as exc:
        return ScenarioOptimizationResult(
            scenario_name=str(scenario_constraint.get("scenario_name", "")),
            allocations=pd.DataFrame(),
            diagnostics={
                "solver_status": "pyomo_exception",
                "solver_backend": "pyomo",
                "solver_error": str(exc),
                **_constraint_diagnostics(scored, scenario_constraint, config, pd.DataFrame()),
            },
        )
    return ScenarioOptimizationResult(
        scenario_name=str(scenario_constraint.get("scenario_name", "")),
        allocations=pd.DataFrame(),
        diagnostics={
            "solver_status": "pyomo_solver_unavailable",
            "solver_backend": "pyomo",
            "solver_candidates": "|".join(solver_candidates),
            **_constraint_diagnostics(scored, scenario_constraint, config, pd.DataFrame()),
        },
    )


def build_pyomo_model(
    scored: pd.DataFrame,
    scenario_constraint: dict[str, object],
    config: "PlanningConfig",
):  # pragma: no cover - optional solver construction
    if pyo is None:
        raise RuntimeError("pyomo is not installed.")

    subtype_groups = _subtype_groups(scored)
    effective_budget = _required_constraint_value(scenario_constraint, "effective_processing_budget_ton_per_year")
    candidate_cap = _required_constraint_value(scenario_constraint, "candidate_share_cap_ton_per_year")
    subtype_cap = _required_constraint_value(scenario_constraint, "subtype_share_cap_ton_per_year")
    carbon_budget = _carbon_budget(scored, scenario_constraint, config)

    _require_numeric_columns(
        scored,
        columns=(
            "weighted_score_per_ton",
            "planning_energy_intensity_mj_per_ton",
            "planning_environment_intensity_kgco2e_per_ton",
            "planning_cost_intensity_proxy_or_real_per_ton",
            "planning_carbon_load_kgco2e_per_ton",
        ),
        context="pyomo model assembly",
    )

    model = pyo.ConcreteModel()
    model.CASES = pyo.RangeSet(0, len(scored) - 1)
    model.SUBTYPES = pyo.Set(initialize=list(subtype_groups))
    model.x = pyo.Var(model.CASES, domain=pyo.NonNegativeReals)
    model.y = pyo.Var(model.CASES, domain=pyo.Binary)
    model.z = pyo.Var(model.SUBTYPES, domain=pyo.Binary)
    model.total_energy = pyo.Var(domain=pyo.NonNegativeReals)
    model.total_environment = pyo.Var(domain=pyo.NonNegativeReals)
    model.total_cost = pyo.Var(domain=pyo.NonNegativeReals)
    model.total_carbon_load = pyo.Var(domain=pyo.NonNegativeReals)

    score_coeff = scored["weighted_score_per_ton"].tolist()
    energy_coeff = scored["planning_energy_intensity_mj_per_ton"].tolist()
    environment_coeff = scored["planning_environment_intensity_kgco2e_per_ton"].tolist()
    cost_coeff = scored["planning_cost_intensity_proxy_or_real_per_ton"].tolist()
    carbon_coeff = scored["planning_carbon_load_kgco2e_per_ton"].tolist()

    model.objective = pyo.Objective(
        expr=sum(score_coeff[i] * model.x[i] for i in model.CASES),
        sense=pyo.maximize,
    )
    model.feed_budget = pyo.Constraint(expr=sum(model.x[i] for i in model.CASES) <= effective_budget)
    model.candidate_limit = pyo.ConstraintList()
    for i in model.CASES:
        cap = candidate_cap if config.enforce_candidate_cap else effective_budget
        model.candidate_limit.add(model.x[i] <= cap * model.y[i])
    if config.enforce_max_selected:
        model.max_selected = pyo.Constraint(expr=sum(model.y[i] for i in model.CASES) <= config.max_portfolio_candidates)

    model.subtype_cap = pyo.ConstraintList()
    model.subtype_activation = pyo.ConstraintList()
    for subtype, indices in subtype_groups.items():
        if config.enforce_subtype_cap:
            model.subtype_cap.add(sum(model.x[i] for i in indices) <= subtype_cap)
        model.subtype_activation.add(model.z[subtype] <= sum(model.y[i] for i in indices))
    if config.enforce_min_distinct_subtypes and len(subtype_groups) >= config.min_distinct_subtypes:
        model.min_diversity = pyo.Constraint(
            expr=sum(model.z[subtype] for subtype in model.SUBTYPES) >= config.min_distinct_subtypes
        )

    model.carbon_budget = pyo.Constraint(expr=sum(carbon_coeff[i] * model.x[i] for i in model.CASES) <= carbon_budget)
    model.energy_balance = pyo.Constraint(
        expr=model.total_energy == sum(energy_coeff[i] * model.x[i] for i in model.CASES)
    )
    model.environment_balance = pyo.Constraint(
        expr=model.total_environment == sum(environment_coeff[i] * model.x[i] for i in model.CASES)
    )
    model.cost_balance = pyo.Constraint(
        expr=model.total_cost == sum(cost_coeff[i] * model.x[i] for i in model.CASES)
    )
    model.carbon_balance = pyo.Constraint(
        expr=model.total_carbon_load == sum(carbon_coeff[i] * model.x[i] for i in model.CASES)
    )
    return model


def _extract_pyomo_allocations(
    scored: pd.DataFrame,
    model,
    scenario_constraint: dict[str, object],
) -> pd.DataFrame:  # pragma: no cover - optional solver extraction
    allocation = pd.Series([pyo.value(model.x[i]) for i in model.CASES], index=scored.index, dtype=float)
    return _build_allocation_frame(scored, allocation, scenario_constraint)


def _solve_with_scipy_milp(
    scored: pd.DataFrame,
    scenario_constraint: dict[str, object],
    config: "PlanningConfig",
) -> ScenarioOptimizationResult:
    if milp is None or LinearConstraint is None or Bounds is None:
        allocation = _documented_allocation_fallback(scored, scenario_constraint, config)
        return ScenarioOptimizationResult(
            scenario_name=str(scenario_constraint.get("scenario_name", "")),
            allocations=allocation,
            diagnostics={
                "solver_status": "documented_greedy_fallback",
                "solver_backend": "none",
                **_constraint_diagnostics(scored, scenario_constraint, config, allocation),
            },
        )

    n_cases = len(scored)
    subtype_groups = _subtype_groups(scored)
    subtype_names = list(subtype_groups)
    n_subtypes = len(subtype_names)
    total_vars = n_cases + n_cases + n_subtypes

    effective_budget = _required_constraint_value(scenario_constraint, "effective_processing_budget_ton_per_year")
    candidate_cap = _required_constraint_value(scenario_constraint, "candidate_share_cap_ton_per_year")
    subtype_cap = _required_constraint_value(scenario_constraint, "subtype_share_cap_ton_per_year")
    carbon_budget = _carbon_budget(scored, scenario_constraint, config)
    _require_numeric_columns(
        scored,
        columns=(
            "weighted_score_per_ton",
            "planning_carbon_load_kgco2e_per_ton",
        ),
        context="scipy milp assembly",
    )

    cost_vector = np.zeros(total_vars, dtype=float)
    cost_vector[:n_cases] = -pd.to_numeric(scored["weighted_score_per_ton"], errors="coerce").to_numpy()

    lower_bounds = np.zeros(total_vars, dtype=float)
    upper_bounds = np.ones(total_vars, dtype=float)
    upper_bounds[:n_cases] = candidate_cap if config.enforce_candidate_cap else effective_budget
    upper_bounds[n_cases : n_cases + n_cases] = 1.0
    upper_bounds[n_cases + n_cases :] = 1.0
    bounds = Bounds(lower_bounds, upper_bounds)

    integrality = np.zeros(total_vars, dtype=int)
    integrality[n_cases:] = 1

    constraints: list[LinearConstraint] = []

    feed_budget_row = np.zeros(total_vars, dtype=float)
    feed_budget_row[:n_cases] = 1.0
    constraints.append(LinearConstraint(feed_budget_row, -np.inf, effective_budget))

    if config.enforce_max_selected:
        selection_row = np.zeros(total_vars, dtype=float)
        selection_row[n_cases : n_cases + n_cases] = 1.0
        constraints.append(LinearConstraint(selection_row, -np.inf, float(config.max_portfolio_candidates)))

    for i in range(n_cases):
        row = np.zeros(total_vars, dtype=float)
        row[i] = 1.0
        row[n_cases + i] = -(candidate_cap if config.enforce_candidate_cap else effective_budget)
        constraints.append(LinearConstraint(row, -np.inf, 0.0))

    for subtype_index, subtype in enumerate(subtype_names):
        row = np.zeros(total_vars, dtype=float)
        subtype_y_row = np.zeros(total_vars, dtype=float)
        for case_index in subtype_groups[subtype]:
            row[case_index] = 1.0
            subtype_y_row[n_cases + case_index] = -1.0
        row[n_cases + n_cases + subtype_index] = 0.0
        if config.enforce_subtype_cap:
            constraints.append(LinearConstraint(row, -np.inf, subtype_cap))

        subtype_activation = np.zeros(total_vars, dtype=float)
        subtype_activation[n_cases + n_cases + subtype_index] = 1.0
        subtype_activation += subtype_y_row
        constraints.append(LinearConstraint(subtype_activation, -np.inf, 0.0))

    if config.enforce_min_distinct_subtypes and n_subtypes >= config.min_distinct_subtypes:
        diversity_row = np.zeros(total_vars, dtype=float)
        diversity_row[n_cases + n_cases :] = 1.0
        constraints.append(LinearConstraint(diversity_row, float(config.min_distinct_subtypes), np.inf))

    carbon_row = np.zeros(total_vars, dtype=float)
    carbon_row[:n_cases] = pd.to_numeric(
        scored["planning_carbon_load_kgco2e_per_ton"], errors="coerce"
    ).to_numpy()
    constraints.append(LinearConstraint(carbon_row, -np.inf, carbon_budget))

    solved = milp(
        c=cost_vector,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
    )
    if not getattr(solved, "success", False):
        allocation = _documented_allocation_fallback(scored, scenario_constraint, config)
        return ScenarioOptimizationResult(
            scenario_name=str(scenario_constraint.get("scenario_name", "")),
            allocations=allocation,
            diagnostics={
                "solver_status": "scipy_milp_failed_fallback",
                "solver_backend": "scipy_milp",
                **_constraint_diagnostics(scored, scenario_constraint, config, allocation),
            },
        )

    allocation = pd.Series(solved.x[:n_cases], index=scored.index, dtype=float)
    allocations = _build_allocation_frame(scored, allocation, scenario_constraint)
    return ScenarioOptimizationResult(
        scenario_name=str(scenario_constraint.get("scenario_name", "")),
        allocations=allocations,
        diagnostics={
            "solver_status": "optimal",
            "solver_backend": "scipy_milp",
            **_constraint_diagnostics(scored, scenario_constraint, config, allocations),
        },
    )


def _documented_allocation_fallback(
    scored: pd.DataFrame,
    scenario_constraint: dict[str, object],
    config: "PlanningConfig",
) -> pd.DataFrame:
    ordered = scored.sort_values(
        ["weighted_score_per_ton", "planning_energy_intensity_mj_per_ton"],
        ascending=[False, False],
    ).copy()
    effective_budget = _required_constraint_value(scenario_constraint, "effective_processing_budget_ton_per_year")
    candidate_cap = _required_constraint_value(scenario_constraint, "candidate_share_cap_ton_per_year")
    subtype_cap = _required_constraint_value(scenario_constraint, "subtype_share_cap_ton_per_year")
    carbon_budget = _carbon_budget(scored, scenario_constraint, config)
    _require_numeric_columns(
        ordered,
        columns=("weighted_score_per_ton", "planning_energy_intensity_mj_per_ton", "planning_carbon_load_kgco2e_per_ton"),
        context="documented allocation fallback",
    )

    allocation = pd.Series(0.0, index=ordered.index, dtype=float)
    subtype_load: dict[str, float] = {}
    carbon_load = 0.0
    chosen = 0
    for idx, row in ordered.iterrows():
        if config.enforce_max_selected and chosen >= config.max_portfolio_candidates:
            break
        subtype = _candidate_subtype_key(row)
        remaining = effective_budget - float(allocation.sum())
        if remaining <= 1e-9:
            break
        allowable = remaining
        if config.enforce_candidate_cap:
            allowable = min(allowable, candidate_cap)
        if config.enforce_subtype_cap:
            allowable = min(allowable, subtype_cap - subtype_load.get(subtype, 0.0))
        if allowable <= 1e-9:
            continue
        row_carbon_intensity = _required_row_numeric_value(
            row,
            "planning_carbon_load_kgco2e_per_ton",
            context="documented allocation fallback",
        )
        if carbon_load + allowable * row_carbon_intensity > carbon_budget + 1e-9:
            allowable = max(0.0, (carbon_budget - carbon_load) / max(row_carbon_intensity, 1e-9))
        if allowable <= 1e-9:
            continue
        allocation.loc[idx] = allowable
        subtype_load[subtype] = subtype_load.get(subtype, 0.0) + allowable
        carbon_load += allowable * row_carbon_intensity
        chosen += 1
    return _build_allocation_frame(ordered, allocation, scenario_constraint)


def _build_allocation_frame(
    scored: pd.DataFrame,
    allocation: pd.Series,
    scenario_constraint: dict[str, object],
) -> pd.DataFrame:
    total_budget = _required_constraint_value(
        scenario_constraint,
        "effective_processing_budget_ton_per_year",
    )
    result = scored.copy()
    result["allocated_feed_ton_per_year"] = allocation.reindex(result.index).fillna(0.0)
    result = result[result["allocated_feed_ton_per_year"] > 1e-6].copy()
    if result.empty:
        return result

    result["allocated_feed_share"] = result["allocated_feed_ton_per_year"] / max(total_budget, 1.0)
    result["candidate_capacity_cap_ton_per_year"] = _required_constraint_value(
        scenario_constraint,
        "candidate_share_cap_ton_per_year",
    )
    result["subtype_capacity_cap_ton_per_year"] = _required_constraint_value(
        scenario_constraint,
        "subtype_share_cap_ton_per_year",
    )
    result["scenario_feed_budget_ton_per_year"] = _required_constraint_value(
        scenario_constraint,
        "scenario_feed_budget_ton_per_year",
    )
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
    result["allocated_carbon_load_kgco2e"] = (
        result["allocated_feed_ton_per_year"] * result["planning_carbon_load_kgco2e_per_ton"]
    )
    result["allocated_score_mass"] = result["allocated_feed_ton_per_year"] * result["weighted_score_per_ton"]
    result = result.sort_values(
        ["allocated_score_mass", "allocated_feed_ton_per_year"],
        ascending=[False, False],
    ).reset_index(drop=True)
    result["portfolio_rank"] = range(1, len(result) + 1)
    return result


def _build_weight_grid(point_count: int) -> list[tuple[float, float, float]]:
    anchors = np.linspace(0.15, 0.70, num=max(3, min(point_count, 7)))
    grid: list[tuple[float, float, float]] = []
    for energy_weight, environment_weight in product(anchors, anchors):
        if energy_weight + environment_weight >= 0.95:
            continue
        cost_weight = 1.0 - energy_weight - environment_weight
        if cost_weight <= 0.05:
            continue
        grid.append((round(float(energy_weight), 4), round(float(environment_weight), 4), round(float(cost_weight), 4)))
    if not grid:
        return [(0.40, 0.35, 0.25)]
    return grid


def _prefix_diagnostics(diagnostics: dict[str, object], prefix: str) -> dict[str, object]:
    return {f"{prefix}{key}": value for key, value in diagnostics.items()}


def _subtype_groups(scored: pd.DataFrame) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, row in scored.iterrows():
        groups.setdefault(_candidate_subtype_key(row), []).append(int(idx))
    return groups


def _carbon_budget(
    scored: pd.DataFrame,
    scenario_constraint: dict[str, object],
    config: "PlanningConfig",
) -> float:
    constraint_reference = _required_constraint_value(
        scenario_constraint,
        "baseline_emission_factor_kgco2e_per_metric_ton",
    )
    if constraint_reference <= 0.0:
        raise ValueError("baseline_emission_factor_kgco2e_per_metric_ton must be positive for carbon-budget construction.")
    baseline_reference = constraint_reference
    effective_budget = _required_constraint_value(scenario_constraint, "effective_processing_budget_ton_per_year")
    return max(0.0, baseline_reference * effective_budget * config.carbon_budget_factor)


def _candidate_subtype_key(row: pd.Series) -> str:
    manure_subtype = str(row.get("manure_subtype", "") or "").strip()
    if manure_subtype:
        return manure_subtype
    return str(row.get("sample_id", "unknown_candidate"))


def _normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.isna().any():
        raise ValueError("Normalization received missing/non-finite values; upstream objective construction should exclude them.")
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return pd.Series(0.0, index=values.index)
    return (values - minimum) / (maximum - minimum)


def _apply_scenario_external_evidence(
    frame: pd.DataFrame,
    config: "PlanningConfig",
) -> pd.DataFrame:
    if frame.empty or not config.scenario_external_evidence:
        return frame
    if "scenario_name" not in frame.columns:
        return frame
    adjusted = frame.copy()
    adjusted["scenario_external_evidence_source"] = adjusted.get(
        "evidence_source",
        pd.Series(["unadjusted"] * len(adjusted), index=adjusted.index),
    ).astype(str)
    adjusted["scenario_external_evidence_reference"] = adjusted.get(
        "evidence_reference",
        pd.Series([""] * len(adjusted), index=adjusted.index),
    ).astype(str)
    adjusted["scenario_external_evidence_rationale"] = adjusted.get(
        "evidence_rationale",
        pd.Series([""] * len(adjusted), index=adjusted.index),
    ).astype(str)
    adjusted["scenario_external_evidence_table_path"] = config.scenario_external_evidence_table_path or ""
    variance_scale = float(config.scenario_metric_variance_scale)
    for evidence in config.scenario_external_evidence:
        mask = adjusted["scenario_name"].astype(str).eq(evidence.scenario_name)
        if not mask.any():
            continue
        scale_factor = _scaled_multiplier(evidence.feedstock_scale_factor, variance_scale)
        cost_scale_multiplier = float(
            np.clip(np.power(max(scale_factor, 1e-6), -float(evidence.feedstock_cost_elasticity)), 0.70, 1.05)
        )
        carbon_tax = float(evidence.carbon_tax_usd_per_ton_co2e) * max(1.0, variance_scale)
        carbon_tax_cost = (
            pd.to_numeric(adjusted.loc[mask, "planning_carbon_load_kgco2e_per_ton"], errors="coerce")
            .clip(lower=0.0)
            * carbon_tax
            / 1000.0
        )
        adjusted.loc[mask, "planning_cost_intensity_proxy_or_real_per_ton"] = (
            pd.to_numeric(adjusted.loc[mask, "planning_cost_intensity_proxy_or_real_per_ton"], errors="coerce")
            * cost_scale_multiplier
            + carbon_tax_cost
        )
        adjusted.loc[mask, "scenario_scale_cost_multiplier"] = cost_scale_multiplier
        adjusted.loc[mask, "carbon_tax_cost_intensity_usd_per_ton"] = carbon_tax_cost
        adjusted.loc[mask, "scenario_external_evidence_source"] = evidence.evidence_source
        adjusted.loc[mask, "scenario_external_evidence_reference"] = evidence.evidence_reference
        adjusted.loc[mask, "scenario_external_evidence_rationale"] = evidence.evidence_rationale
    return adjusted


def _scaled_multiplier(multiplier: float, variance_scale: float) -> float:
    return 1.0 + (float(multiplier) - 1.0) * variance_scale


def _scale_numeric_series(series: pd.Series, multiplier: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values * float(multiplier)


def _constraint_diagnostics(
    scored: pd.DataFrame,
    scenario_constraint: dict[str, object],
    config: "PlanningConfig",
    allocations: pd.DataFrame,
) -> dict[str, object]:
    effective_budget = _required_constraint_value(
        scenario_constraint,
        "effective_processing_budget_ton_per_year",
    )
    candidate_cap = _required_constraint_value(
        scenario_constraint,
        "candidate_share_cap_ton_per_year",
    )
    subtype_cap = _required_constraint_value(
        scenario_constraint,
        "subtype_share_cap_ton_per_year",
    )
    carbon_budget = _carbon_budget(scored, scenario_constraint, config)
    allocation_frame = allocations.copy() if isinstance(allocations, pd.DataFrame) else pd.DataFrame()
    if allocation_frame.empty:
        total_allocated = 0.0
        selected_count = 0
        candidate_load_max = 0.0
        subtype_load_max = 0.0
        carbon_used = 0.0
        distinct_selected = 0
    else:
        _require_numeric_columns(
            allocation_frame,
            columns=("allocated_feed_ton_per_year", "allocated_carbon_load_kgco2e"),
            context="constraint diagnostics allocation frame",
        )
        total_allocated = float(
            pd.to_numeric(allocation_frame["allocated_feed_ton_per_year"], errors="coerce").sum()
        )
        selected_count = int(len(allocation_frame))
        candidate_load_max = float(
            pd.to_numeric(allocation_frame["allocated_feed_ton_per_year"], errors="coerce").max()
        )
        subtype_load_max = (
            float(
                pd.to_numeric(
                    allocation_frame.groupby("manure_subtype", dropna=False)["allocated_feed_ton_per_year"].sum(),
                    errors="coerce",
                )
                .max()
            )
            if "manure_subtype" in allocation_frame.columns
            else 0.0
        )
        carbon_used = float(
            pd.to_numeric(allocation_frame.get("allocated_carbon_load_kgco2e"), errors="coerce").sum()
        )
        distinct_selected = (
            int(allocation_frame["manure_subtype"].astype(str).nunique())
            if "manure_subtype" in allocation_frame.columns
            else 0
        )
    min_distinct_value = pd.to_numeric(
        pd.Series([scenario_constraint.get("min_distinct_subtypes", config.min_distinct_subtypes)]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(min_distinct_value):
        raise ValueError("Scenario optimization constraint contains missing/non-numeric value in 'min_distinct_subtypes'.")
    min_distinct_required = int(min_distinct_value)
    return {
        "constraint_relaxation_ratio": float(config.constraint_relaxation_ratio),
        "subtype_relaxation_ratio": float(config.subtype_relaxation_ratio),
        "candidate_cap_enforced": bool(config.enforce_candidate_cap),
        "subtype_cap_enforced": bool(config.enforce_subtype_cap),
        "max_selected_enforced": bool(config.enforce_max_selected),
        "min_distinct_subtypes_enforced": bool(config.enforce_min_distinct_subtypes),
        "scenario_metric_variance_scale": float(config.scenario_metric_variance_scale),
        "effective_budget_ton_per_year": effective_budget,
        "allocated_feed_ton_per_year": total_allocated,
        "feed_budget_slack_ton_per_year": max(0.0, effective_budget - total_allocated),
        "feed_budget_binding": bool(effective_budget > 0.0 and abs(effective_budget - total_allocated) <= 1e-6),
        "candidate_cap_ton_per_year": candidate_cap,
        "candidate_cap_max_observed_ton_per_year": candidate_load_max,
        "candidate_cap_slack_ton_per_year": max(0.0, candidate_cap - candidate_load_max),
        "candidate_cap_binding": bool(config.enforce_candidate_cap and candidate_cap > 0.0 and abs(candidate_cap - candidate_load_max) <= 1e-6),
        "candidate_cap_shadow_price_proxy": 1.0 if config.enforce_candidate_cap and candidate_cap > 0.0 and abs(candidate_cap - candidate_load_max) <= 1e-6 else 0.0,
        "subtype_cap_ton_per_year": subtype_cap,
        "subtype_cap_max_observed_ton_per_year": subtype_load_max,
        "subtype_cap_slack_ton_per_year": max(0.0, subtype_cap - subtype_load_max),
        "subtype_cap_binding": bool(config.enforce_subtype_cap and subtype_cap > 0.0 and abs(subtype_cap - subtype_load_max) <= 1e-6),
        "subtype_cap_shadow_price_proxy": 1.0 if config.enforce_subtype_cap and subtype_cap > 0.0 and abs(subtype_cap - subtype_load_max) <= 1e-6 else 0.0,
        "max_selected_limit": int(config.max_portfolio_candidates),
        "selected_candidate_count": selected_count,
        "max_selected_slack": max(0, int(config.max_portfolio_candidates) - selected_count),
        "max_selected_binding": bool(config.enforce_max_selected and selected_count >= int(config.max_portfolio_candidates)),
        "max_selected_shadow_price_proxy": 1.0 if config.enforce_max_selected and selected_count >= int(config.max_portfolio_candidates) else 0.0,
        "min_distinct_subtypes_required": min_distinct_required,
        "distinct_subtypes_selected": distinct_selected,
        "min_distinct_subtypes_slack": max(0, distinct_selected - min_distinct_required),
        "min_distinct_subtypes_binding": bool(config.enforce_min_distinct_subtypes and min_distinct_required > 0 and distinct_selected == min_distinct_required),
        "min_distinct_subtypes_shadow_price_proxy": 1.0 if config.enforce_min_distinct_subtypes and min_distinct_required > 0 and distinct_selected == min_distinct_required else 0.0,
        "carbon_budget_kgco2e": carbon_budget,
        "carbon_load_used_kgco2e": carbon_used,
        "carbon_budget_slack_kgco2e": max(0.0, carbon_budget - carbon_used),
        "carbon_budget_binding": bool(carbon_budget > 0.0 and abs(carbon_budget - carbon_used) <= 1e-6),
        "carbon_budget_shadow_price_proxy": 1.0 if carbon_budget > 0.0 and abs(carbon_budget - carbon_used) <= 1e-6 else 0.0,
    }


def _pareto_efficient_mask(
    energy: pd.Series,
    environment: pd.Series,
    cost: pd.Series,
) -> pd.Series:
    frame = pd.DataFrame(
        {
            "energy": pd.to_numeric(energy, errors="coerce"),
            "environment": pd.to_numeric(environment, errors="coerce"),
            "cost": pd.to_numeric(cost, errors="coerce"),
        }
    ).reset_index(drop=True)
    if frame.isna().any().any():
        raise ValueError("Pareto-front construction received missing objective values.")
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


def _require_numeric_columns(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    context: str,
) -> None:
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"{context} requires column '{column}', but it is missing.")
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.isna().any():
            invalid_rows = frame.loc[values.isna(), "optimization_case_id"] if "optimization_case_id" in frame.columns else frame.index[values.isna()]
            preview = ", ".join(str(value) for value in list(invalid_rows[:5]))
            raise ValueError(
                f"{context} encountered missing/non-finite values in '{column}' for row(s): {preview}."
            )


def _required_constraint_value(scenario_constraint: dict[str, object], key: str) -> float:
    if key not in scenario_constraint:
        raise ValueError(f"Scenario optimization constraint is missing required key '{key}'.")
    value = pd.to_numeric(pd.Series([scenario_constraint.get(key)]), errors="coerce").iloc[0]
    if pd.isna(value):
        scenario_name = scenario_constraint.get("scenario_name", "unknown_scenario")
        raise ValueError(
            f"Scenario optimization constraint for '{scenario_name}' contains missing/non-numeric value in '{key}'."
        )
    return float(value)


def _required_row_numeric_value(row: pd.Series, key: str, *, context: str) -> float:
    if key not in row.index:
        raise ValueError(f"{context} requires row key '{key}', but it is missing.")
    value = pd.to_numeric(pd.Series([row[key]]), errors="coerce").iloc[0]
    if pd.isna(value):
        row_id = row.get("optimization_case_id", "unknown_case")
        raise ValueError(f"{context} encountered missing/non-numeric '{key}' for row '{row_id}'.")
    return float(value)
