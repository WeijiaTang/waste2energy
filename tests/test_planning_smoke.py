# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json

import pandas as pd

from waste2energy.planning import optimization as planning_optimization
from waste2energy.planning.solve import PlanningConfig, run_planning_baseline


def test_planning_baseline_smoke(tmp_path):
    output_dir = tmp_path / "planning"
    result = run_planning_baseline(
        output_dir=str(output_dir),
        config=PlanningConfig(pareto_point_count=6),
    )

    assert result["planner_variant"] == "surrogate_driven_robust_multiobjective_optimizer"
    assert result["recommendation_count"] > 0
    assert result["pareto_candidate_count"] > 0

    surrogate_predictions = pd.read_csv(output_dir / "surrogate_predictions.csv")
    optimization_diagnostics = pd.read_csv(output_dir / "optimization_diagnostics.csv")
    portfolio_allocations = pd.read_csv(output_dir / "portfolio_allocations.csv")
    scored_cases = pd.read_csv(output_dir / "scored_cases.csv")
    scenario_constraints = pd.read_csv(output_dir / "scenario_constraints.csv")
    planning_data_quality_summary = pd.read_csv(output_dir / "planning_data_quality_summary.csv")
    scenario_metric_adjustments = pd.read_csv(output_dir / "scenario_metric_adjustments.csv")
    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))

    assert "combined_uncertainty_ratio" in surrogate_predictions.columns
    assert surrogate_predictions["combined_uncertainty_ratio"].ge(0.0).all()
    assert set(optimization_diagnostics["solver_status"]) == {"optimal"}
    assert "candidate_cap_binding" in optimization_diagnostics.columns
    assert "subtype_cap_binding" in optimization_diagnostics.columns
    assert "constraint_relaxation_ratio" in optimization_diagnostics.columns
    assert "planning_score_scope" in scored_cases.columns
    assert set(scored_cases["planning_score_scope"]) == {"scenario_local_optimizer"}
    assert "scenario_metric_adjustment_source" in scored_cases.columns
    assert "planning_imputation_flag" in scored_cases.columns
    assert "scenario_baseline_waste_treatment_emission_factor_kgco2e_per_metric_ton" in scored_cases.columns
    assert "planning_carbon_unit_basis" in scored_cases.columns
    assert set(scored_cases["planning_carbon_unit_basis"]) == {"kgco2e_per_metric_ton"}
    assert scenario_metric_adjustments["adjustment_source"].astype(str).str.len().gt(0).all()
    assert planning_data_quality_summary["evaluated_candidate_count"].gt(0).all()
    assert "baseline_emission_factor_kgco2e_per_metric_ton" in scenario_constraints.columns
    assert "baseline_emission_factor_source_unit" in scenario_constraints.columns
    assert "carbon_budget_basis_note" in scenario_constraints.columns
    assert scenario_constraints["baseline_emission_factor_kgco2e_per_metric_ton"].gt(
        scenario_constraints["baseline_emission_factor_kgco2e_per_short_ton_source"]
    ).all()
    assert run_config["scenario_metric_adjustment_table_path"]
    assert run_config["unit_registry"]["planning_mass_unit_basis"] == "metric_ton"
    assert portfolio_allocations["allocated_feed_ton_per_year"].gt(0.0).all()


def test_planning_solver_falls_back_after_pyomo_exception(monkeypatch):
    scenario_frame = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "sample_id": "sample-1",
                "optimization_case_id": "case-1",
                "manure_subtype": "beef",
                "planning_energy_intensity_mj_per_ton": 10.0,
                "planning_environment_intensity_kgco2e_per_ton": 5.0,
                "planning_cost_intensity_proxy_or_real_per_ton": 2.0,
                "planning_carbon_load_kgco2e_per_ton": 1.0,
                "combined_uncertainty_ratio": 0.1,
            }
        ]
    )
    scenario_constraint = {
        "scenario_name": "baseline_region_case",
        "effective_processing_budget_ton_per_year": 1.0,
        "candidate_share_cap_ton_per_year": 1.0,
        "subtype_share_cap_ton_per_year": 1.0,
        "baseline_emission_factor_kgco2e_per_metric_ton": 11.023113109243878,
    }

    def fake_pyomo(*args, **kwargs):
        return planning_optimization.ScenarioOptimizationResult(
            scenario_name="baseline_region_case",
            allocations=pd.DataFrame(),
            diagnostics={
                "solver_status": "pyomo_exception",
                "solver_backend": "pyomo",
                "solver_error": "boom",
            },
        )

    def fake_scipy(scored, constraint, config):
        return planning_optimization.ScenarioOptimizationResult(
            scenario_name="baseline_region_case",
            allocations=planning_optimization._build_allocation_frame(
                scored,
                pd.Series([1.0], index=scored.index, dtype=float),
                constraint,
            ),
            diagnostics={
                "solver_status": "optimal",
                "solver_backend": "scipy_milp",
            },
        )

    monkeypatch.setattr(planning_optimization, "_solve_with_pyomo_if_available", fake_pyomo)
    monkeypatch.setattr(planning_optimization, "_solve_with_scipy_milp", fake_scipy)

    result = planning_optimization.solve_scenario_optimization(
        scenario_frame,
        scenario_constraint,
        PlanningConfig(),
    )

    assert not result.allocations.empty
    assert result.diagnostics["solver_status"] == "optimal"
    assert result.diagnostics["solver_backend"] == "scipy_milp"
    assert result.diagnostics["pyomo_attempt_solver_status"] == "pyomo_exception"
