from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..audit import build_confirmatory_audit, write_confirmatory_audit
from ..planning.reporting import build_main_results_table, write_main_results_table
from ..planning.inputs import load_planning_input_bundle
from ..planning.solve import PlanningConfig, execute_planning_pipeline
from .artifacts import write_scenario_outputs
from .registry import build_default_stress_registry, registry_to_frame
from .robustness import (
    build_cross_scenario_stability,
    build_decision_stability,
    build_stress_test_summary,
)
from .uncertainty import build_uncertainty_summary


def run_scenario_robustness_baseline(
    dataset_path: str | None = None,
    output_dir: str | None = None,
    planning_dir: str | None = None,
    base_config: PlanningConfig | None = None,
) -> dict[str, Any]:
    baseline_seed = base_config or PlanningConfig()
    baseline_config = PlanningConfig(
        objective_weight_preset=baseline_seed.objective_weight_preset,
        objective_weight_system=baseline_seed.objective_weight_system,
        top_k_per_scenario=baseline_seed.top_k_per_scenario,
        max_portfolio_candidates=baseline_seed.max_portfolio_candidates,
        max_candidate_share=baseline_seed.max_candidate_share,
        max_subtype_share=baseline_seed.max_subtype_share,
        min_distinct_subtypes=baseline_seed.min_distinct_subtypes,
        deployable_capacity_fraction=baseline_seed.deployable_capacity_fraction,
        robustness_factor=baseline_seed.robustness_factor,
        carbon_budget_factor=baseline_seed.carbon_budget_factor,
        constraint_relaxation_ratio=baseline_seed.constraint_relaxation_ratio,
        subtype_relaxation_ratio=baseline_seed.subtype_relaxation_ratio,
        enforce_candidate_cap=baseline_seed.enforce_candidate_cap,
        enforce_subtype_cap=baseline_seed.enforce_subtype_cap,
        enforce_max_selected=baseline_seed.enforce_max_selected,
        enforce_min_distinct_subtypes=baseline_seed.enforce_min_distinct_subtypes,
        scenario_metric_variance_scale=baseline_seed.scenario_metric_variance_scale,
        scenario_external_evidence_table_path=baseline_seed.scenario_external_evidence_table_path,
        scenario_external_evidence=baseline_seed.scenario_external_evidence,
        optimization_method=baseline_seed.optimization_method,
        pyomo_solver_preference=baseline_seed.pyomo_solver_preference,
        pareto_point_count=baseline_seed.pareto_point_count,
        enable_pareto_export=False,
        allow_surrogate_fallback=baseline_seed.allow_surrogate_fallback,
    )
    bundle = load_planning_input_bundle(dataset_path=dataset_path)
    stress_registry = build_default_stress_registry(baseline_config)
    registry_frame = registry_to_frame(stress_registry)

    run_records: list[dict[str, Any]] = []
    portfolio_frames: list[pd.DataFrame] = []
    constraint_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    objective_readiness: dict[str, str] | None = None
    planner_variant = "surrogate_driven_robust_multiobjective_optimizer"

    for stress in stress_registry:
        execution = execute_planning_pipeline(bundle=bundle, config=stress.planning_config)
        objective_readiness = execution["objective_readiness"]

        constraints = execution["scenario_constraints"].copy()
        constraints["stress_test_name"] = stress.name
        constraints["stress_test_description"] = stress.description
        constraint_frames.append(constraints)

        portfolio = execution["portfolio_allocations"].copy()
        portfolio["stress_test_name"] = stress.name
        portfolio["stress_test_description"] = stress.description
        portfolio_frames.append(portfolio)

        summary = execution["portfolio_summary"].copy()
        summary["stress_test_name"] = stress.name
        summary["stress_test_description"] = stress.description
        summary_frames.append(summary)

        run_records.append(
            {
                "stress_test_name": stress.name,
                "scenario_count": int(len(execution["scenario_summary"])),
                "portfolio_allocation_count": int(len(execution["portfolio_allocations"])),
            }
        )

    all_constraints = pd.concat(constraint_frames, ignore_index=True) if constraint_frames else pd.DataFrame()
    all_portfolios = pd.concat(portfolio_frames, ignore_index=True) if portfolio_frames else pd.DataFrame()
    all_summaries = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()

    stress_test_summary = build_stress_test_summary(
        stress_registry=registry_frame,
        scenario_constraints=all_constraints,
        portfolio_summary=all_summaries,
    )
    decision_stability = build_decision_stability(all_portfolios)
    cross_scenario_stability = build_cross_scenario_stability(all_portfolios)
    uncertainty_summary = build_uncertainty_summary(
        stress_test_summary=stress_test_summary,
        decision_stability=decision_stability,
    )
    outputs = write_scenario_outputs(
        registry=registry_frame,
        stress_test_summary=stress_test_summary,
        decision_stability=decision_stability,
        cross_scenario_stability=cross_scenario_stability,
        uncertainty_summary=uncertainty_summary,
        output_dir=output_dir,
        dataset_path=str(bundle.dataset_path),
        planner_variant=planner_variant,
        objective_readiness=objective_readiness or {},
    )
    reporting_outputs: dict[str, str] = {}
    audit_outputs: dict[str, str] = {}
    scenario_output_root = Path(output_dir) if output_dir else None
    planning_output_root = Path(planning_dir) if planning_dir else None
    audit_output_dir = None
    if scenario_output_root is not None:
        if scenario_output_root.parent.name == "scenarios":
            audit_output_dir = scenario_output_root.parent.parent / "audit"
        else:
            audit_output_dir = scenario_output_root.parent / "audit"
    elif planning_output_root is not None:
        if planning_output_root.parent.name == "planning":
            audit_output_dir = planning_output_root.parent.parent / "audit"
        else:
            audit_output_dir = planning_output_root.parent / "audit"
    try:
        table, manifest = build_main_results_table(
            planning_dir=planning_dir,
            scenario_dir=output_dir,
        )
        reporting_outputs = write_main_results_table(
            table,
            manifest,
            planning_dir=planning_dir,
        )
    except FileNotFoundError:
        reporting_outputs = {}
    try:
        audit_payload = build_confirmatory_audit(
            planning_dir=planning_dir,
            scenario_dir=output_dir,
        )
        audit_outputs = write_confirmatory_audit(
            audit_payload,
            output_dir=audit_output_dir,
        )
    except FileNotFoundError:
        audit_outputs = {}
    return {
        "dataset_path": str(bundle.dataset_path),
        "stress_test_count": int(len(stress_registry)),
        "stress_run_count": int(len(run_records)),
        "planner_variant": planner_variant,
        "objective_readiness": objective_readiness or {},
        "outputs": outputs,
        "reporting_outputs": reporting_outputs,
        "audit_outputs": audit_outputs,
    }
