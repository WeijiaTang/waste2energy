from __future__ import annotations

from typing import Any

import pandas as pd

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
    base_config: PlanningConfig | None = None,
) -> dict[str, Any]:
    baseline_config = base_config or PlanningConfig()
    bundle = load_planning_input_bundle(dataset_path=dataset_path)
    stress_registry = build_default_stress_registry(baseline_config)
    registry_frame = registry_to_frame(stress_registry)

    run_records: list[dict[str, Any]] = []
    portfolio_frames: list[pd.DataFrame] = []
    constraint_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    objective_readiness: dict[str, str] | None = None
    planner_variant = "constraint_aware_weighted_portfolio"

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
    return {
        "dataset_path": str(bundle.dataset_path),
        "stress_test_count": int(len(stress_registry)),
        "stress_run_count": int(len(run_records)),
        "planner_variant": planner_variant,
        "objective_readiness": objective_readiness or {},
        "outputs": outputs,
    }
