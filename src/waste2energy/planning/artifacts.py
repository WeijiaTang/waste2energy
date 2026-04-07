# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common import build_run_manifest, write_json
from ..config import PLANNING_OUTPUTS_DIR


def write_planning_outputs(
    *,
    scored: pd.DataFrame,
    scenario_recommendations: pd.DataFrame,
    pareto_candidates: pd.DataFrame,
    scenario_constraints: pd.DataFrame,
    portfolio_allocations: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    pathway_summary: pd.DataFrame,
    surrogate_predictions: pd.DataFrame,
    optimization_diagnostics: pd.DataFrame,
    output_dir: str | None,
    config,
    bundle,
    readiness: dict[str, str],
) -> dict[str, str]:
    target_dir = Path(output_dir) if output_dir else PLANNING_OUTPUTS_DIR / "baseline"
    target_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "scored_cases": target_dir / "scored_cases.csv",
        "scenario_recommendations": target_dir / "scenario_recommendations.csv",
        "pareto_candidates": target_dir / "pareto_candidates.csv",
        "pareto_front": target_dir / "pareto_front.csv",
        "scenario_constraints": target_dir / "scenario_constraints.csv",
        "portfolio_allocations": target_dir / "portfolio_allocations.csv",
        "portfolio_summary": target_dir / "portfolio_summary.csv",
        "scenario_summary": target_dir / "scenario_summary.csv",
        "pathway_summary": target_dir / "pathway_summary.csv",
        "surrogate_predictions": target_dir / "surrogate_predictions.csv",
        "optimization_diagnostics": target_dir / "optimization_diagnostics.csv",
        "run_config": target_dir / "run_config.json",
    }

    scored.to_csv(outputs["scored_cases"], index=False)
    scenario_recommendations.to_csv(outputs["scenario_recommendations"], index=False)
    pareto_candidates.to_csv(outputs["pareto_candidates"], index=False)
    pareto_candidates.to_csv(outputs["pareto_front"], index=False)
    scenario_constraints.to_csv(outputs["scenario_constraints"], index=False)
    portfolio_allocations.to_csv(outputs["portfolio_allocations"], index=False)
    portfolio_summary.to_csv(outputs["portfolio_summary"], index=False)
    scenario_summary.to_csv(outputs["scenario_summary"], index=False)
    pathway_summary.to_csv(outputs["pathway_summary"], index=False)
    surrogate_predictions.to_csv(outputs["surrogate_predictions"], index=False)
    optimization_diagnostics.to_csv(outputs["optimization_diagnostics"], index=False)

    run_config = build_run_manifest(
        dataset_path=str(bundle.dataset_path),
        scenario_names=list(bundle.scenario_names),
        pathways=list(bundle.pathways),
        real_cost_columns=list(bundle.real_cost_columns),
        unit_registry=bundle.unit_registry,
        planning_config=config,
        objective_weights=config.objective_weight_system,
        objective_readiness=readiness,
        row_count=int(len(scored)),
        planner_variant="surrogate_driven_robust_multiobjective_optimizer",
        output_files={key: str(path) for key, path in outputs.items()},
    )
    write_json(outputs["run_config"], run_config)
    return {key: str(path) for key, path in outputs.items()}
