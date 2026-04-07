# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ..config import perturb_objective_weights
from ..operation.baselines import run_baseline_policies
from ..operation.inputs import build_operation_environment_specs
from ..scenarios.run import run_scenario_robustness_baseline
from .solve import PlanningConfig, run_planning_baseline


def analyze_weight_sensitivity(
    *,
    dataset_path: str | None = None,
    base_config: PlanningConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = base_config or PlanningConfig()
    rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []

    for variant in perturb_objective_weights(config.objective_weight_system):
        variant_config = PlanningConfig(
            objective_weight_preset=variant.preset_name,
            objective_weight_system=variant,
            top_k_per_scenario=config.top_k_per_scenario,
            max_portfolio_candidates=config.max_portfolio_candidates,
            max_candidate_share=config.max_candidate_share,
            max_subtype_share=config.max_subtype_share,
            min_distinct_subtypes=config.min_distinct_subtypes,
            deployable_capacity_fraction=config.deployable_capacity_fraction,
            robustness_factor=config.robustness_factor,
            carbon_budget_factor=config.carbon_budget_factor,
            constraint_relaxation_ratio=config.constraint_relaxation_ratio,
            subtype_relaxation_ratio=config.subtype_relaxation_ratio,
            enforce_candidate_cap=config.enforce_candidate_cap,
            enforce_subtype_cap=config.enforce_subtype_cap,
            enforce_max_selected=config.enforce_max_selected,
            enforce_min_distinct_subtypes=config.enforce_min_distinct_subtypes,
            scenario_metric_variance_scale=config.scenario_metric_variance_scale,
            scenario_metric_adjustments=config.scenario_metric_adjustments,
            optimization_method=config.optimization_method,
            pareto_point_count=config.pareto_point_count,
            allow_surrogate_fallback=config.allow_surrogate_fallback,
        )
        with TemporaryDirectory(prefix="wte_weight_sensitivity_") as tmp_dir:
            planning_dir = Path(tmp_dir) / "planning"
            scenario_dir = Path(tmp_dir) / "scenarios"
            planning_result = run_planning_baseline(
                dataset_path=dataset_path,
                output_dir=str(planning_dir),
                config=variant_config,
            )
            run_scenario_robustness_baseline(
                dataset_path=dataset_path,
                output_dir=str(scenario_dir),
                base_config=variant_config,
            )
            specs = build_operation_environment_specs(
                planning_dir=planning_dir,
                scenario_dir=scenario_dir,
            )
            _, rollout_summary = run_baseline_policies(specs, horizon_steps=8)

            scenario_summary = pd.read_csv(planning_dir / "scenario_summary.csv")
            for _, row in scenario_summary.iterrows():
                ranking_rows.append(
                    {
                        "weight_preset": variant.preset_name,
                        "scenario_name": row["scenario_name"],
                        "top_ranked_case_id": row.get("top_ranked_case_id"),
                        "top_portfolio_case_id": row.get("top_portfolio_case_id"),
                        "top_ranked_case_score": row.get("top_ranked_case_score"),
                    }
                )

            hold_plan = rollout_summary[rollout_summary["policy_name"] == "hold_plan"].copy()
            rows.append(
                {
                    "weight_preset": variant.preset_name,
                    "energy_weight": variant.energy,
                    "environment_weight": variant.environment,
                    "cost_weight": variant.cost,
                    "planning_recommendation_count": planning_result["recommendation_count"],
                    "planning_pareto_candidate_count": planning_result["pareto_candidate_count"],
                    "operation_environment_count": int(len(specs)),
                    "hold_plan_reward_mean": float(hold_plan["average_reward"].mean()) if not hold_plan.empty else 0.0,
                    "hold_plan_total_energy_mean": float(hold_plan["total_realized_energy"].mean())
                    if not hold_plan.empty
                    else 0.0,
                    "hold_plan_total_environment_mean": float(hold_plan["total_realized_environment"].mean())
                    if not hold_plan.empty
                    else 0.0,
                    "hold_plan_total_cost_mean": float(hold_plan["total_realized_cost"].mean())
                    if not hold_plan.empty
                    else 0.0,
                }
            )

    return (
        pd.DataFrame(rows).sort_values("weight_preset").reset_index(drop=True),
        pd.DataFrame(ranking_rows).sort_values(["weight_preset", "scenario_name"]).reset_index(drop=True),
    )
