# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from dataclasses import dataclass

from ..config import get_objective_weight_system
from ..planning.solve import PlanningConfig


@dataclass(frozen=True)
class ScenarioStressConfig:
    name: str
    description: str
    planning_config: PlanningConfig


def build_default_stress_registry(base_config: PlanningConfig | None = None) -> list[ScenarioStressConfig]:
    baseline = base_config or PlanningConfig()
    return [
        ScenarioStressConfig(
            name="baseline",
            description="Reference planning configuration carried forward from the planning layer.",
            planning_config=baseline,
        ),
        ScenarioStressConfig(
            name="energy_priority",
            description="Higher energy emphasis with slightly softer environment and cost weighting.",
            planning_config=_replace(
                baseline,
                objective_weight_system=get_objective_weight_system(
                    preset_name=baseline.objective_weight_preset,
                    energy=0.62,
                    environment=0.20,
                    cost=0.18,
                ),
                scenario_metric_variance_scale=1.25,
            ),
        ),
        ScenarioStressConfig(
            name="environment_priority",
            description="Higher environment emphasis to test carbon-benefit sensitivity.",
            planning_config=_replace(
                baseline,
                objective_weight_system=get_objective_weight_system(
                    preset_name=baseline.objective_weight_preset,
                    energy=0.22,
                    environment=0.58,
                    cost=0.20,
                ),
                scenario_metric_variance_scale=1.35,
            ),
        ),
        ScenarioStressConfig(
            name="cost_guardrail",
            description="Higher cost aversion to expose proxy-cost sensitivity.",
            planning_config=_replace(
                baseline,
                objective_weight_system=get_objective_weight_system(
                    preset_name=baseline.objective_weight_preset,
                    energy=0.22,
                    environment=0.20,
                    cost=0.58,
                ),
                scenario_metric_variance_scale=1.20,
            ),
        ),
        ScenarioStressConfig(
            name="tight_capacity",
            description="Tighter capacity deployability and concentration caps.",
            planning_config=_replace(
                baseline,
                deployable_capacity_fraction=0.70,
                max_candidate_share=0.40,
                max_subtype_share=0.50,
                constraint_relaxation_ratio=0.92,
                subtype_relaxation_ratio=0.90,
            ),
        ),
        ScenarioStressConfig(
            name="flexible_capacity",
            description="More permissive capacity and concentration assumptions.",
            planning_config=_replace(
                baseline,
                deployable_capacity_fraction=1.00,
                max_candidate_share=0.55,
                max_subtype_share=0.75,
                constraint_relaxation_ratio=1.25,
                subtype_relaxation_ratio=1.25,
                scenario_metric_variance_scale=1.15,
            ),
        ),
        ScenarioStressConfig(
            name="diversity_strict",
            description="Force broader subtype diversity inside the selected portfolio.",
            planning_config=_replace(
                baseline,
                max_portfolio_candidates=max(3, baseline.max_portfolio_candidates),
                min_distinct_subtypes=min(
                    max(3, baseline.min_distinct_subtypes),
                    max(3, baseline.max_portfolio_candidates),
                ),
                max_candidate_share=0.40,
                max_subtype_share=0.50,
                constraint_relaxation_ratio=0.95,
                subtype_relaxation_ratio=0.92,
            ),
        ),
        ScenarioStressConfig(
            name="unconstrained_scenario",
            description="Budget-only reference that removes concentration and selection caps to reveal natural pathway preference.",
            planning_config=_replace(
                baseline,
                max_portfolio_candidates=max(8, baseline.max_portfolio_candidates),
                max_candidate_share=1.0,
                max_subtype_share=1.0,
                min_distinct_subtypes=1,
                constraint_relaxation_ratio=1.0,
                subtype_relaxation_ratio=1.0,
                enforce_candidate_cap=False,
                enforce_subtype_cap=False,
                enforce_max_selected=False,
                enforce_min_distinct_subtypes=False,
                scenario_metric_variance_scale=1.40,
            ),
        ),
    ]


def registry_to_frame(registry: list[ScenarioStressConfig]):
    import pandas as pd

    rows = []
    for item in registry:
        config = item.planning_config
        rows.append(
            {
                "stress_test_name": item.name,
                "stress_test_description": item.description,
                "objective_weight_preset": config.objective_weight_preset,
                "energy_weight": config.energy_weight,
                "environment_weight": config.environment_weight,
                "cost_weight": config.cost_weight,
                "top_k_per_scenario": config.top_k_per_scenario,
                "max_portfolio_candidates": config.max_portfolio_candidates,
                "max_candidate_share": config.max_candidate_share,
                "max_subtype_share": config.max_subtype_share,
                "min_distinct_subtypes": config.min_distinct_subtypes,
                "deployable_capacity_fraction": config.deployable_capacity_fraction,
                "robustness_factor": config.robustness_factor,
                "carbon_budget_factor": config.carbon_budget_factor,
                "constraint_relaxation_ratio": config.constraint_relaxation_ratio,
                "subtype_relaxation_ratio": config.subtype_relaxation_ratio,
                "candidate_cap_enforced": config.enforce_candidate_cap,
                "subtype_cap_enforced": config.enforce_subtype_cap,
                "max_selected_enforced": config.enforce_max_selected,
                "min_distinct_subtypes_enforced": config.enforce_min_distinct_subtypes,
                "scenario_metric_variance_scale": config.scenario_metric_variance_scale,
                "pyomo_solver_preference": config.pyomo_solver_preference,
            }
        )
    return pd.DataFrame(rows)


def _replace(config: PlanningConfig, **overrides) -> PlanningConfig:
    payload = {
        "objective_weight_preset": config.objective_weight_preset,
        "objective_weight_system": config.objective_weight_system,
        "top_k_per_scenario": config.top_k_per_scenario,
        "max_portfolio_candidates": config.max_portfolio_candidates,
        "max_candidate_share": config.max_candidate_share,
        "max_subtype_share": config.max_subtype_share,
        "min_distinct_subtypes": config.min_distinct_subtypes,
        "deployable_capacity_fraction": config.deployable_capacity_fraction,
        "robustness_factor": config.robustness_factor,
        "carbon_budget_factor": config.carbon_budget_factor,
        "constraint_relaxation_ratio": config.constraint_relaxation_ratio,
        "subtype_relaxation_ratio": config.subtype_relaxation_ratio,
        "enforce_candidate_cap": config.enforce_candidate_cap,
        "enforce_subtype_cap": config.enforce_subtype_cap,
        "enforce_max_selected": config.enforce_max_selected,
        "enforce_min_distinct_subtypes": config.enforce_min_distinct_subtypes,
        "scenario_metric_variance_scale": config.scenario_metric_variance_scale,
        "scenario_metric_adjustment_table_path": config.scenario_metric_adjustment_table_path,
        "scenario_metric_adjustments": config.scenario_metric_adjustments,
        "optimization_method": config.optimization_method,
        "pyomo_solver_preference": config.pyomo_solver_preference,
        "pareto_point_count": config.pareto_point_count,
        "enable_pareto_export": config.enable_pareto_export,
        "allow_surrogate_fallback": config.allow_surrogate_fallback,
    }
    payload.update(overrides)
    return PlanningConfig(**payload)
