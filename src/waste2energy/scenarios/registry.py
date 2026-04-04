from __future__ import annotations

from dataclasses import dataclass

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
                energy_weight=0.55,
                environment_weight=0.25,
                cost_weight=0.20,
            ),
        ),
        ScenarioStressConfig(
            name="environment_priority",
            description="Higher environment emphasis to test carbon-benefit sensitivity.",
            planning_config=_replace(
                baseline,
                energy_weight=0.30,
                environment_weight=0.50,
                cost_weight=0.20,
            ),
        ),
        ScenarioStressConfig(
            name="cost_guardrail",
            description="Higher cost aversion to expose proxy-cost sensitivity.",
            planning_config=_replace(
                baseline,
                energy_weight=0.30,
                environment_weight=0.25,
                cost_weight=0.45,
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
            ),
        ),
        ScenarioStressConfig(
            name="diversity_strict",
            description="Force broader subtype diversity inside the selected portfolio.",
            planning_config=_replace(
                baseline,
                max_portfolio_candidates=max(3, baseline.max_portfolio_candidates),
                min_distinct_subtypes=min(max(3, baseline.min_distinct_subtypes), max(3, baseline.max_portfolio_candidates)),
                max_candidate_share=0.40,
                max_subtype_share=0.50,
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
                "energy_weight": config.energy_weight,
                "environment_weight": config.environment_weight,
                "cost_weight": config.cost_weight,
                "top_k_per_scenario": config.top_k_per_scenario,
                "max_portfolio_candidates": config.max_portfolio_candidates,
                "max_candidate_share": config.max_candidate_share,
                "max_subtype_share": config.max_subtype_share,
                "min_distinct_subtypes": config.min_distinct_subtypes,
                "deployable_capacity_fraction": config.deployable_capacity_fraction,
            }
        )
    return pd.DataFrame(rows)


def _replace(config: PlanningConfig, **overrides) -> PlanningConfig:
    payload = {
        "energy_weight": config.energy_weight,
        "environment_weight": config.environment_weight,
        "cost_weight": config.cost_weight,
        "top_k_per_scenario": config.top_k_per_scenario,
        "max_portfolio_candidates": config.max_portfolio_candidates,
        "max_candidate_share": config.max_candidate_share,
        "max_subtype_share": config.max_subtype_share,
        "min_distinct_subtypes": config.min_distinct_subtypes,
        "deployable_capacity_fraction": config.deployable_capacity_fraction,
    }
    payload.update(overrides)
    return PlanningConfig(**payload)
