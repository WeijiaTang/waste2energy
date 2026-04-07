# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from ..config import (
    DEFAULT_OBJECTIVE_WEIGHT_PRESET,
    ObjectiveWeightSystem,
    get_objective_weight_system,
)
from .artifacts import write_planning_outputs
from .constraints import build_scenario_constraints
from .inputs import load_planning_input_bundle, load_scenario_external_evidence_table
from .optimization import (
    build_candidate_score_frame,
    generate_pareto_front,
    solve_scenario_optimization,
)
from .objectives import assemble_objective_frame
from .surrogate_evaluator import build_surrogate_predictions


@dataclass(frozen=True)
class ScenarioExternalEvidence:
    scenario_name: str
    feedstock_scale_factor: float = 1.0
    feedstock_cost_elasticity: float = 0.0
    carbon_tax_usd_per_ton_co2e: float = 0.0
    evidence_source: str = ""
    evidence_reference: str = ""
    evidence_rationale: str = ""


@dataclass(frozen=True)
class PlanningConfig:
    objective_weight_preset: str = DEFAULT_OBJECTIVE_WEIGHT_PRESET
    objective_weight_system: ObjectiveWeightSystem = field(
        default_factory=lambda: get_objective_weight_system()
    )
    top_k_per_scenario: int = 5
    max_portfolio_candidates: int = 3
    max_candidate_share: float = 0.45
    max_subtype_share: float = 0.60
    min_distinct_subtypes: int = 2
    deployable_capacity_fraction: float = 0.85
    robustness_factor: float = 0.35
    carbon_budget_factor: float = 1.00
    constraint_relaxation_ratio: float = 1.00
    subtype_relaxation_ratio: float = 1.00
    enforce_candidate_cap: bool = True
    enforce_subtype_cap: bool = True
    enforce_max_selected: bool = True
    enforce_min_distinct_subtypes: bool = True
    scenario_metric_variance_scale: float = 1.00
    scenario_external_evidence_table_path: str | None = None
    scenario_external_evidence: tuple[ScenarioExternalEvidence, ...] = field(default_factory=tuple)
    optimization_method: str = "auto"
    pyomo_solver_preference: str = "auto"
    pareto_point_count: int = 12
    enable_pareto_export: bool = True
    allow_surrogate_fallback: bool = True
    partial_surrogate_weight: float = 0.70
    static_fallback_weight: float = 0.35
    unsupported_pathway_weight: float = 0.15
    partial_surrogate_uncertainty_multiplier: float = 1.35
    static_fallback_uncertainty_multiplier: float = 2.10
    unsupported_pathway_uncertainty_multiplier: float = 3.25
    partial_surrogate_information_premium_usd_per_ton: float = 8.0
    static_fallback_information_premium_usd_per_ton: float = 22.0
    unsupported_pathway_information_premium_usd_per_ton: float = 45.0

    @property
    def energy_weight(self) -> float:
        return self.objective_weight_system.energy

    @property
    def environment_weight(self) -> float:
        return self.objective_weight_system.environment

    @property
    def cost_weight(self) -> float:
        return self.objective_weight_system.cost

    def copy_with_weights(
        self,
        *,
        energy_weight: float | None = None,
        environment_weight: float | None = None,
        cost_weight: float | None = None,
    ) -> "PlanningConfig":
        updated_system = get_objective_weight_system(
            preset_name=self.objective_weight_preset,
            energy=self.energy_weight if energy_weight is None else energy_weight,
            environment=self.environment_weight if environment_weight is None else environment_weight,
            cost=self.cost_weight if cost_weight is None else cost_weight,
        )
        return replace(
            self,
            objective_weight_system=updated_system,
        )


def run_planning_baseline(
    dataset_path: str | None = None,
    output_dir: str | None = None,
    config: PlanningConfig | None = None,
) -> dict[str, object]:
    bundle = load_planning_input_bundle(dataset_path=dataset_path)
    active_config = _resolve_planning_config(config or PlanningConfig(), bundle.frame)
    _validate_config(active_config)
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
        surrogate_predictions=execution["surrogate_predictions"],
        optimization_diagnostics=execution["optimization_diagnostics"],
        planning_data_quality_summary=execution["planning_data_quality_summary"],
        planning_candidate_exclusions=execution["planning_candidate_exclusions"],
        scenario_external_evidence=execution["scenario_external_evidence"],
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
        "planner_variant": "surrogate_driven_robust_multiobjective_optimizer",
        "row_count": int(len(execution["scored"])),
        "pareto_candidate_count": int(len(execution["pareto_candidates"])),
        "recommendation_count": int(len(execution["scenario_recommendations"])),
        "portfolio_allocation_count": int(len(execution["portfolio_allocations"])),
        "outputs": outputs,
    }


def execute_planning_pipeline(bundle, config: PlanningConfig) -> dict[str, object]:
    active_config = _resolve_planning_config(config, bundle.frame)
    planning_frame = _attach_scenario_external_evidence(bundle.frame, active_config)
    surrogate_predictions = build_surrogate_predictions(planning_frame)
    objective_frame, readiness, data_quality_summary, candidate_exclusions = assemble_objective_frame(
        base_frame=planning_frame,
        surrogate_predictions=surrogate_predictions,
        robustness_factor=active_config.robustness_factor,
        real_cost_columns=bundle.real_cost_columns,
        config=active_config,
    )
    scenario_constraints = build_scenario_constraints(objective_frame, active_config)
    scored = score_cases(objective_frame, active_config)
    scenario_recommendations = build_scenario_recommendations(scored, active_config.top_k_per_scenario)
    portfolio_allocations, diagnostics = build_scenario_portfolios(scored, scenario_constraints, active_config)
    pareto_candidates = build_portfolio_pareto_candidates(scored, scenario_constraints, active_config)
    portfolio_summary = build_portfolio_summary(portfolio_allocations, scenario_constraints)
    scenario_summary = build_scenario_summary(
        scored=scored,
        recommendations=scenario_recommendations,
        portfolio_summary=portfolio_summary,
        diagnostics=diagnostics,
    )
    pathway_summary = build_pathway_summary(
        scored=scored,
        portfolio_allocations=portfolio_allocations,
    )
    return {
        "objective_frame": objective_frame,
        "objective_readiness": readiness,
        "scenario_constraints": scenario_constraints,
        "surrogate_predictions": surrogate_predictions,
        "scored": scored,
        "scenario_recommendations": scenario_recommendations,
        "pareto_candidates": pareto_candidates,
        "portfolio_allocations": portfolio_allocations,
        "portfolio_summary": portfolio_summary,
        "scenario_summary": scenario_summary,
        "pathway_summary": pathway_summary,
        "optimization_diagnostics": diagnostics,
        "planning_data_quality_summary": data_quality_summary,
        "planning_candidate_exclusions": candidate_exclusions,
        "scenario_external_evidence": _scenario_external_evidence_to_frame(active_config),
    }


def score_cases(frame: pd.DataFrame, config: PlanningConfig) -> pd.DataFrame:
    scenario_frames: list[pd.DataFrame] = []
    for _, scenario_frame in frame.groupby("scenario_name", dropna=False):
        scenario_frames.append(build_candidate_score_frame(scenario_frame, config))
    if not scenario_frames:
        return frame.copy()
    return pd.concat(scenario_frames, ignore_index=True).sort_values(
        ["scenario_name", "planning_score", "planning_energy_intensity_mj_per_ton"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_scenario_recommendations(scored: pd.DataFrame, top_k: int) -> pd.DataFrame:
    ranked = scored.copy()
    ranked["scenario_rank"] = ranked.groupby("scenario_name")["planning_score"].rank(
        method="first", ascending=False
    )
    return ranked[ranked["scenario_rank"] <= top_k].reset_index(drop=True)


def build_scenario_portfolios(
    scored: pd.DataFrame,
    scenario_constraints: pd.DataFrame,
    config: PlanningConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    constraint_map = scenario_constraints.set_index("scenario_name").to_dict("index")
    allocation_rows: list[pd.DataFrame] = []
    diagnostics_rows: list[dict[str, object]] = []

    for scenario_name, scenario_frame in scored.groupby("scenario_name", dropna=False):
        constraint = constraint_map.get(scenario_name, {})
        result = solve_scenario_optimization(scenario_frame, constraint, config)
        allocations = result.allocations.copy()
        if not allocations.empty:
            allocations["scenario_name"] = scenario_name
            allocation_rows.append(allocations)
        diagnostics_rows.append(
            {
                "scenario_name": scenario_name,
                **result.diagnostics,
            }
        )

    portfolio_allocations = (
        pd.concat(allocation_rows, ignore_index=True) if allocation_rows else pd.DataFrame()
    )
    diagnostics = pd.DataFrame(diagnostics_rows).sort_values("scenario_name").reset_index(drop=True)
    return portfolio_allocations, diagnostics


def build_portfolio_pareto_candidates(
    scored: pd.DataFrame,
    scenario_constraints: pd.DataFrame,
    config: PlanningConfig,
) -> pd.DataFrame:
    if not config.enable_pareto_export or config.pareto_point_count <= 0:
        return pd.DataFrame()
    constraint_map = scenario_constraints.set_index("scenario_name").to_dict("index")
    rows: list[pd.DataFrame] = []
    for scenario_name, scenario_frame in scored.groupby("scenario_name", dropna=False):
        pareto = generate_pareto_front(scenario_frame, constraint_map.get(scenario_name, {}), config)
        if pareto.empty:
            continue
        rows.append(pareto)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).reset_index(drop=True)


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
                "portfolio_carbon_load_kgco2e": _sum_column(allocation_frame, "allocated_carbon_load_kgco2e"),
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
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    top1 = recommendations[recommendations["scenario_rank"] == 1].copy()
    top1_columns = [
        "scenario_name",
        "optimization_case_id",
        "planning_score",
        "planning_energy_objective",
        "planning_environment_objective",
        "planning_cost_objective",
        "combined_uncertainty_ratio",
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
            "combined_uncertainty_ratio": "top_ranked_uncertainty_ratio",
            "process_temperature_c": "top_ranked_temperature_c",
            "residence_time_min": "top_ranked_residence_time_min",
            "manure_subtype": "top_ranked_manure_subtype",
        }
    )
    counts = scored.groupby("scenario_name").size().reset_index(name="candidate_count")
    return (
        counts.merge(top1, on="scenario_name", how="left")
        .merge(portfolio_summary, on="scenario_name", how="left")
        .merge(diagnostics, on="scenario_name", how="left")
        .sort_values("scenario_name")
        .reset_index(drop=True)
    )


def build_pathway_summary(
    *,
    scored: pd.DataFrame,
    portfolio_allocations: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if portfolio_allocations.empty or not {"scenario_name", "pathway"}.issubset(portfolio_allocations.columns):
        allocation_map: dict[tuple[object, object], pd.DataFrame] = {}
    else:
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
                "best_case_uncertainty_ratio": _optional_numeric_from_row(
                    best_case,
                    "combined_uncertainty_ratio",
                ),
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
                "portfolio_carbon_load_kgco2e": _sum_column(
                    allocation_frame, "allocated_carbon_load_kgco2e"
                ),
                "portfolio_top_case_id": _first_value(allocation_frame, "optimization_case_id"),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["scenario_name", "best_case_score"], ascending=[True, False]
    ).reset_index(drop=True)


def _sum_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        preview = ", ".join(frame.loc[values.isna(), "optimization_case_id"].astype(str).head(5).tolist())
        raise ValueError(
            f"Planning summary aggregation encountered missing/non-numeric values in '{column}' for allocation row(s): {preview}."
        )
    return float(values.sum())


def _first_value(frame: pd.DataFrame, column: str) -> object:
    if frame.empty or column not in frame.columns:
        return None
    return frame.iloc[0][column]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _optional_numeric_from_row(row: pd.Series, column: str) -> float | object:
    if column not in row.index:
        return pd.NA
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    if pd.isna(value):
        return pd.NA
    return float(value)


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
    if (
        config.enforce_candidate_cap
        and config.enforce_subtype_cap
        and config.max_subtype_share + 1e-9 < config.max_candidate_share
    ):
        raise ValueError("max_subtype_share must be greater than or equal to max_candidate_share.")
    if config.min_distinct_subtypes < 1:
        raise ValueError("min_distinct_subtypes must be at least 1.")
    if config.constraint_relaxation_ratio <= 0.0:
        raise ValueError("constraint_relaxation_ratio must be positive.")
    if config.subtype_relaxation_ratio <= 0.0:
        raise ValueError("subtype_relaxation_ratio must be positive.")
    if config.scenario_metric_variance_scale <= 0.0:
        raise ValueError("scenario_metric_variance_scale must be positive.")
    if not 0.0 < config.deployable_capacity_fraction <= 1.0:
        raise ValueError("deployable_capacity_fraction must be within (0, 1].")
    if not 0.0 <= config.robustness_factor <= 1.0:
        raise ValueError("robustness_factor must be within [0, 1].")
    if config.carbon_budget_factor <= 0.0:
        raise ValueError("carbon_budget_factor must be positive.")
    if config.pyomo_solver_preference not in {"auto", "appsi_highs", "highs", "glpk", "cbc"}:
        raise ValueError("pyomo_solver_preference must be one of: auto, appsi_highs, highs, glpk, cbc.")


def _resolve_planning_config(config: PlanningConfig, frame: pd.DataFrame) -> PlanningConfig:
    evidence = config.scenario_external_evidence
    table_path = config.scenario_external_evidence_table_path
    if not evidence:
        evidence_frame, resolved_path = load_scenario_external_evidence_table(table_path)
        _validate_external_evidence_coverage(evidence_frame, frame)
        evidence = tuple(
            ScenarioExternalEvidence(
                scenario_name=str(row.scenario_name),
                feedstock_scale_factor=float(row.feedstock_scale_factor),
                feedstock_cost_elasticity=float(row.feedstock_cost_elasticity),
                carbon_tax_usd_per_ton_co2e=float(row.carbon_tax_usd_per_ton_co2e),
                evidence_source=str(row.evidence_source),
                evidence_reference=str(row.evidence_reference),
                evidence_rationale=str(row.evidence_rationale),
            )
            for row in evidence_frame.itertuples(index=False)
        )
        table_path = str(resolved_path)
    return replace(
        config,
        scenario_external_evidence_table_path=table_path,
        scenario_external_evidence=evidence,
    )


def _validate_external_evidence_coverage(evidence_frame: pd.DataFrame, planning_frame: pd.DataFrame) -> None:
    expected = {
        str(row.scenario_name)
        for row in planning_frame[["scenario_name"]].drop_duplicates().itertuples(index=False)
    }
    observed = {
        str(row.scenario_name)
        for row in evidence_frame[["scenario_name"]].drop_duplicates().itertuples(index=False)
    }
    missing = sorted(expected - observed)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            "Scenario external evidence table does not cover all planning scenarios. "
            f"Missing: {preview}"
        )


def _scenario_external_evidence_to_frame(config: PlanningConfig) -> pd.DataFrame:
    rows = [
        {
            "scenario_name": evidence.scenario_name,
            "feedstock_scale_factor": evidence.feedstock_scale_factor,
            "feedstock_cost_elasticity": evidence.feedstock_cost_elasticity,
            "carbon_tax_usd_per_ton_co2e": evidence.carbon_tax_usd_per_ton_co2e,
            "evidence_source": evidence.evidence_source,
            "evidence_reference": evidence.evidence_reference,
            "evidence_rationale": evidence.evidence_rationale,
            "evidence_table_path": config.scenario_external_evidence_table_path or "",
        }
        for evidence in config.scenario_external_evidence
    ]
    return pd.DataFrame(rows).sort_values(["scenario_name"]).reset_index(drop=True) if rows else pd.DataFrame()


def _attach_scenario_external_evidence(frame: pd.DataFrame, config: PlanningConfig) -> pd.DataFrame:
    if frame.empty or not config.scenario_external_evidence:
        return frame.copy()
    evidence_frame = _scenario_external_evidence_to_frame(config).drop(columns=["evidence_table_path"], errors="ignore")
    return frame.merge(
        evidence_frame,
        on="scenario_name",
        how="left",
        validate="many_to_one",
    )
