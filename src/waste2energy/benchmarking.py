# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .common import build_run_manifest, write_json
from .config import BENCHMARK_OUTPUTS_DIR
from .planning.inputs import load_planning_input_bundle
from .planning.optimization import _documented_allocation_fallback
from .planning.solve import (
    PlanningConfig,
    build_pathway_summary,
    build_portfolio_summary,
    build_scenario_summary,
    execute_planning_pipeline,
)


@dataclass(frozen=True)
class BenchmarkVariant:
    key: str
    description: str
    config: PlanningConfig
    comparator_family: str = "counterfactual_optimizer"
    allocation_mode: str = "optimizer"


def build_default_benchmark_variants(base_config: PlanningConfig | None = None) -> list[BenchmarkVariant]:
    baseline = base_config or PlanningConfig()
    variants = [
        BenchmarkVariant(
            key="baseline_evidence_aware",
            description="Current evidence-aware robust planning baseline.",
            config=baseline,
            comparator_family="evidence_aware_optimizer",
        ),
        BenchmarkVariant(
            key="classic_multiobjective_optimizer",
            description="Classic weighted multi-objective optimizer without evidence-aware or robustness-aware adjustments.",
            config=replace(
                baseline,
                robustness_factor=0.0,
                partial_surrogate_weight=1.0,
                static_fallback_weight=1.0,
                unsupported_pathway_weight=1.0,
                partial_surrogate_uncertainty_multiplier=1.0,
                static_fallback_uncertainty_multiplier=1.0,
                unsupported_pathway_uncertainty_multiplier=1.0,
                partial_surrogate_information_premium_usd_per_ton=0.0,
                static_fallback_information_premium_usd_per_ton=0.0,
                unsupported_pathway_information_premium_usd_per_ton=0.0,
            ),
            comparator_family="classic_multiobjective_baseline",
        ),
        BenchmarkVariant(
            key="no_robustness_penalty",
            description="Remove uncertainty-aware robustness utility from score construction.",
            config=replace(baseline, robustness_factor=0.0),
            comparator_family="counterfactual_optimizer",
        ),
        BenchmarkVariant(
            key="uncertainty_penalty_max_interval",
            description="Use the maximum target-level interval-derived uncertainty ratio inside the robustness penalty.",
            config=replace(baseline, uncertainty_penalty_mode="max_interval_ratio"),
            comparator_family="uncertainty_mode_stress",
        ),
        BenchmarkVariant(
            key="uncertainty_penalty_combined_only",
            description="Ignore target-level interval aggregation and rely only on an explicit combined uncertainty ratio.",
            config=replace(baseline, uncertainty_penalty_mode="combined_only"),
            comparator_family="uncertainty_mode_stress",
        ),
        BenchmarkVariant(
            key="no_evidence_penalty",
            description="Neutralize evidence-based weighting and information-deficit premiums.",
            config=replace(
                baseline,
                partial_surrogate_weight=1.0,
                static_fallback_weight=1.0,
                unsupported_pathway_weight=1.0,
                partial_surrogate_uncertainty_multiplier=1.0,
                static_fallback_uncertainty_multiplier=1.0,
                unsupported_pathway_uncertainty_multiplier=1.0,
                partial_surrogate_information_premium_usd_per_ton=0.0,
                static_fallback_information_premium_usd_per_ton=0.0,
                unsupported_pathway_information_premium_usd_per_ton=0.0,
            ),
            comparator_family="counterfactual_optimizer",
        ),
        BenchmarkVariant(
            key="no_carbon_constraint",
            description="Relax carbon budget so planning is not carbon-cap constrained.",
            config=replace(baseline, carbon_budget_factor=1000.0),
            comparator_family="counterfactual_optimizer",
        ),
        BenchmarkVariant(
            key="ranking_only_unconstrained",
            description="Remove portfolio diversification and share caps so ranking pressure dominates.",
            config=replace(
                baseline,
                max_candidate_share=1.0,
                max_subtype_share=1.0,
                min_distinct_subtypes=1,
                enforce_candidate_cap=False,
                enforce_subtype_cap=False,
                enforce_max_selected=False,
                enforce_min_distinct_subtypes=False,
                max_portfolio_candidates=max(12, baseline.max_portfolio_candidates),
            ),
            comparator_family="counterfactual_optimizer",
        ),
        BenchmarkVariant(
            key="greedy_weighted_score_heuristic",
            description="Simple greedy heuristic that allocates feed by descending weighted score instead of solving the portfolio optimization.",
            config=baseline,
            comparator_family="heuristic_baseline",
            allocation_mode="greedy",
        ),
    ]
    return variants


def run_planning_benchmark_suite(
    *,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    base_config: PlanningConfig | None = None,
    bootstrap_replicates: int = 0,
    bootstrap_random_seed: int = 42,
) -> dict[str, object]:
    bundle = load_planning_input_bundle(dataset_path=dataset_path)
    variants = build_default_benchmark_variants(base_config)

    summary_rows: list[dict[str, object]] = []
    allocation_rows: list[pd.DataFrame] = []
    scenario_rows: list[pd.DataFrame] = []
    shift_rows: list[dict[str, object]] = []
    diagnostics_rows: list[pd.DataFrame] = []
    baseline_allocations: pd.DataFrame | None = None
    baseline_summary: pd.DataFrame | None = None

    for variant in variants:
        execution = _execute_benchmark_variant(bundle=bundle, variant=variant)
        allocations = execution["portfolio_allocations"].copy()
        scenario_summary = execution["scenario_summary"].copy()
        diagnostics = execution["optimization_diagnostics"].copy()
        portfolio_summary = build_portfolio_summary(allocations, execution["scenario_constraints"]).copy()

        if not allocations.empty:
            allocations["benchmark_variant"] = variant.key
            allocations["comparator_family"] = variant.comparator_family
            allocations["allocation_mode"] = variant.allocation_mode
            allocation_rows.append(allocations)
        if not scenario_summary.empty:
            scenario_summary["benchmark_variant"] = variant.key
            scenario_summary["comparator_family"] = variant.comparator_family
            scenario_summary["allocation_mode"] = variant.allocation_mode
            scenario_rows.append(scenario_summary)
        if not diagnostics.empty:
            diagnostics["benchmark_variant"] = variant.key
            diagnostics["comparator_family"] = variant.comparator_family
            diagnostics["allocation_mode"] = variant.allocation_mode
            diagnostics_rows.append(diagnostics)

        variant_summary = _build_variant_summary(
            variant=variant,
            portfolio_summary=portfolio_summary,
            scenario_summary=scenario_summary,
            allocations=allocations,
        )
        summary_rows.extend(variant_summary)

        if variant.key == "baseline_evidence_aware":
            baseline_allocations = allocations.copy()
            baseline_summary = portfolio_summary.copy()
        else:
            shift_rows.extend(
                _build_variant_shift_rows(
                    variant=variant,
                    variant_key=variant.key,
                    baseline_allocations=baseline_allocations,
                    variant_allocations=allocations,
                    baseline_summary=baseline_summary,
                variant_summary=portfolio_summary,
            )
        )

    bootstrap_shift_frame, bootstrap_stats_frame = _run_bootstrap_benchmark_analysis(
        bundle=bundle,
        variants=variants,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_random_seed=bootstrap_random_seed,
    )

    summary_frame = pd.DataFrame(summary_rows).sort_values(
        ["scenario_name", "benchmark_variant"]
    ).reset_index(drop=True)
    allocation_frame = (
        pd.concat(allocation_rows, ignore_index=True) if allocation_rows else pd.DataFrame()
    )
    scenario_frame = pd.concat(scenario_rows, ignore_index=True) if scenario_rows else pd.DataFrame()
    diagnostics_frame = pd.concat(diagnostics_rows, ignore_index=True) if diagnostics_rows else pd.DataFrame()
    shift_frame = pd.DataFrame(shift_rows).sort_values(
        ["scenario_name", "benchmark_variant"]
    ).reset_index(drop=True) if shift_rows else pd.DataFrame()

    outputs = write_benchmark_outputs(
        summary_frame=summary_frame,
        allocation_frame=allocation_frame,
        scenario_frame=scenario_frame,
        shift_frame=shift_frame,
        diagnostics_frame=diagnostics_frame,
        bootstrap_shift_frame=bootstrap_shift_frame,
        bootstrap_stats_frame=bootstrap_stats_frame,
        output_dir=output_dir,
        dataset_path=str(bundle.dataset_path),
        variants=variants,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_random_seed=bootstrap_random_seed,
    )
    return {
        "dataset_path": str(bundle.dataset_path),
        "variant_count": len(variants),
        "benchmark_variant_keys": [variant.key for variant in variants],
        "bootstrap_replicates": bootstrap_replicates,
        "outputs": outputs,
    }


def write_benchmark_outputs(
    *,
    summary_frame: pd.DataFrame,
    allocation_frame: pd.DataFrame,
    scenario_frame: pd.DataFrame,
    shift_frame: pd.DataFrame,
    diagnostics_frame: pd.DataFrame,
    bootstrap_shift_frame: pd.DataFrame,
    bootstrap_stats_frame: pd.DataFrame,
    output_dir: str | None,
    dataset_path: str,
    variants: list[BenchmarkVariant],
    bootstrap_replicates: int,
    bootstrap_random_seed: int,
) -> dict[str, str]:
    target_dir = Path(output_dir) if output_dir else BENCHMARK_OUTPUTS_DIR / "baseline"
    target_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "benchmark_summary": target_dir / "benchmark_summary.csv",
        "benchmark_allocations": target_dir / "benchmark_allocations.csv",
        "benchmark_scenario_summary": target_dir / "benchmark_scenario_summary.csv",
        "benchmark_shift_summary": target_dir / "benchmark_shift_summary.csv",
        "benchmark_diagnostics": target_dir / "benchmark_diagnostics.csv",
        "benchmark_bootstrap_shift_samples": target_dir / "benchmark_bootstrap_shift_samples.csv",
        "benchmark_statistical_summary": target_dir / "benchmark_statistical_summary.csv",
        "run_config": target_dir / "run_config.json",
    }

    summary_frame.to_csv(outputs["benchmark_summary"], index=False)
    allocation_frame.to_csv(outputs["benchmark_allocations"], index=False)
    scenario_frame.to_csv(outputs["benchmark_scenario_summary"], index=False)
    shift_frame.to_csv(outputs["benchmark_shift_summary"], index=False)
    diagnostics_frame.to_csv(outputs["benchmark_diagnostics"], index=False)
    bootstrap_shift_frame.to_csv(outputs["benchmark_bootstrap_shift_samples"], index=False)
    bootstrap_stats_frame.to_csv(outputs["benchmark_statistical_summary"], index=False)
    write_json(
        outputs["run_config"],
        build_run_manifest(
            dataset_path=dataset_path,
            benchmark_variant_count=len(variants),
            benchmark_variants=[
                {"key": variant.key, "description": variant.description} for variant in variants
            ],
            summary_row_count=int(len(summary_frame)),
            shift_row_count=int(len(shift_frame)),
            bootstrap_replicate_count=int(bootstrap_replicates),
            bootstrap_random_seed=int(bootstrap_random_seed),
            bootstrap_shift_row_count=int(len(bootstrap_shift_frame)),
            bootstrap_statistical_summary_row_count=int(len(bootstrap_stats_frame)),
            output_files={key: str(path) for key, path in outputs.items()},
            purpose="Ablation and benchmark suite proving which planning conclusions depend on evidence-aware and constraint-aware design choices.",
        ),
    )
    return {key: str(path) for key, path in outputs.items()}


def _execute_benchmark_variant(
    *,
    bundle,
    variant: BenchmarkVariant,
) -> dict[str, object]:
    execution = execute_planning_pipeline(bundle=bundle, config=variant.config)
    if variant.allocation_mode == "optimizer":
        return execution
    if variant.allocation_mode == "greedy":
        return _replace_portfolio_with_greedy_baseline(execution=execution, config=variant.config)
    raise ValueError(f"Unsupported benchmark allocation mode: {variant.allocation_mode}")


def _replace_portfolio_with_greedy_baseline(
    *,
    execution: dict[str, object],
    config: PlanningConfig,
) -> dict[str, object]:
    scored = execution["scored"]
    scenario_constraints = execution["scenario_constraints"]
    recommendations = execution["scenario_recommendations"]
    constraint_map = scenario_constraints.set_index("scenario_name").to_dict("index")

    allocation_rows: list[pd.DataFrame] = []
    diagnostics_rows: list[dict[str, object]] = []
    for scenario_name, frame in scored.groupby("scenario_name", dropna=False):
        allocation = _documented_allocation_fallback(frame, constraint_map.get(scenario_name, {}), config)
        if not allocation.empty:
            allocation["scenario_name"] = scenario_name
            allocation_rows.append(allocation)
        diagnostics_rows.append(
            {
                "scenario_name": scenario_name,
                "solver_status": "benchmark_greedy_weighted_score",
                "solver_backend": "documented_greedy_baseline",
            }
        )

    portfolio_allocations = (
        pd.concat(allocation_rows, ignore_index=True) if allocation_rows else pd.DataFrame()
    )
    diagnostics = pd.DataFrame(diagnostics_rows).sort_values("scenario_name").reset_index(drop=True)
    portfolio_summary = build_portfolio_summary(portfolio_allocations, scenario_constraints)
    scenario_summary = build_scenario_summary(
        scored=scored,
        recommendations=recommendations,
        portfolio_summary=portfolio_summary,
        diagnostics=diagnostics,
    )
    pathway_summary = build_pathway_summary(
        scored=scored,
        portfolio_allocations=portfolio_allocations,
    )
    return {
        **execution,
        "portfolio_allocations": portfolio_allocations,
        "portfolio_summary": portfolio_summary,
        "scenario_summary": scenario_summary,
        "pathway_summary": pathway_summary,
        "optimization_diagnostics": diagnostics,
    }


def _run_bootstrap_benchmark_analysis(
    *,
    bundle,
    variants: list[BenchmarkVariant],
    bootstrap_replicates: int,
    bootstrap_random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_columns = [
        "bootstrap_replicate",
        "scenario_name",
        "benchmark_variant",
        "comparator_family",
        "allocation_mode",
        "portfolio_case_shift",
        "portfolio_pathway_shift",
        "baseline_selected_pathways",
        "variant_selected_pathways",
        "baseline_top_portfolio_case_id",
        "variant_top_portfolio_case_id",
        "delta_portfolio_score_mass",
        "delta_portfolio_carbon_load_kgco2e",
        "delta_scenario_feed_coverage_ratio",
    ]
    summary_columns = [
        "scenario_name",
        "benchmark_variant",
        "comparator_family",
        "allocation_mode",
        "bootstrap_replicate_count",
        "pathway_shift_count",
        "pathway_shift_rate",
        "pathway_shift_rate_ci_lower",
        "pathway_shift_rate_ci_upper",
        "case_shift_count",
        "case_shift_rate",
        "case_shift_rate_ci_lower",
        "case_shift_rate_ci_upper",
        "delta_portfolio_score_mass_mean",
        "delta_portfolio_score_mass_median",
        "delta_portfolio_score_mass_ci_lower",
        "delta_portfolio_score_mass_ci_upper",
        "delta_portfolio_score_mass_ci_excludes_zero",
        "delta_portfolio_score_mass_sign_agreement_rate",
        "delta_portfolio_score_mass_empirical_p_value",
        "delta_portfolio_score_mass_direction",
        "delta_portfolio_carbon_load_kgco2e_mean",
        "delta_portfolio_carbon_load_kgco2e_median",
        "delta_portfolio_carbon_load_kgco2e_ci_lower",
        "delta_portfolio_carbon_load_kgco2e_ci_upper",
        "delta_portfolio_carbon_load_kgco2e_ci_excludes_zero",
        "delta_portfolio_carbon_load_kgco2e_sign_agreement_rate",
        "delta_portfolio_carbon_load_kgco2e_empirical_p_value",
        "delta_portfolio_carbon_load_kgco2e_direction",
        "delta_scenario_feed_coverage_ratio_mean",
        "delta_scenario_feed_coverage_ratio_median",
        "delta_scenario_feed_coverage_ratio_ci_lower",
        "delta_scenario_feed_coverage_ratio_ci_upper",
        "delta_scenario_feed_coverage_ratio_ci_excludes_zero",
        "delta_scenario_feed_coverage_ratio_sign_agreement_rate",
        "delta_scenario_feed_coverage_ratio_empirical_p_value",
        "delta_scenario_feed_coverage_ratio_direction",
        "effect_significance_tier",
    ]
    if bootstrap_replicates <= 0:
        return pd.DataFrame(columns=sample_columns), pd.DataFrame(columns=summary_columns)

    baseline_variant = next(
        (variant for variant in variants if variant.key == "baseline_evidence_aware"),
        None,
    )
    if baseline_variant is None:
        return pd.DataFrame(columns=sample_columns), pd.DataFrame(columns=summary_columns)

    rng = np.random.default_rng(bootstrap_random_seed)
    rows: list[dict[str, object]] = []
    for replicate_idx in range(bootstrap_replicates):
        boot_bundle = _bootstrap_planning_bundle(bundle=bundle, rng=rng, replicate_index=replicate_idx)
        baseline_execution = _execute_benchmark_variant(bundle=boot_bundle, variant=baseline_variant)
        baseline_allocations = baseline_execution["portfolio_allocations"].copy()
        baseline_summary = baseline_execution["portfolio_summary"].copy()

        for variant in variants:
            if variant.key == baseline_variant.key:
                continue
            execution = _execute_benchmark_variant(bundle=boot_bundle, variant=variant)
            replicate_shifts = _build_variant_shift_rows(
                variant=variant,
                variant_key=variant.key,
                baseline_allocations=baseline_allocations,
                variant_allocations=execution["portfolio_allocations"].copy(),
                baseline_summary=baseline_summary,
                variant_summary=execution["portfolio_summary"].copy(),
            )
            for row in replicate_shifts:
                rows.append(
                    {
                        "bootstrap_replicate": int(replicate_idx + 1),
                        **row,
                    }
                )

    sample_frame = pd.DataFrame(rows, columns=sample_columns)
    if sample_frame.empty:
        return sample_frame, pd.DataFrame(columns=summary_columns)
    summary_frame = _summarize_bootstrap_shift_samples(sample_frame)
    return sample_frame, summary_frame[summary_columns]


def _build_variant_summary(
    *,
    variant: BenchmarkVariant,
    portfolio_summary: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    allocations: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    summary_map = (
        portfolio_summary.set_index("scenario_name").to_dict("index") if not portfolio_summary.empty else {}
    )
    top_case_map = (
        scenario_summary.set_index("scenario_name").to_dict("index") if not scenario_summary.empty else {}
    )
    allocation_map = {
        scenario_name: frame.copy()
        for scenario_name, frame in allocations.groupby("scenario_name", dropna=False)
    } if not allocations.empty else {}

    scenario_names = sorted(set(summary_map) | set(top_case_map) | set(allocation_map))
    for scenario_name in scenario_names:
        summary_row = summary_map.get(scenario_name, {})
        top_case_row = top_case_map.get(scenario_name, {})
        allocation_frame = allocation_map.get(scenario_name, pd.DataFrame())
        selected_pathways = "|".join(sorted(allocation_frame.get("pathway", pd.Series(dtype="object")).astype(str).unique()))
        rows.append(
            {
                "scenario_name": scenario_name,
                "benchmark_variant": variant.key,
                "benchmark_description": variant.description,
                "comparator_family": variant.comparator_family,
                "allocation_mode": variant.allocation_mode,
                "top_ranked_case_id": top_case_row.get("top_ranked_case_id", ""),
                "top_portfolio_case_id": summary_row.get("top_portfolio_case_id", ""),
                "selected_pathways": selected_pathways,
                "selected_candidate_count": int(summary_row.get("selected_candidate_count", 0) or 0),
                "portfolio_fill_ratio": _float_or_zero(summary_row.get("portfolio_fill_ratio")),
                "scenario_feed_coverage_ratio": _float_or_zero(summary_row.get("scenario_feed_coverage_ratio")),
                "portfolio_energy_objective": _float_or_zero(summary_row.get("portfolio_energy_objective")),
                "portfolio_environment_objective": _float_or_zero(summary_row.get("portfolio_environment_objective")),
                "portfolio_cost_objective": _float_or_zero(summary_row.get("portfolio_cost_objective")),
                "portfolio_score_mass": _float_or_zero(summary_row.get("portfolio_score_mass")),
                "portfolio_carbon_load_kgco2e": _float_or_zero(summary_row.get("portfolio_carbon_load_kgco2e")),
            }
        )
    return rows


def _build_variant_shift_rows(
    *,
    variant: BenchmarkVariant,
    variant_key: str,
    baseline_allocations: pd.DataFrame | None,
    variant_allocations: pd.DataFrame,
    baseline_summary: pd.DataFrame | None,
    variant_summary: pd.DataFrame,
) -> list[dict[str, object]]:
    if baseline_summary is None:
        return []
    rows: list[dict[str, object]] = []
    baseline_allocation_map = (
        {
            scenario_name: frame.copy()
            for scenario_name, frame in baseline_allocations.groupby("scenario_name", dropna=False)
        }
        if baseline_allocations is not None and not baseline_allocations.empty
        else {}
    )
    variant_allocation_map = {
        scenario_name: frame.copy()
        for scenario_name, frame in variant_allocations.groupby("scenario_name", dropna=False)
    } if not variant_allocations.empty else {}
    baseline_summary_map = baseline_summary.set_index("scenario_name").to_dict("index")
    variant_summary_map = variant_summary.set_index("scenario_name").to_dict("index") if not variant_summary.empty else {}

    scenario_names = sorted(set(baseline_summary_map) | set(variant_summary_map))
    for scenario_name in scenario_names:
        baseline_frame = baseline_allocation_map.get(scenario_name, pd.DataFrame())
        current_frame = variant_allocation_map.get(scenario_name, pd.DataFrame())
        baseline_cases = "|".join(baseline_frame.get("optimization_case_id", pd.Series(dtype="object")).astype(str).tolist())
        current_cases = "|".join(current_frame.get("optimization_case_id", pd.Series(dtype="object")).astype(str).tolist())
        baseline_pathways = "|".join(sorted(baseline_frame.get("pathway", pd.Series(dtype="object")).astype(str).unique()))
        current_pathways = "|".join(sorted(current_frame.get("pathway", pd.Series(dtype="object")).astype(str).unique()))
        base_summary_row = baseline_summary_map.get(scenario_name, {})
        current_summary_row = variant_summary_map.get(scenario_name, {})
        rows.append(
            {
                "scenario_name": scenario_name,
                "benchmark_variant": variant_key,
                "comparator_family": variant.comparator_family,
                "allocation_mode": variant.allocation_mode,
                "portfolio_case_shift": "changed" if baseline_cases != current_cases else "unchanged",
                "portfolio_pathway_shift": "changed" if baseline_pathways != current_pathways else "unchanged",
                "baseline_selected_pathways": baseline_pathways,
                "variant_selected_pathways": current_pathways,
                "baseline_top_portfolio_case_id": base_summary_row.get("top_portfolio_case_id", ""),
                "variant_top_portfolio_case_id": current_summary_row.get("top_portfolio_case_id", ""),
                "delta_portfolio_score_mass": _float_or_zero(current_summary_row.get("portfolio_score_mass"))
                - _float_or_zero(base_summary_row.get("portfolio_score_mass")),
                "delta_portfolio_carbon_load_kgco2e": _float_or_zero(current_summary_row.get("portfolio_carbon_load_kgco2e"))
                - _float_or_zero(base_summary_row.get("portfolio_carbon_load_kgco2e")),
                "delta_scenario_feed_coverage_ratio": _float_or_zero(current_summary_row.get("scenario_feed_coverage_ratio"))
                - _float_or_zero(base_summary_row.get("scenario_feed_coverage_ratio")),
            }
        )
    return rows


def _bootstrap_planning_bundle(
    *,
    bundle,
    rng: np.random.Generator,
    replicate_index: int,
):
    sampled_frames: list[pd.DataFrame] = []
    for scenario_name, frame in bundle.frame.groupby("scenario_name", dropna=False, sort=False):
        if frame.empty:
            continue
        sampled = frame.sample(
            n=len(frame),
            replace=True,
            random_state=int(rng.integers(0, 2**31 - 1)),
        ).reset_index(drop=True)
        if "optimization_case_id" in sampled.columns:
            sampled["optimization_case_id"] = sampled["optimization_case_id"].astype(str) + sampled.index.map(
                lambda idx: f"::bootstrap_{replicate_index + 1:03d}_{scenario_name}_{idx:04d}"
            )
        sampled_frames.append(sampled)
    if not sampled_frames:
        return bundle
    return replace(bundle, frame=pd.concat(sampled_frames, ignore_index=True))


def _summarize_bootstrap_shift_samples(sample_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scenario_name, benchmark_variant), frame in sample_frame.groupby(
        ["scenario_name", "benchmark_variant"],
        dropna=False,
    ):
        score_stats = _bootstrap_metric_statistics(frame["delta_portfolio_score_mass"])
        carbon_stats = _bootstrap_metric_statistics(frame["delta_portfolio_carbon_load_kgco2e"])
        coverage_stats = _bootstrap_metric_statistics(frame["delta_scenario_feed_coverage_ratio"])
        pathway_shift_count = int(frame["portfolio_pathway_shift"].astype(str).eq("changed").sum())
        case_shift_count = int(frame["portfolio_case_shift"].astype(str).eq("changed").sum())
        replicate_count = int(len(frame))
        pathway_shift_rate = pathway_shift_count / replicate_count if replicate_count > 0 else 0.0
        case_shift_rate = case_shift_count / replicate_count if replicate_count > 0 else 0.0
        pathway_shift_interval = _wilson_interval(success_count=pathway_shift_count, total_count=replicate_count)
        case_shift_interval = _wilson_interval(success_count=case_shift_count, total_count=replicate_count)
        rows.append(
            {
                "scenario_name": scenario_name,
                "benchmark_variant": benchmark_variant,
                "comparator_family": frame["comparator_family"].iloc[0],
                "allocation_mode": frame["allocation_mode"].iloc[0],
                "bootstrap_replicate_count": replicate_count,
                "pathway_shift_count": pathway_shift_count,
                "pathway_shift_rate": pathway_shift_rate,
                "pathway_shift_rate_ci_lower": pathway_shift_interval["ci_lower"],
                "pathway_shift_rate_ci_upper": pathway_shift_interval["ci_upper"],
                "case_shift_count": case_shift_count,
                "case_shift_rate": case_shift_rate,
                "case_shift_rate_ci_lower": case_shift_interval["ci_lower"],
                "case_shift_rate_ci_upper": case_shift_interval["ci_upper"],
                "delta_portfolio_score_mass_mean": score_stats["mean"],
                "delta_portfolio_score_mass_median": score_stats["median"],
                "delta_portfolio_score_mass_ci_lower": score_stats["ci_lower"],
                "delta_portfolio_score_mass_ci_upper": score_stats["ci_upper"],
                "delta_portfolio_score_mass_ci_excludes_zero": score_stats["ci_excludes_zero"],
                "delta_portfolio_score_mass_sign_agreement_rate": score_stats["sign_agreement_rate"],
                "delta_portfolio_score_mass_empirical_p_value": score_stats["empirical_p_value"],
                "delta_portfolio_score_mass_direction": score_stats["direction"],
                "delta_portfolio_carbon_load_kgco2e_mean": carbon_stats["mean"],
                "delta_portfolio_carbon_load_kgco2e_median": carbon_stats["median"],
                "delta_portfolio_carbon_load_kgco2e_ci_lower": carbon_stats["ci_lower"],
                "delta_portfolio_carbon_load_kgco2e_ci_upper": carbon_stats["ci_upper"],
                "delta_portfolio_carbon_load_kgco2e_ci_excludes_zero": carbon_stats["ci_excludes_zero"],
                "delta_portfolio_carbon_load_kgco2e_sign_agreement_rate": carbon_stats["sign_agreement_rate"],
                "delta_portfolio_carbon_load_kgco2e_empirical_p_value": carbon_stats["empirical_p_value"],
                "delta_portfolio_carbon_load_kgco2e_direction": carbon_stats["direction"],
                "delta_scenario_feed_coverage_ratio_mean": coverage_stats["mean"],
                "delta_scenario_feed_coverage_ratio_median": coverage_stats["median"],
                "delta_scenario_feed_coverage_ratio_ci_lower": coverage_stats["ci_lower"],
                "delta_scenario_feed_coverage_ratio_ci_upper": coverage_stats["ci_upper"],
                "delta_scenario_feed_coverage_ratio_ci_excludes_zero": coverage_stats["ci_excludes_zero"],
                "delta_scenario_feed_coverage_ratio_sign_agreement_rate": coverage_stats["sign_agreement_rate"],
                "delta_scenario_feed_coverage_ratio_empirical_p_value": coverage_stats["empirical_p_value"],
                "delta_scenario_feed_coverage_ratio_direction": coverage_stats["direction"],
                "effect_significance_tier": _classify_bootstrap_effect_tier(
                    pathway_shift_rate=pathway_shift_rate,
                    pathway_shift_rate_ci_lower=pathway_shift_interval["ci_lower"],
                    case_shift_rate=case_shift_rate,
                    case_shift_rate_ci_lower=case_shift_interval["ci_lower"],
                    score_ci_lower=score_stats["ci_lower"],
                    score_ci_upper=score_stats["ci_upper"],
                    score_empirical_p_value=score_stats["empirical_p_value"],
                    carbon_ci_lower=carbon_stats["ci_lower"],
                    carbon_ci_upper=carbon_stats["ci_upper"],
                    carbon_empirical_p_value=carbon_stats["empirical_p_value"],
                    coverage_ci_lower=coverage_stats["ci_lower"],
                    coverage_ci_upper=coverage_stats["ci_upper"],
                    coverage_empirical_p_value=coverage_stats["empirical_p_value"],
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario_name", "benchmark_variant"]).reset_index(drop=True)


def _bootstrap_metric_statistics(series: pd.Series) -> dict[str, float | object]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            "mean": pd.NA,
            "median": pd.NA,
            "ci_lower": pd.NA,
            "ci_upper": pd.NA,
            "ci_excludes_zero": False,
            "sign_agreement_rate": pd.NA,
            "empirical_p_value": pd.NA,
            "direction": "not_available",
        }
    positive_count = int(values.gt(1e-12).sum())
    negative_count = int(values.lt(-1e-12).sum())
    nonzero_count = positive_count + negative_count
    if nonzero_count <= 0:
        sign_agreement_rate = 0.0
        empirical_p_value = 1.0
        direction = "mixed"
    else:
        sign_agreement_rate = max(positive_count, negative_count) / nonzero_count
        empirical_p_value = min(1.0, 2.0 * min(positive_count, negative_count) / nonzero_count)
        direction = "positive" if positive_count > negative_count else "negative" if negative_count > positive_count else "mixed"
    ci_lower = float(values.quantile(0.025))
    ci_upper = float(values.quantile(0.975))
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_excludes_zero": _interval_excludes_zero(ci_lower, ci_upper),
        "sign_agreement_rate": float(sign_agreement_rate),
        "empirical_p_value": float(empirical_p_value),
        "direction": direction,
    }


def _classify_bootstrap_effect_tier(
    *,
    pathway_shift_rate: float,
    pathway_shift_rate_ci_lower: float | object,
    case_shift_rate: float,
    case_shift_rate_ci_lower: float | object,
    score_ci_lower: float | object,
    score_ci_upper: float | object,
    score_empirical_p_value: float | object,
    carbon_ci_lower: float | object,
    carbon_ci_upper: float | object,
    carbon_empirical_p_value: float | object,
    coverage_ci_lower: float | object,
    coverage_ci_upper: float | object,
    coverage_empirical_p_value: float | object,
) -> str:
    score_excludes_zero = _interval_excludes_zero(score_ci_lower, score_ci_upper)
    carbon_excludes_zero = _interval_excludes_zero(carbon_ci_lower, carbon_ci_upper)
    coverage_excludes_zero = _interval_excludes_zero(coverage_ci_lower, coverage_ci_upper)
    strong_directional_p = _any_empirical_pvalue_at_or_below(
        [score_empirical_p_value, carbon_empirical_p_value, coverage_empirical_p_value],
        threshold=0.05,
    )
    weak_directional_p = _any_empirical_pvalue_at_or_below(
        [score_empirical_p_value, carbon_empirical_p_value, coverage_empirical_p_value],
        threshold=0.10,
    )
    pathway_ci_lower = _optional_float(pathway_shift_rate_ci_lower)
    case_ci_lower = _optional_float(case_shift_rate_ci_lower)
    if pathway_shift_rate >= 0.80 and (pd.isna(pathway_ci_lower) or float(pathway_ci_lower) >= 0.50):
        return "highly_consistent"
    if case_shift_rate >= 0.80 and (
        score_excludes_zero or carbon_excludes_zero or coverage_excludes_zero or strong_directional_p
    ):
        return "directionally_consistent"
    if (
        case_shift_rate >= 0.50
        or (pd.notna(case_ci_lower) and float(case_ci_lower) >= 0.25)
        or score_excludes_zero
        or carbon_excludes_zero
        or coverage_excludes_zero
        or weak_directional_p
    ):
        return "suggestive"
    return "unstable"


def _wilson_interval(
    *,
    success_count: int,
    total_count: int,
    z: float = 1.959963984540054,
) -> dict[str, float | object]:
    if total_count <= 0:
        return {"ci_lower": pd.NA, "ci_upper": pd.NA}
    proportion = success_count / total_count
    denominator = 1.0 + (z**2 / total_count)
    center = (proportion + z**2 / (2.0 * total_count)) / denominator
    margin = (
        z
        * np.sqrt((proportion * (1.0 - proportion) / total_count) + (z**2 / (4.0 * total_count**2)))
        / denominator
    )
    return {
        "ci_lower": float(max(0.0, center - margin)),
        "ci_upper": float(min(1.0, center + margin)),
    }


def _any_empirical_pvalue_at_or_below(values: list[float | object], *, threshold: float) -> bool:
    for value in values:
        numeric = _optional_float(value)
        if pd.notna(numeric) and float(numeric) <= threshold:
            return True
    return False


def _interval_excludes_zero(lower: float | object, upper: float | object) -> bool:
    lower_value = pd.to_numeric(pd.Series([lower]), errors="coerce").iloc[0]
    upper_value = pd.to_numeric(pd.Series([upper]), errors="coerce").iloc[0]
    if pd.isna(lower_value) or pd.isna(upper_value):
        return False
    return (float(lower_value) > 0.0 and float(upper_value) > 0.0) or (
        float(lower_value) < 0.0 and float(upper_value) < 0.0
    )


def _float_or_zero(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0.0).iloc[0]
    return float(numeric)


def _optional_float(value: object) -> float | object:
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
