# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import build_run_manifest, write_json
from ..config import BENCHMARK_OUTPUTS_DIR, get_objective_weight_system
from .inputs import PlanningInputBundle, load_planning_input_bundle
from .solve import PlanningConfig, execute_planning_pipeline

ETA_SWEEP_VALUES = (0.0, 0.15, 0.30, 0.50, 0.75, 1.0)
EVIDENCE_LADDER_PRESETS = {
    "mild": {
        "partial_surrogate_weight": 0.85,
        "static_fallback_weight": 0.60,
        "unsupported_pathway_weight": 0.40,
        "partial_surrogate_information_premium_usd_per_ton": 4.0,
        "static_fallback_information_premium_usd_per_ton": 12.0,
        "unsupported_pathway_information_premium_usd_per_ton": 24.0,
        "partial_surrogate_uncertainty_multiplier": 1.15,
        "static_fallback_uncertainty_multiplier": 1.50,
        "unsupported_pathway_uncertainty_multiplier": 2.00,
    },
    "current": {},
    "strict": {
        "partial_surrogate_weight": 0.50,
        "static_fallback_weight": 0.20,
        "unsupported_pathway_weight": 0.05,
        "partial_surrogate_information_premium_usd_per_ton": 16.0,
        "static_fallback_information_premium_usd_per_ton": 40.0,
        "unsupported_pathway_information_premium_usd_per_ton": 80.0,
        "partial_surrogate_uncertainty_multiplier": 1.75,
        "static_fallback_uncertainty_multiplier": 2.75,
        "unsupported_pathway_uncertainty_multiplier": 4.00,
    },
    "binary_support_only": {
        "partial_surrogate_weight": 0.0,
        "static_fallback_weight": 0.0,
        "unsupported_pathway_weight": 0.0,
        "partial_surrogate_information_premium_usd_per_ton": 40.0,
        "static_fallback_information_premium_usd_per_ton": 80.0,
        "unsupported_pathway_information_premium_usd_per_ton": 120.0,
        "partial_surrogate_uncertainty_multiplier": 2.50,
        "static_fallback_uncertainty_multiplier": 4.00,
        "unsupported_pathway_uncertainty_multiplier": 6.00,
    },
}


def run_targeted_planning_ablations(
    *,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    base_config: PlanningConfig | None = None,
    monte_carlo_replicates: int = 48,
    monte_carlo_random_seed: int = 42,
) -> dict[str, str]:
    bundle = load_planning_input_bundle(dataset_path=dataset_path)
    config = base_config or PlanningConfig(enable_pareto_export=False, pareto_point_count=0)
    rows: list[dict[str, object]] = []
    allocation_frames: list[pd.DataFrame] = []

    for cap_key, cap_config, cap_note in (
        (
            "locked_candidate_cap",
            config,
            "Re-runs the declared portfolio rule with the default 45% candidate cap to expose whether selected rows bind the cap.",
        ),
        (
            "candidate_cap_relaxed_100pct",
            replace(config, max_candidate_share=1.0),
            "Relaxes only the single-candidate cap to 100% while retaining subtype and diversity rules.",
        ),
        (
            "candidate_and_subtype_caps_relaxed",
            replace(
                config,
                max_candidate_share=1.0,
                max_subtype_share=1.0,
                enforce_min_distinct_subtypes=False,
            ),
            "Relaxes candidate and subtype caps and disables subtype-diversity forcing to test whether 90/10 is mechanically induced.",
        ),
    ):
        _append_execution_rows(
            rows,
            allocation_frames,
            execution=execute_planning_pipeline(bundle=bundle, config=cap_config),
            ablation_family="constraint_mechanism",
            ablation_key=cap_key,
            ablation_value=cap_key,
            note=cap_note,
        )

    algorithm_config = replace(config, htc_model_priority=_reprioritize_models(config.htc_model_priority, "xgboost"))
    _append_execution_rows(
        rows,
        allocation_frames,
        execution=execute_planning_pipeline(bundle=bundle, config=algorithm_config),
        ablation_family="algorithm",
        ablation_key="htc_xgboost_priority",
        ablation_value="xgboost",
        note="Forces the HTC surrogate selector to fall back to the legacy XGBoost preference order.",
    )

    for eta_value in ETA_SWEEP_VALUES:
        eta_config = replace(config, evidence_utility_factor=float(eta_value))
        _append_execution_rows(
            rows,
            allocation_frames,
            execution=execute_planning_pipeline(bundle=bundle, config=eta_config),
            ablation_family="evidence_sensitivity",
            ablation_key=f"eta_{eta_value:0.2f}",
            ablation_value=float(eta_value),
            note="Sweeps the evidence-utility coefficient that scales the optimizer-side evidence term.",
        )

    gate_config = replace(config, minimum_surrogate_artifact_test_r2=0.0)
    _append_execution_rows(
        rows,
        allocation_frames,
        execution=execute_planning_pipeline(bundle=bundle, config=gate_config),
        ablation_family="surrogate_evidence_gate",
        ablation_key="negative_r2_artifacts_fallback",
        ablation_value="minimum_test_r2_0",
        note="Requires selected surrogate artifacts to have non-negative test R2; negative-R2 artifacts fall back to documented candidate values.",
    )

    for ladder_name, ladder_params in EVIDENCE_LADDER_PRESETS.items():
        ladder_config = replace(config, **ladder_params) if ladder_params else config
        _append_execution_rows(
            rows,
            allocation_frames,
            execution=execute_planning_pipeline(bundle=bundle, config=ladder_config),
            ablation_family="evidence_ladder_sensitivity",
            ablation_key=f"evidence_ladder_{ladder_name}",
            ablation_value=ladder_name,
            note="Changes the full evidence weight, uncertainty-multiplier, and information-premium ladder rather than only the eta coefficient.",
        )

    for preset_name in ("balanced_cleaner_production", "energy_priority", "environment_priority", "cost_guardrail"):
        weight_config = replace(
            config,
            objective_weight_preset=preset_name,
            objective_weight_system=get_objective_weight_system(preset_name=preset_name),
        )
        _append_execution_rows(
            rows,
            allocation_frames,
            execution=execute_planning_pipeline(bundle=bundle, config=weight_config),
            ablation_family="objective_weight_sensitivity",
            ablation_key=preset_name,
            ablation_value=preset_name,
            note="Switches among pre-registered objective-weight presets to test whether pathway choice reflects management priorities rather than a fixed score weighting.",
        )

    for min_share in (0.0, 0.10, 0.20):
        ad_floor_config = replace(
            config,
            min_pathway_share=(("ad", float(min_share)),),
            enforce_min_distinct_subtypes=False,
        )
        _append_execution_rows(
            rows,
            allocation_frames,
            execution=execute_planning_pipeline(bundle=bundle, config=ad_floor_config),
            ablation_family="ad_complementarity",
            ablation_key=f"ad_min_share_{int(min_share * 100):02d}pct",
            ablation_value=float(min_share),
            note="Imposes a minimum AD processing share to quantify cost, carbon, and energy tradeoffs for biological-treatment complementarity.",
        )

    for max_share in (1.0, 0.80, 0.60):
        cap_config = replace(
            config,
            max_pathway_share=(("pyrolysis", float(max_share)),),
            enforce_min_distinct_subtypes=False,
        )
        _append_execution_rows(
            rows,
            allocation_frames,
            execution=execute_planning_pipeline(bundle=bundle, config=cap_config),
            ablation_family="pathway_cap_sensitivity",
            ablation_key=f"pyrolysis_max_share_{int(max_share * 100):03d}pct",
            ablation_value=float(max_share),
            note="Limits maximum pyrolysis throughput share to test whether anti-concentration or technology-diversity policies alter the portfolio.",
        )

    exclusion_bundle = _exclude_pathway(bundle, "htc")
    _append_execution_rows(
        rows,
        allocation_frames,
        execution=execute_planning_pipeline(bundle=exclusion_bundle, config=config),
        ablation_family="pathway_exclusion",
        ablation_key="exclude_htc",
        ablation_value="htc_removed_from_candidate_set",
        note="Removes HTC candidate rows to test whether baseline recommendations depend on allowing transfer-limited HTC participation.",
    )

    symmetry_bundle = _apply_hydrochar_price_symmetry(bundle)
    _append_execution_rows(
        rows,
        allocation_frames,
        execution=execute_planning_pipeline(bundle=symmetry_bundle, config=config),
        ablation_family="economic_symmetry",
        ablation_key="hydrochar_price_matches_biochar",
        ablation_value="pyrolysis_median_unit_product_revenue",
        note="Applies the pyrolysis median unit product-revenue intensity to HTC rows as a matched hydrochar-price symmetry stress.",
    )

    for boundary_key, boundary_bundle, note in _build_economic_baseline_bundles(bundle):
        _append_execution_rows(
            rows,
            allocation_frames,
            execution=execute_planning_pipeline(bundle=boundary_bundle, config=config),
            ablation_family="economic_baseline",
            ablation_key=boundary_key,
            ablation_value="formal_baseline_boundary",
            note=note,
        )

    for boundary_key, boundary_bundle, note in _build_coproduct_boundary_bundles(bundle):
        _append_execution_rows(
            rows,
            allocation_frames,
            execution=execute_planning_pipeline(bundle=boundary_bundle, config=config),
            ablation_family="coproduct_boundary",
            ablation_key=boundary_key,
            ablation_value="pyrolysis_median_unit_product_revenue_basis",
            note=note,
        )

    summary = pd.DataFrame(rows).sort_values(["ablation_family", "ablation_key", "scenario_name"]).reset_index(drop=True)
    allocations = pd.concat(allocation_frames, ignore_index=True) if allocation_frames else pd.DataFrame()
    monte_carlo_samples, monte_carlo_summary = run_monte_carlo_uq(
        bundle=bundle,
        base_config=config,
        replicate_count=monte_carlo_replicates,
        random_seed=monte_carlo_random_seed,
    )
    return write_targeted_planning_ablations(
        summary=summary,
        allocations=allocations,
        monte_carlo_samples=monte_carlo_samples,
        monte_carlo_summary=monte_carlo_summary,
        output_dir=output_dir,
        dataset_path=str(bundle.dataset_path),
        base_config=config,
        monte_carlo_replicates=monte_carlo_replicates,
        monte_carlo_random_seed=monte_carlo_random_seed,
    )


def write_targeted_planning_ablations(
    *,
    summary: pd.DataFrame,
    allocations: pd.DataFrame,
    monte_carlo_samples: pd.DataFrame | None = None,
    monte_carlo_summary: pd.DataFrame | None = None,
    output_dir: str | None,
    dataset_path: str,
    base_config: PlanningConfig,
    monte_carlo_replicates: int = 0,
    monte_carlo_random_seed: int = 42,
) -> dict[str, str]:
    target_dir = Path(output_dir) if output_dir else BENCHMARK_OUTPUTS_DIR / "targeted_planning_ablations"
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary_csv": target_dir / "targeted_planning_ablations_summary.csv",
        "allocations_csv": target_dir / "targeted_planning_ablations_allocations.csv",
        "cap_diagnostics_csv": target_dir / "portfolio_cap_diagnostics.csv",
        "monte_carlo_samples_csv": target_dir / "monte_carlo_uq_samples.csv",
        "monte_carlo_summary_csv": target_dir / "monte_carlo_uq_summary.csv",
        "run_config": target_dir / "run_config.json",
    }
    summary.to_csv(outputs["summary_csv"], index=False)
    allocations.to_csv(outputs["allocations_csv"], index=False)
    _build_cap_diagnostics(summary, allocations).to_csv(outputs["cap_diagnostics_csv"], index=False)
    (monte_carlo_samples if monte_carlo_samples is not None else pd.DataFrame()).to_csv(
        outputs["monte_carlo_samples_csv"],
        index=False,
    )
    (monte_carlo_summary if monte_carlo_summary is not None else pd.DataFrame()).to_csv(
        outputs["monte_carlo_summary_csv"],
        index=False,
    )
    write_json(
        outputs["run_config"],
        build_run_manifest(
            dataset_path=dataset_path,
            eta_sweep_values=list(ETA_SWEEP_VALUES),
            evidence_ladder_presets=list(EVIDENCE_LADDER_PRESETS),
            evidence_utility_factor=float(base_config.evidence_utility_factor),
            htc_model_priority=list(base_config.htc_model_priority),
            monte_carlo_replicate_count=int(monte_carlo_replicates),
            monte_carlo_random_seed=int(monte_carlo_random_seed),
            output_files={key: str(path) for key, path in outputs.items()},
            purpose="Targeted Q1-facing ablations for economic baselines, evidence gates, evidence ladders, product-credit boundaries, and Monte Carlo UQ.",
        ),
    )
    return {key: str(path) for key, path in outputs.items()}


def _append_execution_rows(
    rows: list[dict[str, object]],
    allocation_frames: list[pd.DataFrame],
    *,
    execution: dict[str, object],
    ablation_family: str,
    ablation_key: str,
    ablation_value: object,
    note: str,
) -> None:
    portfolio_summary = execution["portfolio_summary"].copy()
    allocations = execution["portfolio_allocations"].copy()
    if not allocations.empty:
        allocations["ablation_family"] = ablation_family
        allocations["ablation_key"] = ablation_key
        allocations["ablation_value"] = ablation_value
        allocation_frames.append(allocations)
    selected_lookup = (
        allocations.groupby("scenario_name", dropna=False)["pathway"]
        .agg(lambda values: "|".join(sorted(pd.Series(values, dtype="object").astype(str).unique())))
        .to_dict()
        if not allocations.empty
        else {}
    )
    for _, row in portfolio_summary.iterrows():
        rows.append(
            {
                "ablation_family": ablation_family,
                "ablation_key": ablation_key,
                "ablation_value": ablation_value,
                "scenario_name": row["scenario_name"],
                "selected_pathways": selected_lookup.get(row["scenario_name"], ""),
                "top_portfolio_case_id": row.get("top_portfolio_case_id", ""),
                "selected_candidate_count": row.get("selected_candidate_count", 0),
                "portfolio_fill_ratio": row.get("portfolio_fill_ratio", 0.0),
                "scenario_feed_coverage_ratio": row.get("scenario_feed_coverage_ratio", 0.0),
                "portfolio_score_mass": row.get("portfolio_score_mass", 0.0),
                "portfolio_energy_objective": row.get("portfolio_energy_objective", 0.0),
                "portfolio_environment_objective": row.get("portfolio_environment_objective", 0.0),
                "portfolio_cost_objective": row.get("portfolio_cost_objective", 0.0),
                "portfolio_carbon_load_kgco2e": row.get("portfolio_carbon_load_kgco2e", 0.0),
                "pyrolysis_allocated_share_pct": _pathway_share_pct(allocations, row["scenario_name"], "pyrolysis"),
                "htc_allocated_share_pct": _pathway_share_pct(allocations, row["scenario_name"], "htc"),
                "ad_allocated_share_pct": _pathway_share_pct(allocations, row["scenario_name"], "ad"),
                "max_candidate_allocated_share_pct": _max_candidate_share_pct(allocations, row["scenario_name"]),
                "note": note,
            }
        )


def _build_cap_diagnostics(summary: pd.DataFrame, allocations: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    families = {"constraint_mechanism", "pathway_cap_sensitivity"}
    diagnostic = summary[summary["ablation_family"].astype(str).isin(families)].copy()
    if diagnostic.empty:
        return diagnostic
    diagnostic["candidate_cap_artifact_flag"] = diagnostic["max_candidate_allocated_share_pct"].apply(
        lambda value: bool(pd.notna(value) and float(value) >= 44.9)
    )
    diagnostic["interpretation"] = diagnostic.apply(_cap_diagnostic_interpretation, axis=1)
    return diagnostic.reset_index(drop=True)


def _cap_diagnostic_interpretation(row: pd.Series) -> str:
    key = str(row.get("ablation_key", ""))
    pyrolysis_share = float(pd.to_numeric(pd.Series([row.get("pyrolysis_allocated_share_pct")]), errors="coerce").fillna(0.0).iloc[0])
    htc_share = float(pd.to_numeric(pd.Series([row.get("htc_allocated_share_pct")]), errors="coerce").fillna(0.0).iloc[0])
    max_candidate = float(pd.to_numeric(pd.Series([row.get("max_candidate_allocated_share_pct")]), errors="coerce").fillna(0.0).iloc[0])
    if "locked_candidate_cap" in key and max_candidate >= 44.9:
        return "At least one selected candidate binds the 45% cap; pathway shares require cap-diagnostic interpretation."
    if "relaxed" in key and pyrolysis_share >= 99.0:
        return "Relaxing candidate/subtype mechanics collapses the portfolio to pyrolysis-only under this boundary."
    if htc_share > 0.0:
        return "HTC remains present under this cap diagnostic, but participation should be read with the stated portfolio rule."
    return "HTC is absent under this cap diagnostic."


def _pathway_share_pct(allocations: pd.DataFrame, scenario_name: object, pathway_name: str) -> float:
    if allocations.empty or "allocated_feed_ton_per_year" not in allocations.columns:
        return 0.0
    subset = allocations[allocations["scenario_name"].astype(str).eq(str(scenario_name))]
    if subset.empty:
        return 0.0
    total = pd.to_numeric(subset["allocated_feed_ton_per_year"], errors="coerce").fillna(0.0).sum()
    if total <= 0.0:
        return 0.0
    pathway_total = pd.to_numeric(
        subset.loc[subset["pathway"].astype(str).str.lower().eq(str(pathway_name).lower()), "allocated_feed_ton_per_year"],
        errors="coerce",
    ).fillna(0.0).sum()
    return float(pathway_total / total * 100.0)


def _max_candidate_share_pct(allocations: pd.DataFrame, scenario_name: object) -> float:
    if allocations.empty or "allocated_feed_ton_per_year" not in allocations.columns:
        return 0.0
    subset = allocations[allocations["scenario_name"].astype(str).eq(str(scenario_name))]
    if subset.empty:
        return 0.0
    values = pd.to_numeric(subset["allocated_feed_ton_per_year"], errors="coerce").fillna(0.0)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    return float(values.max() / total * 100.0)


def _exclude_pathway(bundle: PlanningInputBundle, pathway_name: str) -> PlanningInputBundle:
    frame = bundle.frame.copy()
    mask = frame["pathway"].astype(str).str.lower().ne(str(pathway_name).lower())
    return replace(bundle, frame=frame.loc[mask].reset_index(drop=True))


def _reprioritize_models(priorities: tuple[str, ...], leading_model: str) -> tuple[str, ...]:
    ordered = [leading_model]
    ordered.extend(model_key for model_key in priorities if model_key != leading_model)
    return tuple(dict.fromkeys(ordered))


def _apply_hydrochar_price_symmetry(bundle: PlanningInputBundle) -> PlanningInputBundle:
    frame = bundle.frame.copy()
    pathway = frame["pathway"].astype(str).str.lower()
    pyrolysis_mask = pathway.eq("pyrolysis")
    htc_mask = pathway.eq("htc")
    if not pyrolysis_mask.any() or not htc_mask.any():
        return bundle

    pyro_unit_revenue = (
        pd.to_numeric(frame.loc[pyrolysis_mask, "product_revenue_usd_per_year"], errors="coerce")
        / pd.to_numeric(frame.loc[pyrolysis_mask, "scenario_total_mixed_feed_ton_per_year_proxy"], errors="coerce").replace(0.0, pd.NA)
    )
    default_unit_revenue = float(pyro_unit_revenue.dropna().median()) if not pyro_unit_revenue.dropna().empty else 0.0
    scenario_unit_revenue = (
        frame.loc[pyrolysis_mask, ["scenario_name", "product_revenue_usd_per_year", "scenario_total_mixed_feed_ton_per_year_proxy"]]
        .assign(
            unit_product_revenue=lambda data: pd.to_numeric(data["product_revenue_usd_per_year"], errors="coerce")
            / pd.to_numeric(data["scenario_total_mixed_feed_ton_per_year_proxy"], errors="coerce").replace(0.0, pd.NA)
        )
        .groupby("scenario_name", dropna=False)["unit_product_revenue"]
        .median()
        .to_dict()
    )

    htc_totals = pd.to_numeric(frame.loc[htc_mask, "scenario_total_mixed_feed_ton_per_year_proxy"], errors="coerce").fillna(0.0)
    unit_values = frame.loc[htc_mask, "scenario_name"].astype(str).map(
        lambda scenario_name: float(scenario_unit_revenue.get(scenario_name, default_unit_revenue))
    )
    current_revenue = pd.to_numeric(frame.loc[htc_mask, "product_revenue_usd_per_year"], errors="coerce").fillna(0.0)
    updated_revenue = htc_totals * unit_values
    delta_revenue = updated_revenue - current_revenue

    frame.loc[htc_mask, "product_revenue_usd_per_year"] = updated_revenue
    if "net_system_cost_usd_per_year" in frame.columns:
        frame.loc[htc_mask, "net_system_cost_usd_per_year"] = (
            pd.to_numeric(frame.loc[htc_mask, "net_system_cost_usd_per_year"], errors="coerce").fillna(0.0)
            - delta_revenue
        )
    if "unit_net_system_cost_usd_per_total_mixed_ton" in frame.columns:
        frame.loc[htc_mask, "unit_net_system_cost_usd_per_total_mixed_ton"] = (
            pd.to_numeric(frame.loc[htc_mask, "net_system_cost_usd_per_year"], errors="coerce").fillna(0.0)
            / htc_totals.replace(0.0, pd.NA)
        ).fillna(0.0)
    if "total_system_cost_usd_per_year" in frame.columns:
        frame.loc[htc_mask, "total_system_cost_usd_per_year"] = (
            pd.to_numeric(frame.loc[htc_mask, "total_system_cost_usd_per_year"], errors="coerce").fillna(0.0)
            - delta_revenue
        )
    return replace(bundle, frame=frame)


def _build_economic_baseline_bundles(bundle: PlanningInputBundle) -> list[tuple[str, PlanningInputBundle, str]]:
    unit_revenue = _median_unit_product_revenue(bundle, pathway_name="pyrolysis")
    return [
        (
            "no_product_credit_baseline",
            _set_all_product_revenue(bundle, unit_revenue_by_pathway={"pyrolysis": 0.0, "htc": 0.0, "ad": 0.0}),
            "Removes matched product credits from pyrolysis, HTC, and AD to create a neutral no-product-credit economic baseline.",
        ),
        (
            "symmetric_product_credit_baseline",
            _set_all_product_revenue(
                bundle,
                unit_revenue_by_pathway={
                    "pyrolysis": unit_revenue,
                    "htc": unit_revenue,
                    "ad": unit_revenue,
                },
            ),
            "Applies the same product-credit intensity to pyrolysis, HTC, and AD as a symmetric economic baseline.",
        ),
    ]



def _build_coproduct_boundary_bundles(bundle: PlanningInputBundle) -> list[tuple[str, PlanningInputBundle, str]]:
    """Create formal coproduct-credit boundary scenarios for manuscript sensitivity.

    These are not tuned recommendations. They use the median pyrolysis product-revenue
    intensity as a transparent reference to test whether portfolio conclusions are
    conditional on asymmetric coproduct market crediting.
    """
    unit_revenue = _median_unit_product_revenue(bundle, pathway_name="pyrolysis")
    return [
        (
            "no_pyrolysis_product_credit",
            _set_pathway_product_revenue(bundle, pathway_name="pyrolysis", unit_revenue_usd_per_ton=0.0),
            "Removes the matched biochar revenue term from pyrolysis rows to test dependence on product-credit inclusion.",
        ),
        (
            "hydrochar_credit_50pct",
            _set_pathway_product_revenue(bundle, pathway_name="htc", unit_revenue_usd_per_ton=0.50 * unit_revenue),
            "Adds a conservative hydrochar credit at 50% of the pyrolysis median unit product-revenue intensity.",
        ),
        (
            "hydrochar_credit_75pct",
            _set_pathway_product_revenue(bundle, pathway_name="htc", unit_revenue_usd_per_ton=0.75 * unit_revenue),
            "Adds a moderate hydrochar credit at 75% of the pyrolysis median unit product-revenue intensity.",
        ),
        (
            "hydrochar_credit_100pct",
            _set_pathway_product_revenue(bundle, pathway_name="htc", unit_revenue_usd_per_ton=unit_revenue),
            "Adds a matched hydrochar credit at the pyrolysis median unit product-revenue intensity.",
        ),
        (
            "digestate_rng_credit_50pct",
            _set_pathway_product_revenue(bundle, pathway_name="ad", unit_revenue_usd_per_ton=0.50 * unit_revenue),
            "Adds a conservative AD digestate/renewable-gas credit at 50% of the pyrolysis median unit product-revenue intensity.",
        ),
        (
            "digestate_rng_credit_100pct",
            _set_pathway_product_revenue(bundle, pathway_name="ad", unit_revenue_usd_per_ton=unit_revenue),
            "Adds a matched AD digestate/renewable-gas credit at the pyrolysis median unit product-revenue intensity.",
        ),
        (
            "digestate_rng_credit_200pct",
            _set_pathway_product_revenue(bundle, pathway_name="ad", unit_revenue_usd_per_ton=2.00 * unit_revenue),
            "Adds a high AD digestate/renewable-gas credit at 200% of the pyrolysis median unit product-revenue intensity to test whether AD is recoverable under aggressive nutrient-market assumptions.",
        ),
        (
            "digestate_rng_credit_300pct",
            _set_pathway_product_revenue(bundle, pathway_name="ad", unit_revenue_usd_per_ton=3.00 * unit_revenue),
            "Adds a very high AD digestate/renewable-gas credit at 300% of the pyrolysis median unit product-revenue intensity.",
        ),
    ]


def _median_unit_product_revenue(bundle: PlanningInputBundle, pathway_name: str) -> float:
    frame = bundle.frame.copy()
    pathway = frame["pathway"].astype(str).str.lower()
    mask = pathway.eq(str(pathway_name).lower())
    if not mask.any():
        return 0.0
    unit_revenue = (
        pd.to_numeric(frame.loc[mask, "product_revenue_usd_per_year"], errors="coerce")
        / pd.to_numeric(frame.loc[mask, "scenario_total_mixed_feed_ton_per_year_proxy"], errors="coerce").replace(0.0, pd.NA)
    )
    values = unit_revenue.dropna()
    return float(values.median()) if not values.empty else 0.0


def _set_pathway_product_revenue(
    bundle: PlanningInputBundle,
    *,
    pathway_name: str,
    unit_revenue_usd_per_ton: float,
) -> PlanningInputBundle:
    frame = bundle.frame.copy()
    pathway = frame["pathway"].astype(str).str.lower()
    mask = pathway.eq(str(pathway_name).lower())
    if not mask.any():
        return bundle

    totals = pd.to_numeric(frame.loc[mask, "scenario_total_mixed_feed_ton_per_year_proxy"], errors="coerce").fillna(0.0)
    current_revenue = pd.to_numeric(frame.loc[mask, "product_revenue_usd_per_year"], errors="coerce").fillna(0.0)
    updated_revenue = totals * float(unit_revenue_usd_per_ton)
    delta_revenue = updated_revenue - current_revenue

    frame.loc[mask, "product_revenue_usd_per_year"] = updated_revenue
    if "net_system_cost_usd_per_year" in frame.columns:
        frame.loc[mask, "net_system_cost_usd_per_year"] = (
            pd.to_numeric(frame.loc[mask, "net_system_cost_usd_per_year"], errors="coerce").fillna(0.0)
            - delta_revenue
        )
    if "unit_net_system_cost_usd_per_total_mixed_ton" in frame.columns:
        frame.loc[mask, "unit_net_system_cost_usd_per_total_mixed_ton"] = (
            pd.to_numeric(frame.loc[mask, "net_system_cost_usd_per_year"], errors="coerce").fillna(0.0)
            / totals.replace(0.0, pd.NA)
        ).fillna(0.0)
    if "total_system_cost_usd_per_year" in frame.columns:
        frame.loc[mask, "total_system_cost_usd_per_year"] = (
            pd.to_numeric(frame.loc[mask, "total_system_cost_usd_per_year"], errors="coerce").fillna(0.0)
            - delta_revenue
        )
    return replace(bundle, frame=frame)


def _set_all_product_revenue(
    bundle: PlanningInputBundle,
    *,
    unit_revenue_by_pathway: dict[str, float],
) -> PlanningInputBundle:
    updated = bundle
    for pathway_name, unit_revenue in unit_revenue_by_pathway.items():
        updated = _set_pathway_product_revenue(
            updated,
            pathway_name=pathway_name,
            unit_revenue_usd_per_ton=float(unit_revenue),
        )
    return updated


def run_monte_carlo_uq(
    *,
    bundle: PlanningInputBundle,
    base_config: PlanningConfig,
    replicate_count: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_columns = [
        "monte_carlo_replicate",
        "scenario_name",
        "economic_boundary",
        "selected_pathways",
        "pyrolysis_share_pct",
        "htc_share_pct",
        "ad_share_pct",
        "allocated_feed_ktpy",
        "portfolio_energy_pj_per_year",
        "portfolio_carbon_load_ktco2e_per_year",
        "portfolio_net_cost_musd_per_year",
        "energy_weight",
        "environment_weight",
        "cost_weight",
        "evidence_ladder",
        "hydrochar_credit_multiplier",
        "digestate_credit_multiplier",
        "pyrolysis_credit_multiplier",
        "carbon_budget_factor",
        "deployable_capacity_fraction",
    ]
    if replicate_count <= 0:
        return pd.DataFrame(columns=sample_columns), _empty_monte_carlo_summary()

    rng = np.random.default_rng(random_seed)
    unit_revenue = _median_unit_product_revenue(bundle, pathway_name="pyrolysis")
    sample_rows: list[dict[str, object]] = []
    ladder_names = tuple(EVIDENCE_LADDER_PRESETS)
    boundary_names = ("current", "no_product_credit", "symmetric_credit", "market_uncertain")

    for replicate_idx in range(int(replicate_count)):
        boundary = str(rng.choice(boundary_names))
        pyrolysis_multiplier = 1.0
        hydrochar_multiplier = 0.0
        digestate_multiplier = 0.0
        uq_bundle = bundle
        if boundary == "no_product_credit":
            pyrolysis_multiplier = 0.0
            uq_bundle = _set_all_product_revenue(
                bundle,
                unit_revenue_by_pathway={"pyrolysis": 0.0, "htc": 0.0, "ad": 0.0},
            )
        elif boundary == "symmetric_credit":
            pyrolysis_multiplier = hydrochar_multiplier = digestate_multiplier = 1.0
            uq_bundle = _set_all_product_revenue(
                bundle,
                unit_revenue_by_pathway={"pyrolysis": unit_revenue, "htc": unit_revenue, "ad": unit_revenue},
            )
        elif boundary == "market_uncertain":
            pyrolysis_multiplier = float(rng.uniform(0.50, 1.25))
            hydrochar_multiplier = float(rng.uniform(0.00, 1.25))
            digestate_multiplier = float(rng.uniform(0.00, 2.00))
            uq_bundle = _set_all_product_revenue(
                bundle,
                unit_revenue_by_pathway={
                    "pyrolysis": pyrolysis_multiplier * unit_revenue,
                    "htc": hydrochar_multiplier * unit_revenue,
                    "ad": digestate_multiplier * unit_revenue,
                },
            )

        ladder_name = str(rng.choice(ladder_names))
        ladder_params = EVIDENCE_LADDER_PRESETS.get(ladder_name, {})
        weights = rng.dirichlet([8.0, 7.0, 5.0])
        uq_config = replace(
            base_config,
            **ladder_params,
            objective_weight_system=base_config.copy_with_weights(
                energy_weight=float(weights[0]),
                environment_weight=float(weights[1]),
                cost_weight=float(weights[2]),
            ).objective_weight_system,
            carbon_budget_factor=float(rng.uniform(0.85, 1.15)),
            deployable_capacity_fraction=float(rng.uniform(0.75, 0.95)),
            robustness_factor=float(rng.uniform(0.15, 0.55)),
            evidence_utility_factor=float(rng.uniform(0.0, 0.75)),
            minimum_surrogate_artifact_test_r2=0.0,
        )
        execution = execute_planning_pipeline(bundle=uq_bundle, config=uq_config)
        allocations = execution["portfolio_allocations"].copy()
        portfolio_summary = execution["portfolio_summary"].copy()
        summary_map = portfolio_summary.set_index("scenario_name").to_dict("index")
        for scenario_name, summary_row in summary_map.items():
            scenario_allocations = allocations[
                allocations.get("scenario_name", pd.Series(dtype="object")).astype(str).eq(str(scenario_name))
            ].copy()
            selected_pathways = "|".join(
                sorted(scenario_allocations.get("pathway", pd.Series(dtype="object")).astype(str).unique())
            )
            sample_rows.append(
                {
                    "monte_carlo_replicate": int(replicate_idx + 1),
                    "scenario_name": scenario_name,
                    "economic_boundary": boundary,
                    "selected_pathways": selected_pathways,
                    "pyrolysis_share_pct": _pathway_share_pct(scenario_allocations, scenario_name, "pyrolysis"),
                    "htc_share_pct": _pathway_share_pct(scenario_allocations, scenario_name, "htc"),
                    "ad_share_pct": _pathway_share_pct(scenario_allocations, scenario_name, "ad"),
                    "allocated_feed_ktpy": float(summary_row.get("allocated_feed_ton_per_year", 0.0)) / 1000.0,
                    "portfolio_energy_pj_per_year": float(summary_row.get("portfolio_energy_objective", 0.0)) / 1e9,
                    "portfolio_carbon_load_ktco2e_per_year": float(summary_row.get("portfolio_carbon_load_kgco2e", 0.0)) / 1e6,
                    "portfolio_net_cost_musd_per_year": float(summary_row.get("portfolio_cost_objective", 0.0)) / 1e6,
                    "energy_weight": float(weights[0]),
                    "environment_weight": float(weights[1]),
                    "cost_weight": float(weights[2]),
                    "evidence_ladder": ladder_name,
                    "hydrochar_credit_multiplier": hydrochar_multiplier,
                    "digestate_credit_multiplier": digestate_multiplier,
                    "pyrolysis_credit_multiplier": pyrolysis_multiplier,
                    "carbon_budget_factor": uq_config.carbon_budget_factor,
                    "deployable_capacity_fraction": uq_config.deployable_capacity_fraction,
                }
            )

    samples = pd.DataFrame(sample_rows, columns=sample_columns)
    return samples, _summarize_monte_carlo_uq(samples)


def _summarize_monte_carlo_uq(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return _empty_monte_carlo_summary()
    rows: list[dict[str, object]] = []
    for scenario_name, frame in samples.groupby("scenario_name", dropna=False):
        replicate_count = int(frame["monte_carlo_replicate"].nunique())
        for pathway in ("pyrolysis", "htc", "ad"):
            share = pd.to_numeric(frame[f"{pathway}_share_pct"], errors="coerce").fillna(0.0)
            selected = share.gt(1e-6)
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "pathway": pathway,
                    "monte_carlo_replicates": replicate_count,
                    "selection_probability": float(selected.mean()) if len(selected) else 0.0,
                    "share_pct_median": float(share.median()),
                    "share_pct_p05": float(share.quantile(0.05)),
                    "share_pct_p95": float(share.quantile(0.95)),
                    "energy_pj_per_year_median": float(
                        pd.to_numeric(frame["portfolio_energy_pj_per_year"], errors="coerce").median()
                    ),
                    "carbon_ktco2e_per_year_median": float(
                        pd.to_numeric(frame["portfolio_carbon_load_ktco2e_per_year"], errors="coerce").median()
                    ),
                    "cost_musd_per_year_median": float(
                        pd.to_numeric(frame["portfolio_net_cost_musd_per_year"], errors="coerce").median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _empty_monte_carlo_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "scenario_name",
            "pathway",
            "monte_carlo_replicates",
            "selection_probability",
            "share_pct_median",
            "share_pct_p05",
            "share_pct_p95",
            "energy_pj_per_year_median",
            "carbon_ktco2e_per_year_median",
            "cost_musd_per_year_median",
        ]
    )
