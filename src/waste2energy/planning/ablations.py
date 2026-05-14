# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from ..common import build_run_manifest, write_json
from ..config import BENCHMARK_OUTPUTS_DIR, get_objective_weight_system
from .inputs import PlanningInputBundle, load_planning_input_bundle
from .solve import PlanningConfig, execute_planning_pipeline

ETA_SWEEP_VALUES = (0.0, 0.15, 0.30, 0.50, 0.75, 1.0)


def run_targeted_planning_ablations(
    *,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    base_config: PlanningConfig | None = None,
) -> dict[str, str]:
    bundle = load_planning_input_bundle(dataset_path=dataset_path)
    config = base_config or PlanningConfig(enable_pareto_export=False, pareto_point_count=0)
    rows: list[dict[str, object]] = []
    allocation_frames: list[pd.DataFrame] = []

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
    return write_targeted_planning_ablations(
        summary=summary,
        allocations=allocations,
        output_dir=output_dir,
        dataset_path=str(bundle.dataset_path),
        base_config=config,
    )


def write_targeted_planning_ablations(
    *,
    summary: pd.DataFrame,
    allocations: pd.DataFrame,
    output_dir: str | None,
    dataset_path: str,
    base_config: PlanningConfig,
) -> dict[str, str]:
    target_dir = Path(output_dir) if output_dir else BENCHMARK_OUTPUTS_DIR / "targeted_planning_ablations"
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary_csv": target_dir / "targeted_planning_ablations_summary.csv",
        "allocations_csv": target_dir / "targeted_planning_ablations_allocations.csv",
        "run_config": target_dir / "run_config.json",
    }
    summary.to_csv(outputs["summary_csv"], index=False)
    allocations.to_csv(outputs["allocations_csv"], index=False)
    write_json(
        outputs["run_config"],
        build_run_manifest(
            dataset_path=dataset_path,
            eta_sweep_values=list(ETA_SWEEP_VALUES),
            evidence_utility_factor=float(base_config.evidence_utility_factor),
            htc_model_priority=list(base_config.htc_model_priority),
            output_files={key: str(path) for key, path in outputs.items()},
            purpose="Targeted Q1-facing ablations for HTC algorithm priority, evidence-utility sensitivity, HTC exclusion, and hydrochar-price symmetry.",
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
                "ad_allocated_share_pct": _pathway_share_pct(allocations, row["scenario_name"], "ad"),
                "note": note,
            }
        )


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
