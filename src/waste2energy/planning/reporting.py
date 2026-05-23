# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..common import build_run_manifest, write_json
from ..config import FIGURES_TABLES_DIR, MODEL_READY_DIR, PLANNING_OUTPUTS_DIR, SCENARIO_OUTPUTS_DIR
from .confidence import build_recommendation_confidence_summary


def build_main_results_table(
    *,
    planning_dir: str | Path | None = None,
    scenario_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    planning_root = Path(planning_dir) if planning_dir else PLANNING_OUTPUTS_DIR
    scenario_root = Path(scenario_dir) if scenario_dir else SCENARIO_OUTPUTS_DIR

    pathway_summary = pd.read_csv(planning_root / "pathway_summary.csv")
    scored_cases = pd.read_csv(planning_root / "scored_cases.csv")
    decision_stability = pd.read_csv(scenario_root / "decision_stability.csv")
    uncertainty_summary = _read_optional_csv(scenario_root / "uncertainty_summary.csv")
    optimization_diagnostics = _read_optional_csv(planning_root / "optimization_diagnostics.csv")
    readiness_summary = pd.read_csv(MODEL_READY_DIR / "optimization_pathway_readiness_summary.csv")
    pathway_summary = _ensure_policy_floor_ad_rows(pathway_summary, scored_cases)

    scenario_best = (
        pathway_summary.groupby("scenario_name")["best_case_score"].max().rename("scenario_best_score").reset_index()
    )
    pathway_summary = pathway_summary.merge(scenario_best, on="scenario_name", how="left")
    pathway_summary["score_gap_to_scenario_best_pct"] = (
        (pathway_summary["scenario_best_score"] - pathway_summary["best_case_score"])
        / pathway_summary["scenario_best_score"].replace(0.0, pd.NA)
    )

    best_case_details = scored_cases[
        [
            "optimization_case_id",
            "process_temperature_c",
            "residence_time_min",
            "heating_rate_c_per_min",
            "planning_energy_intensity_mj_per_ton",
            "planning_environment_intensity_kgco2e_per_ton",
            "planning_score",
        ]
    ].drop_duplicates(subset=["optimization_case_id"])
    pathway_summary = pathway_summary.merge(
        best_case_details,
        left_on="best_case_id",
        right_on="optimization_case_id",
        how="left",
    ).drop(columns=["optimization_case_id"])

    case_pathways = scored_cases[
        ["optimization_case_id", "scenario_name", "sample_id", "pathway"]
    ].drop_duplicates(subset=["optimization_case_id"])
    decision_with_pathway = decision_stability.merge(
        case_pathways.rename(columns={"optimization_case_id": "representative_case_id"})[
            ["representative_case_id", "pathway"]
        ],
        on="representative_case_id",
        how="left",
    )
    unresolved_pathway = decision_with_pathway["pathway"].isna()
    if unresolved_pathway.any():
        decision_with_pathway.loc[unresolved_pathway, "pathway"] = decision_with_pathway.loc[
            unresolved_pathway
        ][["scenario_name", "sample_id"]].merge(
            case_pathways[["scenario_name", "sample_id", "pathway"]].drop_duplicates(
                subset=["scenario_name", "sample_id"]
            ).rename(columns={"pathway": "fallback_pathway"}),
            on=["scenario_name", "sample_id"],
            how="left",
        )["fallback_pathway"].to_numpy()
    stress_summary = _aggregate_pathway_stress(decision_with_pathway)
    pathway_summary = pathway_summary.merge(
        stress_summary,
        on=["scenario_name", "pathway"],
        how="left",
    )
    support_summary = _build_support_summary(scored_cases)
    if not support_summary.empty:
        pathway_summary = pathway_summary.merge(
            support_summary,
            on=["scenario_name", "pathway"],
            how="left",
        )
    uncertainty_merge_columns = [
        "scenario_name",
        "uncertainty_mode_case_switch_count",
        "uncertainty_mode_pathway_switch_count",
        "max_interval_changes_case_vs_baseline",
        "combined_only_changes_case_vs_baseline",
        "max_interval_changes_pathway_vs_baseline",
        "combined_only_changes_pathway_vs_baseline",
    ]
    if not uncertainty_summary.empty:
        available_uncertainty_columns = [
            column for column in uncertainty_merge_columns if column in uncertainty_summary.columns
        ]
        if available_uncertainty_columns:
            pathway_summary = pathway_summary.merge(
                uncertainty_summary[available_uncertainty_columns].drop_duplicates(subset=["scenario_name"]),
                on="scenario_name",
                how="left",
            )
    diagnostics_merge_columns = [
        "scenario_name",
        "active_uncertainty_penalty_mode",
        "interval_mean_top_ranked_case_id",
        "max_interval_top_ranked_case_id",
        "combined_only_top_ranked_case_id",
        "interval_mean_top_ranked_pathway",
        "max_interval_top_ranked_pathway",
        "combined_only_top_ranked_pathway",
        "uncertainty_mode_case_map",
        "uncertainty_mode_pathway_map",
        "uncertainty_mode_ranking_summary",
    ]
    if not optimization_diagnostics.empty:
        available_diagnostics_columns = [
            column for column in diagnostics_merge_columns if column in optimization_diagnostics.columns
        ]
        if available_diagnostics_columns:
            pathway_summary = pathway_summary.merge(
                optimization_diagnostics[available_diagnostics_columns].drop_duplicates(subset=["scenario_name"]),
                on="scenario_name",
                how="left",
            )
    pathway_summary = pathway_summary.merge(
        readiness_summary[["pathway", "process_basis", "performance_basis", "claim_boundary"]],
        on="pathway",
        how="left",
    )
    for column, default in {
        "uncertainty_mode_case_map": "",
        "uncertainty_mode_pathway_map": "",
        "uncertainty_mode_ranking_summary": "",
        "best_case_uncertainty_rank_span": pd.NA,
        "best_case_uncertainty_best_mode": pd.NA,
        "best_case_uncertainty_worst_mode": pd.NA,
        "best_case_rank_interval_mean": pd.NA,
        "best_case_rank_max_interval": pd.NA,
        "best_case_rank_combined_only": pd.NA,
    }.items():
        if column not in pathway_summary.columns:
            pathway_summary[column] = default

    pathway_summary["stress_test_tags"] = pathway_summary["stress_test_tags"].fillna("none")
    pathway_summary = _require_numeric_columns(
        pathway_summary,
        columns=(
            "best_case_score",
            "portfolio_selected_count",
            "portfolio_allocated_feed_share",
            "best_case_energy_objective",
            "best_case_environment_objective",
        ),
        context="Planning main results table",
    )
    selected_count = pd.to_numeric(pathway_summary["portfolio_selected_count"], errors="coerce")
    pathway_summary["selected_in_baseline_portfolio"] = pd.Series(
        np.where(selected_count.isna(), pd.NA, selected_count > 0.0),
        index=pathway_summary.index,
        dtype="boolean",
    )
    pathway_summary["baseline_portfolio_share_pct"] = (
        pd.to_numeric(pathway_summary["portfolio_allocated_feed_share"], errors="coerce") * 100.0
    )
    pathway_summary["best_case_energy_pj_per_year"] = (
        pd.to_numeric(pathway_summary["best_case_energy_objective"], errors="coerce") / 1e9
    )
    pathway_summary["best_case_environment_ktco2e_per_year"] = (
        pd.to_numeric(pathway_summary["best_case_environment_objective"], errors="coerce") / 1e6
    )
    pathway_summary["best_case_operating_window"] = pathway_summary.apply(
        _format_operating_window,
        axis=1,
    )
    pathway_summary["best_case_blend_label"] = pathway_summary.apply(_format_blend_label, axis=1)
    pathway_summary["writing_label"] = pathway_summary.apply(_classify_results_row, axis=1)
    pathway_summary["writing_label_display"] = pathway_summary["writing_label"].map(_format_writing_label)
    pathway_summary["process_basis_label"] = pathway_summary.apply(_format_process_basis_label, axis=1)
    pathway_summary["performance_basis_label"] = pathway_summary.apply(_format_performance_basis_label, axis=1)
    pathway_summary["claim_boundary_label"] = pathway_summary.apply(_format_claim_boundary_label, axis=1)
    pathway_summary["uncertainty_mode_sensitivity"] = pathway_summary.apply(
        _format_uncertainty_mode_sensitivity,
        axis=1,
    )
    pathway_summary["best_case_uq_ranking_note"] = pathway_summary.apply(_build_best_case_uq_ranking_note, axis=1)
    pathway_summary["best_case_uq_rank_profile"] = pathway_summary.apply(_build_best_case_uq_rank_profile, axis=1)
    pathway_summary["uq_mode_comparison_sentence"] = pathway_summary.apply(
        _build_uq_mode_comparison_sentence,
        axis=1,
    )
    pathway_summary["results_sentence"] = pathway_summary.apply(_build_results_sentence, axis=1)

    final_table = pathway_summary[
        [
            "scenario_name",
            "pathway",
            "writing_label_display",
            "best_case_manure_subtype",
            "best_case_blend_label",
            "best_case_operating_window",
            "best_case_score",
            "score_gap_to_scenario_best_pct",
            "best_case_energy_pj_per_year",
            "best_case_environment_ktco2e_per_year",
            "selected_in_baseline_portfolio",
            "baseline_portfolio_share_pct",
            "max_stress_selection_rate",
            "stress_test_tags",
            "uncertainty_stress_support",
            "max_uncertainty_stress_selection_rate",
            "selected_under_max_interval_uncertainty",
            "selected_under_combined_only_uncertainty",
            "uncertainty_mode_sensitivity",
            "uncertainty_mode_case_switch_count",
            "uncertainty_mode_pathway_switch_count",
            "uncertainty_mode_case_map",
            "uncertainty_mode_pathway_map",
            "uncertainty_mode_ranking_summary",
            "best_case_uncertainty_rank_span",
            "best_case_uncertainty_best_mode",
            "best_case_uncertainty_worst_mode",
            "best_case_rank_interval_mean",
            "best_case_rank_max_interval",
            "best_case_rank_combined_only",
            "best_case_uq_ranking_note",
            "best_case_uq_rank_profile",
            "uq_mode_comparison_sentence",
            "process_basis_label",
            "performance_basis_label",
            "claim_boundary_label",
            "Surrogate_Support_Level",
            "results_sentence",
        ]
    ].copy()
    final_table = final_table.rename(
        columns={
            "writing_label_display": "writing_label",
            "best_case_manure_subtype": "dominant_manure_subtype",
            "best_case_score": "best_case_score_index",
            "score_gap_to_scenario_best_pct": "score_gap_to_scenario_best_pct",
            "selected_in_baseline_portfolio": "selected_in_baseline_portfolio",
            "max_stress_selection_rate": "max_stress_selection_rate",
            "stress_test_tags": "stress_tests_supporting_pathway",
            "uncertainty_stress_support": "uq_stress_support",
            "max_uncertainty_stress_selection_rate": "max_uq_stress_selection_rate",
            "process_basis_label": "process_basis",
            "performance_basis_label": "performance_basis",
            "claim_boundary_label": "claim_boundary",
            "Surrogate_Support_Level": "surrogate_support_level",
        }
    )
    final_table["score_gap_to_scenario_best_pct"] = (
        pd.to_numeric(final_table["score_gap_to_scenario_best_pct"], errors="coerce") * 100.0
    )
    final_table["max_stress_selection_rate"] = (
        pd.to_numeric(final_table["max_stress_selection_rate"], errors="coerce") * 100.0
    )
    final_table["best_case_score_index"] = final_table["best_case_score_index"].round(3)
    final_table["score_gap_to_scenario_best_pct"] = final_table["score_gap_to_scenario_best_pct"].round(1)
    final_table["best_case_energy_pj_per_year"] = final_table["best_case_energy_pj_per_year"].round(2)
    final_table["best_case_environment_ktco2e_per_year"] = final_table[
        "best_case_environment_ktco2e_per_year"
    ].round(1)
    final_table["baseline_portfolio_share_pct"] = final_table["baseline_portfolio_share_pct"].round(1)
    final_table["max_stress_selection_rate"] = final_table["max_stress_selection_rate"].round(1)
    final_table["max_uq_stress_selection_rate"] = (
        pd.to_numeric(final_table["max_uq_stress_selection_rate"], errors="coerce") * 100.0
    ).round(1)
    for column in [
        "best_case_uncertainty_rank_span",
        "best_case_rank_interval_mean",
        "best_case_rank_max_interval",
        "best_case_rank_combined_only",
    ]:
        if column in final_table.columns:
            final_table[column] = pd.to_numeric(final_table[column], errors="coerce").round(0)
    confidence_summary = build_recommendation_confidence_summary(final_table)
    if not confidence_summary.empty:
        final_table = final_table.merge(
            confidence_summary[
                [
                    "scenario_name",
                    "pathway",
                    "recommendation_confidence_score",
                    "recommendation_confidence_tier",
                    "recommendation_confidence_note",
                ]
            ],
            on=["scenario_name", "pathway"],
            how="left",
        )
    final_table = final_table.sort_values(
        ["scenario_name", "selected_in_baseline_portfolio", "best_case_score_index"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    manifest = build_run_manifest(
        planning_dir=str(planning_root),
        scenario_dir=str(scenario_root),
        source_files=[
            str(planning_root / "pathway_summary.csv"),
            str(planning_root / "scored_cases.csv"),
            str(scenario_root / "decision_stability.csv"),
            str(scenario_root / "uncertainty_summary.csv"),
            str(planning_root / "optimization_diagnostics.csv"),
            str(MODEL_READY_DIR / "optimization_pathway_readiness_summary.csv"),
        ],
        row_count=int(len(final_table)),
        columns=final_table.columns.tolist(),
        purpose="Manuscript-facing Paper 1 planning Results table for pathway competition and writing guidance.",
    )
    return final_table, manifest


def write_main_results_table(
    table: pd.DataFrame,
    manifest: dict[str, object],
    *,
    planning_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
) -> dict[str, str]:
    planning_root = Path(planning_dir) if planning_dir else PLANNING_OUTPUTS_DIR
    figures_root = Path(figures_dir) if figures_dir else FIGURES_TABLES_DIR
    planning_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    visualization_bundle = build_main_results_visualization_bundle(table)
    confidence_summary = build_recommendation_confidence_summary(table)
    visualization_manifest = build_run_manifest(
        based_on_table="main_results_table.csv",
        row_count=int(len(table)),
        metric_long_row_count=int(len(visualization_bundle["metric_long"])),
        annotation_row_count=int(len(visualization_bundle["annotations"])),
        recommendation_confidence_row_count=int(len(confidence_summary)),
        figure_specs=visualization_bundle["figure_specs"],
        purpose="Visualization-ready Paper 1 planning bundle for manuscript-grade plotting.",
    )

    outputs = {
        "planning_results_table": planning_root / "main_results_table.csv",
        "planning_results_table_thermochemical": planning_root / "main_results_table_thermochemical.csv",
        "planning_ad_reference_diagnostics": planning_root / "ad_reference_diagnostics.csv",
        "planning_results_manifest": planning_root / "main_results_table_manifest.json",
        "figures_results_table": figures_root / "paper1_planning_results_table.csv",
        "planning_visual_metrics_long": planning_root / "main_results_visual_metrics_long.csv",
        "planning_visual_annotations": planning_root / "main_results_visual_annotations.csv",
        "planning_visual_manifest": planning_root / "main_results_visual_manifest.json",
        "figures_visual_metrics_long": figures_root / "paper1_planning_visual_metrics_long.csv",
        "figures_visual_annotations": figures_root / "paper1_planning_visual_annotations.csv",
        "figures_visual_manifest": figures_root / "paper1_planning_visual_manifest.json",
        "planning_recommendation_confidence_summary": planning_root / "recommendation_confidence_summary.csv",
        "figures_recommendation_confidence_summary": figures_root / "paper1_recommendation_confidence_summary.csv",
    }
    thermochemical_table = _thermochemical_main_results_table(table)
    ad_reference_diagnostics = _ad_reference_diagnostics_table(table)
    table.to_csv(outputs["planning_results_table"], index=False)
    thermochemical_table.to_csv(outputs["planning_results_table_thermochemical"], index=False)
    ad_reference_diagnostics.to_csv(outputs["planning_ad_reference_diagnostics"], index=False)
    table.to_csv(outputs["figures_results_table"], index=False)
    visualization_bundle["metric_long"].to_csv(outputs["planning_visual_metrics_long"], index=False)
    visualization_bundle["annotations"].to_csv(outputs["planning_visual_annotations"], index=False)
    visualization_bundle["metric_long"].to_csv(outputs["figures_visual_metrics_long"], index=False)
    visualization_bundle["annotations"].to_csv(outputs["figures_visual_annotations"], index=False)
    confidence_summary.to_csv(outputs["planning_recommendation_confidence_summary"], index=False)
    confidence_summary.to_csv(outputs["figures_recommendation_confidence_summary"], index=False)
    write_json(outputs["planning_results_manifest"], manifest)
    write_json(outputs["planning_visual_manifest"], visualization_manifest)
    write_json(outputs["figures_visual_manifest"], visualization_manifest)
    return {key: str(value) for key, value in outputs.items()}


def _thermochemical_main_results_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "pathway" not in table.columns:
        return table.copy()
    return table[table["pathway"].astype(str).str.lower().isin(["pyrolysis", "htc"])].copy().reset_index(drop=True)


def _ad_reference_diagnostics_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "pathway" not in table.columns:
        return pd.DataFrame(
            [
                {
                    "scenario_name": pd.NA,
                    "pathway": "ad",
                    "reference_role": "biological_reference_policy_floor_diagnostic",
                    "baseline_portfolio_share_pct": pd.NA,
                    "selected_in_baseline_portfolio": False,
                    "claim_boundary": "comparison only",
                    "diagnostic_note": "Main results table unavailable.",
                }
            ]
        )
    ad_rows = table[table["pathway"].astype(str).str.lower().eq("ad")].copy()
    if ad_rows.empty:
        scenarios = (
            table["scenario_name"].dropna().astype(str).drop_duplicates().tolist()
            if "scenario_name" in table.columns
            else [pd.NA]
        )
        ad_rows = pd.DataFrame(
            [
                {
                    "scenario_name": scenario_name,
                    "pathway": "ad",
                    "baseline_portfolio_share_pct": 0.0,
                    "selected_in_baseline_portfolio": False,
                    "claim_boundary": "comparison only",
                    "results_sentence": "AD is reported only as a biological-reference/policy-floor diagnostic.",
                }
                for scenario_name in scenarios
            ]
        )
    ad_rows["reference_role"] = "biological_reference_policy_floor_diagnostic"
    ad_rows["diagnostic_note"] = (
        "AD is excluded from the primary thermochemical optimizer and retained only for "
        "biological-reference/policy-floor diagnostics."
    )
    preferred_columns = [
        "scenario_name",
        "pathway",
        "reference_role",
        "selected_in_baseline_portfolio",
        "baseline_portfolio_share_pct",
        "max_stress_selection_rate",
        "best_case_score_index",
        "claim_boundary",
        "surrogate_support_level",
        "results_sentence",
        "diagnostic_note",
    ]
    available = [column for column in preferred_columns if column in ad_rows.columns]
    return ad_rows[available].reset_index(drop=True)


def build_main_results_visualization_bundle(table: pd.DataFrame) -> dict[str, object]:
    enriched = table.copy()
    scenario_order = {
        name: index
        for index, name in enumerate(enriched["scenario_name"].drop_duplicates().astype(str).tolist(), start=1)
    }
    pathway_order = {
        name: index
        for index, name in enumerate(["htc", "pyrolysis", "ad", "baseline"], start=1)
    }
    enriched["scenario_order"] = enriched["scenario_name"].map(scenario_order).fillna(999).astype(int)
    enriched["pathway_order"] = enriched["pathway"].map(pathway_order).fillna(999).astype(int)
    enriched["selected_flag"] = enriched["selected_in_baseline_portfolio"].map({True: "selected", False: "not_selected"})
    enriched["claim_color_group"] = enriched["claim_boundary"].map(_classify_claim_color_group)
    enriched["pathway_rank_within_scenario"] = (
        enriched.groupby("scenario_name")["best_case_score_index"].rank(method="dense", ascending=False).astype(int)
    )

    metric_specs = [
        ("best_case_score_index", "Best-case score index", "score", "index"),
        ("score_gap_to_scenario_best_pct", "Score gap to scenario best", "score_gap", "pct"),
        ("best_case_energy_pj_per_year", "Best-case energy", "energy", "PJ/year"),
        ("best_case_environment_ktco2e_per_year", "Best-case environment", "environment", "ktCO2e/year"),
        ("baseline_portfolio_share_pct", "Baseline portfolio share", "portfolio_share", "pct"),
        ("max_stress_selection_rate", "Max stress selection rate", "stress_support", "pct"),
    ]
    metric_frames: list[pd.DataFrame] = []
    for metric_key, metric_label, metric_family, unit in metric_specs:
        metric_frame = enriched[
            [
                "scenario_name",
                "scenario_order",
                "pathway",
                "pathway_order",
                "pathway_rank_within_scenario",
                "writing_label",
                "selected_in_baseline_portfolio",
                "selected_flag",
                "claim_boundary",
                "claim_color_group",
                metric_key,
            ]
        ].copy()
        metric_frame["metric_key"] = metric_key
        metric_frame["metric_label"] = metric_label
        metric_frame["metric_family"] = metric_family
        metric_frame["unit"] = unit
        metric_frame["value"] = pd.to_numeric(metric_frame[metric_key], errors="coerce")
        metric_frame["value_available"] = metric_frame["value"].notna()
        metric_frame["value_label"] = metric_frame["value"].map(lambda value: _format_metric_label(value, unit))
        metric_frame = metric_frame.drop(columns=[metric_key])
        metric_frames.append(metric_frame)
    metric_long = pd.concat(metric_frames, ignore_index=True)
    metric_long = metric_long.sort_values(
        ["scenario_order", "pathway_order", "metric_family"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    annotations = enriched[
        [
            "scenario_name",
            "scenario_order",
            "pathway",
            "pathway_order",
            "pathway_rank_within_scenario",
            "writing_label",
            "dominant_manure_subtype",
            "best_case_blend_label",
            "best_case_operating_window",
            "selected_in_baseline_portfolio",
            "selected_flag",
            "stress_tests_supporting_pathway",
            "uq_stress_support",
            "max_uq_stress_selection_rate",
            "uncertainty_mode_sensitivity",
            "uncertainty_mode_case_switch_count",
            "uncertainty_mode_pathway_switch_count",
            "uncertainty_mode_case_map",
            "uncertainty_mode_pathway_map",
            "uncertainty_mode_ranking_summary",
            "best_case_uncertainty_rank_span",
            "best_case_uncertainty_best_mode",
            "best_case_uncertainty_worst_mode",
            "best_case_rank_interval_mean",
            "best_case_rank_max_interval",
            "best_case_rank_combined_only",
            "best_case_uq_ranking_note",
            "best_case_uq_rank_profile",
            "uq_mode_comparison_sentence",
            "process_basis",
            "performance_basis",
            "claim_boundary",
            "claim_color_group",
            "results_sentence",
        ]
    ].copy()
    annotations = annotations.sort_values(
        ["scenario_order", "pathway_order"],
        ascending=[True, True],
    ).reset_index(drop=True)

    figure_specs = [
        {
            "figure_id": "planning_results_panel_a",
            "recommended_chart": "faceted bar chart",
            "data_file": "paper1_planning_visual_metrics_long.csv",
            "filter_metric_key": "best_case_score_index",
            "x": "pathway",
            "y": "value",
            "facet": "scenario_name",
            "color": "selected_flag",
            "annotation_file": "paper1_planning_visual_annotations.csv",
            "purpose": "Show pathway competition and the selected planning winner in each scenario.",
        },
        {
            "figure_id": "planning_results_panel_b",
            "recommended_chart": "paired bar chart",
            "data_file": "paper1_planning_visual_metrics_long.csv",
            "filter_metric_keys": ["best_case_energy_pj_per_year", "best_case_environment_ktco2e_per_year"],
            "x": "pathway",
            "y": "value",
            "facet": "scenario_name",
            "color": "metric_label",
            "annotation_file": "paper1_planning_visual_annotations.csv",
            "purpose": "Compare energy and environment outputs across pathways within each scenario.",
        },
        {
            "figure_id": "planning_results_panel_c",
            "recommended_chart": "bubble plot or heatmap",
            "data_file": "paper1_planning_visual_metrics_long.csv",
            "filter_metric_keys": ["baseline_portfolio_share_pct", "max_stress_selection_rate"],
            "x": "pathway",
            "y": "scenario_name",
            "size_or_fill": "value",
            "color": "claim_color_group",
            "annotation_file": "paper1_planning_visual_annotations.csv",
            "purpose": "Summarize portfolio dominance, robustness support, and claim boundary in one manuscript panel.",
        },
    ]
    return {
        "metric_long": metric_long,
        "annotations": annotations,
        "figure_specs": figure_specs,
    }


def _aggregate_pathway_stress(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "scenario_name",
                "pathway",
                "max_stress_selection_rate",
                "max_uncertainty_stress_selection_rate",
                "stable_case_count",
                "consensus_case_count",
                "stress_test_tags",
                "uncertainty_stress_support",
                "selected_under_max_interval_uncertainty",
                "selected_under_combined_only_uncertainty",
            ]
        )

    rows: list[dict[str, object]] = []
    for (scenario_name, pathway), subset in frame.groupby(["scenario_name", "pathway"], dropna=False):
        tags: set[str] = set()
        for value in subset["stress_tests_selected"].fillna("").astype(str):
            tags.update(part.strip() for part in value.split("|") if part.strip())

        rows.append(
            {
                "scenario_name": scenario_name,
                "pathway": pathway,
                "max_stress_selection_rate": float(
                    pd.to_numeric(subset["selection_rate"], errors="coerce").max()
                ),
                "max_uncertainty_stress_selection_rate": float(
                    pd.to_numeric(
                        subset.get(
                            "uncertainty_stress_selection_rate",
                            pd.Series([0.0] * len(subset), index=subset.index),
                        ),
                        errors="coerce",
                    ).fillna(0.0).max()
                ),
                "stable_case_count": int(_coerce_bool_series(subset["stable_under_majority_rule"]).sum()),
                "consensus_case_count": int(_coerce_bool_series(subset["stable_under_consensus_rule"]).sum()),
                "stress_test_tags": "|".join(sorted(tags)) if tags else "none",
                "uncertainty_stress_support": _format_uncertainty_stress_support(subset),
                "selected_under_max_interval_uncertainty": bool(
                    _coerce_bool_series(
                        subset.get(
                            "selected_under_max_interval_uncertainty",
                            pd.Series([False] * len(subset), index=subset.index),
                        ),
                        index=subset.index,
                    ).any()
                ),
                "selected_under_combined_only_uncertainty": bool(
                    _coerce_bool_series(
                        subset.get(
                            "selected_under_combined_only_uncertainty",
                            pd.Series([False] * len(subset), index=subset.index),
                        ),
                        index=subset.index,
                    ).any()
                ),
            }
        )
    return pd.DataFrame(rows)


def _ensure_policy_floor_ad_rows(pathway_summary: pd.DataFrame, scored_cases: pd.DataFrame) -> pd.DataFrame:
    if (
        pathway_summary.empty
        or "scenario_name" not in pathway_summary.columns
        or "pathway" not in pathway_summary.columns
    ):
        return pathway_summary
    scenarios = (
        scored_cases["scenario_name"].dropna().astype(str).drop_duplicates().tolist()
        if "scenario_name" in scored_cases.columns
        else pathway_summary["scenario_name"].dropna().astype(str).drop_duplicates().tolist()
    )
    existing_pairs = set(
        zip(
            pathway_summary["scenario_name"].astype(str),
            pathway_summary["pathway"].astype(str).str.lower(),
        )
    )
    rows: list[dict[str, object]] = []
    for scenario_name in scenarios:
        if (scenario_name, "ad") in existing_pairs:
            continue
        row = {column: pd.NA for column in pathway_summary.columns}
        row.update(
            {
                "scenario_name": scenario_name,
                "pathway": "ad",
                "candidate_count": 0,
                "best_case_id": "policy_floor_ad_reference",
                "best_case_score": 0.0,
                "best_case_energy_objective": 0.0,
                "best_case_environment_objective": 0.0,
                "best_case_cost_objective": 0.0,
                "best_case_uncertainty_ratio": 0.0,
                "best_case_uncertainty_source": "policy_floor_reference",
                "best_case_uncertainty_rank_span": 0.0,
                "best_case_manure_subtype": "regional proxy",
                "best_case_blend_manure_ratio": 0.7,
                "best_case_blend_wet_waste_ratio": 0.3,
                "portfolio_selected_count": 0,
                "portfolio_allocated_feed_ton_per_year": 0.0,
                "portfolio_allocated_feed_share": 0.0,
                "portfolio_energy_objective": 0.0,
                "portfolio_environment_objective": 0.0,
                "portfolio_cost_objective": 0.0,
                "portfolio_carbon_load_kgco2e": 0.0,
                "portfolio_top_case_id": None,
            }
        )
        rows.append(row)
    if not rows:
        return pathway_summary
    extra = pd.DataFrame(rows).dropna(axis=1, how="all")
    return pd.concat([pathway_summary, extra], ignore_index=True)


def _build_support_summary(scored_cases: pd.DataFrame) -> pd.DataFrame:
    if scored_cases.empty or "surrogate_support_level" not in scored_cases.columns:
        return pd.DataFrame(columns=["scenario_name", "pathway", "Surrogate_Support_Level"])
    working = scored_cases.copy()
    working["surrogate_support_level"] = working["surrogate_support_level"].fillna("unknown").astype(str)
    return (
        working.groupby(["scenario_name", "pathway"], dropna=False)
        .agg(
            Surrogate_Support_Level=(
                "surrogate_support_level",
                lambda series: _mode_or_default(series, "unknown"),
            )
        )
        .reset_index()
    )


def _format_operating_window(row: pd.Series) -> str:
    if str(row.get("pathway", "")) in {"ad", "baseline"}:
        return "regional proxy"
    temperature = pd.to_numeric(pd.Series([row.get("process_temperature_c")]), errors="coerce").iloc[0]
    residence = pd.to_numeric(pd.Series([row.get("residence_time_min")]), errors="coerce").iloc[0]
    heating_rate = pd.to_numeric(pd.Series([row.get("heating_rate_c_per_min")]), errors="coerce").iloc[0]
    base = f"{temperature:.0f} C / {residence:.0f} min" if pd.notna(temperature) and pd.notna(residence) else "n/a"
    if pd.notna(heating_rate) and heating_rate > 0:
        return f"{base} / {heating_rate:.0f} C min-1"
    return base


def _format_blend_label(row: pd.Series) -> str:
    manure = _optional_float(row.get("best_case_blend_manure_ratio"))
    wet = _optional_float(row.get("best_case_blend_wet_waste_ratio"))
    if pd.isna(manure) or pd.isna(wet):
        return "blend not available"
    return f"{manure:.1f} manure / {wet:.1f} wet waste"


def _classify_results_row(row: pd.Series) -> str:
    selected_count = _optional_float(row.get("portfolio_selected_count"))
    if pd.isna(selected_count):
        return "not_evaluated"
    selected = bool(selected_count > 0.0)
    portfolio_share = _optional_float(row.get("portfolio_allocated_feed_share"))
    stress_rate = _optional_float(row.get("max_stress_selection_rate"))
    score_gap = _optional_float(row.get("score_gap_to_scenario_best_pct"))
    pathway = str(row.get("pathway", ""))
    tags = _split_stress_test_tags(row.get("stress_test_tags", ""))

    if selected and pd.notna(portfolio_share) and portfolio_share >= 0.99:
        return "dominant_baseline_portfolio"
    if selected:
        return "supporting_baseline_portfolio"
    if pathway == "baseline":
        return "baseline_comparison_anchor"
    if pd.notna(stress_rate) and stress_rate > 0 and "environment_priority" in tags:
        return "environment_sensitive_alternative"
    if pd.notna(stress_rate) and stress_rate > 0:
        return "stress_sensitive_alternative"
    if pd.notna(score_gap) and score_gap <= 0.15:
        return "competitive_unselected_alternative"
    if pd.isna(stress_rate) and pd.isna(score_gap):
        return "not_evaluated"
    return "comparison_only_pathway"


def _build_results_sentence(row: pd.Series) -> str:
    scenario = str(row.get("scenario_name", "scenario"))
    pathway = str(row.get("pathway", "pathway"))
    label = str(row.get("writing_label", ""))
    share = _optional_float(row.get("portfolio_allocated_feed_share"))
    stress_tags = str(row.get("stress_test_tags", "none"))

    if label == "dominant_baseline_portfolio":
        base = f"In {scenario}, {pathway} is the dominant baseline portfolio pathway with full allocated-share coverage."
        return _append_uncertainty_sentence_clause(base, row)
    if label == "supporting_baseline_portfolio":
        if pd.isna(share):
            base = f"In {scenario}, {pathway} enters the baseline portfolio, but the allocated-share value is not available in the current outputs."
            return _append_uncertainty_sentence_clause(base, row)
        base = f"In {scenario}, {pathway} enters the baseline portfolio with {share * 100.0:.1f}% allocated share."
        return _append_uncertainty_sentence_clause(base, row)
    if label == "environment_sensitive_alternative":
        base = f"In {scenario}, {pathway} is not selected in the baseline portfolio but enters under environment-priority stress."
        return _append_uncertainty_sentence_clause(base, row)
    if label == "stress_sensitive_alternative":
        base = f"In {scenario}, {pathway} is a stress-sensitive alternative supported by {stress_tags}."
        return _append_uncertainty_sentence_clause(base, row)
    if label == "competitive_unselected_alternative":
        base = f"In {scenario}, {pathway} remains competitive in score but is not selected in the baseline portfolio."
        return _append_uncertainty_sentence_clause(base, row)
    if label == "baseline_comparison_anchor":
        base = f"In {scenario}, baseline is retained as the comparison anchor rather than as a selected optimization pathway."
        return _append_uncertainty_sentence_clause(base, row)
    if label == "not_evaluated":
        base = f"In {scenario}, {pathway} remains available for comparison, but the current outputs do not fully quantify its manuscript-facing planning status."
        return _append_uncertainty_sentence_clause(base, row)
    base = f"In {scenario}, {pathway} acts as a comparison-only pathway under the current planning evidence."
    return _append_uncertainty_sentence_clause(base, row)


def _append_uncertainty_sentence_clause(base: str, row: pd.Series) -> str:
    note = str(row.get("best_case_uq_ranking_note", "")).strip()
    if not note:
        return base
    return f"{base} {note}"


def _format_writing_label(label: object) -> str:
    mapping = {
        "dominant_baseline_portfolio": "dominant portfolio",
        "supporting_baseline_portfolio": "supporting portfolio",
        "environment_sensitive_alternative": "environment-sensitive alternative",
        "stress_sensitive_alternative": "stress-sensitive alternative",
        "competitive_unselected_alternative": "competitive but unselected",
        "baseline_comparison_anchor": "comparison anchor",
        "comparison_only_pathway": "comparison only",
        "not_evaluated": "not fully evaluated",
    }
    return mapping.get(str(label), str(label))


def _format_process_basis_label(row: pd.Series) -> str:
    pathway = str(row.get("pathway", ""))
    if pathway == "htc":
        return "observed HTC operating window"
    if pathway == "pyrolysis":
        return "observed pyrolysis operating window"
    if pathway == "ad":
        return "regional AD proxy conditions"
    if pathway == "baseline":
        return "regional management-mix proxy"
    return "pathway-specific process basis"


def _format_performance_basis_label(row: pd.Series) -> str:
    pathway = str(row.get("pathway", ""))
    if pathway == "htc":
        return "mixed-feed proxy on observed HTC anchor"
    if pathway == "pyrolysis":
        return "mixed-feed proxy on observed pyrolysis anchor"
    if pathway == "ad":
        return "food-waste AD energy/emission proxy"
    if pathway == "baseline":
        return "regional weighted baseline proxy"
    return "pathway-specific performance basis"


def _format_claim_boundary_label(row: pd.Series) -> str:
    pathway = str(row.get("pathway", ""))
    if pathway == "htc":
        return "planning-ready candidate with cross-study caution"
    if pathway == "pyrolysis":
        return "planning-ready candidate with blended-feed caution"
    if pathway == "ad":
        return "planning comparison only"
    if pathway == "baseline":
        return "comparison anchor only"
    return "claim boundary not classified"


def _classify_claim_color_group(claim_boundary: object) -> str:
    value = str(claim_boundary)
    if "planning-ready candidate" in value:
        return "planning_ready"
    if "comparison anchor" in value:
        return "anchor_only"
    if "comparison only" in value:
        return "comparison_only"
    return "other"


def _format_metric_label(value: float, unit: str) -> str:
    if pd.isna(value):
        return "not available"
    if unit == "index":
        return f"{value:.3f}"
    if unit == "pct":
        return f"{value:.1f}%"
    if unit == "PJ/year":
        return f"{value:.2f}"
    if unit == "ktCO2e/year":
        return f"{value:.1f}"
    return str(value)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _format_uncertainty_stress_support(frame: pd.DataFrame) -> str:
    tags: list[str] = []
    max_interval = frame.get(
        "selected_under_max_interval_uncertainty",
        pd.Series([False] * len(frame), index=frame.index),
    )
    max_interval = _coerce_bool_series(max_interval, index=frame.index)
    combined_only = frame.get(
        "selected_under_combined_only_uncertainty",
        pd.Series([False] * len(frame), index=frame.index),
    )
    combined_only = _coerce_bool_series(combined_only, index=frame.index)
    if max_interval.any():
        tags.append("max_interval")
    if combined_only.any():
        tags.append("combined_only")
    return "|".join(tags) if tags else "none"


def _format_uncertainty_mode_sensitivity(row: pd.Series) -> str:
    case_switch_count = _optional_float(row.get("uncertainty_mode_case_switch_count"))
    pathway_switch_count = _optional_float(row.get("uncertainty_mode_pathway_switch_count"))
    if pd.isna(case_switch_count) or pd.isna(pathway_switch_count):
        return "not evaluated"
    if pathway_switch_count > 1.0:
        return "pathway-sensitive"
    if case_switch_count > 1.0:
        return "case-sensitive, pathway-stable"
    return "stable across tested UQ modes"


def _build_best_case_uq_ranking_note(row: pd.Series) -> str:
    best_case_id = str(row.get("best_case_id", "") or "")
    if not best_case_id:
        return ""
    modes: list[str] = []
    if best_case_id and best_case_id == str(row.get("max_interval_top_ranked_case_id", "") or ""):
        modes.append("max-interval")
    if best_case_id and best_case_id == str(row.get("combined_only_top_ranked_case_id", "") or ""):
        modes.append("combined-only")
    baseline_case_id = str(row.get("interval_mean_top_ranked_case_id", "") or "")
    case_switch_count = _optional_float(row.get("uncertainty_mode_case_switch_count"))
    if best_case_id == baseline_case_id:
        if modes == ["max-interval", "combined-only"] or modes == ["combined-only", "max-interval"]:
            return "The same case remains top-ranked under all tested uncertainty definitions."
        if pd.notna(case_switch_count) and case_switch_count > 1.0:
            sensitivity = str(row.get("uncertainty_mode_sensitivity", "case-sensitive, pathway-stable"))
            return f"The interval-mean top case shifts under at least one alternative uncertainty definition, but the scenario remains {sensitivity}."
        return "The same case remains top-ranked under all tested uncertainty definitions."
    if modes:
        return f"This case becomes top-ranked only under {' and '.join(modes)} uncertainty aggregation."
    return "This case is not the interval-mean top-ranked candidate."


def _build_best_case_uq_rank_profile(row: pd.Series) -> str:
    parts: list[str] = []
    for column, label in [
        ("best_case_rank_interval_mean", "interval-mean"),
        ("best_case_rank_max_interval", "max-interval"),
        ("best_case_rank_combined_only", "combined-only"),
    ]:
        rank_value = _optional_float(row.get(column))
        if pd.notna(rank_value):
            parts.append(f"{label} #{int(float(rank_value))}")
    return "; ".join(parts) if parts else "not available"


def _build_uq_mode_comparison_sentence(row: pd.Series) -> str:
    profile = str(row.get("best_case_uq_rank_profile", "")).strip()
    sensitivity = str(row.get("uncertainty_mode_sensitivity", "")).strip()
    if not profile or profile == "not available":
        pathway = str(row.get("pathway", "pathway"))
        return (
            f"UQ-mode ranking is not available for the best {pathway} row; "
            "interpret this row as a comparison or policy-floor diagnostic rather than a ranked operating recommendation."
        )

    pathway = str(row.get("pathway", "pathway"))
    clauses = [f"Across tested UQ modes, the best {pathway} case ranks as {profile}."]
    rank_span = _optional_float(row.get("best_case_uncertainty_rank_span"))
    if pd.notna(rank_span):
        clauses.append(f"The within-pathway rank span is {int(float(rank_span))}.")
    best_mode = _uq_mode_label(row.get("best_case_uncertainty_best_mode"))
    worst_mode = _uq_mode_label(row.get("best_case_uncertainty_worst_mode"))
    if best_mode != "not available" and worst_mode != "not available":
        if best_mode == worst_mode:
            clauses.append(f"It performs most strongly under {best_mode} scoring.")
        else:
            clauses.append(
                f"It performs best under {best_mode} scoring and worst under {worst_mode} scoring."
            )
    if sensitivity and sensitivity != "not evaluated":
        clauses.append(f"At the scenario level, the recommendation is {sensitivity}.")
    return " ".join(clauses)


def _uq_mode_label(mode: object) -> str:
    mapping = {
        "interval_mean": "interval-mean",
        "max_interval": "max-interval",
        "combined_only": "combined-only",
    }
    value = str(mode or "").strip()
    if not value:
        return "not available"
    return mapping.get(value, value.replace("_", "-"))


def _split_stress_test_tags(raw_value: object) -> set[str]:
    return {
        part.strip()
        for part in str(raw_value or "").split("|")
        if part.strip() and part.strip().lower() != "none"
    }


def _optional_float(value: object) -> float | object:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return pd.NA
    return float(numeric)


def _coerce_bool_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "selected"}


def _coerce_bool_series(values: object, *, index: pd.Index | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(values, index=index)
    return series.map(_coerce_bool_flag)


def _require_numeric_columns(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    context: str,
) -> pd.DataFrame:
    validated = frame.copy()
    for column in columns:
        if column not in validated.columns:
            raise ValueError(f"{context} requires column '{column}', but it is missing.")
        values = pd.to_numeric(validated[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.isna().any():
            row_preview = ", ".join(validated.loc[values.isna(), "pathway"].astype(str).head(5).tolist())
            raise ValueError(
                f"{context} encountered missing/non-finite values in '{column}' for pathway row(s): {row_preview}."
            )
        validated[column] = values
    return validated


def _mode_or_default(series: pd.Series, default: str) -> str:
    values = series.dropna().astype(str)
    if values.empty:
        return default
    modes = values.mode()
    if modes.empty:
        return default
    return str(modes.iloc[0])
