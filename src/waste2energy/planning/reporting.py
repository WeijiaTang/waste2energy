from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common import build_run_manifest, write_json
from ..config import FIGURES_TABLES_DIR, MODEL_READY_DIR, PLANNING_OUTPUTS_DIR, SCENARIO_OUTPUTS_DIR


def build_main_results_table(
    *,
    planning_dir: str | Path | None = None,
    scenario_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    planning_root = Path(planning_dir) if planning_dir else PLANNING_OUTPUTS_DIR / "baseline"
    scenario_root = Path(scenario_dir) if scenario_dir else SCENARIO_OUTPUTS_DIR / "baseline"

    pathway_summary = pd.read_csv(planning_root / "pathway_summary.csv")
    scored_cases = pd.read_csv(planning_root / "scored_cases.csv")
    decision_stability = pd.read_csv(scenario_root / "decision_stability.csv")
    readiness_summary = pd.read_csv(MODEL_READY_DIR / "optimization_pathway_readiness_summary.csv")

    scenario_best = (
        pathway_summary.groupby("scenario_name")["best_case_score"].max().rename("scenario_best_score").reset_index()
    )
    pathway_summary = pathway_summary.merge(scenario_best, on="scenario_name", how="left")
    pathway_summary["score_gap_to_scenario_best_pct"] = (
        (pathway_summary["scenario_best_score"] - pathway_summary["best_case_score"])
        / pathway_summary["scenario_best_score"].replace(0.0, pd.NA)
    ).fillna(0.0)

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
        ].merge(
            case_pathways[["scenario_name", "sample_id", "pathway"]].drop_duplicates(
                subset=["scenario_name", "sample_id"]
            ),
            on=["scenario_name", "sample_id"],
            how="left",
        )["pathway"].to_numpy()
    stress_summary = _aggregate_pathway_stress(decision_with_pathway)
    pathway_summary = pathway_summary.merge(
        stress_summary,
        on=["scenario_name", "pathway"],
        how="left",
    )
    pathway_summary = pathway_summary.merge(
        readiness_summary[["pathway", "process_basis", "performance_basis", "claim_boundary"]],
        on="pathway",
        how="left",
    )

    pathway_summary["stress_test_tags"] = pathway_summary["stress_test_tags"].fillna("none")
    pathway_summary["selected_in_baseline_portfolio"] = (
        pd.to_numeric(pathway_summary["portfolio_selected_count"], errors="coerce").fillna(0.0) > 0
    )
    pathway_summary["baseline_portfolio_share_pct"] = pathway_summary[
        "portfolio_allocated_feed_share"
    ].fillna(0.0) * 100.0
    pathway_summary["best_case_energy_pj_per_year"] = pathway_summary[
        "best_case_energy_objective"
    ].fillna(0.0) / 1e9
    pathway_summary["best_case_environment_ktco2e_per_year"] = pathway_summary[
        "best_case_environment_objective"
    ].fillna(0.0) / 1e6
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
            "process_basis_label",
            "performance_basis_label",
            "claim_boundary_label",
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
            "process_basis_label": "process_basis",
            "performance_basis_label": "performance_basis",
            "claim_boundary_label": "claim_boundary",
        }
    )
    final_table["score_gap_to_scenario_best_pct"] = (
        final_table["score_gap_to_scenario_best_pct"].fillna(0.0) * 100.0
    )
    final_table["max_stress_selection_rate"] = final_table["max_stress_selection_rate"].fillna(0.0) * 100.0
    final_table["best_case_score_index"] = final_table["best_case_score_index"].round(3)
    final_table["score_gap_to_scenario_best_pct"] = final_table["score_gap_to_scenario_best_pct"].round(1)
    final_table["best_case_energy_pj_per_year"] = final_table["best_case_energy_pj_per_year"].round(2)
    final_table["best_case_environment_ktco2e_per_year"] = final_table[
        "best_case_environment_ktco2e_per_year"
    ].round(1)
    final_table["baseline_portfolio_share_pct"] = final_table["baseline_portfolio_share_pct"].round(1)
    final_table["max_stress_selection_rate"] = final_table["max_stress_selection_rate"].round(1)
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
    planning_root = Path(planning_dir) if planning_dir else PLANNING_OUTPUTS_DIR / "baseline"
    figures_root = Path(figures_dir) if figures_dir else FIGURES_TABLES_DIR
    planning_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    visualization_bundle = build_main_results_visualization_bundle(table)
    visualization_manifest = build_run_manifest(
        based_on_table="main_results_table.csv",
        row_count=int(len(table)),
        metric_long_row_count=int(len(visualization_bundle["metric_long"])),
        annotation_row_count=int(len(visualization_bundle["annotations"])),
        figure_specs=visualization_bundle["figure_specs"],
        purpose="Visualization-ready Paper 1 planning bundle for manuscript-grade plotting.",
    )

    outputs = {
        "planning_results_table": planning_root / "main_results_table.csv",
        "planning_results_manifest": planning_root / "main_results_table_manifest.json",
        "figures_results_table": figures_root / "paper1_planning_results_table.csv",
        "planning_visual_metrics_long": planning_root / "main_results_visual_metrics_long.csv",
        "planning_visual_annotations": planning_root / "main_results_visual_annotations.csv",
        "planning_visual_manifest": planning_root / "main_results_visual_manifest.json",
        "figures_visual_metrics_long": figures_root / "paper1_planning_visual_metrics_long.csv",
        "figures_visual_annotations": figures_root / "paper1_planning_visual_annotations.csv",
        "figures_visual_manifest": figures_root / "paper1_planning_visual_manifest.json",
    }
    table.to_csv(outputs["planning_results_table"], index=False)
    table.to_csv(outputs["figures_results_table"], index=False)
    visualization_bundle["metric_long"].to_csv(outputs["planning_visual_metrics_long"], index=False)
    visualization_bundle["annotations"].to_csv(outputs["planning_visual_annotations"], index=False)
    visualization_bundle["metric_long"].to_csv(outputs["figures_visual_metrics_long"], index=False)
    visualization_bundle["annotations"].to_csv(outputs["figures_visual_annotations"], index=False)
    write_json(outputs["planning_results_manifest"], manifest)
    write_json(outputs["planning_visual_manifest"], visualization_manifest)
    write_json(outputs["figures_visual_manifest"], visualization_manifest)
    return {key: str(value) for key, value in outputs.items()}


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
        metric_frame["value"] = pd.to_numeric(metric_frame[metric_key], errors="coerce").fillna(0.0)
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
                "stable_case_count",
                "consensus_case_count",
                "stress_test_tags",
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
                    pd.to_numeric(subset["selection_rate"], errors="coerce").fillna(0.0).max()
                ),
                "stable_case_count": int(subset["stable_under_majority_rule"].fillna(False).astype(bool).sum()),
                "consensus_case_count": int(subset["stable_under_consensus_rule"].fillna(False).astype(bool).sum()),
                "stress_test_tags": "|".join(sorted(tags)) if tags else "none",
            }
        )
    return pd.DataFrame(rows)


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
    manure = pd.to_numeric(pd.Series([row.get("best_case_blend_manure_ratio")]), errors="coerce").fillna(0.0).iloc[0]
    wet = pd.to_numeric(pd.Series([row.get("best_case_blend_wet_waste_ratio")]), errors="coerce").fillna(0.0).iloc[0]
    return f"{manure:.1f} manure / {wet:.1f} wet waste"


def _classify_results_row(row: pd.Series) -> str:
    selected = bool(row.get("portfolio_selected_count", 0) > 0)
    portfolio_share = float(pd.to_numeric(pd.Series([row.get("portfolio_allocated_feed_share")]), errors="coerce").fillna(0.0).iloc[0])
    stress_rate = float(pd.to_numeric(pd.Series([row.get("max_stress_selection_rate")]), errors="coerce").fillna(0.0).iloc[0])
    score_gap = float(pd.to_numeric(pd.Series([row.get("score_gap_to_scenario_best_pct")]), errors="coerce").fillna(0.0).iloc[0])
    pathway = str(row.get("pathway", ""))
    tags = str(row.get("stress_test_tags", ""))

    if selected and portfolio_share >= 0.99:
        return "dominant_baseline_portfolio"
    if selected:
        return "supporting_baseline_portfolio"
    if pathway == "baseline":
        return "baseline_comparison_anchor"
    if stress_rate > 0 and tags == "environment_priority":
        return "environment_sensitive_alternative"
    if stress_rate > 0:
        return "stress_sensitive_alternative"
    if score_gap <= 0.15:
        return "competitive_unselected_alternative"
    return "comparison_only_pathway"


def _build_results_sentence(row: pd.Series) -> str:
    scenario = str(row.get("scenario_name", "scenario"))
    pathway = str(row.get("pathway", "pathway"))
    label = str(row.get("writing_label", ""))
    share = float(pd.to_numeric(pd.Series([row.get("portfolio_allocated_feed_share")]), errors="coerce").fillna(0.0).iloc[0]) * 100.0
    stress_tags = str(row.get("stress_test_tags", "none"))

    if label == "dominant_baseline_portfolio":
        return f"In {scenario}, {pathway} is the dominant baseline portfolio pathway with full allocated-share coverage."
    if label == "supporting_baseline_portfolio":
        return f"In {scenario}, {pathway} enters the baseline portfolio with {share:.1f}% allocated share."
    if label == "environment_sensitive_alternative":
        return f"In {scenario}, {pathway} is not selected in the baseline portfolio but enters under environment-priority stress."
    if label == "stress_sensitive_alternative":
        return f"In {scenario}, {pathway} is a stress-sensitive alternative supported by {stress_tags}."
    if label == "competitive_unselected_alternative":
        return f"In {scenario}, {pathway} remains competitive in score but is not selected in the baseline portfolio."
    if label == "baseline_comparison_anchor":
        return f"In {scenario}, baseline is retained as the comparison anchor rather than as a selected optimization pathway."
    return f"In {scenario}, {pathway} acts as a comparison-only pathway under the current planning evidence."


def _format_writing_label(label: object) -> str:
    mapping = {
        "dominant_baseline_portfolio": "dominant portfolio",
        "supporting_baseline_portfolio": "supporting portfolio",
        "environment_sensitive_alternative": "environment-sensitive alternative",
        "stress_sensitive_alternative": "stress-sensitive alternative",
        "competitive_unselected_alternative": "competitive but unselected",
        "baseline_comparison_anchor": "comparison anchor",
        "comparison_only_pathway": "comparison only",
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
    if unit == "index":
        return f"{value:.3f}"
    if unit == "pct":
        return f"{value:.1f}%"
    if unit == "PJ/year":
        return f"{value:.2f}"
    if unit == "ktCO2e/year":
        return f"{value:.1f}"
    return str(value)
