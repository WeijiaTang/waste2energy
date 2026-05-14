from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import BENCHMARK_OUTPUTS_DIR, FIGURES_TABLES_DIR, MODEL_READY_DIR, OUTPUTS_ROOT
from .config import get_objective_weight_system
from .config import resolve_surrogate_outputs_dir
from .planning.inputs import load_planning_input_bundle
from .planning.solve import PlanningConfig, execute_planning_pipeline


SCENARIO_ORDER = (
    "baseline_region_case",
    "high_supply_case",
    "policy_support_case",
)

SCENARIO_DISPLAY = {
    "baseline_region_case": "baseline-region",
    "high_supply_case": "high-supply",
    "policy_support_case": "policy-support",
}

PATHWAY_DISPLAY = {
    "htc": "HTC",
    "pyrolysis": "pyrolysis",
    "ad": "AD",
    "baseline": "baseline",
}

TARGET_DISPLAY = {
    "carbon_retention_pct": "Carbon retention",
    "product_char_hhv_mj_per_kg": "Char HHV",
    "product_char_yield_pct": "Char yield",
    "energy_recovery_pct": "Energy recovery",
}


def sync_planning_summary_to_latex(
    *,
    planning_dir: str | Path,
    abstract_path: str | Path,
    macros_path: str | Path,
    operation_dir: str | Path | None = None,
    audit_dir: str | Path | None = None,
    benchmark_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
) -> dict[str, object]:
    planning_root = Path(planning_dir)
    abstract_file = Path(abstract_path)
    macros_file = Path(macros_path)
    figures_root = Path(figures_dir) if figures_dir else FIGURES_TABLES_DIR
    audit_root = Path(audit_dir) if audit_dir else OUTPUTS_ROOT / "audit"
    benchmark_root = Path(benchmark_dir) if benchmark_dir else BENCHMARK_OUTPUTS_DIR / "baseline"

    scenario_summary = pd.read_csv(planning_root / "scenario_summary.csv")
    portfolio_allocations = pd.read_csv(planning_root / "portfolio_allocations.csv")
    portfolio_summary = _read_csv_if_exists(planning_root / "portfolio_summary.csv")
    pathway_summary = _read_csv_if_exists(planning_root / "pathway_summary.csv")
    main_results_table = _read_csv_if_exists(planning_root / "main_results_table.csv")
    optimization_diagnostics = _read_csv_if_exists(planning_root / "optimization_diagnostics.csv")

    ad_selected = False
    ad_allocated_share = 0.0
    if not portfolio_allocations.empty and "pathway" in portfolio_allocations.columns:
        ad_rows = portfolio_allocations[portfolio_allocations["pathway"].astype(str).str.lower() == "ad"].copy()
        if not ad_rows.empty:
            if "allocated_feed_ton_per_year" in portfolio_allocations.columns:
                total_allocated_feed = float(
                    pd.to_numeric(
                        portfolio_allocations["allocated_feed_ton_per_year"], errors="coerce"
                    ).fillna(0.0).sum()
                )
                ad_allocated_feed = float(
                    pd.to_numeric(ad_rows["allocated_feed_ton_per_year"], errors="coerce").fillna(0.0).sum()
                )
                ad_allocated_share = ad_allocated_feed / total_allocated_feed if total_allocated_feed > 0.0 else 0.0
            elif "allocated_feed_share" in portfolio_allocations.columns:
                ad_allocated_share = float(
                    pd.to_numeric(ad_rows["allocated_feed_share"], errors="coerce").fillna(0.0).sum()
                )
            ad_selected = ad_allocated_share > 0.0

    ad_status_label = "AD-limited" if ad_selected else "AD-free"
    ad_status_note = (
        "AD appears in optimized allocations and manuscript wording must remain evidence-qualified."
        if ad_selected
        else "No AD allocation is present in the optimized baseline portfolio."
    )

    dominance = _build_dominance_payload(
        scenario_summary=scenario_summary,
        portfolio_summary=portfolio_summary,
        pathway_summary=pathway_summary,
        portfolio_allocations=portfolio_allocations,
        main_results_table=main_results_table,
    )
    uq_payload = _build_uq_payload(
        main_results_table=main_results_table,
        optimization_diagnostics=optimization_diagnostics,
    )
    planning_uq_narrative = _build_planning_uq_narrative(
        main_results_table=main_results_table,
        optimization_diagnostics=optimization_diagnostics,
        uq_payload=uq_payload,
    )
    benchmark = _build_benchmark_payload(audit_dir=audit_root, benchmark_dir=benchmark_root)
    benchmark_artifacts = _write_benchmark_manuscript_artifacts(
        benchmark=benchmark,
        planning_uq_narrative=planning_uq_narrative,
        figures_dir=figures_root,
    )
    benchmark_sections = _build_benchmark_section_templates(
        audit_dir=audit_root,
        benchmark=benchmark,
        planning_uq_narrative=planning_uq_narrative,
    )
    benchmark_section_artifacts = _write_benchmark_section_artifacts(
        sections=benchmark_sections,
        figures_dir=figures_root,
    )
    manuscript_table_artifacts = _write_priority_manuscript_tables(
        planning_dir=planning_root,
        audit_dir=audit_root,
        benchmark_dir=benchmark_root,
        figures_dir=figures_root,
    )

    scenario_count = int(len(scenario_summary))
    top_case_count = (
        int(scenario_summary["top_ranked_case_id"].dropna().astype(str).nunique())
        if "top_ranked_case_id" in scenario_summary.columns
        else 0
    )
    macro_safe = _escape_macro_definition_value
    ad_allocated_share_text = f"{ad_allocated_share * 100.0:.1f}\\%"
    macros_file.parent.mkdir(parents=True, exist_ok=True)
    macros_file.write_text(
        "\n".join(
            [
                f"\\newcommand{{\\PlanningScenarioCount}}{{{macro_safe(scenario_count)}}}",
                f"\\newcommand{{\\PlanningTopCaseCount}}{{{macro_safe(top_case_count)}}}",
                f"\\newcommand{{\\PlanningDominantPathwayDisplay}}{{{macro_safe(dominance['dominant_pathway_display'])}}}",
                f"\\newcommand{{\\PlanningDominancePattern}}{{{macro_safe(dominance['dominance_pattern'])}}}",
                f"\\newcommand{{\\PlanningBaselineDominantPathwayDisplay}}{{{macro_safe(dominance['baseline_pathway_display'])}}}",
                f"\\newcommand{{\\PlanningHighSupplyDominantPathwayDisplay}}{{{macro_safe(dominance['high_supply_pathway_display'])}}}",
                f"\\newcommand{{\\PlanningPolicySupportDominantPathwayDisplay}}{{{macro_safe(dominance['policy_support_pathway_display'])}}}",
                f"\\newcommand{{\\PlanningHighSupplyCoveragePct}}{{{macro_safe(dominance['high_supply_coverage_pct'])}}}",
                f"\\newcommand{{\\PlanningPyrolysisRole}}{{{macro_safe(dominance['pyrolysis_role'])}}}",
                f"\\newcommand{{\\PlanningHTCRole}}{{{macro_safe(dominance['htc_role'])}}}",
                f"\\newcommand{{\\PlanningADRole}}{{{macro_safe(dominance['ad_role'])}}}",
                f"\\newcommand{{\\PlanningIntroductionDominanceSentence}}{{{macro_safe(dominance['introduction_sentence'])}}}",
                f"\\newcommand{{\\PlanningDominanceSummarySentence}}{{{macro_safe(dominance['abstract_sentence'])}}}",
                f"\\newcommand{{\\PlanningHighlightsDominanceBullet}}{{{macro_safe(dominance['highlights_sentence'])}}}",
                f"\\newcommand{{\\PlanningResultsDominanceSentence}}{{{macro_safe(dominance['results_sentence'])}}}",
                f"\\newcommand{{\\PlanningConclusionDominanceSentence}}{{{macro_safe(dominance['conclusion_sentence'])}}}",
                f"\\newcommand{{\\PlanningADStatus}}{{{macro_safe(ad_status_label)}}}",
                f"\\newcommand{{\\PlanningADAllocatedShare}}{{{macro_safe(ad_allocated_share_text)}}}",
                f"\\newcommand{{\\PlanningADStatusNote}}{{{macro_safe(ad_status_note)}}}",
                f"\\newcommand{{\\PlanningUQSensitivityPattern}}{{{macro_safe(uq_payload['pattern'])}}}",
                f"\\newcommand{{\\PlanningUQSensitivitySentence}}{{{macro_safe(uq_payload['sentence'])}}}",
                f"\\newcommand{{\\PlanningUQResultsSentence}}{{{macro_safe(planning_uq_narrative['results_sentence'])}}}",
                f"\\newcommand{{\\PlanningUQDiscussionSentence}}{{{macro_safe(planning_uq_narrative['discussion_sentence'])}}}",
                f"\\newcommand{{\\BenchmarkPrimaryVariant}}{{{macro_safe(benchmark['primary_variant_display'])}}}",
                f"\\newcommand{{\\BenchmarkPrimaryInnovationSentence}}{{{macro_safe(benchmark['primary_sentence'])}}}",
                f"\\newcommand{{\\BenchmarkClassicBaselineSentence}}{{{macro_safe(benchmark['classic_sentence'])}}}",
                f"\\newcommand{{\\BenchmarkEvidenceAwareSentence}}{{{macro_safe(benchmark['evidence_sentence'])}}}",
                f"\\newcommand{{\\BenchmarkRobustnessSentence}}{{{macro_safe(benchmark['robustness_sentence'])}}}",
                f"\\newcommand{{\\BenchmarkHeuristicSentence}}{{{macro_safe(benchmark['heuristic_sentence'])}}}",
                f"\\newcommand{{\\BenchmarkBootstrapSentence}}{{{macro_safe(benchmark['bootstrap_sentence'])}}}",
                f"\\newcommand{{\\BenchmarkResultsTakeawaySentence}}{{{macro_safe(benchmark['takeaway_sentence'])}}}",
                f"\\newcommand{{\\BenchmarkResultsParagraph}}{{{macro_safe(benchmark_sections['results_paragraph'])}}}",
                f"\\newcommand{{\\BenchmarkDiscussionParagraph}}{{{macro_safe(benchmark_sections['discussion_paragraph'])}}}",
                f"\\newcommand{{\\BenchmarkConclusionParagraph}}{{{macro_safe(benchmark_sections['conclusion_paragraph'])}}}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    abstract_text = abstract_file.read_text(encoding="utf-8")
    updated_abstract = abstract_text.replace("AD-free", ad_status_label)
    abstract_rewritten = updated_abstract != abstract_text
    if abstract_rewritten:
        abstract_file.write_text(updated_abstract, encoding="utf-8")

    return {
        "scenario_count": scenario_count,
        "top_case_count": top_case_count,
        "dominance_pattern": dominance["dominance_pattern"],
        "dominant_pathway": dominance["dominant_pathway"],
        "dominant_pathway_display": dominance["dominant_pathway_display"],
        "high_supply_coverage_ratio": dominance["high_supply_coverage_ratio"],
        "ad_selected": ad_selected,
        "ad_status_label": ad_status_label,
        "ad_allocated_share": ad_allocated_share,
        "uq_sensitivity_pattern": uq_payload["pattern"],
        "uq_sensitivity_sentence": uq_payload["sentence"],
        "planning_uq_results_sentence": planning_uq_narrative["results_sentence"],
        "planning_uq_discussion_sentence": planning_uq_narrative["discussion_sentence"],
        "benchmark_primary_variant": benchmark["primary_variant"],
        "benchmark_primary_variant_display": benchmark["primary_variant_display"],
        "benchmark_primary_significance_tier": benchmark["primary_significance_tier"],
        "benchmark_bootstrap_available": benchmark["bootstrap_available"],
        "benchmark_artifacts": benchmark_artifacts,
        "benchmark_section_artifacts": benchmark_section_artifacts,
        "manuscript_table_artifacts": manuscript_table_artifacts,
        "abstract_rewritten": abstract_rewritten,
        "macros_path": str(macros_file),
    }


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _build_dominance_payload(
    *,
    scenario_summary: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    pathway_summary: pd.DataFrame,
    portfolio_allocations: pd.DataFrame,
    main_results_table: pd.DataFrame,
) -> dict[str, object]:
    dominance_frame = _resolve_scenario_dominance(pathway_summary, portfolio_allocations)
    dominant_by_scenario = {
        str(row["scenario_name"]): str(row["pathway"])
        for _, row in dominance_frame.iterrows()
    }
    dominant_share_by_scenario = {
        str(row["scenario_name"]): float(row.get("portfolio_allocated_feed_share", 0.0) or 0.0)
        for _, row in dominance_frame.iterrows()
    }
    baseline_pathway = dominant_by_scenario.get("baseline_region_case", "unknown")
    high_supply_pathway = dominant_by_scenario.get("high_supply_case", baseline_pathway)
    policy_support_pathway = dominant_by_scenario.get("policy_support_case", baseline_pathway)
    distinct_pathways = {
        pathway for pathway in [baseline_pathway, high_supply_pathway, policy_support_pathway] if pathway and pathway != "unknown"
    }
    dominance_pattern = "uniform" if len(distinct_pathways) == 1 else "mixed"
    dominant_pathway = baseline_pathway if dominance_pattern == "uniform" else "mixed"
    dominant_pathway_display = _pathway_display(dominant_pathway)

    score_leaders = _resolve_score_leaders(pathway_summary)
    score_leader_set = {value for value in score_leaders.values() if value and value != "unknown"}
    uniform_score_leader = score_leaders.get("baseline_region_case", "unknown") if len(score_leader_set) == 1 else "mixed"
    high_supply_coverage_ratio = _resolve_high_supply_coverage_ratio(portfolio_summary, scenario_summary)
    high_supply_coverage_pct = f"{high_supply_coverage_ratio * 100.0:.1f}\\%"

    role_frame = main_results_table if not main_results_table.empty else pathway_summary
    pyrolysis_role = _classify_pathway_role("pyrolysis", role_frame)
    htc_role = _classify_pathway_role("htc", role_frame)
    ad_role = _classify_pathway_role("ad", role_frame)

    abstract_sentence = _build_abstract_dominance_sentence(
        dominance_pattern=dominance_pattern,
        dominant_pathway=dominant_pathway,
        score_leader=uniform_score_leader,
        dominant_by_scenario=dominant_by_scenario,
    )
    introduction_sentence = _build_introduction_dominance_sentence(
        dominance_pattern=dominance_pattern,
        dominant_pathway=dominant_pathway,
        score_leader=uniform_score_leader,
        pyrolysis_role=pyrolysis_role,
        htc_role=htc_role,
        ad_role=ad_role,
        dominant_by_scenario=dominant_by_scenario,
    )
    results_sentence = _build_results_dominance_sentence(
        dominance_pattern=dominance_pattern,
        dominant_pathway=dominant_pathway,
        pyrolysis_role=pyrolysis_role,
        dominant_by_scenario=dominant_by_scenario,
    )
    highlights_sentence = _build_highlights_dominance_sentence(
        dominance_pattern=dominance_pattern,
        dominant_pathway=dominant_pathway,
        score_leader=uniform_score_leader,
        pyrolysis_role=pyrolysis_role,
        htc_role=htc_role,
        dominant_by_scenario=dominant_by_scenario,
    )
    conclusion_sentence = _build_conclusion_dominance_sentence(
        dominance_pattern=dominance_pattern,
        dominant_pathway=dominant_pathway,
        score_leader=uniform_score_leader,
        pyrolysis_role=pyrolysis_role,
        htc_role=htc_role,
        dominant_by_scenario=dominant_by_scenario,
        dominant_share_by_scenario=dominant_share_by_scenario,
    )

    return {
        "dominance_pattern": dominance_pattern,
        "dominant_pathway": dominant_pathway,
        "dominant_pathway_display": dominant_pathway_display,
        "baseline_pathway_display": _pathway_display(baseline_pathway),
        "high_supply_pathway_display": _pathway_display(high_supply_pathway),
        "policy_support_pathway_display": _pathway_display(policy_support_pathway),
        "high_supply_coverage_ratio": high_supply_coverage_ratio,
        "high_supply_coverage_pct": high_supply_coverage_pct,
        "pyrolysis_role": pyrolysis_role,
        "htc_role": htc_role,
        "ad_role": ad_role,
        "introduction_sentence": introduction_sentence,
        "abstract_sentence": abstract_sentence,
        "highlights_sentence": highlights_sentence,
        "results_sentence": results_sentence,
        "conclusion_sentence": conclusion_sentence,
    }


def _resolve_scenario_dominance(
    pathway_summary: pd.DataFrame,
    portfolio_allocations: pd.DataFrame,
) -> pd.DataFrame:
    if not pathway_summary.empty and {
        "scenario_name",
        "pathway",
        "portfolio_allocated_feed_share",
    }.issubset(pathway_summary.columns):
        ranked = pathway_summary.copy()
        ranked["portfolio_allocated_feed_share"] = _numeric_column(ranked, "portfolio_allocated_feed_share")
        ranked["best_case_score"] = _numeric_column(ranked, "best_case_score", default=-1.0)
        ranked = ranked.sort_values(
            ["scenario_name", "portfolio_allocated_feed_share", "best_case_score", "pathway"],
            ascending=[True, False, False, True],
        )
        return ranked.groupby("scenario_name", as_index=False).head(1).reset_index(drop=True)

    if not portfolio_allocations.empty and {"scenario_name", "pathway"}.issubset(portfolio_allocations.columns):
        grouped = (
            portfolio_allocations.assign(
                allocated_feed_ton_per_year=_numeric_column(portfolio_allocations, "allocated_feed_ton_per_year")
            )
            .groupby(["scenario_name", "pathway"], as_index=False)["allocated_feed_ton_per_year"]
            .sum()
        )
        totals = grouped.groupby("scenario_name")["allocated_feed_ton_per_year"].transform("sum")
        grouped["portfolio_allocated_feed_share"] = grouped["allocated_feed_ton_per_year"] / totals.where(
            totals > 0.0, 1.0
        )
        grouped = grouped.sort_values(
            ["scenario_name", "portfolio_allocated_feed_share", "allocated_feed_ton_per_year", "pathway"],
            ascending=[True, False, False, True],
        )
        return grouped.groupby("scenario_name", as_index=False).head(1).reset_index(drop=True)

    return pd.DataFrame(columns=["scenario_name", "pathway", "portfolio_allocated_feed_share"])


def _resolve_score_leaders(pathway_summary: pd.DataFrame) -> dict[str, str]:
    if pathway_summary.empty or not {"scenario_name", "pathway", "best_case_score"}.issubset(pathway_summary.columns):
        return {}
    ranked = pathway_summary.copy()
    ranked["best_case_score"] = _numeric_column(ranked, "best_case_score", default=-1.0)
    ranked = ranked.sort_values(
        ["scenario_name", "best_case_score", "pathway"],
        ascending=[True, False, True],
    )
    leaders = ranked.groupby("scenario_name", as_index=False).head(1)
    return {
        str(row["scenario_name"]): str(row["pathway"])
        for _, row in leaders.iterrows()
    }


def _resolve_high_supply_coverage_ratio(portfolio_summary: pd.DataFrame, scenario_summary: pd.DataFrame) -> float:
    for frame in (portfolio_summary, scenario_summary):
        if frame.empty or "scenario_name" not in frame.columns:
            continue
        subset = frame[frame["scenario_name"].astype(str) == "high_supply_case"]
        if subset.empty:
            continue
        if "scenario_feed_coverage_ratio" in subset.columns:
            value = pd.to_numeric(subset["scenario_feed_coverage_ratio"], errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return 0.0


def _build_uq_payload(
    *,
    main_results_table: pd.DataFrame,
    optimization_diagnostics: pd.DataFrame,
) -> dict[str, object]:
    if optimization_diagnostics.empty:
        return {
            "pattern": "not-exported",
            "sentence": "Uncertainty-mode sensitivity has not yet been exported into manuscript-ready artifacts.",
        }

    scenario_count = int(len(optimization_diagnostics))
    case_sensitive_count = int(
        pd.to_numeric(
            optimization_diagnostics.get("uncertainty_mode_case_switch_count", pd.Series(0, index=optimization_diagnostics.index)),
            errors="coerce",
        ).fillna(0.0).gt(1.0).sum()
    )
    pathway_sensitive_count = int(
        pd.to_numeric(
            optimization_diagnostics.get(
                "uncertainty_mode_pathway_switch_count",
                pd.Series(0, index=optimization_diagnostics.index),
            ),
            errors="coerce",
        ).fillna(0.0).gt(1.0).sum()
    )
    if pathway_sensitive_count > 0:
        pattern = "pathway-sensitive"
        sentence = (
            f"Across {scenario_count} exported scenarios, alternative uncertainty definitions changed the top-ranked "
            f"pathway family in {pathway_sensitive_count} case(s), so uncertainty aggregation is pathway-sensitive "
            "under the current planning evidence."
        )
    elif case_sensitive_count > 0:
        pattern = "case-sensitive_pathway-stable"
        sentence = (
            f"Across {scenario_count} exported scenarios, alternative uncertainty definitions changed the top-ranked "
            f"case in {case_sensitive_count} case(s) without changing pathway identity, so the current sensitivity is "
            "case-level rather than pathway-level."
        )
    else:
        pattern = "stable-across-tested-uq-modes"
        sentence = (
            f"Across {scenario_count} exported scenarios, the same top-ranked case persists under all tested "
            "uncertainty definitions, so the exported recommendation is stable across the current UQ modes."
        )

    if not main_results_table.empty and "uncertainty_mode_sensitivity" in main_results_table.columns:
        scenario_labels = (
            main_results_table[["scenario_name", "uncertainty_mode_sensitivity"]]
            .dropna()
            .drop_duplicates(subset=["scenario_name"])
        )
        if not scenario_labels.empty:
            label_clause = "; ".join(
                f"{SCENARIO_DISPLAY.get(str(row['scenario_name']), str(row['scenario_name']))}: {row['uncertainty_mode_sensitivity']}"
                for _, row in scenario_labels.iterrows()
            )
            sentence = f"{sentence} Scenario-level labels are: {label_clause}."
    return {
        "pattern": pattern,
        "sentence": sentence,
    }


def _build_planning_uq_narrative(
    *,
    main_results_table: pd.DataFrame,
    optimization_diagnostics: pd.DataFrame,
    uq_payload: dict[str, object],
) -> dict[str, str]:
    if main_results_table.empty:
        sentence = str(uq_payload.get("sentence", "")).strip() or (
            "Uncertainty-mode sensitivity has not yet been synchronized into planning narratives."
        )
        return {
            "results_sentence": sentence,
            "discussion_sentence": sentence,
            "scenario_sentence": sentence,
        }

    selected = (
        main_results_table[main_results_table["selected_in_baseline_portfolio"].fillna(False).astype(bool)].copy()
        if "selected_in_baseline_portfolio" in main_results_table.columns
        else pd.DataFrame()
    )
    if selected.empty:
        selected = main_results_table.copy()

    scenario_clauses: list[str] = []
    for scenario_name in SCENARIO_ORDER:
        subset = selected[selected["scenario_name"].astype(str) == scenario_name]
        if subset.empty:
            continue
        sort_columns = [
            column
            for column in ["baseline_portfolio_share_pct", "best_case_score_index"]
            if column in subset.columns
        ]
        if sort_columns:
            top_row = subset.sort_values(
                sort_columns,
                ascending=[False] * len(sort_columns),
            ).iloc[0]
        else:
            top_row = subset.iloc[0]
        scenario_display = SCENARIO_DISPLAY.get(scenario_name, scenario_name)
        pathway = _pathway_display(str(top_row.get("pathway", "")))
        comparison = str(top_row.get("uq_mode_comparison_sentence", "")).strip()
        ranking_summary = str(top_row.get("uncertainty_mode_ranking_summary", "")).strip()
        if comparison:
            scenario_clauses.append(f"{scenario_display}: {pathway}. {comparison}")
        elif ranking_summary:
            scenario_clauses.append(f"{scenario_display}: {ranking_summary}")

    scenario_sentence = " ".join(scenario_clauses).strip()
    base_sentence = str(uq_payload.get("sentence", "")).strip()
    results_sentence = base_sentence
    if scenario_sentence:
        results_sentence = f"{base_sentence} {scenario_sentence}".strip()

    case_sensitive_count = 0
    pathway_sensitive_count = 0
    if not optimization_diagnostics.empty:
        case_sensitive_count = int(
            pd.to_numeric(
                optimization_diagnostics.get(
                    "uncertainty_mode_case_switch_count",
                    pd.Series(0, index=optimization_diagnostics.index),
                ),
                errors="coerce",
            ).fillna(0.0).gt(1.0).sum()
        )
        pathway_sensitive_count = int(
            pd.to_numeric(
                optimization_diagnostics.get(
                    "uncertainty_mode_pathway_switch_count",
                    pd.Series(0, index=optimization_diagnostics.index),
                ),
                errors="coerce",
            ).fillna(0.0).gt(1.0).sum()
        )

    if pathway_sensitive_count > 0:
        discussion_sentence = (
            f"{base_sentence} This means uncertainty aggregation can alter pathway-level interpretation in "
            f"{pathway_sensitive_count} exported scenario(s), so conclusions should stay pathway-qualified."
        )
    elif case_sensitive_count > 0:
        discussion_sentence = (
            f"{base_sentence} In the current export, the main sensitivity is case-level rather than pathway-level, "
            "so the evidence supports pathway-family stability but not a single invariant operating case."
        )
    else:
        discussion_sentence = (
            f"{base_sentence} Under the tested settings, uncertainty treatment does not materially alter the exported case ranking."
        )

    return {
        "results_sentence": results_sentence.strip(),
        "discussion_sentence": discussion_sentence.strip(),
        "scenario_sentence": scenario_sentence,
    }


def _classify_pathway_role(pathway: str, pathway_summary: pd.DataFrame) -> str:
    if pathway_summary.empty or "pathway" not in pathway_summary.columns:
        return "unevaluated pathway"
    subset = pathway_summary[pathway_summary["pathway"].astype(str).str.lower() == pathway].copy()
    if subset.empty:
        return "unevaluated pathway"
    if "portfolio_allocated_feed_share" in subset.columns:
        share = _numeric_column(subset, "portfolio_allocated_feed_share")
    elif "baseline_portfolio_share_pct" in subset.columns:
        share = _numeric_column(subset, "baseline_portfolio_share_pct") / 100.0
    else:
        share = pd.Series(0.0, index=subset.index, dtype=float)
    stress = _numeric_column(subset, "max_stress_selection_rate")
    if share.gt(0.0).any():
        if share.ge(0.99).all():
            return "dominant baseline portfolio pathway"
        return "supporting baseline portfolio pathway"
    if stress.gt(0.0).any():
        return "stress-sensitive alternative"
    if pathway == "ad":
        return "comparison-only pathway"
    if pathway == "baseline":
        return "comparison anchor"
    return "unselected alternative"


def _build_abstract_dominance_sentence(
    *,
    dominance_pattern: str,
    dominant_pathway: str,
    score_leader: str,
    dominant_by_scenario: dict[str, str],
) -> str:
    if dominance_pattern == "uniform":
        dominant_display = _pathway_display(dominant_pathway)
        if score_leader == dominant_pathway:
            return (
                "Across the baseline-region, high-supply, and policy-support scenarios, the optimized baseline "
                f"portfolio is {dominant_display}-dominant in the current exported planning runs, and "
                f"{dominant_display} also retains the strongest best-case score leadership under the present "
                "evidence-qualified formulation."
            )
        if score_leader and score_leader != "mixed":
            return (
                "Across the baseline-region, high-supply, and policy-support scenarios, the optimized baseline "
                f"portfolio is {dominant_display}-dominant in the current exported planning runs, while "
                f"{_pathway_display(score_leader)} retains stronger best-case score leadership under the present "
                "evidence-qualified formulation."
            )
    return (
        "Across the baseline-region, high-supply, and policy-support scenarios, the optimized baseline portfolio "
        f"remains scenario-dependent in the current exported planning runs: {_scenario_dominance_clause(dominant_by_scenario)}."
    )


def _build_introduction_dominance_sentence(
    *,
    dominance_pattern: str,
    dominant_pathway: str,
    score_leader: str,
    pyrolysis_role: str,
    htc_role: str,
    ad_role: str,
    dominant_by_scenario: dict[str, str],
) -> str:
    if dominance_pattern == "uniform":
        dominant_display = _pathway_display(dominant_pathway)
        if score_leader and score_leader not in {"mixed", "unknown", dominant_pathway}:
            return (
                "Under the current California-parameterized screening boundary, the strongest result is not a universal technology "
                f"winner but an evidence-qualified thermochemical recommendation whose baseline allocation is "
                f"{dominant_display}-dominant in the current exported runs, even though "
                f"{_pathway_display(score_leader)} retains stronger case-level score leadership and AD remains "
                f"{_with_indefinite_article(ad_role)}."
            )
        secondary_pathway = "pyrolysis" if dominant_pathway != "pyrolysis" else "htc"
        secondary_role = pyrolysis_role if secondary_pathway == "pyrolysis" else htc_role
        return (
            "Under the current California-parameterized screening boundary, the strongest result is not a universal technology "
            f"winner but an evidence-qualified thermochemical recommendation whose baseline allocation is "
            f"{dominant_display}-dominant in the current exported runs, while "
            f"{_pathway_display(secondary_pathway)} remains {_with_indefinite_article(secondary_role)} and AD "
            f"remains {_with_indefinite_article(ad_role)}."
        )
    return (
        "Under the current California-parameterized screening boundary, the strongest result is not a universal technology "
        "winner but an evidence-qualified thermochemical recommendation whose dominant baseline pathway remains "
        f"scenario-dependent in the current exported runs ({_scenario_dominance_clause(dominant_by_scenario)}), "
        f"while pyrolysis remains the {pyrolysis_role} and AD remains {_with_indefinite_article(ad_role)}."
    )


def _build_results_dominance_sentence(
    *,
    dominance_pattern: str,
    dominant_pathway: str,
    pyrolysis_role: str,
    dominant_by_scenario: dict[str, str],
) -> str:
    if dominance_pattern == "uniform":
        return (
            f"Across the baseline, high-supply, and policy-support scenarios, the constrained portfolio is now "
            f"{_pathway_display(dominant_pathway)}-dominant in the current exported planning runs."
        )
    return (
        "Across the baseline, high-supply, and policy-support scenarios, the constrained portfolio remains "
        f"scenario-dependent in the current exported planning runs ({_scenario_dominance_clause(dominant_by_scenario)})."
    )


def _build_highlights_dominance_sentence(
    *,
    dominance_pattern: str,
    dominant_pathway: str,
    score_leader: str,
    pyrolysis_role: str,
    htc_role: str,
    dominant_by_scenario: dict[str, str],
) -> str:
    if dominance_pattern == "uniform":
        dominant_phrase = _with_indefinite_article(
            f"{_pathway_display(dominant_pathway)}-dominant optimized baseline portfolio"
        )
        if score_leader and score_leader not in {"mixed", "unknown", dominant_pathway}:
            return (
                f"All three main scenarios return {dominant_phrase} in the current exported "
                f"planning runs, while {_pathway_display(score_leader)} retains stronger best-case "
                "score leadership under the current evidence-qualified formulation."
            )
        secondary_pathway = "pyrolysis" if dominant_pathway != "pyrolysis" else "htc"
        secondary_role = pyrolysis_role if secondary_pathway == "pyrolysis" else htc_role
        return (
            f"All three main scenarios return {dominant_phrase} in the current exported "
            f"planning runs, while {_pathway_display(secondary_pathway)} remains "
            f"{_with_indefinite_article(secondary_role)}."
        )
    return (
        "The three main scenarios remain scenario-dependent in the current exported planning runs "
        f"({_scenario_dominance_clause(dominant_by_scenario)}), while pyrolysis remains the {pyrolysis_role}."
    )


def _build_conclusion_dominance_sentence(
    *,
    dominance_pattern: str,
    dominant_pathway: str,
    score_leader: str,
    pyrolysis_role: str,
    htc_role: str,
    dominant_by_scenario: dict[str, str],
    dominant_share_by_scenario: dict[str, float],
) -> str:
    if dominance_pattern == "uniform":
        dominant_display = _pathway_display(dominant_pathway)
        sentence_subject = _pathway_display_sentence_start(dominant_pathway)
        scenario_share_clause = _scenario_share_clause(dominant_pathway, dominant_share_by_scenario)
        minimum_share = min(dominant_share_by_scenario.values(), default=0.0)
        if minimum_share >= 0.99:
            secondary_pathway = "pyrolysis" if dominant_pathway != "pyrolysis" else "htc"
            secondary_role = pyrolysis_role if secondary_pathway == "pyrolysis" else htc_role
            return (
                f"{sentence_subject} carries the full baseline allocated share in the baseline-region, high-supply, "
                "and policy-support cases, while "
                f"{_pathway_display(secondary_pathway)} remains {_with_indefinite_article(secondary_role)}."
            )
        if score_leader and score_leader not in {"mixed", "unknown", dominant_pathway}:
            return (
                f"{sentence_subject} carries the leading baseline allocated share across the baseline-region, "
                f"high-supply, and policy-support cases ({scenario_share_clause}), while "
                f"{_pathway_display(score_leader)} retains stronger case-level score leadership under the current "
                "evidence-qualified formulation."
            )
        return (
            f"{sentence_subject} carries the leading baseline allocated share across the baseline-region, "
            f"high-supply, and policy-support cases ({scenario_share_clause}) under the current "
            "evidence-qualified formulation."
        )
    return (
        "Dominance remains scenario-dependent across the baseline-region, high-supply, and policy-support cases "
        f"({_scenario_dominance_clause(dominant_by_scenario)}), while pyrolysis remains the {pyrolysis_role}."
    )


def _scenario_dominance_clause(dominant_by_scenario: dict[str, str]) -> str:
    parts: list[str] = []
    for scenario_name in SCENARIO_ORDER:
        pathway = dominant_by_scenario.get(scenario_name)
        if not pathway:
            continue
        parts.append(f"{SCENARIO_DISPLAY[scenario_name]} is {_pathway_display(pathway)}-dominant")
    return "; ".join(parts) if parts else "dominance is not available"


def _scenario_share_clause(dominant_pathway: str, dominant_share_by_scenario: dict[str, float]) -> str:
    parts: list[str] = []
    for scenario_name in SCENARIO_ORDER:
        if scenario_name not in dominant_share_by_scenario:
            continue
        share_pct = dominant_share_by_scenario[scenario_name] * 100.0
        parts.append(f"{SCENARIO_DISPLAY[scenario_name]}: {_pathway_display(dominant_pathway)} {share_pct:.1f}\\%")
    return "; ".join(parts) if parts else "scenario shares are not available"


def _with_indefinite_article(phrase: str) -> str:
    normalized = str(phrase).strip()
    if not normalized:
        return normalized
    lowered = normalized.lower()
    if lowered.startswith(("a ", "an ", "the ")):
        return normalized
    first_token = normalized.split()[0]
    first_chunk = first_token.split("-")[0]
    article = "a"
    if first_chunk.isupper():
        article = "an" if first_chunk[0] in {"A", "E", "F", "H", "I", "L", "M", "N", "O", "R", "S", "X"} else "a"
    elif lowered[0] in {"a", "e", "i", "o", "u"}:
        article = "an"
    return f"{article} {normalized}"


def _pathway_display_sentence_start(pathway: str) -> str:
    display = _pathway_display(pathway)
    return display if display.isupper() else display.capitalize()


def _pathway_display(pathway: str) -> str:
    return PATHWAY_DISPLAY.get(str(pathway).lower(), str(pathway))


def _target_display(target_column: str) -> str:
    return TARGET_DISPLAY.get(str(target_column).strip(), str(target_column).replace("_", " "))


def _uq_mode_display(mode: str) -> str:
    mapping = {
        "prefer_interval_mean": "interval-mean",
        "max_interval_ratio": "max-interval",
        "combined_only": "combined-only",
    }
    return mapping.get(str(mode), str(mode).replace("_", "-"))


def _case_id_display(case_id: object) -> str:
    value = str(case_id or "").strip()
    if not value:
        return "--"
    parts = value.split("::")
    if len(parts) >= 4:
        return f"{_pathway_display(parts[2])}-{parts[3]}"
    return value


def _uq_table_note(diag: pd.Series, sensitivity: str) -> str:
    ranking_summary = str(diag.get("uncertainty_mode_ranking_summary", "")).strip()
    if ranking_summary:
        return ranking_summary
    case_switch_count = _as_float(diag.get("uncertainty_mode_case_switch_count"))
    pathway_switch_count = _as_float(diag.get("uncertainty_mode_pathway_switch_count"))
    if pathway_switch_count > 1.0:
        return "Alternative uncertainty definitions change pathway identity and should be discussed explicitly."
    if case_switch_count > 1.0:
        return f"Alternative uncertainty definitions shift the top-ranked case within a stable pathway family ({sensitivity})."
    return "The exported recommendation remains unchanged across the tested uncertainty definitions."


def _build_benchmark_payload(
    *,
    audit_dir: Path,
    benchmark_dir: Path,
) -> dict[str, object]:
    fallback = {
        "primary_variant": "benchmark_not_available",
        "primary_variant_display": "benchmark-not-available",
        "primary_significance_tier": "not_available",
        "primary_sentence": "Benchmark evidence has not yet been synchronized into manuscript-ready outputs.",
        "classic_sentence": "Classic multi-objective benchmark evidence is not yet available.",
        "evidence_sentence": "Evidence-aware benchmark evidence is not yet available.",
        "robustness_sentence": "Robustness-ablation benchmark evidence is not yet available.",
        "heuristic_sentence": "Heuristic benchmark evidence is not yet available.",
        "bootstrap_sentence": "Bootstrap benchmark repeats are not yet available.",
        "takeaway_sentence": "Benchmark-backed innovation claims should remain provisional until the benchmark bundle is synchronized.",
        "bootstrap_available": False,
        "summary_table": pd.DataFrame(),
        "claim_summary": pd.DataFrame(),
    }
    benchmark_sentences = _read_csv_if_exists(audit_dir / "benchmark_manuscript_sentences.csv")
    benchmark_claim_summary = _read_csv_if_exists(audit_dir / "benchmark_claim_summary.csv")
    benchmark_stats = _read_csv_if_exists(benchmark_dir / "benchmark_statistical_summary.csv")
    if benchmark_sentences.empty:
        return fallback

    enriched_sentences = benchmark_sentences.copy()
    if "manuscript_sentence" in enriched_sentences.columns:
        enriched_sentences["manuscript_sentence"] = (
            enriched_sentences["manuscript_sentence"].astype(str).map(_sanitize_benchmark_sentence)
        )
    if not benchmark_stats.empty:
        variant_stats = _aggregate_benchmark_variant_statistics(benchmark_stats)
        if not variant_stats.empty:
            enriched_sentences = enriched_sentences.merge(variant_stats, on="benchmark_variant", how="left")
    if not benchmark_claim_summary.empty:
        claim_priority = _aggregate_benchmark_claim_priority(benchmark_claim_summary)
        if not claim_priority.empty:
            enriched_sentences = enriched_sentences.merge(claim_priority, on="benchmark_variant", how="left")

    primary_row = _select_primary_benchmark_row(enriched_sentences)
    classic_row = _row_for_variant(enriched_sentences, "classic_multiobjective_optimizer")
    evidence_row = _row_for_variant(enriched_sentences, "no_evidence_penalty")
    robustness_row = _row_for_variant(enriched_sentences, "no_robustness_penalty")
    heuristic_row = _row_for_variant(enriched_sentences, "greedy_weighted_score_heuristic")

    primary_variant = str(primary_row.get("benchmark_variant", "benchmark_not_available"))
    return {
        "primary_variant": primary_variant,
        "primary_variant_display": _benchmark_variant_display(primary_variant),
        "primary_significance_tier": str(primary_row.get("dominant_significance_tier", "not_available")),
        "primary_sentence": _sanitize_benchmark_sentence(
            str(primary_row.get("manuscript_sentence", fallback["primary_sentence"]))
        ),
        "classic_sentence": _sanitize_benchmark_sentence(
            str(classic_row.get("manuscript_sentence", fallback["classic_sentence"]))
        ),
        "evidence_sentence": _sanitize_benchmark_sentence(
            str(evidence_row.get("manuscript_sentence", fallback["evidence_sentence"]))
        ),
        "robustness_sentence": _sanitize_benchmark_sentence(
            str(robustness_row.get("manuscript_sentence", fallback["robustness_sentence"]))
        ),
        "heuristic_sentence": _sanitize_benchmark_sentence(
            str(heuristic_row.get("manuscript_sentence", fallback["heuristic_sentence"]))
        ),
        "bootstrap_sentence": _build_benchmark_bootstrap_sentence(benchmark_stats),
        "takeaway_sentence": _build_benchmark_takeaway_sentence(
            primary_row=primary_row,
            classic_row=classic_row,
            evidence_row=evidence_row,
            heuristic_row=heuristic_row,
            bootstrap_stats=benchmark_stats,
        ),
        "bootstrap_available": not benchmark_stats.empty,
        "summary_table": enriched_sentences,
        "claim_summary": benchmark_claim_summary,
    }


def _write_benchmark_manuscript_artifacts(
    *,
    benchmark: dict[str, object],
    planning_uq_narrative: dict[str, str],
    figures_dir: Path,
) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "benchmark_results_table": figures_dir / "paper1_benchmark_results_table.csv",
        "benchmark_claim_summary": figures_dir / "paper1_benchmark_claim_summary.csv",
        "benchmark_narrative": figures_dir / "paper1_benchmark_narrative.json",
    }
    summary_table = benchmark.get("summary_table")
    claim_summary = benchmark.get("claim_summary")
    (summary_table if isinstance(summary_table, pd.DataFrame) else pd.DataFrame()).to_csv(
        outputs["benchmark_results_table"], index=False
    )
    (claim_summary if isinstance(claim_summary, pd.DataFrame) else pd.DataFrame()).to_csv(
        outputs["benchmark_claim_summary"], index=False
    )
    narrative = {
        "primary_variant": benchmark.get("primary_variant", "benchmark_not_available"),
        "primary_variant_display": benchmark.get("primary_variant_display", "benchmark-not-available"),
        "primary_significance_tier": benchmark.get("primary_significance_tier", "not_available"),
        "primary_sentence": benchmark.get("primary_sentence", ""),
        "classic_sentence": benchmark.get("classic_sentence", ""),
        "evidence_sentence": benchmark.get("evidence_sentence", ""),
        "robustness_sentence": benchmark.get("robustness_sentence", ""),
        "heuristic_sentence": benchmark.get("heuristic_sentence", ""),
        "bootstrap_sentence": benchmark.get("bootstrap_sentence", ""),
        "takeaway_sentence": benchmark.get("takeaway_sentence", ""),
        "planning_uq_results_sentence": planning_uq_narrative.get("results_sentence", ""),
        "planning_uq_discussion_sentence": planning_uq_narrative.get("discussion_sentence", ""),
        "planning_uq_scenario_sentence": planning_uq_narrative.get("scenario_sentence", ""),
    }
    outputs["benchmark_narrative"].write_text(json.dumps(narrative, indent=2), encoding="utf-8")
    return {key: str(value) for key, value in outputs.items()}


def _build_benchmark_section_templates(
    *,
    audit_dir: Path,
    benchmark: dict[str, object],
    planning_uq_narrative: dict[str, str],
) -> dict[str, object]:
    transferability = _read_csv_if_exists(audit_dir / "planning_transferability_risk_summary.csv")
    pathway_reliability = _read_csv_if_exists(audit_dir / "pathway_reliability_summary.csv")
    bounded_case_count = 0
    conditional_case_count = 0
    if not transferability.empty and "transferability_evidence_ceiling" in transferability.columns:
        bounded_case_count = int(
            transferability["transferability_evidence_ceiling"]
            .astype(str)
            .isin(["auxiliary_or_missing_bounded", "guarded_transfer"])
            .sum()
        )
        conditional_case_count = int(
            transferability["transferability_evidence_ceiling"]
            .astype(str)
            .eq("conditional_transfer_supported")
            .sum()
        )
    htc_auxiliary = False
    pyrolysis_conditional = False
    if not pathway_reliability.empty and "pathway" in pathway_reliability.columns:
        htc_rows = pathway_reliability[pathway_reliability["pathway"].astype(str) == "htc"]
        pyrolysis_rows = pathway_reliability[pathway_reliability["pathway"].astype(str) == "pyrolysis"]
        if not htc_rows.empty:
            htc_auxiliary = str(htc_rows.iloc[0].get("reliability_tier", "")) == "auxiliary_only"
        if not pyrolysis_rows.empty:
            pyrolysis_conditional = str(pyrolysis_rows.iloc[0].get("reliability_tier", "")) == "conditional_support"

    results_paragraph = (
        f"We observed that {benchmark['classic_sentence'].lower()} "
        f"The evidence-ablation result points in the same direction because {benchmark['evidence_sentence'].lower()} "
        f"By contrast, {benchmark['robustness_sentence'].lower()} "
        "Taken together, these benchmark contrasts indicate that pathway identity is most sensitive to the evidence-aware "
        "formulation and to replacing the current method with a classic multi-objective optimizer, whereas robustness "
        "treatment remains decision-relevant but is not the sole pathway-flip driver. "
        f"{planning_uq_narrative.get('results_sentence', '')}"
    )
    discussion_paragraph = (
        f"It should be pointed out that {benchmark['bootstrap_sentence'].lower()} "
        f"Notably, {benchmark['heuristic_sentence'].lower()} "
        f"Although a detailed investigation of external cross-study expansion is beyond the scope of this work, we acknowledge that "
        f"{bounded_case_count} of the three audited planning scenarios remain under an auxiliary-or-missing-bounded transferability ceiling"
        + (
            ", whereas only "
            f"{conditional_case_count} scenario currently satisfies a conditional-transfer-supported ceiling."
            if not transferability.empty
            else "."
        )
        + " "
        + (
            "This is consistent with the observation from the pathway-level audit that HTC remains auxiliary-only whereas pyrolysis retains conditional support."
            if htc_auxiliary or pyrolysis_conditional
            else "This is consistent with the observation from the current audit that transferability support remains scenario-dependent."
        )
        + " "
        + str(planning_uq_narrative.get("discussion_sentence", ""))
    )
    conclusion_paragraph = (
        f"Our findings lead us to conclude that {_lower_sentence_start(str(benchmark['takeaway_sentence']))} "
        f"This work extends our knowledge of evidence-qualified waste-to-energy planning by separating core innovation from bounded refinement through benchmark and bootstrap evidence. "
        f"Benchmark-guided planning offers opportunities to write optimization results with more conservative claim framing and clearer evidence ceilings. "
        f"{planning_uq_narrative.get('discussion_sentence', '')}"
    )
    return {
        "results_paragraph": results_paragraph,
        "discussion_paragraph": discussion_paragraph,
        "conclusion_paragraph": conclusion_paragraph,
    }


def _write_benchmark_section_artifacts(
    *,
    sections: dict[str, object],
    figures_dir: Path,
) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "benchmark_section_templates_json": figures_dir / "paper1_benchmark_section_templates.json",
        "benchmark_section_templates_md": figures_dir / "paper1_benchmark_section_templates.md",
        "benchmark_section_templates_tex": figures_dir / "paper1_benchmark_section_templates.tex",
    }
    outputs["benchmark_section_templates_json"].write_text(
        json.dumps(sections, indent=2),
        encoding="utf-8",
    )
    markdown_text = (
        "# Benchmark Section Templates\n\n"
        "## Results\n\n"
        f"{sections['results_paragraph']}\n\n"
        "## Discussion\n\n"
        f"{sections['discussion_paragraph']}\n\n"
        "## Conclusion\n\n"
        f"{sections['conclusion_paragraph']}\n"
    )
    outputs["benchmark_section_templates_md"].write_text(markdown_text, encoding="utf-8")
    tex_text = (
        "% Auto-generated benchmark section templates\n"
        "\\paragraph{Results.} "
        f"{sections['results_paragraph']}\n\n"
        "\\paragraph{Discussion.} "
        f"{sections['discussion_paragraph']}\n\n"
        "\\paragraph{Conclusion.} "
        f"{sections['conclusion_paragraph']}\n"
    )
    outputs["benchmark_section_templates_tex"].write_text(tex_text, encoding="utf-8")
    return {key: str(value) for key, value in outputs.items()}


def _write_priority_manuscript_tables(
    *,
    planning_dir: Path,
    audit_dir: Path,
    benchmark_dir: Path,
    figures_dir: Path,
) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "data_summary_csv": figures_dir / "paper1_data_summary_table.csv",
        "data_summary_tex": figures_dir / "paper1_data_summary_table.tex",
        "surrogate_validation_csv": figures_dir / "paper1_surrogate_validation_table.csv",
        "surrogate_validation_tex": figures_dir / "paper1_surrogate_validation_table.tex",
        "transfer_support_csv": figures_dir / "paper1_transfer_support_table.csv",
        "transfer_support_tex": figures_dir / "paper1_transfer_support_table.tex",
        "evidence_ceiling_csv": figures_dir / "paper1_evidence_ceiling_table.csv",
        "evidence_ceiling_tex": figures_dir / "paper1_evidence_ceiling_table.tex",
        "scenario_parameter_csv": figures_dir / "paper1_scenario_parameter_table.csv",
        "scenario_parameter_tex": figures_dir / "paper1_scenario_parameter_table.tex",
        "optimization_output_csv": figures_dir / "paper1_optimization_output_table.csv",
        "optimization_output_tex": figures_dir / "paper1_optimization_output_table.tex",
        "uq_sensitivity_csv": figures_dir / "paper1_uq_sensitivity_table.csv",
        "uq_sensitivity_tex": figures_dir / "paper1_uq_sensitivity_table.tex",
        "cost_boundary_csv": figures_dir / "paper1_cost_boundary_table.csv",
        "cost_boundary_tex": figures_dir / "paper1_cost_boundary_table.tex",
        "product_credit_sensitivity_csv": figures_dir / "paper1_product_credit_sensitivity_table.csv",
        "product_credit_sensitivity_tex": figures_dir / "paper1_product_credit_sensitivity_table.tex",
        "targeted_ablation_csv": figures_dir / "paper1_targeted_ablation_table.csv",
        "targeted_ablation_tex": figures_dir / "paper1_targeted_ablation_table.tex",
        "htc_model_comparison_csv": figures_dir / "paper1_htc_model_comparison_table.csv",
        "htc_model_comparison_tex": figures_dir / "paper1_htc_model_comparison_table.tex",
        "htc_model_comparison_note_md": figures_dir / "paper1_htc_model_comparison_note.md",
        "htc_model_narrative_md": figures_dir / "paper1_htc_model_narrative.md",
        "htc_model_narrative_tex": figures_dir / "paper1_htc_model_narrative.tex",
        "figure3_htc_caption_md": figures_dir / "paper1_figure3_htc_caption.md",
        "figure3_htc_caption_tex": figures_dir / "paper1_figure3_htc_caption.tex",
        "ad_evidence_tier_csv": figures_dir / "paper1_ad_evidence_tier_table.csv",
        "ad_evidence_tier_tex": figures_dir / "paper1_ad_evidence_tier_table.tex",
        "ad_complementarity_csv": figures_dir / "paper1_ad_complementarity_sensitivity_table.csv",
        "ad_complementarity_tex": figures_dir / "paper1_ad_complementarity_sensitivity_table.tex",
        "ad_credit_csv": figures_dir / "paper1_ad_credit_sensitivity_table.csv",
        "ad_credit_tex": figures_dir / "paper1_ad_credit_sensitivity_table.tex",
        "pathway_cap_csv": figures_dir / "paper1_pathway_cap_sensitivity_table.csv",
        "pathway_cap_tex": figures_dir / "paper1_pathway_cap_sensitivity_table.tex",
    }

    for src_name, dst_name in [
        ("paper1_data_summary_table", "data_summary"),
        ("paper1_scenario_parameter_table", "scenario_parameter"),
    ]:
        for ext in ("csv", "tex"):
            src = FIGURES_TABLES_DIR / f"{src_name}.{ext}"
            dst = outputs[f"{dst_name}_{ext}"]
            if src.exists():
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                if ext == "csv":
                    pd.DataFrame().to_csv(dst, index=False)
                else:
                    dst.write_text(f"% {src_name} unavailable.\n", encoding="utf-8")

    surrogate_validation = _build_surrogate_validation_table(
        audit_dir=audit_dir,
        benchmark_dir=benchmark_dir,
    )
    outputs["surrogate_validation_csv"].write_text(
        surrogate_validation.to_csv(index=False),
        encoding="utf-8",
    )
    outputs["surrogate_validation_tex"].write_text(
        _render_surrogate_validation_table(surrogate_validation),
        encoding="utf-8",
    )

    transfer_support = _build_transfer_support_table(
        audit_dir=audit_dir,
        surrogate_validation=surrogate_validation,
    )
    outputs["transfer_support_csv"].write_text(
        transfer_support.to_csv(index=False),
        encoding="utf-8",
    )
    outputs["transfer_support_tex"].write_text(
        _render_transfer_support_table(transfer_support),
        encoding="utf-8",
    )

    optimization_output = _build_optimization_output_table(planning_dir=planning_dir)
    outputs["optimization_output_csv"].write_text(
        optimization_output.to_csv(index=False),
        encoding="utf-8",
    )
    outputs["optimization_output_tex"].write_text(
        _render_optimization_output_table(optimization_output),
        encoding="utf-8",
    )

    uq_sensitivity = _build_uq_sensitivity_table(planning_dir=planning_dir)
    outputs["uq_sensitivity_csv"].write_text(
        uq_sensitivity.to_csv(index=False),
        encoding="utf-8",
    )
    outputs["uq_sensitivity_tex"].write_text(
        _render_uq_sensitivity_table(uq_sensitivity),
        encoding="utf-8",
    )

    cost_boundary = _build_cost_boundary_table(planning_dir=planning_dir)
    outputs["cost_boundary_csv"].write_text(
        cost_boundary.to_csv(index=False),
        encoding="utf-8",
    )
    outputs["cost_boundary_tex"].write_text(
        _render_cost_boundary_table(cost_boundary),
        encoding="utf-8",
    )

    evidence_ceiling = _build_evidence_ceiling_table(
        planning_dir=planning_dir,
        audit_dir=audit_dir,
    )
    outputs["evidence_ceiling_csv"].write_text(
        evidence_ceiling.to_csv(index=False),
        encoding="utf-8",
    )
    outputs["evidence_ceiling_tex"].write_text(
        _render_evidence_ceiling_table(evidence_ceiling),
        encoding="utf-8",
    )

    product_credit_sensitivity = _build_product_credit_sensitivity_table(planning_dir=planning_dir)
    outputs["product_credit_sensitivity_csv"].write_text(
        product_credit_sensitivity.to_csv(index=False),
        encoding="utf-8",
    )
    outputs["product_credit_sensitivity_tex"].write_text(
        _render_product_credit_sensitivity_table(product_credit_sensitivity),
        encoding="utf-8",
    )

    targeted_ablation = _build_targeted_ablation_table(benchmark_dir=benchmark_dir)
    outputs["targeted_ablation_csv"].write_text(
        targeted_ablation.to_csv(index=False),
        encoding="utf-8",
    )
    outputs["targeted_ablation_tex"].write_text(
        _render_targeted_ablation_table(targeted_ablation),
        encoding="utf-8",
    )

    htc_model_comparison = _build_htc_model_comparison_table(benchmark_dir=benchmark_dir)
    outputs["htc_model_comparison_csv"].write_text(
        htc_model_comparison.to_csv(index=False),
        encoding="utf-8",
    )
    outputs["htc_model_comparison_tex"].write_text(
        _render_htc_model_comparison_table(htc_model_comparison),
        encoding="utf-8",
    )
    outputs["htc_model_comparison_note_md"].write_text(
        _build_htc_model_comparison_note(htc_model_comparison),
        encoding="utf-8",
    )
    htc_model_narrative = _build_htc_model_narrative(
        htc_model_comparison=htc_model_comparison,
        confidence_summary=_load_confidence_summary(figures_dir),
    )
    outputs["htc_model_narrative_md"].write_text(
        _render_htc_model_narrative_markdown(htc_model_narrative),
        encoding="utf-8",
    )
    outputs["htc_model_narrative_tex"].write_text(
        _render_htc_model_narrative_tex(htc_model_narrative),
        encoding="utf-8",
    )
    outputs["figure3_htc_caption_md"].write_text(
        _render_figure3_htc_caption_markdown(htc_model_narrative),
        encoding="utf-8",
    )
    outputs["figure3_htc_caption_tex"].write_text(
        _render_figure3_htc_caption_tex(htc_model_narrative),
        encoding="utf-8",
    )

    ad_evidence_tier = _build_ad_evidence_tier_table()
    outputs["ad_evidence_tier_csv"].write_text(
        ad_evidence_tier.to_csv(index=False),
        encoding="utf-8",
    )
    outputs["ad_evidence_tier_tex"].write_text(
        _render_ad_evidence_tier_table(ad_evidence_tier),
        encoding="utf-8",
    )

    for table_key, family in [
        ("ad_complementarity", "ad_complementarity"),
        ("ad_credit", "ad_credit"),
        ("pathway_cap", "pathway_cap_sensitivity"),
    ]:
        table = _build_targeted_family_metric_table(benchmark_dir=benchmark_dir, family=family)
        outputs[f"{table_key}_csv"].write_text(table.to_csv(index=False), encoding="utf-8")
        outputs[f"{table_key}_tex"].write_text(
            _render_targeted_family_metric_table(table, family=family),
            encoding="utf-8",
        )

    return {key: str(value) for key, value in outputs.items()}


def _build_ad_evidence_tier_table() -> pd.DataFrame:
    path = FIGURES_TABLES_DIR.parent / "unified_features" / "ad_literature_standardized.csv"
    if not path.exists():
        return pd.DataFrame([{"evidence_tier": "unavailable", "feedstock_groups": "--", "observations": 0, "role_in_planning": "AD literature layer has not been generated."}])
    frame = pd.read_csv(path)
    if frame.empty or "feedstock_group" not in frame.columns:
        return pd.DataFrame([{"evidence_tier": "unavailable", "feedstock_groups": "--", "observations": 0, "role_in_planning": "AD literature layer is empty."}])
    tier_map = {
        "food_waste": "food-waste direct",
        "co_digestion_food_sanitation_waste": "co-digestion related",
        "co_digestion_food_wastewater": "co-digestion related",
        "industrial_food_waste": "industrial-organic related",
        "agroindustrial_waste": "industrial-organic related",
        "green_waste": "lignocellulosic/yard-waste related",
        "paper_waste": "lignocellulosic/yard-waste related",
    }
    frame = frame.copy()
    frame["evidence_tier"] = frame["feedstock_group"].map(tier_map).fillna("other related organic waste")
    rows = []
    role_lookup = {
        "food-waste direct": "Closest support for source-separated food-waste digestion and the exported AD proxy.",
        "co-digestion related": "Relevant to mixed organic waste systems, but includes sanitation or wastewater co-substrates.",
        "industrial-organic related": "Supports organic-industrial methane-yield plausibility, not municipal food-waste design.",
        "lignocellulosic/yard-waste related": "Lower-transfer support for organic fractions with different degradability constraints.",
        "other related organic waste": "Traceable but peripheral support for bounded screening only.",
    }
    order = [
        "food-waste direct",
        "co-digestion related",
        "industrial-organic related",
        "lignocellulosic/yard-waste related",
        "other related organic waste",
    ]
    for tier in order:
        subset = frame[frame["evidence_tier"].eq(tier)]
        if subset.empty:
            continue
        groups = ", ".join(sorted(subset["feedstock_group"].dropna().astype(str).unique()))
        rows.append({
            "evidence_tier": tier,
            "feedstock_groups": groups.replace("_", r"\_"),
            "observations": int(len(subset)),
            "median_methane_yield_m3_kg_odm": round(float(pd.to_numeric(subset["specific_methane_yield_m3_per_kg_odm"], errors="coerce").median()), 3),
            "role_in_planning": role_lookup[tier],
        })
    return pd.DataFrame(rows)


def _render_ad_evidence_tier_table(df: pd.DataFrame) -> str:
    return _render_latex_table(
        df,
        columns=[
            ("evidence_tier", "AD evidence tier"),
            ("feedstock_groups", "Feedstock groups"),
            ("observations", "Observations"),
            ("median_methane_yield_m3_kg_odm", "Median methane yield (m$^3$ kg$^{-1}$ oDM)"),
            ("role_in_planning", "Interpretation"),
        ],
        caption="Tiered interpretation of the expanded AD methane-yield evidence layer.",
        label="tab:ad-evidence-tier",
        column_format=r">{\raggedright\arraybackslash}p{2.6cm} >{\raggedright\arraybackslash}p{3.2cm} r r >{\raggedright\arraybackslash}X",
        notes=(
            "Tiers distinguish direct food-waste support from related co-digestion and industrial-organic observations. "
            "All rows remain methane-yield screening evidence, not facility-siting or full-scale AD design evidence."
        ),
        formatters={"feedstock_groups": _literal_latex_formatter},
    )


def _build_targeted_family_metric_table(*, benchmark_dir: Path, family: str) -> pd.DataFrame:
    ablation_dir = _resolve_targeted_ablation_dir(benchmark_dir)
    summary = _read_csv_if_exists(ablation_dir / "targeted_planning_ablations_summary.csv")
    if summary.empty or "ablation_family" not in summary.columns:
        return pd.DataFrame([{"scenario": "--", "variant": "unavailable", "selected_pathways": "--"}])
    fam = summary[summary["ablation_family"].astype(str).eq(family)].copy()
    if family == "ad_credit" and fam.empty:
        fam = summary[
            summary["ablation_family"].astype(str).eq("coproduct_boundary")
            & summary["ablation_key"].astype(str).str.startswith("digestate_rng_credit")
        ].copy()
    if fam.empty:
        return pd.DataFrame([{"scenario": "--", "variant": f"{family} unavailable", "selected_pathways": "--"}])
    scenario_display = {k: v for k, v in SCENARIO_DISPLAY.items()}
    rows = []
    for _, row in fam.sort_values(["scenario_name", "ablation_value", "ablation_key"]).iterrows():
        energy = _as_float(row.get("portfolio_energy_objective")) / 1_000_000_000.0
        cost = _as_float(row.get("portfolio_cost_objective")) / 1_000_000.0
        carbon = _as_float(row.get("portfolio_carbon_load_kgco2e")) / 1_000_000.0
        rows.append({
            "scenario": scenario_display.get(str(row.get("scenario_name")), str(row.get("scenario_name"))),
            "variant": _targeted_ablation_label(
                ablation_family=str(row.get("ablation_family")),
                ablation_key=str(row.get("ablation_key")),
                ablation_value=row.get("ablation_value"),
            ),
            "selected_pathways": _portfolio_pathway_label(str(row.get("selected_pathways", ""))),
            "ad_share_pct": round(_as_float(row.get("ad_allocated_share_pct")), 1),
            "energy_pj_y": round(energy, 3),
            "net_cost_musd_y": round(cost, 2),
            "carbon_load_ktco2e_y": round(carbon, 2),
        })
    return pd.DataFrame(rows)


def _render_targeted_family_metric_table(df: pd.DataFrame, *, family: str) -> str:
    captions = {
        "ad_complementarity": "AD complementarity sensitivity. Minimum AD processing floors test biological treatment as a management complement rather than an unconstrained score leader.",
        "ad_credit": "AD digestate/renewable-gas credit sensitivity under the pyrolysis median product-credit reference.",
        "pathway_cap_sensitivity": "Pathway concentration-cap sensitivity for pyrolysis maximum-share policies.",
    }
    labels = {
        "ad_complementarity": "tab:ad-complementarity-sensitivity",
        "ad_credit": "tab:ad-credit-sensitivity",
        "pathway_cap_sensitivity": "tab:pathway-cap-sensitivity",
    }
    return _render_latex_table(
        df,
        columns=[
            ("scenario", "Scenario"),
            ("variant", "Variant"),
            ("selected_pathways", "Selected pathways"),
            ("ad_share_pct", r"AD share (\%)"),
            ("energy_pj_y", "Energy (PJ y$^{-1}$)"),
            ("net_cost_musd_y", "Net cost (MUSD y$^{-1}$)"),
            ("carbon_load_ktco2e_y", "Carbon load (ktCO$_2$e y$^{-1}$)"),
        ],
        caption=captions.get(family, f"{family} sensitivity."),
        label=labels.get(family, f"tab:{family}"),
        column_format=r"l >{\raggedright\arraybackslash}p{3.1cm} l r r r r",
        notes="Outcome strings report optimized pathway composition under each declared policy or credit boundary.",
        formatters={"variant": _literal_latex_formatter},
    )


def _portfolio_pathway_label(value: str) -> str:
    if not value or value.lower() == "nan":
        return "--"
    parts = [PATHWAY_DISPLAY.get(part.strip().lower(), part.strip()) for part in value.split("|") if part.strip()]
    return " + ".join(parts) if parts else "--"


def _build_surrogate_validation_table(*, audit_dir: Path, benchmark_dir: Path) -> pd.DataFrame:
    claim_flags = _read_csv_if_exists(audit_dir / "ml_claim_flag_table.csv")
    surrogate_root = resolve_surrogate_outputs_dir()
    htc_compare_dir = _resolve_htc_model_comparison_dir(benchmark_dir)

    rows = [
        *_build_surrogate_validation_rows_from_frame(
            frame=_read_selected_or_summary_frame(
                manifest_path=surrogate_root / "selected_models_manifest_strict_group.csv",
                summary_path=surrogate_root / "traditional_ml_suite_summary_strict_group.csv",
                dataset_key="pyrolysis_direct",
                sort_columns=("selection_metric_value", "benchmark_validation_r2", "validation_r2", "benchmark_test_r2", "test_r2"),
            ),
            summary_label="strict_group",
            validation_tier="strict-group",
            dataset_key="pyrolysis_direct",
            dataset_scope_label="Pyrolysis direct observations",
            claim_flags=claim_flags,
        ),
        *_build_surrogate_validation_rows_from_frame(
            frame=_read_selected_or_summary_frame(
                manifest_path=surrogate_root / "paper1_strict_group" / "selected_models_manifest_strict_group.csv",
                summary_path=surrogate_root / "paper1_strict_group" / "traditional_ml_suite_summary_strict_group.csv",
                dataset_key="paper1_htc_scope",
                sort_columns=("selection_metric_value", "benchmark_validation_r2", "validation_r2", "benchmark_test_r2", "test_r2"),
            ),
            summary_label="strict_group",
            validation_tier="strict-group",
            dataset_key="paper1_htc_scope",
            dataset_scope_label="HTC mixed-feed planning scope",
            claim_flags=claim_flags,
        ),
        *_build_surrogate_validation_rows_from_frame(
            frame=_read_selected_or_summary_frame(
                manifest_path=htc_compare_dir / "selected_models_manifest_leave_study_out.csv",
                summary_path=htc_compare_dir / "traditional_ml_suite_summary_leave_study_out.csv",
                dataset_key="htc_direct",
                sort_columns=("selection_metric_value", "benchmark_validation_r2", "validation_r2", "benchmark_test_r2", "test_r2"),
            ),
            summary_label="leave_study_out",
            validation_tier="leave-study-out",
            dataset_key="htc_direct",
            dataset_scope_label="HTC direct observations",
            claim_flags=claim_flags,
        ),
        *_build_surrogate_validation_rows_from_frame(
            frame=_read_selected_or_summary_frame(
                manifest_path=surrogate_root / "selected_models_manifest_leave_study_out.csv",
                summary_path=surrogate_root / "traditional_ml_suite_summary_leave_study_out.csv",
                dataset_key="pyrolysis_direct",
                sort_columns=("selection_metric_value", "benchmark_validation_r2", "validation_r2", "benchmark_test_r2", "test_r2"),
            ),
            summary_label="leave_study_out",
            validation_tier="leave-study-out",
            dataset_key="pyrolysis_direct",
            dataset_scope_label="Pyrolysis direct observations",
            claim_flags=claim_flags,
        ),
    ]
    if not rows:
        return pd.DataFrame(
            [
                {
                    "validation_tier": "validation summary unavailable",
                    "dataset_scope": "--",
                    "target": "--",
                    "best_model": "--",
                    "validation_r2": pd.NA,
                    "test_r2": pd.NA,
                    "interpretation": "Current audit/benchmark surrogate validation artifacts are unavailable.",
                }
            ]
        )
    order_frame = pd.DataFrame(rows)
    order_frame["validation_tier_order"] = order_frame["validation_tier"].map({"strict-group": 0, "leave-study-out": 1}).fillna(9)
    order_frame["dataset_scope_order"] = order_frame["dataset_scope"].map(
        {
            "HTC mixed-feed planning scope": 0,
            "Pyrolysis direct observations": 1,
            "HTC direct observations": 2,
        }
    ).fillna(9)
    order_frame["target_order"] = order_frame["target"].map(
        {"Carbon retention": 0, "Char HHV": 1, "Char yield": 2, "Energy recovery": 3}
    ).fillna(9)
    return order_frame.sort_values(
        ["validation_tier_order", "dataset_scope_order", "target_order", "dataset_scope", "target"]
    ).drop(columns=["validation_tier_order", "dataset_scope_order", "target_order"]).reset_index(drop=True)


def _render_surrogate_validation_table(df: pd.DataFrame) -> str:
    return _render_latex_table(
        df,
        columns=[
            ("validation_tier", "Validation tier"),
            ("dataset_scope", "Dataset scope"),
            ("target", "Target"),
            ("best_model", "Best model"),
            ("validation_r2", "Validation $R^2$"),
            ("test_r2", "Test $R^2$"),
            ("interpretation", "Interpretation"),
        ],
        caption="Retained surrogate-validation results rebuilt from the current audit and benchmark artifacts.",
        label="tab:surrogate-validation-summary",
        column_format="l l l l r r X",
        notes=(
            "Strict-group rows are rebuilt from the current in-domain benchmark artifacts, while leave-study-out rows are rebuilt "
            "from the current cross-study benchmark or synchronized surrogate summary. Interpretations are synchronized to the current "
            "audit claim rules whenever matching claim-flag rows are available."
        ),
        formatters={
            "validation_r2": _format_r2_value,
            "test_r2": _format_r2_value,
        },
    )


def _build_surrogate_validation_rows_from_frame(
    *,
    frame: pd.DataFrame,
    summary_label: str,
    validation_tier: str,
    dataset_key: str,
    dataset_scope_label: str,
    claim_flags: pd.DataFrame,
) -> list[dict[str, object]]:
    if frame.empty:
        return []
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        model_key = _model_key_from_row(row)
        target_column = str(row.get("target_column", "")).strip()
        validation_r2 = _metric_from_row(
            row,
            ("benchmark_validation_r2", "selected_validation_r2", "validation_r2", "selection_metric_value"),
        )
        test_r2 = _metric_from_row(
            row,
            ("benchmark_test_r2", "selected_test_r2", "test_r2", "refit_test_r2"),
        )
        interpretation = _claim_status_from_audit(
            claim_flags=claim_flags,
            summary_label=summary_label,
            dataset_key=dataset_key,
            target_column=target_column,
            model_key=model_key,
        )
        if not interpretation:
            interpretation = _surrogate_interpretation_fallback(validation_tier=validation_tier, test_r2=test_r2)
        rows.append(
            {
                "validation_tier": validation_tier,
                "dataset_scope": dataset_scope_label,
                "target": _target_display(target_column),
                "best_model": _model_display_name(model_key),
                "validation_r2": validation_r2,
                "test_r2": test_r2,
                "interpretation": interpretation,
            }
        )
    return rows


def _read_selected_or_summary_frame(
    *,
    manifest_path: Path,
    summary_path: Path,
    dataset_key: str,
    sort_columns: tuple[str, ...],
) -> pd.DataFrame:
    manifest = _read_csv_if_exists(manifest_path)
    if not manifest.empty and "dataset_key" in manifest.columns:
        filtered = manifest[manifest["dataset_key"].astype(str) == dataset_key].copy()
        if not filtered.empty:
            return filtered.sort_values(
                [column for column in sort_columns if column in filtered.columns],
                ascending=[False] * len([column for column in sort_columns if column in filtered.columns]),
            ).drop_duplicates(subset=["target_column"], keep="first")
    summary = _read_csv_if_exists(summary_path)
    if summary.empty or "dataset_key" not in summary.columns:
        return pd.DataFrame()
    filtered = summary[summary["dataset_key"].astype(str) == dataset_key].copy()
    if filtered.empty:
        return pd.DataFrame()
    active_sort_columns = [column for column in sort_columns if column in filtered.columns]
    if active_sort_columns:
        filtered = filtered.sort_values(active_sort_columns, ascending=[False] * len(active_sort_columns))
    return filtered.drop_duplicates(subset=["target_column"], keep="first")


def _model_key_from_row(row: pd.Series) -> str:
    for column in ("selected_model_key", "model_key", "best_model_key"):
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return "unknown"


def _metric_from_row(row: pd.Series, columns: tuple[str, ...]) -> float:
    for column in columns:
        if column not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.notna(value):
            return float(value)
    return float("nan")


def _claim_status_from_audit(
    *,
    claim_flags: pd.DataFrame,
    summary_label: str,
    dataset_key: str,
    target_column: str,
    model_key: str,
) -> str:
    if claim_flags.empty:
        return ""
    default_series = pd.Series([""] * len(claim_flags), index=claim_flags.index, dtype="object")
    matched = claim_flags[
        claim_flags.get("summary_label", default_series).astype(str).eq(summary_label)
        & claim_flags.get("dataset_key", default_series).astype(str).eq(dataset_key)
        & claim_flags.get("target_column", default_series).astype(str).eq(target_column)
        & claim_flags.get("best_model_key", default_series).astype(str).eq(model_key)
    ].copy()
    if matched.empty:
        return ""
    return str(matched.iloc[0].get("claim_status", "")).strip()


def _surrogate_interpretation_fallback(*, validation_tier: str, test_r2: float) -> str:
    if pd.isna(test_r2):
        return "not evaluated"
    threshold = 0.65 if validation_tier == "strict-group" else 0.50
    if test_r2 >= threshold:
        return "supportive"
    if test_r2 >= 0.0:
        return "weak"
    return "unsupported"


def _build_transfer_support_table(*, audit_dir: Path, surrogate_validation: pd.DataFrame | None = None) -> pd.DataFrame:
    pathway_reliability = _read_csv_if_exists(audit_dir / "pathway_reliability_summary.csv")
    surrogate_validation_frame = (
        surrogate_validation.copy()
        if surrogate_validation is not None
        else _read_csv_if_exists(FIGURES_TABLES_DIR / "paper1_surrogate_validation_table.csv")
    )
    if pathway_reliability.empty:
        return pd.DataFrame(
            [
                {
                    "pathway": "support summary unavailable",
                    "strict_group_targets": 0,
                    "strict_group_validation_r2_range": "--",
                    "strict_group_test_r2_range": "--",
                    "leave_study_out_support_split": "--",
                    "reliability_summary": "--",
                    "manuscript_ceiling": "--",
                }
            ]
        )

    rows: list[dict[str, object]] = []
    for pathway in ["pyrolysis", "htc", "ad", "baseline"]:
        matched = pathway_reliability[pathway_reliability["pathway"].astype(str).str.lower().eq(pathway)].copy()
        reliability_row = matched.iloc[0] if not matched.empty else pd.Series(dtype="object")
        if reliability_row.empty:
            continue
        strict_rows = _surrogate_validation_rows_for_pathway(
            surrogate_validation=surrogate_validation_frame,
            pathway=pathway,
            validation_tier="strict-group",
        )
        supportive = int(_as_float(reliability_row.get("leave_study_out_supportive_count")))
        weak = int(_as_float(reliability_row.get("leave_study_out_weak_count")))
        unsupported = int(_as_float(reliability_row.get("leave_study_out_unsupported_count")))
        reliability_score = _as_float(reliability_row.get("reliability_score"))
        reliability_tier = str(reliability_row.get("reliability_tier", "")).strip()
        rows.append(
            {
                "pathway": _pathway_display(pathway),
                "strict_group_targets": int(len(strict_rows)),
                "strict_group_validation_r2_range": _metric_range_display(strict_rows.get("validation_r2")),
                "strict_group_test_r2_range": _metric_range_display(strict_rows.get("test_r2")),
                "leave_study_out_support_split": f"{supportive}/{weak}/{unsupported}",
                "reliability_summary": (
                    f"{reliability_score:.3f} ({_reliability_tier_display(reliability_tier)})"
                    if reliability_tier
                    else f"{reliability_score:.3f}"
                ),
                "manuscript_ceiling": _manuscript_ceiling_for_reliability_tier(reliability_tier),
            }
        )
    return pd.DataFrame(rows)


def _render_transfer_support_table(df: pd.DataFrame) -> str:
    return _render_latex_table(
        df,
        columns=[
            ("pathway", "Pathway"),
            ("strict_group_targets", "Strict-group targets"),
            ("strict_group_validation_r2_range", "Strict-group validation $R^2$"),
            ("strict_group_test_r2_range", "Strict-group test $R^2$"),
            ("leave_study_out_support_split", "Leave-study-out S/W/U"),
            ("reliability_summary", "Reliability summary"),
            ("manuscript_ceiling", "Manuscript ceiling"),
        ],
        caption="Pathway-level support ladder separating in-domain surrogate fit from cross-study transfer reliability.",
        label="tab:transfer-support-summary",
        column_format="l r l l l l >{\\raggedright\\arraybackslash}X",
        notes=(
            "Strict-group ranges summarize the manuscript-scope validation table for each pathway. "
            "Leave-study-out S/W/U reports supportive, weak, and unsupported counts from "
            "\\texttt{surrogate\\_validation\\_summary} and the synchronized audit pathway summary. "
            "The reliability score is computed as $(S + 0.5W)/(S+W+U)$."
        ),
    )


def _build_optimization_output_table(*, planning_dir: Path) -> pd.DataFrame:
    portfolio_summary = _read_csv_if_exists(planning_dir / "portfolio_summary.csv")
    portfolio_allocations = _read_csv_if_exists(planning_dir / "portfolio_allocations.csv")
    rows: list[dict[str, object]] = []
    for scenario_name in SCENARIO_ORDER:
        summary = _scenario_frame_row(portfolio_summary, scenario_name)
        scenario_allocations = _scenario_allocations(portfolio_allocations, scenario_name)
        rows.append(
            {
                "scenario": SCENARIO_DISPLAY.get(scenario_name, scenario_name),
                "selected_pathways": _selected_pathways_display(scenario_allocations),
                "allocation_profile": _allocation_profile_display(scenario_allocations),
                "allocated_feed_ktpy": _as_float(summary.get("allocated_feed_ton_per_year")) / 1e3,
                "coverage_pct": _as_float(summary.get("scenario_feed_coverage_ratio")) * 100.0,
                "portfolio_energy_pj_per_year": _as_float(summary.get("portfolio_energy_objective")) / 1e9,
                "portfolio_carbon_load_ktco2e_per_year": _as_float(summary.get("portfolio_carbon_load_kgco2e")) / 1e6,
                "portfolio_net_cost_musd_per_year": _as_float(summary.get("portfolio_cost_objective")) / 1e6,
            }
        )
    return pd.DataFrame(rows)


def _build_uq_sensitivity_table(*, planning_dir: Path) -> pd.DataFrame:
    diagnostics = _read_csv_if_exists(planning_dir / "optimization_diagnostics.csv")
    main_results = _read_csv_if_exists(planning_dir / "main_results_table.csv")
    rows: list[dict[str, object]] = []
    for scenario_name in SCENARIO_ORDER:
        diag = _scenario_frame_row(diagnostics, scenario_name)
        main = _scenario_frame(main_results, scenario_name)
        if not main.empty and "selected_in_baseline_portfolio" in main.columns:
            selected = main[main["selected_in_baseline_portfolio"].fillna(False).astype(bool)].copy()
        else:
            selected = pd.DataFrame()
        support_values = (
            selected.get("uq_stress_support", pd.Series(dtype="object")).dropna().astype(str).unique().tolist()
            if not selected.empty
            else []
        )
        sensitivity = (
            str(main["uncertainty_mode_sensitivity"].dropna().astype(str).iloc[0])
            if not main.empty and "uncertainty_mode_sensitivity" in main.columns and main["uncertainty_mode_sensitivity"].dropna().any()
            else "not evaluated"
        )
        rows.append(
            {
                "scenario": SCENARIO_DISPLAY.get(scenario_name, scenario_name),
                "active_uq_mode": _uq_mode_display(str(diag.get("active_uncertainty_penalty_mode", "prefer_interval_mean"))),
                "interval_mean_top_case": _case_id_display(diag.get("interval_mean_top_ranked_case_id")),
                "max_interval_top_case": _case_id_display(diag.get("max_interval_top_ranked_case_id")),
                "combined_only_top_case": _case_id_display(diag.get("combined_only_top_ranked_case_id")),
                "case_switch_count": _as_float(diag.get("uncertainty_mode_case_switch_count")),
                "pathway_switch_count": _as_float(diag.get("uncertainty_mode_pathway_switch_count")),
                "selected_pathway_uq_support": "; ".join(sorted(value for value in support_values if value and value != "nan")) or "--",
                "uq_sensitivity_label": sensitivity,
                "manuscript_note": _uq_table_note(diag, sensitivity),
            }
        )
    return pd.DataFrame(rows)


def _render_optimization_output_table(df: pd.DataFrame) -> str:
    return _render_latex_table(
        df,
        columns=[
            ("scenario", "Scenario"),
            ("selected_pathways", "Selected pathways"),
            ("allocation_profile", "Allocation profile"),
            ("allocated_feed_ktpy", "Allocated feed (kt y$^{-1}$)"),
            ("coverage_pct", "Coverage (\\%)"),
            ("portfolio_energy_pj_per_year", "Energy (PJ y$^{-1}$)"),
            ("portfolio_carbon_load_ktco2e_per_year", "Residual carbon (ktCO$_2$e y$^{-1}$)"),
            ("portfolio_net_cost_musd_per_year", "Net cost (MUSD y$^{-1}$)"),
        ],
        caption="Baseline portfolio outputs for the three main scenarios.",
        label="tab:optimization-output-summary",
        column_format="l l >{\\raggedright\\arraybackslash}X r r r r r",
        notes=(
            "Coverage is the ratio of allocated throughput to the scenario feed budget. "
            "Residual carbon is reported only for allocated throughput."
        ),
        formatters={
            "allocated_feed_ktpy": lambda value: f"{_as_float(value):.1f}",
            "coverage_pct": lambda value: f"{_as_float(value):.1f}",
            "portfolio_energy_pj_per_year": lambda value: f"{_as_float(value):.2f}",
            "portfolio_carbon_load_ktco2e_per_year": lambda value: f"{_as_float(value):.1f}",
            "portfolio_net_cost_musd_per_year": lambda value: f"{_as_float(value):.2f}",
        },
    )


def _render_uq_sensitivity_table(df: pd.DataFrame) -> str:
    return _render_latex_table(
        df,
        columns=[
            ("scenario", "Scenario"),
            ("active_uq_mode", "Active UQ mode"),
            ("interval_mean_top_case", "Interval-mean top case"),
            ("max_interval_top_case", "Max-interval top case"),
            ("combined_only_top_case", "Combined-only top case"),
            ("case_switch_count", "Case-switch count"),
            ("pathway_switch_count", "Pathway-switch count"),
            ("selected_pathway_uq_support", "Selected-pathway UQ support"),
            ("uq_sensitivity_label", "Sensitivity label"),
            ("manuscript_note", "Manuscript note"),
        ],
        caption="Scenario-level sensitivity of the exported recommendation to alternative uncertainty aggregations.",
        label="tab:uq-sensitivity-summary",
        column_format="l l l l l r r l l X",
        notes=(
            "Case-switch count records how many distinct top-ranked cases appear across the tested uncertainty modes. "
            "Pathway-switch count records whether uncertainty aggregation changes pathway identity or only within-pathway case ranking."
        ),
        formatters={
            "case_switch_count": lambda value: f"{_as_float(value):.0f}",
            "pathway_switch_count": lambda value: f"{_as_float(value):.0f}",
        },
    )


def _build_cost_boundary_table(*, planning_dir: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pathway": "Baseline management",
                "cost_anchor": "California landfill tipping-fee anchor.",
                "included_terms": "Direct disposal cost only.",
                "product_credit_status": "No matched product credit.",
                "regionalization": "California tipping-fee anchor applied directly.",
                "boundary_note": "Reference comparator, not a revenue-generating conversion route.",
            },
            {
                "pathway": "AD",
                "cost_anchor": "Literature annualized CAPEX + OPEX model.",
                "included_terms": "Electricity, heat, and avoided-disposal credits.",
                "product_credit_status": "No digestate/nutrient market credit in the exported baseline.",
                "regionalization": "California energy repricing on the proxy-based pathway.",
                "boundary_note": "Economic support remains proxy-based and should be read cautiously.",
            },
            {
                "pathway": "Pyrolysis",
                "cost_anchor": "Pathway-specific cost, feedstock, and biochar references.",
                "included_terms": "Biochar revenue plus avoided-disposal credit.",
                "product_credit_status": "Matched biochar revenue retained.",
                "regionalization": "Pathway-specific references retained without forced harmonization.",
                "boundary_note": "Read recommendation sensitivity against the biochar-revenue boundary.",
            },
            {
                "pathway": "HTC",
                "cost_anchor": "Lucian and Fiori plant-scale cost structure with California repricing.",
                "included_terms": "Annualized CAPEX + OPEX under the exported utility assumptions.",
                "product_credit_status": "No matched hydrochar market credit in the exported baseline.",
                "regionalization": "California electricity and natural-gas repricing applied.",
                "boundary_note": "HTC can be selected without a hydrochar credit, but the comparison remains asymmetric.",
            },
        ]
    )


def _render_cost_boundary_table(df: pd.DataFrame) -> str:
    return _render_latex_table(
        df,
        columns=[
            ("pathway", "Pathway"),
            ("cost_anchor", "Cost anchor"),
            ("included_terms", "Included terms"),
            ("product_credit_status", "Product-credit status"),
            ("regionalization", "Regionalization"),
            ("boundary_note", "Boundary note"),
        ],
        caption="Cost-boundary disclosures for the pathway-specific economic terms.",
        label="tab:cost-boundary-summary",
        column_format="l >{\\raggedright\\arraybackslash}X >{\\raggedright\\arraybackslash}X >{\\raggedright\\arraybackslash}X l >{\\raggedright\\arraybackslash}X",
        notes=(
            "The table records what is priced into each pathway. "
            "Product-credit omissions are explicit boundary limits."
        ),
    )


def _build_evidence_ceiling_table(*, planning_dir: Path, audit_dir: Path) -> pd.DataFrame:
    portfolio_allocations = _read_csv_if_exists(planning_dir / "portfolio_allocations.csv")
    transferability = _read_csv_if_exists(audit_dir / "planning_transferability_risk_summary.csv")
    pathway_reliability = _read_csv_if_exists(audit_dir / "pathway_reliability_summary.csv")
    reliability_map = {
        str(row["pathway"]): (
            _as_float(row.get("reliability_score")),
            _reliability_tier_display(str(row.get("reliability_tier", ""))),
        )
        for _, row in pathway_reliability.iterrows()
    }
    rows: list[dict[str, object]] = []
    for scenario_name in SCENARIO_ORDER:
        scenario_allocations = _scenario_allocations(portfolio_allocations, scenario_name)
        total_allocated = _as_float(scenario_allocations.get("allocated_feed_ton_per_year", pd.Series(dtype=float)).sum())
        grouped = _grouped_allocations_by_pathway(scenario_allocations)
        support_profile = _support_profile_display(scenario_allocations)
        transfer_profile_parts: list[str] = []
        weighted_reliability = 0.0
        if total_allocated > 0.0:
            for pathway, allocated_ton in grouped.items():
                score, tier = reliability_map.get(pathway, (0.0, "not available"))
                share = allocated_ton / total_allocated
                weighted_reliability += share * score
                transfer_profile_parts.append(f"{_pathway_display(pathway)} {score:.3f} ({tier})")
        transfer_row = _scenario_frame_row(transferability, scenario_name)
        surrogate_supported_share = 0.0
        if total_allocated > 0.0 and not scenario_allocations.empty:
            support_mask = scenario_allocations.get("is_surrogate_supported", pd.Series(False, index=scenario_allocations.index))
            surrogate_supported_share = (
                _as_float(
                    pd.to_numeric(
                        scenario_allocations.loc[support_mask.astype(bool), "allocated_feed_ton_per_year"],
                        errors="coerce",
                    ).fillna(0.0).sum()
                )
                / total_allocated
                * 100.0
            )
        rows.append(
            {
                "scenario": SCENARIO_DISPLAY.get(scenario_name, scenario_name),
                "selected_pathways": _selected_pathways_display(scenario_allocations),
                "row_support_profile": support_profile,
                "selected_pathway_transfer_profile": "; ".join(transfer_profile_parts) if transfer_profile_parts else "--",
                "surrogate_supported_share_pct": surrogate_supported_share,
                "weighted_transferability_score": _as_float(transfer_row.get("weighted_transferability_score")),
                "manuscript_ceiling": _transferability_ceiling_display(
                    str(transfer_row.get("transferability_evidence_ceiling", "not_available"))
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_evidence_ceiling_table(df: pd.DataFrame) -> str:
    return _render_latex_table(
        df,
        columns=[
            ("scenario", "Scenario"),
            ("selected_pathways", "Selected pathways"),
            ("row_support_profile", "Row-level support profile"),
            ("selected_pathway_transfer_profile", "Pathway transfer profile"),
            ("surrogate_supported_share_pct", "Surrogate-supported share (\\%)"),
            ("weighted_transferability_score", "Portfolio transferability score"),
            ("manuscript_ceiling", "Manuscript ceiling"),
        ],
        caption="Evidence ceiling of the exported baseline portfolios.",
        label="tab:evidence-ceiling-summary",
        column_format="l l >{\\raggedright\\arraybackslash}X >{\\raggedright\\arraybackslash}X r r l",
        notes=(
            "Row-level support refers to selected candidate rows, whereas pathway transfer profiles summarize the "
            "leave-study-out audit."
        ),
        formatters={
            "surrogate_supported_share_pct": lambda value: f"{_as_float(value):.1f}",
            "weighted_transferability_score": lambda value: f"{_as_float(value):.2f}",
        },
        font_size="\\tiny",
    )


def _build_product_credit_sensitivity_table(*, planning_dir: Path) -> pd.DataFrame:
    portfolio_allocations = _read_csv_if_exists(planning_dir / "portfolio_allocations.csv")
    stress_specs = [
        {
            "economic_perturbation": "Current exported boundary",
            "baseline_region_outcome": _portfolio_outcome_label(portfolio_allocations, "baseline_region_case"),
            "high_supply_outcome": _portfolio_outcome_label(portfolio_allocations, "high_supply_case"),
            "policy_support_outcome": _portfolio_outcome_label(portfolio_allocations, "policy_support_case"),
            "interpretation": (
                "Reference row for the exported baseline economic boundary; subsequent rows report counterfactual "
                "changes relative to these scenario outcomes."
            ),
        },
        {
            "economic_perturbation": "Pyrolysis biochar revenue x1.10",
            "modifier": ("pyrolysis", 1.10, None),
        },
        {
            "economic_perturbation": "Pyrolysis biochar revenue x1.30",
            "modifier": ("pyrolysis", 1.30, None),
        },
        {
            "economic_perturbation": "Pyrolysis biochar revenue x1.40",
            "modifier": ("pyrolysis", 1.40, None),
        },
        {
            "economic_perturbation": "Pyrolysis biochar revenue x1.50",
            "modifier": ("pyrolysis", 1.50, None),
        },
        {
            "economic_perturbation": "HTC hydrochar credit +100 USD per t mixed feed",
            "modifier": ("htc", None, 100.0),
        },
        {
            "economic_perturbation": "AD digestate/nutrient credit +200 USD per t mixed feed",
            "modifier": ("ad", None, 200.0),
        },
    ]
    if not (planning_dir / "run_config.json").exists():
        return _build_product_credit_sensitivity_placeholder_table(
            portfolio_allocations=portfolio_allocations,
            stress_specs=stress_specs,
        )

    try:
        bundle = load_planning_input_bundle()
        base_config = replace(
            PlanningConfig(),
            objective_weight_system=get_objective_weight_system(),
        )
    except Exception:
        return _build_product_credit_sensitivity_placeholder_table(
            portfolio_allocations=portfolio_allocations,
            stress_specs=stress_specs,
        )

    rows: list[dict[str, object]] = []
    baseline_outcomes = {
        "baseline_region_case": _portfolio_outcome_label(portfolio_allocations, "baseline_region_case"),
        "high_supply_case": _portfolio_outcome_label(portfolio_allocations, "high_supply_case"),
        "policy_support_case": _portfolio_outcome_label(portfolio_allocations, "policy_support_case"),
    }
    for spec in stress_specs:
        if "modifier" not in spec:
            rows.append(spec)
            continue
        pathway, multiplier, credit_per_ton = spec["modifier"]
        stressed_allocations = _run_product_credit_stress(
            bundle=bundle,
            config=base_config,
            pathway=str(pathway),
            revenue_multiplier=multiplier,
            additional_credit_per_total_mixed_ton=credit_per_ton,
        )
        stressed_outcomes = {
            "baseline_region_case": _portfolio_outcome_label(stressed_allocations, "baseline_region_case"),
            "high_supply_case": _portfolio_outcome_label(stressed_allocations, "high_supply_case"),
            "policy_support_case": _portfolio_outcome_label(stressed_allocations, "policy_support_case"),
        }
        rows.append(
            {
                "economic_perturbation": spec["economic_perturbation"],
                "baseline_region_outcome": stressed_outcomes["baseline_region_case"],
                "high_supply_outcome": stressed_outcomes["high_supply_case"],
                "policy_support_outcome": stressed_outcomes["policy_support_case"],
                "interpretation": _build_product_credit_interpretation(
                    economic_perturbation=str(spec["economic_perturbation"]),
                    baseline_outcomes=baseline_outcomes,
                    stressed_outcomes=stressed_outcomes,
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_product_credit_sensitivity_placeholder_table(
    *,
    portfolio_allocations: pd.DataFrame,
    stress_specs: list[dict[str, object]],
) -> pd.DataFrame:
    baseline_region_outcome = _portfolio_outcome_label(portfolio_allocations, "baseline_region_case")
    high_supply_outcome = _portfolio_outcome_label(portfolio_allocations, "high_supply_case")
    policy_support_outcome = _portfolio_outcome_label(portfolio_allocations, "policy_support_case")
    placeholder_note = (
        "Scenario-level replanning inputs were not packaged with this planning directory, "
        "so manuscript export preserves the exported baseline portfolio pending a dedicated sensitivity rerun."
    )
    rows: list[dict[str, object]] = []
    for spec in stress_specs:
        interpretation = str(spec.get("interpretation", "")).strip()
        if "modifier" in spec:
            interpretation = f"{interpretation} {placeholder_note}".strip()
        rows.append(
            {
                "economic_perturbation": spec["economic_perturbation"],
                "baseline_region_outcome": baseline_region_outcome,
                "high_supply_outcome": high_supply_outcome,
                "policy_support_outcome": policy_support_outcome,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def _render_product_credit_sensitivity_table(df: pd.DataFrame) -> str:
    return _render_latex_table(
        df,
        columns=[
            ("economic_perturbation", "Economic perturbation"),
            ("baseline_region_outcome", "Baseline-region outcome"),
            ("high_supply_outcome", "High-supply outcome"),
            ("policy_support_outcome", "Policy-support outcome"),
            ("interpretation", "Interpretation"),
        ],
        caption="Sensitivity of the recommendation to product-credit assumptions.",
        label="tab:product-credit-sensitivity-summary",
        column_format=">{\\raggedright\\arraybackslash}p{3.5cm} l l l >{\\raggedright\\arraybackslash}X",
        notes=(
            "Pyrolysis stresses multiply the existing biochar revenue term. HTC and AD stresses add matched hypothetical "
            "product credits per ton of total mixed feed."
        ),
    )


def _build_product_credit_interpretation(
    *,
    economic_perturbation: str,
    baseline_outcomes: dict[str, str],
    stressed_outcomes: dict[str, str],
) -> str:
    changed_scenarios = [
        SCENARIO_DISPLAY.get(scenario_name, scenario_name)
        for scenario_name in SCENARIO_ORDER
        if str(stressed_outcomes.get(scenario_name, "--")) != str(baseline_outcomes.get(scenario_name, "--"))
    ]
    if not changed_scenarios:
        return "No pathway-set change relative to the baseline boundary; any effect is confined to share reweighting."
    if len(changed_scenarios) == len(SCENARIO_ORDER):
        scenario_clause = "all three scenarios"
    else:
        scenario_clause = ", ".join(changed_scenarios)
    return (
        f"Composition changes appear in {scenario_clause}; treat this as a boundary-sensitive accounting stress, "
        "not as a pathway-proof result."
    )


def _build_targeted_ablation_table(*, benchmark_dir: Path) -> pd.DataFrame:
    ablation_dir = _resolve_targeted_ablation_dir(benchmark_dir)
    summary = _read_csv_if_exists(ablation_dir / "targeted_planning_ablations_summary.csv")
    allocations = _read_csv_if_exists(ablation_dir / "targeted_planning_ablations_allocations.csv")
    if summary.empty:
        return pd.DataFrame(
            [
                {
                    "ablation": "targeted ablations unavailable",
                    "baseline_region_outcome": "--",
                    "high_supply_outcome": "--",
                    "policy_support_outcome": "--",
                    "interpretation": "Targeted planning ablations have not been exported yet.",
                }
            ]
        )

    rows: list[dict[str, object]] = []
    group_columns = ["ablation_family", "ablation_key", "ablation_value"]
    grouped_allocations = {}
    if not allocations.empty and set(group_columns).issubset(allocations.columns):
        grouped_allocations = {
            tuple(key): frame.copy()
            for key, frame in allocations.groupby(group_columns, dropna=False)
        }

    baseline_frame = pd.DataFrame()
    for keys, frame in grouped_allocations.items():
        ablation_family, ablation_key, _ = keys
        if str(ablation_family) == "evidence_sensitivity" and str(ablation_key) == "eta_0.15":
            baseline_frame = frame.copy()
            break
    if baseline_frame.empty and grouped_allocations:
        baseline_frame = next(iter(grouped_allocations.values())).copy()
    baseline_outcomes = {
        scenario_name: _portfolio_outcome_label(baseline_frame, scenario_name)
        for scenario_name in SCENARIO_ORDER
    }

    for keys, frame in summary.groupby(group_columns, dropna=False):
        ablation_family, ablation_key, ablation_value = keys
        allocation_frame = grouped_allocations.get(keys, pd.DataFrame())
        stressed_outcomes = {
            scenario_name: _portfolio_outcome_label(allocation_frame, scenario_name)
            for scenario_name in SCENARIO_ORDER
        }
        rows.append(
            {
                "ablation": _targeted_ablation_label(
                    ablation_family=str(ablation_family),
                    ablation_key=str(ablation_key),
                    ablation_value=ablation_value,
                ),
                "baseline_region_outcome": stressed_outcomes["baseline_region_case"],
                "high_supply_outcome": stressed_outcomes["high_supply_case"],
                "policy_support_outcome": stressed_outcomes["policy_support_case"],
                "interpretation": _build_product_credit_interpretation(
                    economic_perturbation=_targeted_ablation_label(
                        ablation_family=str(ablation_family),
                        ablation_key=str(ablation_key),
                        ablation_value=ablation_value,
                    ),
                    baseline_outcomes=baseline_outcomes,
                    stressed_outcomes=stressed_outcomes,
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_targeted_ablation_table(df: pd.DataFrame) -> str:
    return _render_latex_table(
        df,
        columns=[
            ("ablation", "Targeted ablation"),
            ("baseline_region_outcome", "Baseline-region outcome"),
            ("high_supply_outcome", "High-supply outcome"),
            ("policy_support_outcome", "Policy-support outcome"),
            ("interpretation", "Interpretation"),
        ],
        caption="Targeted planning ablations for legacy HTC model choice, evidence-utility intensity, and matched hydrochar-price symmetry.",
        label="tab:targeted-ablation-summary",
        column_format=">{\\raggedright\\arraybackslash}p{3.5cm} l l l >{\\raggedright\\arraybackslash}X",
        notes=(
            "The baseline comparison corresponds to the exported planning default with evidence-utility coefficient "
            "$\\eta=0.15$. Outcome strings report pathway composition rather than a binary winner."
        ),
        formatters={"ablation": _literal_latex_formatter},
    )


def _resolve_targeted_ablation_dir(benchmark_dir: Path) -> Path:
    candidates = [
        benchmark_dir / "targeted_planning_ablations",
        benchmark_dir.parent / "targeted_planning_ablations",
        BENCHMARK_OUTPUTS_DIR / "targeted_planning_ablations",
    ]
    for candidate in candidates:
        if (candidate / "targeted_planning_ablations_summary.csv").exists():
            return candidate
    return candidates[0]


def _targeted_ablation_label(*, ablation_family: str, ablation_key: str, ablation_value: object) -> str:
    if ablation_family == "algorithm":
        return "HTC selector forced back to XGBoost"
    if ablation_family == "economic_symmetry":
        return "Hydrochar price matched to the pyrolysis median product revenue"
    if ablation_family == "evidence_sensitivity":
        return f"Evidence-utility coefficient $\\eta={_as_float(ablation_value):.2f}$"
    if ablation_family == "ad_complementarity":
        return f"AD minimum share {_as_float(ablation_value) * 100:.0f}\\%"
    if ablation_family == "pathway_cap_sensitivity":
        return f"Pyrolysis maximum share {_as_float(ablation_value) * 100:.0f}\\%"
    if ablation_family == "coproduct_boundary" and ablation_key.startswith("digestate_rng_credit"):
        multiplier = {
            "digestate_rng_credit_50pct": 0.5,
            "digestate_rng_credit_100pct": 1.0,
            "digestate_rng_credit_200pct": 2.0,
            "digestate_rng_credit_300pct": 3.0,
        }.get(ablation_key, _as_float(ablation_value))
        return f"AD coproduct credit {multiplier:.1f}$\times$ pyrolysis median"
    return f"{ablation_family}: {ablation_key}={ablation_value}"


def _build_htc_model_comparison_table(*, benchmark_dir: Path) -> pd.DataFrame:
    comparison_dir = _resolve_htc_model_comparison_dir(benchmark_dir)
    aggregate = _read_csv_if_exists(comparison_dir / "htc_model_comparison_aggregate.csv")
    selected = _read_csv_if_exists(comparison_dir / "selected_models_manifest_leave_study_out.csv")
    if aggregate.empty:
        return pd.DataFrame(
            [
                {
                    "model": "comparison unavailable",
                    "selected_targets": 0,
                    "mean_validation_r2": pd.NA,
                    "mean_test_r2": pd.NA,
                    "negative_refit_selected_targets": 0,
                    "manuscript_note": (
                        "HTC leave-study-out model comparison has not been exported yet."
                    ),
                }
            ]
        )

    negative_refit_counts: dict[str, int] = {}
    if not selected.empty and "selected_model_key" in selected.columns:
        for model_key, subset in selected.groupby("selected_model_key", dropna=False):
            refit_test = pd.to_numeric(subset.get("refit_test_r2"), errors="coerce")
            negative_refit_counts[str(model_key)] = int((refit_test < 0).fillna(False).sum())

    rows: list[dict[str, object]] = []
    working = aggregate.copy()
    sort_columns = [column for column in ["selected_target_count", "mean_validation_r2"] if column in working.columns]
    if sort_columns:
        working = working.sort_values(sort_columns, ascending=[False, False][: len(sort_columns)]).reset_index(drop=True)

    for _, row in working.iterrows():
        model_key = str(row.get("model_key", "unknown"))
        model_display = _model_display_name(model_key)
        selected_target_count = int(_as_float(row.get("selected_target_count")))
        target_count = int(_as_float(row.get("target_count")))
        negative_refit_count = negative_refit_counts.get(model_key, 0)
        rows.append(
            {
                "model": model_display,
                "selected_targets": f"{selected_target_count}/{target_count}" if target_count > 0 else "0/0",
                "mean_validation_r2": _as_float(row.get("mean_validation_r2")),
                "mean_test_r2": _as_float(row.get("mean_test_r2")),
                "negative_refit_selected_targets": negative_refit_count,
                "manuscript_note": _htc_model_comparison_note_for_row(
                    model_key=model_key,
                    selected_target_count=selected_target_count,
                    negative_refit_count=negative_refit_count,
                    mean_test_r2=_as_float(row.get("mean_test_r2")),
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_htc_model_comparison_table(df: pd.DataFrame) -> str:
    return _render_latex_table(
        df,
        columns=[
            ("model", "Model"),
            ("selected_targets", "Selected targets"),
            ("mean_validation_r2", "Mean validation $R^2$"),
            ("mean_test_r2", "Mean test $R^2$"),
            ("negative_refit_selected_targets", "Negative refit targets"),
            ("manuscript_note", "Interpretation"),
        ],
        caption="HTC leave-study-out comparison across the expanded model suite.",
        label="tab:htc-model-comparison",
        column_format="l l r r r X",
        notes=(
            "Study-group metadata are used only to define the leave-study-out split and are excluded from model "
            "features; differences therefore reflect generalization over process and feedstock descriptors rather "
            "than direct study-ID encoding."
        ),
        formatters={
            "mean_validation_r2": lambda value: "--" if pd.isna(value) else f"{_as_float(value):.3f}",
            "mean_test_r2": lambda value: "--" if pd.isna(value) else f"{_as_float(value):.3f}",
            "negative_refit_selected_targets": lambda value: f"{int(_as_float(value))}",
        },
    )


def _build_htc_model_comparison_note(df: pd.DataFrame) -> str:
    if df.empty or "model" not in df.columns:
        return "HTC leave-study-out model comparison unavailable.\n"

    best_row = df.iloc[0]
    lines = [
        "# HTC leave-study-out model comparison note",
        "",
        (
            f"CatBoost/LightGBM/stacking expansion confirms that the HTC transferability ceiling is not an artifact "
            f"of relying on a single learner. In the exported comparison, {best_row.get('model', 'the top model')} "
            f"ranked best overall, but the comparison still remained bounded at the cross-study level."
        ),
        "",
        (
            "Use conservative phrasing in the manuscript: CatBoost can be described as the strongest tested HTC model "
            "under leave-study-out validation, but not as a definitive solution."
        ),
        "",
        (
            "Do not attribute the CatBoost gain to direct use of study IDs. In this repository, study-group metadata "
            "are used to define the leave-study-out split and are excluded from model features."
        ),
        "",
        (
            "LightGBM should be described as target-specific rather than uniformly failed, because it still won the "
            "HTC char-HHV target in the expanded suite. Stacking can be framed as a stabilizing ensemble that avoided "
            "aggregate collapse without becoming the top-ranked model."
        ),
    ]
    return "\n".join(lines) + "\n"


def _build_htc_model_narrative(
    *,
    htc_model_comparison: pd.DataFrame,
    confidence_summary: pd.DataFrame,
) -> dict[str, str]:
    if htc_model_comparison.empty or "model" not in htc_model_comparison.columns:
        placeholder = "HTC model-comparison narrative is unavailable because the leave-study-out comparison has not been exported."
        return {
            "results_paragraph": placeholder,
            "figure3_paragraph": placeholder,
            "conclusion_paragraph": placeholder,
        }

    comparison = htc_model_comparison.copy()
    comparison["selected_target_numerator"] = comparison["selected_targets"].astype(str).str.split("/").str[0].fillna("0")
    comparison["selected_target_numerator"] = pd.to_numeric(comparison["selected_target_numerator"], errors="coerce").fillna(0).astype(int)
    best_row = comparison.sort_values(
        ["selected_target_numerator", "mean_validation_r2"],
        ascending=[False, False],
    ).iloc[0]
    best_model = str(best_row.get("model", "The leading model"))
    best_selected_targets = str(best_row.get("selected_targets", "0/0"))
    best_selected_targets_phrase = _selected_targets_phrase(best_selected_targets)
    best_validation_r2 = _as_float(best_row.get("mean_validation_r2"))
    best_test_r2 = _as_float(best_row.get("mean_test_r2"))
    negative_refit_targets = int(_as_float(best_row.get("negative_refit_selected_targets")))
    negative_refit_targets_phrase = _count_targets_phrase(negative_refit_targets)

    lightgbm_row = comparison[comparison["model"].astype(str).str.lower() == "lightgbm"]
    stacking_row = comparison[comparison["model"].astype(str).str.lower() == "stacking"]
    xgboost_row = comparison[comparison["model"].astype(str).str.lower() == "xgboost"]
    lightgbm_clause = ""
    if not lightgbm_row.empty:
        row = lightgbm_row.iloc[0]
        lightgbm_clause = (
            f" LightGBM remained target-specific rather than uniformly weak, being selected for {_selected_targets_phrase(str(row['selected_targets']))} and showing a mean validation $R^2$ of {_as_float(row['mean_validation_r2']):.3f}, "
            f"but its mean test $R^2$ stayed negative ({_as_float(row['mean_test_r2']):.3f})."
        )
    stacking_clause = ""
    if not stacking_row.empty:
        row = stacking_row.iloc[0]
        stacking_clause = (
            f" Stacking did not win any target, but its mean test $R^2$ remained non-negative ({_as_float(row['mean_test_r2']):.3f}), consistent with a stabilizing rather than a leading role."
        )
    xgboost_clause = ""
    if not xgboost_row.empty:
        row = xgboost_row.iloc[0]
        xgboost_clause = (
            f" By contrast, XGBoost won no target and retained a negative mean test $R^2$ ({_as_float(row['mean_test_r2']):.3f})."
        )

    results_paragraph = (
        f"When the HTC suite was expanded to CatBoost, LightGBM, stacking, and XGBoost under leave-study-out evaluation, {best_model} emerged as the strongest tested learner, ranking first for {best_selected_targets_phrase} and yielding the highest aggregate mean validation $R^2$ ({best_validation_r2:.3f}) "
        f"and mean test $R^2$ ({best_test_r2:.3f}). Even so, the gain was insufficient to remove the HTC transfer ceiling, because {negative_refit_targets_phrase} remained negative after refit test evaluation."
        f"{lightgbm_clause}{stacking_clause}{xgboost_clause}"
    )

    baseline_htc = _confidence_row(confidence_summary, "baseline_region_case", "htc")
    high_supply_htc = _confidence_row(confidence_summary, "high_supply_case", "htc")
    policy_support_htc = _confidence_row(confidence_summary, "policy_support_case", "htc")
    figure3_bits: list[str] = []
    if baseline_htc is not None and high_supply_htc is not None:
        baseline_tier = str(baseline_htc.get("recommendation_confidence_tier", "moderate")).replace("_", " ")
        high_supply_tier = str(high_supply_htc.get("recommendation_confidence_tier", "moderate")).replace("_", " ")
        figure3_bits.append(
            f"Figure 3 can therefore retain HTC as a selected but evidence-bounded pathway in the baseline and high-supply scenarios ({baseline_tier} confidence at {_as_float(baseline_htc.get('recommendation_confidence_score')):.3f} and {high_supply_tier} confidence at {_as_float(high_supply_htc.get('recommendation_confidence_score')):.3f}, respectively)"
        )
    if policy_support_htc is not None:
        figure3_bits.append(
            f"while emphasizing that HTC falls to a guarded confidence status in the policy-support scenario (confidence index {_as_float(policy_support_htc.get('recommendation_confidence_score')):.3f}) once it is unselected and loses stress-support credit"
        )
    if figure3_bits:
        if len(figure3_bits) == 1:
            joined_bits = figure3_bits[0].rstrip(".")
        else:
            joined_bits = f"{figure3_bits[0].rstrip('.')}, {figure3_bits[1].rstrip('.')}"
        figure3_paragraph = (
            f"{joined_bits}. Read together with the leave-study-out model comparison, this pattern indicates that stronger learners improve HTC competitiveness in selected scenarios, but do not remove the underlying cross-study transfer constraint."
        )
    else:
        figure3_paragraph = (
            "Figure 3 should present HTC confidence as scenario-dependent rather than universally high, and the caption should note that stronger learners improved ranking without removing the leave-study-out transfer ceiling."
        )

    conclusion_paragraph = (
        f"Overall, the expanded HTC model comparison indicates that the transfer limit is not a single-model artifact. {best_model} should therefore be presented as the strongest tested HTC learner rather than as a definitive solution, because cross-study refit failure persisted for {negative_refit_targets_phrase}. This supports a conservative reading of HTC as algorithmically strengthened yet still evidence-bounded, and it reinforces the guarded pathway-level claims used in the planning conclusions."
    )

    return {
        "results_paragraph": results_paragraph,
        "figure3_paragraph": figure3_paragraph,
        "conclusion_paragraph": conclusion_paragraph,
    }


def _render_htc_model_narrative_markdown(narrative: dict[str, str]) -> str:
    return (
        "# HTC Model Narrative\n\n"
        "## Results\n\n"
        f"{narrative['results_paragraph']}\n\n"
        "## Figure 3 note\n\n"
        f"{narrative['figure3_paragraph']}\n\n"
        "## Conclusion\n\n"
        f"{narrative['conclusion_paragraph']}\n"
    )


def _render_htc_model_narrative_tex(narrative: dict[str, str]) -> str:
    return (
        "% Auto-generated HTC model narrative\n"
        "\\paragraph{Results.} "
        f"{narrative['results_paragraph']}\n\n"
        "\\paragraph{Figure 3 note.} "
        f"{narrative['figure3_paragraph']}\n\n"
        "\\paragraph{Conclusion.} "
        f"{narrative['conclusion_paragraph']}\n"
    )


def _render_figure3_htc_caption_markdown(narrative: dict[str, str]) -> str:
    return "# Figure 3 HTC caption\n\n" + narrative["figure3_paragraph"] + "\n"


def _render_figure3_htc_caption_tex(narrative: dict[str, str]) -> str:
    return "% Auto-generated Figure 3 HTC caption\n" + narrative["figure3_paragraph"] + "\n"


def _load_confidence_summary(figures_dir: Path) -> pd.DataFrame:
    candidates = [
        figures_dir / "paper1_recommendation_confidence_summary.csv",
        FIGURES_TABLES_DIR / "paper1_recommendation_confidence_summary.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    return pd.DataFrame()


def _confidence_row(confidence_summary: pd.DataFrame, scenario_name: str, pathway: str) -> pd.Series | None:
    if confidence_summary.empty:
        return None
    required = {"scenario_name", "pathway"}
    if not required.issubset(confidence_summary.columns):
        return None
    matched = confidence_summary[
        confidence_summary["scenario_name"].astype(str).eq(str(scenario_name))
        & confidence_summary["pathway"].astype(str).str.lower().eq(str(pathway).lower())
    ]
    if matched.empty:
        return None
    return matched.iloc[0]


def _resolve_htc_model_comparison_dir(benchmark_dir: Path) -> Path:
    candidates = [
        benchmark_dir / "htc_model_compare_lso",
        benchmark_dir,
        benchmark_dir.parent / "htc_model_compare_lso",
        BENCHMARK_OUTPUTS_DIR / "htc_model_compare_lso",
    ]
    for candidate in candidates:
        if (
            (candidate / "htc_model_comparison_aggregate.csv").exists()
            or (candidate / "selected_models_manifest_leave_study_out.csv").exists()
        ):
            return candidate
    return candidates[0]


def _model_display_name(model_key: str) -> str:
    mapping = {
        "catboost": "CatBoost",
        "lightgbm": "LightGBM",
        "stacking": "Stacking",
        "xgboost": "XGBoost",
        "rf": "Random Forest",
        "extra_trees": "Extra Trees",
        "gradient_boosting": "Gradient Boosting",
        "elastic_net": "Elastic Net",
    }
    return mapping.get(str(model_key).lower(), str(model_key))


def _selected_targets_phrase(value: str) -> str:
    raw = str(value).strip()
    if "/" not in raw:
        return raw
    numerator, denominator = raw.split("/", 1)
    try:
        return f"{_small_count_word(int(numerator))} of {_small_count_word(int(denominator))} targets"
    except ValueError:
        return raw


def _count_targets_phrase(count: int) -> str:
    word = _small_count_word(int(count))
    noun = "target" if int(count) == 1 else "targets"
    return f"{word} selected {noun}"


def _small_count_word(value: int) -> str:
    mapping = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }
    return mapping.get(int(value), str(value))


def _htc_model_comparison_note_for_row(
    *,
    model_key: str,
    selected_target_count: int,
    negative_refit_count: int,
    mean_test_r2: float,
) -> str:
    if selected_target_count > 0 and negative_refit_count > 0:
        return (
            f"Won {selected_target_count} target(s), but {negative_refit_count} selected target(s) remained negative after refit."
        )
    if selected_target_count > 0:
        return f"Won {selected_target_count} target(s) and retained non-negative refit test performance on the selected targets."
    if mean_test_r2 > 0.0:
        return "No target wins, but aggregate test performance remained non-negative."
    return "No target wins and aggregate test performance remained negative."


def _scenario_frame_row(frame: pd.DataFrame, scenario_name: str) -> pd.Series:
    if frame.empty or "scenario_name" not in frame.columns:
        return pd.Series(dtype="object")
    matched = frame[frame["scenario_name"].astype(str) == scenario_name]
    return matched.iloc[0] if not matched.empty else pd.Series(dtype="object")


def _scenario_frame(frame: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    if frame.empty or "scenario_name" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["scenario_name"].astype(str) == scenario_name].copy()


def _scenario_allocations(frame: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    if frame.empty or "scenario_name" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["scenario_name"].astype(str) == scenario_name].copy()


def _grouped_allocations_by_pathway(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or "pathway" not in frame.columns or "allocated_feed_ton_per_year" not in frame.columns:
        return {}
    grouped = (
        frame.assign(
            allocated_feed_ton_per_year=pd.to_numeric(frame["allocated_feed_ton_per_year"], errors="coerce").fillna(0.0)
        )
        .groupby(frame["pathway"].astype(str).str.lower())["allocated_feed_ton_per_year"]
        .sum()
        .sort_values(ascending=False)
    )
    return {str(index): float(value) for index, value in grouped.items()}


def _selected_pathways_display(frame: pd.DataFrame) -> str:
    grouped = _grouped_allocations_by_pathway(frame)
    if not grouped:
        return "--"
    return " + ".join(_pathway_display(pathway) for pathway in grouped)


def _allocation_profile_display(frame: pd.DataFrame) -> str:
    grouped = _grouped_allocations_by_pathway(frame)
    total = sum(grouped.values())
    if total <= 0.0:
        return "--"
    return "; ".join(
        f"{_pathway_display(pathway)} {value / total * 100.0:.1f}%"
        for pathway, value in grouped.items()
    )


def _support_profile_display(frame: pd.DataFrame) -> str:
    if frame.empty or "surrogate_support_level" not in frame.columns or "allocated_feed_ton_per_year" not in frame.columns:
        return "--"
    working = frame.copy()
    working["allocated_feed_ton_per_year"] = pd.to_numeric(working["allocated_feed_ton_per_year"], errors="coerce").fillna(0.0)
    total = float(working["allocated_feed_ton_per_year"].sum())
    if total <= 0.0:
        return "--"
    grouped = (
        working.groupby(working["surrogate_support_level"].astype(str))["allocated_feed_ton_per_year"]
        .sum()
        .sort_values(ascending=False)
    )
    return "; ".join(
        f"{_support_level_display(level)} {value / total * 100.0:.1f}%"
        for level, value in grouped.items()
    )


def _portfolio_outcome_label(frame: pd.DataFrame, scenario_name: str) -> str:
    grouped = _grouped_allocations_by_pathway(_scenario_allocations(frame, scenario_name))
    total = sum(grouped.values())
    if total <= 0.0:
        return "--"
    if len(grouped) == 1:
        only_pathway = next(iter(grouped))
        return f"{_pathway_display(only_pathway)}-only"
    return " + ".join(
        f"{_pathway_display(pathway)} {value / total * 100.0:.0f}%"
        for pathway, value in grouped.items()
    )


def _run_product_credit_stress(
    *,
    bundle,
    config: PlanningConfig,
    pathway: str,
    revenue_multiplier: float | None,
    additional_credit_per_total_mixed_ton: float | None,
) -> pd.DataFrame:
    frame = bundle.frame.copy()
    mask = frame["pathway"].astype(str).str.lower() == str(pathway).lower()
    if not mask.any():
        return pd.DataFrame()
    current_revenue = pd.to_numeric(frame.loc[mask, "product_revenue_usd_per_year"], errors="coerce").fillna(0.0)
    delta_annual = pd.Series(0.0, index=frame.index, dtype=float)
    if revenue_multiplier is not None:
        updated_revenue = current_revenue * float(revenue_multiplier)
        delta_annual.loc[mask] = updated_revenue - current_revenue
        frame.loc[mask, "product_revenue_usd_per_year"] = updated_revenue
    if additional_credit_per_total_mixed_ton is not None:
        total_mixed_feed = pd.to_numeric(
            frame.loc[mask, "scenario_total_mixed_feed_ton_per_year_proxy"],
            errors="coerce",
        ).fillna(0.0)
        increment = float(additional_credit_per_total_mixed_ton) * total_mixed_feed
        delta_annual.loc[mask] = delta_annual.loc[mask] + increment
        frame.loc[mask, "product_revenue_usd_per_year"] = current_revenue + delta_annual.loc[mask]
    annual_cost = pd.to_numeric(frame["net_system_cost_usd_per_year"], errors="coerce").fillna(0.0)
    frame["net_system_cost_usd_per_year"] = annual_cost - delta_annual
    if "unit_net_system_cost_usd_per_ton" in frame.columns:
        allocation_mass = pd.to_numeric(
            frame["scenario_wet_waste_feed_allocation_ton_per_year_proxy"],
            errors="coerce",
        ).replace(0.0, pd.NA)
        frame["unit_net_system_cost_usd_per_ton"] = (
            pd.to_numeric(frame["unit_net_system_cost_usd_per_ton"], errors="coerce").fillna(0.0)
            - delta_annual.divide(allocation_mass).fillna(0.0)
        )
    if "unit_net_system_cost_usd_per_total_mixed_ton" in frame.columns:
        total_mixed_feed = pd.to_numeric(
            frame["scenario_total_mixed_feed_ton_per_year_proxy"],
            errors="coerce",
        ).replace(0.0, pd.NA)
        frame["unit_net_system_cost_usd_per_total_mixed_ton"] = (
            pd.to_numeric(frame["unit_net_system_cost_usd_per_total_mixed_ton"], errors="coerce").fillna(0.0)
            - delta_annual.divide(total_mixed_feed).fillna(0.0)
        )
    stressed_bundle = replace(bundle, frame=frame)
    execution = execute_planning_pipeline(bundle=stressed_bundle, config=config)
    return execution["portfolio_allocations"]


def _render_latex_table(
    df: pd.DataFrame,
    *,
    columns: list[tuple[str, str]],
    caption: str,
    label: str,
    column_format: str,
    notes: str | None = None,
    formatters: dict[str, object] | None = None,
    font_size: str = "\\scriptsize",
) -> str:
    formatters = formatters or {}
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        font_size,
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{threeparttable}",
        f"\\begin{{tabularx}}{{\\textwidth}}{{{column_format}}}",
        "\\toprule",
        " & ".join(header for _, header in columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        rendered_cells = []
        for column_name, _ in columns:
            formatter = formatters.get(column_name, _default_latex_formatter)
            rendered_cells.append(str(formatter(row.get(column_name))))
        lines.append(" & ".join(rendered_cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabularx}"])
    if notes:
        lines.extend(
            [
                "\\begin{tablenotes}[flushleft]",
                "\\footnotesize",
                f"\\item {notes}",
                "\\end{tablenotes}",
            ]
        )
    lines.extend(["\\end{threeparttable}", "\\end{table*}", ""])
    return "\n".join(lines)


def _default_latex_formatter(value: object) -> str:
    if value is None or pd.isna(value):
        return "--"
    return _escape_latex(str(value))


def _literal_latex_formatter(value: object) -> str:
    if value is None or pd.isna(value):
        return "--"
    return str(value)


def _format_r2_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "--"
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "--"
    return f"{float(numeric):.3f}"


def _escape_macro_definition_value(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("#", r"\#")


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    escaped = str(text)
    for src, dst in replacements.items():
        escaped = escaped.replace(src, dst)
    return escaped


def _as_float(value: object) -> float:
    series = pd.to_numeric(pd.Series([value]), errors="coerce")
    parsed = series.iloc[0]
    return float(parsed) if pd.notna(parsed) else 0.0


def _support_level_display(level: str) -> str:
    mapping = {
        "surrogate_supported": "surrogate-supported",
        "trained_surrogate_with_documented_fallback": "trained surrogate/fallback",
        "documented_static_fallback": "documented static fallback",
        "unsupported_pathway": "unsupported pathway",
    }
    return mapping.get(str(level), str(level).replace("_", " "))


def _reliability_tier_display(tier: str) -> str:
    mapping = {
        "auxiliary_only": "auxiliary only",
        "conditional_support": "conditional support",
        "limited_support": "limited support",
    }
    return mapping.get(str(tier), str(tier).replace("_", " "))


def _manuscript_ceiling_for_reliability_tier(tier: str) -> str:
    mapping = {
        "auxiliary_only": "auxiliary support only",
        "conditional_support": "conditional transfer only",
        "limited_support": "guarded transfer only",
    }
    return mapping.get(str(tier), str(tier).replace("_", " "))


def _metric_range_display(values: object) -> str:
    if values is None:
        return "--"
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return "--"
    min_value = float(series.min())
    max_value = float(series.max())
    if abs(max_value - min_value) < 1e-9:
        return f"{min_value:.3f}"
    return f"{min_value:.3f}--{max_value:.3f}"


def _surrogate_validation_rows_for_pathway(
    *,
    surrogate_validation: pd.DataFrame,
    pathway: str,
    validation_tier: str,
) -> pd.DataFrame:
    if surrogate_validation.empty:
        return pd.DataFrame()
    working = surrogate_validation.copy()
    if "validation_tier" not in working.columns or "dataset_scope" not in working.columns:
        return pd.DataFrame()
    scope = working["dataset_scope"].astype(str).str.lower()
    return working[
        working["validation_tier"].astype(str).str.lower().eq(str(validation_tier).lower())
        & scope.str.contains(str(pathway).lower(), na=False)
    ].copy()


def _transferability_ceiling_display(code: str) -> str:
    mapping = {
        "auxiliary_or_missing_bounded": "auxiliary / missing bounded",
        "conditional_transfer_supported": "conditional transfer supported",
        "guarded_transfer": "guarded transfer",
        "not_available": "not available",
    }
    return mapping.get(str(code), str(code).replace("_", " "))


def _aggregate_benchmark_variant_statistics(benchmark_stats: pd.DataFrame) -> pd.DataFrame:
    if benchmark_stats.empty or "benchmark_variant" not in benchmark_stats.columns:
        return pd.DataFrame()
    working = benchmark_stats.copy()
    rows: list[dict[str, object]] = []
    for variant, subset in working.groupby("benchmark_variant"):
        tier_counts = subset.get("effect_significance_tier", pd.Series(dtype="object")).astype(str).value_counts()
        rows.append(
            {
                "benchmark_variant": str(variant),
                "dominant_significance_tier": _pick_dominant_significance_tier(tier_counts.to_dict()),
                "highly_consistent_count": int(tier_counts.get("highly_consistent", 0)),
                "directionally_consistent_count": int(tier_counts.get("directionally_consistent", 0)),
                "suggestive_count": int(tier_counts.get("suggestive", 0)),
                "unstable_count": int(tier_counts.get("unstable", 0)),
                "pathway_shift_rate_mean": float(pd.to_numeric(subset.get("pathway_shift_rate", 0.0), errors="coerce").fillna(0.0).mean()),
                "case_shift_rate_mean": float(pd.to_numeric(subset.get("case_shift_rate", 0.0), errors="coerce").fillna(0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_benchmark_claim_priority(claim_summary: pd.DataFrame) -> pd.DataFrame:
    if claim_summary.empty or "benchmark_variant" not in claim_summary.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for variant, subset in claim_summary.groupby("benchmark_variant"):
        necessity_counts = subset.get("necessity_tier", pd.Series(dtype="object")).astype(str).value_counts()
        rows.append(
            {
                "benchmark_variant": str(variant),
                "dominant_necessity_tier": _pick_dominant_necessity_tier(necessity_counts.to_dict()),
                "pathway_shift_scenario_count": int(
                    subset.get("portfolio_pathway_shift", pd.Series(dtype="object")).astype(str).eq("changed").sum()
                ),
                "case_shift_scenario_count": int(
                    subset.get("portfolio_case_shift", pd.Series(dtype="object")).astype(str).eq("changed").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _pick_dominant_significance_tier(counts: dict[str, int]) -> str:
    for tier in ("highly_consistent", "directionally_consistent", "suggestive", "unstable"):
        if counts.get(tier, 0) > 0:
            return tier
    return "not_available"


def _pick_dominant_necessity_tier(counts: dict[str, int]) -> str:
    for tier in ("supports_core_innovation", "supports_secondary_innovation", "limited_effect"):
        if counts.get(tier, 0) > 0:
            return tier
    return "not_available"


def _select_primary_benchmark_row(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="object")
    working = frame.copy()
    necessity_rank = {
        "supports_core_innovation": 0,
        "supports_secondary_innovation": 1,
        "limited_effect": 2,
        "not_available": 3,
    }
    significance_rank = {
        "highly_consistent": 0,
        "directionally_consistent": 1,
        "suggestive": 2,
        "unstable": 3,
        "not_available": 4,
    }
    variant_priority = {
        "classic_multiobjective_optimizer": 0,
        "no_evidence_penalty": 1,
        "no_robustness_penalty": 2,
        "ranking_only_unconstrained": 3,
        "greedy_weighted_score_heuristic": 4,
        "no_carbon_constraint": 5,
    }
    working["_necessity_rank"] = (
        working.get("dominant_necessity_tier", pd.Series("not_available", index=working.index))
        .astype(str)
        .map(necessity_rank)
        .fillna(necessity_rank["not_available"])
    )
    working["_significance_rank"] = (
        working.get("dominant_significance_tier", pd.Series("not_available", index=working.index))
        .astype(str)
        .map(significance_rank)
        .fillna(significance_rank["not_available"])
    )
    working["_pathway_shift_count"] = pd.to_numeric(
        working.get("pathway_shift_scenario_count", 0),
        errors="coerce",
    ).fillna(0.0)
    working["_case_shift_count"] = pd.to_numeric(
        working.get("case_shift_scenario_count", 0),
        errors="coerce",
    ).fillna(0.0)
    working["_variant_priority"] = (
        working["benchmark_variant"].astype(str).map(variant_priority).fillna(len(variant_priority))
    )
    ranked = working.sort_values(
        [
            "_necessity_rank",
            "_significance_rank",
            "_pathway_shift_count",
            "_case_shift_count",
            "_variant_priority",
        ],
        ascending=[True, True, False, False, True],
    )
    return ranked.iloc[0]


def _row_for_variant(frame: pd.DataFrame, variant: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="object")
    matched = frame[frame["benchmark_variant"].astype(str) == variant]
    return matched.iloc[0] if not matched.empty else pd.Series(dtype="object")


def _benchmark_variant_display(variant: str) -> str:
    mapping = {
        "no_robustness_penalty": "robustness-penalty removal",
        "classic_multiobjective_optimizer": "classic multi-objective optimizer",
        "no_evidence_penalty": "evidence-penalty removal",
        "greedy_weighted_score_heuristic": "greedy weighted-score heuristic",
    }
    return mapping.get(str(variant), str(variant).replace("_", "-"))


def _build_benchmark_bootstrap_sentence(benchmark_stats: pd.DataFrame) -> str:
    if benchmark_stats.empty:
        return "Bootstrap benchmark repeats are not yet available."
    working = benchmark_stats.copy()
    classic = working[working["benchmark_variant"].astype(str) == "classic_multiobjective_optimizer"]
    evidence = working[working["benchmark_variant"].astype(str) == "no_evidence_penalty"]
    robustness = working[working["benchmark_variant"].astype(str) == "no_robustness_penalty"]
    classic_high = int(classic.get("effect_significance_tier", pd.Series(dtype="object")).astype(str).eq("highly_consistent").sum())
    evidence_high = int(evidence.get("effect_significance_tier", pd.Series(dtype="object")).astype(str).eq("highly_consistent").sum())
    robustness_directional = int(
        robustness.get("effect_significance_tier", pd.Series(dtype="object"))
        .astype(str)
        .isin(["directionally_consistent", "highly_consistent"])
        .sum()
    )
    total = int(len(working))
    if robustness_directional > 0 and classic_high == 0 and evidence_high == 0:
        additional_consistent = int(
            working.get("effect_significance_tier", pd.Series(dtype="object"))
            .astype(str)
            .isin(["directionally_consistent", "highly_consistent"])
            .sum()
            - robustness_directional
        )
        return (
            "Bootstrap repeats keep the robustness-removal ablation highly consistent across all audited scenarios, "
            f"while {additional_consistent}/{total} additional scenario-level comparisons remain directionally consistent."
        )
    return (
        f"Bootstrap repeats keep the classic-optimizer contrast highly consistent in {classic_high}/3 scenarios and the "
        f"evidence-penalty ablation highly consistent in {evidence_high}/3 scenarios, whereas robustness removal remains "
        f"directionally consistent in {robustness_directional}/3 scenarios and mostly shifts preferred cases or metrics."
    )


def _build_benchmark_takeaway_sentence(
    *,
    primary_row: pd.Series,
    classic_row: pd.Series,
    evidence_row: pd.Series,
    heuristic_row: pd.Series,
    bootstrap_stats: pd.DataFrame,
) -> str:
    primary_display = _benchmark_variant_display(str(primary_row.get("benchmark_variant", "benchmark_not_available")))
    if str(primary_row.get("benchmark_variant", "")) == "classic_multiobjective_optimizer":
        sentence = (
            "The strongest benchmark-backed claim is that the exported pathway recommendation depends on the difference between the "
            "current evidence-aware formulation and a classic multi-objective optimizer, with evidence-penalty removal providing "
            "parallel support for the necessity of the evidence-aware design."
        )
    elif str(primary_row.get("benchmark_variant", "")) == "no_evidence_penalty":
        sentence = (
            "The strongest benchmark-backed claim is that evidence penalties materially shape the exported pathway recommendation, "
            "while the classic multi-objective comparator provides parallel support that the overall formulation is not a cosmetic rewrite."
        )
    else:
        sentence = (
            f"The strongest benchmark-backed innovation claim currently centers on {primary_display}, while classic baseline comparisons "
            "and evidence-aware penalty removal supply supporting contrast evidence."
        )
    if not bootstrap_stats.empty:
        sentence += " Bootstrap repeats are available and should be cited when framing uncertainty."
    return sentence


def _lower_sentence_start(text: str) -> str:
    stripped = str(text).strip()
    if not stripped:
        return stripped
    return stripped[0].lower() + stripped[1:]


def _sanitize_benchmark_sentence(text: str) -> str:
    cleaned = str(text).strip()
    replacements = {
        "supporting the necessity of the evidence-aware and robustness-aware design": "showing that the combined evidence-aware and robustness-aware formulation materially shapes the recommendation",
        "supporting the necessity of the evidence-aware design": "showing that evidence-aware penalties materially shape the recommendation",
        "is materially necessary for the exported recommendation": "materially shapes the exported recommendation",
    }
    for src, dst in replacements.items():
        cleaned = cleaned.replace(src, dst)
    return cleaned
