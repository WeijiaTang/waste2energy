from __future__ import annotations

from pathlib import Path

import pandas as pd


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


def sync_planning_summary_to_latex(
    *,
    planning_dir: str | Path,
    abstract_path: str | Path,
    macros_path: str | Path,
) -> dict[str, object]:
    planning_root = Path(planning_dir)
    abstract_file = Path(abstract_path)
    macros_file = Path(macros_path)

    scenario_summary = pd.read_csv(planning_root / "scenario_summary.csv")
    portfolio_allocations = pd.read_csv(planning_root / "portfolio_allocations.csv")
    portfolio_summary = _read_csv_if_exists(planning_root / "portfolio_summary.csv")
    pathway_summary = _read_csv_if_exists(planning_root / "pathway_summary.csv")
    main_results_table = _read_csv_if_exists(planning_root / "main_results_table.csv")

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

    scenario_count = int(len(scenario_summary))
    top_case_count = (
        int(scenario_summary["top_ranked_case_id"].dropna().astype(str).nunique())
        if "top_ranked_case_id" in scenario_summary.columns
        else 0
    )
    macros_file.parent.mkdir(parents=True, exist_ok=True)
    macros_file.write_text(
        "\n".join(
            [
                f"\\newcommand{{\\PlanningScenarioCount}}{{{scenario_count}}}",
                f"\\newcommand{{\\PlanningTopCaseCount}}{{{top_case_count}}}",
                f"\\newcommand{{\\PlanningDominantPathwayDisplay}}{{{dominance['dominant_pathway_display']}}}",
                f"\\newcommand{{\\PlanningDominancePattern}}{{{dominance['dominance_pattern']}}}",
                f"\\newcommand{{\\PlanningBaselineDominantPathwayDisplay}}{{{dominance['baseline_pathway_display']}}}",
                f"\\newcommand{{\\PlanningHighSupplyDominantPathwayDisplay}}{{{dominance['high_supply_pathway_display']}}}",
                f"\\newcommand{{\\PlanningPolicySupportDominantPathwayDisplay}}{{{dominance['policy_support_pathway_display']}}}",
                f"\\newcommand{{\\PlanningHighSupplyCoveragePct}}{{{dominance['high_supply_coverage_pct']}}}",
                f"\\newcommand{{\\PlanningPyrolysisRole}}{{{dominance['pyrolysis_role']}}}",
                f"\\newcommand{{\\PlanningHTCRole}}{{{dominance['htc_role']}}}",
                f"\\newcommand{{\\PlanningADRole}}{{{dominance['ad_role']}}}",
                f"\\newcommand{{\\PlanningIntroductionDominanceSentence}}{{{dominance['introduction_sentence']}}}",
                f"\\newcommand{{\\PlanningDominanceSummarySentence}}{{{dominance['abstract_sentence']}}}",
                f"\\newcommand{{\\PlanningHighlightsDominanceBullet}}{{{dominance['highlights_sentence']}}}",
                f"\\newcommand{{\\PlanningResultsDominanceSentence}}{{{dominance['results_sentence']}}}",
                f"\\newcommand{{\\PlanningConclusionDominanceSentence}}{{{dominance['conclusion_sentence']}}}",
                f"\\newcommand{{\\PlanningADStatus}}{{{ad_status_label}}}",
                f"\\newcommand{{\\PlanningADAllocatedShare}}{{{ad_allocated_share * 100.0:.1f}\\%}}",
                f"\\newcommand{{\\PlanningADStatusNote}}{{{ad_status_note}}}",
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
                "Under the current California-oriented assumptions, the strongest result is not a universal technology "
                f"winner but an evidence-qualified thermochemical recommendation whose baseline allocation is "
                f"{dominant_display}-dominant in the current exported runs, even though "
                f"{_pathway_display(score_leader)} retains stronger case-level score leadership and AD remains "
                f"{_with_indefinite_article(ad_role)}."
            )
        secondary_pathway = "pyrolysis" if dominant_pathway != "pyrolysis" else "htc"
        secondary_role = pyrolysis_role if secondary_pathway == "pyrolysis" else htc_role
        return (
            "Under the current California-oriented assumptions, the strongest result is not a universal technology "
            f"winner but an evidence-qualified thermochemical recommendation whose baseline allocation is "
            f"{dominant_display}-dominant in the current exported runs, while "
            f"{_pathway_display(secondary_pathway)} remains {_with_indefinite_article(secondary_role)} and AD "
            f"remains {_with_indefinite_article(ad_role)}."
        )
    return (
        "Under the current California-oriented assumptions, the strongest result is not a universal technology "
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
