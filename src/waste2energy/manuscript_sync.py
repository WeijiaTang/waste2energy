from __future__ import annotations

from pathlib import Path

import pandas as pd


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

    scenario_count = int(len(scenario_summary))
    top_case_count = int(scenario_summary["top_ranked_case_id"].dropna().astype(str).nunique()) if "top_ranked_case_id" in scenario_summary.columns else 0
    macros_file.parent.mkdir(parents=True, exist_ok=True)
    macros_file.write_text(
        "\n".join(
            [
                f"\\newcommand{{\\PlanningScenarioCount}}{{{scenario_count}}}",
                f"\\newcommand{{\\PlanningTopCaseCount}}{{{top_case_count}}}",
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
        "ad_selected": ad_selected,
        "ad_status_label": ad_status_label,
        "ad_allocated_share": ad_allocated_share,
        "abstract_rewritten": abstract_rewritten,
        "macros_path": str(macros_file),
    }
