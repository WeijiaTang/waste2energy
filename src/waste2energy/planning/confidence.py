# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import numpy as np
import pandas as pd

from ..evidence_policy import (
    CONFIDENCE_COMPONENT_WEIGHTS,
    EVIDENCE_POLICY_VERSION,
    classify_confidence_tier,
    support_score_for_level,
)


def build_recommendation_confidence_summary(main_results: pd.DataFrame) -> pd.DataFrame:
    if main_results.empty:
        return pd.DataFrame(
            columns=[
                "scenario_name",
                "pathway",
                "surrogate_support_level",
                "support_score_component",
                "stress_support_score_component",
                "role_score_component",
                "confidence_cap_rule",
                "recommendation_confidence_policy_version",
                "recommendation_confidence_score",
                "recommendation_confidence_tier",
                "recommendation_confidence_note",
            ]
        )

    working = main_results.copy()
    working["surrogate_support_level"] = (
        working.get("surrogate_support_level", pd.Series(["unknown"] * len(working), index=working.index))
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .replace("", "unknown")
    )
    working["support_score"] = working["surrogate_support_level"].map(support_score_for_level).astype(float)
    working["selected_flag"] = (
        working.get("selected_in_baseline_portfolio", pd.Series([False] * len(working), index=working.index))
        .fillna(False)
        .astype(bool)
    )
    working["baseline_portfolio_share_score"] = _normalized_pct(
        working.get("baseline_portfolio_share_pct", pd.Series([0.0] * len(working), index=working.index))
    )
    working["stress_support_score"] = _normalized_pct(
        working.get("max_stress_selection_rate", pd.Series([0.0] * len(working), index=working.index))
    )
    working["score_competitiveness"] = 1.0 - _normalized_pct(
        working.get("score_gap_to_scenario_best_pct", pd.Series([100.0] * len(working), index=working.index))
    )

    role_score = np.where(
        working["selected_flag"],
        working["baseline_portfolio_share_score"],
        0.5 * working["score_competitiveness"] + 0.5 * working["stress_support_score"],
    )
    working["support_score_component"] = working["support_score"]
    working["stress_support_score_component"] = working["stress_support_score"]
    working["role_score_component"] = role_score
    working["recommendation_confidence_score"] = (
        CONFIDENCE_COMPONENT_WEIGHTS["support"] * working["support_score_component"]
        + CONFIDENCE_COMPONENT_WEIGHTS["stress"] * working["stress_support_score_component"]
        + CONFIDENCE_COMPONENT_WEIGHTS["role"] * working["role_score_component"]
    )

    claim_boundary = working.get("claim_boundary", pd.Series([""] * len(working), index=working.index)).astype(str)
    comparison_anchor_mask = claim_boundary.str.contains("anchor", case=False, na=False)
    comparison_only_mask = claim_boundary.str.contains("comparison only", case=False, na=False)
    low_stress_unselected_mask = (~working["selected_flag"]) & working["stress_support_score"].le(0.0)
    working["confidence_cap_rule"] = "none"

    working.loc[comparison_anchor_mask, "recommendation_confidence_score"] = working.loc[
        comparison_anchor_mask, "recommendation_confidence_score"
    ].clip(upper=0.25)
    working.loc[comparison_anchor_mask, "confidence_cap_rule"] = "comparison_anchor_cap"
    working.loc[comparison_only_mask, "recommendation_confidence_score"] = working.loc[
        comparison_only_mask, "recommendation_confidence_score"
    ].clip(upper=0.40)
    working.loc[comparison_only_mask, "confidence_cap_rule"] = "comparison_only_cap"
    working.loc[low_stress_unselected_mask, "recommendation_confidence_score"] = working.loc[
        low_stress_unselected_mask, "recommendation_confidence_score"
    ].clip(upper=0.54)
    uncapped_rules = working["confidence_cap_rule"].eq("none")
    working.loc[low_stress_unselected_mask & uncapped_rules, "confidence_cap_rule"] = "unselected_no_stress_cap"

    working["recommendation_confidence_tier"] = working["recommendation_confidence_score"].map(classify_confidence_tier)
    working["recommendation_confidence_note"] = working.apply(_build_confidence_note, axis=1)
    working["recommendation_confidence_score"] = working["recommendation_confidence_score"].round(3)
    working["support_score_component"] = working["support_score_component"].round(3)
    working["stress_support_score_component"] = working["stress_support_score_component"].round(3)
    working["role_score_component"] = pd.Series(working["role_score_component"], index=working.index).round(3)
    working["recommendation_confidence_policy_version"] = EVIDENCE_POLICY_VERSION

    columns = [
        "scenario_name",
        "pathway",
        "surrogate_support_level",
        "support_score_component",
        "stress_support_score_component",
        "role_score_component",
        "confidence_cap_rule",
        "recommendation_confidence_policy_version",
        "recommendation_confidence_score",
        "recommendation_confidence_tier",
        "recommendation_confidence_note",
    ]
    return working[columns].sort_values(
        ["scenario_name", "recommendation_confidence_score", "pathway"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _normalized_pct(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    return (numeric / 100.0).clip(lower=0.0, upper=1.0)


def _build_confidence_note(row: pd.Series) -> str:
    tier = str(row.get("recommendation_confidence_tier", "low"))
    selected = bool(row.get("selected_flag", False))
    support_level = str(row.get("surrogate_support_level", "unknown"))
    stress_score = float(row.get("stress_support_score", 0.0))

    if tier == "high":
        return "Selected and stress-persistent under the strongest currently available surrogate support."
    if tier == "moderate":
        if selected:
            return "Selected under current constraints, but either stress persistence or evidence maturity remains partial."
        return "Competitive under current constraints with partial stress support, but not strong enough for an unrestricted claim."
    if tier == "guarded":
        if support_level in {"documented_static_fallback", "unsupported_pathway"}:
            return "Recommendation remains evidence-bounded because pathway support relies on fallback or unsupported evidence."
        if selected:
            return "Selected in the current baseline portfolio, but recommendation confidence should remain guarded."
        if stress_score > 0.0:
            return "Not baseline-selected, yet stress sensitivity keeps the pathway relevant as a guarded alternative."
        return "Competitive signal is visible, but evidence and robustness support remain too limited for a strong recommendation."
    if selected:
        return "Current selection should be written conservatively because confidence remains low under evidence or robustness screening."
    return "Use for comparison only; current evidence is insufficient for a confident recommendation."
