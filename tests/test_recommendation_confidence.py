# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import pandas as pd

from waste2energy.evidence_policy import EVIDENCE_POLICY_VERSION
from waste2energy.planning.confidence import build_recommendation_confidence_summary


def test_recommendation_confidence_rewards_supported_selected_pathway():
    frame = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "surrogate_support_level": "surrogate_supported",
                "selected_in_baseline_portfolio": True,
                "baseline_portfolio_share_pct": 88.0,
                "max_stress_selection_rate": 75.0,
                "score_gap_to_scenario_best_pct": 12.0,
                "claim_boundary": "planning-ready candidate with blended-feed caution",
            },
            {
                "scenario_name": "baseline_region_case",
                "pathway": "ad",
                "surrogate_support_level": "unsupported_pathway",
                "selected_in_baseline_portfolio": False,
                "baseline_portfolio_share_pct": 0.0,
                "max_stress_selection_rate": 0.0,
                "score_gap_to_scenario_best_pct": 80.0,
                "claim_boundary": "planning comparison only",
            },
        ]
    )

    summary = build_recommendation_confidence_summary(frame)
    pyrolysis = summary.loc[summary["pathway"] == "pyrolysis"].iloc[0]
    ad = summary.loc[summary["pathway"] == "ad"].iloc[0]

    assert pyrolysis["recommendation_confidence_tier"] in {"high", "moderate"}
    assert ad["recommendation_confidence_tier"] == "low"
    assert pyrolysis["recommendation_confidence_score"] > ad["recommendation_confidence_score"]
    assert pyrolysis["support_score_component"] > ad["support_score_component"]
    assert pyrolysis["stress_support_score_component"] > ad["stress_support_score_component"]
    assert pyrolysis["recommendation_confidence_policy_version"] == EVIDENCE_POLICY_VERSION


def test_recommendation_confidence_caps_anchor_only_pathway():
    frame = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "baseline",
                "surrogate_support_level": "unsupported_pathway",
                "selected_in_baseline_portfolio": False,
                "baseline_portfolio_share_pct": 0.0,
                "max_stress_selection_rate": 100.0,
                "score_gap_to_scenario_best_pct": 0.0,
                "claim_boundary": "comparison anchor only",
            }
        ]
    )

    summary = build_recommendation_confidence_summary(frame)
    row = summary.iloc[0]

    assert row["recommendation_confidence_tier"] == "low"
    assert row["recommendation_confidence_score"] <= 0.25
    assert row["confidence_cap_rule"] == "comparison_anchor_cap"
