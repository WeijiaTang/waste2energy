# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import pandas as pd

from waste2energy.audit import (
    build_operation_claim_flag_table,
    build_pathway_reliability_summary,
    build_planning_ml_consistency_summary,
)


def test_operation_claim_flag_downgrades_underperforming_rl(tmp_path):
    operation_dir = tmp_path / "operation"
    operation_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "method_type": "baseline_policy",
                "method_name": "hold_plan",
                "reward_mean": 10.0,
                "reward_std": 0.0,
                "max_violation_mean": 0.0,
                "average_reward_mean": 1.0,
                "hold_plan_reward_mean": 10.0,
                "hold_plan_average_reward_mean": 1.0,
                "hold_plan_max_violation_mean": 0.0,
                "reward_improvement_vs_hold_plan_abs": 0.0,
                "reward_improvement_vs_hold_plan_pct": 0.0,
                "average_reward_improvement_vs_hold_plan_abs": 0.0,
                "violation_delta_vs_hold_plan": 0.0,
                "violation_aware_score": 10.0,
                "reward_rank_within_scenario": 1.0,
                "violation_aware_rank_within_scenario": 1.0,
            },
            {
                "scenario_name": "baseline_region_case",
                "method_type": "rl_agent",
                "method_name": "td3",
                "reward_mean": 8.7,
                "reward_std": 0.0,
                "max_violation_mean": 0.0,
                "average_reward_mean": 0.87,
                "hold_plan_reward_mean": 10.0,
                "hold_plan_average_reward_mean": 1.0,
                "hold_plan_max_violation_mean": 0.0,
                "reward_improvement_vs_hold_plan_abs": -1.3,
                "reward_improvement_vs_hold_plan_pct": -0.13,
                "average_reward_improvement_vs_hold_plan_abs": -0.13,
                "violation_delta_vs_hold_plan": 0.0,
                "violation_aware_score": 8.7,
                "reward_rank_within_scenario": 2.0,
                "violation_aware_rank_within_scenario": 2.0,
            },
        ]
    ).to_csv(operation_dir / "rl_vs_baseline_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "method_name": "td3",
                "throughput_nonzero_rate_mean": 0.2,
                "severity_nonzero_rate_mean": 0.8,
            }
        ]
    ).to_csv(operation_dir / "policy_behavior_comparison.csv", index=False)

    flags = build_operation_claim_flag_table(operation_dir)
    td3 = flags.loc[flags["method_name"] == "td3"].iloc[0]

    assert td3["claim_status"] == "unsupported"
    assert "reward_below_90pct_of_hold_plan" in td3["notes"]


def test_operation_claim_flag_handles_negative_hold_plan_rewards(tmp_path):
    operation_dir = tmp_path / "operation_negative_hold_plan"
    operation_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "high_supply_case",
                "method_type": "baseline_policy",
                "method_name": "hold_plan",
                "reward_mean": -0.5,
                "reward_std": 0.0,
                "max_violation_mean": 0.0,
                "average_reward_mean": -0.05,
                "hold_plan_reward_mean": -0.5,
                "hold_plan_average_reward_mean": -0.05,
                "hold_plan_max_violation_mean": 0.0,
                "reward_improvement_vs_hold_plan_abs": 0.0,
                "reward_improvement_vs_hold_plan_pct": 0.0,
                "average_reward_improvement_vs_hold_plan_abs": 0.0,
                "violation_delta_vs_hold_plan": 0.0,
                "violation_aware_score": -0.5,
                "reward_rank_within_scenario": 1.0,
                "violation_aware_rank_within_scenario": 1.0,
            },
            {
                "scenario_name": "high_supply_case",
                "method_type": "rl_agent",
                "method_name": "td3",
                "reward_mean": -0.6,
                "reward_std": 0.0,
                "max_violation_mean": 0.0,
                "average_reward_mean": -0.06,
                "hold_plan_reward_mean": -0.5,
                "hold_plan_average_reward_mean": -0.05,
                "hold_plan_max_violation_mean": 0.0,
                "reward_improvement_vs_hold_plan_abs": -0.1,
                "reward_improvement_vs_hold_plan_pct": -0.2,
                "average_reward_improvement_vs_hold_plan_abs": -0.01,
                "violation_delta_vs_hold_plan": 0.0,
                "violation_aware_score": -0.6,
                "reward_rank_within_scenario": 2.0,
                "violation_aware_rank_within_scenario": 2.0,
            },
        ]
    ).to_csv(operation_dir / "rl_vs_baseline_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "high_supply_case",
                "method_name": "td3",
                "throughput_nonzero_rate_mean": 0.0,
                "severity_nonzero_rate_mean": 0.0,
            }
        ]
    ).to_csv(operation_dir / "policy_behavior_comparison.csv", index=False)

    flags = build_operation_claim_flag_table(operation_dir)
    td3 = flags.loc[flags["method_name"] == "td3"].iloc[0]

    assert td3["reward_ratio_vs_hold_plan"] == 0.8
    assert td3["claim_status"] == "unsupported"
    assert "reward_below_90pct_of_hold_plan" in td3["notes"]


def test_pathway_reliability_summary_adds_htc_restriction():
    ml_flags = pd.DataFrame(
        [
            {"summary_label": "leave_study_out", "dataset_key": "htc_direct", "target_column": "a", "claim_status": "unsupported"},
            {"summary_label": "leave_study_out", "dataset_key": "htc_direct", "target_column": "b", "claim_status": "weak"},
            {"summary_label": "leave_study_out", "dataset_key": "htc_direct", "target_column": "c", "claim_status": "supportive"},
            {"summary_label": "leave_study_out", "dataset_key": "pyrolysis_direct", "target_column": "a", "claim_status": "supportive"},
            {"summary_label": "leave_study_out", "dataset_key": "pyrolysis_direct", "target_column": "b", "claim_status": "supportive"},
        ]
    )

    summary = build_pathway_reliability_summary(ml_flags)
    htc = summary.loc[summary["pathway"] == "htc"].iloc[0]

    assert htc["reliability_tier"] == "auxiliary_only"
    assert "lack cross-study generalizability" in htc["reviewer_restriction_sentence"]


def test_planning_ml_consistency_flags_high_risk(tmp_path):
    planning_dir = tmp_path / "planning"
    planning_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "htc",
                "portfolio_allocated_feed_share": 0.7,
            },
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "portfolio_allocated_feed_share": 0.3,
            },
        ]
    ).to_csv(planning_dir / "pathway_summary.csv", index=False)
    reliability = pd.DataFrame(
        [
            {"pathway": "htc", "reliability_score": 0.2},
            {"pathway": "pyrolysis", "reliability_score": 0.9},
        ]
    )

    summary = build_planning_ml_consistency_summary(planning_dir, reliability)
    row = summary.iloc[0]

    assert row["risk_tier"] == "high_risk"
    assert row["unsupported_allocation_share"] == 0.7
