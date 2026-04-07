# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import warnings

import pandas as pd

from waste2energy.audit import (
    InconsistencyWarning,
    build_operation_claim_flag_table,
    build_pathway_reliability_summary,
    build_planning_claim_flag_table,
    build_planning_ml_consistency_summary,
    build_confirmatory_audit,
)
from waste2energy.manuscript_sync import sync_planning_summary_to_latex


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


def test_operation_claim_flag_marks_missing_hold_plan_reference_as_not_evaluated(tmp_path):
    operation_dir = tmp_path / "operation_missing_hold_plan"
    operation_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "method_type": "rl_agent",
                "method_name": "td3",
                "reward_mean": 8.7,
                "reward_std": 0.0,
                "max_violation_mean": 0.0,
                "average_reward_mean": 0.87,
                "hold_plan_reward_mean": pd.NA,
                "hold_plan_average_reward_mean": pd.NA,
                "hold_plan_max_violation_mean": pd.NA,
                "reward_improvement_vs_hold_plan_abs": pd.NA,
                "reward_improvement_vs_hold_plan_pct": pd.NA,
                "average_reward_improvement_vs_hold_plan_abs": pd.NA,
                "violation_delta_vs_hold_plan": pd.NA,
                "violation_aware_score": 8.7,
                "reward_rank_within_scenario": 1.0,
                "violation_aware_rank_within_scenario": 1.0,
            },
        ]
    ).to_csv(operation_dir / "rl_vs_baseline_comparison.csv", index=False)

    flags = build_operation_claim_flag_table(operation_dir)
    td3 = flags.loc[flags["method_name"] == "td3"].iloc[0]

    assert td3["claim_status"] == "not_evaluated"
    assert pd.isna(td3["reward_improvement_vs_hold_plan_pct"])
    assert pd.isna(td3["reward_ratio_vs_hold_plan"])
    assert "missing_hold_plan_reference" in td3["notes"]


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


def test_planning_ml_consistency_separates_missing_mapping_from_unsupported(tmp_path):
    planning_dir = tmp_path / "planning_missing_mapping"
    planning_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "htc",
                "portfolio_allocated_feed_share": 0.2,
            },
            {
                "scenario_name": "baseline_region_case",
                "pathway": "gasification",
                "portfolio_allocated_feed_share": 0.8,
            },
        ]
    ).to_csv(planning_dir / "pathway_summary.csv", index=False)
    reliability = pd.DataFrame(
        [
            {"pathway": "htc", "reliability_score": 0.75, "reliability_tier": "conditional_support"},
        ]
    )

    summary = build_planning_ml_consistency_summary(planning_dir, reliability)
    row = summary.iloc[0]

    assert row["risk_tier"] == "high_risk"
    assert row["missing_reliability_allocation_share"] == 0.8
    assert row["unsupported_allocation_share"] == 0.0
    assert "without any pathway-level reliability mapping" in row["risk_note"]


def test_planning_ml_consistency_marks_data_gap_when_no_evidence_scores_are_available(tmp_path):
    planning_dir = tmp_path / "planning_data_gap"
    planning_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "htc",
                "portfolio_allocated_feed_share": 1.0,
            }
        ]
    ).to_csv(planning_dir / "pathway_summary.csv", index=False)
    reliability = pd.DataFrame(
        [
            {"pathway": "htc", "reliability_score": pd.NA, "reliability_tier": "not_evaluated"},
        ]
    )

    summary = build_planning_ml_consistency_summary(planning_dir, reliability)
    row = summary.iloc[0]

    assert row["risk_tier"] == "high_risk"
    assert row["not_evaluated_allocation_share"] == 1.0
    assert pd.isna(row["planning_ml_consistency_correlation"])


def test_confirmatory_audit_warns_when_surrogate_supported_share_is_below_threshold(tmp_path):
    planning_dir = tmp_path / "planning"
    scenario_dir = tmp_path / "scenarios"
    operation_dir = tmp_path / "operation"
    surrogate_dir = tmp_path / "surrogate"
    outputs_root = tmp_path / "outputs"
    planning_dir.mkdir()
    scenario_dir.mkdir()
    operation_dir.mkdir()
    surrogate_dir.mkdir()
    (outputs_root / "surrogate").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"pathway": "pyrolysis", "reliability_score": 0.9, "reliability_tier": "conditional_support"},
            {"pathway": "ad", "reliability_score": 0.1, "reliability_tier": "limited_support"},
        ]
    )
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "portfolio_allocated_feed_share": 0.3},
            {"scenario_name": "baseline_region_case", "pathway": "ad", "portfolio_allocated_feed_share": 0.7},
        ]
    ).to_csv(planning_dir / "pathway_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "allocated_feed_ton_per_year": 30.0, "allocated_feed_share": 0.3, "surrogate_support_level": "surrogate_supported"},
            {"scenario_name": "baseline_region_case", "pathway": "ad", "allocated_feed_ton_per_year": 70.0, "allocated_feed_share": 0.7, "surrogate_support_level": "unsupported_pathway"},
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "writing_label": "supporting portfolio", "selected_in_baseline_portfolio": True, "baseline_portfolio_share_pct": 30.0, "max_stress_selection_rate": 50.0, "best_case_score_index": 1.0, "claim_boundary": "planning-ready", "results_sentence": "x"},
            {"scenario_name": "baseline_region_case", "pathway": "ad", "writing_label": "supporting portfolio", "selected_in_baseline_portfolio": True, "baseline_portfolio_share_pct": 70.0, "max_stress_selection_rate": 50.0, "best_case_score_index": 0.9, "claim_boundary": "comparison only", "results_sentence": "y"},
        ]
    ).to_csv(planning_dir / "main_results_table.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "feedstock_scale_factor": 1.0, "feedstock_cost_elasticity": 0.0, "carbon_tax_usd_per_ton_co2e": 0.0, "evidence_source": "x", "evidence_reference": "y", "evidence_rationale": "z"},
        ]
    ).to_csv(planning_dir / "scenario_external_evidence.csv", index=False)
    pd.DataFrame([]).to_csv(scenario_dir / "stress_test_summary.csv", index=False)
    pd.DataFrame([]).to_csv(operation_dir / "rl_vs_baseline_comparison.csv", index=False)
    pd.DataFrame([]).to_csv(outputs_root / "surrogate" / "traditional_ml_suite_summary.csv", index=False)
    pd.DataFrame([]).to_csv(outputs_root / "surrogate" / "traditional_ml_suite_summary_strict_group.csv", index=False)
    pd.DataFrame([]).to_csv(outputs_root / "surrogate" / "traditional_ml_suite_summary_leave_study_out.csv", index=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = build_planning_ml_consistency_summary(
            planning_dir,
            pd.DataFrame(
                [
                    {"pathway": "pyrolysis", "reliability_score": 0.9, "reliability_tier": "conditional_support"},
                    {"pathway": "ad", "reliability_score": 0.1, "reliability_tier": "limited_support"},
                ]
            ),
        )
        from waste2energy.audit import _emit_surrogate_led_inconsistency_warnings

        _emit_surrogate_led_inconsistency_warnings(summary)

    assert any(issubclass(item.category, InconsistencyWarning) for item in caught)


def test_planning_claim_flag_table_adds_support_level_and_evidence_gap(tmp_path):
    planning_dir = tmp_path / "planning_flags"
    scenario_dir = tmp_path / "scenario_flags"
    planning_dir.mkdir()
    scenario_dir.mkdir()

    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "ad", "writing_label": "supporting portfolio", "selected_in_baseline_portfolio": True, "baseline_portfolio_share_pct": 60.0, "max_stress_selection_rate": 0.0, "best_case_score_index": 0.8, "claim_boundary": "comparison only", "results_sentence": "ad selected"},
        ]
    ).to_csv(planning_dir / "main_results_table.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "ad", "portfolio_selected_count": 1, "portfolio_allocated_feed_share": 0.6, "portfolio_top_case_id": "case-ad"},
        ]
    ).to_csv(planning_dir / "pathway_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "ad", "allocated_feed_share": 0.6, "surrogate_support_level": "unsupported_pathway"},
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "feedstock_scale_factor": 1.0, "feedstock_cost_elasticity": 0.0, "carbon_tax_usd_per_ton_co2e": 0.0, "evidence_source": "x", "evidence_reference": "y", "evidence_rationale": "z"},
        ]
    ).to_csv(planning_dir / "scenario_external_evidence.csv", index=False)
    pd.DataFrame([]).to_csv(scenario_dir / "stress_test_summary.csv", index=False)

    flags = build_planning_claim_flag_table(planning_dir, scenario_dir)
    row = flags.iloc[0]

    assert row["Surrogate_Support_Level"] == "unsupported_pathway"
    assert row["evidence_gap_flag"] == "Evidence Gap: Unsupported Pathway"


def test_planning_claim_flag_table_backfills_support_level_from_scored_cases(tmp_path):
    planning_dir = tmp_path / "planning_flags_scored_cases"
    scenario_dir = tmp_path / "scenario_flags_scored_cases"
    planning_dir.mkdir()
    scenario_dir.mkdir()

    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "writing_label": "stress-sensitive alternative",
                "selected_in_baseline_portfolio": False,
                "baseline_portfolio_share_pct": 0.0,
                "max_stress_selection_rate": 12.5,
                "best_case_score_index": 0.8,
                "claim_boundary": "planning-ready candidate with blended-feed caution",
                "results_sentence": "pyrolysis stress support",
            }
        ]
    ).to_csv(planning_dir / "main_results_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "portfolio_selected_count": 0,
                "portfolio_allocated_feed_share": 0.0,
                "portfolio_top_case_id": "case-pyro",
            }
        ]
    ).to_csv(planning_dir / "pathway_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "surrogate_mode": "trained_surrogate_with_documented_fallback",
            }
        ]
    ).to_csv(planning_dir / "scored_cases.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "feedstock_scale_factor": 1.0,
                "feedstock_cost_elasticity": 0.0,
                "carbon_tax_usd_per_ton_co2e": 0.0,
                "evidence_source": "x",
                "evidence_reference": "y",
                "evidence_rationale": "z",
            }
        ]
    ).to_csv(planning_dir / "scenario_external_evidence.csv", index=False)
    pd.DataFrame([]).to_csv(scenario_dir / "stress_test_summary.csv", index=False)

    flags = build_planning_claim_flag_table(planning_dir, scenario_dir)
    row = flags.iloc[0]

    assert row["Surrogate_Support_Level"] == "trained_surrogate_with_documented_fallback"
    assert row["evidence_gap_flag"] == ""


def test_manuscript_sync_writes_macros_and_relabels_ad_status(tmp_path):
    planning_dir = tmp_path / "planning_sync"
    paper_dir = tmp_path / "paper"
    planning_dir.mkdir()
    paper_dir.mkdir()
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "top_ranked_case_id": "case-1"},
        ]
    ).to_csv(planning_dir / "scenario_summary.csv", index=False)
    pd.DataFrame(
        [
            {"pathway": "ad", "allocated_feed_share": 0.25},
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)
    abstract_path = paper_dir / "00-abstract.tex"
    abstract_path.write_text("This abstract describes an AD-free portfolio.\n", encoding="utf-8")
    macros_path = paper_dir / "99-auto-macros.tex"

    result = sync_planning_summary_to_latex(
        planning_dir=planning_dir,
        abstract_path=abstract_path,
        macros_path=macros_path,
    )

    assert result["ad_status_label"] == "AD-limited"
    assert "AD-limited" in abstract_path.read_text(encoding="utf-8")
    assert "\\newcommand{\\PlanningADStatus}{AD-limited}" in macros_path.read_text(encoding="utf-8")
