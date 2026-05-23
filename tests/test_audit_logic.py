# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json
import warnings

import pandas as pd
import pytest

import waste2energy.manuscript_sync as manuscript_sync_module
import waste2energy.audit as audit_module
from waste2energy.audit import (
    InconsistencyWarning,
    build_artifact_inventory,
    build_benchmark_claim_summary,
    build_benchmark_manuscript_sentences,
    build_ad_boundary_fairness_audit,
    build_binding_constraint_audit,
    build_duplicate_candidate_audit,
    build_hhv_dominance_audit,
    build_hhv_imputation_sensitivity,
    build_hhv_replanning_sensitivity,
    build_ml_best_result_summary,
    build_ml_claim_flag_table,
    build_ml_refit_provenance_summary,
    build_operation_claim_flag_table,
    build_planning_artifact_consistency_summary,
    build_pathway_reliability_summary,
    build_planning_claim_flag_table,
    build_planning_transferability_risk_summary,
    build_planning_ml_consistency_summary,
    build_surrogate_extrapolation_audit,
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


def test_artifact_inventory_flags_stale_operation_comparison_outputs(tmp_path):
    planning_dir = tmp_path / "planning"
    scenario_dir = tmp_path / "scenario"
    operation_dir = tmp_path / "operation"
    benchmark_dir = tmp_path / "benchmark"
    for directory in (planning_dir, scenario_dir, operation_dir, benchmark_dir):
        directory.mkdir()

    (planning_dir / "main_results_table.csv").write_text("", encoding="utf-8")
    (planning_dir / "main_results_table_manifest.json").write_text("{}", encoding="utf-8")
    (planning_dir / "pathway_summary.csv").write_text("", encoding="utf-8")
    (planning_dir / "portfolio_allocations.csv").write_text("", encoding="utf-8")
    (scenario_dir / "stress_test_summary.csv").write_text("", encoding="utf-8")
    (scenario_dir / "decision_stability.csv").write_text("", encoding="utf-8")
    (scenario_dir / "cross_scenario_stability.csv").write_text("", encoding="utf-8")
    (scenario_dir / "uncertainty_summary.csv").write_text("", encoding="utf-8")

    (planning_dir / "run_config.json").write_text(
        json.dumps({"generated_at_utc": "2026-04-08T00:29:16+00:00"}),
        encoding="utf-8",
    )
    (scenario_dir / "run_config.json").write_text(
        json.dumps({"generated_at_utc": "2026-04-08T00:29:25+00:00"}),
        encoding="utf-8",
    )
    (benchmark_dir / "run_config.json").write_text(
        json.dumps({"generated_at_utc": "2026-04-08T00:29:30+00:00"}),
        encoding="utf-8",
    )
    (operation_dir / "run_config.json").write_text(
        json.dumps({"generated_at_utc": "2026-04-07T08:35:10+00:00"}),
        encoding="utf-8",
    )
    for file_name in [
        "benchmark_summary.csv",
        "benchmark_allocations.csv",
        "benchmark_scenario_summary.csv",
        "benchmark_shift_summary.csv",
        "benchmark_diagnostics.csv",
        "benchmark_bootstrap_shift_samples.csv",
        "benchmark_statistical_summary.csv",
    ]:
        (benchmark_dir / file_name).write_text("", encoding="utf-8")
    for file_name in [
        "baseline_policy_summary.csv",
        "baseline_rollout_steps.csv",
        "policy_behavior_comparison.csv",
        "rl_vs_baseline_comparison.csv",
        "sac_training_summary.csv",
        "sac_evaluation_rollouts.csv",
        "sac_evaluation_episode_summary.csv",
        "sac_seed_aggregate_summary.csv",
        "td3_training_summary.csv",
        "td3_evaluation_rollouts.csv",
        "td3_evaluation_episode_summary.csv",
        "td3_seed_aggregate_summary.csv",
    ]:
        (operation_dir / file_name).write_text("", encoding="utf-8")

    inventory = build_artifact_inventory(
        summary_paths={"strict_group": tmp_path / "strict_group.csv"},
        operation_dir=operation_dir,
        planning_dir=planning_dir,
        scenario_dir=scenario_dir,
        benchmark_dir=benchmark_dir,
    )

    operation_rows = inventory[inventory["artifact_group"] == "operation_comparison"]
    assert not operation_rows.empty
    assert set(operation_rows["freshness_status"]) == {"stale"}
    assert operation_rows["freshness_note"].str.contains("should be regenerated").all()

    benchmark_rows = inventory[inventory["artifact_group"] == "benchmark"]
    assert set(benchmark_rows["artifact_label"]) == {
        "benchmark_summary.csv",
        "benchmark_allocations.csv",
        "benchmark_scenario_summary.csv",
        "benchmark_shift_summary.csv",
        "benchmark_diagnostics.csv",
        "benchmark_bootstrap_shift_samples.csv",
        "benchmark_statistical_summary.csv",
        "run_config.json",
    }
    assert benchmark_rows["exists"].all()


def test_planning_artifact_consistency_flags_stale_allocation_tables(tmp_path):
    planning_dir = tmp_path / "planning"
    figures_dir = tmp_path / "figures_tables"
    audit_dir = tmp_path / "audit"
    for directory in (planning_dir, figures_dir, audit_dir):
        directory.mkdir()

    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "allocated_feed_ton_per_year": 88.3},
            {"scenario_name": "baseline_region_case", "pathway": "htc", "allocated_feed_ton_per_year": 11.7},
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)
    stale_rows = pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "baseline_portfolio_share_pct": 90.0},
            {"scenario_name": "baseline_region_case", "pathway": "htc", "baseline_portfolio_share_pct": 10.0},
            {"scenario_name": "baseline_region_case", "pathway": "ad", "baseline_portfolio_share_pct": 0.0},
        ]
    )
    stale_rows.to_csv(planning_dir / "main_results_table.csv", index=False)
    stale_rows.to_csv(figures_dir / "paper1_planning_results_table.csv", index=False)
    stale_rows.to_csv(audit_dir / "planning_claim_flag_table.csv", index=False)

    summary = build_planning_artifact_consistency_summary(
        planning_dir,
        figures_dir=figures_dir,
        audit_dir=audit_dir,
        tolerance_pct_point=0.1,
    )

    failed = summary[summary["consistency_status"] == "fail"]
    assert not failed.empty
    assert set(failed["artifact_label"]) == {
        "main_results_table.csv",
        "paper1_planning_results_table.csv",
        "planning_claim_flag_table.csv",
    }
    pyrolysis = failed[
        (failed["artifact_label"] == "main_results_table.csv")
        & (failed["pathway"] == "pyrolysis")
    ].iloc[0]
    assert pyrolysis["expected_share_pct"] == 88.3
    assert pyrolysis["observed_share_pct"] == 90.0
    assert pyrolysis["absolute_difference_pct_point"] == 1.7


def test_planning_artifact_consistency_checks_compact_profiles_and_costs(tmp_path):
    planning_dir = tmp_path / "planning"
    figures_dir = tmp_path / "figures_tables"
    audit_dir = tmp_path / "audit"
    for directory in (planning_dir, figures_dir, audit_dir):
        directory.mkdir()

    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "allocated_feed_ton_per_year": 89.3},
            {"scenario_name": "baseline_region_case", "pathway": "htc", "allocated_feed_ton_per_year": 10.7},
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "portfolio_cost_objective": 2_950_000.0,
            }
        ]
    ).to_csv(planning_dir / "portfolio_summary.csv", index=False)
    fresh_rows = pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "baseline_portfolio_share_pct": 89.3},
            {"scenario_name": "baseline_region_case", "pathway": "htc", "baseline_portfolio_share_pct": 10.7},
            {"scenario_name": "baseline_region_case", "pathway": "ad", "baseline_portfolio_share_pct": 0.0},
        ]
    )
    fresh_rows.to_csv(planning_dir / "main_results_table.csv", index=False)
    fresh_rows.to_csv(figures_dir / "paper1_planning_results_table.csv", index=False)
    fresh_rows.to_csv(audit_dir / "planning_claim_flag_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "diagnostic": "Declared asymmetric baseline",
                "baseline_region": "P 90.0 / H 10.0",
                "high_supply": "--",
                "policy_support": "--",
            }
        ]
    ).to_csv(figures_dir / "paper1_driver_decomposition_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "boundary_regime": "Declared asymmetric credit + diversification rule",
                "baseline_region": "P 89.3 / H 10.7",
                "high_supply": "--",
                "policy_support": "--",
            }
        ]
    ).to_csv(figures_dir / "paper1_core_boundary_regime_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "baseline-region",
                "final_net_cost_musd_per_year": 1.00,
            }
        ]
    ).to_csv(figures_dir / "paper1_policy_cost_decomposition_table.csv", index=False)

    summary = build_planning_artifact_consistency_summary(
        planning_dir,
        figures_dir=figures_dir,
        audit_dir=audit_dir,
        tolerance_pct_point=0.1,
    )

    failed = summary[summary["consistency_status"] == "fail"]
    assert {
        "paper1_driver_decomposition_table.csv",
        "paper1_policy_cost_decomposition_table.csv",
    }.issubset(set(failed["artifact_label"]))
    driver_pyrolysis = failed[
        (failed["artifact_label"] == "paper1_driver_decomposition_table.csv")
        & (failed["scenario_name"] == "baseline_region_case")
        & (failed["pathway"] == "pyrolysis")
    ].iloc[0]
    assert driver_pyrolysis["expected_share_pct"] == 89.3
    assert driver_pyrolysis["observed_share_pct"] == 90.0
    cost_row = failed[failed["artifact_label"] == "paper1_policy_cost_decomposition_table.csv"].iloc[0]
    assert cost_row["pathway"] == "portfolio_cost_musd_per_year"
    assert cost_row["expected_share_pct"] == 2.95
    assert cost_row["observed_share_pct"] == 1.0


def test_build_benchmark_claim_summary_flags_core_innovation_when_pathways_shift(tmp_path):
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()

    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "benchmark_variant": "baseline_evidence_aware",
                "selected_pathways": "pyrolysis|htc",
                "portfolio_score_mass": 100.0,
                "portfolio_carbon_load_kgco2e": 50.0,
                "scenario_feed_coverage_ratio": 1.0,
            },
            {
                "scenario_name": "baseline_region_case",
                "benchmark_variant": "no_evidence_penalty",
                "selected_pathways": "pyrolysis",
                "portfolio_score_mass": 110.0,
                "portfolio_carbon_load_kgco2e": 60.0,
                "scenario_feed_coverage_ratio": 1.0,
            },
        ]
    ).to_csv(benchmark_dir / "benchmark_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "benchmark_variant": "no_evidence_penalty",
                "portfolio_case_shift": "changed",
                "portfolio_pathway_shift": "changed",
                "comparator_family": "counterfactual_optimizer",
                "allocation_mode": "optimizer",
                "baseline_selected_pathways": "pyrolysis|htc",
                "variant_selected_pathways": "pyrolysis",
                "baseline_top_portfolio_case_id": "case-a",
                "variant_top_portfolio_case_id": "case-b",
                "delta_portfolio_score_mass": 10.0,
                "delta_portfolio_carbon_load_kgco2e": 10.0,
                "delta_scenario_feed_coverage_ratio": 0.0,
            }
        ]
    ).to_csv(benchmark_dir / "benchmark_shift_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "benchmark_variant": "no_evidence_penalty",
                "bootstrap_replicate_count": 4,
                "pathway_shift_count": 4,
                "pathway_shift_rate": 1.0,
                "pathway_shift_rate_ci_lower": 0.51,
                "pathway_shift_rate_ci_upper": 1.0,
                "case_shift_count": 4,
                "case_shift_rate": 1.0,
                "case_shift_rate_ci_lower": 0.51,
                "case_shift_rate_ci_upper": 1.0,
                "delta_portfolio_score_mass_median": 9.0,
                "delta_portfolio_score_mass_ci_lower": 5.0,
                "delta_portfolio_score_mass_ci_upper": 12.0,
                "delta_portfolio_score_mass_ci_excludes_zero": True,
                "delta_portfolio_score_mass_sign_agreement_rate": 1.0,
                "delta_portfolio_score_mass_empirical_p_value": 0.0,
                "delta_portfolio_score_mass_direction": "positive",
                "delta_portfolio_carbon_load_kgco2e_median": 9.0,
                "delta_portfolio_carbon_load_kgco2e_ci_lower": 5.0,
                "delta_portfolio_carbon_load_kgco2e_ci_upper": 15.0,
                "delta_portfolio_carbon_load_kgco2e_ci_excludes_zero": True,
                "delta_portfolio_carbon_load_kgco2e_sign_agreement_rate": 1.0,
                "delta_portfolio_carbon_load_kgco2e_empirical_p_value": 0.0,
                "delta_portfolio_carbon_load_kgco2e_direction": "positive",
                "delta_scenario_feed_coverage_ratio_median": 0.0,
                "delta_scenario_feed_coverage_ratio_ci_lower": 0.0,
                "delta_scenario_feed_coverage_ratio_ci_upper": 0.0,
                "delta_scenario_feed_coverage_ratio_ci_excludes_zero": False,
                "delta_scenario_feed_coverage_ratio_sign_agreement_rate": 0.0,
                "delta_scenario_feed_coverage_ratio_empirical_p_value": 1.0,
                "delta_scenario_feed_coverage_ratio_direction": "mixed",
                "effect_significance_tier": "highly_consistent",
            }
        ]
    ).to_csv(benchmark_dir / "benchmark_statistical_summary.csv", index=False)

    summary = build_benchmark_claim_summary(benchmark_dir)
    row = summary.iloc[0]

    assert row["necessity_tier"] == "supports_core_innovation"
    assert "evidence-aware design" in row["necessity_note"]
    assert "changed the selected pathways" in row["manuscript_sentence"]
    assert row["effect_significance_tier"] == "highly_consistent"
    assert row["pathway_shift_rate_ci_lower"] == 0.51
    assert row["delta_portfolio_score_mass_empirical_p_value"] == 0.0


def test_build_benchmark_manuscript_sentences_aggregate_variant_effects(tmp_path):
    benchmark_dir = tmp_path / "benchmark_sentence"
    benchmark_dir.mkdir()

    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "benchmark_variant": "baseline_evidence_aware",
                "selected_pathways": "pyrolysis|htc",
                "portfolio_score_mass": 100.0,
                "portfolio_carbon_load_kgco2e": 50.0,
                "scenario_feed_coverage_ratio": 1.0,
            },
            {
                "scenario_name": "high_supply_case",
                "benchmark_variant": "baseline_evidence_aware",
                "selected_pathways": "htc",
                "portfolio_score_mass": 120.0,
                "portfolio_carbon_load_kgco2e": 40.0,
                "scenario_feed_coverage_ratio": 0.9,
            },
            {
                "scenario_name": "baseline_region_case",
                "benchmark_variant": "no_robustness_penalty",
                "selected_pathways": "pyrolysis",
                "portfolio_score_mass": 103.0,
                "portfolio_carbon_load_kgco2e": 55.0,
                "scenario_feed_coverage_ratio": 1.0,
            },
            {
                "scenario_name": "high_supply_case",
                "benchmark_variant": "no_robustness_penalty",
                "selected_pathways": "htc",
                "portfolio_score_mass": 121.0,
                "portfolio_carbon_load_kgco2e": 40.0,
                "scenario_feed_coverage_ratio": 0.9,
            },
        ]
    ).to_csv(benchmark_dir / "benchmark_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "benchmark_variant": "no_robustness_penalty",
                "portfolio_case_shift": "changed",
                "portfolio_pathway_shift": "changed",
                "comparator_family": "counterfactual_optimizer",
                "allocation_mode": "optimizer",
                "baseline_selected_pathways": "pyrolysis|htc",
                "variant_selected_pathways": "pyrolysis",
                "baseline_top_portfolio_case_id": "case-a",
                "variant_top_portfolio_case_id": "case-b",
                "delta_portfolio_score_mass": 3.0,
                "delta_portfolio_carbon_load_kgco2e": 5.0,
                "delta_scenario_feed_coverage_ratio": 0.0,
            },
            {
                "scenario_name": "high_supply_case",
                "benchmark_variant": "no_robustness_penalty",
                "portfolio_case_shift": "unchanged",
                "portfolio_pathway_shift": "unchanged",
                "comparator_family": "counterfactual_optimizer",
                "allocation_mode": "optimizer",
                "baseline_selected_pathways": "htc",
                "variant_selected_pathways": "htc",
                "baseline_top_portfolio_case_id": "case-c",
                "variant_top_portfolio_case_id": "case-c",
                "delta_portfolio_score_mass": 1.5,
                "delta_portfolio_carbon_load_kgco2e": 0.0,
                "delta_scenario_feed_coverage_ratio": 0.0,
            },
        ]
    ).to_csv(benchmark_dir / "benchmark_shift_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "benchmark_variant": "no_robustness_penalty",
                "bootstrap_replicate_count": 4,
                "pathway_shift_count": 4,
                "pathway_shift_rate": 1.0,
                "pathway_shift_rate_ci_lower": 0.51,
                "pathway_shift_rate_ci_upper": 1.0,
                "case_shift_count": 4,
                "case_shift_rate": 1.0,
                "case_shift_rate_ci_lower": 0.51,
                "case_shift_rate_ci_upper": 1.0,
                "delta_portfolio_score_mass_median": 2.5,
                "delta_portfolio_score_mass_ci_lower": 1.0,
                "delta_portfolio_score_mass_ci_upper": 4.0,
                "delta_portfolio_score_mass_ci_excludes_zero": True,
                "delta_portfolio_score_mass_sign_agreement_rate": 1.0,
                "delta_portfolio_score_mass_empirical_p_value": 0.0,
                "delta_portfolio_score_mass_direction": "positive",
                "delta_portfolio_carbon_load_kgco2e_median": 4.0,
                "delta_portfolio_carbon_load_kgco2e_ci_lower": 2.0,
                "delta_portfolio_carbon_load_kgco2e_ci_upper": 6.0,
                "delta_portfolio_carbon_load_kgco2e_ci_excludes_zero": True,
                "delta_portfolio_carbon_load_kgco2e_sign_agreement_rate": 1.0,
                "delta_portfolio_carbon_load_kgco2e_empirical_p_value": 0.0,
                "delta_portfolio_carbon_load_kgco2e_direction": "positive",
                "delta_scenario_feed_coverage_ratio_median": 0.0,
                "delta_scenario_feed_coverage_ratio_ci_lower": 0.0,
                "delta_scenario_feed_coverage_ratio_ci_upper": 0.0,
                "delta_scenario_feed_coverage_ratio_ci_excludes_zero": False,
                "delta_scenario_feed_coverage_ratio_sign_agreement_rate": 0.0,
                "delta_scenario_feed_coverage_ratio_empirical_p_value": 1.0,
                "delta_scenario_feed_coverage_ratio_direction": "mixed",
                "effect_significance_tier": "highly_consistent",
            },
            {
                "scenario_name": "high_supply_case",
                "benchmark_variant": "no_robustness_penalty",
                "bootstrap_replicate_count": 4,
                "pathway_shift_count": 1,
                "pathway_shift_rate": 0.25,
                "pathway_shift_rate_ci_lower": 0.05,
                "pathway_shift_rate_ci_upper": 0.70,
                "case_shift_count": 2,
                "case_shift_rate": 0.5,
                "case_shift_rate_ci_lower": 0.15,
                "case_shift_rate_ci_upper": 0.85,
                "delta_portfolio_score_mass_median": 1.0,
                "delta_portfolio_score_mass_ci_lower": 0.5,
                "delta_portfolio_score_mass_ci_upper": 2.0,
                "delta_portfolio_score_mass_ci_excludes_zero": True,
                "delta_portfolio_score_mass_sign_agreement_rate": 0.75,
                "delta_portfolio_score_mass_empirical_p_value": 0.0,
                "delta_portfolio_score_mass_direction": "positive",
                "delta_portfolio_carbon_load_kgco2e_median": 0.0,
                "delta_portfolio_carbon_load_kgco2e_ci_lower": 0.0,
                "delta_portfolio_carbon_load_kgco2e_ci_upper": 0.0,
                "delta_portfolio_carbon_load_kgco2e_ci_excludes_zero": False,
                "delta_portfolio_carbon_load_kgco2e_sign_agreement_rate": 0.0,
                "delta_portfolio_carbon_load_kgco2e_empirical_p_value": 1.0,
                "delta_portfolio_carbon_load_kgco2e_direction": "mixed",
                "delta_scenario_feed_coverage_ratio_median": 0.0,
                "delta_scenario_feed_coverage_ratio_ci_lower": 0.0,
                "delta_scenario_feed_coverage_ratio_ci_upper": 0.0,
                "delta_scenario_feed_coverage_ratio_ci_excludes_zero": False,
                "delta_scenario_feed_coverage_ratio_sign_agreement_rate": 0.0,
                "delta_scenario_feed_coverage_ratio_empirical_p_value": 1.0,
                "delta_scenario_feed_coverage_ratio_direction": "mixed",
                "effect_significance_tier": "suggestive",
            },
        ]
    ).to_csv(benchmark_dir / "benchmark_statistical_summary.csv", index=False)

    claim_summary = build_benchmark_claim_summary(benchmark_dir)
    sentences = build_benchmark_manuscript_sentences(claim_summary)
    row = sentences.iloc[0]

    assert row["supports_core_innovation_count"] == 1
    assert row["changed_pathway_count"] == 1
    assert row["supports_secondary_innovation_count"] == 1
    assert "robustness-aware design" in row["manuscript_sentence"]


def test_pathway_reliability_summary_uses_evidence_thresholds_without_pathway_bias():
    ml_flags = pd.DataFrame(
        [
            {"summary_label": "leave_study_out", "dataset_key": "htc_direct", "target_column": "a", "claim_status": "unsupported"},
            {"summary_label": "leave_study_out", "dataset_key": "htc_direct", "target_column": "b", "claim_status": "weak"},
            {"summary_label": "leave_study_out", "dataset_key": "htc_direct", "target_column": "c", "claim_status": "supportive"},
            {"summary_label": "leave_study_out", "dataset_key": "pyrolysis_direct", "target_column": "a", "claim_status": "supportive"},
            {"summary_label": "leave_study_out", "dataset_key": "pyrolysis_direct", "target_column": "b", "claim_status": "weak"},
            {"summary_label": "leave_study_out", "dataset_key": "pyrolysis_direct", "target_column": "c", "claim_status": "unsupported"},
            {"summary_label": "leave_study_out", "dataset_key": "ad_proxy", "target_column": "a", "claim_status": "unsupported"},
            {"summary_label": "leave_study_out", "dataset_key": "ad_proxy", "target_column": "b", "claim_status": "unsupported"},
            {"summary_label": "leave_study_out", "dataset_key": "ad_proxy", "target_column": "c", "claim_status": "weak"},
        ]
    )

    summary = build_pathway_reliability_summary(ml_flags)
    htc = summary.loc[summary["pathway"] == "htc"].iloc[0]
    pyrolysis = summary.loc[summary["pathway"] == "pyrolysis"].iloc[0]
    ad = summary.loc[summary["pathway"] == "ad"].iloc[0]

    assert htc["reliability_tier"] == "limited_support"
    assert pyrolysis["reliability_tier"] == "limited_support"
    assert htc["reviewer_restriction_sentence"] == pyrolysis["reviewer_restriction_sentence"]
    assert ad["reliability_tier"] == "auxiliary_only"
    assert "does not support strong generalization" in ad["reviewer_restriction_sentence"]


def test_planning_claim_flags_merge_pathway_reliability_and_evidence_ceiling(tmp_path):
    planning_dir = tmp_path / "planning"
    scenario_dir = tmp_path / "scenario"
    planning_dir.mkdir()
    scenario_dir.mkdir()

    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "writing_label": "supporting portfolio",
                "selected_in_baseline_portfolio": True,
                "baseline_portfolio_share_pct": 87.6,
                "max_stress_selection_rate": 37.5,
                "stress_tests_supporting_pathway": "baseline|environment_priority",
                "best_case_score_index": 0.913,
                "claim_boundary": "planning-ready candidate with blended-feed caution",
                "surrogate_support_level": "trained_surrogate_with_documented_fallback",
                "score_gap_to_scenario_best_pct": 10.0,
            }
        ]
    ).to_csv(planning_dir / "main_results_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "portfolio_selected_count": 1,
                "portfolio_allocated_feed_share": 0.876,
                "portfolio_top_case_id": "case-1",
            }
        ]
    ).to_csv(planning_dir / "pathway_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "surrogate_support_level": "trained_surrogate_with_documented_fallback",
            }
        ]
    ).to_csv(planning_dir / "scored_cases.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "surrogate_support_level": "trained_surrogate_with_documented_fallback",
            }
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "evidence_source": "test",
            }
        ]
    ).to_csv(planning_dir / "scenario_external_evidence.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "top_portfolio_case_id": "case-1",
            }
        ]
    ).to_csv(scenario_dir / "stress_test_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "surrogate_support_level": "trained_surrogate_with_documented_fallback",
                "recommendation_confidence_score": 0.64,
                "recommendation_confidence_tier": "moderate",
                "recommendation_confidence_note": "Selected under current constraints, but either stress persistence or evidence maturity remains partial.",
            }
        ]
    ).to_csv(planning_dir / "recommendation_confidence_summary.csv", index=False)

    reliability = pd.DataFrame(
        [
            {
                "pathway": "pyrolysis",
                "reliability_score": 0.625,
                "reliability_tier": "conditional_support",
                "reviewer_restriction_sentence": "Cross-study evidence remains pathway-specific and should be written with claim discipline.",
            }
        ]
    )

    flags = build_planning_claim_flag_table(planning_dir, scenario_dir, pathway_reliability=reliability)
    row = flags.iloc[0]

    assert row["reliability_tier"] == "conditional_support"
    assert row["recommendation_evidence_ceiling"] in {
        "conditional_transfer_supported",
        "conditional_transfer_caution",
    }


def test_planning_claim_flag_table_preserves_uq_fields(tmp_path):
    planning_dir = tmp_path / "planning_uq_flags"
    scenario_dir = tmp_path / "scenario_uq_flags"
    planning_dir.mkdir()
    scenario_dir.mkdir()

    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "writing_label": "dominant baseline portfolio pathway",
                "selected_in_baseline_portfolio": True,
                "baseline_portfolio_share_pct": 100.0,
                "max_stress_selection_rate": 80.0,
                "stress_tests_supporting_pathway": 8,
                "uq_stress_support": "interval_supported",
                "max_uq_stress_selection_rate": 66.7,
                "uncertainty_mode_sensitivity": "case-sensitive_pathway-stable",
                "uncertainty_mode_case_switch_count": 2,
                "uncertainty_mode_pathway_switch_count": 1,
                "best_case_uq_ranking_note": "Top case changes across UQ modes, but pathway identity remains pyrolysis.",
                "best_case_score_index": 0.91,
                "claim_boundary": "planning-ready candidate with evidence-qualified scope",
                "results_sentence": "Pyrolysis remains selected under the exported UQ modes.",
            }
        ]
    ).to_csv(planning_dir / "main_results_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "portfolio_selected_count": 1,
                "portfolio_allocated_feed_share": 1.0,
                "portfolio_top_case_id": "case-1",
            }
        ]
    ).to_csv(planning_dir / "pathway_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "surrogate_support_level": "surrogate_supported",
                "recommendation_confidence_score": 0.72,
                "recommendation_confidence_tier": "moderate",
                "recommendation_confidence_note": "Selected across the exported stress settings.",
            }
        ]
    ).to_csv(planning_dir / "recommendation_confidence_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "surrogate_support_level": "surrogate_supported",
            }
        ]
    ).to_csv(planning_dir / "scored_cases.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "top_portfolio_case_id": "case-1",
            }
        ]
    ).to_csv(scenario_dir / "stress_test_summary.csv", index=False)

    flags = build_planning_claim_flag_table(planning_dir, scenario_dir)
    row = flags.iloc[0]

    assert row["uq_stress_support"] == "interval_supported"
    assert row["max_uq_stress_selection_rate"] == 66.7
    assert row["uncertainty_mode_sensitivity"] == "case-sensitive_pathway-stable"
    assert row["uncertainty_mode_case_switch_count"] == 2
    assert row["uncertainty_mode_pathway_switch_count"] == 1
    assert "pathway identity remains pyrolysis" in row["best_case_uq_ranking_note"]


def test_planning_transferability_risk_summary_flags_auxiliary_share(tmp_path):
    planning_dir = tmp_path / "planning"
    planning_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "portfolio_allocated_feed_share": 0.7,
                "portfolio_selected_count": 1,
            },
            {
                "scenario_name": "baseline_region_case",
                "pathway": "htc",
                "portfolio_allocated_feed_share": 0.3,
                "portfolio_selected_count": 1,
            },
        ]
    ).to_csv(planning_dir / "pathway_summary.csv", index=False)

    reliability = pd.DataFrame(
        [
            {
                "pathway": "pyrolysis",
                "reliability_score": 0.625,
                "reliability_tier": "conditional_support",
                "reviewer_restriction_sentence": "Cross-study evidence remains pathway-specific and should be written with claim discipline.",
            },
            {
                "pathway": "htc",
                "reliability_score": 0.25,
                "reliability_tier": "auxiliary_only",
                "reviewer_restriction_sentence": "The findings for HTC are auxiliary and lack cross-study generalizability.",
            },
        ]
    )

    summary = build_planning_transferability_risk_summary(planning_dir, reliability)
    row = summary.iloc[0]

    assert row["auxiliary_transfer_share"] == 0.3
    assert row["transferability_evidence_ceiling"] == "auxiliary_or_missing_bounded"


def test_build_ml_best_result_summary_prefers_selected_manifest_identity(tmp_path):
    summary = pd.DataFrame(
        [
            {
                "dataset_key": "demo",
                "target_column": "target_a",
                "model_key": "xgboost",
                "test_r2": 0.95,
                "test_rmse": 1.0,
                "test_mae": 0.8,
            },
            {
                "dataset_key": "demo",
                "target_column": "target_a",
                "model_key": "rf",
                "test_r2": 0.70,
                "test_rmse": 2.0,
                "test_mae": 1.5,
            },
        ]
    )
    summary_path = tmp_path / "strict_summary.csv"
    summary.to_csv(summary_path, index=False)

    selected_manifest = pd.DataFrame(
        [
            {
                "dataset_key": "demo",
                "target_column": "target_a",
                "split_strategy": "strict_group",
                "selected_model_key": "rf",
                "selection_metric_name": "validation_r2",
                "selection_metric_value": 0.82,
                "selected_test_r2": 0.70,
                "selected_test_rmse": 2.0,
                "selected_test_mae": 1.5,
            }
        ]
    )
    selected_manifest_path = tmp_path / "strict_selected_manifest.csv"
    selected_manifest.to_csv(selected_manifest_path, index=False)

    result = build_ml_best_result_summary(
        {"strict_group": summary_path},
        {"strict_group": selected_manifest_path},
    )

    row = result.iloc[0]
    assert row["best_model_key"] == "rf"
    assert row["selection_metric_name"] == "validation_r2"
    assert row["best_test_r2"] == 0.70


def test_build_ml_claim_flag_table_uses_selected_manifest_identity_with_test_thresholds(tmp_path):
    strict_summary = pd.DataFrame(
        [
            {"dataset_key": "demo", "target_column": "target_a", "model_key": "xgboost", "test_r2": 0.95},
            {"dataset_key": "demo", "target_column": "target_a", "model_key": "rf", "test_r2": 0.40},
        ]
    )
    strict_summary_path = tmp_path / "strict_summary.csv"
    strict_summary.to_csv(strict_summary_path, index=False)

    strict_manifest = pd.DataFrame(
        [
            {
                "dataset_key": "demo",
                "target_column": "target_a",
                "split_strategy": "strict_group",
                "selected_model_key": "rf",
                "selection_metric_name": "validation_r2",
                "selection_metric_value": 0.80,
                "selected_test_r2": 0.40,
                "selected_test_rmse": 2.0,
                "selected_test_mae": 1.5,
            }
        ]
    )
    strict_manifest_path = tmp_path / "strict_selected_manifest.csv"
    strict_manifest.to_csv(strict_manifest_path, index=False)

    leave_summary_path = tmp_path / "leave_summary.csv"
    pd.DataFrame(columns=["dataset_key", "target_column", "model_key", "test_r2"]).to_csv(
        leave_summary_path, index=False
    )
    leave_manifest_path = tmp_path / "leave_selected_manifest.csv"
    pd.DataFrame(
        columns=[
            "dataset_key",
            "target_column",
            "split_strategy",
            "selected_model_key",
            "selection_metric_name",
            "selection_metric_value",
            "selected_test_r2",
        ]
    ).to_csv(leave_manifest_path, index=False)

    result = build_ml_claim_flag_table(
        {"strict_group": strict_summary_path, "leave_study_out": leave_summary_path},
        {"strict_group": strict_manifest_path, "leave_study_out": leave_manifest_path},
    )

    row = result.loc[result["summary_label"] == "strict_group"].iloc[0]
    assert row["best_model_key"] == "rf"
    assert row["claim_status"] == "weak"
    assert row["selection_metric_name"] == "validation_r2"


def test_build_ml_best_result_summary_recovers_validation_selected_model_from_summary_when_manifest_missing(tmp_path):
    summary = pd.DataFrame(
        [
            {
                "dataset_key": "demo",
                "target_column": "target_a",
                "model_key": "xgboost",
                "validation_r2": 0.81,
                "validation_rmse": 1.1,
                "validation_mae": 0.9,
                "test_r2": 0.65,
                "test_rmse": 1.8,
                "test_mae": 1.4,
            },
            {
                "dataset_key": "demo",
                "target_column": "target_a",
                "model_key": "rf",
                "validation_r2": 0.92,
                "validation_rmse": 0.9,
                "validation_mae": 0.7,
                "test_r2": 0.55,
                "test_rmse": 2.1,
                "test_mae": 1.6,
            },
        ]
    )
    summary_path = tmp_path / "strict_summary_no_manifest.csv"
    summary.to_csv(summary_path, index=False)

    result = build_ml_best_result_summary({"strict_group": summary_path})

    row = result.iloc[0]
    assert row["best_model_key"] == "rf"
    assert row["selection_metric_name"] == "validation_r2"
    assert row["selection_metric_value"] == 0.92
    assert row["best_test_r2"] == 0.55


def test_build_ml_claim_flag_table_recovers_validation_selected_model_from_summary_when_manifest_missing(tmp_path):
    strict_summary = pd.DataFrame(
        [
            {
                "dataset_key": "demo",
                "target_column": "target_a",
                "model_key": "xgboost",
                "validation_r2": 0.75,
                "validation_rmse": 1.1,
                "validation_mae": 0.8,
                "test_r2": 0.91,
            },
            {
                "dataset_key": "demo",
                "target_column": "target_a",
                "model_key": "rf",
                "validation_r2": 0.88,
                "validation_rmse": 0.9,
                "validation_mae": 0.7,
                "test_r2": 0.40,
            },
        ]
    )
    strict_summary_path = tmp_path / "strict_summary_no_manifest.csv"
    strict_summary.to_csv(strict_summary_path, index=False)

    leave_summary_path = tmp_path / "leave_summary.csv"
    pd.DataFrame(columns=["dataset_key", "target_column", "model_key", "test_r2"]).to_csv(
        leave_summary_path, index=False
    )

    result = build_ml_claim_flag_table(
        {"strict_group": strict_summary_path, "leave_study_out": leave_summary_path},
    )

    row = result.loc[result["summary_label"] == "strict_group"].iloc[0]
    assert row["best_model_key"] == "rf"
    assert row["selection_metric_name"] == "validation_r2"
    assert row["claim_status"] == "weak"


def test_build_ml_claim_flag_table_augments_leave_study_out_with_htc_benchmark_compare(monkeypatch, tmp_path):
    strict_summary = pd.DataFrame(columns=["dataset_key", "target_column", "model_key", "test_r2"])
    leave_summary = pd.DataFrame(columns=["dataset_key", "target_column", "model_key", "test_r2"])
    strict_summary_path = tmp_path / "strict_summary.csv"
    leave_summary_path = tmp_path / "leave_summary.csv"
    strict_summary.to_csv(strict_summary_path, index=False)
    leave_summary.to_csv(leave_summary_path, index=False)

    benchmark_root = tmp_path / "benchmark" / "htc_model_compare_lso"
    benchmark_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "dataset_key": "htc_direct",
                "target_column": "product_char_yield_pct",
                "model_key": "catboost",
                "validation_r2": 0.44,
                "test_rmse": 1.5,
                "test_mae": 1.1,
                "test_r2": 0.41,
            },
            {
                "dataset_key": "htc_direct",
                "target_column": "product_char_yield_pct",
                "model_key": "lightgbm",
                "validation_r2": 0.53,
                "test_rmse": 1.7,
                "test_mae": 1.2,
                "test_r2": 0.21,
            },
        ]
    ).to_csv(benchmark_root / "traditional_ml_suite_summary_leave_study_out.csv", index=False)

    monkeypatch.setattr("waste2energy.audit.BENCHMARK_OUTPUTS_DIR", tmp_path / "benchmark")

    result = build_ml_claim_flag_table(
        {"strict_group": strict_summary_path, "leave_study_out": leave_summary_path},
    )

    row = result.loc[result["summary_label"] == "leave_study_out"].iloc[0]
    assert row["best_model_key"] == "catboost"
    assert row["claim_status"] == "weak"


def test_build_ml_claim_flag_table_uses_benchmark_metric_from_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr("waste2energy.audit.BENCHMARK_OUTPUTS_DIR", tmp_path / "empty_benchmark")
    strict_summary = pd.DataFrame(columns=["dataset_key", "target_column", "model_key", "test_r2"])
    leave_summary = pd.DataFrame(
        [
            {
                "dataset_key": "htc_direct",
                "target_column": "energy_recovery_pct",
                "model_key": "catboost",
                "validation_r2": 0.16,
                "test_r2": 0.004,
                "test_rmse": 1.0,
                "test_mae": 0.8,
            }
        ]
    )
    manifest = pd.DataFrame(
        [
            {
                "dataset_key": "htc_direct",
                "target_column": "energy_recovery_pct",
                "selected_model_key": "catboost",
                "selection_metric_name": "validation_r2",
                "selection_metric_value": 0.16,
                "benchmark_test_r2": 0.004,
                "selected_test_r2": -0.226,
                "benchmark_test_rmse": 1.0,
                "selected_test_rmse": 2.0,
                "benchmark_test_mae": 0.8,
                "selected_test_mae": 1.8,
            }
        ]
    )
    strict_summary_path = tmp_path / "strict_summary.csv"
    leave_summary_path = tmp_path / "leave_summary.csv"
    manifest_path = tmp_path / "selected_leave.csv"
    strict_summary.to_csv(strict_summary_path, index=False)
    leave_summary.to_csv(leave_summary_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    result = build_ml_claim_flag_table(
        {"strict_group": strict_summary_path, "leave_study_out": leave_summary_path},
        {"leave_study_out": manifest_path},
    )

    row = result.loc[result["summary_label"] == "leave_study_out"].iloc[0]
    assert row["best_test_r2"] == pytest.approx(0.004)
    assert row["claim_status"] == "weak"


def test_build_ml_claim_flag_table_keeps_summary_groups_missing_from_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr("waste2energy.audit.BENCHMARK_OUTPUTS_DIR", tmp_path / "empty_benchmark")
    strict_summary = pd.DataFrame(columns=["dataset_key", "target_column", "model_key", "test_r2"])
    leave_summary = pd.DataFrame(
        [
            {
                "dataset_key": "pyrolysis_direct",
                "target_column": "product_char_yield_pct",
                "model_key": "rf",
                "validation_r2": 0.75,
                "test_r2": 0.70,
                "test_rmse": 1.0,
                "test_mae": 0.8,
            }
        ]
    )
    manifest = pd.DataFrame(
        [
            {
                "dataset_key": "htc_direct",
                "target_column": "energy_recovery_pct",
                "selected_model_key": "catboost",
                "selection_metric_name": "validation_r2",
                "selection_metric_value": 0.16,
                "benchmark_test_r2": 0.004,
                "selected_test_r2": -0.226,
            }
        ]
    )
    strict_summary_path = tmp_path / "strict_summary.csv"
    leave_summary_path = tmp_path / "leave_summary.csv"
    manifest_path = tmp_path / "selected_leave.csv"
    strict_summary.to_csv(strict_summary_path, index=False)
    leave_summary.to_csv(leave_summary_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    result = build_ml_claim_flag_table(
        {"strict_group": strict_summary_path, "leave_study_out": leave_summary_path},
        {"leave_study_out": manifest_path},
    )

    pyrolysis = result[
        result["dataset_key"].eq("pyrolysis_direct")
        & result["target_column"].eq("product_char_yield_pct")
    ].iloc[0]
    assert pyrolysis["best_test_r2"] == pytest.approx(0.70)
    assert pyrolysis["claim_status"] == "supportive"


def test_build_ml_refit_provenance_summary_reports_complete_trace(tmp_path):
    run_config_path = tmp_path / "run_config.json"
    run_config_path.write_text(
        """
        {
          "model_config": {"random_state": 42},
          "dataset_version_label": "demo:demo.csv",
          "dataset_fingerprint": "abc123",
          "refit_config_source": "selected_benchmark_run_config"
        }
        """.strip(),
        encoding="utf-8",
    )
    manifest = pd.DataFrame(
        [
            {
                "dataset_key": "demo",
                "target_column": "target_a",
                "selected_model_key": "rf",
                "artifact_role": "selected_model_refit",
                "training_scope": "train_plus_validation",
                "selection_trace_id": "demo::target_a::strict_group::rf",
                "selection_evidence_source": "validation_selected_benchmark_then_refit",
                "selection_benchmark_manifest_path": "selected_models_manifest_benchmark_strict_group.csv",
                "selection_data_version": "demo:demo.csv",
                "selection_data_fingerprint": "abc123",
                "selection_random_state": 42,
                "benchmark_data_version": "demo:demo.csv",
                "benchmark_data_fingerprint": "abc123",
                "benchmark_random_state": 42,
                "refit_data_version": "demo:demo.csv",
                "refit_data_fingerprint": "def456",
                "refit_test_data_fingerprint": "ghi789",
                "refit_random_state": 42,
                "run_config_path": str(run_config_path),
            }
        ]
    )
    manifest_path = tmp_path / "selected_models_manifest_strict_group.csv"
    manifest.to_csv(manifest_path, index=False)

    summary = build_ml_refit_provenance_summary({"strict_group": manifest_path})
    row = summary.iloc[0]

    assert bool(row["provenance_complete"])
    assert row["missing_provenance_fields"] == ""
    assert row["selection_evidence_source"] == "validation_selected_benchmark_then_refit"
    assert row["run_config_random_state"] == 42


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


def test_planning_ml_consistency_reports_surrogate_feature_imputation_share(tmp_path):
    planning_dir = tmp_path / "planning_imputation"
    planning_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "portfolio_allocated_feed_share": 0.9,
            },
            {
                "scenario_name": "baseline_region_case",
                "pathway": "htc",
                "portfolio_allocated_feed_share": 0.1,
            },
        ]
    ).to_csv(planning_dir / "pathway_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "allocated_feed_ton_per_year": 90.0,
                "surrogate_support_level": "surrogate_supported",
                "surrogate_feature_imputation_flag": "True",
                "surrogate_imputed_feature_columns": "feedstock_hhv_mj_per_kg",
            },
            {
                "scenario_name": "baseline_region_case",
                "pathway": "htc",
                "allocated_feed_ton_per_year": 10.0,
                "surrogate_support_level": "surrogate_supported",
                "surrogate_feature_imputation_flag": "False",
                "surrogate_imputed_feature_columns": "",
            },
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)
    reliability = pd.DataFrame(
        [
            {"pathway": "pyrolysis", "reliability_score": 0.5},
            {"pathway": "htc", "reliability_score": 0.4},
        ]
    )

    summary = build_planning_ml_consistency_summary(planning_dir, reliability)
    row = summary.iloc[0]

    assert row["surrogate_supported_allocation_share"] == pytest.approx(1.0)
    assert row["surrogate_feature_imputed_allocation_share"] == pytest.approx(0.9)
    assert row["surrogate_supported_with_imputed_key_feature_allocation_share"] == pytest.approx(0.9)
    assert row["fully_observed_surrogate_supported_allocation_share"] == pytest.approx(0.1)
    assert row["surrogate_support_evidence_tier"] == "surrogate_supported_with_imputed_key_feature"
    assert row["surrogate_imputed_feature_columns"] == "feedstock_hhv_mj_per_kg"


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


def test_hhv_imputation_sensitivity_exports_imputed_key_feature_tier(tmp_path):
    planning_dir = tmp_path / "planning_hhv"
    planning_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "allocated_feed_ton_per_year": 90.0,
                "allocated_feed_share": 0.9,
                "sample_id": "P-1",
                "manure_subtype": "dairy",
                "feedstock_carbon_pct": 44.0,
                "feedstock_hydrogen_pct": 6.0,
                "feedstock_nitrogen_pct": 3.0,
                "feedstock_oxygen_pct": 34.0,
                "feedstock_ash_pct": 12.0,
                "surrogate_feature_imputation_flag": True,
                "surrogate_imputed_feature_columns": "feedstock_hhv_mj_per_kg",
            },
            {
                "scenario_name": "baseline_region_case",
                "pathway": "htc",
                "allocated_feed_ton_per_year": 10.0,
                "allocated_feed_share": 0.1,
                "sample_id": "H-1",
                "manure_subtype": "dairy",
                "surrogate_feature_imputation_flag": True,
                "surrogate_imputed_feature_columns": "some_other_feature",
            },
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)

    result = build_hhv_imputation_sensitivity(planning_dir)

    assert set(result["stress_case"]) == {
        "composition-derived baseline",
        "HHV imputation -10%",
        "HHV imputation -5%",
        "HHV imputation +5%",
        "HHV imputation +10%",
    }
    assert result["allocated_share_pct"].iloc[0] == pytest.approx(90.0)
    assert set(result["evidence_tier"]) == {"surrogate_supported_with_imputed_key_feature"}


def test_hhv_replanning_sensitivity_reruns_optimizer_with_perturbed_hhv(tmp_path, monkeypatch):
    from waste2energy.planning.inputs import PlanningInputBundle

    planning_dir = tmp_path / "planning_hhv_replan"
    planning_dir.mkdir()
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "allocated_feed_ton_per_year": 90.0, "sample_id": "P-base"},
            {"scenario_name": "baseline_region_case", "pathway": "htc", "allocated_feed_ton_per_year": 10.0, "sample_id": "H-base"},
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)
    (planning_dir / "run_config.json").write_text(
        json.dumps({"dataset_path": "dummy.csv", "planning_config": {"primary_optimization_pathways": ["pyrolysis", "htc"]}}),
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "sample_id": "P-stress",
                "feedstock_hhv_mj_per_kg": pd.NA,
                "feedstock_carbon_pct": 44.0,
                "feedstock_hydrogen_pct": 6.0,
                "feedstock_nitrogen_pct": 3.0,
                "feedstock_oxygen_pct": 34.0,
                "feedstock_ash_pct": 12.0,
            }
        ]
    )
    bundle = PlanningInputBundle(
        frame=frame,
        dataset_path=tmp_path / "dummy.csv",
        scenario_names=("baseline_region_case",),
        pathways=("pyrolysis", "htc"),
        real_cost_columns=(),
        surrogate_feature_columns=("feedstock_hhv_mj_per_kg",),
        unit_registry={},
    )

    monkeypatch.setattr(audit_module, "load_planning_input_bundle", lambda dataset_path=None: bundle)

    baseline_hhv = float(audit_module._derive_feedstock_hhv_from_ultimate_analysis(frame.iloc[0]))

    def fake_execute_planning_pipeline(*, bundle, config):
        stressed_hhv = float(pd.to_numeric(bundle.frame["feedstock_hhv_mj_per_kg"], errors="coerce").iloc[0])
        pyro_share = 95.0 if stressed_hhv > baseline_hhv else 85.0 if stressed_hhv < baseline_hhv else 90.0
        return {
            "portfolio_allocations": pd.DataFrame(
                [
                    {
                        "scenario_name": "baseline_region_case",
                        "pathway": "pyrolysis",
                        "allocated_feed_ton_per_year": pyro_share,
                        "sample_id": "P-stress",
                    },
                    {
                        "scenario_name": "baseline_region_case",
                        "pathway": "htc",
                        "allocated_feed_ton_per_year": 100.0 - pyro_share,
                        "sample_id": "H-stress",
                    },
                ]
            )
        }

    monkeypatch.setattr(audit_module, "execute_planning_pipeline", fake_execute_planning_pipeline)

    result = build_hhv_replanning_sensitivity(planning_dir)

    plus = result[
        (result["pathway"] == "pyrolysis")
        & (result["stress_case"] == "HHV replanning +10%")
    ].iloc[0]
    baseline = result[
        (result["pathway"] == "pyrolysis")
        & (result["stress_case"] == "HHV replanning baseline-derived")
    ].iloc[0]
    assert plus["replanning_status"] == "replanned"
    assert plus["share_change_pct_point"] == pytest.approx(5.0)
    assert baseline["share_change_pct_point"] == pytest.approx(0.0)


def test_hhv_dominance_audit_distinguishes_case_switch_from_pathway_dominance(tmp_path):
    planning_dir = tmp_path / "planning_hhv_dominance"
    planning_dir.mkdir()
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "allocated_feed_ton_per_year": 90.0},
            {"scenario_name": "baseline_region_case", "pathway": "htc", "allocated_feed_ton_per_year": 10.0},
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)
    imputation = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "stress_case": "composition-derived baseline",
                "allocated_share_pct": 90.0,
            }
        ]
    )
    replanning = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "stress_case": "HHV replanning +10%",
                "baseline_share_pct": 90.0,
                "stressed_share_pct": 89.4,
                "share_change_pct_point": -0.6,
                "baseline_selected_sample_ids": "P-1",
                "stressed_selected_sample_ids": "P-2",
                "replanning_status": "replanned",
            },
            {
                "scenario_name": "baseline_region_case",
                "pathway": "htc",
                "stress_case": "HHV replanning +10%",
                "baseline_share_pct": 10.0,
                "stressed_share_pct": 10.6,
                "share_change_pct_point": 0.6,
                "baseline_selected_sample_ids": "H-1",
                "stressed_selected_sample_ids": "H-1",
                "replanning_status": "replanned",
            },
        ]
    )

    result = build_hhv_dominance_audit(
        planning_dir,
        hhv_imputation_sensitivity=imputation,
        hhv_replanning_sensitivity=replanning,
    )
    row = result.iloc[0]

    assert row["affected_imputed_share_pct"] == pytest.approx(90.0)
    assert row["max_abs_pathway_share_change_pct_point"] == pytest.approx(0.6)
    assert bool(row["selected_case_changed"]) is True
    assert row["hhv_dominance_conclusion"] == "not_pathway_dominant_but_case_sensitive"


def test_binding_constraint_audit_reports_binding_and_cap_relaxation_shift(tmp_path):
    planning_dir = tmp_path / "planning_binding"
    benchmark_dir = tmp_path / "benchmark" / "baseline"
    ablation_dir = benchmark_dir / "targeted_planning_ablations"
    planning_dir.mkdir()
    ablation_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "allocated_feed_ton_per_year": 90.0,
            },
            {
                "scenario_name": "baseline_region_case",
                "pathway": "htc",
                "allocated_feed_ton_per_year": 10.0,
            },
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "candidate_cap_binding": True,
                "candidate_cap_slack_ton_per_year": 0.0,
                "subtype_cap_binding": False,
                "subtype_cap_slack_ton_per_year": 5.0,
                "carbon_budget_binding": False,
                "carbon_budget_slack_kgco2e": 1000.0,
                "min_distinct_subtypes_binding": True,
                "max_selected_binding": True,
            }
        ]
    ).to_csv(planning_dir / "optimization_diagnostics.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "ablation_key": "candidate_and_subtype_caps_relaxed",
                "pyrolysis_allocated_share_pct": 95.0,
                "htc_allocated_share_pct": 5.0,
            },
            {
                "scenario_name": "baseline_region_case",
                "ablation_key": "candidate_cap_relaxed_100pct",
                "max_candidate_allocated_share_pct": 55.0,
            },
        ]
    ).to_csv(ablation_dir / "portfolio_cap_diagnostics.csv", index=False)

    result = build_binding_constraint_audit(planning_dir, benchmark_dir)
    row = result.iloc[0]

    assert bool(row["candidate_cap_binding"]) is True
    assert bool(row["residual_carbon_constraint_binding"]) is False
    assert row["baseline_pyrolysis_share_pct"] == pytest.approx(90.0)
    assert row["cap_relaxed_pyrolysis_share_change_pct_point"] == pytest.approx(5.0)
    assert "candidate cap" in row["interpretation"]


def test_duplicate_candidate_audit_flags_same_pyrolysis_signature_across_subtypes(tmp_path):
    planning_dir = tmp_path / "planning_dup"
    planning_dir.mkdir()
    base = {
        "scenario_name": "baseline_region_case",
        "pathway": "pyrolysis",
        "allocated_feed_ton_per_year": 45.0,
        "allocated_feed_share": 0.45,
        "process_temperature_c": 400.0,
        "residence_time_min": 30.0,
        "heating_rate_c_per_min": 10.0,
        "predicted_product_char_yield_pct": 40.0,
        "predicted_product_char_hhv_mj_per_kg": 24.0,
        "predicted_energy_recovery_pct": 50.0,
        "predicted_carbon_retention_pct": 51.0,
        "feedstock_carbon_pct": 44.0,
        "feedstock_hydrogen_pct": 6.0,
        "feedstock_nitrogen_pct": 3.0,
        "feedstock_oxygen_pct": 34.0,
        "feedstock_ash_pct": 12.0,
    }
    pd.DataFrame(
        [
            {**base, "sample_id": "P-1", "manure_subtype": "dairy"},
            {**base, "sample_id": "P-2", "manure_subtype": "swine"},
            {**base, "sample_id": "H-1", "pathway": "htc", "manure_subtype": "dairy", "allocated_feed_ton_per_year": 10.0},
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)

    result = build_duplicate_candidate_audit(planning_dir)
    row = result.iloc[0]

    assert row["rows_in_group"] == 2
    assert row["distinct_subtypes"] == 2
    assert row["allocated_share_pct"] == pytest.approx(90.0)
    assert row["audit_finding"] == "duplicate_operating_and_target_signature"


def test_surrogate_extrapolation_audit_caps_weak_lso_claims_at_screening(tmp_path):
    planning_dir = tmp_path / "planning_surrogate_extrap"
    planning_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "allocated_feed_ton_per_year": 90.0,
                "feedstock_carbon_pct": 44.0,
                "process_temperature_c": 400.0,
                "residence_time_min": 30.0,
            }
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)
    ml_flags = pd.DataFrame(
        [
            {
                "summary_label": "leave_study_out",
                "dataset_key": "pyrolysis_direct",
                "target_column": "energy_recovery_pct",
                "best_test_r2": -0.01,
                "claim_status": "unsupported",
            },
            {
                "summary_label": "leave_study_out",
                "dataset_key": "pyrolysis_direct",
                "target_column": "product_char_yield_pct",
                "best_test_r2": 0.76,
                "claim_status": "supportive",
            },
        ]
    )
    training = tmp_path / "training.csv"
    pd.DataFrame(
        [
            {
                "pathway": "pyrolysis",
                "feedstock_carbon_pct": 40.0,
                "process_temperature_c": 350.0,
                "residence_time_min": 20.0,
            },
            {
                "pathway": "pyrolysis",
                "feedstock_carbon_pct": 50.0,
                "process_temperature_c": 500.0,
                "residence_time_min": 60.0,
            },
        ]
    ).to_csv(training, index=False)

    result = build_surrogate_extrapolation_audit(
        planning_dir,
        ml_flags=ml_flags,
        training_dataset_path=training,
    )
    row = result.iloc[0]

    assert row["weakest_leave_study_out_claim_status"] == "unsupported"
    assert row["min_leave_study_out_test_r2"] == pytest.approx(-0.01)
    assert bool(row["within_training_range_all_features"]) is True
    assert row["extrapolation_evidence_ceiling"] == "screening_only_external_validity_not_established"


def test_ad_boundary_fairness_audit_prevents_technical_inferiority_claim(tmp_path):
    planning_dir = tmp_path / "planning_ad_boundary"
    benchmark_dir = tmp_path / "benchmark" / "baseline"
    ablation_dir = benchmark_dir / "targeted_planning_ablations"
    planning_dir.mkdir()
    ablation_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "ad",
                "baseline_portfolio_share_pct": 0.0,
            }
        ]
    ).to_csv(planning_dir / "ad_reference_diagnostics.csv", index=False)
    pd.DataFrame(
        [
            {
                "ablation_family": "ad_complementarity",
                "ablation_key": "ad_min_share_10pct",
                "scenario_name": "baseline_region_case",
                "ad_allocated_share_pct": 18.0,
            },
            {
                "ablation_family": "ad_complementarity",
                "ablation_key": "ad_min_share_20pct",
                "scenario_name": "baseline_region_case",
                "ad_allocated_share_pct": 25.0,
            },
            {
                "ablation_family": "coproduct_boundary",
                "ablation_key": "digestate_rng_credit_300pct",
                "scenario_name": "baseline_region_case",
                "ad_allocated_share_pct": 0.0,
            },
        ]
    ).to_csv(ablation_dir / "targeted_planning_ablations_summary.csv", index=False)

    result = build_ad_boundary_fairness_audit(planning_dir, benchmark_dir)
    row = result.iloc[0]

    assert row["primary_optimizer_ad_share_pct"] == pytest.approx(0.0)
    assert row["ad_min_10pct_floor_share_pct"] == pytest.approx(18.0)
    assert bool(row["ad_policy_floor_feasible"]) is True
    assert row["ad_boundary_evidence_status"] == "evaluated"
    assert row["ad_role_conclusion"] == "boundary_reference_not_technical_inferiority"
    assert "not interpreted as technically inferior" in row["not_technical_inferiority_sentence"]


def test_ad_boundary_fairness_audit_flags_missing_ad_reference(tmp_path):
    planning_dir = tmp_path / "planning_ad_missing_reference"
    benchmark_dir = tmp_path / "benchmark" / "baseline"
    ablation_dir = benchmark_dir / "targeted_planning_ablations"
    planning_dir.mkdir()
    ablation_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ablation_family": "ad_complementarity",
                "ablation_key": "ad_min_share_10pct",
                "scenario_name": "baseline_region_case",
                "ad_allocated_share_pct": 18.0,
            },
            {
                "ablation_family": "ad_complementarity",
                "ablation_key": "ad_min_share_20pct",
                "scenario_name": "baseline_region_case",
                "ad_allocated_share_pct": 25.0,
            },
            {
                "ablation_family": "coproduct_boundary",
                "ablation_key": "digestate_rng_credit_300pct",
                "scenario_name": "baseline_region_case",
                "ad_allocated_share_pct": 0.0,
            },
        ]
    ).to_csv(ablation_dir / "targeted_planning_ablations_summary.csv", index=False)

    result = build_ad_boundary_fairness_audit(planning_dir, benchmark_dir)
    row = result.iloc[0]

    assert row["ad_boundary_evidence_status"] == "missing_ad_reference"
    assert row["ad_role_conclusion"] == "boundary_evidence_incomplete"
    assert pd.isna(row["primary_optimizer_ad_share_pct"])


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
            {"scenario_name": "high_supply_case", "top_ranked_case_id": "case-2"},
            {"scenario_name": "policy_support_case", "top_ranked_case_id": "case-3"},
        ]
    ).to_csv(planning_dir / "scenario_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "htc", "allocated_feed_ton_per_year": 75.0},
            {"scenario_name": "baseline_region_case", "pathway": "ad", "allocated_feed_ton_per_year": 25.0},
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "scenario_feed_coverage_ratio": 1.0},
            {"scenario_name": "high_supply_case", "scenario_feed_coverage_ratio": 0.9176628320631297},
            {"scenario_name": "policy_support_case", "scenario_feed_coverage_ratio": 1.0},
        ]
    ).to_csv(planning_dir / "portfolio_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "htc", "portfolio_allocated_feed_share": 1.0, "best_case_score": 1.30},
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "portfolio_allocated_feed_share": 0.0, "best_case_score": 0.80},
            {"scenario_name": "baseline_region_case", "pathway": "ad", "portfolio_allocated_feed_share": 0.0, "best_case_score": 0.10},
            {"scenario_name": "high_supply_case", "pathway": "htc", "portfolio_allocated_feed_share": 1.0, "best_case_score": 1.31},
            {"scenario_name": "high_supply_case", "pathway": "pyrolysis", "portfolio_allocated_feed_share": 0.0, "best_case_score": 0.79},
            {"scenario_name": "high_supply_case", "pathway": "ad", "portfolio_allocated_feed_share": 0.0, "best_case_score": 0.08},
            {"scenario_name": "policy_support_case", "pathway": "htc", "portfolio_allocated_feed_share": 1.0, "best_case_score": 1.32},
            {"scenario_name": "policy_support_case", "pathway": "pyrolysis", "portfolio_allocated_feed_share": 0.0, "best_case_score": 0.78},
            {"scenario_name": "policy_support_case", "pathway": "ad", "portfolio_allocated_feed_share": 0.0, "best_case_score": 0.09},
        ]
    ).to_csv(planning_dir / "pathway_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "htc", "baseline_portfolio_share_pct": 100.0, "max_stress_selection_rate": 100.0},
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "baseline_portfolio_share_pct": 0.0, "max_stress_selection_rate": 12.5},
            {"scenario_name": "baseline_region_case", "pathway": "ad", "baseline_portfolio_share_pct": 0.0, "max_stress_selection_rate": 0.0},
            {"scenario_name": "high_supply_case", "pathway": "htc", "baseline_portfolio_share_pct": 100.0, "max_stress_selection_rate": 100.0},
            {"scenario_name": "high_supply_case", "pathway": "pyrolysis", "baseline_portfolio_share_pct": 0.0, "max_stress_selection_rate": 12.5},
            {"scenario_name": "high_supply_case", "pathway": "ad", "baseline_portfolio_share_pct": 0.0, "max_stress_selection_rate": 0.0},
            {"scenario_name": "policy_support_case", "pathway": "htc", "baseline_portfolio_share_pct": 100.0, "max_stress_selection_rate": 87.5},
            {"scenario_name": "policy_support_case", "pathway": "pyrolysis", "baseline_portfolio_share_pct": 0.0, "max_stress_selection_rate": 12.5},
            {"scenario_name": "policy_support_case", "pathway": "ad", "baseline_portfolio_share_pct": 0.0, "max_stress_selection_rate": 0.0},
        ]
    ).to_csv(planning_dir / "main_results_table.csv", index=False)
    abstract_path = paper_dir / "00-abstract.tex"
    abstract_path.write_text("This abstract describes an AD-free portfolio.\n", encoding="utf-8")
    macros_path = paper_dir / "99-auto-macros.tex"

    result = sync_planning_summary_to_latex(
        planning_dir=planning_dir,
        abstract_path=abstract_path,
        macros_path=macros_path,
    )

    macros_text = macros_path.read_text(encoding="utf-8")

    assert result["ad_status_label"] == "AD-limited"
    assert result["dominant_pathway"] == "htc"
    assert result["dominance_pattern"] == "uniform"
    assert "AD-limited" in abstract_path.read_text(encoding="utf-8")
    assert "\\newcommand{\\PlanningDominantPathwayDisplay}{HTC}" in macros_text
    assert "\\newcommand{\\PlanningDominancePattern}{uniform}" in macros_text
    assert "\\newcommand{\\PlanningHighSupplyCoveragePct}{91.8\\%}" in macros_text
    assert "\\newcommand{\\PlanningPyrolysisRole}{stress-sensitive alternative}" in macros_text
    assert "AD remains a comparison-only pathway." in macros_text
    assert "\\newcommand{\\PlanningHighlightsDominanceBullet}{" in macros_text
    assert "HTC-dominant optimized baseline portfolio" in macros_text
    assert "pyrolysis remains a stress-sensitive alternative" in macros_text
    assert "\\newcommand{\\PlanningResultsDominanceSentence}{" in macros_text
    assert "the constrained portfolio is now HTC-dominant" in macros_text
    assert "\\newcommand{\\PlanningADStatus}{AD-limited}" in macros_text


def test_manuscript_sync_copies_canonical_planning_artifacts(tmp_path):
    planning_dir = tmp_path / "planning_artifacts"
    figures_dir = tmp_path / "figures_artifacts"
    planning_dir.mkdir()
    figures_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "htc",
                "baseline_portfolio_share_pct": 11.1,
            }
        ]
    ).to_csv(planning_dir / "main_results_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "htc",
                "baseline_portfolio_share_pct": 0.0,
            }
        ]
    ).to_csv(figures_dir / "paper1_planning_results_table.csv", index=False)

    manuscript_sync_module._sync_planning_result_artifacts(
        planning_dir=planning_dir,
        figures_dir=figures_dir,
    )

    synchronized = pd.read_csv(figures_dir / "paper1_planning_results_table.csv")
    assert synchronized.loc[0, "baseline_portfolio_share_pct"] == pytest.approx(11.1)


def test_manuscript_sync_filters_ad_from_primary_planning_artifacts(tmp_path):
    planning_dir = tmp_path / "planning_artifacts_ad"
    figures_dir = tmp_path / "figures_artifacts_ad"
    planning_dir.mkdir()
    figures_dir.mkdir()
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "baseline_portfolio_share_pct": 90.0},
            {"scenario_name": "baseline_region_case", "pathway": "htc", "baseline_portfolio_share_pct": 10.0},
            {"scenario_name": "baseline_region_case", "pathway": "ad", "baseline_portfolio_share_pct": 0.0},
        ]
    ).to_csv(planning_dir / "main_results_table.csv", index=False)

    manuscript_sync_module._sync_planning_result_artifacts(
        planning_dir=planning_dir,
        figures_dir=figures_dir,
    )

    synchronized = pd.read_csv(figures_dir / "paper1_planning_results_table.csv")
    assert set(synchronized["pathway"]) == {"pyrolysis", "htc"}


def test_monte_carlo_uq_table_excludes_ad_from_primary_summary(tmp_path):
    benchmark_dir = tmp_path / "benchmark" / "baseline"
    ablation_dir = benchmark_dir / "targeted_planning_ablations"
    ablation_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "monte_carlo_replicates": 10,
                "selection_probability": 1.0,
                "share_pct_median": 90.0,
                "share_pct_p05": 70.0,
                "share_pct_p95": 100.0,
                "cost_musd_per_year_median": 0.1,
                "carbon_ktco2e_per_year_median": 1.0,
            },
            {
                "scenario_name": "baseline_region_case",
                "pathway": "ad",
                "monte_carlo_replicates": 10,
                "selection_probability": 0.0,
                "share_pct_median": 0.0,
                "share_pct_p05": 0.0,
                "share_pct_p95": 0.0,
                "cost_musd_per_year_median": 0.1,
                "carbon_ktco2e_per_year_median": 1.0,
            },
        ]
    ).to_csv(ablation_dir / "monte_carlo_uq_summary.csv", index=False)
    pd.DataFrame([{"ablation_key": "placeholder"}]).to_csv(
        ablation_dir / "targeted_planning_ablations_summary.csv",
        index=False,
    )

    table = manuscript_sync_module._build_monte_carlo_uq_table(benchmark_dir=benchmark_dir)

    assert set(table["pathway"]) == {"pyrolysis"}


def test_manuscript_sync_writes_mixed_dominance_sentence_when_scenarios_diverge(tmp_path):
    planning_dir = tmp_path / "planning_sync_mixed"
    paper_dir = tmp_path / "paper"
    planning_dir.mkdir()
    paper_dir.mkdir()
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "top_ranked_case_id": "case-1"},
            {"scenario_name": "high_supply_case", "top_ranked_case_id": "case-2"},
            {"scenario_name": "policy_support_case", "top_ranked_case_id": "case-3"},
        ]
    ).to_csv(planning_dir / "scenario_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "htc", "allocated_feed_ton_per_year": 60.0},
            {"scenario_name": "high_supply_case", "pathway": "pyrolysis", "allocated_feed_ton_per_year": 70.0},
            {"scenario_name": "policy_support_case", "pathway": "htc", "allocated_feed_ton_per_year": 65.0},
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "high_supply_case", "scenario_feed_coverage_ratio": 0.9},
        ]
    ).to_csv(planning_dir / "portfolio_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "htc", "portfolio_allocated_feed_share": 1.0},
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "portfolio_allocated_feed_share": 0.0},
            {"scenario_name": "high_supply_case", "pathway": "htc", "portfolio_allocated_feed_share": 0.4},
            {"scenario_name": "high_supply_case", "pathway": "pyrolysis", "portfolio_allocated_feed_share": 0.6},
            {"scenario_name": "policy_support_case", "pathway": "htc", "portfolio_allocated_feed_share": 1.0},
            {"scenario_name": "policy_support_case", "pathway": "pyrolysis", "portfolio_allocated_feed_share": 0.0},
        ]
    ).to_csv(planning_dir / "pathway_summary.csv", index=False)
    abstract_path = paper_dir / "00-abstract.tex"
    abstract_path.write_text("AD-free.\n", encoding="utf-8")
    macros_path = paper_dir / "99-auto-macros.tex"

    result = sync_planning_summary_to_latex(
        planning_dir=planning_dir,
        abstract_path=abstract_path,
        macros_path=macros_path,
    )

    macros_text = macros_path.read_text(encoding="utf-8")

    assert result["dominance_pattern"] == "mixed"
    assert result["dominant_pathway"] == "mixed"
    assert "\\newcommand{\\PlanningDominantPathwayDisplay}{mixed}" in macros_text
    assert "\\newcommand{\\PlanningHighlightsDominanceBullet}{" in macros_text
    assert "baseline-region is HTC-dominant; high-supply is pyrolysis-dominant; policy-support is HTC-dominant" in macros_text
    assert "pyrolysis remains the supporting baseline portfolio pathway" in macros_text
    assert "\\newcommand{\\PlanningResultsDominanceSentence}{" in macros_text
    assert "the constrained portfolio remains scenario-dependent" in macros_text
    assert "baseline-region is HTC-dominant; high-supply is pyrolysis-dominant; policy-support is HTC-dominant" in macros_text


def test_manuscript_sync_uses_score_leader_clause_when_pyrolysis_dominates_allocations(tmp_path):
    planning_dir = tmp_path / "planning_sync_pyrolysis"
    paper_dir = tmp_path / "paper_pyrolysis"
    planning_dir.mkdir()
    paper_dir.mkdir()
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "top_ranked_case_id": "case-1"},
            {"scenario_name": "high_supply_case", "top_ranked_case_id": "case-2"},
            {"scenario_name": "policy_support_case", "top_ranked_case_id": "case-3"},
        ]
    ).to_csv(planning_dir / "scenario_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "allocated_feed_ton_per_year": 87.6},
            {"scenario_name": "baseline_region_case", "pathway": "htc", "allocated_feed_ton_per_year": 12.4},
            {"scenario_name": "high_supply_case", "pathway": "pyrolysis", "allocated_feed_ton_per_year": 87.9},
            {"scenario_name": "high_supply_case", "pathway": "htc", "allocated_feed_ton_per_year": 12.1},
            {"scenario_name": "policy_support_case", "pathway": "pyrolysis", "allocated_feed_ton_per_year": 100.0},
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "scenario_feed_coverage_ratio": 1.0},
            {"scenario_name": "high_supply_case", "scenario_feed_coverage_ratio": 0.9176628320631297},
            {"scenario_name": "policy_support_case", "scenario_feed_coverage_ratio": 1.0},
        ]
    ).to_csv(planning_dir / "portfolio_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "htc", "portfolio_allocated_feed_share": 0.124, "best_case_score": 1.226},
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "portfolio_allocated_feed_share": 0.876, "best_case_score": 0.913},
            {"scenario_name": "baseline_region_case", "pathway": "ad", "portfolio_allocated_feed_share": 0.0, "best_case_score": 0.081},
            {"scenario_name": "high_supply_case", "pathway": "htc", "portfolio_allocated_feed_share": 0.121, "best_case_score": 1.226},
            {"scenario_name": "high_supply_case", "pathway": "pyrolysis", "portfolio_allocated_feed_share": 0.879, "best_case_score": 0.913},
            {"scenario_name": "high_supply_case", "pathway": "ad", "portfolio_allocated_feed_share": 0.0, "best_case_score": 0.080},
            {"scenario_name": "policy_support_case", "pathway": "htc", "portfolio_allocated_feed_share": 0.0, "best_case_score": 1.227},
            {"scenario_name": "policy_support_case", "pathway": "pyrolysis", "portfolio_allocated_feed_share": 1.0, "best_case_score": 0.913},
            {"scenario_name": "policy_support_case", "pathway": "ad", "portfolio_allocated_feed_share": 0.0, "best_case_score": 0.084},
        ]
    ).to_csv(planning_dir / "pathway_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "htc", "baseline_portfolio_share_pct": 12.4, "max_stress_selection_rate": 75.0},
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "baseline_portfolio_share_pct": 87.6, "max_stress_selection_rate": 37.5},
            {"scenario_name": "baseline_region_case", "pathway": "ad", "baseline_portfolio_share_pct": 0.0, "max_stress_selection_rate": 0.0},
            {"scenario_name": "high_supply_case", "pathway": "htc", "baseline_portfolio_share_pct": 12.1, "max_stress_selection_rate": 75.0},
            {"scenario_name": "high_supply_case", "pathway": "pyrolysis", "baseline_portfolio_share_pct": 87.9, "max_stress_selection_rate": 37.5},
            {"scenario_name": "high_supply_case", "pathway": "ad", "baseline_portfolio_share_pct": 0.0, "max_stress_selection_rate": 0.0},
            {"scenario_name": "policy_support_case", "pathway": "htc", "baseline_portfolio_share_pct": 0.0, "max_stress_selection_rate": 0.0},
            {"scenario_name": "policy_support_case", "pathway": "pyrolysis", "baseline_portfolio_share_pct": 100.0, "max_stress_selection_rate": 62.5},
            {"scenario_name": "policy_support_case", "pathway": "ad", "baseline_portfolio_share_pct": 0.0, "max_stress_selection_rate": 0.0},
        ]
    ).to_csv(planning_dir / "main_results_table.csv", index=False)
    abstract_path = paper_dir / "00-abstract.tex"
    abstract_path.write_text("AD-free.\n", encoding="utf-8")
    macros_path = paper_dir / "99-auto-macros.tex"

    result = sync_planning_summary_to_latex(
        planning_dir=planning_dir,
        abstract_path=abstract_path,
        macros_path=macros_path,
    )

    macros_text = macros_path.read_text(encoding="utf-8")

    assert result["dominant_pathway"] == "pyrolysis"
    assert result["dominance_pattern"] == "uniform"
    assert "while HTC retains stronger best-case score leadership" in macros_text
    assert "while pyrolysis remains the supporting baseline portfolio pathway" not in macros_text
    assert (
        "\\newcommand{\\PlanningConclusionDominanceSentence}{Pyrolysis carries the leading baseline allocated "
        "share across the baseline-region, high-supply, and policy-support cases (baseline-region: pyrolysis "
        "87.6\\%; high-supply: pyrolysis 87.9\\%; policy-support: pyrolysis 100.0\\%), while HTC retains stronger "
        "case-level score leadership under the current evidence-qualified formulation.}"
    ) in macros_text


def test_manuscript_sync_ignores_operation_appendix_inputs_for_manuscript_macros(tmp_path):
    planning_dir = tmp_path / "planning_sync_operation"
    operation_dir = tmp_path / "operation_sync"
    paper_dir = tmp_path / "paper_operation"
    planning_dir.mkdir()
    operation_dir.mkdir()
    paper_dir.mkdir()

    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "top_ranked_case_id": "case-1"},
            {"scenario_name": "high_supply_case", "top_ranked_case_id": "case-2"},
            {"scenario_name": "policy_support_case", "top_ranked_case_id": "case-3"},
        ]
    ).to_csv(planning_dir / "scenario_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "allocated_feed_ton_per_year": 100.0},
            {"scenario_name": "high_supply_case", "pathway": "htc", "allocated_feed_ton_per_year": 100.0},
            {"scenario_name": "policy_support_case", "pathway": "pyrolysis", "allocated_feed_ton_per_year": 100.0},
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "high_supply_case", "scenario_feed_coverage_ratio": 0.9},
        ]
    ).to_csv(planning_dir / "portfolio_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "portfolio_allocated_feed_share": 1.0, "best_case_score": 0.91},
            {"scenario_name": "baseline_region_case", "pathway": "htc", "portfolio_allocated_feed_share": 0.0, "best_case_score": 0.88},
            {"scenario_name": "high_supply_case", "pathway": "htc", "portfolio_allocated_feed_share": 1.0, "best_case_score": 0.92},
            {"scenario_name": "high_supply_case", "pathway": "pyrolysis", "portfolio_allocated_feed_share": 0.0, "best_case_score": 0.86},
            {"scenario_name": "policy_support_case", "pathway": "pyrolysis", "portfolio_allocated_feed_share": 1.0, "best_case_score": 0.93},
            {"scenario_name": "policy_support_case", "pathway": "htc", "portfolio_allocated_feed_share": 0.0, "best_case_score": 0.87},
        ]
    ).to_csv(planning_dir / "pathway_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "method_name": "hold_plan",
                "dominant_case_id": "Waste2Energy::planning::pyrolysis::0011::baseline_region_case",
            },
            {
                "scenario_name": "high_supply_case",
                "method_name": "hold_plan",
                "dominant_case_id": "Waste2Energy::planning::htc::0011::high_supply_case",
            },
            {
                "scenario_name": "policy_support_case",
                "method_name": "hold_plan",
                "dominant_case_id": "Waste2Energy::planning::pyrolysis::0022::policy_support_case",
            },
        ]
    ).to_csv(operation_dir / "baseline_policy_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "method_type": "baseline_policy",
                "method_name": "hold_plan",
                "reward_mean": 10.0,
                "reward_std": 0.0,
                "average_reward_mean": 1.0,
                "max_violation_mean": 0.0,
                "violation_rate_mean": 0.0,
                "resilience_index_mean": 1.0,
                "hold_plan_reward_mean": 10.0,
                "reward_improvement_vs_hold_plan_abs": 0.0,
                "violation_aware_rank_within_scenario": 1.0,
            },
            {
                "scenario_name": "baseline_region_case",
                "method_type": "rl_agent",
                "method_name": "sac",
                "reward_mean": 10.0,
                "reward_std": 0.2,
                "average_reward_mean": 1.0,
                "max_violation_mean": 0.0,
                "violation_rate_mean": 0.0,
                "resilience_index_mean": 0.96,
                "hold_plan_reward_mean": 10.0,
                "reward_improvement_vs_hold_plan_abs": 0.0,
                "violation_aware_rank_within_scenario": 2.0,
            },
            {
                "scenario_name": "baseline_region_case",
                "method_type": "rl_agent",
                "method_name": "td3",
                "reward_mean": 9.5,
                "reward_std": 0.2,
                "average_reward_mean": 0.95,
                "max_violation_mean": 0.0,
                "violation_rate_mean": 0.0,
                "resilience_index_mean": 0.95,
                "hold_plan_reward_mean": 10.0,
                "reward_improvement_vs_hold_plan_abs": -0.5,
                "violation_aware_rank_within_scenario": 3.0,
            },
            {
                "scenario_name": "high_supply_case",
                "method_type": "baseline_policy",
                "method_name": "hold_plan",
                "reward_mean": 10.0,
                "reward_std": 0.0,
                "average_reward_mean": 1.0,
                "max_violation_mean": 0.0,
                "violation_rate_mean": 0.0,
                "resilience_index_mean": 1.0,
                "hold_plan_reward_mean": 10.0,
                "reward_improvement_vs_hold_plan_abs": 0.0,
                "violation_aware_rank_within_scenario": 1.0,
            },
            {
                "scenario_name": "high_supply_case",
                "method_type": "rl_agent",
                "method_name": "sac",
                "reward_mean": 10.0,
                "reward_std": 0.2,
                "average_reward_mean": 1.0,
                "max_violation_mean": 0.0,
                "violation_rate_mean": 0.0,
                "resilience_index_mean": 0.96,
                "hold_plan_reward_mean": 10.0,
                "reward_improvement_vs_hold_plan_abs": 0.0,
                "violation_aware_rank_within_scenario": 2.0,
            },
            {
                "scenario_name": "high_supply_case",
                "method_type": "rl_agent",
                "method_name": "td3",
                "reward_mean": 8.9,
                "reward_std": 0.2,
                "average_reward_mean": 0.89,
                "max_violation_mean": 0.0,
                "violation_rate_mean": 0.0,
                "resilience_index_mean": 0.95,
                "hold_plan_reward_mean": 10.0,
                "reward_improvement_vs_hold_plan_abs": -1.1,
                "violation_aware_rank_within_scenario": 3.0,
            },
            {
                "scenario_name": "policy_support_case",
                "method_type": "baseline_policy",
                "method_name": "hold_plan",
                "reward_mean": 10.0,
                "reward_std": 0.0,
                "average_reward_mean": 1.0,
                "max_violation_mean": 0.0,
                "violation_rate_mean": 0.0,
                "resilience_index_mean": 1.0,
                "hold_plan_reward_mean": 10.0,
                "reward_improvement_vs_hold_plan_abs": 0.0,
                "violation_aware_rank_within_scenario": 1.0,
            },
            {
                "scenario_name": "policy_support_case",
                "method_type": "rl_agent",
                "method_name": "sac",
                "reward_mean": 10.0,
                "reward_std": 0.2,
                "average_reward_mean": 1.0,
                "max_violation_mean": 0.0,
                "violation_rate_mean": 0.0,
                "resilience_index_mean": 0.96,
                "hold_plan_reward_mean": 10.0,
                "reward_improvement_vs_hold_plan_abs": 0.0,
                "violation_aware_rank_within_scenario": 2.0,
            },
            {
                "scenario_name": "policy_support_case",
                "method_type": "rl_agent",
                "method_name": "td3",
                "reward_mean": 8.8,
                "reward_std": 0.2,
                "average_reward_mean": 0.88,
                "max_violation_mean": 0.0,
                "violation_rate_mean": 0.0,
                "resilience_index_mean": 0.95,
                "hold_plan_reward_mean": 10.0,
                "reward_improvement_vs_hold_plan_abs": -1.2,
                "violation_aware_rank_within_scenario": 3.0,
            },
        ]
    ).to_csv(operation_dir / "rl_vs_baseline_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "method_name": "sac", "throughput_nonzero_rate_mean": 0.2, "severity_nonzero_rate_mean": 0.1},
            {"scenario_name": "baseline_region_case", "method_name": "td3", "throughput_nonzero_rate_mean": 0.4, "severity_nonzero_rate_mean": 0.2},
            {"scenario_name": "high_supply_case", "method_name": "sac", "throughput_nonzero_rate_mean": 0.2, "severity_nonzero_rate_mean": 0.1},
            {"scenario_name": "high_supply_case", "method_name": "td3", "throughput_nonzero_rate_mean": 0.5, "severity_nonzero_rate_mean": 0.2},
            {"scenario_name": "policy_support_case", "method_name": "sac", "throughput_nonzero_rate_mean": 0.2, "severity_nonzero_rate_mean": 0.1},
            {"scenario_name": "policy_support_case", "method_name": "td3", "throughput_nonzero_rate_mean": 0.6, "severity_nonzero_rate_mean": 0.2},
        ]
    ).to_csv(operation_dir / "policy_behavior_comparison.csv", index=False)

    abstract_path = paper_dir / "00-abstract.tex"
    abstract_path.write_text("AD-free.\n", encoding="utf-8")
    macros_path = paper_dir / "99-auto-macros.tex"

    result = sync_planning_summary_to_latex(
        planning_dir=planning_dir,
        abstract_path=abstract_path,
        macros_path=macros_path,
        operation_dir=operation_dir,
    )

    macros_text = macros_path.read_text(encoding="utf-8")

    assert "operation_anchor_pattern" not in result
    assert "operation_sac_status" not in result
    assert "operation_td3_status" not in result
    assert "\\newcommand{\\OperationAnchorSentence}{" not in macros_text
    assert "\\newcommand{\\OperationSACSentence}{" not in macros_text
    assert "\\newcommand{\\OperationTDThreeSentence}{" not in macros_text
    assert "\\newcommand{\\OperationTakeawaySentence}{" not in macros_text


def test_manuscript_sync_writes_benchmark_macros_and_artifacts(tmp_path):
    planning_dir = tmp_path / "planning_sync_benchmark"
    benchmark_dir = tmp_path / "benchmark_sync"
    audit_dir = tmp_path / "audit_sync"
    figures_dir = tmp_path / "figures_tables"
    paper_dir = tmp_path / "paper_benchmark"
    for directory in (planning_dir, benchmark_dir, audit_dir, figures_dir, paper_dir):
        directory.mkdir()

    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "top_ranked_case_id": "case-1"},
            {"scenario_name": "high_supply_case", "top_ranked_case_id": "case-2"},
            {"scenario_name": "policy_support_case", "top_ranked_case_id": "case-3"},
        ]
    ).to_csv(planning_dir / "scenario_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "allocated_feed_ton_per_year": 100.0},
            {"scenario_name": "high_supply_case", "pathway": "pyrolysis", "allocated_feed_ton_per_year": 100.0},
            {"scenario_name": "policy_support_case", "pathway": "pyrolysis", "allocated_feed_ton_per_year": 100.0},
        ]
    ).to_csv(planning_dir / "portfolio_allocations.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "scenario_feed_coverage_ratio": 1.0},
            {"scenario_name": "high_supply_case", "scenario_feed_coverage_ratio": 0.9},
            {"scenario_name": "policy_support_case", "scenario_feed_coverage_ratio": 1.0},
        ]
    ).to_csv(planning_dir / "portfolio_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "pathway": "pyrolysis", "portfolio_allocated_feed_share": 1.0, "best_case_score": 0.9},
            {"scenario_name": "high_supply_case", "pathway": "pyrolysis", "portfolio_allocated_feed_share": 1.0, "best_case_score": 0.9},
            {"scenario_name": "policy_support_case", "pathway": "pyrolysis", "portfolio_allocated_feed_share": 1.0, "best_case_score": 0.9},
        ]
    ).to_csv(planning_dir / "pathway_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "selected_in_baseline_portfolio": True,
                "baseline_portfolio_share_pct": 100.0,
                "max_stress_selection_rate": 50.0,
                "uq_stress_support": "interval_supported",
                "uncertainty_mode_sensitivity": "case-sensitive_pathway-stable",
            },
            {
                "scenario_name": "high_supply_case",
                "pathway": "pyrolysis",
                "selected_in_baseline_portfolio": True,
                "baseline_portfolio_share_pct": 100.0,
                "max_stress_selection_rate": 50.0,
                "uq_stress_support": "interval_supported",
                "uncertainty_mode_sensitivity": "case-sensitive_pathway-stable",
            },
            {
                "scenario_name": "policy_support_case",
                "pathway": "pyrolysis",
                "selected_in_baseline_portfolio": True,
                "baseline_portfolio_share_pct": 100.0,
                "max_stress_selection_rate": 50.0,
                "uq_stress_support": "interval_supported",
                "uncertainty_mode_sensitivity": "stable-across-tested-uq-modes",
            },
        ]
    ).to_csv(planning_dir / "main_results_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "active_uncertainty_penalty_mode": "prefer_interval_mean",
                "interval_mean_top_ranked_case_id": "Waste2Energy::planning::pyrolysis::0011::baseline_region_case",
                "max_interval_top_ranked_case_id": "Waste2Energy::planning::pyrolysis::0012::baseline_region_case",
                "combined_only_top_ranked_case_id": "Waste2Energy::planning::pyrolysis::0011::baseline_region_case",
                "uncertainty_mode_case_switch_count": 2,
                "uncertainty_mode_pathway_switch_count": 1,
            },
            {
                "scenario_name": "high_supply_case",
                "active_uncertainty_penalty_mode": "prefer_interval_mean",
                "interval_mean_top_ranked_case_id": "Waste2Energy::planning::pyrolysis::0021::high_supply_case",
                "max_interval_top_ranked_case_id": "Waste2Energy::planning::pyrolysis::0022::high_supply_case",
                "combined_only_top_ranked_case_id": "Waste2Energy::planning::pyrolysis::0021::high_supply_case",
                "uncertainty_mode_case_switch_count": 2,
                "uncertainty_mode_pathway_switch_count": 1,
            },
            {
                "scenario_name": "policy_support_case",
                "active_uncertainty_penalty_mode": "prefer_interval_mean",
                "interval_mean_top_ranked_case_id": "Waste2Energy::planning::pyrolysis::0031::policy_support_case",
                "max_interval_top_ranked_case_id": "Waste2Energy::planning::pyrolysis::0031::policy_support_case",
                "combined_only_top_ranked_case_id": "Waste2Energy::planning::pyrolysis::0031::policy_support_case",
                "uncertainty_mode_case_switch_count": 1,
                "uncertainty_mode_pathway_switch_count": 1,
            },
        ]
    ).to_csv(planning_dir / "optimization_diagnostics.csv", index=False)

    pd.DataFrame(
        [
            {
                "benchmark_variant": "classic_multiobjective_optimizer",
                "scenario_count": 3,
                "changed_pathway_count": 0,
                "changed_case_count": 3,
                "supports_core_innovation_count": 0,
                "supports_secondary_innovation_count": 3,
                "manuscript_sentence": "Classic baseline sentence.",
            },
            {
                "benchmark_variant": "no_evidence_penalty",
                "scenario_count": 3,
                "changed_pathway_count": 0,
                "changed_case_count": 3,
                "supports_core_innovation_count": 0,
                "supports_secondary_innovation_count": 3,
                "manuscript_sentence": "Evidence sentence.",
            },
            {
                "benchmark_variant": "no_robustness_penalty",
                "scenario_count": 3,
                "changed_pathway_count": 3,
                "changed_case_count": 3,
                "supports_core_innovation_count": 3,
                "supports_secondary_innovation_count": 0,
                "manuscript_sentence": "Primary robustness sentence.",
            },
            {
                "benchmark_variant": "greedy_weighted_score_heuristic",
                "scenario_count": 3,
                "changed_pathway_count": 0,
                "changed_case_count": 3,
                "supports_core_innovation_count": 0,
                "supports_secondary_innovation_count": 1,
                "manuscript_sentence": "Heuristic sentence.",
            },
        ]
    ).to_csv(audit_dir / "benchmark_manuscript_sentences.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "benchmark_variant": "no_robustness_penalty", "necessity_tier": "supports_core_innovation"},
        ]
    ).to_csv(audit_dir / "benchmark_claim_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "transferability_evidence_ceiling": "guarded_transfer"},
            {"scenario_name": "high_supply_case", "transferability_evidence_ceiling": "guarded_transfer"},
            {"scenario_name": "policy_support_case", "transferability_evidence_ceiling": "conditional_transfer_supported"},
        ]
    ).to_csv(audit_dir / "planning_transferability_risk_summary.csv", index=False)
    pd.DataFrame(
        [
            {"pathway": "htc", "reliability_tier": "auxiliary_only"},
            {"pathway": "pyrolysis", "reliability_tier": "conditional_support"},
        ]
    ).to_csv(audit_dir / "pathway_reliability_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "baseline_region_case", "benchmark_variant": "no_robustness_penalty", "effect_significance_tier": "highly_consistent", "pathway_shift_rate": 1.0, "case_shift_rate": 1.0},
            {"scenario_name": "high_supply_case", "benchmark_variant": "no_robustness_penalty", "effect_significance_tier": "highly_consistent", "pathway_shift_rate": 1.0, "case_shift_rate": 1.0},
            {"scenario_name": "policy_support_case", "benchmark_variant": "no_robustness_penalty", "effect_significance_tier": "highly_consistent", "pathway_shift_rate": 1.0, "case_shift_rate": 1.0},
            {"scenario_name": "baseline_region_case", "benchmark_variant": "classic_multiobjective_optimizer", "effect_significance_tier": "directionally_consistent", "pathway_shift_rate": 0.0, "case_shift_rate": 1.0},
            {"scenario_name": "high_supply_case", "benchmark_variant": "classic_multiobjective_optimizer", "effect_significance_tier": "directionally_consistent", "pathway_shift_rate": 0.0, "case_shift_rate": 1.0},
            {"scenario_name": "policy_support_case", "benchmark_variant": "classic_multiobjective_optimizer", "effect_significance_tier": "directionally_consistent", "pathway_shift_rate": 0.0, "case_shift_rate": 1.0},
            {"scenario_name": "baseline_region_case", "benchmark_variant": "no_evidence_penalty", "effect_significance_tier": "directionally_consistent", "pathway_shift_rate": 0.0, "case_shift_rate": 1.0},
            {"scenario_name": "high_supply_case", "benchmark_variant": "no_evidence_penalty", "effect_significance_tier": "directionally_consistent", "pathway_shift_rate": 0.0, "case_shift_rate": 1.0},
            {"scenario_name": "policy_support_case", "benchmark_variant": "no_evidence_penalty", "effect_significance_tier": "directionally_consistent", "pathway_shift_rate": 0.0, "case_shift_rate": 1.0},
            {"scenario_name": "baseline_region_case", "benchmark_variant": "greedy_weighted_score_heuristic", "effect_significance_tier": "suggestive", "pathway_shift_rate": 0.0, "case_shift_rate": 1.0},
        ]
    ).to_csv(benchmark_dir / "benchmark_statistical_summary.csv", index=False)
    htc_compare_dir = benchmark_dir / "htc_model_compare_lso"
    htc_compare_dir.mkdir()
    pd.DataFrame(
        [
            {"model_key": "catboost", "target_count": 4, "mean_validation_r2": 0.341, "mean_test_r2": 0.161, "selected_target_count": 3},
            {"model_key": "lightgbm", "target_count": 4, "mean_validation_r2": 0.313, "mean_test_r2": -0.287, "selected_target_count": 1},
            {"model_key": "stacking", "target_count": 4, "mean_validation_r2": 0.207, "mean_test_r2": 0.061, "selected_target_count": 0},
            {"model_key": "xgboost", "target_count": 4, "mean_validation_r2": 0.239, "mean_test_r2": -0.291, "selected_target_count": 0},
        ]
    ).to_csv(htc_compare_dir / "htc_model_comparison_aggregate.csv", index=False)
    pd.DataFrame(
        [
            {"selected_model_key": "catboost", "refit_test_r2": -0.353},
            {"selected_model_key": "catboost", "refit_test_r2": -0.226},
            {"selected_model_key": "lightgbm", "refit_test_r2": 0.279},
            {"selected_model_key": "catboost", "refit_test_r2": 0.329},
        ]
    ).to_csv(htc_compare_dir / "selected_models_manifest_leave_study_out.csv", index=False)

    abstract_path = paper_dir / "00-abstract.tex"
    abstract_path.write_text("AD-free.\n", encoding="utf-8")
    macros_path = paper_dir / "99-auto-macros.tex"

    result = sync_planning_summary_to_latex(
        planning_dir=planning_dir,
        abstract_path=abstract_path,
        macros_path=macros_path,
        audit_dir=audit_dir,
        benchmark_dir=benchmark_dir,
        figures_dir=figures_dir,
    )

    macros_text = macros_path.read_text(encoding="utf-8")
    narrative = json.loads((figures_dir / "paper1_benchmark_narrative.json").read_text(encoding="utf-8"))

    assert result["benchmark_primary_variant"] == "no_robustness_penalty"
    assert result["benchmark_primary_significance_tier"] == "highly_consistent"
    assert result["benchmark_bootstrap_available"] is True
    assert result["uq_sensitivity_pattern"] == "case-sensitive_pathway-stable"
    assert "case-level rather than pathway-level" in result["uq_sensitivity_sentence"]
    assert "case-level rather than pathway-level" in result["planning_uq_results_sentence"]
    assert "pathway-family stability but not a single invariant operating case" in result["planning_uq_discussion_sentence"]
    assert "\\newcommand{\\BenchmarkPrimaryVariant}{robustness-penalty removal}" in macros_text
    assert "\\newcommand{\\BenchmarkPrimaryInnovationSentence}{Primary robustness sentence.}" in macros_text
    assert "\\newcommand{\\BenchmarkClassicBaselineSentence}{Classic baseline sentence.}" in macros_text
    assert "\\newcommand{\\PlanningUQSensitivityPattern}{case-sensitive_pathway-stable}" in macros_text
    assert "\\newcommand{\\PlanningUQSensitivitySentence}{" in macros_text
    assert "\\newcommand{\\PlanningUQResultsSentence}{" in macros_text
    assert "\\newcommand{\\PlanningUQDiscussionSentence}{" in macros_text
    assert "robustness-removal ablation highly consistent" in macros_text
    assert "strongest benchmark-backed innovation claim currently centers on robustness-penalty removal" in macros_text
    assert "\\newcommand{\\BenchmarkResultsParagraph}{" in macros_text
    assert "\\newcommand{\\BenchmarkDiscussionParagraph}{" in macros_text
    assert "\\newcommand{\\BenchmarkConclusionParagraph}{" in macros_text
    assert "baseline-region:" in macros_text
    assert "case-level rather than pathway-level" in macros_text
    assert narrative["primary_variant"] == "no_robustness_penalty"
    assert "planning_uq_results_sentence" in narrative
    assert "planning_uq_discussion_sentence" in narrative
    assert (figures_dir / "paper1_benchmark_results_table.csv").exists()
    assert (figures_dir / "paper1_benchmark_claim_summary.csv").exists()
    assert (figures_dir / "paper1_benchmark_section_templates.json").exists()
    assert (figures_dir / "paper1_benchmark_section_templates.md").exists()
    assert (figures_dir / "paper1_benchmark_section_templates.tex").exists()
    assert "manuscript_table_artifacts" in result
    assert (figures_dir / "paper1_data_summary_table.tex").exists()
    assert (figures_dir / "paper1_surrogate_validation_table.tex").exists()
    assert (figures_dir / "paper1_transfer_support_table.tex").exists()
    assert (figures_dir / "paper1_evidence_ceiling_table.tex").exists()
    assert (figures_dir / "paper1_scenario_parameter_table.tex").exists()
    assert (figures_dir / "paper1_optimization_output_table.tex").exists()
    assert (figures_dir / "paper1_uq_sensitivity_table.tex").exists()
    assert (figures_dir / "paper1_cost_boundary_table.tex").exists()
    assert (figures_dir / "paper1_product_credit_sensitivity_table.tex").exists()
    assert (figures_dir / "paper1_htc_model_comparison_table.tex").exists()
    assert (figures_dir / "paper1_htc_model_comparison_note.md").exists()
    assert (figures_dir / "paper1_htc_model_narrative.md").exists()
    assert (figures_dir / "paper1_htc_model_narrative.tex").exists()
    assert (figures_dir / "paper1_figure3_htc_caption.md").exists()
    assert (figures_dir / "paper1_figure3_htc_caption.tex").exists()
    htc_compare_note = (figures_dir / "paper1_htc_model_comparison_note.md").read_text(encoding="utf-8")
    assert "study-group metadata" in htc_compare_note
    htc_compare_tex = (figures_dir / "paper1_htc_model_comparison_table.tex").read_text(encoding="utf-8")
    assert "CatBoost" in htc_compare_tex
    assert "direct study-ID encoding" in htc_compare_tex
    htc_narrative_md = (figures_dir / "paper1_htc_model_narrative.md").read_text(encoding="utf-8")
    assert "Figure 3 note" in htc_narrative_md
    assert "cross-study transfer constraint" in htc_narrative_md
    figure3_caption_md = (figures_dir / "paper1_figure3_htc_caption.md").read_text(encoding="utf-8")
    assert "HTC caption" in figure3_caption_md
    assert "stronger learners improve HTC competitiveness" in figure3_caption_md


def test_surrogate_validation_table_is_rebuilt_from_current_audit_and_benchmark(tmp_path, monkeypatch):
    audit_dir = tmp_path / "audit"
    benchmark_dir = tmp_path / "benchmark"
    surrogate_root = tmp_path / "surrogates"
    htc_compare_dir = benchmark_dir / "htc_model_compare_lso"
    paper1_strict_group_dir = surrogate_root / "paper1_strict_group"
    for directory in (audit_dir, benchmark_dir, surrogate_root, htc_compare_dir, paper1_strict_group_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "summary_label": "leave_study_out",
                "dataset_key": "htc_direct",
                "target_column": "carbon_retention_pct",
                "best_model_key": "catboost",
                "claim_status": "weak",
            },
            {
                "summary_label": "leave_study_out",
                "dataset_key": "htc_direct",
                "target_column": "product_char_yield_pct",
                "best_model_key": "lightgbm",
                "claim_status": "supportive",
            },
            {
                "summary_label": "leave_study_out",
                "dataset_key": "pyrolysis_direct",
                "target_column": "carbon_retention_pct",
                "best_model_key": "xgboost",
                "claim_status": "supportive",
            },
            {
                "summary_label": "leave_study_out",
                "dataset_key": "pyrolysis_direct",
                "target_column": "product_char_yield_pct",
                "best_model_key": "gradient_boosting",
                "claim_status": "supportive",
            },
        ]
    ).to_csv(audit_dir / "ml_claim_flag_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "pathway": "pyrolysis",
                "leave_study_out_supportive_count": 2,
                "leave_study_out_weak_count": 0,
                "leave_study_out_unsupported_count": 0,
                "reliability_score": 1.0,
                "reliability_tier": "strong_support",
            },
            {
                "pathway": "htc",
                "leave_study_out_supportive_count": 1,
                "leave_study_out_weak_count": 1,
                "leave_study_out_unsupported_count": 0,
                "reliability_score": 0.75,
                "reliability_tier": "conditional_support",
            },
        ]
    ).to_csv(audit_dir / "pathway_reliability_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "dataset_key": "pyrolysis_direct",
                "target_column": "product_char_yield_pct",
                "selected_model_key": "xgboost",
                "selection_metric_value": 0.91,
                "benchmark_validation_r2": 0.91,
                "benchmark_test_r2": 0.83,
            }
        ]
    ).to_csv(surrogate_root / "selected_models_manifest_strict_group.csv", index=False)

    pd.DataFrame(
        [
            {
                "dataset_key": "paper1_htc_scope",
                "target_column": "carbon_retention_pct",
                "model_key": "xgboost",
                "validation_r2": 0.88,
                "test_r2": 0.76,
            },
            {
                "dataset_key": "paper1_htc_scope",
                "target_column": "product_char_hhv_mj_per_kg",
                "model_key": "rf",
                "validation_r2": 0.79,
                "test_r2": 0.67,
            },
        ]
    ).to_csv(paper1_strict_group_dir / "traditional_ml_suite_summary_strict_group.csv", index=False)

    pd.DataFrame(
        [
            {
                "dataset_key": "htc_direct",
                "target_column": "carbon_retention_pct",
                "selected_model_key": "catboost",
                "selection_metric_value": 0.41,
                "selected_validation_r2": 0.41,
                "benchmark_validation_r2": 0.41,
                "benchmark_test_r2": 0.12,
            },
            {
                "dataset_key": "htc_direct",
                "target_column": "product_char_yield_pct",
                "selected_model_key": "lightgbm",
                "selection_metric_value": 0.52,
                "selected_validation_r2": 0.52,
                "benchmark_validation_r2": 0.52,
                "benchmark_test_r2": 0.61,
            },
        ]
    ).to_csv(htc_compare_dir / "selected_models_manifest_leave_study_out.csv", index=False)

    pd.DataFrame(
        [
            {
                "dataset_key": "pyrolysis_direct",
                "target_column": "carbon_retention_pct",
                "model_key": "xgboost",
                "validation_r2": 0.77,
                "test_r2": 0.58,
            },
            {
                "dataset_key": "pyrolysis_direct",
                "target_column": "carbon_retention_pct",
                "model_key": "rf",
                "validation_r2": 0.65,
                "test_r2": 0.49,
            },
            {
                "dataset_key": "pyrolysis_direct",
                "target_column": "product_char_yield_pct",
                "model_key": "gradient_boosting",
                "validation_r2": 0.87,
                "test_r2": 0.82,
            },
            {
                "dataset_key": "pyrolysis_direct",
                "target_column": "product_char_yield_pct",
                "model_key": "xgboost",
                "validation_r2": 0.84,
                "test_r2": 0.80,
            },
        ]
    ).to_csv(surrogate_root / "traditional_ml_suite_summary_leave_study_out.csv", index=False)

    monkeypatch.setattr(manuscript_sync_module, "resolve_surrogate_outputs_dir", lambda: surrogate_root)

    table = manuscript_sync_module._build_surrogate_validation_table(
        audit_dir=audit_dir,
        benchmark_dir=benchmark_dir,
    )

    assert not table.empty
    assert set(table["dataset_scope"]) == {
        "HTC mixed-feed planning scope",
        "Pyrolysis direct observations",
        "HTC direct observations",
    }

    strict_htc = table[
        (table["validation_tier"] == "strict-group") & (table["dataset_scope"] == "HTC mixed-feed planning scope")
    ].reset_index(drop=True)
    assert strict_htc["best_model"].tolist() == ["XGBoost", "Random Forest"]
    assert strict_htc["test_r2"].round(3).tolist() == [0.760, 0.670]
    assert strict_htc["interpretation"].tolist() == ["supportive", "supportive"]

    lso_htc = table[
        (table["validation_tier"] == "leave-study-out") & (table["dataset_scope"] == "HTC direct observations")
    ].reset_index(drop=True)
    assert lso_htc["best_model"].tolist() == ["CatBoost", "LightGBM"]
    assert lso_htc["interpretation"].tolist() == ["weak", "supportive"]

    pyro_lso = table[
        (table["validation_tier"] == "leave-study-out") & (table["dataset_scope"] == "Pyrolysis direct observations")
    ].reset_index(drop=True)
    assert pyro_lso["best_model"].tolist() == ["XGBoost", "Gradient Boosting"]
    assert pyro_lso["test_r2"].round(3).tolist() == [0.580, 0.820]

    transfer_support = manuscript_sync_module._build_transfer_support_table(
        audit_dir=audit_dir,
        surrogate_validation=table,
    )
    pyro_transfer = transfer_support[transfer_support["pathway"] == "pyrolysis"].iloc[0]
    htc_transfer = transfer_support[transfer_support["pathway"] == "HTC"].iloc[0]
    assert int(pyro_transfer["strict_group_targets"]) == 1
    assert pyro_transfer["strict_group_test_r2_range"] == "0.830"
    assert int(htc_transfer["strict_group_targets"]) == 2
    assert htc_transfer["strict_group_test_r2_range"] == "0.670--0.760"
