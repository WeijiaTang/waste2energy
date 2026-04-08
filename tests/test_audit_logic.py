# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import warnings

import pandas as pd

from waste2energy.audit import (
    InconsistencyWarning,
    build_ml_best_result_summary,
    build_ml_claim_flag_table,
    build_ml_refit_provenance_summary,
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

    row = result.iloc[0]
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

    row = result.iloc[0]
    assert row["best_model_key"] == "rf"
    assert row["selection_metric_name"] == "validation_r2"
    assert row["claim_status"] == "weak"


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
    assert (
        "\\newcommand{\\PlanningHighlightsDominanceBullet}{All three main scenarios return an HTC-dominant "
        "optimized baseline portfolio in the current exported planning runs, while pyrolysis remains a "
        "stress-sensitive alternative.}"
    ) in macros_text
    assert (
        "\\newcommand{\\PlanningResultsDominanceSentence}{Across the baseline, high-supply, and policy-support "
        "scenarios, the constrained portfolio is now HTC-dominant in the current exported planning runs.}"
    ) in macros_text
    assert "\\newcommand{\\PlanningADStatus}{AD-limited}" in macros_text


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
    assert (
        "\\newcommand{\\PlanningHighlightsDominanceBullet}{The three main scenarios remain scenario-dependent in "
        "the current exported planning runs (baseline-region is HTC-dominant; high-supply is "
        "pyrolysis-dominant; policy-support is HTC-dominant), while pyrolysis remains the supporting baseline "
        "portfolio pathway.}"
    ) in macros_text
    assert (
        "\\newcommand{\\PlanningResultsDominanceSentence}{Across the baseline, high-supply, and policy-support "
        "scenarios, the constrained portfolio remains scenario-dependent in the current exported planning runs "
        "(baseline-region is HTC-dominant; high-supply is pyrolysis-dominant; policy-support is HTC-dominant).}"
    ) in macros_text


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
