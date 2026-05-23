# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import pandas as pd
import pytest

from waste2energy.planning.reporting import (
    _build_uq_mode_comparison_sentence,
    _build_results_sentence,
    _classify_results_row,
    _format_blend_label,
    build_main_results_table,
)
from waste2energy.scenarios.robustness import build_decision_stability
from waste2energy.scenarios.run import resolve_audit_output_dir, run_scenario_robustness_baseline
from waste2energy.scenarios.uncertainty import build_uncertainty_summary


def test_scenario_robustness_smoke(tmp_path):
    output_dir = tmp_path / "scenarios"
    planning_dir = tmp_path / "planning"
    from waste2energy.planning.solve import PlanningConfig, run_planning_baseline

    run_planning_baseline(output_dir=str(planning_dir), config=PlanningConfig(pareto_point_count=6))
    result = run_scenario_robustness_baseline(
        output_dir=str(output_dir),
        planning_dir=str(planning_dir),
    )

    assert result["planner_variant"] == "surrogate_driven_robust_multiobjective_optimizer"
    assert result["stress_test_count"] >= 3

    stress_registry = pd.read_csv(output_dir / "stress_registry.csv")
    decision_stability = pd.read_csv(output_dir / "decision_stability.csv")
    uncertainty_summary = pd.read_csv(output_dir / "uncertainty_summary.csv")
    planning_claim_flags = pd.read_csv(tmp_path / "audit" / "planning_claim_flag_table.csv")
    main_results_table = pd.read_csv(planning_dir / "main_results_table.csv")
    recommendation_confidence = pd.read_csv(planning_dir / "recommendation_confidence_summary.csv")
    transferability_risk = pd.read_csv(tmp_path / "audit" / "planning_transferability_risk_summary.csv")

    assert "robustness_factor" in stress_registry.columns
    assert "carbon_budget_factor" in stress_registry.columns
    assert "constraint_relaxation_ratio" in stress_registry.columns
    assert "candidate_cap_enforced" in stress_registry.columns
    assert "uncertainty_penalty_mode" in stress_registry.columns
    assert "unconstrained_scenario" in set(stress_registry["stress_test_name"])
    assert "uncertainty_penalty_max_interval" in set(stress_registry["stress_test_name"])
    assert "uncertainty_penalty_combined_only" in set(stress_registry["stress_test_name"])
    uncertainty_modes = stress_registry.set_index("stress_test_name")["uncertainty_penalty_mode"].to_dict()
    assert uncertainty_modes["baseline"] == "prefer_interval_mean"
    assert uncertainty_modes["uncertainty_penalty_max_interval"] == "max_interval_ratio"
    assert uncertainty_modes["uncertainty_penalty_combined_only"] == "combined_only"
    assert not decision_stability.empty
    assert not uncertainty_summary.empty
    assert not planning_claim_flags.empty
    assert not main_results_table.empty
    assert not recommendation_confidence.empty
    assert not transferability_risk.empty
    assert "uncertainty_stress_selection_rate" in decision_stability.columns
    assert "selected_under_max_interval_uncertainty" in decision_stability.columns
    assert "selected_under_combined_only_uncertainty" in decision_stability.columns
    assert "max_interval_changes_case_vs_baseline" in uncertainty_summary.columns
    assert "combined_only_changes_case_vs_baseline" in uncertainty_summary.columns
    assert "uncertainty_mode_case_switch_count" in uncertainty_summary.columns
    assert "uncertainty_mode_sensitivity" in main_results_table.columns
    assert "best_case_uq_ranking_note" in main_results_table.columns
    assert "best_case_uq_rank_profile" in main_results_table.columns
    assert "uq_mode_comparison_sentence" in main_results_table.columns
    assert "uq_stress_support" in main_results_table.columns
    assert "max_uq_stress_selection_rate" in main_results_table.columns
    assert "uncertainty_mode_ranking_summary" in main_results_table.columns
    assert "claim_status" in planning_claim_flags.columns
    assert "recommendation_confidence_tier" in planning_claim_flags.columns
    assert "recommendation_evidence_ceiling" in planning_claim_flags.columns
    assert "recommendation_confidence_tier" in recommendation_confidence.columns
    assert recommendation_confidence["recommendation_confidence_score"].between(0.0, 1.0).all()
    assert "transferability_evidence_ceiling" in transferability_risk.columns
    assert main_results_table["best_case_uq_rank_profile"].astype(str).str.len().gt(0).all()
    assert main_results_table["uq_mode_comparison_sentence"].astype(str).str.len().gt(0).all()


def test_scenario_run_requires_explicit_planning_dir(tmp_path):
    output_dir = tmp_path / "scenarios"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="requires an explicit planning_dir"):
        run_scenario_robustness_baseline(output_dir=str(output_dir))


def test_reporting_preserves_environment_priority_ad_support(workflow_dirs):
    table, _ = build_main_results_table(
        planning_dir=workflow_dirs["planning_dir"],
        scenario_dir=workflow_dirs["scenario_dir"],
    )
    ad_rows = table[table["pathway"] == "ad"].copy()

    assert not ad_rows.empty
    positive_support = (
        ad_rows["max_stress_selection_rate"].gt(0.0)
        & ~ad_rows["selected_in_baseline_portfolio"].fillna(False).astype(bool)
    )
    if positive_support.any():
        assert ad_rows.loc[positive_support, "stress_tests_supporting_pathway"].str.contains(
            "environment_priority"
        ).all()
        assert ad_rows.loc[positive_support, "writing_label"].eq(
            "environment-sensitive alternative"
        ).all()
    else:
        assert (ad_rows["stress_tests_supporting_pathway"] == "none").all()
        assert ad_rows["writing_label"].isin(
            {"competitive but unselected", "comparison only"}
        ).all()


def test_reporting_classifies_environment_priority_when_multiple_stress_tags_exist():
    row = pd.Series(
        {
            "portfolio_selected_count": 0,
            "portfolio_allocated_feed_share": 0.0,
            "max_stress_selection_rate": 0.143,
            "score_gap_to_scenario_best_pct": 0.08,
            "pathway": "ad",
            "stress_test_tags": "cost_guardrail|environment_priority",
        }
    )

    assert _classify_results_row(row) == "environment_sensitive_alternative"


def test_reporting_marks_missing_manuscript_fields_as_not_evaluated():
    row = pd.Series(
        {
            "portfolio_selected_count": 0,
            "portfolio_allocated_feed_share": pd.NA,
            "max_stress_selection_rate": pd.NA,
            "score_gap_to_scenario_best_pct": pd.NA,
            "pathway": "htc",
            "stress_test_tags": "none",
        }
    )

    assert _classify_results_row(row) == "not_evaluated"


def test_reporting_sentence_mentions_missing_allocated_share_for_selected_pathway():
    row = pd.Series(
        {
            "scenario_name": "baseline_region_case",
            "pathway": "pyrolysis",
            "writing_label": "supporting_baseline_portfolio",
            "portfolio_allocated_feed_share": pd.NA,
            "stress_test_tags": "none",
        }
    )

    sentence = _build_results_sentence(row)

    assert "allocated-share value is not available" in sentence


def test_uq_mode_sentence_is_explicit_when_rank_profile_unavailable():
    sentence = _build_uq_mode_comparison_sentence(
        pd.Series(
            {
                "pathway": "ad",
                "best_case_uq_rank_profile": "not available",
                "uncertainty_mode_sensitivity": "not evaluated",
            }
        )
    )

    assert "not available" in sentence
    assert "policy-floor diagnostic" in sentence


def test_reporting_blend_label_exposes_missing_blend_ratios():
    row = pd.Series(
        {
            "best_case_blend_manure_ratio": pd.NA,
            "best_case_blend_wet_waste_ratio": 0.4,
        }
    )

    assert _format_blend_label(row) == "blend not available"


def test_reporting_main_results_fail_fast_on_missing_required_numeric_field(tmp_path):
    planning_dir = tmp_path / "planning"
    scenario_dir = tmp_path / "scenario"
    model_ready_dir = tmp_path / "model_ready"
    planning_dir.mkdir()
    scenario_dir.mkdir()
    model_ready_dir.mkdir()

    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "htc",
                "best_case_score": 1.0,
                "best_case_id": "case-1",
                "portfolio_selected_count": 1,
                "portfolio_allocated_feed_share": pd.NA,
                "best_case_energy_objective": 10.0,
                "best_case_environment_objective": 5.0,
                "best_case_manure_subtype": "dairy",
                "best_case_blend_manure_ratio": 0.6,
                "best_case_blend_wet_waste_ratio": 0.4,
            }
        ]
    ).to_csv(planning_dir / "pathway_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "optimization_case_id": "case-1",
                "scenario_name": "baseline_region_case",
                "sample_id": "sample-1",
                "pathway": "htc",
                "process_temperature_c": 220.0,
                "residence_time_min": 60.0,
                "heating_rate_c_per_min": 5.0,
                "planning_energy_intensity_mj_per_ton": 10.0,
                "planning_environment_intensity_kgco2e_per_ton": 5.0,
                "planning_score": 1.0,
            }
        ]
    ).to_csv(planning_dir / "scored_cases.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "representative_case_id": "case-1",
                "sample_id": "sample-1",
                "selection_rate": 0.5,
                "stable_under_majority_rule": True,
                "stable_under_consensus_rule": False,
                "stress_tests_selected": "environment_priority",
            }
        ]
    ).to_csv(scenario_dir / "decision_stability.csv", index=False)
    pd.DataFrame(
        [
            {
                "pathway": "htc",
                "process_basis": "observed HTC operating window",
                "performance_basis": "mixed-feed proxy on observed HTC anchor",
                "claim_boundary": "planning-ready candidate with cross-study caution",
            }
        ]
    ).to_csv(model_ready_dir / "optimization_pathway_readiness_summary.csv", index=False)

    from waste2energy.planning import reporting

    original_model_ready_dir = reporting.MODEL_READY_DIR
    try:
        reporting.MODEL_READY_DIR = model_ready_dir
        with pytest.raises(ValueError, match="portfolio_allocated_feed_share"):
            build_main_results_table(planning_dir=planning_dir, scenario_dir=scenario_dir)
    finally:
        reporting.MODEL_READY_DIR = original_model_ready_dir


def test_scenario_run_refreshes_audit_outputs(workflow_dirs):
    audit_dir = workflow_dirs["root"] / "audit"
    planning_claim_flags = pd.read_csv(audit_dir / "planning_claim_flag_table.csv")

    assert not planning_claim_flags.empty
    assert planning_claim_flags["pathway"].isin(["pyrolysis", "htc", "ad", "baseline"]).all()
    assert (workflow_dirs["planning_dir"] / "main_results_table.csv").exists()


def test_decision_stability_tracks_uncertainty_mode_selection_rate():
    portfolio = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "stress_test_name": "baseline",
                "sample_id": "sample-1",
                "optimization_case_id": "Waste2Energy::planning::pyrolysis::0010::baseline_region_case",
                "portfolio_rank": 1,
                "allocated_feed_share": 0.50,
                "planning_score": 1.0,
                "manure_subtype": "beef",
            },
            {
                "scenario_name": "baseline_region_case",
                "stress_test_name": "uncertainty_penalty_max_interval",
                "sample_id": "sample-1",
                "optimization_case_id": "Waste2Energy::planning::pyrolysis::0010::baseline_region_case",
                "portfolio_rank": 1,
                "allocated_feed_share": 0.52,
                "planning_score": 1.1,
                "manure_subtype": "beef",
            },
            {
                "scenario_name": "baseline_region_case",
                "stress_test_name": "uncertainty_penalty_combined_only",
                "sample_id": "sample-2",
                "optimization_case_id": "Waste2Energy::planning::pyrolysis::0011::baseline_region_case",
                "portfolio_rank": 1,
                "allocated_feed_share": 0.48,
                "planning_score": 0.9,
                "manure_subtype": "dairy",
            },
        ]
    )

    stability = build_decision_stability(portfolio)
    sample_1 = stability.loc[stability["sample_id"] == "sample-1"].iloc[0]
    sample_2 = stability.loc[stability["sample_id"] == "sample-2"].iloc[0]

    assert sample_1["uncertainty_stress_run_count"] == 1
    assert sample_1["uncertainty_stress_selection_rate"] == pytest.approx(0.5)
    assert bool(sample_1["selected_under_max_interval_uncertainty"]) is True
    assert bool(sample_1["selected_under_combined_only_uncertainty"]) is False
    assert sample_2["uncertainty_stress_run_count"] == 1
    assert sample_2["uncertainty_stress_selection_rate"] == pytest.approx(0.5)
    assert bool(sample_2["selected_under_combined_only_uncertainty"]) is True


def test_uncertainty_summary_exposes_uncertainty_mode_switch_flags():
    stress = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "stress_test_name": "baseline",
                "top_portfolio_case_id": "Waste2Energy::planning::pyrolysis::0010::baseline_region_case",
                "portfolio_energy_objective": 10.0,
                "portfolio_environment_objective": 5.0,
                "portfolio_cost_objective": 1.0,
                "scenario_feed_coverage_ratio": 1.0,
                "remaining_unmet_feed_ton_per_year": 0.0,
            },
            {
                "scenario_name": "baseline_region_case",
                "stress_test_name": "uncertainty_penalty_max_interval",
                "top_portfolio_case_id": "Waste2Energy::planning::pyrolysis::0012::baseline_region_case",
                "portfolio_energy_objective": 10.2,
                "portfolio_environment_objective": 5.1,
                "portfolio_cost_objective": 1.0,
                "scenario_feed_coverage_ratio": 1.0,
                "remaining_unmet_feed_ton_per_year": 0.0,
            },
            {
                "scenario_name": "baseline_region_case",
                "stress_test_name": "uncertainty_penalty_combined_only",
                "top_portfolio_case_id": "Waste2Energy::planning::pyrolysis::0010::baseline_region_case",
                "portfolio_energy_objective": 10.0,
                "portfolio_environment_objective": 5.0,
                "portfolio_cost_objective": 1.0,
                "scenario_feed_coverage_ratio": 1.0,
                "remaining_unmet_feed_ton_per_year": 0.0,
            },
        ]
    )

    summary = build_uncertainty_summary(stress_test_summary=stress, decision_stability=pd.DataFrame())
    row = summary.iloc[0]

    assert row["baseline_top_case_id"].endswith("::0010::baseline_region_case")
    assert row["max_interval_top_case_id"].endswith("::0012::baseline_region_case")
    assert bool(row["max_interval_changes_case_vs_baseline"]) is True
    assert bool(row["combined_only_changes_case_vs_baseline"]) is False
    assert row["uncertainty_mode_case_switch_count"] == 2
    assert row["uncertainty_mode_pathway_switch_count"] == 1


def test_scenario_run_reporting_outputs_stay_with_explicit_planning_dir(tmp_path):
    planning_dir = tmp_path / "planning"
    scenario_dir = tmp_path / "scenarios"

    from waste2energy.planning.solve import PlanningConfig, run_planning_baseline

    run_planning_baseline(output_dir=str(planning_dir), config=PlanningConfig(pareto_point_count=4))
    result = run_scenario_robustness_baseline(
        output_dir=str(scenario_dir),
        planning_dir=str(planning_dir),
        base_config=PlanningConfig(pareto_point_count=4),
    )

    assert result["reporting_outputs"]["planning_results_table"] == str(planning_dir / "main_results_table.csv")
    assert result["reporting_outputs"]["planning_results_table_thermochemical"] == str(
        planning_dir / "main_results_table_thermochemical.csv"
    )
    assert result["reporting_outputs"]["planning_ad_reference_diagnostics"] == str(
        planning_dir / "ad_reference_diagnostics.csv"
    )
    thermochemical = pd.read_csv(planning_dir / "main_results_table_thermochemical.csv")
    assert not thermochemical["pathway"].astype(str).str.lower().eq("ad").any()
    ad_reference = pd.read_csv(planning_dir / "ad_reference_diagnostics.csv")
    assert not ad_reference.empty
    assert result["audit_outputs"]["planning_claim_flag_table"] == str(tmp_path / "audit" / "planning_claim_flag_table.csv")


def test_resolve_audit_output_dir_prefers_shared_outputs_root_shape(tmp_path):
    planning_dir = tmp_path / "outputs" / "planning" / "baseline"
    scenario_dir = tmp_path / "outputs" / "scenarios" / "baseline"

    assert resolve_audit_output_dir(planning_dir=planning_dir) == tmp_path / "outputs" / "audit"
    assert resolve_audit_output_dir(scenario_dir=scenario_dir) == tmp_path / "outputs" / "audit"


def test_resolve_audit_output_dir_falls_back_to_parent_audit_dir(tmp_path):
    planning_dir = tmp_path / "custom-planning-run"
    scenario_dir = tmp_path / "custom-scenarios-run"

    assert resolve_audit_output_dir(planning_dir=planning_dir) == tmp_path / "audit"
    assert resolve_audit_output_dir(scenario_dir=scenario_dir) == tmp_path / "audit"
