# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import pandas as pd
import pytest

from waste2energy.planning.reporting import (
    _build_results_sentence,
    _classify_results_row,
    _format_blend_label,
    build_main_results_table,
)
from waste2energy.scenarios.run import run_scenario_robustness_baseline


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

    assert "robustness_factor" in stress_registry.columns
    assert "carbon_budget_factor" in stress_registry.columns
    assert "constraint_relaxation_ratio" in stress_registry.columns
    assert "candidate_cap_enforced" in stress_registry.columns
    assert "unconstrained_scenario" in set(stress_registry["stress_test_name"])
    assert not decision_stability.empty
    assert not uncertainty_summary.empty
    assert not planning_claim_flags.empty
    assert "claim_status" in planning_claim_flags.columns


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
