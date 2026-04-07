# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import pandas as pd

from waste2energy.planning.reporting import build_main_results_table
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
    positive_support = ad_rows["max_stress_selection_rate"].gt(0.0)
    if positive_support.any():
        assert ad_rows.loc[positive_support, "stress_tests_supporting_pathway"].str.contains(
            "environment_priority"
        ).all()
    else:
        assert (ad_rows["stress_tests_supporting_pathway"] == "none").all()
        assert ad_rows["writing_label"].isin(
            {"competitive but unselected", "comparison only"}
        ).all()


def test_scenario_run_refreshes_audit_outputs(workflow_dirs):
    audit_dir = workflow_dirs["root"] / "audit"
    planning_claim_flags = pd.read_csv(audit_dir / "planning_claim_flag_table.csv")

    assert not planning_claim_flags.empty
    assert planning_claim_flags["pathway"].isin(["pyrolysis", "htc", "ad", "baseline"]).all()
