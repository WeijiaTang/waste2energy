# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import pandas as pd
import pytest

from waste2energy.operation.baselines import run_baseline_policies
from waste2energy.operation.comparison import build_baseline_policy_summary, build_comparison_table
from waste2energy.operation import inputs as operation_inputs
from waste2energy.operation.inputs import OperationInputBundle, build_operation_environment_specs
from waste2energy.scenarios.uncertainty import build_uncertainty_summary


def test_operation_baseline_smoke(workflow_dirs):
    specs = build_operation_environment_specs(
        planning_dir=workflow_dirs["planning_dir"],
        scenario_dir=workflow_dirs["scenario_dir"],
    )
    rollout_steps, rollout_summary = run_baseline_policies(specs, horizon_steps=6)

    assert len(specs) > 0
    assert {"reward_energy_weight", "reward_environment_weight", "reward_cost_weight"}.issubset(specs.columns)
    assert "global_energy_reference_mj_per_year" in specs.columns
    assert "nominal_carbon_load_kgco2e_per_ton" in specs.columns
    assert specs["dominant_sample_id"].nunique() == len(specs)
    assert (specs["candidate_feed_target_ton_per_year"] <= specs["candidate_capacity_cap_ton_per_year"] + 1e-9).all()
    assert len(rollout_steps) > 0
    assert len(rollout_summary) > 0
    assert "violation_rate" in rollout_summary.columns
    assert "resilience_index" in rollout_summary.columns
    assert rollout_summary["policy_name"].isin({"hold_plan", "track_target", "energy_push"}).all()

    baseline_summary = build_baseline_policy_summary(rollout_summary)
    comparison = build_comparison_table(baseline_summary, [])
    assert "violation_rate_mean" in comparison.columns
    assert "resilience_index_mean" in comparison.columns


def test_operation_environment_specs_fail_fast_on_missing_planning_intensity(monkeypatch):
    bundle = OperationInputBundle(
        planning_portfolio_allocations=pd.DataFrame(
            [
                {
                    "scenario_name": "baseline_region_case",
                    "sample_id": "sample-1",
                    "optimization_case_id": "case-1",
                    "allocated_feed_ton_per_year": 10.0,
                    "effective_processing_budget_ton_per_year": 20.0,
                    "candidate_capacity_cap_ton_per_year": 15.0,
                    "scenario_feed_budget_ton_per_year": 20.0,
                    "planning_energy_intensity_mj_per_ton": pd.NA,
                    "planning_environment_intensity_kgco2e_per_ton": 5.0,
                    "planning_carbon_load_kgco2e_per_ton": 1.0,
                    "planning_cost_intensity_proxy_or_real_per_ton": 2.0,
                    "process_temperature_c": 500.0,
                    "residence_time_min": 30.0,
                    "manure_subtype": "beef",
                    "pathway": "pyrolysis",
                }
            ]
        ),
        planning_constraints=pd.DataFrame(
            [
                {
                    "scenario_name": "baseline_region_case",
                    "capacity_binding_reason": "feed_budget",
                }
            ]
        ),
        scenario_decision_stability=pd.DataFrame(
            [
                {
                    "scenario_name": "baseline_region_case",
                    "sample_id": "sample-1",
                    "avg_allocated_feed_share": 0.5,
                    "max_allocated_feed_share": 0.5,
                }
            ]
        ),
        scenario_uncertainty_summary=pd.DataFrame(
            [
                {
                    "scenario_name": "baseline_region_case",
                    "energy_range_ratio": 0.1,
                    "environment_range_ratio": 0.1,
                    "cost_range_ratio": 0.1,
                    "coverage_range_ratio": 0.1,
                    "unmet_feed_max": 0.0,
                    "dominant_selection_rate": 1.0,
                    "stable_candidate_count": 1,
                }
            ]
        ),
        scenario_cross_stability=pd.DataFrame(
            [
                {
                    "sample_id": "sample-1",
                    "cross_scenario_selection_rate": 0.0,
                    "selected_in_all_scenarios": False,
                }
            ]
        ),
        planning_run_config={"objective_weights": {"preset_name": "balanced_cleaner_production", "weights": {}}},
        scenario_run_config={},
    )

    monkeypatch.setattr(operation_inputs, "load_operation_input_bundle", lambda **kwargs: bundle)
    monkeypatch.setattr(operation_inputs, "_validate_operation_input_freshness", lambda _bundle: None)

    with pytest.raises(ValueError, match="planning_energy_intensity_mj_per_ton"):
        build_operation_environment_specs()


def test_operation_environment_specs_expose_fallback_and_clip_sources(monkeypatch):
    bundle = OperationInputBundle(
        planning_portfolio_allocations=pd.DataFrame(
            [
                {
                    "scenario_name": "baseline_region_case",
                    "sample_id": "sample-1",
                    "optimization_case_id": "case-1",
                    "allocated_feed_ton_per_year": 10.0,
                    "effective_processing_budget_ton_per_year": 20.0,
                    "candidate_capacity_cap_ton_per_year": 15.0,
                    "scenario_feed_budget_ton_per_year": 20.0,
                    "planning_energy_intensity_mj_per_ton": 10.0,
                    "planning_environment_intensity_kgco2e_per_ton": 5.0,
                    "planning_carbon_load_kgco2e_per_ton": 1.0,
                    "planning_cost_intensity_proxy_or_real_per_ton": 2.0,
                    "process_temperature_c": 500.0,
                    "residence_time_min": 30.0,
                    "manure_subtype": "beef",
                    "pathway": "pyrolysis",
                }
            ]
        ),
        planning_constraints=pd.DataFrame(
            [
                {
                    "scenario_name": "baseline_region_case",
                    "capacity_binding_reason": "feed_budget",
                }
            ]
        ),
        scenario_decision_stability=pd.DataFrame(
            [
                {
                    "scenario_name": "baseline_region_case",
                    "sample_id": "sample-1",
                }
            ]
        ),
        scenario_uncertainty_summary=pd.DataFrame(
            [
                {
                    "scenario_name": "baseline_region_case",
                    "energy_range_ratio": 0.01,
                    "environment_range_ratio": 0.40,
                    "cost_range_ratio": 0.10,
                    "coverage_range_ratio": 0.02,
                    "unmet_feed_max": 0.0,
                    "dominant_selection_rate": 1.0,
                    "stable_candidate_count": 1,
                }
            ]
        ),
        scenario_cross_stability=pd.DataFrame(
            [
                {
                    "sample_id": "sample-1",
                }
            ]
        ),
        planning_run_config={"objective_weights": {"preset_name": "balanced_cleaner_production", "weights": {}}},
        scenario_run_config={},
    )

    monkeypatch.setattr(operation_inputs, "load_operation_input_bundle", lambda **kwargs: bundle)
    monkeypatch.setattr(operation_inputs, "_validate_operation_input_freshness", lambda _bundle: None)

    specs = build_operation_environment_specs()
    row = specs.iloc[0]

    assert row["avg_share_source"] == "baseline_allocation_share_fallback"
    assert row["max_share_source"] == "baseline_allocation_share_fallback"
    assert row["cross_scenario_selection_rate_source"] == "cross_scenario_default_zero"
    assert row["selected_in_all_scenarios_source"] == "cross_scenario_default_false"
    assert row["energy_disturbance_source"] == "lower_clip"
    assert row["environment_disturbance_source"] == "upper_clip"
    assert row["cost_disturbance_source"] == "direct"


def test_uncertainty_summary_preserves_missing_ratio_when_mean_is_zero():
    stress = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "stress_test_name": "test-a",
                "top_portfolio_case_id": "case-1",
                "portfolio_energy_objective": 0.0,
                "portfolio_environment_objective": 0.0,
                "portfolio_cost_objective": 0.0,
                "scenario_feed_coverage_ratio": 0.0,
                "remaining_unmet_feed_ton_per_year": 0.0,
            },
            {
                "scenario_name": "baseline_region_case",
                "stress_test_name": "test-b",
                "top_portfolio_case_id": "case-2",
                "portfolio_energy_objective": 0.0,
                "portfolio_environment_objective": 0.0,
                "portfolio_cost_objective": 0.0,
                "scenario_feed_coverage_ratio": 0.0,
                "remaining_unmet_feed_ton_per_year": 0.0,
            },
        ]
    )

    summary = build_uncertainty_summary(stress_test_summary=stress, decision_stability=pd.DataFrame())

    assert pd.isna(summary.loc[0, "energy_range_ratio"])
    assert pd.isna(summary.loc[0, "coverage_range_ratio"])
    assert pd.isna(summary.loc[0, "dominant_selection_rate"])
