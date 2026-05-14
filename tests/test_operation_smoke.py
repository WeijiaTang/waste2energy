# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json

import pandas as pd
import pytest

from waste2energy.operation import cli as operation_cli
from waste2energy.operation.baselines import run_baseline_policies
from waste2energy.operation.artifacts import write_operation_outputs
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


def test_write_operation_outputs_records_source_manifest_timestamps(tmp_path, workflow_dirs):
    specs = build_operation_environment_specs(
        planning_dir=workflow_dirs["planning_dir"],
        scenario_dir=workflow_dirs["scenario_dir"],
    )
    rollout_steps, rollout_summary = run_baseline_policies(specs, horizon_steps=4)

    outputs = write_operation_outputs(
        environment_specs=specs,
        rollout_steps=rollout_steps,
        rollout_summary=rollout_summary,
        output_dir=str(tmp_path / "operation_baseline"),
        planning_run_config={"generated_at_utc": "2026-04-08T00:29:16+00:00"},
        scenario_run_config={"generated_at_utc": "2026-04-08T00:29:25+00:00"},
    )
    run_config = json.loads((tmp_path / "operation_baseline" / "run_config.json").read_text(encoding="utf-8"))

    assert outputs["run_config"] == str(tmp_path / "operation_baseline" / "run_config.json")
    assert run_config["source_planning_generated_at_utc"] == "2026-04-08T00:29:16+00:00"
    assert run_config["source_scenario_generated_at_utc"] == "2026-04-08T00:29:25+00:00"


def test_load_or_train_rl_artifacts_reuses_fresh_cached_outputs(tmp_path, monkeypatch):
    operation_root = tmp_path / "operation"
    rl_dir = operation_root / "rl" / "sac"
    rl_dir.mkdir(parents=True)

    training_summary = pd.DataFrame([{"scenario_name": "baseline_region_case", "algorithm": "sac", "seed": 42}])
    evaluation_rollouts = pd.DataFrame([{"scenario_name": "baseline_region_case", "algorithm": "sac", "seed": 42}])
    evaluation_episode_summary = pd.DataFrame([{"scenario_name": "baseline_region_case", "algorithm": "sac", "seed": 42}])
    seed_aggregate_summary = pd.DataFrame(
        [{"scenario_name": "baseline_region_case", "algorithm": "sac", "seed_count": 1}]
    )
    policy_behavior_summary = pd.DataFrame(
        [{"scenario_name": "baseline_region_case", "method_name": "sac", "method_type": "rl_agent"}]
    )
    training_summary.to_csv(rl_dir / "training_summary.csv", index=False)
    evaluation_rollouts.to_csv(rl_dir / "evaluation_rollouts.csv", index=False)
    evaluation_episode_summary.to_csv(rl_dir / "evaluation_episode_summary.csv", index=False)
    seed_aggregate_summary.to_csv(rl_dir / "seed_aggregate_summary.csv", index=False)
    policy_behavior_summary.to_csv(rl_dir / "policy_behavior_summary.csv", index=False)
    (rl_dir / "run_config.json").write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-04-08T01:00:00+00:00",
                "source_planning_generated_at_utc": "2026-04-08T00:29:16+00:00",
                "source_scenario_generated_at_utc": "2026-04-08T00:29:25+00:00",
                "total_timesteps": 512,
                "evaluation_episodes": 5,
                "horizon_steps": 8760,
                "seeds": [42],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(operation_cli, "OPERATION_OUTPUTS_DIR", operation_root)
    monkeypatch.setattr(
        operation_cli,
        "train_rl_agents",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("train_rl_agents should not be called")),
    )

    bundle = OperationInputBundle(
        planning_portfolio_allocations=pd.DataFrame(),
        planning_constraints=pd.DataFrame(),
        scenario_decision_stability=pd.DataFrame(),
        scenario_uncertainty_summary=pd.DataFrame(),
        scenario_cross_stability=pd.DataFrame(),
        planning_run_config={"generated_at_utc": "2026-04-08T00:29:16+00:00"},
        scenario_run_config={"generated_at_utc": "2026-04-08T00:29:25+00:00"},
    )

    outputs = operation_cli._load_or_train_rl_artifacts(
        environment_specs=pd.DataFrame(),
        input_bundle=bundle,
        algorithm="sac",
        total_timesteps=512,
        seeds=[42],
        horizon_steps=8760,
        evaluation_episodes=5,
        force_retrain=False,
    )

    assert outputs[0].equals(training_summary)
    assert outputs[1].equals(evaluation_rollouts)
    assert outputs[2].equals(evaluation_episode_summary)
    assert outputs[3].equals(seed_aggregate_summary)
    assert outputs[4].equals(policy_behavior_summary)
