# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from waste2energy.operation.baselines import run_baseline_policies
from waste2energy.operation.inputs import build_operation_environment_specs


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
    assert rollout_summary["policy_name"].isin({"hold_plan", "track_target", "energy_push"}).all()
