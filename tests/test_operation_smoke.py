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
    assert len(rollout_steps) > 0
    assert len(rollout_summary) > 0
    assert rollout_summary["policy_name"].isin({"hold_plan", "track_target", "energy_push"}).all()

