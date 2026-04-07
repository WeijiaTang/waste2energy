# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from waste2energy.config import get_objective_weight_system
from waste2energy.operation.inputs import (
    _validate_operation_input_freshness,
    build_operation_environment_specs,
    load_operation_input_bundle,
)


def test_weight_system_is_shared_between_planning_and_operation(workflow_dirs):
    run_config = json.loads((workflow_dirs["planning_dir"] / "run_config.json").read_text(encoding="utf-8"))
    weights = run_config["objective_weights"]["weights"]
    expected = get_objective_weight_system(
        preset_name=run_config["objective_weights"]["preset_name"],
        energy=weights["energy"],
        environment=weights["environment"],
        cost=weights["cost"],
    )

    specs = build_operation_environment_specs(
        planning_dir=workflow_dirs["planning_dir"],
        scenario_dir=workflow_dirs["scenario_dir"],
    )
    assert not specs.empty
    assert specs["reward_energy_weight"].nunique() == 1
    assert specs["reward_environment_weight"].nunique() == 1
    assert specs["reward_cost_weight"].nunique() == 1
    assert specs["reward_energy_weight"].iloc[0] == expected.energy
    assert specs["reward_environment_weight"].iloc[0] == expected.environment
    assert specs["reward_cost_weight"].iloc[0] == expected.cost


def test_operation_inputs_require_scenario_outputs_not_older_than_planning(workflow_dirs):
    bundle = load_operation_input_bundle(
        planning_dir=workflow_dirs["planning_dir"],
        scenario_dir=workflow_dirs["scenario_dir"],
    )
    stale_bundle = type(bundle)(
        planning_portfolio_allocations=bundle.planning_portfolio_allocations,
        planning_constraints=bundle.planning_constraints,
        scenario_decision_stability=bundle.scenario_decision_stability,
        scenario_uncertainty_summary=bundle.scenario_uncertainty_summary,
        scenario_cross_stability=bundle.scenario_cross_stability,
        planning_run_config={
            **bundle.planning_run_config,
            "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        },
        scenario_run_config={
            **bundle.scenario_run_config,
            "generated_at_utc": (
                datetime.now(UTC).replace(microsecond=0) - timedelta(minutes=5)
            ).isoformat(),
        },
    )

    with pytest.raises(ValueError, match="Scenario outputs are older than the planning outputs"):
        _validate_operation_input_freshness(stale_bundle)
