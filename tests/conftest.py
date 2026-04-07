# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from pathlib import Path

import pytest

from waste2energy.planning.solve import PlanningConfig, run_planning_baseline
from waste2energy.scenarios.run import run_scenario_robustness_baseline


@pytest.fixture(scope="session")
def workflow_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("wte_workflow")
    planning_dir = root / "planning"
    scenario_dir = root / "scenarios"

    config = PlanningConfig(pareto_point_count=6)
    run_planning_baseline(output_dir=str(planning_dir), config=config)
    run_scenario_robustness_baseline(
        output_dir=str(scenario_dir),
        planning_dir=str(planning_dir),
        base_config=config,
    )

    return {
        "root": root,
        "planning_dir": planning_dir,
        "scenario_dir": scenario_dir,
    }
