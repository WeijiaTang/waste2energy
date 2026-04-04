from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common import build_run_manifest, write_json
from ..config import SCENARIO_OUTPUTS_DIR


def write_scenario_outputs(
    *,
    registry: pd.DataFrame,
    stress_test_summary: pd.DataFrame,
    decision_stability: pd.DataFrame,
    cross_scenario_stability: pd.DataFrame,
    uncertainty_summary: pd.DataFrame,
    output_dir: str | None,
    dataset_path: str,
    planner_variant: str,
    objective_readiness: dict[str, str],
) -> dict[str, str]:
    target_dir = Path(output_dir) if output_dir else SCENARIO_OUTPUTS_DIR / "baseline"
    target_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "stress_registry": target_dir / "stress_registry.csv",
        "stress_test_summary": target_dir / "stress_test_summary.csv",
        "decision_stability": target_dir / "decision_stability.csv",
        "cross_scenario_stability": target_dir / "cross_scenario_stability.csv",
        "uncertainty_summary": target_dir / "uncertainty_summary.csv",
        "run_config": target_dir / "run_config.json",
    }

    registry.to_csv(outputs["stress_registry"], index=False)
    stress_test_summary.to_csv(outputs["stress_test_summary"], index=False)
    decision_stability.to_csv(outputs["decision_stability"], index=False)
    cross_scenario_stability.to_csv(outputs["cross_scenario_stability"], index=False)
    uncertainty_summary.to_csv(outputs["uncertainty_summary"], index=False)

    write_json(
        outputs["run_config"],
        build_run_manifest(
            dataset_path=dataset_path,
            planner_variant=planner_variant,
            objective_readiness=objective_readiness,
            stress_test_count=int(registry["stress_test_name"].nunique()) if not registry.empty else 0,
            output_files={key: str(path) for key, path in outputs.items()},
        ),
    )
    return {key: str(path) for key, path in outputs.items()}
