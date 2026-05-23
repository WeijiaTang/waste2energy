# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

"""High-intensity, reproducible final-case runner for SCI review artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmarking import run_planning_benchmark_suite
from .common import build_reproducibility_manifest, write_reproducibility_manifest
from .config import OUTPUTS_ROOT
from .operation.artifacts import write_operation_outputs
from .operation.baselines import run_baseline_policies
from .operation.inputs import build_operation_environment_specs, load_operation_input_bundle
from .planning.ablations import run_targeted_planning_ablations
from .planning.solve import PlanningConfig, run_planning_baseline
from .results_quality import write_results_quality_report
from .scenarios.run import run_scenario_robustness_baseline


@dataclass(frozen=True)
class HighCaseProfile:
    name: str
    pareto_point_count: int
    bootstrap_replicates: int
    targeted_monte_carlo_replicates: int
    operation_horizon_steps: int


PROFILES = {
    "smoke": HighCaseProfile("smoke", pareto_point_count=6, bootstrap_replicates=2, targeted_monte_carlo_replicates=8, operation_horizon_steps=168),
    "standard": HighCaseProfile("standard", pareto_point_count=24, bootstrap_replicates=16, targeted_monte_carlo_replicates=64, operation_horizon_steps=8760),
    "high": HighCaseProfile("high", pareto_point_count=48, bootstrap_replicates=64, targeted_monte_carlo_replicates=256, operation_horizon_steps=8760),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run final high-case Waste2Energy SCI artifacts.")
    parser.add_argument("--dataset-path", default="", help="Optional explicit planning dataset path.")
    parser.add_argument("--output-root", default="", help="Root output directory for the high-case run.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    parser.add_argument("--optimization-method", choices=["auto", "pyomo", "scipy"], default="scipy")
    parser.add_argument("--pyomo-solver", choices=["auto", "appsi_highs", "highs", "glpk", "cbc"], default="auto")
    parser.add_argument("--minimum-surrogate-artifact-test-r2", type=float, default=0.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--targeted-monte-carlo-replicates", type=int, default=None)
    parser.add_argument("--pareto-point-count", type=int, default=None)
    parser.add_argument("--operation-horizon-steps", type=int, default=None)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-targeted-ablations", action="store_true")
    parser.add_argument("--skip-scenario", action="store_true")
    parser.add_argument("--skip-operation", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_highcase(
        dataset_path=args.dataset_path or None,
        output_root=args.output_root or None,
        profile_name=args.profile,
        optimization_method=args.optimization_method,
        pyomo_solver_preference=args.pyomo_solver,
        minimum_surrogate_artifact_test_r2=args.minimum_surrogate_artifact_test_r2,
        bootstrap_replicates=args.bootstrap_replicates,
        targeted_monte_carlo_replicates=args.targeted_monte_carlo_replicates,
        pareto_point_count=args.pareto_point_count,
        operation_horizon_steps=args.operation_horizon_steps,
        run_benchmark=not args.skip_benchmark,
        run_targeted_ablations=not args.skip_targeted_ablations,
        run_scenario=not args.skip_scenario,
        run_operation=not args.skip_operation,
    )
    print(json.dumps(result, indent=2))
    return 0


def run_highcase(
    *,
    dataset_path: str | None = None,
    output_root: str | Path | None = None,
    profile_name: str = "standard",
    optimization_method: str = "scipy",
    pyomo_solver_preference: str = "auto",
    minimum_surrogate_artifact_test_r2: float | None = 0.0,
    bootstrap_replicates: int | None = None,
    targeted_monte_carlo_replicates: int | None = None,
    pareto_point_count: int | None = None,
    operation_horizon_steps: int | None = None,
    run_benchmark: bool = True,
    run_targeted_ablations: bool = True,
    run_scenario: bool = True,
    run_operation: bool = True,
) -> dict[str, Any]:
    profile = PROFILES[profile_name]
    root = Path(output_root) if output_root else OUTPUTS_ROOT / "sci_highcase" / profile.name
    root.mkdir(parents=True, exist_ok=True)
    planning_dir = root / "planning"
    benchmark_dir = root / "benchmark"
    targeted_dir = root / "targeted_planning_ablations"
    scenario_dir = root / "scenarios"
    operation_dir = root / "operation"
    quality_dir = root / "quality"
    for directory in (planning_dir, benchmark_dir, targeted_dir, scenario_dir, operation_dir, quality_dir):
        directory.mkdir(parents=True, exist_ok=True)

    active_pareto = profile.pareto_point_count if pareto_point_count is None else int(pareto_point_count)
    active_bootstrap = profile.bootstrap_replicates if bootstrap_replicates is None else int(bootstrap_replicates)
    active_mc = (
        profile.targeted_monte_carlo_replicates
        if targeted_monte_carlo_replicates is None
        else int(targeted_monte_carlo_replicates)
    )
    active_horizon = profile.operation_horizon_steps if operation_horizon_steps is None else int(operation_horizon_steps)

    config = PlanningConfig(
        optimization_method=optimization_method,
        pyomo_solver_preference=pyomo_solver_preference,
        pareto_point_count=active_pareto,
        enable_pareto_export=active_pareto > 0,
        minimum_surrogate_artifact_test_r2=minimum_surrogate_artifact_test_r2,
    )

    outputs: dict[str, Any] = {}
    outputs["planning"] = run_planning_baseline(
        dataset_path=dataset_path,
        output_dir=str(planning_dir),
        config=config,
    )
    if run_benchmark:
        outputs["benchmark"] = run_planning_benchmark_suite(
            dataset_path=dataset_path,
            output_dir=str(benchmark_dir),
            base_config=PlanningConfig(
                optimization_method=optimization_method,
                pyomo_solver_preference=pyomo_solver_preference,
                pareto_point_count=0,
                enable_pareto_export=False,
                minimum_surrogate_artifact_test_r2=minimum_surrogate_artifact_test_r2,
            ),
            bootstrap_replicates=active_bootstrap,
            bootstrap_random_seed=42,
        )
    if run_targeted_ablations:
        outputs["targeted_ablations"] = run_targeted_planning_ablations(
            dataset_path=dataset_path,
            output_dir=str(targeted_dir),
            base_config=PlanningConfig(
                optimization_method=optimization_method,
                pyomo_solver_preference=pyomo_solver_preference,
                pareto_point_count=0,
                enable_pareto_export=False,
                minimum_surrogate_artifact_test_r2=minimum_surrogate_artifact_test_r2,
            ),
            monte_carlo_replicates=active_mc,
            monte_carlo_random_seed=42,
        )
    if run_scenario:
        outputs["scenario"] = run_scenario_robustness_baseline(
            dataset_path=dataset_path,
            output_dir=str(scenario_dir),
            planning_dir=str(planning_dir),
            base_config=config,
        )
    if run_operation and run_scenario:
        outputs["operation"] = _run_operation_baseline(
            planning_dir=planning_dir,
            scenario_dir=scenario_dir,
            output_dir=operation_dir,
            horizon_steps=active_horizon,
        )

    quality_outputs = write_results_quality_report(
        output_dir=quality_dir,
        planning_dir=planning_dir,
        benchmark_dir=benchmark_dir if run_benchmark else None,
        scenario_dir=scenario_dir if run_scenario else None,
        operation_dir=operation_dir if run_operation and run_scenario else None,
    )
    manifest = build_reproducibility_manifest(
        command="waste2energy-highcase",
        inputs=[dataset_path] if dataset_path else [],
        outputs=[
            planning_dir / "run_config.json",
            planning_dir / "reproducibility_manifest.json",
            benchmark_dir / "benchmark_statistical_summary.csv",
            scenario_dir / "decision_stability.csv",
            operation_dir / "rollout_summary.csv",
            quality_dir / "quality_gate_summary.csv",
        ],
        parameters={
            "profile": profile.name,
            "pareto_point_count": active_pareto,
            "bootstrap_replicates": active_bootstrap,
            "targeted_monte_carlo_replicates": active_mc,
            "operation_horizon_steps": active_horizon,
            "optimization_method": optimization_method,
            "pyomo_solver_preference": pyomo_solver_preference,
            "minimum_surrogate_artifact_test_r2": minimum_surrogate_artifact_test_r2,
            "run_benchmark": run_benchmark,
            "run_targeted_ablations": run_targeted_ablations,
            "run_scenario": run_scenario,
            "run_operation": run_operation,
        },
    )
    highcase_manifest = write_reproducibility_manifest(root / "highcase_manifest.json", manifest)
    return {
        "profile": profile.name,
        "output_root": str(root),
        "planning_dir": str(planning_dir),
        "benchmark_dir": str(benchmark_dir) if run_benchmark else "",
        "scenario_dir": str(scenario_dir) if run_scenario else "",
        "operation_dir": str(operation_dir) if run_operation and run_scenario else "",
        "quality_outputs": quality_outputs,
        "highcase_manifest": str(highcase_manifest),
        "outputs": outputs,
    }


def _run_operation_baseline(
    *,
    planning_dir: Path,
    scenario_dir: Path,
    output_dir: Path,
    horizon_steps: int,
) -> dict[str, object]:
    input_bundle = load_operation_input_bundle(planning_dir=planning_dir, scenario_dir=scenario_dir)
    environment_specs = build_operation_environment_specs(
        planning_dir=planning_dir,
        scenario_dir=scenario_dir,
        bundle=input_bundle,
    )
    rollout_steps, rollout_summary = run_baseline_policies(
        environment_specs,
        horizon_steps=horizon_steps,
    )
    outputs = write_operation_outputs(
        environment_specs=environment_specs,
        rollout_steps=rollout_steps,
        rollout_summary=rollout_summary,
        output_dir=output_dir,
        planning_run_config=input_bundle.planning_run_config,
        scenario_run_config=input_bundle.scenario_run_config,
        horizon_steps=horizon_steps,
    )
    return {
        "environment_count": int(len(environment_specs)),
        "policy_episode_count": int(len(rollout_summary)),
        "horizon_steps": int(horizon_steps),
        "outputs": outputs,
    }


if __name__ == "__main__":
    raise SystemExit(main())
