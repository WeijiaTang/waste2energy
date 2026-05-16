# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmarking import run_planning_benchmark_suite
from .planning.ablations import run_targeted_planning_ablations
from .planning.solve import PlanningConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Waste2Energy planning ablation and benchmark suite."
    )
    parser.add_argument("--dataset-path", default="", help="Optional explicit planning dataset path.")
    parser.add_argument("--output-dir", default="", help="Optional explicit benchmark output directory.")
    parser.add_argument(
        "--optimization-method",
        choices=["auto", "pyomo", "scipy"],
        default="auto",
        help="Optimization backend preference for planner-based benchmark variants.",
    )
    parser.add_argument(
        "--pyomo-solver",
        choices=["auto", "appsi_highs", "highs", "glpk", "cbc"],
        default="auto",
        help="Preferred Pyomo solver backend when planner variants use Pyomo optimization.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=0,
        help="Optional number of bootstrap benchmark repeats for uncertainty intervals. Default 0 keeps the standard benchmark fast.",
    )
    parser.add_argument(
        "--bootstrap-random-seed",
        type=int,
        default=42,
        help="Random seed used when bootstrap benchmark repeats are enabled.",
    )
    parser.add_argument(
        "--skip-targeted-ablations",
        action="store_true",
        help="Run only the benchmark variants and skip Q1-facing targeted planning ablations.",
    )
    parser.add_argument(
        "--targeted-monte-carlo-replicates",
        type=int,
        default=48,
        help="Monte Carlo replanning samples exported with targeted ablations.",
    )
    parser.add_argument(
        "--targeted-monte-carlo-random-seed",
        type=int,
        default=42,
        help="Random seed used by targeted Monte Carlo UQ.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        base_config = PlanningConfig(
            optimization_method=args.optimization_method,
            pyomo_solver_preference=args.pyomo_solver,
            pareto_point_count=0,
            enable_pareto_export=False,
        )
        result = run_planning_benchmark_suite(
            dataset_path=args.dataset_path or None,
            output_dir=args.output_dir or None,
            base_config=base_config,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_random_seed=args.bootstrap_random_seed,
        )
        if not args.skip_targeted_ablations:
            targeted_output_dir = (
                str(Path(args.output_dir) / "targeted_planning_ablations")
                if args.output_dir
                else None
            )
            result["targeted_ablations"] = run_targeted_planning_ablations(
                dataset_path=args.dataset_path or None,
                output_dir=targeted_output_dir,
                base_config=base_config,
                monte_carlo_replicates=args.targeted_monte_carlo_replicates,
                monte_carlo_random_seed=args.targeted_monte_carlo_random_seed,
            )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
