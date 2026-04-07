# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import argparse
import json
import sys

from ..config import DEFAULT_OBJECTIVE_WEIGHT_PRESET, get_objective_weight_system
from ..planning.solve import PlanningConfig
from .run import run_scenario_robustness_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Waste2Energy scenario and robustness baseline."
    )
    parser.add_argument("--dataset-path", default="", help="Optional explicit planning dataset path.")
    parser.add_argument("--output-dir", default="", help="Optional explicit output directory.")
    parser.add_argument(
        "--planning-dir",
        default="",
        help="Optional explicit planning output directory used to refresh manuscript-facing planning tables.",
    )
    parser.add_argument(
        "--objective-weight-preset",
        default=DEFAULT_OBJECTIVE_WEIGHT_PRESET,
        help="Named objective-weight preset shared by planning and operation.",
    )
    parser.add_argument("--energy-weight", type=float, default=None, help="Optional energy objective weight override.")
    parser.add_argument(
        "--environment-weight",
        type=float,
        default=None,
        help="Optional environmental objective weight override.",
    )
    parser.add_argument("--cost-weight", type=float, default=None, help="Optional cost objective weight override.")
    parser.add_argument(
        "--top-k-per-scenario",
        type=int,
        default=5,
        help="How many scored recommendations to export per scenario in the baseline config.",
    )
    parser.add_argument(
        "--max-portfolio-candidates",
        type=int,
        default=3,
        help="Maximum number of candidate designs kept in the baseline portfolio.",
    )
    parser.add_argument(
        "--max-candidate-share",
        type=float,
        default=0.45,
        help="Maximum share of scenario processing budget allocated to any one candidate.",
    )
    parser.add_argument(
        "--max-subtype-share",
        type=float,
        default=0.60,
        help="Maximum share of scenario processing budget allocated to one manure subtype.",
    )
    parser.add_argument(
        "--min-distinct-subtypes",
        type=int,
        default=2,
        help="Minimum number of manure subtypes targeted during the portfolio diversity pass.",
    )
    parser.add_argument(
        "--deployable-capacity-fraction",
        type=float,
        default=0.85,
        help="Fraction of the scenario capacity gap treated as deployable in the baseline config.",
    )
    parser.add_argument(
        "--robustness-factor",
        type=float,
        default=0.35,
        help="Penalty multiplier applied to surrogate uncertainty during planning.",
    )
    parser.add_argument(
        "--carbon-budget-factor",
        type=float,
        default=1.0,
        help="Fraction of baseline-treatment carbon budget allowed in the optimized portfolio.",
    )
    parser.add_argument(
        "--constraint-relaxation-ratio",
        type=float,
        default=1.0,
        help="Multiplier applied to candidate share caps before stress tests add their own overrides.",
    )
    parser.add_argument(
        "--subtype-relaxation-ratio",
        type=float,
        default=1.0,
        help="Multiplier applied to subtype share caps before stress tests add their own overrides.",
    )
    parser.add_argument(
        "--scenario-metric-variance-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to pathway-level scenario perturbations.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        result = run_scenario_robustness_baseline(
            dataset_path=args.dataset_path or None,
            output_dir=args.output_dir or None,
            planning_dir=args.planning_dir or None,
            base_config=PlanningConfig(
                objective_weight_preset=args.objective_weight_preset,
                objective_weight_system=get_objective_weight_system(
                    preset_name=args.objective_weight_preset,
                    energy=args.energy_weight,
                    environment=args.environment_weight,
                    cost=args.cost_weight,
                ),
                top_k_per_scenario=args.top_k_per_scenario,
                max_portfolio_candidates=args.max_portfolio_candidates,
                max_candidate_share=args.max_candidate_share,
                max_subtype_share=args.max_subtype_share,
                min_distinct_subtypes=args.min_distinct_subtypes,
                deployable_capacity_fraction=args.deployable_capacity_fraction,
                robustness_factor=args.robustness_factor,
                carbon_budget_factor=args.carbon_budget_factor,
                constraint_relaxation_ratio=args.constraint_relaxation_ratio,
                subtype_relaxation_ratio=args.subtype_relaxation_ratio,
                scenario_metric_variance_scale=args.scenario_metric_variance_scale,
            ),
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
