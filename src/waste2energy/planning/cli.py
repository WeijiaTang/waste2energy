from __future__ import annotations

import argparse
import json
import sys

from .solve import PlanningConfig, run_planning_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Waste2Energy constraint-aware planning-layer baseline."
    )
    parser.add_argument("--dataset-path", default="", help="Optional explicit planning dataset path.")
    parser.add_argument("--output-dir", default="", help="Optional explicit output directory.")
    parser.add_argument("--energy-weight", type=float, default=0.40, help="Energy objective weight.")
    parser.add_argument(
        "--environment-weight", type=float, default=0.35, help="Environmental objective weight."
    )
    parser.add_argument("--cost-weight", type=float, default=0.25, help="Cost objective weight.")
    parser.add_argument(
        "--top-k-per-scenario",
        type=int,
        default=5,
        help="How many scored recommendations to export per scenario.",
    )
    parser.add_argument(
        "--max-portfolio-candidates",
        type=int,
        default=3,
        help="Maximum number of candidate designs kept in the scenario portfolio.",
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
        help="Fraction of the scenario capacity gap treated as deployable in the planning baseline.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        result = run_planning_baseline(
            dataset_path=args.dataset_path or None,
            output_dir=args.output_dir or None,
            config=PlanningConfig(
                energy_weight=args.energy_weight,
                environment_weight=args.environment_weight,
                cost_weight=args.cost_weight,
                top_k_per_scenario=args.top_k_per_scenario,
                max_portfolio_candidates=args.max_portfolio_candidates,
                max_candidate_share=args.max_candidate_share,
                max_subtype_share=args.max_subtype_share,
                min_distinct_subtypes=args.min_distinct_subtypes,
                deployable_capacity_fraction=args.deployable_capacity_fraction,
            ),
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
