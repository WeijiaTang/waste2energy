from __future__ import annotations

import argparse
import json
import sys

from .baselines import run_baseline_policies
from .artifacts import (
    write_operation_comparison_outputs,
    write_operation_outputs,
    write_operation_rl_outputs,
)
from .comparison import (
    build_baseline_policy_summary,
    build_comparison_table,
    build_rl_policy_summary,
)
from .inputs import build_operation_environment_specs
from .train_rl import train_rl_agents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Waste2Energy planning-derived appendix operation environment."
    )
    parser.add_argument("--planning-dir", default="", help="Optional explicit planning output directory.")
    parser.add_argument("--scenario-dir", default="", help="Optional explicit scenario output directory.")
    parser.add_argument("--output-dir", default="", help="Optional explicit operation output directory.")
    parser.add_argument("--horizon-steps", type=int, default=12, help="Number of control steps per episode.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "rl", "compare"],
        default="baseline",
        help="Run deterministic baselines, SB3 RL training, or the formal RL-vs-baseline comparison.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["sac", "td3"],
        default="sac",
        help="RL algorithm used when --mode rl is selected.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=256,
        help="Training timesteps per scenario for RL mode.",
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=5,
        help="Number of deterministic episodes used for RL evaluation.",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated random seeds for RL mode or compare mode.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    try:
        environment_specs = build_operation_environment_specs(
            planning_dir=args.planning_dir or None,
            scenario_dir=args.scenario_dir or None,
        )
        if args.mode == "baseline":
            rollout_steps, rollout_summary = run_baseline_policies(
                environment_specs,
                horizon_steps=args.horizon_steps,
            )
            outputs = write_operation_outputs(
                environment_specs=environment_specs,
                rollout_steps=rollout_steps,
                rollout_summary=rollout_summary,
                output_dir=args.output_dir or None,
            )
            payload = {
                "mode": "baseline",
                "environment_count": int(len(environment_specs)),
                "policy_episode_count": int(len(rollout_summary)),
                "outputs": outputs,
            }
        elif args.mode == "rl":
            training_summary, evaluation_rollouts, evaluation_episode_summary, seed_aggregate_summary = train_rl_agents(
                environment_specs,
                algorithm=args.algorithm,
                total_timesteps=args.total_timesteps,
                seeds=seeds,
                horizon_steps=args.horizon_steps,
                evaluation_episodes=args.evaluation_episodes,
            )
            outputs = write_operation_rl_outputs(
                environment_specs=environment_specs,
                training_summary=training_summary,
                evaluation_rollouts=evaluation_rollouts,
                output_dir=args.output_dir or None,
                algorithm=args.algorithm,
            )
            payload = {
                "mode": "rl",
                "algorithm": args.algorithm,
                "environment_count": int(len(environment_specs)),
                "training_run_count": int(len(training_summary)),
                "outputs": outputs,
            }
        else:
            _, baseline_rollout_summary = run_baseline_policies(
                environment_specs,
                horizon_steps=args.horizon_steps,
            )
            baseline_policy_summary = build_baseline_policy_summary(baseline_rollout_summary)

            rl_training_summaries: dict[str, object] = {}
            rl_seed_aggregate_summaries: dict[str, object] = {}
            rl_policy_summaries = []
            for algorithm in ["sac", "td3"]:
                training_summary, _, _, seed_aggregate_summary = train_rl_agents(
                    environment_specs,
                    algorithm=algorithm,
                    total_timesteps=args.total_timesteps,
                    seeds=seeds,
                    horizon_steps=args.horizon_steps,
                    evaluation_episodes=args.evaluation_episodes,
                )
                rl_training_summaries[algorithm] = training_summary
                rl_seed_aggregate_summaries[algorithm] = seed_aggregate_summary
                rl_policy_summaries.append(build_rl_policy_summary(seed_aggregate_summary))

            comparison_table = build_comparison_table(
                baseline_policy_summary,
                rl_policy_summaries,
            )
            outputs = write_operation_comparison_outputs(
                baseline_summary=baseline_policy_summary,
                rl_training_summaries=rl_training_summaries,
                rl_seed_aggregate_summaries=rl_seed_aggregate_summaries,
                comparison_table=comparison_table,
                output_dir=args.output_dir or None,
            )
            payload = {
                "mode": "compare",
                "environment_count": int(len(environment_specs)),
                "comparison_row_count": int(len(comparison_table)),
                "outputs": outputs,
            }
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
