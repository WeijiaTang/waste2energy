from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from ..common import parse_manifest_timestamp
from ..config import OPERATION_OUTPUTS_DIR
from .baselines import run_baseline_policies
from .artifacts import (
    write_operation_comparison_outputs,
    write_operation_outputs,
    write_operation_rl_outputs,
)
from .comparison import (
    build_baseline_policy_summary,
    build_comparison_table,
    build_policy_behavior_comparison,
    build_rl_policy_behavior_summary,
    build_rl_policy_summary,
)
from .inputs import build_operation_environment_specs, load_operation_input_bundle
from .train_rl import train_rl_agents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Waste2Energy planning-derived appendix operation environment."
    )
    parser.add_argument("--planning-dir", default="", help="Optional explicit planning output directory.")
    parser.add_argument("--scenario-dir", default="", help="Optional explicit scenario output directory.")
    parser.add_argument("--output-dir", default="", help="Optional explicit operation output directory.")
    parser.add_argument("--horizon-steps", type=int, default=8760, help="Number of control steps per episode.")
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
        default=8760,
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
        default="42,43,44,45,46",
        help="Comma-separated random seeds for RL mode or compare mode.",
    )
    parser.add_argument(
        "--force-retrain-rl",
        action="store_true",
        help="Ignore any reusable RL artifacts and retrain RL policies from scratch.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    try:
        input_bundle = load_operation_input_bundle(
            planning_dir=args.planning_dir or None,
            scenario_dir=args.scenario_dir or None,
        )
        environment_specs = build_operation_environment_specs(
            planning_dir=args.planning_dir or None,
            scenario_dir=args.scenario_dir or None,
            bundle=input_bundle,
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
                planning_run_config=input_bundle.planning_run_config,
                scenario_run_config=input_bundle.scenario_run_config,
                horizon_steps=args.horizon_steps,
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
            policy_behavior_summary = build_rl_policy_behavior_summary(evaluation_rollouts)
            outputs = write_operation_rl_outputs(
                environment_specs=environment_specs,
                training_summary=training_summary,
                evaluation_rollouts=evaluation_rollouts,
                evaluation_episode_summary=evaluation_episode_summary,
                seed_aggregate_summary=seed_aggregate_summary,
                policy_behavior_summary=policy_behavior_summary,
                output_dir=args.output_dir or None,
                algorithm=args.algorithm,
                planning_run_config=input_bundle.planning_run_config,
                scenario_run_config=input_bundle.scenario_run_config,
                seeds=seeds,
                total_timesteps=args.total_timesteps,
                evaluation_episodes=args.evaluation_episodes,
                horizon_steps=args.horizon_steps,
            )
            payload = {
                "mode": "rl",
                "algorithm": args.algorithm,
                "environment_count": int(len(environment_specs)),
                "training_run_count": int(len(training_summary)),
                "seed_aggregate_count": int(len(seed_aggregate_summary)),
                "outputs": outputs,
            }
        else:
            baseline_rollout_steps, baseline_rollout_summary = run_baseline_policies(
                environment_specs,
                horizon_steps=args.horizon_steps,
            )
            baseline_policy_summary = build_baseline_policy_summary(baseline_rollout_summary)

            rl_training_summaries: dict[str, object] = {}
            rl_evaluation_rollouts: dict[str, object] = {}
            rl_evaluation_episode_summaries: dict[str, object] = {}
            rl_seed_aggregate_summaries: dict[str, object] = {}
            rl_policy_summaries = []
            for algorithm in ["sac", "td3"]:
                (
                    training_summary,
                    evaluation_rollouts,
                    evaluation_episode_summary,
                    seed_aggregate_summary,
                    policy_behavior_summary,
                ) = _load_or_train_rl_artifacts(
                    environment_specs=environment_specs,
                    input_bundle=input_bundle,
                    algorithm=algorithm,
                    total_timesteps=args.total_timesteps,
                    seeds=seeds,
                    horizon_steps=args.horizon_steps,
                    evaluation_episodes=args.evaluation_episodes,
                    force_retrain=args.force_retrain_rl,
                )
                rl_training_summaries[algorithm] = training_summary
                rl_evaluation_rollouts[algorithm] = evaluation_rollouts
                rl_evaluation_episode_summaries[algorithm] = evaluation_episode_summary
                rl_seed_aggregate_summaries[algorithm] = seed_aggregate_summary
                rl_policy_summaries.append(build_rl_policy_summary(seed_aggregate_summary))

            comparison_table = build_comparison_table(
                baseline_policy_summary,
                rl_policy_summaries,
            )
            policy_behavior_comparison = build_policy_behavior_comparison(
                baseline_rollout_steps,
                list(rl_evaluation_rollouts.values()) if rl_evaluation_rollouts else [],
            )
            outputs = write_operation_comparison_outputs(
                baseline_rollout_steps=baseline_rollout_steps,
                baseline_summary=baseline_policy_summary,
                rl_training_summaries=rl_training_summaries,
                rl_evaluation_rollouts=rl_evaluation_rollouts,
                rl_evaluation_episode_summaries=rl_evaluation_episode_summaries,
                rl_seed_aggregate_summaries=rl_seed_aggregate_summaries,
                comparison_table=comparison_table,
                policy_behavior_comparison=policy_behavior_comparison,
                output_dir=args.output_dir or None,
                seeds=seeds,
                total_timesteps=args.total_timesteps,
                planning_run_config=input_bundle.planning_run_config,
                scenario_run_config=input_bundle.scenario_run_config,
                evaluation_episodes=args.evaluation_episodes,
                horizon_steps=args.horizon_steps,
            )
            payload = {
                "mode": "compare",
                "environment_count": int(len(environment_specs)),
                "comparison_row_count": int(len(comparison_table)),
                "policy_behavior_row_count": int(len(policy_behavior_comparison)),
                "outputs": outputs,
            }
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2))
    return 0


def _load_or_train_rl_artifacts(
    *,
    environment_specs: pd.DataFrame,
    input_bundle,
    algorithm: str,
    total_timesteps: int,
    seeds: list[int],
    horizon_steps: int,
    evaluation_episodes: int,
    force_retrain: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not force_retrain:
        cached = _load_cached_rl_artifacts(
            algorithm=algorithm,
            input_bundle=input_bundle,
            total_timesteps=total_timesteps,
            seeds=seeds,
            horizon_steps=horizon_steps,
            evaluation_episodes=evaluation_episodes,
        )
        if cached is not None:
            return cached

    training_summary, evaluation_rollouts, evaluation_episode_summary, seed_aggregate_summary = train_rl_agents(
        environment_specs,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        seeds=seeds,
        horizon_steps=horizon_steps,
        evaluation_episodes=evaluation_episodes,
    )
    policy_behavior_summary = build_rl_policy_behavior_summary(evaluation_rollouts)
    write_operation_rl_outputs(
        environment_specs=environment_specs,
        training_summary=training_summary,
        evaluation_rollouts=evaluation_rollouts,
        evaluation_episode_summary=evaluation_episode_summary,
        seed_aggregate_summary=seed_aggregate_summary,
        policy_behavior_summary=policy_behavior_summary,
        output_dir=None,
        algorithm=algorithm,
        planning_run_config=input_bundle.planning_run_config,
        scenario_run_config=input_bundle.scenario_run_config,
        seeds=seeds,
        total_timesteps=total_timesteps,
        evaluation_episodes=evaluation_episodes,
        horizon_steps=horizon_steps,
    )
    return (
        training_summary,
        evaluation_rollouts,
        evaluation_episode_summary,
        seed_aggregate_summary,
        policy_behavior_summary,
    )


def _load_cached_rl_artifacts(
    *,
    algorithm: str,
    input_bundle,
    total_timesteps: int,
    seeds: list[int],
    horizon_steps: int,
    evaluation_episodes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    rl_dir = OPERATION_OUTPUTS_DIR / "rl" / algorithm
    run_config_path = rl_dir / "run_config.json"
    if not run_config_path.exists():
        return None

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    if not _rl_run_config_is_compatible(
        run_config=run_config,
        input_bundle=input_bundle,
        total_timesteps=total_timesteps,
        seeds=seeds,
        horizon_steps=horizon_steps,
        evaluation_episodes=evaluation_episodes,
    ):
        return None

    required_files = {
        "training_summary": rl_dir / "training_summary.csv",
        "evaluation_rollouts": rl_dir / "evaluation_rollouts.csv",
        "evaluation_episode_summary": rl_dir / "evaluation_episode_summary.csv",
        "seed_aggregate_summary": rl_dir / "seed_aggregate_summary.csv",
        "policy_behavior_summary": rl_dir / "policy_behavior_summary.csv",
    }
    if not all(path.exists() for path in required_files.values()):
        return None

    return tuple(pd.read_csv(path) for path in required_files.values())  # type: ignore[return-value]


def _rl_run_config_is_compatible(
    *,
    run_config: dict[str, object],
    input_bundle,
    total_timesteps: int,
    seeds: list[int],
    horizon_steps: int,
    evaluation_episodes: int,
) -> bool:
    planning_timestamp = parse_manifest_timestamp(input_bundle.planning_run_config)
    scenario_timestamp = parse_manifest_timestamp(input_bundle.scenario_run_config)
    cached_planning_timestamp = _parse_cached_timestamp(run_config.get("source_planning_generated_at_utc"))
    cached_scenario_timestamp = _parse_cached_timestamp(run_config.get("source_scenario_generated_at_utc"))

    if planning_timestamp and cached_planning_timestamp and cached_planning_timestamp < planning_timestamp:
        return False
    if scenario_timestamp and cached_scenario_timestamp and cached_scenario_timestamp < scenario_timestamp:
        return False

    if planning_timestamp and cached_planning_timestamp is None:
        return False
    if scenario_timestamp and cached_scenario_timestamp is None:
        return False

    if int(run_config.get("total_timesteps", -1) or -1) != int(total_timesteps):
        return False
    if int(run_config.get("evaluation_episodes", -1) or -1) != int(evaluation_episodes):
        return False
    if int(run_config.get("horizon_steps", -1) or -1) != int(horizon_steps):
        return False

    cached_seeds = [int(value) for value in list(run_config.get("seeds", []))]
    if sorted(cached_seeds) != sorted(int(seed) for seed in seeds):
        return False

    return True


def _parse_cached_timestamp(value: object):
    if not value:
        return None
    try:
        return parse_manifest_timestamp({"generated_at_utc": str(value)})
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
