from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common import build_run_manifest, write_json
from ..config import OPERATION_OUTPUTS_DIR


def write_operation_outputs(
    *,
    environment_specs: pd.DataFrame,
    rollout_steps: pd.DataFrame,
    rollout_summary: pd.DataFrame,
    output_dir: str | None,
) -> dict[str, str]:
    target_dir = Path(output_dir) if output_dir else OPERATION_OUTPUTS_DIR / "baseline"
    target_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "environment_specs": target_dir / "operation_environment_specs.csv",
        "rollout_steps": target_dir / "baseline_rollout_steps.csv",
        "rollout_summary": target_dir / "baseline_rollout_summary.csv",
        "run_config": target_dir / "run_config.json",
    }

    environment_specs.to_csv(outputs["environment_specs"], index=False)
    rollout_steps.to_csv(outputs["rollout_steps"], index=False)
    rollout_summary.to_csv(outputs["rollout_summary"], index=False)
    write_json(
        outputs["run_config"],
        build_run_manifest(
            scenario_names=sorted(environment_specs["scenario_name"].unique().tolist())
            if not environment_specs.empty
            else [],
            environment_count=int(len(environment_specs)),
            rollout_episode_count=int(len(rollout_summary)),
            output_files={key: str(path) for key, path in outputs.items()},
            operation_layer_status="planning_derived_environment_ready_baseline_policies_only",
        ),
    )
    return {key: str(path) for key, path in outputs.items()}


def write_operation_rl_outputs(
    *,
    environment_specs: pd.DataFrame,
    training_summary: pd.DataFrame,
    evaluation_rollouts: pd.DataFrame,
    evaluation_episode_summary: pd.DataFrame,
    seed_aggregate_summary: pd.DataFrame,
    policy_behavior_summary: pd.DataFrame,
    output_dir: str | None,
    algorithm: str,
) -> dict[str, str]:
    base_dir = Path(output_dir) if output_dir else OPERATION_OUTPUTS_DIR / "rl" / algorithm
    base_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "environment_specs": base_dir / "operation_environment_specs.csv",
        "training_summary": base_dir / "training_summary.csv",
        "evaluation_rollouts": base_dir / "evaluation_rollouts.csv",
        "evaluation_episode_summary": base_dir / "evaluation_episode_summary.csv",
        "seed_aggregate_summary": base_dir / "seed_aggregate_summary.csv",
        "policy_behavior_summary": base_dir / "policy_behavior_summary.csv",
        "run_config": base_dir / "run_config.json",
    }
    environment_specs.to_csv(outputs["environment_specs"], index=False)
    training_summary.to_csv(outputs["training_summary"], index=False)
    evaluation_rollouts.to_csv(outputs["evaluation_rollouts"], index=False)
    evaluation_episode_summary.to_csv(outputs["evaluation_episode_summary"], index=False)
    seed_aggregate_summary.to_csv(outputs["seed_aggregate_summary"], index=False)
    policy_behavior_summary.to_csv(outputs["policy_behavior_summary"], index=False)
    write_json(
        outputs["run_config"],
        build_run_manifest(
            scenario_names=sorted(environment_specs["scenario_name"].unique().tolist())
            if not environment_specs.empty
            else [],
            environment_count=int(len(environment_specs)),
            training_run_count=int(len(training_summary)),
            evaluation_episode_count=int(len(evaluation_episode_summary)),
            algorithm=algorithm,
            output_files={key: str(path) for key, path in outputs.items()},
            operation_layer_status="gymnasium_env_with_sb3_training_and_evaluation",
        ),
    )
    return {key: str(path) for key, path in outputs.items()}


def write_operation_comparison_outputs(
    *,
    baseline_rollout_steps: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    rl_training_summaries: dict[str, pd.DataFrame],
    rl_evaluation_rollouts: dict[str, pd.DataFrame],
    rl_evaluation_episode_summaries: dict[str, pd.DataFrame],
    rl_seed_aggregate_summaries: dict[str, pd.DataFrame],
    comparison_table: pd.DataFrame,
    policy_behavior_comparison: pd.DataFrame,
    output_dir: str | None,
    seeds: list[int] | None = None,
    total_timesteps: int | None = None,
) -> dict[str, str]:
    base_dir = Path(output_dir) if output_dir else OPERATION_OUTPUTS_DIR / "comparison"
    base_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "baseline_rollout_steps": base_dir / "baseline_rollout_steps.csv",
        "baseline_summary": base_dir / "baseline_policy_summary.csv",
        "comparison_table": base_dir / "rl_vs_baseline_comparison.csv",
        "policy_behavior_comparison": base_dir / "policy_behavior_comparison.csv",
        "run_config": base_dir / "run_config.json",
    }
    baseline_rollout_steps.to_csv(outputs["baseline_rollout_steps"], index=False)
    baseline_summary.to_csv(outputs["baseline_summary"], index=False)
    comparison_table.to_csv(outputs["comparison_table"], index=False)
    policy_behavior_comparison.to_csv(outputs["policy_behavior_comparison"], index=False)

    for algorithm, frame in rl_training_summaries.items():
        (base_dir / f"{algorithm}_training_summary.csv").write_text(
            frame.to_csv(index=False), encoding="utf-8"
        )
    for algorithm, frame in rl_evaluation_rollouts.items():
        (base_dir / f"{algorithm}_evaluation_rollouts.csv").write_text(
            frame.to_csv(index=False), encoding="utf-8"
        )
    for algorithm, frame in rl_evaluation_episode_summaries.items():
        (base_dir / f"{algorithm}_evaluation_episode_summary.csv").write_text(
            frame.to_csv(index=False), encoding="utf-8"
        )
    for algorithm, frame in rl_seed_aggregate_summaries.items():
        (base_dir / f"{algorithm}_seed_aggregate_summary.csv").write_text(
            frame.to_csv(index=False), encoding="utf-8"
        )

    extra_files = {
        f"{algorithm}_training_summary": str(base_dir / f"{algorithm}_training_summary.csv")
        for algorithm in rl_training_summaries
    }
    extra_files.update(
        {
            f"{algorithm}_evaluation_rollouts": str(base_dir / f"{algorithm}_evaluation_rollouts.csv")
            for algorithm in rl_evaluation_rollouts
        }
    )
    extra_files.update(
        {
            f"{algorithm}_evaluation_episode_summary": str(base_dir / f"{algorithm}_evaluation_episode_summary.csv")
            for algorithm in rl_evaluation_episode_summaries
        }
    )
    extra_files.update(
        {
            f"{algorithm}_seed_aggregate_summary": str(base_dir / f"{algorithm}_seed_aggregate_summary.csv")
            for algorithm in rl_seed_aggregate_summaries
        }
    )
    write_json(
        outputs["run_config"],
        build_run_manifest(
            baseline_step_count=int(len(baseline_rollout_steps)),
            baseline_policy_count=int(len(baseline_summary)),
            comparison_row_count=int(len(comparison_table)),
            policy_behavior_row_count=int(len(policy_behavior_comparison)),
            algorithms=sorted(rl_training_summaries.keys()),
            seeds=list(seeds or []),
            total_timesteps=total_timesteps,
            output_files={**{key: str(path) for key, path in outputs.items()}, **extra_files},
            operation_layer_status="multi_seed_rl_vs_baseline_comparison_ready",
        ),
    )
    return {
        **{key: str(path) for key, path in outputs.items()},
        **extra_files,
    }
