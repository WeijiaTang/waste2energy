from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import RANDOM_STATE
from .environment import OperationEnvironmentSpec
from .evaluate import evaluate_trained_model, run_environment_check, save_model
from .gym_env import build_gym_env

try:
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.monitor import Monitor
except Exception as exc:  # pragma: no cover - optional dependency guard
    SAC = None
    TD3 = None
    Monitor = None
    _SB3_IMPORT_ERROR = exc
else:
    _SB3_IMPORT_ERROR = None


ALGORITHM_REGISTRY = {
    "sac": SAC,
    "td3": TD3,
}


def train_rl_agents(
    specs: pd.DataFrame,
    *,
    algorithm: str,
    total_timesteps: int,
    seeds: list[int] | None = None,
    horizon_steps: int = 12,
    evaluation_episodes: int = 5,
    model_output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_sb3()
    algorithm_key = algorithm.lower()
    if algorithm_key not in ALGORITHM_REGISTRY or ALGORITHM_REGISTRY[algorithm_key] is None:
        raise ValueError(f"Unsupported RL algorithm: {algorithm}")

    model_root = Path(model_output_dir) if model_output_dir else Path("outputs") / "operation" / "rl_models"
    active_seeds = list(seeds) if seeds else [RANDOM_STATE]
    summary_rows: list[dict[str, object]] = []
    rollout_rows: list[pd.DataFrame] = []
    episode_rows: list[pd.DataFrame] = []

    for _, row in specs.iterrows():
        spec = OperationEnvironmentSpec(**row.to_dict())
        run_environment_check(spec, horizon_steps=horizon_steps)
        for seed in active_seeds:
            env = Monitor(build_gym_env(spec, horizon_steps=horizon_steps))
            model_class = ALGORITHM_REGISTRY[algorithm_key]
            model = model_class(
                "MlpPolicy",
                env,
                verbose=0,
                seed=seed,
                learning_starts=max(10, min(100, total_timesteps // 4)),
                batch_size=min(64, max(16, total_timesteps // 4)),
            )
            model.learn(total_timesteps=total_timesteps, progress_bar=False)

            metrics, evaluation_rollout, evaluation_episode_summary = evaluate_trained_model(
                model,
                spec,
                horizon_steps=horizon_steps,
                evaluation_episodes=evaluation_episodes,
            )
            model_path = model_root / algorithm_key / spec.scenario_name / f"seed_{seed}" / "model"
            save_model(model, model_path)
            env.close()

            summary_rows.append(
                {
                    "scenario_name": spec.scenario_name,
                    "algorithm": algorithm_key,
                    "seed": int(seed),
                    "total_timesteps": int(total_timesteps),
                    "model_path": str(model_path),
                    **metrics,
                    "dominant_case_id": spec.dominant_case_id,
                    "dominant_sample_id": spec.dominant_sample_id,
                }
            )
            evaluation_rollout["scenario_name"] = spec.scenario_name
            evaluation_rollout["algorithm"] = algorithm_key
            evaluation_rollout["seed"] = int(seed)
            rollout_rows.append(evaluation_rollout)

            evaluation_episode_summary["scenario_name"] = spec.scenario_name
            evaluation_episode_summary["algorithm"] = algorithm_key
            evaluation_episode_summary["seed"] = int(seed)
            evaluation_episode_summary["dominant_case_id"] = spec.dominant_case_id
            evaluation_episode_summary["dominant_sample_id"] = spec.dominant_sample_id
            episode_rows.append(evaluation_episode_summary)

    training_summary = pd.DataFrame(summary_rows).sort_values(
        ["scenario_name", "algorithm", "seed"]
    ).reset_index(drop=True)
    evaluation_rollouts = pd.concat(rollout_rows, ignore_index=True) if rollout_rows else pd.DataFrame()
    evaluation_episode_summary = pd.concat(episode_rows, ignore_index=True) if episode_rows else pd.DataFrame()
    seed_aggregate = build_seed_aggregate_summary(evaluation_episode_summary)
    return training_summary, evaluation_rollouts, evaluation_episode_summary, seed_aggregate


def build_seed_aggregate_summary(evaluation_episode_summary: pd.DataFrame) -> pd.DataFrame:
    if evaluation_episode_summary.empty:
        return pd.DataFrame()

    per_seed = (
        evaluation_episode_summary.groupby(
            ["scenario_name", "algorithm", "seed", "dominant_case_id", "dominant_sample_id"],
            dropna=False,
        )
        .agg(
            mean_reward=("total_reward", "mean"),
            mean_average_reward=("average_reward", "mean"),
            mean_realized_energy=("total_realized_energy", "mean"),
            mean_realized_environment=("total_realized_environment", "mean"),
            mean_realized_cost=("total_realized_cost", "mean"),
            mean_max_violation_penalty=("max_violation_penalty", "mean"),
        )
        .reset_index()
    )
    aggregate = (
        per_seed.groupby(["scenario_name", "algorithm", "dominant_case_id", "dominant_sample_id"], dropna=False)
        .agg(
            seed_count=("seed", "nunique"),
            reward_mean=("mean_reward", "mean"),
            reward_std=("mean_reward", "std"),
            average_reward_mean=("mean_average_reward", "mean"),
            energy_mean=("mean_realized_energy", "mean"),
            environment_mean=("mean_realized_environment", "mean"),
            cost_mean=("mean_realized_cost", "mean"),
            max_violation_mean=("mean_max_violation_penalty", "mean"),
        )
        .reset_index()
        .fillna({"reward_std": 0.0})
    )
    return aggregate.sort_values(["scenario_name", "algorithm"]).reset_index(drop=True)


def _require_sb3() -> None:
    if SAC is None or TD3 is None or Monitor is None:
        raise ImportError(
            "stable-baselines3 is required for RL training/evaluation. "
            "Install the project dependencies to enable this layer."
        ) from _SB3_IMPORT_ERROR
