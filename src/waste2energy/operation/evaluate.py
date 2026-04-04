from __future__ import annotations

from pathlib import Path

import pandas as pd

from .environment import OperationEnvironmentSpec
from .gym_env import build_gym_env

try:
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
except Exception as exc:  # pragma: no cover - optional dependency guard
    check_env = None
    evaluate_policy = None
    Monitor = None
    _SB3_IMPORT_ERROR = exc
else:
    _SB3_IMPORT_ERROR = None


def run_environment_check(spec: OperationEnvironmentSpec, *, horizon_steps: int = 12) -> None:
    _require_sb3()
    env = Monitor(build_gym_env(spec, horizon_steps=horizon_steps))
    check_env(env, warn=True)
    env.close()


def evaluate_trained_model(
    model,
    spec: OperationEnvironmentSpec,
    *,
    horizon_steps: int = 12,
    evaluation_episodes: int = 5,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    _require_sb3()
    env = Monitor(build_gym_env(spec, horizon_steps=horizon_steps))
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=evaluation_episodes,
        deterministic=True,
        return_episode_rewards=False,
    )

    rollout_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    for episode_index in range(1, evaluation_episodes + 1):
        observation, info = env.reset()
        done = False
        step_index = 0
        episode_reward = 0.0
        episode_energy = 0.0
        episode_environment = 0.0
        episode_cost = 0.0
        max_violation = 0.0
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, step_info = env.step(action)
            done = bool(terminated or truncated)
            step_index += 1
            episode_reward += float(reward)
            components = step_info["reward_components"]
            episode_energy += float(components["realized_energy"])
            episode_environment += float(components["realized_environment"])
            episode_cost += float(components["realized_cost"])
            max_violation = max(max_violation, float(components["violation_penalty"]))
            rollout_rows.append(
                {
                    "episode_index": episode_index,
                    "step_index": step_index,
                    "reward": reward,
                    "throughput_ton_per_year": step_info["state"]["throughput_ton_per_year"],
                    "candidate_share_of_effective_budget": step_info["state"][
                        "candidate_share_of_effective_budget"
                    ],
                    "severity_offset": step_info["state"]["severity_offset"],
                    "realized_energy": components["realized_energy"],
                    "realized_environment": components["realized_environment"],
                    "realized_cost": components["realized_cost"],
                    "violation_penalty": components["violation_penalty"],
                }
            )
        episode_rows.append(
            {
                "episode_index": episode_index,
                "total_reward": episode_reward,
                "average_reward": episode_reward / horizon_steps,
                "total_realized_energy": episode_energy,
                "total_realized_environment": episode_environment,
                "total_realized_cost": episode_cost,
                "max_violation_penalty": max_violation,
                "final_candidate_share": step_info["state"]["candidate_share_of_effective_budget"],
                "final_severity_offset": step_info["state"]["severity_offset"],
            }
        )

    env.close()
    return {
        "evaluation_mean_reward": float(mean_reward),
        "evaluation_std_reward": float(std_reward),
        "evaluation_episodes": int(evaluation_episodes),
    }, pd.DataFrame(rollout_rows), pd.DataFrame(episode_rows)


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def _require_sb3() -> None:
    if check_env is None or evaluate_policy is None or Monitor is None:
        raise ImportError(
            "stable-baselines3 is required for RL training/evaluation. "
            "Install the project dependencies to enable this layer."
        ) from _SB3_IMPORT_ERROR
