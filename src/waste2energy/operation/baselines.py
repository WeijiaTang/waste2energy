from __future__ import annotations

from dataclasses import asdict
from typing import Callable

import pandas as pd

from .environment import OperationEnvironment, OperationEnvironmentSpec


PolicyFn = Callable[[dict[str, float], OperationEnvironmentSpec], tuple[int, int]]


def run_baseline_policies(
    specs: pd.DataFrame,
    *,
    horizon_steps: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    step_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []

    policies: dict[str, PolicyFn] = {
        "hold_plan": hold_plan_policy,
        "track_target": track_target_policy,
        "energy_push": energy_push_policy,
    }

    for _, row in specs.iterrows():
        spec = OperationEnvironmentSpec(**row.to_dict())
        for policy_name, policy_fn in policies.items():
            environment = OperationEnvironment(spec, horizon_steps=horizon_steps)
            state = environment.reset()

            total_reward = 0.0
            total_energy = 0.0
            total_environment = 0.0
            total_cost = 0.0
            max_violation = 0.0

            for step_index in range(1, horizon_steps + 1):
                action = policy_fn(state, spec)
                next_state, reward, done, info = environment.step(action)
                components = info["reward_components"]

                total_reward += reward
                total_energy += components["realized_energy"]
                total_environment += components["realized_environment"]
                total_cost += components["realized_cost"]
                max_violation = max(max_violation, components["violation_penalty"])

                step_rows.append(
                    {
                        "scenario_name": spec.scenario_name,
                        "policy_name": policy_name,
                        "step_index": step_index,
                        "throughput_action": action[0],
                        "severity_action": action[1],
                        "reward": reward,
                        "throughput_ton_per_year": next_state["throughput_ton_per_year"],
                        "candidate_share_of_effective_budget": next_state[
                            "candidate_share_of_effective_budget"
                        ],
                        "severity_offset": next_state["severity_offset"],
                        "energy_disturbance_multiplier": next_state["energy_disturbance_multiplier"],
                        "environment_disturbance_multiplier": next_state[
                            "environment_disturbance_multiplier"
                        ],
                        "cost_disturbance_multiplier": next_state["cost_disturbance_multiplier"],
                        "capacity_pressure": next_state["capacity_pressure"],
                        "coverage_pressure": next_state["coverage_pressure"],
                        "realized_energy": components["realized_energy"],
                        "realized_environment": components["realized_environment"],
                        "realized_cost": components["realized_cost"],
                        "violation_penalty": components["violation_penalty"],
                        "switching_penalty": components["switching_penalty"],
                    }
                )
                state = next_state
                if done:
                    break

            episode_rows.append(
                {
                    "scenario_name": spec.scenario_name,
                    "policy_name": policy_name,
                    "total_reward": total_reward,
                    "average_reward": total_reward / horizon_steps,
                    "total_realized_energy": total_energy,
                    "total_realized_environment": total_environment,
                    "total_realized_cost": total_cost,
                    "max_violation_penalty": max_violation,
                    "final_candidate_share": state["candidate_share_of_effective_budget"],
                    "final_severity_offset": state["severity_offset"],
                    **_spec_summary(spec),
                }
            )

    return (
        pd.DataFrame(step_rows).sort_values(["scenario_name", "policy_name", "step_index"]).reset_index(drop=True),
        pd.DataFrame(episode_rows).sort_values(["scenario_name", "total_reward"], ascending=[True, False]).reset_index(
            drop=True
        ),
    )


def hold_plan_policy(state: dict[str, float], spec: OperationEnvironmentSpec) -> tuple[int, int]:
    return (0, 0)


def track_target_policy(state: dict[str, float], spec: OperationEnvironmentSpec) -> tuple[int, int]:
    gap = state["candidate_share_target_gap"]
    if gap < -0.01:
        throughput_action = 1
    elif gap > 0.01:
        throughput_action = -1
    else:
        throughput_action = 0
    severity_action = 0
    return (throughput_action, severity_action)


def energy_push_policy(state: dict[str, float], spec: OperationEnvironmentSpec) -> tuple[int, int]:
    throughput_action = 0
    if state["candidate_share_of_effective_budget"] < spec.scenario_candidate_share_target - 0.02:
        throughput_action = 1
    elif state["capacity_pressure"] > 0.0:
        throughput_action = -1

    if state["energy_disturbance_multiplier"] < 1.0:
        severity_action = 1
    elif state["cost_disturbance_multiplier"] > 1.08:
        severity_action = -1
    else:
        severity_action = 0
    return (throughput_action, severity_action)


def _spec_summary(spec: OperationEnvironmentSpec) -> dict[str, object]:
    payload = asdict(spec)
    keys = [
        "dominant_sample_id",
        "dominant_case_id",
        "manure_subtype",
        "candidate_feed_target_ton_per_year",
        "candidate_capacity_cap_ton_per_year",
        "dominant_selection_rate",
    ]
    return {key: payload[key] for key in keys}
