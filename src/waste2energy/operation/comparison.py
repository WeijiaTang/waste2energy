from __future__ import annotations

import pandas as pd


def build_baseline_policy_summary(baseline_rollout_summary: pd.DataFrame) -> pd.DataFrame:
    if baseline_rollout_summary.empty:
        return pd.DataFrame()

    renamed = baseline_rollout_summary.rename(
        columns={
            "policy_name": "method_name",
            "total_reward": "reward_mean",
            "average_reward": "average_reward_mean",
            "total_realized_energy": "energy_mean",
            "total_realized_environment": "environment_mean",
            "total_realized_cost": "cost_mean",
            "max_violation_penalty": "max_violation_mean",
            "dominant_case_id": "dominant_case_id",
            "dominant_sample_id": "dominant_sample_id",
        }
    ).copy()
    renamed["method_type"] = "baseline_policy"
    renamed["reward_std"] = 0.0
    renamed["seed_count"] = 1
    columns = [
        "scenario_name",
        "method_type",
        "method_name",
        "seed_count",
        "reward_mean",
        "reward_std",
        "average_reward_mean",
        "energy_mean",
        "environment_mean",
        "cost_mean",
        "max_violation_mean",
        "dominant_case_id",
        "dominant_sample_id",
    ]
    return renamed[columns].sort_values(["scenario_name", "method_name"]).reset_index(drop=True)


def build_rl_policy_summary(seed_aggregate_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_aggregate_summary.empty:
        return pd.DataFrame()

    renamed = seed_aggregate_summary.rename(columns={"algorithm": "method_name"}).copy()
    renamed["method_type"] = "rl_agent"
    columns = [
        "scenario_name",
        "method_type",
        "method_name",
        "seed_count",
        "reward_mean",
        "reward_std",
        "average_reward_mean",
        "energy_mean",
        "environment_mean",
        "cost_mean",
        "max_violation_mean",
        "dominant_case_id",
        "dominant_sample_id",
    ]
    return renamed[columns].sort_values(["scenario_name", "method_name"]).reset_index(drop=True)


def build_comparison_table(
    baseline_summary: pd.DataFrame,
    rl_summaries: list[pd.DataFrame],
) -> pd.DataFrame:
    frames = [frame for frame in [baseline_summary, *rl_summaries] if not frame.empty]
    if not frames:
        return pd.DataFrame()

    comparison = pd.concat(frames, ignore_index=True)
    comparison["reward_rank_within_scenario"] = comparison.groupby("scenario_name")["reward_mean"].rank(
        method="dense", ascending=False
    )
    return comparison.sort_values(
        ["scenario_name", "reward_rank_within_scenario", "method_type", "method_name"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
