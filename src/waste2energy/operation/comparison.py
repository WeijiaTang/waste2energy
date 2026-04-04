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


def build_baseline_policy_behavior_summary(baseline_rollout_steps: pd.DataFrame) -> pd.DataFrame:
    summary = _summarize_rollout_behavior(
        baseline_rollout_steps,
        group_columns=["scenario_name", "policy_name"],
        method_name_column="policy_name",
        method_type="baseline_policy",
    )
    if summary.empty:
        return summary

    summary["seed_count"] = 1
    renamed_columns = {}
    for column in _behavior_metric_columns():
        renamed_columns[column] = f"{column}_mean"
        summary[f"{column}_std"] = 0.0
    summary = summary.rename(columns=renamed_columns)
    summary["behavior_metric_std"] = 0.0
    return summary.sort_values(["scenario_name", "method_name"]).reset_index(drop=True)


def build_rl_policy_behavior_summary(evaluation_rollouts: pd.DataFrame) -> pd.DataFrame:
    per_seed = _summarize_rollout_behavior(
        evaluation_rollouts,
        group_columns=["scenario_name", "algorithm", "seed"],
        method_name_column="algorithm",
        method_type="rl_agent",
    )
    if per_seed.empty:
        return per_seed

    metric_columns = _behavior_metric_columns()
    aggregate_map = {"seed": ("seed", "nunique")}
    for column in metric_columns:
        aggregate_map[f"{column}_mean"] = (column, "mean")
        aggregate_map[f"{column}_std"] = (column, "std")

    aggregated = (
        per_seed.groupby(["scenario_name", "method_name", "method_type"], dropna=False)
        .agg(seed_count=("seed", "nunique"), **{key: value for key, value in aggregate_map.items() if key != "seed"})
        .reset_index()
        .fillna(0.0)
    )
    aggregated["behavior_metric_std"] = aggregated["throughput_nonzero_rate_std"] + aggregated[
        "severity_nonzero_rate_std"
    ]
    return aggregated.sort_values(["scenario_name", "method_name"]).reset_index(drop=True)


def build_policy_behavior_comparison(
    baseline_rollout_steps: pd.DataFrame,
    rl_evaluation_rollouts: list[pd.DataFrame],
) -> pd.DataFrame:
    frames = [
        frame
        for frame in [
            build_baseline_policy_behavior_summary(baseline_rollout_steps),
            *[build_rl_policy_behavior_summary(frame) for frame in rl_evaluation_rollouts],
        ]
        if not frame.empty
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["scenario_name", "method_type", "method_name"]
    ).reset_index(drop=True)


def build_comparison_table(
    baseline_summary: pd.DataFrame,
    rl_summaries: list[pd.DataFrame],
) -> pd.DataFrame:
    frames = [frame for frame in [baseline_summary, *rl_summaries] if not frame.empty]
    if not frames:
        return pd.DataFrame()

    comparison = pd.concat(frames, ignore_index=True)
    comparison = _attach_hold_plan_improvement(comparison)
    comparison["violation_aware_score"] = comparison["reward_mean"] - comparison["max_violation_mean"]
    comparison["reward_rank_within_scenario"] = comparison.groupby("scenario_name")["reward_mean"].rank(
        method="dense", ascending=False
    )
    comparison["violation_aware_rank_within_scenario"] = comparison.groupby("scenario_name")[
        "violation_aware_score"
    ].rank(method="dense", ascending=False)
    return comparison.sort_values(
        [
            "scenario_name",
            "violation_aware_rank_within_scenario",
            "reward_rank_within_scenario",
            "method_type",
            "method_name",
        ],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)


def _attach_hold_plan_improvement(comparison: pd.DataFrame) -> pd.DataFrame:
    hold_plan = comparison[comparison["method_name"] == "hold_plan"][
        ["scenario_name", "reward_mean", "average_reward_mean", "max_violation_mean"]
    ].rename(
        columns={
            "reward_mean": "hold_plan_reward_mean",
            "average_reward_mean": "hold_plan_average_reward_mean",
            "max_violation_mean": "hold_plan_max_violation_mean",
        }
    )
    merged = comparison.merge(hold_plan, on="scenario_name", how="left")
    merged["reward_improvement_vs_hold_plan_abs"] = (
        merged["reward_mean"] - merged["hold_plan_reward_mean"]
    )
    merged["reward_improvement_vs_hold_plan_pct"] = (
        merged["reward_improvement_vs_hold_plan_abs"] / merged["hold_plan_reward_mean"].replace(0.0, pd.NA)
    ).fillna(0.0)
    merged["average_reward_improvement_vs_hold_plan_abs"] = (
        merged["average_reward_mean"] - merged["hold_plan_average_reward_mean"]
    )
    merged["violation_delta_vs_hold_plan"] = (
        merged["max_violation_mean"] - merged["hold_plan_max_violation_mean"]
    )
    return merged


def _summarize_rollout_behavior(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    method_name_column: str,
    method_type: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    summary = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            step_count=("step_index", "count"),
            episode_count=("episode_index", "nunique"),
            throughput_nonzero_rate=("throughput_action", lambda series: float((series != 0).mean())),
            throughput_up_rate=("throughput_action", lambda series: float((series > 0).mean())),
            throughput_down_rate=("throughput_action", lambda series: float((series < 0).mean())),
            severity_nonzero_rate=("severity_action", lambda series: float((series != 0).mean())),
            severity_up_rate=("severity_action", lambda series: float((series > 0).mean())),
            severity_down_rate=("severity_action", lambda series: float((series < 0).mean())),
            mean_abs_throughput_action=("throughput_action", lambda series: float(series.abs().mean())),
            mean_abs_severity_action=("severity_action", lambda series: float(series.abs().mean())),
            reward_per_step_mean=("reward", "mean"),
            candidate_share_mean=("candidate_share_of_effective_budget", "mean"),
            severity_offset_mean=("severity_offset", "mean"),
            capacity_pressure_mean=("capacity_pressure", "mean"),
            coverage_pressure_mean=("coverage_pressure", "mean"),
            violation_penalty_mean=("violation_penalty", "mean"),
            switching_penalty_mean=("switching_penalty", "mean"),
        )
        .reset_index()
        .rename(columns={method_name_column: "method_name"})
    )
    summary["method_type"] = method_type
    ordered_columns = ["scenario_name", "method_type", "method_name"]
    if "seed" in summary.columns:
        ordered_columns.append("seed")
    ordered_columns.extend(_behavior_metric_columns())
    return summary[ordered_columns]


def _behavior_metric_columns() -> list[str]:
    return [
        "step_count",
        "episode_count",
        "throughput_nonzero_rate",
        "throughput_up_rate",
        "throughput_down_rate",
        "severity_nonzero_rate",
        "severity_up_rate",
        "severity_down_rate",
        "mean_abs_throughput_action",
        "mean_abs_severity_action",
        "reward_per_step_mean",
        "candidate_share_mean",
        "severity_offset_mean",
        "capacity_pressure_mean",
        "coverage_pressure_mean",
        "violation_penalty_mean",
        "switching_penalty_mean",
    ]
