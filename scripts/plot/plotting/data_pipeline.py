from __future__ import annotations

from pathlib import Path
import math

import pandas as pd

from scripts.plot.common import PATHWAY_LABELS, SCENARIO_LABELS


def build_figure_ready_tables(metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    working = metrics.copy()
    working["scenario_display"] = working["scenario_name"].map(SCENARIO_LABELS).fillna(
        working["scenario_name"].astype(str)
    )
    working["pathway_display"] = working["pathway"].map(PATHWAY_LABELS).fillna(working["pathway"].astype(str))

    figure1 = _build_main_figure_table(working)
    figure2 = _build_tradeoff_table(working)
    figure3 = _build_robustness_table(working)
    figure_s1 = _build_sup_s1_scenario_fingerprint_table(working)
    figure_s2 = _build_sup_s2_dominance_landscape_table(working)
    return {
        "figure1_main": figure1,
        "figure2_tradeoff": figure2,
        "figure3_robustness": figure3,
        "paper1_sup_s1_scenario_fingerprint": figure_s1,
        "paper1_sup_s2_dominance_evidence_landscape": figure_s2,
    }


def write_figure_ready_tables(
    tables: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for name, frame in tables.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        outputs[name] = str(path)
    return outputs


def _build_main_figure_table(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[
        metrics["metric_key"].isin(
            [
                "best_case_score_index",
                "best_case_energy_pj_per_year",
                "best_case_environment_ktco2e_per_year",
                "baseline_portfolio_share_pct",
                "max_stress_selection_rate",
            ]
        )
    ].copy()
    frame = (
        selected.pivot_table(
            index=[
                "scenario_name",
                "scenario_order",
                "scenario_display",
                "pathway",
                "pathway_order",
                "pathway_display",
                "writing_label",
                "selected_in_baseline_portfolio",
                "selected_flag",
                "claim_boundary",
                "claim_color_group",
            ],
            columns="metric_key",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .rename(
            columns={
                "best_case_score_index": "score_value",
                "best_case_energy_pj_per_year": "energy_value",
                "best_case_environment_ktco2e_per_year": "environment_value",
                "baseline_portfolio_share_pct": "portfolio_share_pct",
                "max_stress_selection_rate": "stress_support_pct",
            }
        )
    )
    return frame.sort_values(["scenario_order", "pathway_order"]).reset_index(drop=True)


def _build_tradeoff_table(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[
        metrics["metric_key"].isin(
            [
                "best_case_energy_pj_per_year",
                "best_case_environment_ktco2e_per_year",
                "baseline_portfolio_share_pct",
            ]
        )
    ].copy()
    frame = (
        selected.pivot_table(
            index=[
                "scenario_name",
                "scenario_order",
                "scenario_display",
                "pathway",
                "pathway_order",
                "pathway_display",
                "selected_in_baseline_portfolio",
                "claim_color_group",
            ],
            columns="metric_key",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .rename(
            columns={
                "best_case_energy_pj_per_year": "energy_value",
                "best_case_environment_ktco2e_per_year": "environment_value",
                "baseline_portfolio_share_pct": "portfolio_share_pct",
            }
        )
    )
    return frame.sort_values(["scenario_order", "pathway_order"]).reset_index(drop=True)


def _build_robustness_table(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[
        metrics["metric_key"].isin(["baseline_portfolio_share_pct", "max_stress_selection_rate"])
    ].copy()
    frame = (
        selected.pivot_table(
            index=[
                "scenario_name",
                "scenario_order",
                "scenario_display",
                "pathway",
                "pathway_order",
                "pathway_display",
                "selected_in_baseline_portfolio",
                "claim_color_group",
                "claim_boundary",
            ],
            columns="metric_key",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .rename(
            columns={
                "baseline_portfolio_share_pct": "portfolio_share_pct",
                "max_stress_selection_rate": "stress_support_pct",
            }
        )
    )
    return frame.sort_values(["scenario_order", "pathway_order"]).reset_index(drop=True)


def _build_sup_s1_scenario_fingerprint_table(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_order = [
        "best_case_score_index",
        "baseline_portfolio_share_pct",
        "max_stress_selection_rate",
        "best_case_energy_pj_per_year",
        "best_case_environment_ktco2e_per_year",
        "score_gap_to_scenario_best_pct",
    ]
    metric_display = {
        "best_case_score_index": "Score",
        "baseline_portfolio_share_pct": "Share",
        "max_stress_selection_rate": "Support",
        "best_case_energy_pj_per_year": "Energy",
        "best_case_environment_ktco2e_per_year": "Environment",
        "score_gap_to_scenario_best_pct": "Score gap",
    }
    selected = metrics.loc[metrics["metric_key"].isin(metric_order)].copy()
    selected["metric_order"] = selected["metric_key"].map({key: index for index, key in enumerate(metric_order)})
    selected["metric_display"] = selected["metric_key"].map(metric_display)

    normalized_rows: list[pd.DataFrame] = []
    for metric_key, metric_frame in selected.groupby("metric_key", sort=False):
        available = metric_frame["value"].fillna(0.0).astype(float)
        minimum = available.min()
        maximum = available.max()
        span = maximum - minimum
        if math.isclose(span, 0.0):
            normalized = pd.Series([0.5] * len(metric_frame), index=metric_frame.index, dtype=float)
        else:
            normalized = (available - minimum) / span
        if metric_key == "score_gap_to_scenario_best_pct":
            normalized = 1.0 - normalized
        normalized = normalized * 0.78 + 0.14
        metric_copy = metric_frame.copy()
        metric_copy["normalized_value"] = normalized.round(6)
        normalized_rows.append(metric_copy)

    frame = pd.concat(normalized_rows, ignore_index=True)
    frame = frame[
        [
            "scenario_name",
            "scenario_order",
            "scenario_display",
            "pathway",
            "pathway_order",
            "pathway_display",
            "selected_in_baseline_portfolio",
            "claim_color_group",
            "metric_key",
            "metric_display",
            "metric_order",
            "value",
            "normalized_value",
        ]
    ].rename(columns={"value": "metric_value"})
    return frame.sort_values(["scenario_order", "pathway_order", "metric_order"]).reset_index(drop=True)


def _build_sup_s2_dominance_landscape_table(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[
        metrics["metric_key"].isin(
            [
                "baseline_portfolio_share_pct",
                "max_stress_selection_rate",
                "best_case_score_index",
                "score_gap_to_scenario_best_pct",
            ]
        )
    ].copy()
    frame = (
        selected.pivot_table(
            index=[
                "scenario_name",
                "scenario_order",
                "scenario_display",
                "pathway",
                "pathway_order",
                "pathway_display",
                "selected_in_baseline_portfolio",
                "selected_flag",
                "claim_boundary",
                "claim_color_group",
            ],
            columns="metric_key",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .rename(
            columns={
                "baseline_portfolio_share_pct": "portfolio_share_pct",
                "max_stress_selection_rate": "stress_support_pct",
                "best_case_score_index": "score_value",
                "score_gap_to_scenario_best_pct": "score_gap_pct",
            }
        )
    )
    frame["stress_support_pct"] = frame["stress_support_pct"].fillna(0.0)
    frame["score_gap_pct"] = frame["score_gap_pct"].fillna(100.0)
    frame["selected_core_zone"] = frame["portfolio_share_pct"].ge(50.0) & frame["stress_support_pct"].ge(30.0)
    frame["score_competitor_zone"] = frame["portfolio_share_pct"].lt(20.0) & frame["score_value"].ge(
        frame["score_value"].max() * 0.75
    )
    frame["evidence_limited_zone"] = frame["claim_color_group"].isin(["comparison_only", "anchor_only"])
    return frame.sort_values(["scenario_order", "pathway_order"]).reset_index(drop=True)
