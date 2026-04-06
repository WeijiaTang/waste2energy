from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from adjustText import adjust_text

from common import (
    CLAIM_COLORS,
    PATHWAY_COLORS,
    PATHWAY_LABELS,
    PATHWAY_ORDER,
    RESULTS_PAPER_DIR,
    SCENARIO_LABELS,
    SELECTION_COLORS,
    add_panel_label,
    configure_plotting,
    format_pct,
    load_planning_visual_bundle,
    save_figure_set,
    scenario_label,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot manuscript-grade planning figures for Paper 1.",
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Directory containing paper1_planning_visual_*.csv files. Defaults to data/processed/figures_tables.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write manuscript figures. Defaults to results/paper.",
    )
    return parser


def prepare_competition_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.loc[metrics["metric_key"] == "best_case_score_index"].copy()
    frame["scenario_display"] = frame["scenario_name"].map(SCENARIO_LABELS)
    frame["pathway_display"] = frame["pathway"].map(PATHWAY_LABELS)
    return frame.sort_values(["scenario_order", "pathway_order"]).reset_index(drop=True)


def prepare_tradeoff_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[
        metrics["metric_key"].isin(["best_case_energy_pj_per_year", "best_case_environment_ktco2e_per_year"])
    ].copy()
    frame = (
        selected.pivot_table(
            index=[
                "scenario_name",
                "scenario_order",
                "pathway",
                "pathway_order",
                "pathway_rank_within_scenario",
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
    )
    frame["scenario_display"] = frame["scenario_name"].map(SCENARIO_LABELS)
    frame["pathway_display"] = frame["pathway"].map(PATHWAY_LABELS)
    return frame.sort_values(["scenario_order", "pathway_order"]).reset_index(drop=True)


def prepare_robustness_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[
        metrics["metric_key"].isin(["baseline_portfolio_share_pct", "max_stress_selection_rate"])
    ].copy()
    frame = (
        selected.pivot_table(
            index=[
                "scenario_name",
                "scenario_order",
                "pathway",
                "pathway_order",
                "pathway_rank_within_scenario",
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
    )
    frame["scenario_display"] = frame["scenario_name"].map(SCENARIO_LABELS)
    frame["pathway_display"] = frame["pathway"].map(PATHWAY_LABELS)
    return frame.sort_values(["scenario_order", "pathway_order"]).reset_index(drop=True)


def plot_score_competition(plt, frame: pd.DataFrame):
    scenarios = frame["scenario_name"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7.2, 2.6), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    for idx, (ax, scenario_name) in enumerate(zip(axes, scenarios, strict=True)):
        subset = frame.loc[frame["scenario_name"] == scenario_name].sort_values("pathway_order")
        y_positions = list(range(len(subset)))[::-1]
        colors = [
            PATHWAY_COLORS.get(pathway, "#777777") if selected else "#D9DDE3"
            for pathway, selected in zip(
                subset["pathway"],
                subset["selected_in_baseline_portfolio"],
                strict=True,
            )
        ]
        edges = [
            "#111111" if selected else "#D9DDE3"
            for selected in subset["selected_in_baseline_portfolio"]
        ]
        bars = ax.barh(
            y_positions,
            subset["value"],
            color=colors,
            edgecolor=edges,
            linewidth=1.0,
            height=0.62,
        )
        for bar, (_, row) in zip(bars, subset.iterrows(), strict=True):
            ax.text(
                bar.get_width() + 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{row['value']:.3f}",
                va="center",
                ha="left",
                fontsize=7.0,
                color="#222222",
            )
        ax.set_yticks(y_positions, subset["pathway_display"])
        ax.set_title(scenario_label(scenario_name), pad=8)
        ax.set_xlabel("Best-case score index")
        ax.set_xlim(0.0, max(0.78, subset["value"].max() + 0.06))
        ax.grid(axis="x", color="#D7DBE2", linewidth=0.7)
        ax.grid(axis="y", visible=False)
        if idx == 0:
            ax.set_ylabel("Pathway")
        else:
            ax.set_ylabel("")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    add_panel_label(axes[0], "A")
    fig.suptitle(
        "Pyrolysis becomes the leading planning pathway after real cost activation",
        y=0.995,
        fontsize=9.8,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return fig


def plot_tradeoff_map(plt, frame: pd.DataFrame):
    scenarios = frame["scenario_name"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7.2, 2.8), sharex=True, sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    for idx, (ax, scenario_name) in enumerate(zip(axes, scenarios, strict=True)):
        subset = frame.loc[frame["scenario_name"] == scenario_name].sort_values("pathway_order")
        texts = []
        for _, row in subset.iterrows():
            score = float(row["pathway_rank_within_scenario"])
            bubble_size = max(85.0, 320.0 - 35.0 * score)
            ax.scatter(
                row["best_case_energy_pj_per_year"],
                row["best_case_environment_ktco2e_per_year"],
                s=bubble_size,
                color=PATHWAY_COLORS.get(str(row["pathway"]), "#777777"),
                edgecolor="#111111" if bool(row["selected_in_baseline_portfolio"]) else "white",
                linewidth=1.0,
                zorder=3,
            )
            texts.append(
                ax.text(
                    row["best_case_energy_pj_per_year"],
                    row["best_case_environment_ktco2e_per_year"],
                    row["pathway_display"],
                    fontsize=7.0,
                    color="#222222",
                )
            )
        adjust_text(
            texts,
            ax=ax,
            expand=(1.08, 1.18),
            force_points=0.25,
            force_text=0.3,
        )
        ax.set_title(scenario_label(scenario_name), pad=8)
        ax.set_xlabel("Energy benefit (PJ/year)")
        ax.grid(color="#D7DBE2", linewidth=0.7)
        if idx == 0:
            ax.set_ylabel("Environment benefit (ktCO2e/year)")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    add_panel_label(axes[0], "B")
    fig.suptitle(
        "HTC remains energy-rich, but pyrolysis is selected after economic scoring",
        y=0.995,
        fontsize=9.8,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return fig


def plot_robustness_matrix(plt, frame: pd.DataFrame):
    metric_specs = [
        ("baseline_portfolio_share_pct", "Baseline portfolio share"),
        ("max_stress_selection_rate", "Stress-test support"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    y_order = [SCENARIO_LABELS[name] for name in frame["scenario_name"].drop_duplicates().tolist()]
    x_order = [PATHWAY_LABELS[name] for name in PATHWAY_ORDER]

    for idx, (ax, (metric_key, title)) in enumerate(zip(axes, metric_specs, strict=True)):
        for _, row in frame.iterrows():
            x = x_order.index(PATHWAY_LABELS[str(row["pathway"])])
            y = y_order.index(SCENARIO_LABELS[str(row["scenario_name"])])
            value = float(row[metric_key])
            ax.scatter(
                x,
                y,
                s=max(40.0, value * 8.0),
                color=CLAIM_COLORS.get(str(row["claim_color_group"]), "#7C7C7C"),
                edgecolor="#111111" if bool(row["selected_in_baseline_portfolio"]) else "white",
                linewidth=0.9,
                zorder=3,
            )
            ax.text(
                x,
                y,
                format_pct(value),
                ha="center",
                va="center",
                fontsize=6.6,
                color="#111111" if value >= 25.0 else "#333333",
                zorder=4,
            )
        ax.set_title(title, pad=8)
        ax.set_xticks(range(len(x_order)), x_order, rotation=18)
        ax.set_yticks(range(len(y_order)))
        ax.set_ylim(-0.5, len(y_order) - 0.5)
        ax.invert_yaxis()
        ax.set_xlim(-0.6, len(x_order) - 0.4)
        ax.set_xlabel("Pathway")
        ax.grid(color="#E1E5EA", linewidth=0.7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        if idx == 0:
            ax.set_yticklabels([""] * len(y_order))
            for y_value, label in enumerate(y_order):
                ax.text(
                    -0.72,
                    y_value,
                    label,
                    ha="right",
                    va="center",
                    fontsize=7.2,
                    color="#222222",
                    clip_on=False,
                )
            ax.set_ylabel("Scenario")
        else:
            ax.set_yticklabels([""] * len(y_order))
            ax.set_ylabel("")

    add_panel_label(axes[0], "C")
    fig.suptitle(
        "Portfolio dominance stays with pyrolysis across the tested stress set",
        y=0.995,
        fontsize=9.8,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return fig


def build_figure_manifest(
    *,
    outputs: dict[str, dict[str, str]],
    source_manifest: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    return {
        "output_dir": str(output_dir),
        "source_manifest": source_manifest,
        "figure_outputs": outputs,
        "recommended_usage": {
            "figure_1": "score competition panel",
            "figure_2": "energy-environment tradeoff panel",
            "figure_3": "portfolio and robustness matrix",
        },
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir) if args.figures_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_PAPER_DIR

    plt, _ = configure_plotting()
    metrics, _, source_manifest = load_planning_visual_bundle(figures_dir=figures_dir)

    competition = prepare_competition_frame(metrics)
    tradeoff = prepare_tradeoff_frame(metrics)
    robustness = prepare_robustness_frame(metrics)

    outputs: dict[str, dict[str, str]] = {}

    fig1 = plot_score_competition(plt, competition)
    outputs["paper1_fig1_planning_score_competition"] = save_figure_set(
        fig1,
        "paper1_fig1_planning_score_competition",
        output_dir=output_dir,
    )
    plt.close(fig1)

    fig2 = plot_tradeoff_map(plt, tradeoff)
    outputs["paper1_fig2_planning_tradeoff_map"] = save_figure_set(
        fig2,
        "paper1_fig2_planning_tradeoff_map",
        output_dir=output_dir,
    )
    plt.close(fig2)

    fig3 = plot_robustness_matrix(plt, robustness)
    outputs["paper1_fig3_planning_robustness_matrix"] = save_figure_set(
        fig3,
        "paper1_fig3_planning_robustness_matrix",
        output_dir=output_dir,
    )
    plt.close(fig3)

    manifest = build_figure_manifest(
        outputs=outputs,
        source_manifest=source_manifest,
        output_dir=output_dir,
    )
    manifest_path = output_dir / "paper1_planning_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "figures": outputs,
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
