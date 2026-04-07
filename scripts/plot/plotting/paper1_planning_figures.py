from __future__ import annotations

from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from scripts.plot.common import scenario_label

from .annotations import add_caption_note, add_header_block, add_panel_label, add_zone_label
from .layout import create_main_figure, create_three_panel_polar_figure, create_three_panel_supporting_figure
from .theme import (
    add_landscape_zones,
    claim_color,
    configure_publication_theme,
    narrative_background,
    pathway_color,
    scenario_marker,
    soften_hex,
    style_axis,
    style_polar_axis,
)


SCENARIO_ORDER = ["baseline_region_case", "high_supply_case", "policy_support_case"]


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "scenario_display" not in working.columns:
        working["scenario_display"] = working["scenario_name"].map(scenario_label)
    if "stress_support_pct" not in working.columns:
        working["stress_support_pct"] = 0.0
    working["stress_support_pct"] = working["stress_support_pct"].fillna(0.0)
    if "claim_boundary" not in working.columns:
        working["claim_boundary"] = "evidence-qualified comparison"
    if "claim_color_group" not in working.columns:
        working["claim_color_group"] = "other"
    return working


def _pathway_handles():
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=pathway_color("pyrolysis"), markeredgecolor="none", markersize=6, label="Pyrolysis"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=pathway_color("htc"), markeredgecolor="none", markersize=6, label="HTC"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=pathway_color("ad"), markeredgecolor="none", markersize=6, label="AD"),
    ]


def _scenario_handles():
    return [
        Line2D([0], [0], marker=scenario_marker(name), color="#64748B", linestyle="none", markersize=5, label=label)
        for name, label in [
            ("baseline_region_case", "Baseline"),
            ("high_supply_case", "High supply"),
            ("policy_support_case", "Policy"),
        ]
    ]


def _close_polar(values: list[float]) -> list[float]:
    return values + values[:1]


def build_figure1_main(frame: pd.DataFrame):
    plt = configure_publication_theme()
    fig, axes = create_main_figure(plt)
    header_ax = axes["header"]
    score_ax = axes["score"]
    tradeoff_ax = axes["tradeoff"]
    evidence_ax = axes["evidence"]

    ordered = _prepare_frame(frame).sort_values(["scenario_order", "pathway_order"]).reset_index(drop=True)
    pyro = ordered.loc[ordered["pathway"] == "pyrolysis"].copy()
    htc = ordered.loc[ordered["pathway"] == "htc"].copy()

    add_header_block(
        header_ax,
        title="Figure 1. Planning decision narrative",
        subtitle="Score leadership and portfolio leadership diverge after evidence-aware constraints are applied.",
        takeaway="Pyrolysis leads selected share; HTC leads score.",
    )

    narrative_background(score_ax, facecolor="#FBFCFE")
    score_ax.set_title("A  Score", loc="left", pad=6)
    positions = list(range(len(pyro)))
    band = 0.30
    for pos, (_, row) in enumerate(pyro.iterrows()):
        paired = htc.loc[htc["scenario_name"] == row["scenario_name"]]
        paired_row = paired.iloc[0] if not paired.empty else None
        score_ax.barh(
            pos + band / 2,
            row["score_value"],
            height=band,
            color=soften_hex(pathway_color("pyrolysis"), weight=0.12),
            edgecolor="none",
            zorder=3,
        )
        if paired_row is not None:
            score_ax.barh(
                pos - band / 2,
                paired_row["score_value"],
                height=band,
                color=soften_hex(pathway_color("htc"), weight=0.12),
                edgecolor="none",
                zorder=3,
            )
        score_ax.text(-0.01, pos, row["scenario_display"], fontsize=6.7, ha="right", va="center", color="#334155")
    score_ax.set_yticks([])
    score_ax.invert_yaxis()
    score_ax.set_xlabel("Best-case score index")
    style_axis(score_ax, grid_axis="x")
    score_ax.legend(handles=_pathway_handles()[:2], loc="lower right", frameon=False, ncol=2, handletextpad=0.4, columnspacing=0.8)
    add_caption_note(score_ax, "Two bars per scenario: pyrolysis and HTC.")

    narrative_background(tradeoff_ax, facecolor="#FBFCFE")
    tradeoff_ax.set_title("B  Frontier", loc="left", pad=6)
    for _, row in ordered.loc[ordered["pathway"] != "baseline"].iterrows():
        tradeoff_ax.scatter(
            row["energy_value"],
            row["environment_value"],
            s=row["portfolio_share_pct"] * 3.2 + 28.0,
            color=pathway_color(row["pathway"]),
            marker=scenario_marker(row["scenario_name"]),
            edgecolors="white",
            linewidths=0.5,
            zorder=4 if row["selected_in_baseline_portfolio"] else 3,
        )
    tradeoff_ax.set_xlabel("Energy benefit (PJ/year)")
    tradeoff_ax.set_ylabel("Environment benefit (ktCO2e/year)")
    style_axis(tradeoff_ax, grid_axis="both")
    tradeoff_ax.legend(handles=_scenario_handles(), loc="lower right", frameon=False, ncol=1, handletextpad=0.5)
    add_caption_note(tradeoff_ax, "Area scales with share.")

    narrative_background(evidence_ax, facecolor="#FBFCFE")
    evidence_ax.set_title("C  Support", loc="left", pad=6)
    visible = ordered.loc[ordered["pathway"] != "baseline"].copy()
    for _, row in visible.iterrows():
        evidence_ax.scatter(
            row["portfolio_share_pct"],
            row["stress_support_pct"],
            s=52,
            color=claim_color(row["claim_color_group"]),
            marker=scenario_marker(row["scenario_name"]),
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )
    evidence_ax.axvline(10.0, color="#C5D0DD", linestyle=(0, (2, 2)), linewidth=0.8)
    evidence_ax.axhline(50.0, color="#C5D0DD", linestyle=(0, (2, 2)), linewidth=0.8)
    evidence_ax.set_xlim(-2, 108)
    evidence_ax.set_ylim(-2, 82)
    evidence_ax.set_xlabel("Portfolio share (%)")
    evidence_ax.set_ylabel("Stress support (%)")
    style_axis(evidence_ax, grid_axis="both")
    evidence_ax.legend(handles=_scenario_handles(), loc="upper left", frameon=False, ncol=1, handletextpad=0.5)
    add_caption_note(evidence_ax, "Guides mark visible share and moderate support.")

    fig.subplots_adjust(top=0.92, bottom=0.18, left=0.08, right=0.985)
    return fig


def build_figure2_tradeoff(frame: pd.DataFrame):
    plt = configure_publication_theme()
    fig, axes = create_three_panel_supporting_figure(plt, figsize=(10.6, 3.9))
    ordered = _prepare_frame(frame).sort_values(["scenario_order", "pathway_order"]).reset_index(drop=True)
    non_baseline = ordered.loc[ordered["pathway"] != "baseline"].copy()

    for index, scenario_name in enumerate(SCENARIO_ORDER):
        ax = axes[index]
        scenario_frame = non_baseline.loc[non_baseline["scenario_name"] == scenario_name]
        narrative_background(ax, facecolor="#FBFCFE")
        for _, row in scenario_frame.iterrows():
            ax.scatter(
                row["energy_value"],
                row["environment_value"],
                s=row["portfolio_share_pct"] * 3.8 + 34.0,
                color=pathway_color(row["pathway"]),
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
            )
            ax.text(
                row["energy_value"] + 0.12,
                row["environment_value"] + 1.4,
                row["pathway_display"],
                fontsize=6.4,
                color="#475569",
            )
        ax.set_title(scenario_label(scenario_name), fontsize=8.0, pad=6)
        ax.set_xlabel("Energy (PJ/year)")
        if index == 0:
            ax.set_ylabel("Environment (ktCO2e/year)")
        else:
            ax.set_ylabel("")
        style_axis(ax, grid_axis="both")
        add_panel_label(ax, chr(ord("A") + index))

    fig.suptitle("Figure 2. Energy-environment tradeoff by scenario", y=0.98, fontsize=11)
    axes[-1].legend(handles=_pathway_handles(), loc="lower right", frameon=False, ncol=1, handletextpad=0.5)
    fig.subplots_adjust(top=0.82, bottom=0.20, left=0.07, right=0.98, wspace=0.26)
    return fig


def build_figure3_robustness(frame: pd.DataFrame):
    plt = configure_publication_theme()
    fig, axes = create_three_panel_supporting_figure(plt, figsize=(10.6, 3.9))
    ordered = _prepare_frame(frame).sort_values(["scenario_order", "pathway_order"]).reset_index(drop=True)
    visible = ordered.loc[ordered["pathway"] != "baseline"].copy()

    claim_labels = {
        "planning_ready": "planning-ready",
        "comparison_only": "comparison-only",
        "anchor_only": "anchor",
        "other": "other",
    }

    for index, scenario_name in enumerate(SCENARIO_ORDER):
        ax = axes[index]
        scenario_frame = visible.loc[visible["scenario_name"] == scenario_name].copy()
        scenario_frame = scenario_frame.sort_values("portfolio_share_pct")
        narrative_background(ax, facecolor="#FBFCFE")
        y_positions = list(range(len(scenario_frame)))
        ax.barh(
            y_positions,
            scenario_frame["portfolio_share_pct"],
            color=[soften_hex(pathway_color(value), weight=0.15) for value in scenario_frame["pathway"]],
            edgecolor="none",
            height=0.58,
            zorder=2,
        )
        ax.scatter(
            scenario_frame["stress_support_pct"],
            y_positions,
            s=40,
            color=[claim_color(value) for value in scenario_frame["claim_color_group"]],
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )
        ax.set_title(scenario_label(scenario_name), fontsize=8.0, pad=6)
        ax.set_xlim(0, 105)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([value for value in scenario_frame["pathway_display"]], fontsize=6.6)
        ax.set_xlabel("Share / support (%)")
        if index == 0:
            ax.set_ylabel("Pathway")
        else:
            ax.set_ylabel("")
        style_axis(ax, grid_axis="x")
        add_panel_label(ax, chr(ord("A") + index))

    fig.suptitle("Figure 3. Portfolio share and robustness support by scenario", y=0.98, fontsize=11)
    legend_handles = [
        Line2D([0], [0], color="#94A3B8", linewidth=4, label="Portfolio share"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=claim_color("planning_ready"), markeredgecolor="none", markersize=5, label=claim_labels["planning_ready"]),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=claim_color("comparison_only"), markeredgecolor="none", markersize=5, label=claim_labels["comparison_only"]),
    ]
    axes[-1].legend(handles=legend_handles, loc="center right", bbox_to_anchor=(0.98, 0.18), frameon=False, ncol=1, handletextpad=0.5)
    fig.subplots_adjust(top=0.82, bottom=0.20, left=0.08, right=0.98, wspace=0.24)
    return fig


def build_sup_figure_s1_scenario_fingerprint(frame: pd.DataFrame):
    plt = configure_publication_theme()
    fig, axes = create_three_panel_polar_figure(plt, figsize=(11.0, 4.2))
    ordered = _prepare_frame(frame).sort_values(["scenario_order", "pathway_order", "metric_order"]).reset_index(drop=True)

    metric_order = (
        ordered[["metric_order", "metric_display"]]
        .drop_duplicates()
        .sort_values("metric_order")
    )
    labels = metric_order["metric_display"].tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    closed_angles = _close_polar(angles)

    for index, scenario_name in enumerate(SCENARIO_ORDER):
        ax = axes[index]
        scenario_frame = ordered.loc[ordered["scenario_name"] == scenario_name]
        style_polar_axis(ax)
        for pathway in ["pyrolysis", "htc", "ad", "baseline"]:
            path_frame = scenario_frame.loc[scenario_frame["pathway"] == pathway].sort_values("metric_order")
            if path_frame.empty:
                continue
            values = _close_polar(path_frame["normalized_value"].astype(float).tolist())
            color = pathway_color(pathway)
            fill_color = soften_hex(color, weight=0.18 if pathway in {"pyrolysis", "htc"} else 0.55)
            linewidth = 2.1 if pathway == "pyrolysis" else 1.6 if pathway == "htc" else 1.0
            ax.plot(closed_angles, values, color=color, linewidth=linewidth, zorder=4)
            if pathway in {"pyrolysis", "htc"}:
                ax.fill(closed_angles, values, color=fill_color, zorder=2)
            else:
                ax.fill(closed_angles, values, color=fill_color, zorder=1, alpha=0.58)
        ax.fill(closed_angles, [0.12] * len(closed_angles), color="white", zorder=3)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=6.6, color="#475569")
        ax.set_title(scenario_label(scenario_name), fontsize=9.0, pad=14)
        add_panel_label(ax, chr(ord("A") + index))

    fig.suptitle("Supplementary Figure S1. Scenario fingerprint", y=0.99, fontsize=11.5)
    fig.legend(
        handles=[
            Line2D([0], [0], color=pathway_color("pyrolysis"), linewidth=2.2, label="Pyrolysis"),
            Line2D([0], [0], color=pathway_color("htc"), linewidth=1.8, label="HTC"),
            Line2D([0], [0], color=pathway_color("ad"), linewidth=1.2, label="AD"),
            Line2D([0], [0], color=pathway_color("baseline"), linewidth=1.2, label="Baseline"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
    )
    add_caption_note(axes[0], "Metric spokes are normalized for cross-metric comparison.")
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.04, right=0.98, wspace=0.38)
    return fig


def build_sup_figure_s2_dominance_evidence_landscape(frame: pd.DataFrame):
    plt = configure_publication_theme()
    fig, ax = plt.subplots(1, 1, figsize=(9.6, 5.2))
    ordered = _prepare_frame(frame).sort_values(["scenario_order", "pathway_order"]).reset_index(drop=True)

    add_landscape_zones(ax)
    narrative_background(ax, facecolor="white")
    ax.set_title("Supplementary Figure S2. Dominance / evidence landscape", loc="left", pad=10)

    zone_labels = [
        (4.2, 76, "Evidence-limited"),
        (31, 76, "Latent competitor"),
        (80, 76, "Selected core"),
    ]
    for x, y, text in zone_labels:
        add_zone_label(ax, x, y, text)

    for _, row in ordered.loc[ordered["pathway"] != "baseline"].iterrows():
        x = float(row["portfolio_share_pct"])
        y = float(row["stress_support_pct"])
        color = pathway_color(row["pathway"])
        halo_color = soften_hex(color, weight=0.42)
        score_size = float(row["score_value"]) * 780.0
        ax.scatter(x, y, s=score_size * 1.55, facecolors="none", edgecolors=halo_color, linewidths=1.0, zorder=2)
        ax.scatter(x, y, s=score_size * 0.92, color=soften_hex(color, weight=0.16), edgecolors="white", linewidths=0.7, zorder=3)
        ax.scatter(
            x,
            y,
            s=42,
            color=claim_color(row["claim_color_group"]),
            edgecolors="white",
            linewidths=0.6,
            marker=scenario_marker(row["scenario_name"]),
            zorder=4,
        )

    label_rows = (
        ordered.loc[ordered["pathway"].isin(["pyrolysis", "htc", "ad"])]
        .groupby("pathway", as_index=False)
        .agg({"portfolio_share_pct": "max", "stress_support_pct": "max"})
    )
    for _, row in label_rows.iterrows():
        ax.text(
            float(row["portfolio_share_pct"]) + 2.2,
            float(row["stress_support_pct"]) + 1.8,
            {"pyrolysis": "Pyrolysis", "htc": "HTC", "ad": "AD"}.get(str(row["pathway"]), str(row["pathway"])),
            fontsize=8.4,
            color="#0F172A",
            zorder=5,
        )

    ax.set_xlim(-2, 108)
    ax.set_ylim(-2, 84)
    ax.set_xlabel("Portfolio dominance (%)")
    ax.set_ylabel("Robustness support (%)")
    style_axis(ax, grid_axis="both")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=claim_color("planning_ready"), markeredgecolor="none", markersize=6, label="planning-ready"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=claim_color("comparison_only"), markeredgecolor="none", markersize=6, label="comparison-only"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#64748B", markersize=9, label="score halo"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.23),
        frameon=False,
        ncol=1,
        handletextpad=0.5,
    )
    add_caption_note(ax, "Ring size scales with best-case score; marker color preserves evidence tier.")
    fig.subplots_adjust(top=0.90, bottom=0.16, left=0.10, right=0.97)
    return fig
