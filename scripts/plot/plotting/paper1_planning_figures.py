from __future__ import annotations

from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.plot.common import scenario_label

from .theme import (
    add_innovation_glow,
    claim_color,
    configure_publication_theme,
    draw_gradient_barh,
    pathway_color,
    scenario_marker,
    soften_hex,
    style_axis,
    confidence_color,
    add_status_badge
)

SCENARIO_ORDER = ['baseline_region_case', 'high_supply_case', 'policy_support_case']

def build_figure_score_comparison(frame: pd.DataFrame):
    plt = configure_publication_theme()
    fig, ax = plt.subplots(figsize=(8, 6))
    ordered = frame.sort_values(['scenario_order', 'pathway_order'])
    pathways = ['pyrolysis', 'htc']
    for i, scenario in enumerate(SCENARIO_ORDER):
        scen_data = ordered[ordered['scenario_name'] == scenario]
        y_center = (2 - i) * 1.5
        for j, p in enumerate(pathways):
            row = scen_data[scen_data['pathway'] == p]
            if not row.empty:
                val = row['score_value'].iloc[0]
                color = pathway_color(p)
                ax.hlines(y_center + (j-1)*0.25, 0, val, color=color, alpha=0.3, lw=2)
                ax.scatter(val, y_center + (j-1)*0.25, color=color, s=80, edgecolors='white', zorder=4)
        ax.text(-0.02, y_center, scenario_label(scenario), transform=ax.get_yaxis_transform(), ha='right', va='center', fontweight='bold', color='#475569')
    ax.set_yticks([])
    ax.set_xlabel('Planning Score Index (Normalized)', fontweight='bold')
    ax.set_title("Thermochemical score diagnostic (AD retained as reference only)", fontweight="bold")
    style_axis(ax, grid_axis='x')
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=pathway_color(p), markersize=10, label=p.upper()) for p in pathways]
    ax.legend(handles=handles, loc='lower right', frameon=False)
    plt.tight_layout()
    return fig



def build_figure_allocation_stack(pathway_summary: pd.DataFrame):
    plt = configure_publication_theme()
    working = pathway_summary.copy()
    if working.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.text(0.5, 0.5, "No allocation data available", ha="center", va="center")
        ax.axis("off")
        return fig
    working["pathway"] = working["pathway"].astype(str).str.lower()
    share = pd.to_numeric(working.get("portfolio_allocated_feed_share"), errors="coerce").fillna(0.0) * 100.0
    working["share_pct"] = share
    scenarios = [s for s in SCENARIO_ORDER if s in set(working["scenario_name"].astype(str))]
    pathways = [p for p in ["pyrolysis", "htc", "ad", "baseline"] if p in set(working["pathway"])]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    y = np.arange(len(scenarios))
    left = np.zeros(len(scenarios))
    for pathway in pathways:
        vals = []
        for scenario in scenarios:
            rows = working[(working["scenario_name"].astype(str) == scenario) & (working["pathway"] == pathway)]
            vals.append(float(rows["share_pct"].sum()) if not rows.empty else 0.0)
        ax.barh(
            y,
            vals,
            left=left,
            color=pathway_color(pathway),
            height=0.52,
            edgecolor="white",
            linewidth=0.8,
            label=pathway.upper() if pathway != "ad" else "AD",
        )
        for idx, value in enumerate(vals):
            if value >= 6.0:
                ax.text(left[idx] + value / 2.0, idx, f"{value:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8)
        left += np.asarray(vals)
    ax.set_yticks(y)
    ax.set_yticklabels([scenario_label(s) for s in scenarios], fontweight="bold", color="#334155")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Allocated throughput share (%)", fontweight="bold")
    ax.set_title("Optimized pathway allocation under the synchronized baseline", fontweight="bold")
    style_axis(ax, grid_axis="x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=max(1, len(pathways)), frameon=False)
    plt.tight_layout()
    return fig

def build_figure_evidence_composition(confidence_df):
    plt = configure_publication_theme()
    fig, ax = plt.subplots(figsize=(8, 5))
    working = confidence_df.copy()
    if working.empty or "allocated_feed_ton_per_year" not in working.columns:
        ax.text(0.5, 0.5, "No allocated evidence-tier data available", ha="center", va="center")
        ax.axis("off")
        return fig

    def _tier(row: pd.Series) -> str:
        level = str(row.get("surrogate_support_level", "")).lower()
        pathway = str(row.get("pathway", "")).lower()
        if level == "surrogate_supported":
            return "Surrogate-supported"
        if "unsupported" in level or pathway == "ad":
            return "Proxy/reference"
        return "Fallback-backed"

    working["evidence_tier"] = working.apply(_tier, axis=1)
    working["allocated_feed_ton_per_year"] = pd.to_numeric(
        working["allocated_feed_ton_per_year"],
        errors="coerce",
    ).fillna(0.0)
    grouped = (
        working.groupby(["scenario_name", "evidence_tier"], dropna=False)["allocated_feed_ton_per_year"]
        .sum()
        .reset_index()
    )
    totals = grouped.groupby("scenario_name")["allocated_feed_ton_per_year"].transform("sum").replace(0.0, np.nan)
    grouped["share"] = (grouped["allocated_feed_ton_per_year"] / totals).fillna(0.0)
    modes = ["Surrogate-supported", "Fallback-backed", "Proxy/reference"]
    colors = ["#059669", "#D97706", "#94A3B8"]
    scenarios = [s for s in SCENARIO_ORDER if s in set(grouped["scenario_name"].astype(str))]
    y = np.arange(len(scenarios))
    left = np.zeros(len(scenarios))
    for mode, color in zip(modes, colors):
        vals = []
        for scenario in scenarios:
            rows = grouped[(grouped["scenario_name"].astype(str) == scenario) & (grouped["evidence_tier"] == mode)]
            vals.append(float(rows["share"].sum()) if not rows.empty else 0.0)
        ax.barh(y, vals, left=left, label=mode, color=color, height=0.5, edgecolor='white')
        for idx, value in enumerate(vals):
            if value >= 0.08:
                ax.text(left[idx] + value / 2, idx, f"{value * 100:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8)
        left += np.asarray(vals)
    ax.set_yticks(y)
    ax.set_yticklabels([scenario_label(s) for s in scenarios], fontweight='bold', color='#334155')
    ax.set_xlabel('Share of Allocated Throughput by Evidence Tier', fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False)
    style_axis(ax, grid_axis='x')
    plt.tight_layout()
    return fig


def build_figure_boundary_regime_map(ablation_df: pd.DataFrame):
    plt = configure_publication_theme()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    if ablation_df.empty:
        ax.text(0.5, 0.5, "No boundary-regime data available", ha="center", va="center")
        ax.axis("off")
        return fig

    row_specs = [
        ("Locked baseline", "ad_complementarity", "ad_min_share_00pct"),
        ("No product credit", "economic_baseline", "no_product_credit_baseline"),
        ("Symmetric credits", "economic_baseline", "symmetric_product_credit_baseline"),
        ("No pyrolysis credit", "coproduct_boundary", "no_pyrolysis_product_credit"),
        ("Hydrochar 75%", "coproduct_boundary", "hydrochar_credit_75pct"),
        ("Hydrochar 100%", "coproduct_boundary", "hydrochar_credit_100pct"),
    ]
    scenarios = [s for s in SCENARIO_ORDER if s in set(ablation_df["scenario_name"].astype(str))]
    display_rows = []
    for label, family, key in row_specs:
        rows = ablation_df[
            ablation_df["ablation_family"].astype(str).eq(family)
            & ablation_df["ablation_key"].astype(str).eq(key)
        ].copy()
        if rows.empty:
            continue
        display_rows.append((label, rows))

    if not display_rows or not scenarios:
        ax.text(0.5, 0.5, "Boundary-regime rows not found", ha="center", va="center")
        ax.axis("off")
        return fig

    tier_color = {
        "pyrolysis": soften_hex(pathway_color("pyrolysis"), 0.18),
        "htc": soften_hex(pathway_color("htc"), 0.18),
        "ad": soften_hex(pathway_color("ad"), 0.18),
        "mixed": "#E2E8F0",
    }
    for r, (label, rows) in enumerate(display_rows):
        for c, scenario in enumerate(scenarios):
            match = rows[rows["scenario_name"].astype(str).eq(scenario)]
            if match.empty:
                text = "n/a"
                dominant = "mixed"
            else:
                row = match.iloc[0]
                p = float(pd.to_numeric(pd.Series([row.get("pyrolysis_allocated_share_pct")]), errors="coerce").fillna(0).iloc[0])
                h = float(pd.to_numeric(pd.Series([row.get("htc_allocated_share_pct")]), errors="coerce").fillna(0).iloc[0])
                a = float(pd.to_numeric(pd.Series([row.get("ad_allocated_share_pct")]), errors="coerce").fillna(0).iloc[0])
                text = f"P {p:.0f}\\nH {h:.0f}\\nA {a:.0f}"
                vals = {"pyrolysis": p, "htc": h, "ad": a}
                dominant = max(vals, key=vals.get)
                if sorted(vals.values(), reverse=True)[0] < 70:
                    dominant = "mixed"
            rect = plt.Rectangle((c, r), 1, 1, facecolor=tier_color[dominant], edgecolor="white", linewidth=2)
            ax.add_patch(rect)
            ax.text(c + 0.5, r + 0.5, text, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1E293B")

    ax.set_xlim(0, len(scenarios))
    ax.set_ylim(0, len(display_rows))
    ax.set_xticks(np.arange(len(scenarios)) + 0.5)
    ax.set_xticklabels([scenario_label(s) for s in scenarios], fontweight="bold", color="#334155")
    ax.set_yticks(np.arange(len(display_rows)) + 0.5)
    ax.set_yticklabels([label for label, _ in display_rows], fontweight="bold", color="#334155")
    ax.invert_yaxis()
    ax.set_title("Boundary-regime diagnostic map", fontweight="bold", pad=14)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=tier_color["pyrolysis"], markersize=12, label="Pyrolysis-led"),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=tier_color["htc"], markersize=12, label="HTC-led"),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=tier_color["mixed"], markersize=12, label="Mixed"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    plt.tight_layout()
    return fig

def build_figure_confidence_decomposition(confidence_df):
    plt = configure_publication_theme()
    fig, ax = plt.subplots(figsize=(9, 7))
    df = confidence_df.copy().sort_values('recommendation_confidence_score')
    df['display_label'] = df['scenario_name'].map(scenario_label) + ' - ' + df['pathway'].str.upper()
    y = np.arange(len(df))
    c1 = df['support_score_component']
    c2 = df['stress_support_score_component']
    c3 = df['role_score_component']
    ax.barh(y, c1, color='#94A3B8', alpha=0.4, label='Process Support', height=0.6)
    ax.barh(y, c2, left=c1, color='#64748B', alpha=0.7, label='Stress Support', height=0.6)
    ax.barh(y, c3, left=c1+c2, color='#1E293B', label='Portfolio Role', height=0.6)
    for i, (_, row) in enumerate(df.iterrows()):
        tier = str(row['recommendation_confidence_tier']).lower()
        add_status_badge(ax, row['recommendation_confidence_score'] + 0.12, i, row['recommendation_confidence_tier'], confidence_color(tier), transform=ax.get_yaxis_transform())
    ax.set_yticks(y)
    ax.set_yticklabels(df['display_label'], fontsize=8, fontweight='bold', color='#475569')
    ax.set_xlabel('Recommendation Confidence Index', fontweight='bold')
    ax.legend(loc='lower right', frameon=False)
    style_axis(ax, grid_axis='x')
    ax.set_xlim(0, 1.35)
    plt.tight_layout()
    return fig

def build_figure_necessity_matrix(benchmark_df):
    plt = configure_publication_theme()
    fig, ax = plt.subplots(figsize=(9, 6))
    working = benchmark_df.copy()
    if "benchmark_variant" not in working.columns and "benchmark_variant_display" in working.columns:
        working["benchmark_variant"] = working["benchmark_variant_display"].astype(str)
    if "delta_portfolio_carbon_load_kgco2e" not in working.columns:
        working["delta_portfolio_carbon_load_kgco2e"] = 0.0
    pivot = working.pivot(index='benchmark_variant', columns='scenario_name', values='delta_portfolio_carbon_load_kgco2e')
    pivot.columns = [scenario_label(c) for c in pivot.columns]
    pivot.index = [i.replace('_', ' ').title() for i in pivot.index]
    pivot_kt = pivot / 1e6
    cmap = sns.diverging_palette(220, 20, s=90, l=50, as_cmap=True)
    sns.heatmap(pivot_kt, annot=True, fmt='.2f', cmap=cmap, center=0, linewidths=2, linecolor='white', cbar_kws={'label': 'Delta Carbon Load (ktCO2e/y)'}, ax=ax)
    row_idx = [i for i, idx in enumerate(pivot.index) if 'Robustness' in idx]
    if row_idx:
        rect = plt.Rectangle((0, row_idx[0]), len(pivot.columns), 1, fill=False, edgecolor='#EA580C', lw=4, zorder=10)
        ax.add_patch(rect)
    ax.set_title('Methodological Ablation Impact Matrix', fontweight='bold', pad=20)
    ax.set_ylabel('Methodology Variant (Ablation)', fontweight='bold')
    ax.set_xlabel('Scenario Context', fontweight='bold')
    plt.tight_layout()
    return fig

def build_figure_mechanism_frontier(frame: pd.DataFrame):
    plt = configure_publication_theme()
    fig, ax = plt.subplots(figsize=(8, 7))
    ordered = frame[frame['pathway'] != 'baseline'].copy()
    for _, row in ordered.iterrows():
        p = row['pathway']
        color = pathway_color(p)
        share = row['portfolio_share_pct']
        marker = scenario_marker(row['scenario_name'])
        env_val = row['environment_value']
        if env_val > 1000: env_val /= 1000
        ax.scatter(row['energy_value'], env_val, s=share*4+60, color=color, marker=marker, edgecolors='white', linewidths=1.0, alpha=0.9, zorder=4)
        if p == 'pyrolysis' and row['selected_in_baseline_portfolio']:
            add_innovation_glow(ax, row['energy_value'], env_val, color, s=share*4+60)
    ax.set_xlabel('Energy Benefit (PJ/y)', fontweight='bold')
    ax.set_ylabel('Gross Carbon Load (ktCO2e/y)', fontweight='bold')
    style_axis(ax, grid_axis='both')
    handles = [Line2D([0], [0], marker=scenario_marker(s), color='w', markerfacecolor='#64748B', markersize=10, label=scenario_label(s)) for s in SCENARIO_ORDER]
    ax.legend(handles=handles, loc='upper left', title='Scenario Context', frameon=False)
    plt.tight_layout()
    return fig


def build_figure1_main(frame: pd.DataFrame):
    return build_figure_score_comparison(frame)


def build_figure2_tradeoff(frame: pd.DataFrame):
    return build_figure_mechanism_frontier(frame)


def build_figure3_robustness(frame: pd.DataFrame):
    plt = configure_publication_theme()
    fig, ax = plt.subplots(figsize=(9, 6))
    ordered = frame.sort_values(['scenario_order', 'pathway_order']).copy()
    ordered["display_label"] = ordered["scenario_display"].astype(str) + " - " + ordered["pathway_display"].astype(str)
    y = np.arange(len(ordered))
    portfolio_share = pd.to_numeric(ordered.get("portfolio_share_pct"), errors="coerce").fillna(0.0)
    stress_support = pd.to_numeric(ordered.get("stress_support_pct"), errors="coerce").fillna(0.0)
    ax.barh(y, portfolio_share, color="#CBD5E1", height=0.62, label="Portfolio share (%)")
    ax.barh(y, stress_support, color="#334155", alpha=0.85, height=0.34, label="Stress support (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["display_label"], fontsize=8, fontweight='bold', color='#475569')
    ax.set_xlabel('Portfolio Share / Stress Support (%)', fontweight='bold')
    style_axis(ax, grid_axis='x')
    ax.legend(loc='lower right', frameon=False)
    plt.tight_layout()
    return fig


def build_figure2_evidence_ceiling(
    confidence_df: pd.DataFrame,
    evidence_ceiling_df: pd.DataFrame,
    transfer_support_df: pd.DataFrame,
):
    plt = configure_publication_theme()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    working = evidence_ceiling_df.copy()
    if working.empty:
        ax.text(0.5, 0.5, "No evidence-ceiling data available", ha="center", va="center")
        ax.axis("off")
        return fig
    working["scenario"] = working["scenario"].astype(str)
    strong_share = pd.to_numeric(working.get("surrogate_supported_share_pct"), errors="coerce").fillna(0.0)
    weak_share = (100.0 - strong_share).clip(lower=0.0)
    y = np.arange(len(working))
    ax.barh(y, strong_share, color="#059669", height=0.5, label="Strong surrogate (%)")
    ax.barh(y, weak_share, left=strong_share, color="#D97706", height=0.5, label="Fallback-supported (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(working["scenario"], fontweight='bold', color='#334155')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Allocated throughput share by evidence tier (%)', fontweight='bold')
    style_axis(ax, grid_axis='x')
    ax.legend(loc='lower right', frameon=False)
    plt.tight_layout()
    return fig


def build_figure3_benchmark_necessity(benchmark_df: pd.DataFrame):
    return build_figure_necessity_matrix(benchmark_df)


def build_sup_figure_dominance_landscape(frame: pd.DataFrame):
    plt = configure_publication_theme()
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ordered = frame.sort_values(['scenario_order', 'pathway_order']).copy()
    score = pd.to_numeric(ordered.get("score_value"), errors="coerce").fillna(0.0)
    support = pd.to_numeric(ordered.get("stress_support_pct"), errors="coerce").fillna(0.0)
    bubble = pd.to_numeric(ordered.get("portfolio_share_pct"), errors="coerce").fillna(0.0)
    ordered["score_value"] = score
    ordered["stress_support_pct"] = support
    ordered["portfolio_share_pct"] = bubble
    for _, row in ordered.iterrows():
        ax.scatter(
            float(row["score_value"]),
            float(row["stress_support_pct"]),
            s=float(row["portfolio_share_pct"]) * 8 + 60,
            color=pathway_color(str(row.get("pathway", ""))),
            alpha=0.85,
            edgecolors='white',
            linewidths=1.0,
            marker=scenario_marker(str(row.get("scenario_name", ""))),
        )
    ax.set_xlabel('Case-level score index', fontweight='bold')
    ax.set_ylabel('Stress support (%)', fontweight='bold')
    style_axis(ax, grid_axis='both')
    ax.set_xlim(left=max(0.0, float(score.min()) - 0.05), right=float(score.max()) + 0.1 if len(score) else 1.0)
    ax.set_ylim(bottom=0.0, top=float(support.max()) + 10.0 if len(support) else 100.0)
    plt.tight_layout()
    return fig
