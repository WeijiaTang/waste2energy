from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot.common import save_figure_set, RESULTS_PLOT_DIR

SUMMARY = ROOT / "outputs" / "benchmark" / "targeted_planning_ablations" / "targeted_planning_ablations_summary.csv"
FIGURES_DIR = ROOT / "data" / "processed" / "figures_tables"

SCENARIO_ORDER = ["baseline-region", "high-supply", "policy-support"]


def _scenario_display(value: object) -> str:
    return {
        "baseline_region_case": "baseline-region",
        "high_supply_case": "high-supply",
        "policy_support_case": "policy-support",
    }.get(str(value), str(value))


def build_ad_complementarity_figure(summary: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    data = summary[summary["ablation_family"].eq("ad_complementarity")].copy()
    base = data[data["ablation_value"].astype(float).eq(0.0)][
        ["scenario_name", "portfolio_energy_objective", "portfolio_cost_objective", "portfolio_carbon_load_kgco2e"]
    ].rename(columns={
        "portfolio_energy_objective": "base_energy",
        "portfolio_cost_objective": "base_cost",
        "portfolio_carbon_load_kgco2e": "base_carbon",
    })
    data = data.merge(base, on="scenario_name", how="left")
    data["scenario_label"] = data["scenario_name"].map(_scenario_display)
    data["ad_floor_pct"] = data["ablation_value"].astype(float) * 100.0
    data["energy_delta_pj"] = (data["portfolio_energy_objective"] - data["base_energy"]) / 1e9
    data["cost_delta_musd"] = (data["portfolio_cost_objective"] - data["base_cost"]) / 1e6
    data["carbon_delta_kt"] = (data["portfolio_carbon_load_kgco2e"] - data["base_carbon"]) / 1e6

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharex=True)
    metrics = [
        ("energy_delta_pj", "Energy change\n(PJ y$^{-1}$)", "#3B6EA8"),
        ("cost_delta_musd", "Net cost change\n(MUSD y$^{-1}$)", "#8F5A9B"),
        ("carbon_delta_kt", "Carbon-load change\n(kt CO$_2$e y$^{-1}$)", "#2F7D32"),
    ]
    for ax, (column, ylabel, color) in zip(axes, metrics, strict=True):
        for scenario in SCENARIO_ORDER:
            subset = data[data["scenario_label"].eq(scenario)].sort_values("ad_floor_pct")
            ax.plot(subset["ad_floor_pct"], subset[column], marker="o", linewidth=1.4, label=scenario)
        ax.axhline(0, color="#444444", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Minimum AD share (%)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_xticks([0, 10, 20])
        if column == "energy_delta_pj":
            ax.set_title("Energy penalty", color=color)
        elif column == "cost_delta_musd":
            ax.set_title("Cost benefit", color=color)
        else:
            ax.set_title("Carbon benefit", color=color)
    axes[0].legend(loc="lower left", fontsize=6.5, frameon=False)
    fig.suptitle("AD participation floors expose a biological-treatment complementarity tradeoff", y=1.03, fontsize=9.5)
    fig.tight_layout(w_pad=1.1)
    return fig, data


def main() -> int:
    summary = pd.read_csv(SUMMARY)
    fig, data = build_ad_complementarity_figure(summary)
    outputs = save_figure_set(fig, "paper1_fig7_ad_complementarity_tradeoff", output_dir=RESULTS_PLOT_DIR)
    data.to_csv(FIGURES_DIR / "paper1_ad_complementarity_figure_data.csv", index=False)
    print(json.dumps({"figure": outputs, "data_rows": int(len(data))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
