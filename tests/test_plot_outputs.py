from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.plot.common import ensure_results_dir
from scripts.plot.plotting.exports import build_plot_manifest, save_plot_figure_set
from scripts.plot.plotting.paper1_planning_figures import (
    build_figure1_main,
    build_figure2_tradeoff,
    build_figure3_robustness,
    build_sup_figure_s1_scenario_fingerprint,
    build_sup_figure_s2_dominance_evidence_landscape,
)


def test_ensure_results_dir_uses_results_plot(tmp_path):
    target = ensure_results_dir(tmp_path)

    assert target == tmp_path
    assert (tmp_path / "pdf").exists()
    assert (tmp_path / "png").exists()
    assert (tmp_path / "tiff").exists()
    assert (tmp_path / "eps").exists()


def test_save_plot_figure_set_writes_multi_format_outputs(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    outputs = save_plot_figure_set(fig, "demo_plot", output_dir=tmp_path)
    plt.close(fig)

    assert outputs["pdf"].endswith("demo_plot.pdf")
    assert (tmp_path / "pdf" / "demo_plot.pdf").exists()
    assert (tmp_path / "png" / "demo_plot.png").exists()


def test_build_plot_manifest_records_pdf_targets():
    manifest = build_plot_manifest(
        outputs={
            "figure1": {"pdf": "results/plot/pdf/figure1.pdf"},
            "paper1_sup_fig_s1_scenario_fingerprint": {"pdf": "results/plot/pdf/paper1_sup_fig_s1_scenario_fingerprint.pdf"},
        },
        data_outputs={
            "figure1_main": "results/plot/data/figure1_main.csv",
            "paper1_sup_s1_scenario_fingerprint": "results/plot/data/paper1_sup_s1_scenario_fingerprint.csv",
        },
        output_dir=Path("results/plot"),
    )

    assert manifest["output_dir"].endswith("results\\plot") or manifest["output_dir"].endswith("results/plot")
    assert "latex_pdf_targets" in manifest
    assert "paper1_sup_fig_s1_scenario_fingerprint" in manifest["latex_pdf_targets"]


def test_figure_builders_return_figures():
    figure1_data = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "scenario_order": 1,
                "pathway": "pyrolysis",
                "pathway_order": 1,
                "pathway_display": "Pyrolysis",
                "score_value": 0.91,
                "energy_value": 9.65,
                "environment_value": 214.8,
                "portfolio_share_pct": 87.6,
                "stress_support_pct": 50.0,
                "selected_in_baseline_portfolio": True,
                "claim_color_group": "planning_ready",
                "claim_boundary": "planning-ready candidate",
            }
        ]
    )
    figure2_data = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "scenario_order": 1,
                "pathway": "pyrolysis",
                "pathway_order": 1,
                "pathway_display": "Pyrolysis",
                "energy_value": 9.65,
                "environment_value": 214.8,
                "portfolio_share_pct": 87.6,
                "selected_in_baseline_portfolio": True,
            }
        ]
    )
    figure3_data = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "scenario_order": 1,
                "scenario_display": "Baseline region",
                "pathway": "pyrolysis",
                "pathway_order": 1,
                "pathway_display": "Pyrolysis",
                "portfolio_share_pct": 87.6,
                "stress_support_pct": 50.0,
                "claim_color_group": "planning_ready",
                "selected_in_baseline_portfolio": True,
            }
        ]
    )
    sup_s1_data = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "scenario_order": 1,
                "scenario_display": "Baseline region",
                "pathway": "pyrolysis",
                "pathway_order": 1,
                "pathway_display": "Pyrolysis",
                "selected_in_baseline_portfolio": True,
                "claim_color_group": "planning_ready",
                "metric_key": "best_case_score_index",
                "metric_display": "Score",
                "metric_order": 0,
                "metric_value": 0.91,
                "normalized_value": 0.72,
            },
            {
                "scenario_name": "baseline_region_case",
                "scenario_order": 1,
                "scenario_display": "Baseline region",
                "pathway": "pyrolysis",
                "pathway_order": 1,
                "pathway_display": "Pyrolysis",
                "selected_in_baseline_portfolio": True,
                "claim_color_group": "planning_ready",
                "metric_key": "baseline_portfolio_share_pct",
                "metric_display": "Share",
                "metric_order": 1,
                "metric_value": 87.6,
                "normalized_value": 0.92,
            },
        ]
    )
    sup_s2_data = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "scenario_order": 1,
                "scenario_display": "Baseline region",
                "pathway": "pyrolysis",
                "pathway_order": 1,
                "pathway_display": "Pyrolysis",
                "portfolio_share_pct": 87.6,
                "stress_support_pct": 50.0,
                "score_value": 0.91,
                "score_gap_pct": 25.5,
                "selected_in_baseline_portfolio": True,
                "selected_flag": "selected",
                "claim_boundary": "planning-ready candidate",
                "claim_color_group": "planning_ready",
            }
        ]
    )

    fig1 = build_figure1_main(figure1_data)
    fig2 = build_figure2_tradeoff(figure2_data)
    fig3 = build_figure3_robustness(figure3_data)
    fig4 = build_sup_figure_s1_scenario_fingerprint(sup_s1_data)
    fig5 = build_sup_figure_s2_dominance_evidence_landscape(sup_s2_data)

    assert fig1 is not None
    assert fig2 is not None
    assert fig3 is not None
    assert fig4 is not None
    assert fig5 is not None
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)
