from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.plot.common import ensure_results_dir
from scripts.plot.plotting.exports import build_plot_manifest, save_plot_figure_set
from scripts.plot.plotting.paper1_planning_figures import (
    build_figure1_main,
    build_figure2_evidence_ceiling,
    build_figure2_tradeoff,
    build_figure3_robustness,
    build_figure3_benchmark_necessity,
    build_sup_figure_dominance_landscape,
)
from scripts.plot.plotting.theme import (
    ULTRA_PREMIUM_FONT_FAMILY,
    ULTRA_PREMIUM_FONT_SIZE,
    configure_publication_theme,
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
                "recommendation_confidence_score": 0.678,
                "recommendation_confidence_tier": "moderate",
                "scenario_transferability_ceiling": "guarded transfer",
                "full_support_share_pct": 12.4,
            },
            {
                "scenario_name": "high_supply_case",
                "scenario_order": 2,
                "scenario_display": "High supply",
                "pathway": "pyrolysis",
                "pathway_order": 1,
                "pathway_display": "Pyrolysis",
                "portfolio_share_pct": 87.9,
                "stress_support_pct": 48.0,
                "claim_color_group": "planning_ready",
                "selected_in_baseline_portfolio": True,
                "recommendation_confidence_score": 0.679,
                "recommendation_confidence_tier": "moderate",
                "scenario_transferability_ceiling": "guarded transfer",
                "full_support_share_pct": 12.1,
            },
            {
                "scenario_name": "policy_support_case",
                "scenario_order": 3,
                "scenario_display": "Policy support",
                "pathway": "pyrolysis",
                "pathway_order": 1,
                "pathway_display": "Pyrolysis",
                "portfolio_share_pct": 100.0,
                "stress_support_pct": 65.0,
                "claim_color_group": "planning_ready",
                "selected_in_baseline_portfolio": True,
                "recommendation_confidence_score": 0.797,
                "recommendation_confidence_tier": "high",
                "scenario_transferability_ceiling": "conditional, supported",
                "full_support_share_pct": 0.0,
            }
        ]
    )
    confidence_data = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "recommendation_confidence_score": 0.678,
                "recommendation_confidence_tier": "moderate",
                "support_score_component": 0.28,
                "stress_support_score_component": 0.20,
                "role_score_component": 0.20,
            },
            {
                "scenario_name": "baseline_region_case",
                "pathway": "htc",
                "recommendation_confidence_score": 0.694,
                "recommendation_confidence_tier": "moderate",
                "support_score_component": 0.25,
                "stress_support_score_component": 0.19,
                "role_score_component": 0.25,
            },
        ]
    )
    evidence_ceiling_data = pd.DataFrame(
        [
            {
                "scenario": "Baseline region",
                "surrogate_supported_share_pct": 12.4,
                "transferability_ceiling": "guarded transfer",
                "selected_pathways": "Pyrolysis + HTC",
            }
        ]
    )
    transfer_support_data = pd.DataFrame(
        [
            {
                "pathway": "pyrolysis",
                "reliability_summary": "0.625 (conditional)",
                "leave_study_out_support_split": "1 supportive / 3 weak / 0 unsupported",
                "manuscript_ceiling": "conditional_support",
            },
            {
                "pathway": "htc",
                "reliability_summary": "0.250 (auxiliary)",
                "leave_study_out_support_split": "0 supportive / 4 weak / 4 unsupported",
                "manuscript_ceiling": "auxiliary_only",
            },
        ]
    )
    benchmark_data = pd.DataFrame(
        [
            {
                "benchmark_variant_display": "No robustness",
                "scenario_name": "baseline_region_case",
                "necessity_rank": 2,
                "necessity_tier": "supports_core_innovation",
                "pathway_shift_rate_pct": 100.0,
                "significance_abbrev": "HC",
            }
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
    fig4 = build_figure2_evidence_ceiling(confidence_data, evidence_ceiling_data, transfer_support_data)
    fig5 = build_figure3_benchmark_necessity(benchmark_data)
    fig6 = build_sup_figure_dominance_landscape(sup_s2_data)

    assert fig1 is not None
    assert fig2 is not None
    assert fig3 is not None
    assert fig4 is not None
    assert fig5 is not None
    assert fig6 is not None
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)
    plt.close(fig6)


def test_benchmark_necessity_figure_uses_kt_carbon_units():
    benchmark_data = pd.DataFrame(
        [
            {
                "benchmark_variant_display": "No robustness",
                "scenario_name": "baseline_region_case",
                "delta_portfolio_carbon_load_kgco2e": -18269435.16325231,
            }
        ]
    )

    fig = build_figure3_benchmark_necessity(benchmark_data)
    colorbar_axis = fig.axes[-1]

    assert colorbar_axis.get_ylabel() == "Delta Carbon Load (ktCO2e/y)"
    plt.close(fig)


def test_publication_theme_uses_ultra_premium_typography_defaults():
    themed = configure_publication_theme()

    assert themed.rcParams["font.size"] == ULTRA_PREMIUM_FONT_SIZE
    assert themed.rcParams["font.family"][: len(ULTRA_PREMIUM_FONT_FAMILY)] == ULTRA_PREMIUM_FONT_FAMILY
    assert themed.rcParams["savefig.dpi"] == 300
