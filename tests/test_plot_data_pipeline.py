from __future__ import annotations

import pandas as pd

from scripts.plot.plotting.data_pipeline import (
    build_figure_ready_tables,
    write_figure_ready_tables,
)


def test_build_figure_ready_tables_returns_main_and_supporting_tables():
    metrics = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "scenario_order": 1,
                "pathway": "pyrolysis",
                "pathway_order": 2,
                "pathway_rank_within_scenario": 1,
                "writing_label": "supporting portfolio",
                "selected_in_baseline_portfolio": True,
                "selected_flag": "selected",
                "claim_boundary": "planning-ready candidate",
                "claim_color_group": "planning_ready",
                "metric_key": "best_case_score_index",
                "metric_label": "Best-case score index",
                "metric_family": "score",
                "unit": "index",
                "value": 0.91,
                "value_available": True,
                "value_label": "0.91",
            },
            {
                "scenario_name": "baseline_region_case",
                "scenario_order": 1,
                "pathway": "pyrolysis",
                "pathway_order": 2,
                "pathway_rank_within_scenario": 1,
                "writing_label": "supporting portfolio",
                "selected_in_baseline_portfolio": True,
                "selected_flag": "selected",
                "claim_boundary": "planning-ready candidate",
                "claim_color_group": "planning_ready",
                "metric_key": "best_case_energy_pj_per_year",
                "metric_label": "Best-case energy",
                "metric_family": "energy",
                "unit": "PJ/year",
                "value": 9.65,
                "value_available": True,
                "value_label": "9.65",
            },
            {
                "scenario_name": "baseline_region_case",
                "scenario_order": 1,
                "pathway": "pyrolysis",
                "pathway_order": 2,
                "pathway_rank_within_scenario": 1,
                "writing_label": "supporting portfolio",
                "selected_in_baseline_portfolio": True,
                "selected_flag": "selected",
                "claim_boundary": "planning-ready candidate",
                "claim_color_group": "planning_ready",
                "metric_key": "best_case_environment_ktco2e_per_year",
                "metric_label": "Best-case environment",
                "metric_family": "environment",
                "unit": "ktCO2e/year",
                "value": 214.8,
                "value_available": True,
                "value_label": "214.8",
            },
            {
                "scenario_name": "baseline_region_case",
                "scenario_order": 1,
                "pathway": "pyrolysis",
                "pathway_order": 2,
                "pathway_rank_within_scenario": 1,
                "writing_label": "supporting portfolio",
                "selected_in_baseline_portfolio": True,
                "selected_flag": "selected",
                "claim_boundary": "planning-ready candidate",
                "claim_color_group": "planning_ready",
                "metric_key": "baseline_portfolio_share_pct",
                "metric_label": "Baseline portfolio share",
                "metric_family": "portfolio_share",
                "unit": "pct",
                "value": 87.6,
                "value_available": True,
                "value_label": "87.6%",
            },
            {
                "scenario_name": "baseline_region_case",
                "scenario_order": 1,
                "pathway": "pyrolysis",
                "pathway_order": 2,
                "pathway_rank_within_scenario": 1,
                "writing_label": "supporting portfolio",
                "selected_in_baseline_portfolio": True,
                "selected_flag": "selected",
                "claim_boundary": "planning-ready candidate",
                "claim_color_group": "planning_ready",
                "metric_key": "max_stress_selection_rate",
                "metric_label": "Max stress selection rate",
                "metric_family": "stress_support",
                "unit": "pct",
                "value": 50.0,
                "value_available": True,
                "value_label": "50.0%",
            },
            {
                "scenario_name": "baseline_region_case",
                "scenario_order": 1,
                "pathway": "pyrolysis",
                "pathway_order": 2,
                "pathway_rank_within_scenario": 1,
                "writing_label": "supporting portfolio",
                "selected_in_baseline_portfolio": True,
                "selected_flag": "selected",
                "claim_boundary": "planning-ready candidate",
                "claim_color_group": "planning_ready",
                "metric_key": "score_gap_to_scenario_best_pct",
                "metric_label": "Score gap to scenario best",
                "metric_family": "score_gap",
                "unit": "pct",
                "value": 25.5,
                "value_available": True,
                "value_label": "25.5%",
            },
        ]
    )

    tables = build_figure_ready_tables(metrics)

    assert set(tables) == {
        "figure1_main",
        "figure2_tradeoff",
        "figure3_robustness",
        "paper1_sup_s1_scenario_fingerprint",
        "paper1_sup_s2_dominance_evidence_landscape",
    }
    assert not tables["figure1_main"].empty
    assert not tables["figure2_tradeoff"].empty
    assert not tables["figure3_robustness"].empty
    assert "normalized_value" in tables["paper1_sup_s1_scenario_fingerprint"].columns
    assert "selected_core_zone" in tables["paper1_sup_s2_dominance_evidence_landscape"].columns


def test_write_figure_ready_tables_writes_csvs(tmp_path):
    tables = {
        "figure1_main": pd.DataFrame([{"a": 1}]),
        "figure2_tradeoff": pd.DataFrame([{"b": 2}]),
        "figure3_robustness": pd.DataFrame([{"c": 3}]),
        "paper1_sup_s1_scenario_fingerprint": pd.DataFrame([{"d": 4}]),
        "paper1_sup_s2_dominance_evidence_landscape": pd.DataFrame([{"e": 5}]),
    }

    outputs = write_figure_ready_tables(tables, output_dir=tmp_path)

    assert outputs["figure1_main"].endswith("figure1_main.csv")
    assert (tmp_path / "figure1_main.csv").exists()
    assert (tmp_path / "figure2_tradeoff.csv").exists()
    assert (tmp_path / "figure3_robustness.csv").exists()
    assert (tmp_path / "paper1_sup_s1_scenario_fingerprint.csv").exists()
    assert (tmp_path / "paper1_sup_s2_dominance_evidence_landscape.csv").exists()
