# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from pathlib import Path

import pandas as pd

from waste2energy.planning import ablations
from waste2energy.planning.inputs import PlanningInputBundle
from waste2energy.planning.solve import PlanningConfig


def test_targeted_ablations_export_reviewer_requested_outputs(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": pathway,
                "product_revenue_usd_per_year": revenue,
                "scenario_total_mixed_feed_ton_per_year_proxy": 1000.0,
                "net_system_cost_usd_per_year": 10000.0 - revenue,
                "unit_net_system_cost_usd_per_total_mixed_ton": (10000.0 - revenue) / 1000.0,
                "total_system_cost_usd_per_year": 10000.0,
            }
            for pathway, revenue in (("pyrolysis", 2500.0), ("htc", 0.0), ("ad", 0.0))
        ]
    )
    bundle = PlanningInputBundle(
        frame=frame,
        dataset_path=Path("dummy.csv"),
        scenario_names=("baseline_region_case",),
        pathways=("ad", "htc", "pyrolysis"),
        real_cost_columns=("net_system_cost_usd_per_year",),
        surrogate_feature_columns=(),
        unit_registry={},
    )

    def fake_execute_planning_pipeline(*, bundle, config):
        selected = "htc" if getattr(config, "minimum_surrogate_artifact_test_r2", None) == 0.0 else "pyrolysis"
        portfolio_summary = pd.DataFrame(
            [
                {
                    "scenario_name": "baseline_region_case",
                    "top_portfolio_case_id": f"case-{selected}",
                    "selected_candidate_count": 1,
                    "portfolio_fill_ratio": 1.0,
                    "scenario_feed_coverage_ratio": 1.0,
                    "portfolio_score_mass": 10.0,
                    "portfolio_energy_objective": 1.0e9,
                    "portfolio_environment_objective": 2.0,
                    "portfolio_cost_objective": 3.0e6,
                    "portfolio_carbon_load_kgco2e": 4.0e6,
                    "allocated_feed_ton_per_year": 1000.0,
                }
            ]
        )
        portfolio_allocations = pd.DataFrame(
            [
                {
                    "scenario_name": "baseline_region_case",
                    "optimization_case_id": f"case-{selected}",
                    "pathway": selected,
                    "allocated_feed_ton_per_year": 1000.0,
                }
            ]
        )
        return {
            "portfolio_summary": portfolio_summary,
            "portfolio_allocations": portfolio_allocations,
        }

    monkeypatch.setattr(ablations, "load_planning_input_bundle", lambda dataset_path=None: bundle)
    monkeypatch.setattr(ablations, "execute_planning_pipeline", fake_execute_planning_pipeline)

    result = ablations.run_targeted_planning_ablations(
        output_dir=str(tmp_path),
        base_config=PlanningConfig(optimization_method="scipy", pareto_point_count=0, enable_pareto_export=False),
        monte_carlo_replicates=2,
        monte_carlo_random_seed=11,
    )

    summary = pd.read_csv(result["summary_csv"])
    cap_diagnostics = pd.read_csv(result["cap_diagnostics_csv"])
    monte_carlo_summary = pd.read_csv(result["monte_carlo_summary_csv"])
    assert {
        "economic_baseline",
        "surrogate_evidence_gate",
        "evidence_ladder_sensitivity",
        "constraint_mechanism",
        "objective_weight_sweep",
    }.issubset(set(summary["ablation_family"]))
    assert {"no_product_credit_baseline", "symmetric_product_credit_baseline"}.issubset(
        set(summary["ablation_key"])
    )
    assert {"equal_weight", "cost_high"}.issubset(set(summary["ablation_key"]))
    assert not monte_carlo_summary.empty
    assert not cap_diagnostics.empty
    assert "candidate_cap_artifact_flag" in cap_diagnostics.columns
    assert {"locked_candidate_cap", "candidate_cap_relaxed_100pct"}.issubset(set(cap_diagnostics["ablation_key"]))
    assert "selection_probability" in monte_carlo_summary.columns
    assert Path(result["monte_carlo_samples_csv"]).exists()
