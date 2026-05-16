# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json

import pandas as pd

from waste2energy.benchmarking import run_planning_benchmark_suite
from waste2energy.planning.solve import PlanningConfig


def test_benchmark_suite_smoke(tmp_path):
    output_dir = tmp_path / "benchmark"
    result = run_planning_benchmark_suite(
        output_dir=str(output_dir),
        base_config=PlanningConfig(
            optimization_method="scipy",
            pareto_point_count=0,
            enable_pareto_export=False,
        ),
        bootstrap_replicates=2,
        bootstrap_random_seed=7,
    )

    assert result["variant_count"] >= 6
    assert "baseline_evidence_aware" in result["benchmark_variant_keys"]
    assert "no_evidence_penalty" in result["benchmark_variant_keys"]
    assert "classic_multiobjective_optimizer" in result["benchmark_variant_keys"]
    assert "greedy_weighted_score_heuristic" in result["benchmark_variant_keys"]
    assert "mcda_weighted_sum_comparator" in result["benchmark_variant_keys"]
    assert "topsis_comparator" in result["benchmark_variant_keys"]
    assert "uncertainty_penalty_max_interval" in result["benchmark_variant_keys"]
    assert "uncertainty_penalty_combined_only" in result["benchmark_variant_keys"]

    summary = pd.read_csv(output_dir / "benchmark_summary.csv")
    shifts = pd.read_csv(output_dir / "benchmark_shift_summary.csv")
    allocations = pd.read_csv(output_dir / "benchmark_allocations.csv")
    diagnostics = pd.read_csv(output_dir / "benchmark_diagnostics.csv")
    bootstrap_samples = pd.read_csv(output_dir / "benchmark_bootstrap_shift_samples.csv")
    statistical_summary = pd.read_csv(output_dir / "benchmark_statistical_summary.csv")
    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))

    assert not summary.empty
    assert not shifts.empty
    assert not allocations.empty
    assert not diagnostics.empty
    assert not bootstrap_samples.empty
    assert not statistical_summary.empty
    assert summary["benchmark_variant"].isin(result["benchmark_variant_keys"]).all()
    assert "comparator_family" in summary.columns
    assert "allocation_mode" in summary.columns
    assert "portfolio_pathway_shift" in shifts.columns
    assert shifts["benchmark_variant"].ne("baseline_evidence_aware").all()
    assert "effect_significance_tier" in statistical_summary.columns
    assert summary[summary["benchmark_variant"].eq("mcda_weighted_sum_comparator")]["allocation_mode"].eq(
        "mcda_weighted_sum"
    ).all()
    assert summary[summary["benchmark_variant"].eq("topsis_comparator")]["allocation_mode"].eq("topsis").all()
    assert "pathway_shift_rate_ci_lower" in statistical_summary.columns
    assert "case_shift_rate_ci_upper" in statistical_summary.columns
    assert "delta_portfolio_score_mass_empirical_p_value" in statistical_summary.columns
    assert "delta_portfolio_carbon_load_kgco2e_direction" in statistical_summary.columns
    assert bootstrap_samples["bootstrap_replicate"].nunique() == 2
    assert run_config["benchmark_variant_count"] == result["variant_count"]
    assert run_config["bootstrap_replicate_count"] == 2
