# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json

import pandas as pd

from waste2energy.results_quality import (
    CORE_BENCHMARK_COMPARATORS,
    DIAGNOSTIC_BENCHMARK_COMPARATORS,
    MAJOR_BENCHMARK_COMPARATORS,
    build_results_quality_report,
    build_validated_pyrolysis_superiority_claim,
    write_results_quality_report,
)


def test_results_quality_report_flags_planning_boundary_and_hashes(tmp_path):
    planning_dir = tmp_path / "planning"
    planning_dir.mkdir()
    (planning_dir / "run_config.json").write_text("{}", encoding="utf-8")
    pd.DataFrame([{"allocated_feed_ton_per_year": 1.0}]).to_csv(
        planning_dir / "portfolio_allocations.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "summary_scope": "all_planning_rows",
                "row_count": 10,
                "independent_observation_count": 0,
                "scenario_expanded_count": 10,
            }
        ]
    ).to_csv(planning_dir / "planning_data_contract_summary.csv", index=False)
    pd.DataFrame([{"warning": "scenario_expanded_rows_must_not_be_counted_as_independent_evidence"}]).to_csv(
        planning_dir / "planning_data_contract_warnings.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "transferability_risk_label": "selected_share_conditionally_transferable",
            }
        ]
    ).to_csv(planning_dir / "surrogate_transferability_summary.csv", index=False)
    (planning_dir / "reproducibility_manifest.json").write_text(
        json.dumps(
            {
                "inputs": [{"exists": True, "sha256": "abc"}],
                "outputs": [{"exists": True, "sha256": "def"}],
            }
        ),
        encoding="utf-8",
    )

    frame, summary = build_results_quality_report(planning_dir=planning_dir)

    assert summary["overall_status"] == "fail"
    assert summary["strong_claim_status"] == "fail"
    assert "planning_data_contract_boundary" in set(frame["gate"])
    assert "validated_pyrolysis_superiority_claim" in set(frame["gate"])
    assert frame[frame["gate"].eq("reproducibility_hashes_present")].iloc[0]["status"] == "pass"


def test_write_results_quality_report_outputs_csv_and_json(tmp_path):
    planning_dir = tmp_path / "planning"
    planning_dir.mkdir()

    outputs = write_results_quality_report(output_dir=tmp_path / "quality", planning_dir=planning_dir)

    assert (tmp_path / "quality" / "quality_gate_summary.csv").exists()
    assert (tmp_path / "quality" / "quality_gate_summary.json").exists()
    assert set(outputs) == {
        "quality_gate_summary",
        "quality_gate_summary_json",
        "validated_pyrolysis_superiority_claim",
        "validated_pyrolysis_superiority_claim_json",
    }


def test_validated_pyrolysis_superiority_claim_passes_when_all_gates_are_met(tmp_path):
    planning_dir = tmp_path / "planning"
    benchmark_dir = tmp_path / "benchmark"
    scenario_dir = tmp_path / "scenarios"
    planning_dir.mkdir()
    benchmark_dir.mkdir()
    scenario_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "pathway": "pyrolysis",
                "selected_allocated_feed_share": 1.0,
                "worst_surrogate_evidence_gate": "conditional_transfer",
                "optimization_supported_prediction_fraction": 1.0,
            }
        ]
    ).to_csv(planning_dir / "surrogate_transferability_summary.csv", index=False)
    pd.DataFrame(
        [{"scenario_name": "s1", "benchmark_variant": "baseline_evidence_aware", "selected_pathways": "pyrolysis"}]
        + [
            {"scenario_name": "s1", "benchmark_variant": variant, "selected_pathways": "pyrolysis"}
            for variant in MAJOR_BENCHMARK_COMPARATORS
        ]
    ).to_csv(benchmark_dir / "benchmark_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "benchmark_variant": variant,
                "bootstrap_replicate_count": 16,
                "effect_significance_tier": "directionally_consistent",
            }
            for variant in MAJOR_BENCHMARK_COMPARATORS
        ]
    ).to_csv(benchmark_dir / "benchmark_statistical_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "baseline_top_pathway": "pyrolysis",
                "combined_only_top_pathway": "pyrolysis",
                "max_interval_top_pathway": "pyrolysis",
                "dominant_selection_rate": 0.9,
            }
        ]
    ).to_csv(scenario_dir / "uncertainty_summary.csv", index=False)

    _, summary = build_validated_pyrolysis_superiority_claim(
        planning_dir=planning_dir,
        benchmark_dir=benchmark_dir,
        scenario_dir=scenario_dir,
    )

    assert summary["claim_status"] == "pass"
    assert "Stress-tested evidence supports pyrolysis as the leading pathway" in summary["claim_text"]


def test_validated_claim_fails_without_benchmark_and_scenario_evidence(tmp_path):
    planning_dir = tmp_path / "planning"
    planning_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "pathway": "pyrolysis",
                "selected_allocated_feed_share": 1.0,
                "worst_surrogate_evidence_gate": "conditional_transfer",
                "optimization_supported_prediction_fraction": 1.0,
            }
        ]
    ).to_csv(planning_dir / "surrogate_transferability_summary.csv", index=False)

    frame, summary = build_validated_pyrolysis_superiority_claim(planning_dir=planning_dir)

    assert summary["claim_status"] == "fail"
    assert {
        "benchmark_evidence_available",
        "scenario_stress_evidence_available",
    }.issubset(set(frame["claim_gate"]))


def test_validated_claim_fails_when_major_comparator_or_bootstrap_is_missing(tmp_path):
    planning_dir = tmp_path / "planning"
    benchmark_dir = tmp_path / "benchmark"
    scenario_dir = tmp_path / "scenarios"
    planning_dir.mkdir()
    benchmark_dir.mkdir()
    scenario_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "pathway": "pyrolysis",
                "selected_allocated_feed_share": 1.0,
                "worst_surrogate_evidence_gate": "conditional_transfer",
                "optimization_supported_prediction_fraction": 1.0,
            }
        ]
    ).to_csv(planning_dir / "surrogate_transferability_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": "s1", "benchmark_variant": "baseline_evidence_aware", "selected_pathways": "pyrolysis"},
            {"scenario_name": "s1", "benchmark_variant": "classic_multiobjective_optimizer", "selected_pathways": "pyrolysis"},
        ]
    ).to_csv(benchmark_dir / "benchmark_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "benchmark_variant": "classic_multiobjective_optimizer",
                "bootstrap_replicate_count": 2,
                "effect_significance_tier": "suggestive",
            }
        ]
    ).to_csv(benchmark_dir / "benchmark_statistical_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "baseline_top_pathway": "pyrolysis",
                "combined_only_top_pathway": "pyrolysis",
                "max_interval_top_pathway": "pyrolysis",
                "dominant_selection_rate": 0.9,
            }
        ]
    ).to_csv(scenario_dir / "uncertainty_summary.csv", index=False)

    frame, summary = build_validated_pyrolysis_superiority_claim(
        planning_dir=planning_dir,
        benchmark_dir=benchmark_dir,
        scenario_dir=scenario_dir,
    )

    assert summary["claim_status"] == "fail"
    failed = set(frame[frame["status"].eq("fail")]["claim_gate"])
    assert "benchmark_core_comparators_select_pyrolysis" in failed
    assert "benchmark_bootstrap_replicates_sufficient" in failed


def test_validated_claim_fails_when_uncertainty_modes_are_incomplete(tmp_path):
    planning_dir = tmp_path / "planning"
    benchmark_dir = tmp_path / "benchmark"
    scenario_dir = tmp_path / "scenarios"
    planning_dir.mkdir()
    benchmark_dir.mkdir()
    scenario_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "pathway": "pyrolysis",
                "selected_allocated_feed_share": 1.0,
                "worst_surrogate_evidence_gate": "conditional_transfer",
                "optimization_supported_prediction_fraction": 1.0,
            }
        ]
    ).to_csv(planning_dir / "surrogate_transferability_summary.csv", index=False)
    pd.DataFrame(
        [{"scenario_name": "s1", "benchmark_variant": "baseline_evidence_aware", "selected_pathways": "pyrolysis"}]
        + [
            {"scenario_name": "s1", "benchmark_variant": variant, "selected_pathways": "pyrolysis"}
            for variant in MAJOR_BENCHMARK_COMPARATORS
        ]
    ).to_csv(benchmark_dir / "benchmark_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "benchmark_variant": variant,
                "bootstrap_replicate_count": 16,
                "effect_significance_tier": "directionally_consistent",
            }
            for variant in MAJOR_BENCHMARK_COMPARATORS
        ]
    ).to_csv(benchmark_dir / "benchmark_statistical_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "baseline_top_pathway": "pyrolysis",
                "dominant_selection_rate": 0.9,
            }
        ]
    ).to_csv(scenario_dir / "uncertainty_summary.csv", index=False)

    frame, summary = build_validated_pyrolysis_superiority_claim(
        planning_dir=planning_dir,
        benchmark_dir=benchmark_dir,
        scenario_dir=scenario_dir,
    )

    assert summary["claim_status"] == "fail"
    assert frame[frame["claim_gate"].eq("uncertainty_modes_keep_pyrolysis_top")].iloc[0]["status"] == "fail"


def test_validated_claim_fails_when_benchmark_and_uncertainty_do_not_cover_all_planning_scenarios(tmp_path):
    planning_dir = tmp_path / "planning"
    benchmark_dir = tmp_path / "benchmark"
    scenario_dir = tmp_path / "scenarios"
    planning_dir.mkdir()
    benchmark_dir.mkdir()
    scenario_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": scenario,
                "pathway": "pyrolysis",
                "selected_allocated_feed_share": 1.0,
                "worst_surrogate_evidence_gate": "conditional_transfer",
                "optimization_supported_prediction_fraction": 1.0,
            }
            for scenario in ("s1", "s2")
        ]
    ).to_csv(planning_dir / "surrogate_transferability_summary.csv", index=False)
    pd.DataFrame(
        [{"scenario_name": "s1", "benchmark_variant": "baseline_evidence_aware", "selected_pathways": "pyrolysis"}]
        + [
            {"scenario_name": "s1", "benchmark_variant": variant, "selected_pathways": "pyrolysis"}
            for variant in MAJOR_BENCHMARK_COMPARATORS
        ]
    ).to_csv(benchmark_dir / "benchmark_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "benchmark_variant": variant,
                "bootstrap_replicate_count": 16,
                "effect_significance_tier": "directionally_consistent",
            }
            for variant in MAJOR_BENCHMARK_COMPARATORS
        ]
    ).to_csv(benchmark_dir / "benchmark_statistical_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "baseline_top_pathway": "pyrolysis",
                "combined_only_top_pathway": "pyrolysis",
                "max_interval_top_pathway": "pyrolysis",
                "dominant_selection_rate": 0.9,
            }
        ]
    ).to_csv(scenario_dir / "uncertainty_summary.csv", index=False)

    frame, summary = build_validated_pyrolysis_superiority_claim(
        planning_dir=planning_dir,
        benchmark_dir=benchmark_dir,
        scenario_dir=scenario_dir,
    )

    assert summary["claim_status"] == "fail"
    failed = set(frame[frame["status"].eq("fail")]["claim_gate"])
    assert "benchmark_baseline_selects_pyrolysis" in failed
    assert "benchmark_core_comparators_select_pyrolysis" in failed
    assert "scenario_uncertainty_scenario_coverage_complete" in failed


def test_validated_claim_fails_when_benchmark_statistics_do_not_cover_all_planning_scenarios(tmp_path):
    planning_dir = tmp_path / "planning"
    benchmark_dir = tmp_path / "benchmark"
    scenario_dir = tmp_path / "scenarios"
    planning_dir.mkdir()
    benchmark_dir.mkdir()
    scenario_dir.mkdir()
    expected_scenarios = ("s1", "s2")
    pd.DataFrame(
        [
            {
                "scenario_name": scenario,
                "pathway": "pyrolysis",
                "selected_allocated_feed_share": 1.0,
                "worst_surrogate_evidence_gate": "conditional_transfer",
                "optimization_supported_prediction_fraction": 1.0,
            }
            for scenario in expected_scenarios
        ]
    ).to_csv(planning_dir / "surrogate_transferability_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_name": scenario, "benchmark_variant": "baseline_evidence_aware", "selected_pathways": "pyrolysis"}
            for scenario in expected_scenarios
        ]
        + [
            {"scenario_name": scenario, "benchmark_variant": variant, "selected_pathways": "pyrolysis"}
            for scenario in expected_scenarios
            for variant in MAJOR_BENCHMARK_COMPARATORS
        ]
    ).to_csv(benchmark_dir / "benchmark_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "benchmark_variant": variant,
                "bootstrap_replicate_count": 16,
                "effect_significance_tier": "directionally_consistent",
            }
            for variant in MAJOR_BENCHMARK_COMPARATORS
        ]
    ).to_csv(benchmark_dir / "benchmark_statistical_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": scenario,
                "baseline_top_pathway": "pyrolysis",
                "combined_only_top_pathway": "pyrolysis",
                "max_interval_top_pathway": "pyrolysis",
                "dominant_selection_rate": 0.9,
            }
            for scenario in expected_scenarios
        ]
    ).to_csv(scenario_dir / "uncertainty_summary.csv", index=False)

    frame, summary = build_validated_pyrolysis_superiority_claim(
        planning_dir=planning_dir,
        benchmark_dir=benchmark_dir,
        scenario_dir=scenario_dir,
    )

    assert summary["claim_status"] == "fail"
    failed = set(frame[frame["status"].eq("fail")]["claim_gate"])
    assert "benchmark_core_statistical_scenario_comparator_coverage_complete" in failed
    assert "benchmark_bootstrap_replicates_sufficient" in failed


def test_validated_claim_fails_when_benchmark_effect_tiers_are_missing(tmp_path):
    planning_dir = tmp_path / "planning"
    benchmark_dir = tmp_path / "benchmark"
    scenario_dir = tmp_path / "scenarios"
    planning_dir.mkdir()
    benchmark_dir.mkdir()
    scenario_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "pathway": "pyrolysis",
                "selected_allocated_feed_share": 1.0,
                "worst_surrogate_evidence_gate": "conditional_transfer",
                "optimization_supported_prediction_fraction": 1.0,
            }
        ]
    ).to_csv(planning_dir / "surrogate_transferability_summary.csv", index=False)
    pd.DataFrame(
        [{"scenario_name": "s1", "benchmark_variant": "baseline_evidence_aware", "selected_pathways": "pyrolysis"}]
        + [
            {"scenario_name": "s1", "benchmark_variant": variant, "selected_pathways": "pyrolysis"}
            for variant in MAJOR_BENCHMARK_COMPARATORS
        ]
    ).to_csv(benchmark_dir / "benchmark_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "benchmark_variant": variant,
                "bootstrap_replicate_count": 16,
            }
            for variant in MAJOR_BENCHMARK_COMPARATORS
        ]
    ).to_csv(benchmark_dir / "benchmark_statistical_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "baseline_top_pathway": "pyrolysis",
                "combined_only_top_pathway": "pyrolysis",
                "max_interval_top_pathway": "pyrolysis",
                "dominant_selection_rate": 0.9,
            }
        ]
    ).to_csv(scenario_dir / "uncertainty_summary.csv", index=False)

    frame, summary = build_validated_pyrolysis_superiority_claim(
        planning_dir=planning_dir,
        benchmark_dir=benchmark_dir,
        scenario_dir=scenario_dir,
    )

    assert summary["claim_status"] == "fail"
    failed = set(frame[frame["status"].eq("fail")]["claim_gate"])
    assert "benchmark_core_statistical_effect_tiers_available" in failed
    assert "benchmark_bootstrap_replicates_sufficient" in failed


def test_validated_claim_bootstrap_count_uses_expected_scenario_comparator_rows(tmp_path):
    planning_dir = tmp_path / "planning"
    benchmark_dir = tmp_path / "benchmark"
    scenario_dir = tmp_path / "scenarios"
    planning_dir.mkdir()
    benchmark_dir.mkdir()
    scenario_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "pathway": "pyrolysis",
                "selected_allocated_feed_share": 1.0,
                "worst_surrogate_evidence_gate": "conditional_transfer",
                "optimization_supported_prediction_fraction": 1.0,
            }
        ]
    ).to_csv(planning_dir / "surrogate_transferability_summary.csv", index=False)
    pd.DataFrame(
        [{"scenario_name": "s1", "benchmark_variant": "baseline_evidence_aware", "selected_pathways": "pyrolysis"}]
        + [
            {"scenario_name": "s1", "benchmark_variant": variant, "selected_pathways": "pyrolysis"}
            for variant in MAJOR_BENCHMARK_COMPARATORS
        ]
    ).to_csv(benchmark_dir / "benchmark_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "benchmark_variant": variant,
                "bootstrap_replicate_count": 16,
                "effect_significance_tier": "directionally_consistent",
            }
            for variant in MAJOR_BENCHMARK_COMPARATORS
        ]
        + [
            {
                "scenario_name": "non_claim_extra",
                "benchmark_variant": MAJOR_BENCHMARK_COMPARATORS[0],
                "bootstrap_replicate_count": 2,
                "effect_significance_tier": "smoke_only",
            }
        ]
    ).to_csv(benchmark_dir / "benchmark_statistical_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "baseline_top_pathway": "pyrolysis",
                "combined_only_top_pathway": "pyrolysis",
                "max_interval_top_pathway": "pyrolysis",
                "dominant_selection_rate": 0.9,
            }
        ]
    ).to_csv(scenario_dir / "uncertainty_summary.csv", index=False)

    frame, summary = build_validated_pyrolysis_superiority_claim(
        planning_dir=planning_dir,
        benchmark_dir=benchmark_dir,
        scenario_dir=scenario_dir,
    )

    assert summary["claim_status"] == "pass"
    assert (
        frame[frame["claim_gate"].eq("benchmark_bootstrap_replicates_sufficient")].iloc[0]["status"]
        == "pass"
    )


def test_validated_claim_fails_when_expected_benchmark_effect_tier_is_unstable(tmp_path):
    planning_dir = tmp_path / "planning"
    benchmark_dir = tmp_path / "benchmark"
    scenario_dir = tmp_path / "scenarios"
    planning_dir.mkdir()
    benchmark_dir.mkdir()
    scenario_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "pathway": "pyrolysis",
                "selected_allocated_feed_share": 1.0,
                "worst_surrogate_evidence_gate": "conditional_transfer",
                "optimization_supported_prediction_fraction": 1.0,
            }
        ]
    ).to_csv(planning_dir / "surrogate_transferability_summary.csv", index=False)
    pd.DataFrame(
        [{"scenario_name": "s1", "benchmark_variant": "baseline_evidence_aware", "selected_pathways": "pyrolysis"}]
        + [
            {"scenario_name": "s1", "benchmark_variant": variant, "selected_pathways": "pyrolysis"}
            for variant in MAJOR_BENCHMARK_COMPARATORS
        ]
    ).to_csv(benchmark_dir / "benchmark_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "benchmark_variant": variant,
                "bootstrap_replicate_count": 16,
                "effect_significance_tier": (
                    "unstable" if variant == MAJOR_BENCHMARK_COMPARATORS[0] else "directionally_consistent"
                ),
            }
            for variant in MAJOR_BENCHMARK_COMPARATORS
        ]
    ).to_csv(benchmark_dir / "benchmark_statistical_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "baseline_top_pathway": "pyrolysis",
                "combined_only_top_pathway": "pyrolysis",
                "max_interval_top_pathway": "pyrolysis",
                "dominant_selection_rate": 0.9,
            }
        ]
    ).to_csv(scenario_dir / "uncertainty_summary.csv", index=False)

    frame, summary = build_validated_pyrolysis_superiority_claim(
        planning_dir=planning_dir,
        benchmark_dir=benchmark_dir,
        scenario_dir=scenario_dir,
    )

    assert summary["claim_status"] == "fail"
    failed = set(frame[frame["status"].eq("fail")]["claim_gate"])
    assert "benchmark_core_statistical_effect_tiers_not_unstable" in failed
    assert (
        frame[frame["claim_gate"].eq("benchmark_bootstrap_replicates_sufficient")].iloc[0]["status"]
        == "pass"
    )


def test_validated_claim_treats_diagnostic_unstable_tiers_as_warning(tmp_path):
    planning_dir = tmp_path / "planning"
    benchmark_dir = tmp_path / "benchmark"
    scenario_dir = tmp_path / "scenarios"
    planning_dir.mkdir()
    benchmark_dir.mkdir()
    scenario_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "pathway": "pyrolysis",
                "selected_allocated_feed_share": 1.0,
                "worst_surrogate_evidence_gate": "conditional_transfer",
                "optimization_supported_prediction_fraction": 1.0,
            }
        ]
    ).to_csv(planning_dir / "surrogate_transferability_summary.csv", index=False)
    pd.DataFrame(
        [{"scenario_name": "s1", "benchmark_variant": "baseline_evidence_aware", "selected_pathways": "pyrolysis"}]
        + [
            {"scenario_name": "s1", "benchmark_variant": variant, "selected_pathways": "pyrolysis"}
            for variant in MAJOR_BENCHMARK_COMPARATORS
        ]
    ).to_csv(benchmark_dir / "benchmark_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "benchmark_variant": variant,
                "bootstrap_replicate_count": 16,
                "effect_significance_tier": "directionally_consistent",
            }
            for variant in CORE_BENCHMARK_COMPARATORS
        ]
        + [
            {
                "scenario_name": "s1",
                "benchmark_variant": variant,
                "bootstrap_replicate_count": 16,
                "effect_significance_tier": "unstable",
            }
            for variant in DIAGNOSTIC_BENCHMARK_COMPARATORS
        ]
    ).to_csv(benchmark_dir / "benchmark_statistical_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "s1",
                "baseline_top_pathway": "pyrolysis",
                "combined_only_top_pathway": "pyrolysis",
                "max_interval_top_pathway": "pyrolysis",
                "dominant_selection_rate": 0.9,
            }
        ]
    ).to_csv(scenario_dir / "uncertainty_summary.csv", index=False)

    frame, summary = build_validated_pyrolysis_superiority_claim(
        planning_dir=planning_dir,
        benchmark_dir=benchmark_dir,
        scenario_dir=scenario_dir,
    )

    assert summary["claim_status"] == "pass"
    assert frame[frame["claim_gate"].eq("benchmark_diagnostic_effect_tiers_contextual")].iloc[0]["status"] == "warning"
