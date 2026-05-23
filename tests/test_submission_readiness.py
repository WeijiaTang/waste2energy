# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json

import pandas as pd

from scripts.check_submission_readiness import build_readiness_report


def _write_ready_tree(root):
    audit_dir = root / "outputs" / "audit"
    planning_dir = root / "outputs" / "planning"
    figures_dir = root / "data" / "processed" / "figures_tables"
    benchmark_dir = root / "outputs" / "benchmark" / "baseline"
    targeted_dir = benchmark_dir / "targeted_planning_ablations"
    paper_dir = root / "waste2energy-paper"
    for directory in (audit_dir, planning_dir, figures_dir, benchmark_dir, targeted_dir, paper_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"consistency_status": "pass"}]).to_csv(
        audit_dir / "planning_artifact_consistency_summary.csv",
        index=False,
    )
    (benchmark_dir / "run_config.json").write_text(
        json.dumps({"bootstrap_replicate_count": 8}),
        encoding="utf-8",
    )
    pd.DataFrame([{"benchmark_variant": "classic_multiobjective_optimizer"}]).to_csv(
        benchmark_dir / "benchmark_statistical_summary.csv",
        index=False,
    )
    (targeted_dir / "run_config.json").write_text(
        json.dumps({"monte_carlo_replicate_count": 16}),
        encoding="utf-8",
    )
    pd.DataFrame([{"ablation_family": "objective_weight_sweep"}]).to_csv(
        targeted_dir / "targeted_planning_ablations_summary.csv",
        index=False,
    )
    pd.DataFrame([{"scenario_name": "baseline_region_case"}]).to_csv(
        targeted_dir / "monte_carlo_uq_summary.csv",
        index=False,
    )
    pd.DataFrame([{"scenario_name": "baseline_region_case", "pathway": "pyrolysis"}]).to_csv(
        audit_dir / "hhv_imputation_sensitivity.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "replanning_status": "replanned",
            }
        ]
    ).to_csv(audit_dir / "hhv_replanning_sensitivity.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "hhv_dominance_conclusion": "not_pathway_dominant_but_case_sensitive",
                "audit_status": "evaluated",
                "selected_pathways_changed": False,
                "max_abs_pathway_share_change_pct_point": 0.5,
            }
        ]
    ).to_csv(audit_dir / "hhv_dominance_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "leave_study_out_target_count": 4,
                "feature_range_status": "evaluated",
                "extrapolation_evidence_ceiling": "screening_only_external_validity_not_established",
            }
        ]
    ).to_csv(audit_dir / "surrogate_extrapolation_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "ad_min_10pct_floor_share_pct": 18.0,
                "ad_min_20pct_floor_share_pct": 25.0,
                "ad_policy_floor_feasible": True,
                "ad_boundary_evidence_status": "evaluated",
                "ad_role_conclusion": "boundary_reference_not_technical_inferiority",
            }
        ]
    ).to_csv(audit_dir / "ad_boundary_fairness_audit.csv", index=False)
    pd.DataFrame([{"scenario_name": "baseline_region_case", "candidate_cap_binding": True}]).to_csv(
        audit_dir / "binding_constraint_audit.csv",
        index=False,
    )
    pd.DataFrame([{"scenario_name": "baseline_region_case", "audit_finding": "duplicate"}]).to_csv(
        audit_dir / "duplicate_candidate_audit.csv",
        index=False,
    )
    pd.DataFrame([{"scenario_name": "baseline_region_case", "pathway": "pyrolysis"}]).to_csv(
        planning_dir / "main_results_table_thermochemical.csv",
        index=False,
    )
    pd.DataFrame([{"scenario_name": "baseline_region_case", "pathway": "ad"}]).to_csv(
        planning_dir / "ad_reference_diagnostics.csv",
        index=False,
    )
    pd.DataFrame([{"scenario_name": "baseline_region_case", "pathway": "pyrolysis"}]).to_csv(
        figures_dir / "paper1_planning_results_table.csv",
        index=False,
    )
    pd.DataFrame([{"scenario": "baseline-region", "pathway": "pyrolysis"}]).to_csv(
        figures_dir / "paper1_monte_carlo_uq_table.csv",
        index=False,
    )
    for stem, rows in {
        "paper1_hhv_dominance_audit_table": [
            {
                "scenario": "baseline-region",
                "hhv_dominance_conclusion": "not_pathway_dominant_but_case_sensitive",
                "max_abs_pathway_share_change_pct_point": 0.5,
            }
        ],
        "paper1_surrogate_extrapolation_audit_table": [{"scenario": "baseline-region", "pathway": "pyrolysis", "extrapolation_evidence_ceiling": "screening_only_external_validity_not_established"}],
        "paper1_ad_boundary_fairness_audit_table": [
            {
                "scenario": "baseline-region",
                "ad_min_10pct_floor_share_pct": 18.0,
                "ad_min_20pct_floor_share_pct": 25.0,
                "ad_boundary_evidence_status": "evaluated",
                "ad_role_conclusion": "boundary_reference_not_technical_inferiority",
            }
        ],
    }.items():
        pd.DataFrame(rows).to_csv(figures_dir / f"{stem}.csv", index=False)
        (figures_dir / f"{stem}.tex").write_text("table content", encoding="utf-8")
    (paper_dir / "main.pdf").write_bytes(b"%PDF-1.7\n")


def test_submission_readiness_passes_when_required_artifacts_exist(tmp_path):
    _write_ready_tree(tmp_path)

    report = build_readiness_report(
        repo_root=tmp_path,
        require_bootstrap=True,
        min_bootstrap_replicates=4,
        require_targeted_ablations=True,
        min_monte_carlo_replicates=8,
        require_pdf=True,
    )

    assert report["ready"] is True
    assert not report["failures"]


def test_submission_readiness_fails_on_stale_planning_artifacts(tmp_path):
    _write_ready_tree(tmp_path)
    pd.DataFrame([{"consistency_status": "fail"}]).to_csv(
        tmp_path / "outputs" / "audit" / "planning_artifact_consistency_summary.csv",
        index=False,
    )

    report = build_readiness_report(repo_root=tmp_path)

    assert report["ready"] is False
    assert report["failures"][0]["name"] == "planning_artifact_consistency"


def test_submission_readiness_requires_bootstrap_when_requested(tmp_path):
    _write_ready_tree(tmp_path)
    (tmp_path / "outputs" / "benchmark" / "baseline" / "run_config.json").write_text(
        json.dumps({"bootstrap_replicate_count": 2}),
        encoding="utf-8",
    )

    report = build_readiness_report(
        repo_root=tmp_path,
        require_bootstrap=True,
        min_bootstrap_replicates=8,
    )

    assert report["ready"] is False
    assert any(failure["name"] == "benchmark_bootstrap_statistics" for failure in report["failures"])


def test_submission_readiness_fails_when_phase2_main_table_contains_ad(tmp_path):
    _write_ready_tree(tmp_path)
    pd.DataFrame([{"scenario_name": "baseline_region_case", "pathway": "ad"}]).to_csv(
        tmp_path / "data" / "processed" / "figures_tables" / "paper1_planning_results_table.csv",
        index=False,
    )

    report = build_readiness_report(repo_root=tmp_path)

    assert report["ready"] is False
    assert any(failure["name"] == "phase2_manuscript_main_table_excludes_ad" for failure in report["failures"])


def test_submission_readiness_fails_when_hhv_audit_flags_pathway_dominance(tmp_path):
    _write_ready_tree(tmp_path)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "hhv_dominance_conclusion": "potentially_pathway_dominant",
            }
        ]
    ).to_csv(tmp_path / "outputs" / "audit" / "hhv_dominance_audit.csv", index=False)

    report = build_readiness_report(repo_root=tmp_path)

    assert report["ready"] is False
    assert any(failure["name"] == "phase2_hhv_not_pathway_dominant" for failure in report["failures"])


def test_submission_readiness_fails_when_hhv_audit_not_evaluated(tmp_path):
    _write_ready_tree(tmp_path)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "hhv_dominance_conclusion": "not_evaluated",
                "audit_status": "not_evaluated",
                "selected_pathways_changed": False,
                "max_abs_pathway_share_change_pct_point": pd.NA,
            }
        ]
    ).to_csv(tmp_path / "outputs" / "audit" / "hhv_dominance_audit.csv", index=False)

    report = build_readiness_report(repo_root=tmp_path)

    assert report["ready"] is False
    assert any(failure["name"] == "phase2_hhv_not_pathway_dominant" for failure in report["failures"])


def test_submission_readiness_requires_surrogate_screening_ceiling(tmp_path):
    _write_ready_tree(tmp_path)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "extrapolation_evidence_ceiling": "external_validation_established",
            }
        ]
    ).to_csv(tmp_path / "outputs" / "audit" / "surrogate_extrapolation_audit.csv", index=False)

    report = build_readiness_report(repo_root=tmp_path)

    assert report["ready"] is False
    assert any(failure["name"] == "phase2_surrogate_screening_ceiling" for failure in report["failures"])


def test_submission_readiness_rejects_incomplete_surrogate_validation(tmp_path):
    _write_ready_tree(tmp_path)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "leave_study_out_target_count": 0,
                "feature_range_status": "training_range_unavailable",
                "extrapolation_evidence_ceiling": "screening_only_validation_incomplete",
            }
        ]
    ).to_csv(tmp_path / "outputs" / "audit" / "surrogate_extrapolation_audit.csv", index=False)

    report = build_readiness_report(repo_root=tmp_path)

    assert report["ready"] is False
    assert any(failure["name"] == "phase2_surrogate_screening_ceiling" for failure in report["failures"])


def test_submission_readiness_requires_ad_non_inferiority_boundary(tmp_path):
    _write_ready_tree(tmp_path)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "ad_role_conclusion": "ad_underperforms",
            }
        ]
    ).to_csv(tmp_path / "outputs" / "audit" / "ad_boundary_fairness_audit.csv", index=False)

    report = build_readiness_report(repo_root=tmp_path)

    assert report["ready"] is False
    assert any(failure["name"] == "phase2_ad_not_technical_inferiority" for failure in report["failures"])


def test_submission_readiness_fails_when_ad_floor_ablation_missing(tmp_path):
    _write_ready_tree(tmp_path)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "ad_min_10pct_floor_share_pct": pd.NA,
                "ad_min_20pct_floor_share_pct": 20.0,
                "ad_policy_floor_feasible": False,
                "ad_boundary_evidence_status": "missing_ad_floor_10pct",
                "ad_role_conclusion": "boundary_evidence_incomplete",
            }
        ]
    ).to_csv(tmp_path / "outputs" / "audit" / "ad_boundary_fairness_audit.csv", index=False)

    report = build_readiness_report(repo_root=tmp_path)

    assert report["ready"] is False
    assert any(failure["name"] == "phase2_ad_not_technical_inferiority" for failure in report["failures"])


def test_submission_readiness_checks_new_manuscript_audit_tables(tmp_path):
    _write_ready_tree(tmp_path)
    (tmp_path / "data" / "processed" / "figures_tables" / "paper1_hhv_dominance_audit_table.tex").write_text(
        "unavailable",
        encoding="utf-8",
    )

    report = build_readiness_report(repo_root=tmp_path)

    assert report["ready"] is False
    assert any(failure["name"] == "phase2_manuscript_hhv_dominance_audit_table" for failure in report["failures"])


def test_submission_readiness_fails_when_manuscript_hhv_audit_disagrees_with_source(tmp_path):
    _write_ready_tree(tmp_path)
    pd.DataFrame(
        [
            {
                "scenario": "baseline-region",
                "hhv_dominance_conclusion": "potentially_pathway_dominant",
                "max_abs_pathway_share_change_pct_point": 0.5,
            }
        ]
    ).to_csv(
        tmp_path / "data" / "processed" / "figures_tables" / "paper1_hhv_dominance_audit_table.csv",
        index=False,
    )

    report = build_readiness_report(repo_root=tmp_path)

    assert report["ready"] is False
    assert any(failure["name"] == "phase2_manuscript_hhv_dominance_audit_table" for failure in report["failures"])


def test_submission_readiness_fails_when_manuscript_surrogate_audit_disagrees_with_source(tmp_path):
    _write_ready_tree(tmp_path)
    pd.DataFrame(
        [
            {
                "scenario": "baseline-region",
                "pathway": "pyrolysis",
                "extrapolation_evidence_ceiling": "evidence_gated_screening_only",
            }
        ]
    ).to_csv(
        tmp_path / "data" / "processed" / "figures_tables" / "paper1_surrogate_extrapolation_audit_table.csv",
        index=False,
    )

    report = build_readiness_report(repo_root=tmp_path)

    assert report["ready"] is False
    assert any(
        failure["name"] == "phase2_manuscript_surrogate_extrapolation_audit_table"
        for failure in report["failures"]
    )


def test_submission_readiness_fails_when_manuscript_ad_audit_disagrees_with_source(tmp_path):
    _write_ready_tree(tmp_path)
    pd.DataFrame(
        [
            {
                "scenario": "baseline-region",
                "ad_min_10pct_floor_share_pct": 18.0,
                "ad_min_20pct_floor_share_pct": 5.0,
                "ad_boundary_evidence_status": "evaluated",
                "ad_role_conclusion": "boundary_reference_not_technical_inferiority",
            }
        ]
    ).to_csv(
        tmp_path / "data" / "processed" / "figures_tables" / "paper1_ad_boundary_fairness_audit_table.csv",
        index=False,
    )

    report = build_readiness_report(repo_root=tmp_path)

    assert report["ready"] is False
    assert any(
        failure["name"] == "phase2_manuscript_ad_boundary_fairness_audit_table"
        for failure in report["failures"]
    )


def test_submission_readiness_fails_when_ad_20pct_floor_not_met(tmp_path):
    _write_ready_tree(tmp_path)
    pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "ad_min_10pct_floor_share_pct": 10.0,
                "ad_min_20pct_floor_share_pct": 5.0,
                "ad_policy_floor_feasible": True,
                "ad_boundary_evidence_status": "evaluated",
                "ad_role_conclusion": "boundary_reference_not_technical_inferiority",
            }
        ]
    ).to_csv(tmp_path / "outputs" / "audit" / "ad_boundary_fairness_audit.csv", index=False)

    report = build_readiness_report(repo_root=tmp_path)

    assert report["ready"] is False
    assert any(failure["name"] == "phase2_ad_not_technical_inferiority" for failure in report["failures"])
