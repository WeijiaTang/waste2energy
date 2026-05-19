# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json

import pandas as pd

from scripts.check_submission_readiness import build_readiness_report


def _write_ready_tree(root):
    audit_dir = root / "outputs" / "audit"
    benchmark_dir = root / "outputs" / "benchmark" / "baseline"
    targeted_dir = benchmark_dir / "targeted_planning_ablations"
    paper_dir = root / "waste2energy-paper"
    for directory in (audit_dir, benchmark_dir, targeted_dir, paper_dir):
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
