from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "selected"}


def _scenario_display(value: object) -> str:
    mapping = {
        "baseline_region_case": "baseline-region",
        "high_supply_case": "high-supply",
        "policy_support_case": "policy-support",
    }
    text = str(value)
    return mapping.get(text, text)


def _pathway_display(value: object) -> str:
    text = str(value).strip().lower()
    return {"htc": "htc", "pyrolysis": "pyrolysis", "ad": "ad"}.get(text, text)


def _hhv_dominance_display(value: object) -> str:
    text = str(value or "").strip()
    mapping = {
        "not_pathway_dominant": "Pathway-stable",
        "not_pathway_dominant_but_case_sensitive": "Pathway-stable; selected cases can switch",
        "limited_share_sensitivity_not_pathway_dominant": "Small share shifts; pathway-stable",
        "potentially_pathway_dominant": "Potentially pathway-dominant",
        "not_evaluated": "Not evaluated",
        "unavailable": "Unavailable",
    }
    return mapping.get(text, text.replace("_", " ") if text else "--")


def _source_with_display_keys(source: pd.DataFrame) -> pd.DataFrame:
    working = source.copy()
    if "scenario_name" in working.columns:
        working["_scenario_key"] = working["scenario_name"].map(_scenario_display)
    if "pathway" in working.columns:
        working["_pathway_key"] = working["pathway"].map(_pathway_display)
    return working


def _table_with_display_keys(table: pd.DataFrame) -> pd.DataFrame:
    working = table.copy()
    if "scenario" in working.columns:
        working["_scenario_key"] = working["scenario"].astype(str)
    if "pathway" in working.columns:
        working["_pathway_key"] = working["pathway"].map(_pathway_display)
    return working


def _numeric_columns_close(
    merged: pd.DataFrame,
    *,
    columns: list[str],
    tolerance: float = 0.11,
) -> bool:
    for column in columns:
        source_column = f"{column}__source"
        table_column = f"{column}__table"
        if source_column not in merged.columns or table_column not in merged.columns:
            return False
        source_values = pd.to_numeric(merged[source_column], errors="coerce")
        table_values = pd.to_numeric(merged[table_column], errors="coerce")
        if source_values.isna().any() or table_values.isna().any():
            return False
        if (source_values - table_values).abs().gt(tolerance).any():
            return False
    return True


def _string_columns_match(merged: pd.DataFrame, *, columns: list[str]) -> bool:
    for column in columns:
        source_column = f"{column}__source"
        table_column = f"{column}__table"
        if source_column not in merged.columns or table_column not in merged.columns:
            return False
        if not merged[source_column].astype(str).eq(merged[table_column].astype(str)).all():
            return False
    return True


def _manuscript_audit_table_matches_source(name: str, source: pd.DataFrame, table: pd.DataFrame) -> bool:
    if source.empty or table.empty:
        return False
    source_keyed = _source_with_display_keys(source)
    table_keyed = _table_with_display_keys(table)
    if name == "hhv_dominance_audit":
        keys = ["_scenario_key"]
        required = {"hhv_dominance_conclusion", "max_abs_pathway_share_change_pct_point"}
        numeric = ["max_abs_pathway_share_change_pct_point"]
        strings = ["hhv_dominance_conclusion"]
        source_keyed["hhv_dominance_conclusion"] = source_keyed["hhv_dominance_conclusion"].map(
            _hhv_dominance_display
        )
        table_keyed["hhv_dominance_conclusion"] = table_keyed["hhv_dominance_conclusion"].map(
            _hhv_dominance_display
        )
    elif name == "surrogate_extrapolation_audit":
        keys = ["_scenario_key", "_pathway_key"]
        required = {"extrapolation_evidence_ceiling"}
        numeric = []
        strings = ["extrapolation_evidence_ceiling"]
    elif name == "ad_boundary_fairness_audit":
        keys = ["_scenario_key"]
        required = {
            "ad_boundary_evidence_status",
            "ad_role_conclusion",
            "ad_min_10pct_floor_share_pct",
            "ad_min_20pct_floor_share_pct",
        }
        numeric = ["ad_min_10pct_floor_share_pct", "ad_min_20pct_floor_share_pct"]
        strings = ["ad_boundary_evidence_status", "ad_role_conclusion"]
    else:
        return True
    if any(key not in source_keyed.columns or key not in table_keyed.columns for key in keys):
        return False
    if not required.issubset(source_keyed.columns) or not required.issubset(table_keyed.columns):
        return False
    source_compare = source_keyed[keys + sorted(required)].copy()
    table_compare = table_keyed[keys + sorted(required)].copy()
    merged = source_compare.merge(
        table_compare,
        on=keys,
        how="outer",
        suffixes=("__source", "__table"),
        indicator=True,
    )
    if merged.empty or not merged["_merge"].eq("both").all():
        return False
    return _string_columns_match(merged, columns=strings) and _numeric_columns_close(
        merged,
        columns=numeric,
    )


def _check(
    checks: list[dict[str, object]],
    *,
    name: str,
    passed: bool,
    detail: str,
    path: Path | None = None,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "detail": detail,
            "path": str(path) if path is not None else "",
        }
    )


def build_readiness_report(
    *,
    repo_root: Path = ROOT,
    require_bootstrap: bool = False,
    min_bootstrap_replicates: int = 1,
    require_targeted_ablations: bool = False,
    min_monte_carlo_replicates: int = 1,
    require_pdf: bool = False,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    consistency_path = repo_root / "outputs" / "audit" / "planning_artifact_consistency_summary.csv"
    consistency = _read_csv(consistency_path)
    consistency_passed = (
        not consistency.empty
        and "consistency_status" in consistency.columns
        and consistency["consistency_status"].astype(str).eq("pass").all()
    )
    _check(
        checks,
        name="planning_artifact_consistency",
        passed=consistency_passed,
        detail=(
            "All manuscript-facing planning shares/profiles and cost summaries match canonical planning outputs."
            if consistency_passed
            else "Missing or failing planning artifact consistency rows for shares/profiles/costs."
        ),
        path=consistency_path,
    )

    benchmark_dir = repo_root / "outputs" / "benchmark" / "baseline"
    benchmark_run_config_path = benchmark_dir / "run_config.json"
    benchmark_run_config = _read_json(benchmark_run_config_path)
    bootstrap_count = int(benchmark_run_config.get("bootstrap_replicate_count", 0) or 0)
    bootstrap_stats_path = benchmark_dir / "benchmark_statistical_summary.csv"
    bootstrap_stats = _read_csv(bootstrap_stats_path)
    bootstrap_passed = bootstrap_count >= int(min_bootstrap_replicates) and not bootstrap_stats.empty
    _check(
        checks,
        name="benchmark_bootstrap_statistics",
        passed=(bootstrap_passed if require_bootstrap else True),
        detail=(
            f"Bootstrap replicate count={bootstrap_count}; statistical rows={len(bootstrap_stats)}."
            if require_bootstrap
            else "Bootstrap statistics not required by this readiness invocation."
        ),
        path=bootstrap_stats_path,
    )

    targeted_dirs = [
        benchmark_dir / "targeted_planning_ablations",
        repo_root / "outputs" / "benchmark" / "targeted_planning_ablations",
    ]
    targeted_dir = next((path for path in targeted_dirs if (path / "run_config.json").exists()), targeted_dirs[0])
    targeted_run_config_path = targeted_dir / "run_config.json"
    targeted_run_config = _read_json(targeted_run_config_path)
    monte_carlo_count = int(targeted_run_config.get("monte_carlo_replicate_count", 0) or 0)
    targeted_summary_path = targeted_dir / "targeted_planning_ablations_summary.csv"
    monte_carlo_summary_path = targeted_dir / "monte_carlo_uq_summary.csv"
    targeted_summary = _read_csv(targeted_summary_path)
    monte_carlo_summary = _read_csv(monte_carlo_summary_path)
    targeted_passed = (
        monte_carlo_count >= int(min_monte_carlo_replicates)
        and not targeted_summary.empty
        and not monte_carlo_summary.empty
    )
    _check(
        checks,
        name="targeted_ablation_monte_carlo",
        passed=(targeted_passed if require_targeted_ablations else True),
        detail=(
            f"Monte Carlo replicate count={monte_carlo_count}; targeted rows={len(targeted_summary)}; "
            f"Monte Carlo summary rows={len(monte_carlo_summary)}."
            if require_targeted_ablations
            else "Targeted ablations not required by this readiness invocation."
        ),
        path=targeted_dir,
    )

    phase2_audit_specs = [
        ("hhv_imputation_sensitivity", repo_root / "outputs" / "audit" / "hhv_imputation_sensitivity.csv"),
        ("hhv_replanning_sensitivity", repo_root / "outputs" / "audit" / "hhv_replanning_sensitivity.csv"),
        ("hhv_dominance_audit", repo_root / "outputs" / "audit" / "hhv_dominance_audit.csv"),
        ("surrogate_extrapolation_audit", repo_root / "outputs" / "audit" / "surrogate_extrapolation_audit.csv"),
        ("ad_boundary_fairness_audit", repo_root / "outputs" / "audit" / "ad_boundary_fairness_audit.csv"),
        ("binding_constraint_audit", repo_root / "outputs" / "audit" / "binding_constraint_audit.csv"),
        ("duplicate_candidate_audit", repo_root / "outputs" / "audit" / "duplicate_candidate_audit.csv"),
    ]
    for name, path in phase2_audit_specs:
        frame = _read_csv(path)
        status_ok = (
            "replanning_status" not in frame.columns
            or not frame["replanning_status"].astype(str).str.lower().isin(["failed", "failed_to_initialize"]).any()
        )
        passed = not frame.empty and status_ok
        _check(
            checks,
            name=f"phase2_{name}",
            passed=passed,
            detail=(
                f"Phase 2 artifact exists with {len(frame)} row(s)."
                if passed
                else "Phase 2 artifact is missing, empty, or contains failed replanning rows."
            ),
            path=path,
        )

    hhv_dominance_path = repo_root / "outputs" / "audit" / "hhv_dominance_audit.csv"
    hhv_dominance = _read_csv(hhv_dominance_path)
    hhv_allowed = {
        "not_pathway_dominant",
        "not_pathway_dominant_but_case_sensitive",
        "limited_share_sensitivity_not_pathway_dominant",
    }
    hhv_required = {
        "audit_status",
        "hhv_dominance_conclusion",
        "selected_pathways_changed",
        "max_abs_pathway_share_change_pct_point",
    }
    hhv_not_dominant = False
    if not hhv_dominance.empty and hhv_required.issubset(hhv_dominance.columns):
        hhv_not_dominant = (
            hhv_dominance["audit_status"].astype(str).eq("evaluated").all()
            and hhv_dominance["hhv_dominance_conclusion"].astype(str).isin(hhv_allowed).all()
            and not hhv_dominance["selected_pathways_changed"].map(_coerce_bool).any()
            and pd.to_numeric(
                hhv_dominance["max_abs_pathway_share_change_pct_point"],
                errors="coerce",
            )
            .notna()
            .all()
        )
    _check(
        checks,
        name="phase2_hhv_not_pathway_dominant",
        passed=hhv_not_dominant,
        detail=(
            "HHV dominance audit does not flag pathway-dominant HHV stress."
            if hhv_not_dominant
            else "HHV dominance audit is missing or flags potentially pathway-dominant stress."
        ),
        path=hhv_dominance_path,
    )

    surrogate_extrapolation_path = repo_root / "outputs" / "audit" / "surrogate_extrapolation_audit.csv"
    surrogate_extrapolation = _read_csv(surrogate_extrapolation_path)
    surrogate_allowed = {
        "screening_only_external_validity_not_established",
        "evidence_gated_screening_only",
        "interpolation_range_flag_screening_only",
    }
    surrogate_required = {
        "leave_study_out_target_count",
        "feature_range_status",
        "extrapolation_evidence_ceiling",
    }
    surrogate_ceiling_ok = False
    if not surrogate_extrapolation.empty and surrogate_required.issubset(surrogate_extrapolation.columns):
        surrogate_ceiling_ok = (
            pd.to_numeric(surrogate_extrapolation["leave_study_out_target_count"], errors="coerce")
            .fillna(0.0)
            .gt(0.0)
            .all()
            and surrogate_extrapolation["feature_range_status"].astype(str).eq("evaluated").all()
            and surrogate_extrapolation["extrapolation_evidence_ceiling"].astype(str).isin(surrogate_allowed).all()
        )
    _check(
        checks,
        name="phase2_surrogate_screening_ceiling",
        passed=surrogate_ceiling_ok,
        detail=(
            "Surrogate extrapolation audit explicitly caps selected-pathway claims at screening scope."
            if surrogate_ceiling_ok
            else "Surrogate extrapolation audit is missing or lacks explicit screening claim ceilings."
        ),
        path=surrogate_extrapolation_path,
    )

    ad_boundary_path = repo_root / "outputs" / "audit" / "ad_boundary_fairness_audit.csv"
    ad_boundary = _read_csv(ad_boundary_path)
    ad_required = {
        "ad_role_conclusion",
        "ad_boundary_evidence_status",
        "ad_policy_floor_feasible",
        "ad_min_10pct_floor_share_pct",
        "ad_min_20pct_floor_share_pct",
    }
    ad_boundary_ok = False
    if not ad_boundary.empty and ad_required.issubset(ad_boundary.columns):
        ad_min_10 = pd.to_numeric(ad_boundary["ad_min_10pct_floor_share_pct"], errors="coerce")
        ad_min_20 = pd.to_numeric(ad_boundary["ad_min_20pct_floor_share_pct"], errors="coerce")
        ad_boundary_ok = (
            ad_boundary["ad_role_conclusion"].astype(str).eq("boundary_reference_not_technical_inferiority").all()
            and ad_boundary["ad_boundary_evidence_status"].astype(str).eq("evaluated").all()
            and ad_boundary["ad_policy_floor_feasible"].map(_coerce_bool).all()
            and ad_min_10.notna().all()
            and ad_min_20.notna().all()
            and ad_min_10.ge(9.9).all()
            and ad_min_20.ge(19.9).all()
        )
    _check(
        checks,
        name="phase2_ad_not_technical_inferiority",
        passed=ad_boundary_ok,
        detail=(
            "AD boundary audit explicitly prevents a technical-inferiority interpretation."
            if ad_boundary_ok
            else "AD boundary audit is missing or does not state the non-inferiority boundary."
        ),
        path=ad_boundary_path,
    )

    figures_dir = repo_root / "data" / "processed" / "figures_tables"
    manuscript_audit_specs = [
        (
            "hhv_dominance_audit",
            hhv_dominance_path,
            figures_dir / "paper1_hhv_dominance_audit_table.csv",
            figures_dir / "paper1_hhv_dominance_audit_table.tex",
        ),
        (
            "surrogate_extrapolation_audit",
            surrogate_extrapolation_path,
            figures_dir / "paper1_surrogate_extrapolation_audit_table.csv",
            figures_dir / "paper1_surrogate_extrapolation_audit_table.tex",
        ),
        (
            "ad_boundary_fairness_audit",
            ad_boundary_path,
            figures_dir / "paper1_ad_boundary_fairness_audit_table.csv",
            figures_dir / "paper1_ad_boundary_fairness_audit_table.tex",
        ),
    ]
    for name, source_path, csv_path, tex_path in manuscript_audit_specs:
        source = _read_csv(source_path)
        table = _read_csv(csv_path)
        csv_text = csv_path.read_text(encoding="utf-8") if csv_path.exists() else ""
        tex_text = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
        no_unavailable = "unavailable" not in csv_text.lower() and "unavailable" not in tex_text.lower()
        source_scenarios = (
            set(source["scenario_name"].dropna().astype(str))
            if "scenario_name" in source.columns
            else set()
        )
        table_scenarios = (
            set(table["scenario"].dropna().astype(str))
            if "scenario" in table.columns
            else set()
        )
        expected_display_scenarios = {_scenario_display(value) for value in source_scenarios}
        scenario_coverage_ok = not expected_display_scenarios or expected_display_scenarios.issubset(table_scenarios)
        source_pathways = (
            {value.lower() for value in source["pathway"].dropna().astype(str)}
            if "pathway" in source.columns
            else set()
        )
        table_pathways = (
            {value.lower() for value in table["pathway"].dropna().astype(str)}
            if "pathway" in table.columns
            else set()
        )
        expected_pathways = {_pathway_display(value) for value in source_pathways}
        pathway_coverage_ok = not expected_pathways or expected_pathways.issubset(table_pathways)
        content_consistent = _manuscript_audit_table_matches_source(name, source, table)
        passed = (
            not source.empty
            and not table.empty
            and csv_path.exists()
            and tex_path.exists()
            and bool(tex_text.strip())
            and no_unavailable
            and scenario_coverage_ok
            and pathway_coverage_ok
            and content_consistent
        )
        _check(
            checks,
            name=f"phase2_manuscript_{name}_table",
            passed=passed,
            detail=(
                f"Manuscript-facing {name} CSV/TEX exists and covers source audit rows."
                if passed
                else (
                    f"Manuscript-facing {name} table is missing, stale, contains unavailable, lacks "
                    "scenario/pathway coverage, or disagrees with source audit fields."
                )
            ),
            path=csv_path,
        )

    planning_thermo_path = repo_root / "outputs" / "planning" / "main_results_table_thermochemical.csv"
    planning_thermo = _read_csv(planning_thermo_path)
    planning_thermo_passed = (
        not planning_thermo.empty
        and "pathway" in planning_thermo.columns
        and set(planning_thermo["pathway"].astype(str).str.lower().dropna().unique()).issubset({"pyrolysis", "htc"})
    )
    _check(
        checks,
        name="phase2_thermochemical_main_results",
        passed=planning_thermo_passed,
        detail=(
            "Thermochemical main-results table exists and excludes AD."
            if planning_thermo_passed
            else "Thermochemical main-results table missing, empty, or contains non-thermochemical pathways."
        ),
        path=planning_thermo_path,
    )

    ad_reference_path = repo_root / "outputs" / "planning" / "ad_reference_diagnostics.csv"
    ad_reference = _read_csv(ad_reference_path)
    _check(
        checks,
        name="phase2_ad_reference_diagnostics",
        passed=not ad_reference.empty,
        detail=(
            f"AD reference diagnostics exists with {len(ad_reference)} row(s)."
            if not ad_reference.empty
            else "AD reference diagnostics is missing or empty."
        ),
        path=ad_reference_path,
    )

    manuscript_planning_path = figures_dir / "paper1_planning_results_table.csv"
    manuscript_planning = _read_csv(manuscript_planning_path)
    manuscript_planning_excludes_ad = (
        not manuscript_planning.empty
        and "pathway" in manuscript_planning.columns
        and not manuscript_planning["pathway"].astype(str).str.lower().eq("ad").any()
    )
    _check(
        checks,
        name="phase2_manuscript_main_table_excludes_ad",
        passed=manuscript_planning_excludes_ad,
        detail=(
            "Manuscript main planning table excludes AD."
            if manuscript_planning_excludes_ad
            else "Manuscript main planning table is missing, empty, or still contains AD."
        ),
        path=manuscript_planning_path,
    )

    manuscript_mc_path = figures_dir / "paper1_monte_carlo_uq_table.csv"
    manuscript_mc = _read_csv(manuscript_mc_path)
    manuscript_mc_excludes_ad = (
        not manuscript_mc.empty
        and "pathway" in manuscript_mc.columns
        and not manuscript_mc["pathway"].astype(str).str.lower().eq("ad").any()
    )
    _check(
        checks,
        name="phase2_monte_carlo_table_excludes_ad",
        passed=manuscript_mc_excludes_ad,
        detail=(
            "Manuscript Monte Carlo UQ table excludes AD."
            if manuscript_mc_excludes_ad
            else "Manuscript Monte Carlo UQ table is missing, empty, or still contains AD."
        ),
        path=manuscript_mc_path,
    )

    pdf_path = repo_root / "waste2energy-paper" / "main.pdf"
    pdf_passed = pdf_path.exists() and pdf_path.stat().st_size > 0
    _check(
        checks,
        name="manuscript_pdf",
        passed=(pdf_passed if require_pdf else True),
        detail=(
            "Manuscript PDF exists."
            if pdf_passed
            else "Manuscript PDF missing or empty."
        ),
        path=pdf_path,
    )

    failures = [check for check in checks if not check["passed"]]
    return {
        "ready": not failures,
        "checks": checks,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check reviewer-remediation submission artifact readiness.")
    parser.add_argument("--repo-root", default=str(ROOT), help="Repository root containing outputs/ and waste2energy-paper/.")
    parser.add_argument("--require-bootstrap", action="store_true", help="Require non-empty bootstrap benchmark statistics.")
    parser.add_argument("--min-bootstrap-replicates", type=int, default=1)
    parser.add_argument("--require-targeted-ablations", action="store_true", help="Require targeted ablations and Monte Carlo UQ outputs.")
    parser.add_argument("--min-monte-carlo-replicates", type=int, default=1)
    parser.add_argument("--require-pdf", action="store_true", help="Require waste2energy-paper/main.pdf.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_readiness_report(
        repo_root=Path(args.repo_root),
        require_bootstrap=args.require_bootstrap,
        min_bootstrap_replicates=args.min_bootstrap_replicates,
        require_targeted_ablations=args.require_targeted_ablations,
        min_monte_carlo_replicates=args.min_monte_carlo_replicates,
        require_pdf=args.require_pdf,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
