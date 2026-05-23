# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

"""Reviewer-facing quality gates for final Waste2Energy result folders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PASS = "pass"
WARN = "warning"
FAIL = "fail"

CORE_BENCHMARK_COMPARATORS = (
    "classic_multiobjective_optimizer",
    "no_robustness_penalty",
    "uncertainty_penalty_max_interval",
    "uncertainty_penalty_combined_only",
    "no_carbon_constraint",
)

DIAGNOSTIC_BENCHMARK_COMPARATORS = (
    "ranking_only_unconstrained",
    "greedy_weighted_score_heuristic",
    "mcda_weighted_sum_comparator",
    "topsis_comparator",
)

MAJOR_BENCHMARK_COMPARATORS = CORE_BENCHMARK_COMPARATORS + DIAGNOSTIC_BENCHMARK_COMPARATORS

REQUIRED_SCENARIO_UNCERTAINTY_COLUMNS = (
    "baseline_top_pathway",
    "combined_only_top_pathway",
    "max_interval_top_pathway",
)

MIN_REVIEWER_BOOTSTRAP_REPLICATES = 16

ALLOWED_STRONG_CLAIM_EFFECT_TIERS = (
    "highly_consistent",
    "directionally_consistent",
    "suggestive",
)


@dataclass(frozen=True)
class QualityGate:
    gate: str
    status: str
    detail: str
    evidence_path: str = ""


def build_results_quality_report(
    *,
    planning_dir: str | Path,
    benchmark_dir: str | Path | None = None,
    scenario_dir: str | Path | None = None,
    operation_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build conservative SCI-readiness checks from generated artifacts."""

    gates: list[QualityGate] = []
    planning_path = Path(planning_dir)
    gates.extend(_planning_gates(planning_path))
    if benchmark_dir is not None:
        gates.extend(_benchmark_gates(Path(benchmark_dir)))
    if scenario_dir is not None:
        gates.extend(_scenario_gates(Path(scenario_dir)))
    if operation_dir is not None:
        gates.extend(_operation_gates(Path(operation_dir)))

    _, claim_summary = build_validated_pyrolysis_superiority_claim(
        planning_dir=planning_path,
        benchmark_dir=benchmark_dir,
        scenario_dir=scenario_dir,
    )
    gates.append(
        QualityGate(
            "validated_pyrolysis_superiority_claim",
            str(claim_summary["claim_status"]),
            str(claim_summary["claim_text"]),
            str(planning_path),
        )
    )

    frame = pd.DataFrame([gate.__dict__ for gate in gates])
    status_order = {FAIL: 2, WARN: 1, PASS: 0}
    worst_status = max(frame["status"], key=lambda status: status_order.get(status, 2)) if not frame.empty else FAIL
    summary = {
        "overall_status": worst_status,
        "gate_count": int(len(frame)),
        "pass_count": int(frame["status"].eq(PASS).sum()) if not frame.empty else 0,
        "warning_count": int(frame["status"].eq(WARN).sum()) if not frame.empty else 0,
        "fail_count": int(frame["status"].eq(FAIL).sum()) if not frame.empty else 0,
        "interpretation": _interpret_status(worst_status),
        "strong_claim_status": claim_summary["claim_status"],
        "strong_claim_text": claim_summary["claim_text"],
        "strong_claim_fail_count": claim_summary["fail_count"],
    }
    return frame, summary


def write_results_quality_report(
    *,
    output_dir: str | Path,
    planning_dir: str | Path,
    benchmark_dir: str | Path | None = None,
    scenario_dir: str | Path | None = None,
    operation_dir: str | Path | None = None,
) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    frame, summary = build_results_quality_report(
        planning_dir=planning_dir,
        benchmark_dir=benchmark_dir,
        scenario_dir=scenario_dir,
        operation_dir=operation_dir,
    )
    gate_path = target / "quality_gate_summary.csv"
    summary_path = target / "quality_gate_summary.json"
    claim_path = target / "validated_pyrolysis_superiority_claim.csv"
    claim_json_path = target / "validated_pyrolysis_superiority_claim.json"
    claim_frame, claim_summary = build_validated_pyrolysis_superiority_claim(
        planning_dir=planning_dir,
        benchmark_dir=benchmark_dir,
        scenario_dir=scenario_dir,
    )
    frame.to_csv(gate_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    claim_frame.to_csv(claim_path, index=False)
    claim_json_path.write_text(json.dumps(claim_summary, indent=2), encoding="utf-8")
    return {
        "quality_gate_summary": str(gate_path),
        "quality_gate_summary_json": str(summary_path),
        "validated_pyrolysis_superiority_claim": str(claim_path),
        "validated_pyrolysis_superiority_claim_json": str(claim_json_path),
    }


def build_validated_pyrolysis_superiority_claim(
    *,
    planning_dir: str | Path,
    benchmark_dir: str | Path | None = None,
    scenario_dir: str | Path | None = None,
    selected_share_threshold: float = 0.80,
    surrogate_support_threshold: float = 0.75,
    stability_threshold: float = 0.70,
    minimum_benchmark_bootstrap_replicates: int = MIN_REVIEWER_BOOTSTRAP_REPLICATES,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate the strong pyrolysis-superiority claim against generated outputs."""

    planning_path = Path(planning_dir)
    rows: list[dict[str, object]] = []
    expected_scenarios = _expected_scenarios_from_planning(planning_path)
    rows.append(
        _claim_row(
            "planning_claim_scenario_scope_available",
            PASS if expected_scenarios else FAIL,
            (
                f"Expected planning scenarios: {sorted(expected_scenarios)}."
                if expected_scenarios
                else "surrogate_transferability_summary.csv lacks a nonempty scenario_name scope."
            ),
            planning_path / "surrogate_transferability_summary.csv",
        )
    )
    rows.extend(
        _claim_planning_rows(
            planning_path,
            selected_share_threshold=selected_share_threshold,
            surrogate_support_threshold=surrogate_support_threshold,
        )
    )
    if benchmark_dir is None:
        rows.append(
            _claim_row(
                "benchmark_evidence_available",
                FAIL,
                "Strong pyrolysis-superiority wording requires a benchmark directory; no benchmark_dir was provided.",
                planning_path,
            )
        )
    else:
        rows.extend(
            _claim_benchmark_rows(
                Path(benchmark_dir),
                expected_scenarios=expected_scenarios,
                minimum_bootstrap_replicates=minimum_benchmark_bootstrap_replicates,
            )
        )
    if scenario_dir is None:
        rows.append(
            _claim_row(
                "scenario_stress_evidence_available",
                FAIL,
                "Strong pyrolysis-superiority wording requires a scenario robustness directory; no scenario_dir was provided.",
                planning_path,
            )
        )
    else:
        rows.extend(
            _claim_scenario_rows(
                Path(scenario_dir),
                expected_scenarios=expected_scenarios,
                stability_threshold=stability_threshold,
            )
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["claim_gate", "status", "detail", "evidence_path"])
    pass_count = int(frame["status"].eq(PASS).sum()) if "status" in frame else 0
    fail_count = int(frame["status"].eq(FAIL).sum()) if "status" in frame else 0
    warning_count = int(frame["status"].eq(WARN).sum()) if "status" in frame else 0
    claim_status = PASS if fail_count == 0 else FAIL
    claim_text = (
        "Stress-tested evidence supports pyrolysis as the leading pathway within the configured evidence-aware "
        "screening boundary."
        if claim_status == PASS
        else "The strong pyrolysis-superiority claim is not validated by the current generated artifacts."
    )
    return frame, {
        "claim_status": claim_status,
        "claim_text": claim_text,
        "pass_count": pass_count,
        "warning_count": warning_count,
        "fail_count": fail_count,
        "selected_share_threshold": selected_share_threshold,
        "surrogate_support_threshold": surrogate_support_threshold,
        "stability_threshold": stability_threshold,
        "minimum_benchmark_bootstrap_replicates": int(minimum_benchmark_bootstrap_replicates),
    }


def _planning_gates(planning_dir: Path) -> list[QualityGate]:
    gates = [
        _exists_gate("planning_run_config_exists", planning_dir / "run_config.json"),
        _exists_gate("planning_reproducibility_manifest_exists", planning_dir / "reproducibility_manifest.json"),
        _exists_gate("planning_portfolio_allocations_exist", planning_dir / "portfolio_allocations.csv"),
        _exists_gate("planning_data_contract_exists", planning_dir / "planning_data_contract_summary.csv"),
        _exists_gate("surrogate_transferability_summary_exists", planning_dir / "surrogate_transferability_summary.csv"),
    ]
    gates.append(_portfolio_gate(planning_dir / "portfolio_allocations.csv"))
    gates.append(_data_contract_gate(planning_dir / "planning_data_contract_summary.csv"))
    gates.append(_data_contract_warning_gate(planning_dir / "planning_data_contract_warnings.csv"))
    gates.append(_transferability_gate(planning_dir / "surrogate_transferability_summary.csv"))
    gates.append(_manifest_hash_gate(planning_dir / "reproducibility_manifest.json"))
    return gates


def _benchmark_gates(benchmark_dir: Path) -> list[QualityGate]:
    gates = [
        _exists_gate("benchmark_summary_exists", benchmark_dir / "benchmark_summary.csv"),
        _exists_gate("benchmark_shift_summary_exists", benchmark_dir / "benchmark_shift_summary.csv"),
        _exists_gate("benchmark_statistical_summary_exists", benchmark_dir / "benchmark_statistical_summary.csv"),
    ]
    stats = _read_csv(benchmark_dir / "benchmark_statistical_summary.csv")
    if stats.empty:
        gates.append(QualityGate("benchmark_uncertainty_intervals", WARN, "No bootstrap statistical summary rows.", str(benchmark_dir)))
    elif "effect_significance_tier" in stats.columns:
        replicate_count = _minimum_numeric_value(stats, "bootstrap_replicate_count")
        replicate_status = (
            PASS
            if replicate_count is not None and replicate_count >= MIN_REVIEWER_BOOTSTRAP_REPLICATES
            else WARN
        )
        gates.append(
            QualityGate(
                "benchmark_uncertainty_intervals",
                replicate_status,
                (
                    f"Bootstrap summary contains {len(stats)} rows and effect tiers; "
                    f"minimum replicate count is {replicate_count if replicate_count is not None else 'unknown'}."
                ),
                str(benchmark_dir / "benchmark_statistical_summary.csv"),
            )
        )
    else:
        gates.append(
            QualityGate(
                "benchmark_uncertainty_intervals",
                WARN,
                "Benchmark summary exists but lacks effect_significance_tier.",
                str(benchmark_dir / "benchmark_statistical_summary.csv"),
            )
        )
    return gates


def _scenario_gates(scenario_dir: Path) -> list[QualityGate]:
    gates = [
        _exists_gate("scenario_stress_summary_exists", scenario_dir / "stress_test_summary.csv"),
        _exists_gate("scenario_decision_stability_exists", scenario_dir / "decision_stability.csv"),
    ]
    stability = _read_csv(scenario_dir / "decision_stability.csv")
    if stability.empty:
        gates.append(QualityGate("scenario_decision_stability_nonempty", WARN, "No decision stability rows.", str(scenario_dir)))
    else:
        gates.append(
            QualityGate(
                "scenario_decision_stability_nonempty",
                PASS,
                f"Decision stability contains {len(stability)} rows.",
                str(scenario_dir / "decision_stability.csv"),
            )
        )
    return gates


def _operation_gates(operation_dir: Path) -> list[QualityGate]:
    summary_path = operation_dir / "rollout_summary.csv"
    if not summary_path.exists():
        summary_path = operation_dir / "baseline_rollout_summary.csv"
    if not summary_path.exists():
        summary_path = operation_dir / "baseline" / "rollout_summary.csv"
    gates = [_exists_gate("operation_baseline_rollout_summary_exists", summary_path)]
    summary = _read_csv(summary_path)
    gates.append(
        QualityGate(
            "operation_appendix_nonempty",
            PASS if not summary.empty else WARN,
            f"Operation appendix rollout rows: {len(summary)}.",
            str(summary_path),
        )
    )
    return gates


def _claim_planning_rows(
    planning_dir: Path,
    *,
    selected_share_threshold: float,
    surrogate_support_threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    transferability_path = planning_dir / "surrogate_transferability_summary.csv"
    transferability = _read_csv(transferability_path)
    if transferability.empty:
        return [
            _claim_row(
                "predictive_transferability_available",
                FAIL,
                "surrogate_transferability_summary.csv is missing or empty.",
                transferability_path,
            )
        ]
    required_columns = {"pathway", "selected_allocated_feed_share", "optimization_supported_prediction_fraction"}
    missing_columns = sorted(required_columns.difference(transferability.columns))
    if missing_columns:
        return [
            _claim_row(
                "predictive_transferability_available",
                FAIL,
                f"surrogate_transferability_summary.csv lacks required columns: {missing_columns}.",
                transferability_path,
            )
        ]
    expected_scenarios = _scenario_values(transferability)
    selected_share = pd.to_numeric(
        transferability.get(
            "selected_allocated_feed_share",
            pd.Series([0.0] * len(transferability), index=transferability.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    selected = transferability[selected_share.gt(0)].copy()
    selected_pathway = selected["pathway"].astype(str).str.strip().str.lower()
    pyrolysis_selected = selected[selected_pathway.eq("pyrolysis")].copy()
    non_pyrolysis_selected = selected[~selected_pathway.eq("pyrolysis")].copy()
    selected_scenarios = _scenario_values(selected)
    pyrolysis_selected_scenarios = _scenario_values(pyrolysis_selected)
    missing_selected_scenarios = sorted(expected_scenarios.difference(selected_scenarios))
    missing_pyrolysis_scenarios = sorted(expected_scenarios.difference(pyrolysis_selected_scenarios))
    if non_pyrolysis_selected.empty and not pyrolysis_selected.empty and not missing_pyrolysis_scenarios:
        min_share = float(pd.to_numeric(pyrolysis_selected["selected_allocated_feed_share"], errors="coerce").min())
        rows.append(
            _claim_row(
                "pyrolysis_selected_in_all_planning_scenarios",
                PASS if min_share >= selected_share_threshold and not missing_selected_scenarios else FAIL,
                (
                    f"Minimum selected pyrolysis share is {min_share:.3f}; "
                    f"missing selected scenarios: {missing_selected_scenarios}; "
                    f"missing pyrolysis scenarios: {missing_pyrolysis_scenarios}."
                ),
                transferability_path,
            )
        )
    else:
        rows.append(
            _claim_row(
                "pyrolysis_selected_in_all_planning_scenarios",
                FAIL,
                (
                    "At least one selected scenario-pathway is not pyrolysis, pyrolysis is absent, "
                    f"or planning scenarios are uncovered; missing selected scenarios: {missing_selected_scenarios}; "
                    f"missing pyrolysis scenarios: {missing_pyrolysis_scenarios}."
                ),
                transferability_path,
            )
        )

    if not pyrolysis_selected.empty:
        support_fraction = pd.to_numeric(
            pyrolysis_selected.get("optimization_supported_prediction_fraction", 0),
            errors="coerce",
        ).fillna(0.0)
        worst_gates = set(pyrolysis_selected.get("worst_surrogate_evidence_gate", pd.Series(dtype="object")).astype(str))
        support_ok = bool(support_fraction.ge(surrogate_support_threshold).all())
        gate_ok = worst_gates == {"conditional_transfer"}
        rows.append(
            _claim_row(
                "selected_pyrolysis_predictions_are_transferable",
                PASS if support_ok and gate_ok else FAIL,
                (
                    f"Minimum optimization-supported target fraction is {float(support_fraction.min()):.3f}; "
                    f"worst gates: {sorted(worst_gates)}."
                ),
                transferability_path,
            )
        )
    return rows


def _claim_benchmark_rows(
    benchmark_dir: Path,
    *,
    expected_scenarios: set[str],
    minimum_bootstrap_replicates: int,
) -> list[dict[str, object]]:
    summary_path = benchmark_dir / "benchmark_summary.csv"
    summary = _read_csv(summary_path)
    if summary.empty:
        return [_claim_row("benchmark_comparators_available", FAIL, "Benchmark summary is missing.", summary_path)]
    baseline = summary[summary["benchmark_variant"].astype(str).eq("baseline_evidence_aware")]
    missing_baseline_scenarios = _missing_variant_scenario_pairs(
        summary,
        expected_scenarios=expected_scenarios,
        variants=("baseline_evidence_aware",),
    )
    expected_baseline = _filter_expected_scenarios(baseline, expected_scenarios)
    baseline_ok = (
        not missing_baseline_scenarios
        and not expected_baseline.empty
        and expected_baseline.get("selected_pathways", pd.Series(dtype="object")).astype(str).eq("pyrolysis").all()
    )
    rows = [
        _claim_row(
            "benchmark_baseline_selects_pyrolysis",
            PASS if baseline_ok else FAIL,
            (
                f"Baseline benchmark rows in expected scenarios: {len(expected_baseline)}; "
                f"missing scenario pairs: {_format_pairs(missing_baseline_scenarios)}."
            ),
            summary_path,
        )
    ]
    available_variants = set(summary["benchmark_variant"].astype(str))
    missing_core_comparators = sorted(set(CORE_BENCHMARK_COMPARATORS).difference(available_variants))
    core_mask = summary["benchmark_variant"].astype(str).isin(CORE_BENCHMARK_COMPARATORS)
    core_comparators = summary[core_mask]
    expected_core_comparators = _filter_expected_scenarios(core_comparators, expected_scenarios)
    missing_core_pairs = _missing_variant_scenario_pairs(
        summary,
        expected_scenarios=expected_scenarios,
        variants=CORE_BENCHMARK_COMPARATORS,
    )
    core_comparator_ok = (
        not missing_core_comparators
        and not missing_core_pairs
        and not expected_core_comparators.empty
        and expected_core_comparators.get("selected_pathways", pd.Series(dtype="object")).astype(str).eq("pyrolysis").all()
    )
    rows.append(
        _claim_row(
            "benchmark_core_comparators_select_pyrolysis",
            PASS if core_comparator_ok else FAIL,
            (
                f"Checked {len(expected_core_comparators)} expected core comparator-scenario rows; "
                f"missing core comparator names: {missing_core_comparators}; "
                f"missing scenario pairs: {_format_pairs(missing_core_pairs)}."
            ),
            summary_path,
        )
    )
    diagnostic_mask = summary["benchmark_variant"].astype(str).isin(DIAGNOSTIC_BENCHMARK_COMPARATORS)
    diagnostic_comparators = summary[diagnostic_mask]
    expected_diagnostic_comparators = _filter_expected_scenarios(diagnostic_comparators, expected_scenarios)
    missing_diagnostic_comparators = sorted(set(DIAGNOSTIC_BENCHMARK_COMPARATORS).difference(available_variants))
    missing_diagnostic_pairs = _missing_variant_scenario_pairs(
        summary,
        expected_scenarios=expected_scenarios,
        variants=DIAGNOSTIC_BENCHMARK_COMPARATORS,
    )
    diagnostic_selects_pyrolysis = (
        not expected_diagnostic_comparators.empty
        and expected_diagnostic_comparators.get("selected_pathways", pd.Series(dtype="object")).astype(str).eq("pyrolysis").all()
    )
    diagnostic_ok = (
        not missing_diagnostic_comparators
        and not missing_diagnostic_pairs
        and diagnostic_selects_pyrolysis
    )
    rows.append(
        _claim_row(
            "benchmark_diagnostic_comparators_contextual",
            PASS if diagnostic_ok else WARN,
            (
                f"Checked {len(expected_diagnostic_comparators)} expected diagnostic comparator-scenario rows; "
                f"missing diagnostic comparator names: {missing_diagnostic_comparators}; "
                f"missing scenario pairs: {_format_pairs(missing_diagnostic_pairs)}; "
                f"all diagnostic comparators select pyrolysis: {diagnostic_selects_pyrolysis}."
            ),
            summary_path,
        )
    )
    stats_path = benchmark_dir / "benchmark_statistical_summary.csv"
    stats = _read_csv(stats_path)
    missing_core_statistical_pairs = _missing_variant_scenario_pairs(
        stats,
        expected_scenarios=expected_scenarios,
        variants=CORE_BENCHMARK_COMPARATORS,
    )
    if "benchmark_variant" in stats.columns:
        expected_core_stats = _filter_expected_scenarios(
            stats[
                stats["benchmark_variant"]
                .astype(str)
                .str.strip()
                .isin(CORE_BENCHMARK_COMPARATORS)
            ],
            expected_scenarios,
        )
        expected_diagnostic_stats = _filter_expected_scenarios(
            stats[
                stats["benchmark_variant"]
                .astype(str)
                .str.strip()
                .isin(DIAGNOSTIC_BENCHMARK_COMPARATORS)
            ],
            expected_scenarios,
        )
    else:
        expected_core_stats = stats.iloc[0:0].copy()
        expected_diagnostic_stats = stats.iloc[0:0].copy()
    core_statistical_coverage_ok = bool(expected_scenarios and not missing_core_statistical_pairs)
    rows.append(
        _claim_row(
            "benchmark_core_statistical_scenario_comparator_coverage_complete",
            PASS if core_statistical_coverage_ok else FAIL,
            (
                "Benchmark statistical summary must cover every expected core scenario-comparator pair; "
                f"missing pairs: {_format_pairs(missing_core_statistical_pairs)}."
            ),
            stats_path,
        )
    )
    missing_diagnostic_statistical_pairs = _missing_variant_scenario_pairs(
        stats,
        expected_scenarios=expected_scenarios,
        variants=DIAGNOSTIC_BENCHMARK_COMPARATORS,
    )
    diagnostic_statistical_coverage_ok = bool(expected_scenarios and not missing_diagnostic_statistical_pairs)
    rows.append(
        _claim_row(
            "benchmark_diagnostic_statistical_coverage_contextual",
            PASS if diagnostic_statistical_coverage_ok else WARN,
            (
                "Diagnostic benchmark statistical rows are contextual rather than blocking; "
                f"missing pairs: {_format_pairs(missing_diagnostic_statistical_pairs)}."
            ),
            stats_path,
        )
    )
    has_effect_tier_column = "effect_significance_tier" in expected_core_stats.columns
    core_effect_tiers = (
        expected_core_stats["effect_significance_tier"].astype(str).str.strip().str.lower()
        if has_effect_tier_column
        else pd.Series(dtype="object")
    )
    missing_core_effect_tier_count = int(
        core_effect_tiers.eq("").sum() + core_effect_tiers.isin({"nan", "none", "<na>"}).sum()
    )
    unexpected_core_effect_tiers = sorted(
        tier
        for tier in set(core_effect_tiers)
        if tier and tier not in {"nan", "none", "<na>"} and tier not in ALLOWED_STRONG_CLAIM_EFFECT_TIERS
    )
    effect_tier_ok = bool(
        core_statistical_coverage_ok
        and not expected_core_stats.empty
        and has_effect_tier_column
        and missing_core_effect_tier_count == 0
    )
    rows.append(
        _claim_row(
            "benchmark_core_statistical_effect_tiers_available",
            PASS if effect_tier_ok else FAIL,
            (
                "Expected core benchmark statistical rows must include nonempty effect_significance_tier values; "
                f"core statistical coverage complete: {core_statistical_coverage_ok}."
            ),
            stats_path,
        )
    )
    effect_tier_stability_ok = bool(effect_tier_ok and not unexpected_core_effect_tiers)
    rows.append(
        _claim_row(
            "benchmark_core_statistical_effect_tiers_not_unstable",
            PASS if effect_tier_stability_ok else FAIL,
            (
                "Strong claim permits only stable/suggestive core benchmark statistical tiers "
                f"{list(ALLOWED_STRONG_CLAIM_EFFECT_TIERS)}; observed tier counts: "
                f"{core_effect_tiers.value_counts().to_dict()}; unexpected tiers: {unexpected_core_effect_tiers}."
            ),
            stats_path,
        )
    )
    diagnostic_effect_tiers = (
        expected_diagnostic_stats["effect_significance_tier"].astype(str).str.strip().str.lower()
        if "effect_significance_tier" in expected_diagnostic_stats.columns
        else pd.Series(dtype="object")
    )
    unexpected_diagnostic_effect_tiers = sorted(
        tier
        for tier in set(diagnostic_effect_tiers)
        if tier and tier not in {"nan", "none", "<na>"} and tier not in ALLOWED_STRONG_CLAIM_EFFECT_TIERS
    )
    rows.append(
        _claim_row(
            "benchmark_diagnostic_effect_tiers_contextual",
            PASS if not unexpected_diagnostic_effect_tiers else WARN,
            (
                "Diagnostic comparator effect tiers are reported as sensitivity context, not blocking evidence; "
                f"observed tier counts: {diagnostic_effect_tiers.value_counts().to_dict()}; "
                f"contextual unstable tiers: {unexpected_diagnostic_effect_tiers}."
            ),
            stats_path,
        )
    )
    replicate_count = _minimum_numeric_value(expected_core_stats, "bootstrap_replicate_count")
    replicate_ok = (
        core_statistical_coverage_ok
        and effect_tier_ok
        and replicate_count is not None
        and replicate_count >= minimum_bootstrap_replicates
    )
    rows.append(
        _claim_row(
            "benchmark_bootstrap_replicates_sufficient",
            PASS if replicate_ok else FAIL,
            (
                f"Minimum bootstrap replicate count is "
                f"{replicate_count if replicate_count is not None else 'unknown'}; "
                f"required >= {minimum_bootstrap_replicates}; "
                f"core statistical coverage complete: {core_statistical_coverage_ok}."
            ),
            stats_path,
        )
    )
    return rows


def _claim_scenario_rows(
    scenario_dir: Path,
    *,
    expected_scenarios: set[str],
    stability_threshold: float,
) -> list[dict[str, object]]:
    uncertainty_path = scenario_dir / "uncertainty_summary.csv"
    uncertainty = _read_csv(uncertainty_path)
    if uncertainty.empty:
        return [_claim_row("scenario_uncertainty_stability_available", FAIL, "Uncertainty summary is missing.", uncertainty_path)]
    observed_scenarios = _scenario_values(uncertainty)
    missing_scenarios = sorted(expected_scenarios.difference(observed_scenarios))
    rows = [
        _claim_row(
            "scenario_uncertainty_scenario_coverage_complete",
            PASS if expected_scenarios and not missing_scenarios else FAIL,
            (
                f"Expected scenarios: {sorted(expected_scenarios)}; observed scenarios: {sorted(observed_scenarios)}; "
                f"missing scenarios: {missing_scenarios}."
            ),
            uncertainty_path,
        )
    ]
    available_pathway_columns = [
        column for column in REQUIRED_SCENARIO_UNCERTAINTY_COLUMNS if column in uncertainty.columns
    ]
    missing_pathway_columns = sorted(set(REQUIRED_SCENARIO_UNCERTAINTY_COLUMNS).difference(available_pathway_columns))
    expected_uncertainty = _filter_expected_scenarios(uncertainty, expected_scenarios)
    pathway_ok = bool(
        not missing_scenarios
        and not missing_pathway_columns
        and available_pathway_columns
        and not expected_uncertainty.empty
        and expected_uncertainty[available_pathway_columns].astype(str).eq("pyrolysis").all().all()
    )
    rows.append(
        _claim_row(
            "uncertainty_modes_keep_pyrolysis_top",
            PASS if pathway_ok else FAIL,
            (
                f"Checked pathway columns: {available_pathway_columns}; missing columns: {missing_pathway_columns}; "
                f"missing scenarios: {missing_scenarios}."
            ),
            uncertainty_path,
        )
    )
    if "dominant_selection_rate" in uncertainty.columns:
        min_stability = float(pd.to_numeric(expected_uncertainty["dominant_selection_rate"], errors="coerce").fillna(0).min())
        rows.append(
            _claim_row(
                "scenario_stress_selection_rate_sufficient",
                PASS if not expected_uncertainty.empty and min_stability >= stability_threshold else FAIL,
                f"Minimum dominant selection rate is {min_stability:.3f}.",
                uncertainty_path,
            )
        )
    return rows


def _expected_scenarios_from_planning(planning_dir: Path) -> set[str]:
    transferability = _read_csv(planning_dir / "surrogate_transferability_summary.csv")
    return _scenario_values(transferability)


def _scenario_values(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "scenario_name" not in frame.columns:
        return set()
    values = frame["scenario_name"].astype(str).str.strip()
    return {
        value
        for value in values
        if value and value.lower() not in {"nan", "none", "<na>"}
    }


def _filter_expected_scenarios(frame: pd.DataFrame, expected_scenarios: set[str]) -> pd.DataFrame:
    if frame.empty or not expected_scenarios or "scenario_name" not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame[frame["scenario_name"].astype(str).str.strip().isin(expected_scenarios)].copy()


def _missing_variant_scenario_pairs(
    frame: pd.DataFrame,
    *,
    expected_scenarios: set[str],
    variants: tuple[str, ...],
) -> list[tuple[str, str]]:
    if not expected_scenarios:
        return [(scenario, variant) for scenario in [] for variant in variants]
    if frame.empty or not {"scenario_name", "benchmark_variant"}.issubset(frame.columns):
        return sorted((scenario, variant) for scenario in expected_scenarios for variant in variants)
    observed = set(
        zip(
            frame["scenario_name"].astype(str).str.strip(),
            frame["benchmark_variant"].astype(str).str.strip(),
        )
    )
    return sorted(
        (scenario, variant)
        for scenario in expected_scenarios
        for variant in variants
        if (scenario, variant) not in observed
    )


def _format_pairs(pairs: list[tuple[str, str]], *, limit: int = 8) -> str:
    if not pairs:
        return "[]"
    shown = [f"{scenario}::{variant}" for scenario, variant in pairs[:limit]]
    suffix = "" if len(pairs) <= limit else f"; +{len(pairs) - limit} more"
    return "[" + ", ".join(shown) + suffix + "]"


def _minimum_numeric_value(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.min())


def _claim_row(gate: str, status: str, detail: str, path: Path) -> dict[str, object]:
    return {
        "claim_gate": gate,
        "status": status,
        "detail": detail,
        "evidence_path": str(path),
    }


def _exists_gate(name: str, path: Path) -> QualityGate:
    return QualityGate(
        name,
        PASS if path.exists() else FAIL,
        "Artifact exists." if path.exists() else "Required artifact is missing.",
        str(path),
    )


def _portfolio_gate(path: Path) -> QualityGate:
    frame = _read_csv(path)
    if frame.empty:
        return QualityGate("planning_portfolio_nonempty", FAIL, "Portfolio allocation table is empty.", str(path))
    if "allocated_feed_ton_per_year" in frame.columns and pd.to_numeric(
        frame["allocated_feed_ton_per_year"], errors="coerce"
    ).gt(0).all():
        return QualityGate("planning_portfolio_nonempty", PASS, f"Portfolio has {len(frame)} positive allocation rows.", str(path))
    return QualityGate("planning_portfolio_nonempty", WARN, "Portfolio exists but some allocations are not positive.", str(path))


def _data_contract_gate(path: Path) -> QualityGate:
    frame = _read_csv(path)
    if frame.empty:
        return QualityGate("planning_data_contract_boundary", FAIL, "Data contract summary is empty.", str(path))
    row = frame[frame["summary_scope"].astype(str).eq("all_planning_rows")].head(1)
    if row.empty:
        return QualityGate("planning_data_contract_boundary", WARN, "No all_planning_rows contract row.", str(path))
    independent = int(pd.to_numeric(row.iloc[0].get("independent_observation_count"), errors="coerce") or 0)
    scenario_expanded = int(pd.to_numeric(row.iloc[0].get("scenario_expanded_count"), errors="coerce") or 0)
    if scenario_expanded > 0 and independent == 0:
        return QualityGate(
            "planning_data_contract_boundary",
            WARN,
            "Planning rows are scenario-expanded candidates, not independent observations; manuscript claims must say so.",
            str(path),
        )
    return QualityGate("planning_data_contract_boundary", PASS, "Planning data contract is explicit.", str(path))


def _data_contract_warning_gate(path: Path) -> QualityGate:
    frame = _read_csv(path)
    if frame.empty:
        return QualityGate("planning_data_contract_warnings", WARN, "No data contract warning table.", str(path))
    warnings = set(frame.get("warning", pd.Series(dtype="object")).astype(str))
    if "duplicate_optimization_case_id" in warnings:
        return QualityGate("planning_data_contract_warnings", FAIL, "Duplicate optimization_case_id detected.", str(path))
    return QualityGate("planning_data_contract_warnings", PASS, "No blocking data contract warnings.", str(path))


def _transferability_gate(path: Path) -> QualityGate:
    frame = _read_csv(path)
    if frame.empty:
        return QualityGate("surrogate_transferability_risk", WARN, "No transferability summary rows.", str(path))
    risk = frame.get("transferability_risk_label", pd.Series(dtype="object")).astype(str)
    selected_high_risk = risk.str.contains("non_transferable|fallback|partial|screening_only", regex=True).sum()
    if selected_high_risk:
        return QualityGate(
            "surrogate_transferability_risk",
            WARN,
            f"{int(selected_high_risk)} selected pathway-scenario rows require conservative surrogate wording.",
            str(path),
        )
    return QualityGate("surrogate_transferability_risk", PASS, "Selected shares are conditionally transferable or unselected.", str(path))


def _manifest_hash_gate(path: Path) -> QualityGate:
    if not path.exists():
        return QualityGate("reproducibility_hashes_present", FAIL, "Manifest is missing.", str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return QualityGate("reproducibility_hashes_present", FAIL, "Manifest JSON is invalid.", str(path))
    input_hashes = [item.get("sha256") for item in payload.get("inputs", []) if item.get("exists")]
    output_hashes = [item.get("sha256") for item in payload.get("outputs", []) if item.get("exists")]
    if input_hashes and output_hashes:
        return QualityGate("reproducibility_hashes_present", PASS, "Input and output hashes are recorded.", str(path))
    return QualityGate("reproducibility_hashes_present", WARN, "Manifest lacks at least one input or output hash.", str(path))


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _interpret_status(status: str) -> str:
    if status == PASS:
        return "Artifacts pass the configured SCI readiness gates."
    if status == WARN:
        return "Artifacts are usable only with conservative manuscript wording and explicit limitations."
    return "At least one blocking artifact or consistency gate failed."
