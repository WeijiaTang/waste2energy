# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUTS_ROOT, resolve_surrogate_outputs_dir
from .data import DATASET_KEYS, TARGET_COLUMNS
from .models import MODEL_KEYS


class InconsistencyWarning(UserWarning):
    pass


def _default_split_summary_paths() -> dict[str, Path]:
    surrogate_root = resolve_surrogate_outputs_dir()
    return {
        "recommended": surrogate_root / "traditional_ml_suite_summary.csv",
        "strict_group": surrogate_root / "traditional_ml_suite_summary_strict_group.csv",
        "leave_study_out": surrogate_root / "traditional_ml_suite_summary_leave_study_out.csv",
    }


def _default_selected_manifest_paths() -> dict[str, Path]:
    surrogate_root = resolve_surrogate_outputs_dir()
    return {
        "recommended": surrogate_root / "selected_models_manifest.csv",
        "strict_group": surrogate_root / "selected_models_manifest_strict_group.csv",
        "leave_study_out": surrogate_root / "selected_models_manifest_leave_study_out.csv",
    }


DEFAULT_OPERATION_COMPARISON_DIR = OUTPUTS_ROOT / "operation" / "comparison"
DEFAULT_PLANNING_DIR = OUTPUTS_ROOT / "planning" / "baseline"
DEFAULT_SCENARIO_DIR = OUTPUTS_ROOT / "scenarios" / "baseline"


def build_confirmatory_audit(
    *,
    outputs_root: str | Path | None = None,
    planning_dir: str | Path | None = None,
    scenario_dir: str | Path | None = None,
    operation_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame | dict[str, object] | list[dict[str, object]]]:
    active_outputs_root = Path(outputs_root) if outputs_root else OUTPUTS_ROOT
    default_paths = _default_split_summary_paths()
    default_selected_paths = _default_selected_manifest_paths()
    ml_paths = {
        key: (active_outputs_root / path.relative_to(OUTPUTS_ROOT))
        for key, path in default_paths.items()
    }
    selected_manifest_paths = {
        key: (active_outputs_root / path.relative_to(OUTPUTS_ROOT))
        for key, path in default_selected_paths.items()
    }
    active_planning_dir = (
        Path(planning_dir)
        if planning_dir
        else active_outputs_root / DEFAULT_PLANNING_DIR.relative_to(OUTPUTS_ROOT)
    )
    active_scenario_dir = (
        Path(scenario_dir)
        if scenario_dir
        else active_outputs_root / DEFAULT_SCENARIO_DIR.relative_to(OUTPUTS_ROOT)
    )
    active_operation_dir = (
        Path(operation_dir)
        if operation_dir
        else active_outputs_root / DEFAULT_OPERATION_COMPARISON_DIR.relative_to(OUTPUTS_ROOT)
    )

    ml_summary = build_ml_split_coverage_summary(ml_paths)
    ml_best = build_ml_best_result_summary(ml_paths, selected_manifest_paths)
    ml_flags = build_ml_claim_flag_table(ml_paths, selected_manifest_paths)
    pathway_reliability = build_pathway_reliability_summary(ml_flags)
    planning_flags = build_planning_claim_flag_table(active_planning_dir, active_scenario_dir)
    planning_ml_consistency = build_planning_ml_consistency_summary(active_planning_dir, pathway_reliability)
    _emit_surrogate_led_inconsistency_warnings(planning_ml_consistency)
    planning_data_quality = _read_csv_if_exists(active_planning_dir / "planning_data_quality_summary.csv")
    operation_table = build_operation_comparison_summary(active_operation_dir)
    operation_flags = build_operation_claim_flag_table(active_operation_dir)
    artifact_inventory = build_artifact_inventory(
        ml_paths,
        active_operation_dir,
        active_planning_dir,
        active_scenario_dir,
    )
    audit_manifest = build_audit_manifest(
        ml_paths,
        active_operation_dir,
        active_planning_dir,
        active_scenario_dir,
    )

    return {
        "ml_split_coverage_summary": ml_summary,
        "ml_best_result_summary": ml_best,
        "ml_claim_flag_table": ml_flags,
        "pathway_reliability_summary": pathway_reliability,
        "planning_claim_flag_table": planning_flags,
        "planning_ml_consistency_summary": planning_ml_consistency,
        "planning_data_quality_summary": planning_data_quality,
        "operation_comparison_summary": operation_table,
        "operation_claim_flag_table": operation_flags,
        "artifact_inventory": artifact_inventory,
        "audit_manifest": audit_manifest,
    }


def write_confirmatory_audit(
    audit_payload: dict[str, pd.DataFrame | dict[str, object] | list[dict[str, object]]],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    base_dir = Path(output_dir) if output_dir else OUTPUTS_ROOT / "audit"
    base_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    for key, value in audit_payload.items():
        if isinstance(value, pd.DataFrame):
            path = base_dir / f"{key}.csv"
            value.to_csv(path, index=False)
            outputs[key] = str(path)
        else:
            path = base_dir / f"{key}.json"
            path.write_text(json.dumps(value, indent=2), encoding="utf-8")
            outputs[key] = str(path)
    return outputs


def main() -> int:
    payload = build_confirmatory_audit()
    outputs = write_confirmatory_audit(payload)
    print(json.dumps({"outputs": outputs}, indent=2))
    return 0


def build_ml_split_coverage_summary(summary_paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expected_pairs = len(DATASET_KEYS) * len(TARGET_COLUMNS)
    for label, path in summary_paths.items():
        frame = _read_csv_if_exists(path)
        rows.append(
            {
                "summary_label": label,
                "path": str(path),
                "exists": path.exists(),
                "row_count": int(len(frame)),
                "dataset_count": int(frame["dataset_key"].nunique()) if not frame.empty else 0,
                "target_count": int(frame["target_column"].nunique()) if not frame.empty else 0,
                "model_count": int(frame["model_key"].nunique()) if not frame.empty else 0,
                "contains_gradient_boosting": bool(
                    not frame.empty and (frame["model_key"] == "gradient_boosting").any()
                ),
                "contains_split_strategy_column": bool(not frame.empty and "split_strategy" in frame.columns),
                "expected_dataset_target_pairs": expected_pairs,
            }
        )
    return pd.DataFrame(rows)


def build_ml_best_result_summary(
    summary_paths: dict[str, Path],
    selected_manifest_paths: dict[str, Path] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, path in summary_paths.items():
        frame = _read_csv_if_exists(path)
        if frame.empty or "test_r2" not in frame.columns:
            continue
        manifest = _read_selected_manifest(selected_manifest_paths, label)
        best = _selected_or_best_frame(frame, manifest)
        for _, row in best.iterrows():
            rows.append(
                {
                    "summary_label": label,
                    "dataset_key": row["dataset_key"],
                    "target_column": row["target_column"],
                    "best_model_key": row["selected_model_key"],
                    "selection_metric_name": row.get("selection_metric_name", "validation_r2"),
                    "selection_metric_value": row.get("selection_metric_value", pd.NA),
                    "best_test_r2": row["selected_test_r2"],
                    "best_test_rmse": row["selected_test_rmse"],
                    "best_test_mae": row["selected_test_mae"],
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["summary_label", "dataset_key", "target_column"]).reset_index(drop=True)


def build_ml_claim_flag_table(
    summary_paths: dict[str, Path],
    selected_manifest_paths: dict[str, Path] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strict_group = _read_csv_if_exists(summary_paths["strict_group"])
    leave_study_out = _read_csv_if_exists(summary_paths["leave_study_out"])

    rows.extend(
        _build_claim_rows_from_frame(
            strict_group,
            selected_manifest=_read_selected_manifest(selected_manifest_paths, "strict_group"),
            summary_label="strict_group",
            claim_rule="Paper 1 main-table benchmark evidence tier",
            positive_threshold=0.65,
        )
    )
    rows.extend(
        _build_claim_rows_from_frame(
            leave_study_out,
            selected_manifest=_read_selected_manifest(selected_manifest_paths, "leave_study_out"),
            summary_label="leave_study_out",
            claim_rule="Cross-study stress test across available study-labeled datasets",
            positive_threshold=0.50,
        )
    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["summary_label", "dataset_key", "target_column"]).reset_index(drop=True)


def build_operation_comparison_summary(operation_dir: Path) -> pd.DataFrame:
    comparison_path = operation_dir / "rl_vs_baseline_comparison.csv"
    frame = _read_csv_if_exists(comparison_path)
    if frame.empty:
        return frame
    columns = [
        "scenario_name",
        "method_type",
        "method_name",
        "seed_count",
        "reward_mean",
        "reward_std",
        "max_violation_mean",
        "violation_rate_mean",
        "resilience_index_mean",
        "reward_improvement_vs_hold_plan_abs",
        "reward_improvement_vs_hold_plan_pct",
        "violation_aware_score",
        "violation_aware_rank_within_scenario",
    ]
    available = [column for column in columns if column in frame.columns]
    return frame[available].sort_values(
        ["scenario_name", "violation_aware_rank_within_scenario", "method_type", "method_name"]
    ).reset_index(drop=True)


def build_operation_claim_flag_table(operation_dir: Path) -> pd.DataFrame:
    comparison = _read_csv_if_exists(operation_dir / "rl_vs_baseline_comparison.csv")
    behavior = _read_csv_if_exists(operation_dir / "policy_behavior_comparison.csv")
    rows: list[dict[str, object]] = []
    if comparison.empty:
        return pd.DataFrame(rows)

    for _, row in comparison.iterrows():
        method_name = row["method_name"]
        scenario_name = row["scenario_name"]
        behavior_row = pd.Series(dtype="object")
        if not behavior.empty:
            matched = behavior[
                (behavior["scenario_name"] == scenario_name) & (behavior["method_name"] == method_name)
            ]
            if not matched.empty:
                behavior_row = matched.iloc[0]

        claim_status = "supportive"
        notes = []
        hold_plan_reward = _optional_float(row.get("hold_plan_reward_mean"))
        reward_delta = _optional_float(row.get("reward_improvement_vs_hold_plan_abs"))
        if pd.isna(hold_plan_reward) or pd.isna(reward_delta):
            relative_reward_change = pd.NA
            reward_ratio = pd.NA
            notes.append("missing_hold_plan_reference")
            if row["method_type"] == "rl_agent":
                claim_status = "not_evaluated"
        else:
            reward_scale = abs(float(hold_plan_reward))
            relative_reward_change = (
                float(reward_delta) / reward_scale if reward_scale > 0.0 else pd.NA
            )
            reward_ratio = 1.0 + float(relative_reward_change) if pd.notna(relative_reward_change) else pd.NA
        if row["method_type"] == "rl_agent":
            if pd.notna(reward_ratio) and float(reward_ratio) < 0.90:
                claim_status = "unsupported"
                notes.append("reward_below_90pct_of_hold_plan")
            elif pd.notna(reward_ratio) and float(reward_ratio) < 0.98:
                claim_status = "weak"
                notes.append("reward_below_98pct_of_hold_plan")
        if (
            row["method_type"] == "rl_agent"
            and pd.notna(reward_delta)
            and abs(float(reward_delta)) < 1e-9
        ):
            claim_status = "conservative_match"
            notes.append("matches_hold_plan_reward")
        if float(row["max_violation_mean"]) > 0.0:
            notes.append("nonzero_violation")
            if claim_status == "supportive":
                claim_status = "violation_prone"
        if _optional_float(row.get("violation_rate_mean")) not in {pd.NA} and pd.notna(_optional_float(row.get("violation_rate_mean"))) and float(_optional_float(row.get("violation_rate_mean"))) > 0.05:
            notes.append("elevated_violation_rate")
            if claim_status == "supportive":
                claim_status = "violation_prone"
        if _optional_float(row.get("resilience_index_mean")) not in {pd.NA} and pd.notna(_optional_float(row.get("resilience_index_mean"))) and float(_optional_float(row.get("resilience_index_mean"))) < 0.85:
            notes.append("low_resilience_index")
            if claim_status == "supportive":
                claim_status = "unstable"
        if float(row["reward_std"]) > 1.0:
            notes.append("high_seed_variation")
            if claim_status == "supportive":
                claim_status = "unstable"
        rows.append(
            {
                "scenario_name": scenario_name,
                "method_name": method_name,
                "method_type": row["method_type"],
                "claim_status": claim_status,
                "reward_mean": row["reward_mean"],
                "reward_std": row["reward_std"],
                "max_violation_mean": row["max_violation_mean"],
                "violation_rate_mean": row.get("violation_rate_mean", pd.NA),
                "resilience_index_mean": row.get("resilience_index_mean", pd.NA),
                "reward_improvement_vs_hold_plan_pct": relative_reward_change,
                "reward_ratio_vs_hold_plan": reward_ratio,
                "violation_aware_rank_within_scenario": row["violation_aware_rank_within_scenario"],
                "throughput_nonzero_rate_mean": behavior_row.get("throughput_nonzero_rate_mean", pd.NA),
                "severity_nonzero_rate_mean": behavior_row.get("severity_nonzero_rate_mean", pd.NA),
                "notes": ";".join(notes),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["scenario_name", "method_type", "method_name"]).reset_index(drop=True)


def build_pathway_reliability_summary(ml_flags: pd.DataFrame) -> pd.DataFrame:
    if ml_flags.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    leave_study_out = ml_flags[ml_flags["summary_label"] == "leave_study_out"].copy()
    if leave_study_out.empty:
        return pd.DataFrame()
    leave_study_out["pathway"] = leave_study_out["dataset_key"].apply(_pathway_from_dataset_key)
    for pathway, pathway_frame in leave_study_out.groupby("pathway", dropna=False):
        supportive = int((pathway_frame["claim_status"] == "supportive").sum())
        weak = int((pathway_frame["claim_status"] == "weak").sum())
        unsupported = int((pathway_frame["claim_status"] == "unsupported").sum())
        total = max(int(len(pathway_frame)), 1)
        weak_or_unsupported_ratio = (weak + unsupported) / total
        reliability_score = (supportive + 0.5 * weak) / total
        if weak_or_unsupported_ratio > 0.50 and pathway == "htc":
            reviewer_sentence = "The findings for HTC are auxiliary and lack cross-study generalizability."
            reliability_tier = "auxiliary_only"
        elif reliability_score >= 0.60:
            reviewer_sentence = "Cross-study evidence remains pathway-specific and should be written with claim discipline."
            reliability_tier = "conditional_support"
        else:
            reviewer_sentence = "Cross-study evidence is limited and should not be generalized without qualification."
            reliability_tier = "limited_support"
        rows.append(
            {
                "pathway": pathway,
                "leave_study_out_supportive_count": supportive,
                "leave_study_out_weak_count": weak,
                "leave_study_out_unsupported_count": unsupported,
                "leave_study_out_total_count": total,
                "weak_or_unsupported_ratio": weak_or_unsupported_ratio,
                "reliability_score": reliability_score,
                "reliability_tier": reliability_tier,
                "reviewer_restriction_sentence": reviewer_sentence,
            }
        )
    return pd.DataFrame(rows).sort_values("pathway").reset_index(drop=True)


def build_planning_ml_consistency_summary(
    planning_dir: Path,
    pathway_reliability: pd.DataFrame,
) -> pd.DataFrame:
    pathway_summary = _read_csv_if_exists(planning_dir / "pathway_summary.csv")
    portfolio_allocations = _read_csv_if_exists(planning_dir / "portfolio_allocations.csv")
    if pathway_summary.empty or pathway_reliability.empty:
        return pd.DataFrame()
    reliability_columns = [
        column
        for column in ["pathway", "reliability_score", "reliability_tier"]
        if column in pathway_reliability.columns
    ]
    reliability_lookup = pathway_reliability[reliability_columns].drop_duplicates(subset=["pathway"])
    rows: list[dict[str, object]] = []
    for scenario_name, frame in pathway_summary.groupby("scenario_name", dropna=False):
        working = frame.copy().merge(reliability_lookup, on="pathway", how="left")
        share = pd.to_numeric(working["portfolio_allocated_feed_share"], errors="coerce")
        if share.isna().any():
            missing_pathways = ", ".join(
                working.loc[share.isna(), "pathway"].astype(str).drop_duplicates().tolist()
            )
            raise ValueError(
                "Planning/ML consistency summary encountered missing "
                f"'portfolio_allocated_feed_share' for scenario '{scenario_name}' and pathway(s): {missing_pathways}."
            )
        working["portfolio_allocated_feed_share"] = share
        working["reliability_score"] = pd.to_numeric(working.get("reliability_score"), errors="coerce")
        working["evidence_mapping_status"] = np.where(
            ~working["pathway"].isin(reliability_lookup["pathway"]),
            "missing",
            np.where(working["reliability_score"].isna(), "not_evaluated", "evaluated"),
        )
        working["evidence_support_status"] = np.select(
            [
                working["evidence_mapping_status"] == "missing",
                working["evidence_mapping_status"] == "not_evaluated",
                working["reliability_score"] < 0.34,
                working["reliability_score"] < 0.60,
            ],
            ["missing", "not_evaluated", "unsupported", "weak"],
            default="supportive",
        )
        evaluated = working.loc[working["evidence_mapping_status"] == "evaluated"].copy()
        if (
            len(evaluated) >= 2
            and evaluated["reliability_score"].nunique() > 1
            and evaluated["portfolio_allocated_feed_share"].nunique() > 1
        ):
            consistency_correlation = float(
                evaluated["portfolio_allocated_feed_share"].corr(evaluated["reliability_score"])
            )
        else:
            consistency_correlation = pd.NA
        weighted_evidence_support = float(
            (
                evaluated["portfolio_allocated_feed_share"] * evaluated["reliability_score"]
            ).sum()
        )
        unsupported_mass = float(
            working.loc[working["evidence_support_status"] == "unsupported", "portfolio_allocated_feed_share"].sum()
        )
        missing_mass = float(
            working.loc[working["evidence_support_status"] == "missing", "portfolio_allocated_feed_share"].sum()
        )
        not_evaluated_mass = float(
            working.loc[
                working["evidence_support_status"] == "not_evaluated",
                "portfolio_allocated_feed_share",
            ].sum()
        )
        weak_mass = float(
            working.loc[working["evidence_support_status"] == "weak", "portfolio_allocated_feed_share"].sum()
        )
        evaluated_mass = float(
            working.loc[working["evidence_mapping_status"] == "evaluated", "portfolio_allocated_feed_share"].sum()
        )
        if missing_mass > 0.25:
            risk_tier = "high_risk"
            risk_note = "High allocation mass is assigned to pathways without any pathway-level reliability mapping."
        elif not_evaluated_mass > 0.25:
            risk_tier = "high_risk"
            risk_note = "High allocation mass is assigned to pathways lacking usable cross-study reliability scores."
        elif unsupported_mass > 0.25:
            risk_tier = "high_risk"
            risk_note = "High allocation mass is assigned to pathways with weak or unsupported cross-study evidence."
        elif pd.notna(consistency_correlation) and float(consistency_correlation) < 0.0:
            risk_tier = "medium_risk"
            risk_note = "Planning allocation is inversely aligned with pathway-level ML reliability."
        elif evaluated_mass <= 0.0:
            risk_tier = "data_gap"
            risk_note = "No pathway-level reliability scores were available to evaluate planning alignment."
        else:
            risk_tier = "managed_risk"
            risk_note = "Planning allocation remains broadly aligned with available pathway-level ML evidence."
        surrogate_supported_share = pd.NA
        surrogate_led_consistency = "not_evaluated"
        if not portfolio_allocations.empty and "scenario_name" in portfolio_allocations.columns:
            scenario_allocations = portfolio_allocations[
                portfolio_allocations["scenario_name"].astype(str) == str(scenario_name)
            ].copy()
            if not scenario_allocations.empty:
                allocated_feed = pd.to_numeric(
                    scenario_allocations.get("allocated_feed_ton_per_year"),
                    errors="coerce",
                ).fillna(0.0)
                total_feed = float(allocated_feed.sum())
                if total_feed > 0.0:
                    support_mask = scenario_allocations.get(
                        "surrogate_support_level",
                        pd.Series([""] * len(scenario_allocations), index=scenario_allocations.index),
                    ).astype(str).eq("surrogate_supported")
                    surrogate_supported_share = float(allocated_feed.loc[support_mask].sum() / total_feed)
                    surrogate_led_consistency = (
                        "consistent" if surrogate_supported_share >= 0.80 else "inconsistent"
                    )
        rows.append(
            {
                "scenario_name": scenario_name,
                "planning_ml_consistency_correlation": consistency_correlation,
                "weighted_evidence_support": weighted_evidence_support,
                "evaluated_allocation_share": evaluated_mass,
                "missing_reliability_allocation_share": missing_mass,
                "not_evaluated_allocation_share": not_evaluated_mass,
                "unsupported_allocation_share": unsupported_mass,
                "weak_allocation_share": weak_mass,
                "surrogate_supported_allocation_share": surrogate_supported_share,
                "surrogate_led_consistency": surrogate_led_consistency,
                "risk_tier": risk_tier,
                "risk_note": risk_note,
            }
        )
    return pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)


def build_planning_claim_flag_table(planning_dir: Path, scenario_dir: Path) -> pd.DataFrame:
    main_results = _read_csv_if_exists(planning_dir / "main_results_table.csv")
    pathway_summary = _read_csv_if_exists(planning_dir / "pathway_summary.csv")
    portfolio_allocations = _read_csv_if_exists(planning_dir / "portfolio_allocations.csv")
    scored_cases = _read_csv_if_exists(planning_dir / "scored_cases.csv")
    scenario_external_evidence = _read_csv_if_exists(planning_dir / "scenario_external_evidence.csv")
    stress_summary = _read_csv_if_exists(scenario_dir / "stress_test_summary.csv")
    if main_results.empty:
        return pd.DataFrame()

    planning_flags = main_results.copy()
    if not pathway_summary.empty:
        planning_flags = planning_flags.merge(
            pathway_summary[
                [
                    "scenario_name",
                    "pathway",
                    "portfolio_selected_count",
                    "portfolio_allocated_feed_share",
                    "portfolio_top_case_id",
                ]
            ],
            on=["scenario_name", "pathway"],
            how="left",
        )
    support_sources: list[pd.DataFrame] = []
    for support_frame in [scored_cases, portfolio_allocations]:
        normalized_support = _normalize_surrogate_support_levels(support_frame)
        if not normalized_support.empty:
            support_sources.append(
                normalized_support[["scenario_name", "pathway", "surrogate_support_level"]]
            )
    if support_sources:
        support_summary = (
            pd.concat(support_sources, ignore_index=True)
            .groupby(["scenario_name", "pathway"], dropna=False)
            .agg(
                Surrogate_Support_Level=(
                    "surrogate_support_level",
                    lambda series: _mode_or_default(series, "unknown"),
                )
            )
            .reset_index()
        )
        planning_flags = planning_flags.merge(
            support_summary,
            on=["scenario_name", "pathway"],
            how="left",
        )
    if not stress_summary.empty:
        top_case_counts = (
            stress_summary.groupby(["scenario_name", "top_portfolio_case_id"], dropna=False)
            .size()
            .rename("top_case_stress_support_count")
            .reset_index()
        )
        planning_flags = planning_flags.merge(
            top_case_counts,
            left_on=["scenario_name", "portfolio_top_case_id"],
            right_on=["scenario_name", "top_portfolio_case_id"],
            how="left",
        ).drop(columns=["top_portfolio_case_id"], errors="ignore")

    planning_flags["claim_status"] = planning_flags.apply(_classify_planning_claim_status, axis=1)
    planning_flags["claim_rule"] = planning_flags.apply(_describe_planning_claim_rule, axis=1)
    support_levels = planning_flags["Surrogate_Support_Level"].fillna("unknown").astype(str)
    evidence_gap_flag = np.where(
        support_levels.eq("documented_static_fallback"),
        "Evidence Gap: Documented Static Fallback",
        np.where(
            support_levels.eq("unsupported_pathway"),
            "Evidence Gap: Unsupported Pathway",
            np.where(
                support_levels.eq("unknown")
                & planning_flags["scenario_name"].isin(
                    scenario_external_evidence.get("scenario_name", pd.Series(dtype="object")).astype(str)
                ),
                "Evidence Gap: Support Level Unresolved",
                "",
            ),
        ),
    )
    planning_flags["evidence_gap_flag"] = pd.Series(
        evidence_gap_flag,
        index=planning_flags.index,
    )
    selected_columns = [
        "scenario_name",
        "pathway",
        "writing_label",
        "claim_status",
        "claim_rule",
        "Surrogate_Support_Level",
        "evidence_gap_flag",
        "selected_in_baseline_portfolio",
        "baseline_portfolio_share_pct",
        "max_stress_selection_rate",
        "stress_tests_supporting_pathway",
        "best_case_score_index",
        "claim_boundary",
        "results_sentence",
    ]
    available = [column for column in selected_columns if column in planning_flags.columns]
    return planning_flags[available].sort_values(
        ["scenario_name", "selected_in_baseline_portfolio", "max_stress_selection_rate", "best_case_score_index"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def build_artifact_inventory(
    summary_paths: dict[str, Path],
    operation_dir: Path,
    planning_dir: Path,
    scenario_dir: Path,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for label, path in summary_paths.items():
        records.append(
            {
                "artifact_group": "ml_summary",
                "artifact_label": label,
                "path": str(path),
                "exists": path.exists(),
            }
        )
    for file_name in [
        "main_results_table.csv",
        "main_results_table_manifest.json",
        "pathway_summary.csv",
        "portfolio_allocations.csv",
        "run_config.json",
    ]:
        path = planning_dir / file_name
        records.append(
            {
                "artifact_group": "planning",
                "artifact_label": file_name,
                "path": str(path),
                "exists": path.exists(),
            }
        )
    for file_name in [
        "stress_test_summary.csv",
        "decision_stability.csv",
        "cross_scenario_stability.csv",
        "uncertainty_summary.csv",
        "run_config.json",
    ]:
        path = scenario_dir / file_name
        records.append(
            {
                "artifact_group": "scenario",
                "artifact_label": file_name,
                "path": str(path),
                "exists": path.exists(),
            }
        )
    for file_name in [
        "baseline_policy_summary.csv",
        "baseline_rollout_steps.csv",
        "policy_behavior_comparison.csv",
        "rl_vs_baseline_comparison.csv",
        "sac_training_summary.csv",
        "sac_evaluation_rollouts.csv",
        "sac_evaluation_episode_summary.csv",
        "sac_seed_aggregate_summary.csv",
        "td3_training_summary.csv",
        "td3_evaluation_rollouts.csv",
        "td3_evaluation_episode_summary.csv",
        "td3_seed_aggregate_summary.csv",
        "run_config.json",
    ]:
        path = operation_dir / file_name
        records.append(
            {
                "artifact_group": "operation_comparison",
                "artifact_label": file_name,
                "path": str(path),
                "exists": path.exists(),
            }
        )
    return pd.DataFrame(records)


def build_audit_manifest(
    summary_paths: dict[str, Path],
    operation_dir: Path,
    planning_dir: Path,
    scenario_dir: Path,
) -> dict[str, object]:
    return {
        "expected_models": list(MODEL_KEYS),
        "expected_datasets": list(DATASET_KEYS),
        "expected_targets": list(TARGET_COLUMNS),
        "ml_summary_paths": {key: str(value) for key, value in summary_paths.items()},
        "planning_dir": str(planning_dir),
        "scenario_dir": str(scenario_dir),
        "operation_comparison_dir": str(operation_dir),
        "confirmation_rules": [
            "strict_group is the main-table benchmark evidence tier",
            "leave_study_out is the stronger cross-study stress test",
            "planning_claim_flag_table is the manuscript-facing planning claim inventory",
            "pyrolysis may retain partial cross-study generalization on some targets",
            "HTC and Paper 1 HTC scope should not be written as uniformly strong cross-study generalization",
            "SAC matching hold_plan should be written as conservative behavior, not superior control",
            "TD3 gains with nonzero violation or high reward_std should be written as scenario-conditional and unstable",
        ],
    }


def _build_claim_rows_from_frame(
    frame: pd.DataFrame,
    *,
    selected_manifest: pd.DataFrame | None = None,
    summary_label: str,
    claim_rule: str,
    positive_threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if frame.empty:
        return rows

    best = _selected_or_best_frame(frame, selected_manifest)
    for _, row in best.iterrows():
        test_r2 = float(row["selected_test_r2"])
        if test_r2 >= positive_threshold:
            claim_status = "supportive"
        elif test_r2 >= 0.0:
            claim_status = "weak"
        else:
            claim_status = "unsupported"
        rows.append(
            {
                "summary_label": summary_label,
                "claim_rule": claim_rule,
                "dataset_key": row["dataset_key"],
                "target_column": row["target_column"],
                "best_model_key": row["selected_model_key"],
                "selection_metric_name": row.get("selection_metric_name", "validation_r2"),
                "selection_metric_value": row.get("selection_metric_value", pd.NA),
                "best_test_r2": test_r2,
                "claim_status": claim_status,
            }
        )
    return rows


def _read_selected_manifest(
    selected_manifest_paths: dict[str, Path] | None,
    label: str,
) -> pd.DataFrame:
    if not selected_manifest_paths:
        return pd.DataFrame()
    path = selected_manifest_paths.get(label)
    if path is None:
        return pd.DataFrame()
    return _read_csv_if_exists(path)


def _selected_or_best_frame(frame: pd.DataFrame, selected_manifest: pd.DataFrame | None) -> pd.DataFrame:
    if selected_manifest is not None and not selected_manifest.empty:
        manifest = selected_manifest.copy()
        manifest = manifest.rename(
            columns={
                "selected_model_key": "selected_model_key",
                "selected_test_r2": "selected_test_r2",
                "selected_test_rmse": "selected_test_rmse",
                "selected_test_mae": "selected_test_mae",
            }
        )
        return manifest.reset_index(drop=True)

    summary_selected = _selected_from_summary_frame(frame)
    if summary_selected is not None and not summary_selected.empty:
        return summary_selected

    fallback = (
        frame.sort_values("test_r2", ascending=False)
        .groupby(["dataset_key", "target_column"], dropna=False)
        .first()
        .reset_index()
    )
    fallback["selected_model_key"] = fallback["model_key"]
    fallback["selected_test_r2"] = fallback["test_r2"]
    fallback["selected_test_rmse"] = fallback["test_rmse"]
    fallback["selected_test_mae"] = fallback["test_mae"]
    fallback["selection_metric_name"] = "legacy_test_r2"
    fallback["selection_metric_value"] = fallback["test_r2"]
    return fallback


def _selected_from_summary_frame(frame: pd.DataFrame) -> pd.DataFrame | None:
    if frame.empty or "model_key" not in frame.columns:
        return None

    selected = pd.DataFrame()
    if "is_selected_model" in frame.columns:
        selected = frame[frame["is_selected_model"].astype(bool)].copy()

    if selected.empty and "selection_rank_within_dataset_target" in frame.columns:
        ranked = frame.copy()
        ranked["selection_rank_within_dataset_target"] = pd.to_numeric(
            ranked["selection_rank_within_dataset_target"],
            errors="coerce",
        )
        selected = ranked.loc[ranked["selection_rank_within_dataset_target"] == 1].copy()

    if selected.empty and "validation_r2" in frame.columns:
        selected_groups: list[pd.DataFrame] = []
        for _, subset in frame.groupby(["dataset_key", "target_column"], dropna=False, sort=False):
            working = subset.copy()
            working["_validation_r2_sort"] = pd.to_numeric(
                working.get("validation_r2"),
                errors="coerce",
            ).fillna(float("-inf"))
            working["_validation_rmse_sort"] = pd.to_numeric(
                working.get("validation_rmse"),
                errors="coerce",
            ).fillna(float("inf"))
            working["_validation_mae_sort"] = pd.to_numeric(
                working.get("validation_mae"),
                errors="coerce",
            ).fillna(float("inf"))
            working = working.sort_values(
                ["_validation_r2_sort", "_validation_rmse_sort", "_validation_mae_sort", "model_key"],
                ascending=[False, True, True, True],
            ).head(1)
            selected_groups.append(
                working.drop(
                    columns=["_validation_r2_sort", "_validation_rmse_sort", "_validation_mae_sort"],
                    errors="ignore",
                )
            )
        if selected_groups:
            selected = pd.concat(selected_groups, ignore_index=True)

    if selected.empty:
        return None

    selected = selected.copy()
    selected["selected_model_key"] = selected["model_key"]
    selected["selected_test_r2"] = pd.to_numeric(selected.get("test_r2"), errors="coerce")
    selected["selected_test_rmse"] = pd.to_numeric(selected.get("test_rmse"), errors="coerce")
    selected["selected_test_mae"] = pd.to_numeric(selected.get("test_mae"), errors="coerce")
    if "selection_metric_name" not in selected.columns:
        selected["selection_metric_name"] = "validation_r2"
    if "selection_metric_value" not in selected.columns:
        selected["selection_metric_value"] = pd.to_numeric(selected.get("validation_r2"), errors="coerce")
    return selected.reset_index(drop=True)


def _pathway_from_dataset_key(dataset_key: object) -> str:
    value = str(dataset_key)
    if "pyrolysis" in value:
        return "pyrolysis"
    if "htc" in value:
        return "htc"
    if "ad" in value:
        return "ad"
    return value


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _classify_planning_claim_status(row: pd.Series) -> str:
    selected = bool(row.get("selected_in_baseline_portfolio", False))
    stress_rate = _optional_float(row.get("max_stress_selection_rate"))
    writing_label = str(row.get("writing_label", ""))
    if selected:
        return "supportive"
    if pd.isna(stress_rate):
        if "comparison anchor" in writing_label:
            return "anchor_only"
        return "not_evaluated"
    if stress_rate > 0.0 and "environment-sensitive" in writing_label:
        return "conditional_support"
    if stress_rate > 0.0:
        return "stress_sensitive"
    if "comparison anchor" in writing_label:
        return "anchor_only"
    return "comparison_only"


def _describe_planning_claim_rule(row: pd.Series) -> str:
    selected = bool(row.get("selected_in_baseline_portfolio", False))
    stress_rate = _optional_float(row.get("max_stress_selection_rate"))
    if str(row.get("writing_label", "")) == "comparison anchor":
        return "Retained as the manuscript comparison anchor rather than as a selected pathway."
    if selected:
        return "Selected in the baseline optimized portfolio under the current planning configuration."
    if pd.isna(stress_rate):
        return "Pathway robustness support is not available in the current scenario audit outputs, so manuscript claims should remain descriptive only."
    if stress_rate > 0.0:
        return "Not selected in the baseline portfolio but supported in at least one planning stress test."
    return "Available for pathway comparison but not currently supported as a selected planning recommendation."


def _optional_float(value: object) -> float | object:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return pd.NA
    return float(numeric)


def _mode_or_default(series: pd.Series, default: str) -> str:
    values = series.astype(str).replace("nan", "").replace("", pd.NA).dropna()
    if values.empty:
        return default
    mode = values.mode(dropna=True)
    if mode.empty:
        return default
    return str(mode.iloc[0])


def _normalize_surrogate_support_levels(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not {"scenario_name", "pathway"}.issubset(frame.columns):
        return pd.DataFrame()
    normalized = frame.copy()
    if "surrogate_support_level" in normalized.columns:
        return normalized
    support_level_series = normalized.get(
        "surrogate_mode",
        pd.Series(["unknown"] * len(normalized), index=normalized.index),
    ).astype(str)
    pathway_series = normalized.get(
        "pathway",
        pd.Series([""] * len(normalized), index=normalized.index),
    ).astype(str).str.lower()
    normalized["surrogate_support_level"] = np.where(
        pathway_series.isin(["pyrolysis", "htc"]) & support_level_series.eq("trained_surrogate"),
        "surrogate_supported",
        np.where(
            pathway_series.isin(["pyrolysis", "htc"])
            & support_level_series.eq("trained_surrogate_with_documented_fallback"),
            "trained_surrogate_with_documented_fallback",
            np.where(
                pathway_series.isin(["pyrolysis", "htc"]),
                "documented_static_fallback",
                "unsupported_pathway",
            ),
        ),
    )
    return normalized


def _emit_surrogate_led_inconsistency_warnings(consistency: pd.DataFrame) -> None:
    if consistency.empty or "surrogate_led_consistency" not in consistency.columns:
        return
    inconsistent = consistency[consistency["surrogate_led_consistency"].astype(str) == "inconsistent"]
    for _, row in inconsistent.iterrows():
        share = _optional_float(row.get("surrogate_supported_allocation_share"))
        share_pct = "unknown" if pd.isna(share) else f"{float(share) * 100.0:.1f}%"
        warnings.warn(
            (
                f"Scenario '{row.get('scenario_name', 'unknown_scenario')}' is inconsistent with a surrogate-led claim: "
                f"surrogate-supported allocated share is {share_pct}, below the 80.0% threshold."
            ),
            InconsistencyWarning,
        )


if __name__ == "__main__":
    raise SystemExit(main())
