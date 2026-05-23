# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json
import re
import warnings
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .common import parse_manifest_timestamp
from .config import (
    BENCHMARK_OUTPUTS_DIR,
    FIGURES_TABLES_DIR,
    MODEL_READY_DIR,
    OUTPUTS_ROOT,
    get_objective_weight_system,
    resolve_surrogate_outputs_dir,
)
from .data import DATASET_KEYS, TARGET_COLUMNS
from .evidence_policy import (
    SURROGATE_LED_SHARE_THRESHOLD,
    build_transferability_note,
    classify_pathway_reliability,
    classify_recommendation_evidence_ceiling,
    classify_scenario_transferability_ceiling,
)
from .models import MODEL_KEYS
from .planning.confidence import build_recommendation_confidence_summary
from .planning.inputs import load_planning_input_bundle
from .planning.solve import PlanningConfig, execute_planning_pipeline


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
DEFAULT_PLANNING_DIR = OUTPUTS_ROOT / "planning"
DEFAULT_SCENARIO_DIR = OUTPUTS_ROOT / "scenarios"
DEFAULT_BENCHMARK_DIR = BENCHMARK_OUTPUTS_DIR / "baseline"
HTC_PRIORITY_DATASETS = frozenset({"htc_direct", "paper1_htc_scope"})
HTC_MODEL_PRIORITY = (
    "catboost",
    "lightgbm",
    "stacking",
    "xgboost",
    "extra_trees",
    "rf",
    "gradient_boosting",
    "elastic_net",
)


def build_confirmatory_audit(
    *,
    outputs_root: str | Path | None = None,
    planning_dir: str | Path | None = None,
    scenario_dir: str | Path | None = None,
    operation_dir: str | Path | None = None,
    benchmark_dir: str | Path | None = None,
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
    active_benchmark_dir = (
        Path(benchmark_dir)
        if benchmark_dir
        else active_outputs_root / DEFAULT_BENCHMARK_DIR.relative_to(OUTPUTS_ROOT)
    )

    ml_summary = build_ml_split_coverage_summary(ml_paths)
    ml_best = build_ml_best_result_summary(ml_paths, selected_manifest_paths)
    ml_flags = build_ml_claim_flag_table(ml_paths, selected_manifest_paths)
    ml_provenance = build_ml_refit_provenance_summary(selected_manifest_paths)
    pathway_reliability = build_pathway_reliability_summary(ml_flags)
    planning_flags = build_planning_claim_flag_table(
        active_planning_dir,
        active_scenario_dir,
        pathway_reliability=pathway_reliability,
    )
    planning_recommendation_confidence = _load_or_build_planning_recommendation_confidence(
        active_planning_dir
    )
    planning_transferability_risk = build_planning_transferability_risk_summary(
        active_planning_dir,
        pathway_reliability,
    )
    planning_ml_consistency = build_planning_ml_consistency_summary(active_planning_dir, pathway_reliability)
    _emit_surrogate_led_inconsistency_warnings(planning_ml_consistency)
    hhv_imputation_sensitivity = build_hhv_imputation_sensitivity(active_planning_dir)
    hhv_replanning_sensitivity = build_hhv_replanning_sensitivity(active_planning_dir)
    hhv_dominance_audit = build_hhv_dominance_audit(
        active_planning_dir,
        hhv_imputation_sensitivity=hhv_imputation_sensitivity,
        hhv_replanning_sensitivity=hhv_replanning_sensitivity,
    )
    surrogate_extrapolation_audit = build_surrogate_extrapolation_audit(
        active_planning_dir,
        ml_flags=ml_flags,
    )
    binding_constraint_audit = build_binding_constraint_audit(
        active_planning_dir,
        active_benchmark_dir,
    )
    duplicate_candidate_audit = build_duplicate_candidate_audit(active_planning_dir)
    ad_boundary_fairness_audit = build_ad_boundary_fairness_audit(
        active_planning_dir,
        active_benchmark_dir,
    )
    planning_data_quality = _read_csv_if_exists(active_planning_dir / "planning_data_quality_summary.csv")
    benchmark_claim_summary = build_benchmark_claim_summary(active_benchmark_dir)
    benchmark_manuscript_sentences = build_benchmark_manuscript_sentences(benchmark_claim_summary)
    operation_table = build_operation_comparison_summary(active_operation_dir)
    operation_flags = build_operation_claim_flag_table(active_operation_dir)
    artifact_inventory = build_artifact_inventory(
        ml_paths,
        active_operation_dir,
        active_planning_dir,
        active_scenario_dir,
        active_benchmark_dir,
    )
    planning_artifact_consistency = build_planning_artifact_consistency_summary(
        active_planning_dir,
        figures_dir=FIGURES_TABLES_DIR,
        audit_dir=active_outputs_root / "audit",
        planning_claim_flags=planning_flags,
    )
    audit_manifest = build_audit_manifest(
        ml_paths,
        active_operation_dir,
        active_planning_dir,
        active_scenario_dir,
        active_benchmark_dir,
    )

    return {
        "ml_split_coverage_summary": ml_summary,
        "ml_best_result_summary": ml_best,
        "ml_claim_flag_table": ml_flags,
        "ml_refit_provenance_summary": ml_provenance,
        "pathway_reliability_summary": pathway_reliability,
        "planning_claim_flag_table": planning_flags,
        "planning_recommendation_confidence_summary": planning_recommendation_confidence,
        "planning_transferability_risk_summary": planning_transferability_risk,
        "planning_ml_consistency_summary": planning_ml_consistency,
        "hhv_imputation_sensitivity": hhv_imputation_sensitivity,
        "hhv_replanning_sensitivity": hhv_replanning_sensitivity,
        "hhv_dominance_audit": hhv_dominance_audit,
        "surrogate_extrapolation_audit": surrogate_extrapolation_audit,
        "binding_constraint_audit": binding_constraint_audit,
        "duplicate_candidate_audit": duplicate_candidate_audit,
        "ad_boundary_fairness_audit": ad_boundary_fairness_audit,
        "planning_data_quality_summary": planning_data_quality,
        "benchmark_claim_summary": benchmark_claim_summary,
        "benchmark_manuscript_sentences": benchmark_manuscript_sentences,
        "operation_comparison_summary": operation_table,
        "operation_claim_flag_table": operation_flags,
        "artifact_inventory": artifact_inventory,
        "planning_artifact_consistency_summary": planning_artifact_consistency,
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
        frame = _read_ml_summary_frame(path, label)
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
        frame = _read_ml_summary_frame(path, label)
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
    strict_group = _read_ml_summary_frame(summary_paths["strict_group"], "strict_group")
    leave_study_out = _read_ml_summary_frame(summary_paths["leave_study_out"], "leave_study_out")

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


def build_ml_refit_provenance_summary(
    selected_manifest_paths: dict[str, Path] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not selected_manifest_paths:
        return pd.DataFrame()

    for summary_label, manifest_path in selected_manifest_paths.items():
        manifest = _read_csv_if_exists(manifest_path)
        if manifest.empty:
            continue
        for _, row in manifest.iterrows():
            run_config_path = Path(str(row.get("run_config_path", "")))
            run_config = _read_json_if_exists(run_config_path)
            required_columns = [
                "selection_trace_id",
                "selection_evidence_source",
                "selection_data_version",
                "selection_data_fingerprint",
                "selection_random_state",
                "benchmark_data_version",
                "benchmark_data_fingerprint",
                "benchmark_random_state",
                "refit_data_version",
                "refit_data_fingerprint",
                "refit_test_data_fingerprint",
                "refit_random_state",
            ]
            missing_fields = [
                column
                for column in required_columns
                if pd.isna(row.get(column)) or str(row.get(column)).strip() == ""
            ]
            model_config = run_config.get("model_config", {})
            rows.append(
                {
                    "summary_label": summary_label,
                    "dataset_key": row.get("dataset_key"),
                    "target_column": row.get("target_column"),
                    "selected_model_key": row.get("selected_model_key"),
                    "artifact_role": row.get("artifact_role", pd.NA),
                    "training_scope": row.get("training_scope", pd.NA),
                    "selection_trace_id": row.get("selection_trace_id", pd.NA),
                    "selection_evidence_source": row.get("selection_evidence_source", pd.NA),
                    "selection_data_version": row.get("selection_data_version", pd.NA),
                    "selection_data_fingerprint": row.get("selection_data_fingerprint", pd.NA),
                    "selection_random_state": row.get("selection_random_state", pd.NA),
                    "benchmark_data_version": row.get("benchmark_data_version", pd.NA),
                    "benchmark_data_fingerprint": row.get("benchmark_data_fingerprint", pd.NA),
                    "benchmark_random_state": row.get("benchmark_random_state", pd.NA),
                    "refit_data_version": row.get("refit_data_version", pd.NA),
                    "refit_data_fingerprint": row.get("refit_data_fingerprint", pd.NA),
                    "refit_test_data_fingerprint": row.get("refit_test_data_fingerprint", pd.NA),
                    "refit_random_state": row.get("refit_random_state", pd.NA),
                    "run_config_exists": run_config_path.exists(),
                    "run_config_random_state": model_config.get("random_state", pd.NA),
                    "provenance_complete": len(missing_fields) == 0,
                    "missing_provenance_fields": ";".join(missing_fields),
                }
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
        mean_best_test_r2 = pd.to_numeric(pathway_frame.get("best_test_r2"), errors="coerce").mean()
        decision = classify_pathway_reliability(
            pathway=pathway,
            reliability_score=reliability_score,
            weak_or_unsupported_ratio=weak_or_unsupported_ratio,
        )
        rows.append(
            {
                "pathway": pathway,
                "leave_study_out_supportive_count": supportive,
                "leave_study_out_weak_count": weak,
                "leave_study_out_unsupported_count": unsupported,
                "leave_study_out_total_count": total,
                "weak_or_unsupported_ratio": weak_or_unsupported_ratio,
                "reliability_score": reliability_score,
                "mean_best_test_r2": mean_best_test_r2,
                "reliability_tier": decision.tier,
                "reviewer_restriction_sentence": decision.reviewer_sentence,
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
        surrogate_supported_with_imputed_key_feature_share = pd.NA
        fully_observed_surrogate_supported_share = pd.NA
        surrogate_feature_imputed_share = pd.NA
        surrogate_imputed_feature_columns = ""
        surrogate_support_evidence_tier = "not_evaluated"
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
                    imputed_flag = scenario_allocations.get(
                        "surrogate_feature_imputation_flag",
                        pd.Series([False] * len(scenario_allocations), index=scenario_allocations.index),
                    ).map(_coerce_bool_flag)
                    imputed_columns = scenario_allocations.get(
                        "surrogate_imputed_feature_columns",
                        pd.Series([""] * len(scenario_allocations), index=scenario_allocations.index),
                    ).fillna("").astype(str)
                    imputed_mask = imputed_flag | imputed_columns.str.len().gt(0)
                    key_imputed_mask = imputed_columns.str.contains(
                        "feedstock_hhv_mj_per_kg",
                        case=False,
                        na=False,
                    )
                    surrogate_feature_imputed_share = float(allocated_feed.loc[imputed_mask].sum() / total_feed)
                    surrogate_supported_with_imputed_key_feature_share = float(
                        allocated_feed.loc[support_mask & key_imputed_mask].sum() / total_feed
                    )
                    fully_observed_surrogate_supported_share = float(
                        allocated_feed.loc[support_mask & ~key_imputed_mask].sum() / total_feed
                    )
                    surrogate_imputed_feature_columns = _join_pipe_values(imputed_columns)
                    if surrogate_supported_share >= SURROGATE_LED_SHARE_THRESHOLD:
                        surrogate_support_evidence_tier = (
                            "surrogate_supported_with_imputed_key_feature"
                            if surrogate_supported_with_imputed_key_feature_share > 0.0
                            else "surrogate_supported"
                        )
                    elif surrogate_supported_with_imputed_key_feature_share > 0.0:
                        surrogate_support_evidence_tier = "partial_surrogate_supported_with_imputed_key_feature"
                    else:
                        surrogate_support_evidence_tier = "insufficient_surrogate_supported_share"
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
                "surrogate_supported_with_imputed_key_feature_allocation_share": (
                    surrogate_supported_with_imputed_key_feature_share
                ),
                "fully_observed_surrogate_supported_allocation_share": fully_observed_surrogate_supported_share,
                "surrogate_feature_imputed_allocation_share": surrogate_feature_imputed_share,
                "surrogate_imputed_feature_columns": surrogate_imputed_feature_columns,
                "surrogate_support_evidence_tier": surrogate_support_evidence_tier,
                "surrogate_led_consistency": surrogate_led_consistency,
                "risk_tier": risk_tier,
                "risk_note": risk_note,
            }
        )
    return pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)


def build_hhv_imputation_sensitivity(planning_dir: Path) -> pd.DataFrame:
    """Summarize ±5/±10% stresses for selected rows with imputed feedstock HHV."""

    allocations = _read_csv_if_exists(planning_dir / "portfolio_allocations.csv")
    required = {"scenario_name", "pathway", "allocated_feed_ton_per_year"}
    if allocations.empty or not required.issubset(allocations.columns):
        return pd.DataFrame()

    working = allocations.copy()
    imputed_columns = working.get(
        "surrogate_imputed_feature_columns",
        pd.Series([""] * len(working), index=working.index),
    ).fillna("").astype(str)
    hhv_imputed_mask = imputed_columns.str.contains(
        "feedstock_hhv_mj_per_kg",
        case=False,
        na=False,
    )
    selected = working.loc[hhv_imputed_mask].copy()
    if selected.empty:
        return pd.DataFrame(
            [
                {
                    "scenario_name": pd.NA,
                    "pathway": pd.NA,
                    "stress_case": "no_hhv_imputed_selected_rows",
                    "allocated_share_pct": 0.0,
                    "selected_row_count": 0,
                    "sample_ids": "",
                    "manure_subtypes": "",
                    "baseline_composition_hhv_mj_per_kg": pd.NA,
                    "stressed_hhv_mj_per_kg": pd.NA,
                    "hhv_delta_pct": 0.0,
                    "evidence_tier": "surrogate_supported",
                    "interpretation": "No selected allocation row used feedstock_hhv_mj_per_kg imputation.",
                }
            ]
        )

    allocated = _numeric_column(working, "allocated_feed_ton_per_year")
    totals = allocated.groupby(working["scenario_name"].astype(str)).transform("sum").replace(0.0, pd.NA)
    working["_allocated_share_pct"] = (allocated / totals * 100.0).fillna(0.0)
    selected["_allocated_share_pct"] = working.loc[selected.index, "_allocated_share_pct"]
    selected["_derived_hhv"] = selected.apply(_derive_feedstock_hhv_from_ultimate_analysis, axis=1)

    stress_specs = [
        ("composition-derived baseline", 0.00),
        ("HHV imputation -10%", -0.10),
        ("HHV imputation -5%", -0.05),
        ("HHV imputation +5%", 0.05),
        ("HHV imputation +10%", 0.10),
    ]
    rows: list[dict[str, object]] = []
    for (scenario_name, pathway), subset in selected.groupby(["scenario_name", "pathway"], dropna=False):
        mass = _numeric_column(subset, "allocated_feed_ton_per_year")
        total_mass = float(mass.sum())
        weights = mass / total_mass if total_mass > 0.0 else pd.Series(0.0, index=subset.index)
        derived = pd.to_numeric(subset["_derived_hhv"], errors="coerce")
        weighted_hhv = float((derived.fillna(0.0) * weights).sum()) if total_mass > 0.0 else pd.NA
        share_pct = float(pd.to_numeric(subset["_allocated_share_pct"], errors="coerce").fillna(0.0).sum())
        sample_ids = _join_pipe_values(subset.get("sample_id", pd.Series([""] * len(subset), index=subset.index)))
        subtypes = _join_pipe_values(subset.get("manure_subtype", pd.Series([""] * len(subset), index=subset.index)))
        for label, delta in stress_specs:
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "pathway": str(pathway).lower(),
                    "stress_case": label,
                    "allocated_share_pct": round(share_pct, 3),
                    "selected_row_count": int(len(subset)),
                    "sample_ids": sample_ids,
                    "manure_subtypes": subtypes,
                    "baseline_composition_hhv_mj_per_kg": round(float(weighted_hhv), 3)
                    if pd.notna(weighted_hhv)
                    else pd.NA,
                    "stressed_hhv_mj_per_kg": round(float(weighted_hhv) * (1.0 + delta), 3)
                    if pd.notna(weighted_hhv)
                    else pd.NA,
                    "hhv_delta_pct": round(delta * 100.0, 1),
                    "evidence_tier": "surrogate_supported_with_imputed_key_feature",
                    "interpretation": (
                        "Selected surrogate rows use composition-derived feedstock_hhv_mj_per_kg; "
                        "the stress cases disclose feature-input dependence rather than claiming new validation."
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["scenario_name", "pathway", "hhv_delta_pct"]).reset_index(drop=True)


def build_hhv_replanning_sensitivity(planning_dir: Path) -> pd.DataFrame:
    """Run true replanning after perturbing composition-derived feedstock HHV.

    Unlike :func:`build_hhv_imputation_sensitivity`, this executes the surrogate
    and optimizer again after materializing ``feedstock_hhv_mj_per_kg`` at
    ±5/±10%.  The output reports pathway-share changes relative to the exported
    baseline portfolio.
    """

    run_config = _read_json_if_exists(planning_dir / "run_config.json")
    dataset_path = run_config.get("dataset_path") if isinstance(run_config, dict) else None
    try:
        bundle = load_planning_input_bundle(dataset_path=dataset_path or None)
        config = _planning_config_from_run_config(run_config)
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "scenario_name": pd.NA,
                    "pathway": pd.NA,
                    "stress_case": "replanning_unavailable",
                    "hhv_delta_pct": pd.NA,
                    "baseline_share_pct": pd.NA,
                    "stressed_share_pct": pd.NA,
                    "share_change_pct_point": pd.NA,
                    "baseline_selected_sample_ids": "",
                    "stressed_selected_sample_ids": "",
                    "replanning_status": "failed_to_initialize",
                    "note": str(exc),
                }
            ]
        )

    frame = bundle.frame.copy()
    if "feedstock_hhv_mj_per_kg" not in frame.columns:
        return pd.DataFrame()
    hhv_missing = pd.to_numeric(frame["feedstock_hhv_mj_per_kg"], errors="coerce").isna()
    derived_hhv = frame.apply(_derive_feedstock_hhv_from_ultimate_analysis, axis=1)
    derivable = pd.to_numeric(derived_hhv, errors="coerce").notna()
    pathway_mask = frame.get("pathway", pd.Series([""] * len(frame), index=frame.index)).astype(str).str.lower().isin(
        ["pyrolysis", "htc"]
    )
    stress_mask = hhv_missing & derivable & pathway_mask
    if not stress_mask.any():
        return pd.DataFrame(
            [
                {
                    "scenario_name": pd.NA,
                    "pathway": pd.NA,
                    "stress_case": "no_derivable_hhv_rows",
                    "hhv_delta_pct": 0.0,
                    "baseline_share_pct": pd.NA,
                    "stressed_share_pct": pd.NA,
                    "share_change_pct_point": pd.NA,
                    "baseline_selected_sample_ids": "",
                    "stressed_selected_sample_ids": "",
                    "replanning_status": "not_evaluated",
                    "note": "No thermochemical rows with missing but derivable feedstock_hhv_mj_per_kg were found.",
                }
            ]
        )

    baseline_allocations = _read_csv_if_exists(planning_dir / "portfolio_allocations.csv")
    stress_specs = [
        ("HHV replanning -10%", -0.10),
        ("HHV replanning -5%", -0.05),
        ("HHV replanning baseline-derived", 0.00),
        ("HHV replanning +5%", 0.05),
        ("HHV replanning +10%", 0.10),
    ]
    rows: list[dict[str, object]] = []
    for label, delta in stress_specs:
        stressed_frame = frame.copy()
        stressed_frame.loc[stress_mask, "feedstock_hhv_mj_per_kg"] = (
            pd.to_numeric(derived_hhv.loc[stress_mask], errors="coerce") * (1.0 + delta)
        )
        try:
            execution = execute_planning_pipeline(bundle=replace(bundle, frame=stressed_frame), config=config)
            stressed_allocations = execution["portfolio_allocations"].copy()
            status = "replanned"
            note = (
                f"Replanned with {int(stress_mask.sum())} thermochemical rows using "
                f"composition-derived feedstock_hhv_mj_per_kg x {1.0 + delta:.2f}."
            )
        except Exception as exc:
            stressed_allocations = pd.DataFrame()
            status = "failed"
            note = str(exc)
        scenarios = sorted(
            set(baseline_allocations.get("scenario_name", pd.Series(dtype="object")).dropna().astype(str))
            | set(stressed_allocations.get("scenario_name", pd.Series(dtype="object")).dropna().astype(str))
        )
        for scenario_name in scenarios:
            for pathway in ("pyrolysis", "htc"):
                baseline_share = _allocated_pathway_share_pct(baseline_allocations, scenario_name, pathway)
                stressed_share = _allocated_pathway_share_pct(stressed_allocations, scenario_name, pathway)
                rows.append(
                    {
                        "scenario_name": scenario_name,
                        "pathway": pathway,
                        "stress_case": label,
                        "hhv_delta_pct": round(delta * 100.0, 1),
                        "baseline_share_pct": round(float(baseline_share), 3),
                        "stressed_share_pct": round(float(stressed_share), 3),
                        "share_change_pct_point": round(float(stressed_share) - float(baseline_share), 3),
                        "baseline_selected_sample_ids": _selected_sample_ids_for_pathway(
                            baseline_allocations,
                            scenario_name=scenario_name,
                            pathway=pathway,
                        ),
                        "stressed_selected_sample_ids": _selected_sample_ids_for_pathway(
                            stressed_allocations,
                            scenario_name=scenario_name,
                            pathway=pathway,
                        ),
                        "replanning_status": status,
                        "note": note,
                    }
                )
    return pd.DataFrame(rows).sort_values(["scenario_name", "pathway", "hhv_delta_pct"]).reset_index(drop=True)


def build_hhv_dominance_audit(
    planning_dir: Path,
    *,
    hhv_imputation_sensitivity: pd.DataFrame | None = None,
    hhv_replanning_sensitivity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Condense HHV-imputation stresses into reviewer-facing dominance evidence.

    The audit asks whether the imputed feedstock HHV feature changes the pathway
    recommendation, as opposed to merely swapping same-pathway case IDs.
    """

    replanning = (
        hhv_replanning_sensitivity.copy()
        if hhv_replanning_sensitivity is not None
        else _read_csv_if_exists(planning_dir / "hhv_replanning_sensitivity.csv")
    )
    if replanning.empty:
        replanning = _read_csv_if_exists(OUTPUTS_ROOT / "audit" / "hhv_replanning_sensitivity.csv")
    imputation = (
        hhv_imputation_sensitivity.copy()
        if hhv_imputation_sensitivity is not None
        else _read_csv_if_exists(OUTPUTS_ROOT / "audit" / "hhv_imputation_sensitivity.csv")
    )
    allocations = _read_csv_if_exists(planning_dir / "portfolio_allocations.csv")
    scenarios = sorted(
        set(allocations.get("scenario_name", pd.Series(dtype="object")).dropna().astype(str))
        | set(replanning.get("scenario_name", pd.Series(dtype="object")).dropna().astype(str))
    )
    if not scenarios:
        return pd.DataFrame()

    affected_lookup: dict[str, float] = {}
    if not imputation.empty and {"scenario_name", "stress_case", "allocated_share_pct"}.issubset(imputation.columns):
        baseline_rows = imputation[
            imputation["stress_case"].astype(str).eq("composition-derived baseline")
        ].copy()
        for scenario_name, subset in baseline_rows.groupby("scenario_name", dropna=False):
            affected_lookup[str(scenario_name)] = float(
                pd.to_numeric(subset["allocated_share_pct"], errors="coerce").fillna(0.0).sum()
            )

    rows: list[dict[str, object]] = []
    for scenario_name in scenarios:
        subset = replanning[replanning.get("scenario_name", pd.Series(dtype="object")).astype(str).eq(scenario_name)].copy()
        if subset.empty:
            max_abs_change = pd.NA
            pathway_changed = False
            case_changed = False
            status = "not_evaluated"
        else:
            max_abs_change = float(pd.to_numeric(subset["share_change_pct_point"], errors="coerce").abs().max())
            baseline_pathways = set(
                subset.loc[
                    pd.to_numeric(
                        subset.get("baseline_share_pct", pd.Series([0.0] * len(subset), index=subset.index)),
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .gt(1.0),
                    "pathway",
                ]
                .astype(str)
                .str.lower()
            )
            pathway_changed = False
            for _stress, stress_rows in subset.groupby("stress_case", dropna=False):
                stressed_pathways = set(
                    stress_rows.loc[
                        pd.to_numeric(
                            stress_rows.get(
                                "stressed_share_pct",
                                pd.Series([0.0] * len(stress_rows), index=stress_rows.index),
                            ),
                            errors="coerce",
                        )
                        .fillna(0.0)
                        .gt(1.0),
                        "pathway",
                    ]
                    .astype(str)
                    .str.lower()
                )
                if stressed_pathways != baseline_pathways:
                    pathway_changed = True
                    break
            case_changed = bool(
                (
                    subset.get("baseline_selected_sample_ids", pd.Series([""] * len(subset), index=subset.index))
                    .fillna("")
                    .astype(str)
                    != subset.get("stressed_selected_sample_ids", pd.Series([""] * len(subset), index=subset.index))
                    .fillna("")
                    .astype(str)
                ).any()
            )
            failed = subset.get("replanning_status", pd.Series([""] * len(subset), index=subset.index)).astype(str).str.contains(
                "failed", case=False, na=False
            )
            status = "failed" if bool(failed.any()) else "evaluated"
        affected_share = affected_lookup.get(scenario_name, 0.0)
        conclusion = _classify_hhv_dominance(
            max_abs_change=max_abs_change,
            pathway_changed=pathway_changed,
            case_changed=case_changed,
        )
        rows.append(
            {
                "scenario_name": scenario_name,
                "affected_imputed_share_pct": round(float(affected_share), 3),
                "max_abs_pathway_share_change_pct_point": round(float(max_abs_change), 3)
                if pd.notna(max_abs_change)
                else pd.NA,
                "selected_pathways_changed": pathway_changed,
                "selected_case_changed": case_changed,
                "hhv_dominance_conclusion": conclusion,
                "reviewer_safe_sentence": _build_hhv_dominance_sentence(
                    scenario_name=scenario_name,
                    affected_share=affected_share,
                    max_abs_change=max_abs_change,
                    conclusion=conclusion,
                ),
                "audit_status": status,
            }
        )
    return pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)


def build_surrogate_extrapolation_audit(
    planning_dir: Path,
    *,
    ml_flags: pd.DataFrame | None = None,
    training_dataset_path: Path | None = None,
) -> pd.DataFrame:
    """Audit the external-validity ceiling of selected thermochemical portfolios."""

    allocations = _read_csv_if_exists(planning_dir / "portfolio_allocations.csv")
    if allocations.empty or not {"scenario_name", "pathway"}.issubset(allocations.columns):
        return pd.DataFrame()
    flags = ml_flags.copy() if ml_flags is not None else _read_csv_if_exists(OUTPUTS_ROOT / "audit" / "ml_claim_flag_table.csv")
    training = _read_csv_if_exists(training_dataset_path or (MODEL_READY_DIR / "ml_training_dataset.csv"))
    range_lookup = _build_training_range_lookup(training)

    selected = allocations[
        allocations["pathway"].astype(str).str.lower().isin(["pyrolysis", "htc"])
        & (_numeric_column(allocations, "allocated_feed_ton_per_year") > 0.0)
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    allocated = _numeric_column(selected, "allocated_feed_ton_per_year")
    scenario_totals = allocated.groupby(selected["scenario_name"].astype(str)).transform("sum").replace(0.0, pd.NA)
    selected["_allocated_share_pct"] = (allocated / scenario_totals * 100.0).fillna(0.0)

    rows: list[dict[str, object]] = []
    for (scenario_name, pathway), subset in selected.groupby(["scenario_name", "pathway"], dropna=False):
        pathway_key = str(pathway).lower()
        lso = _leave_study_out_rows_for_pathway(flags, pathway_key)
        weakest_status = _weakest_claim_status(lso.get("claim_status", pd.Series(dtype="object")))
        weak_targets = _join_pipe_values(
            lso.loc[
                lso.get("claim_status", pd.Series(dtype="object")).astype(str).isin(["weak", "unsupported"]),
                "target_column",
            ]
            if "target_column" in lso.columns
            else pd.Series(dtype="object")
        )
        min_r2 = (
            float(pd.to_numeric(lso.get("best_test_r2"), errors="coerce").min())
            if not lso.empty and "best_test_r2" in lso.columns
            else pd.NA
        )
        range_summary = _selected_range_summary(subset, range_lookup.get(pathway_key, {}))
        ceiling = _classify_surrogate_extrapolation_ceiling(
            weakest_status=weakest_status,
            all_in_range=range_summary["within_training_range_all_features"],
            missing_range=range_summary["feature_range_status"] == "training_range_unavailable",
        )
        share_pct = float(pd.to_numeric(subset["_allocated_share_pct"], errors="coerce").fillna(0.0).sum())
        rows.append(
            {
                "scenario_name": str(scenario_name),
                "pathway": pathway_key,
                "allocated_share_pct": round(share_pct, 3),
                "leave_study_out_target_count": int(len(lso)),
                "min_leave_study_out_test_r2": round(float(min_r2), 3) if pd.notna(min_r2) else pd.NA,
                "weakest_leave_study_out_claim_status": weakest_status,
                "weak_or_unsupported_targets": weak_targets,
                "selected_rows_evaluated": int(len(subset)),
                "feature_range_status": range_summary["feature_range_status"],
                "within_training_range_all_features": range_summary["within_training_range_all_features"],
                "out_of_range_feature_count": range_summary["out_of_range_feature_count"],
                "out_of_range_features": range_summary["out_of_range_features"],
                "extrapolation_evidence_ceiling": ceiling,
                "reviewer_safe_sentence": _build_surrogate_extrapolation_sentence(
                    pathway=pathway_key,
                    weakest_status=weakest_status,
                    min_r2=min_r2,
                    ceiling=ceiling,
                    out_of_range_features=range_summary["out_of_range_features"],
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario_name", "pathway"]).reset_index(drop=True)


def build_binding_constraint_audit(planning_dir: Path, benchmark_dir: Path) -> pd.DataFrame:
    """Expose which portfolio constraints bind and how cap relaxation changes shares."""

    diagnostics = _read_csv_if_exists(planning_dir / "optimization_diagnostics.csv")
    allocations = _read_csv_if_exists(planning_dir / "portfolio_allocations.csv")
    cap_diagnostics = _read_csv_if_exists(
        _resolve_targeted_ablation_dir_for_audit(benchmark_dir) / "portfolio_cap_diagnostics.csv"
    )
    if diagnostics.empty or "scenario_name" not in diagnostics.columns:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, row in diagnostics.iterrows():
        scenario_name = str(row.get("scenario_name"))
        base_pyro = _allocated_pathway_share_pct(allocations, scenario_name, "pyrolysis")
        base_htc = _allocated_pathway_share_pct(allocations, scenario_name, "htc")
        relaxed = _cap_diagnostic_row(
            cap_diagnostics,
            scenario_name=scenario_name,
            ablation_key="candidate_and_subtype_caps_relaxed",
        )
        candidate_relaxed = _cap_diagnostic_row(
            cap_diagnostics,
            scenario_name=scenario_name,
            ablation_key="candidate_cap_relaxed_100pct",
        )
        relaxed_pyro = _optional_float(relaxed.get("pyrolysis_allocated_share_pct")) if not relaxed.empty else pd.NA
        relaxed_htc = _optional_float(relaxed.get("htc_allocated_share_pct")) if not relaxed.empty else pd.NA
        candidate_relaxed_max = (
            _optional_float(candidate_relaxed.get("max_candidate_allocated_share_pct"))
            if not candidate_relaxed.empty
            else pd.NA
        )
        rows.append(
            {
                "scenario_name": scenario_name,
                "candidate_cap_binding": _coerce_bool_flag(row.get("candidate_cap_binding")),
                "candidate_cap_slack_ton_per_year": _optional_float(row.get("candidate_cap_slack_ton_per_year")),
                "subtype_cap_binding": _coerce_bool_flag(row.get("subtype_cap_binding")),
                "subtype_cap_slack_ton_per_year": _optional_float(row.get("subtype_cap_slack_ton_per_year")),
                "residual_carbon_constraint_binding": _coerce_bool_flag(row.get("carbon_budget_binding")),
                "residual_carbon_slack_kgco2e": _optional_float(row.get("carbon_budget_slack_kgco2e")),
                "min_distinct_subtypes_binding": _coerce_bool_flag(row.get("min_distinct_subtypes_binding")),
                "max_selected_binding": _coerce_bool_flag(row.get("max_selected_binding")),
                "baseline_pyrolysis_share_pct": round(float(base_pyro), 3),
                "baseline_htc_share_pct": round(float(base_htc), 3),
                "cap_relaxed_pyrolysis_share_pct": round(float(relaxed_pyro), 3)
                if pd.notna(relaxed_pyro)
                else pd.NA,
                "cap_relaxed_htc_share_pct": round(float(relaxed_htc), 3) if pd.notna(relaxed_htc) else pd.NA,
                "cap_relaxed_pyrolysis_share_change_pct_point": round(float(relaxed_pyro) - float(base_pyro), 3)
                if pd.notna(relaxed_pyro)
                else pd.NA,
                "cap_relaxed_htc_share_change_pct_point": round(float(relaxed_htc) - float(base_htc), 3)
                if pd.notna(relaxed_htc)
                else pd.NA,
                "candidate_cap_relaxed_max_candidate_share_pct": round(float(candidate_relaxed_max), 3)
                if pd.notna(candidate_relaxed_max)
                else pd.NA,
                "interpretation": _binding_constraint_interpretation(row, relaxed_pyro, base_pyro),
            }
        )
    return pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)


def build_duplicate_candidate_audit(planning_dir: Path) -> pd.DataFrame:
    """Find selected pyrolysis rows with identical rounded condition/target signatures."""

    allocations = _read_csv_if_exists(planning_dir / "portfolio_allocations.csv")
    if allocations.empty or "pathway" not in allocations.columns:
        return pd.DataFrame()
    selected = allocations[
        allocations["pathway"].astype(str).str.lower().eq("pyrolysis")
        & (_numeric_column(allocations, "allocated_feed_ton_per_year") > 0.0)
    ].copy()
    if selected.empty:
        return pd.DataFrame()

    key_columns = [
        "process_temperature_c",
        "residence_time_min",
        "heating_rate_c_per_min",
        "predicted_product_char_yield_pct",
        "predicted_product_char_hhv_mj_per_kg",
        "predicted_energy_recovery_pct",
        "predicted_carbon_retention_pct",
        "product_char_yield_pct",
        "product_char_hhv_mj_per_kg",
        "energy_recovery_pct",
        "carbon_retention_pct",
        "feedstock_carbon_pct",
        "feedstock_hydrogen_pct",
        "feedstock_nitrogen_pct",
        "feedstock_oxygen_pct",
        "feedstock_ash_pct",
    ]
    for column in key_columns:
        selected[f"_dup_{column}"] = pd.to_numeric(
            selected.get(column, pd.Series([pd.NA] * len(selected), index=selected.index)),
            errors="coerce",
        ).round(3)
    group_cols = [f"_dup_{column}" for column in key_columns]
    selected["_duplicate_signature"] = selected[group_cols].apply(
        lambda row: "|".join(str(value) for value in row.to_numpy()),
        axis=1,
    )

    rows: list[dict[str, object]] = []
    for (scenario_name, _signature), subset in selected.groupby(["scenario_name", "_duplicate_signature"], dropna=False):
        if len(subset) < 2:
            continue
        subtype_count = subset.get("manure_subtype", pd.Series(dtype=object)).astype(str).nunique(dropna=True)
        sample_count = subset.get("sample_id", pd.Series(dtype=object)).astype(str).nunique(dropna=True)
        share_pct = float(pd.to_numeric(subset.get("allocated_feed_share"), errors="coerce").fillna(0.0).sum() * 100.0)
        rows.append(
            {
                "scenario_name": scenario_name,
                "pathway": "pyrolysis",
                "duplicate_group_id": f"{scenario_name}::pyrolysis::{len(rows) + 1:02d}",
                "rows_in_group": int(len(subset)),
                "distinct_sample_ids": int(sample_count),
                "distinct_subtypes": int(subtype_count),
                "allocated_share_pct": round(share_pct, 3),
                "sample_ids": _join_pipe_values(subset.get("sample_id", pd.Series([""] * len(subset), index=subset.index))),
                "manure_subtypes": _join_pipe_values(
                    subset.get("manure_subtype", pd.Series([""] * len(subset), index=subset.index))
                ),
                "operating_condition": _format_duplicate_operating_condition(subset.iloc[0]),
                "predicted_target_signature": _format_duplicate_target_signature(subset.iloc[0]),
                "audit_finding": (
                    "duplicate_operating_and_target_signature"
                    if subtype_count > 1
                    else "duplicate_operating_signature_same_subtype"
                ),
                "interpretation": (
                    "Selected pyrolysis rows share the same rounded operating-condition and predicted-target signature; "
                    "portfolio diversity should be interpreted as subtype/constraint diversification rather than "
                    "independent thermochemical regimes."
                ),
            }
        )
    if rows:
        return pd.DataFrame(rows).sort_values(["scenario_name", "allocated_share_pct"], ascending=[True, False]).reset_index(drop=True)
    return pd.DataFrame(
        [
            {
                "scenario_name": pd.NA,
                "pathway": "pyrolysis",
                "duplicate_group_id": "none",
                "rows_in_group": 0,
                "distinct_sample_ids": 0,
                "distinct_subtypes": 0,
                "allocated_share_pct": 0.0,
                "sample_ids": "",
                "manure_subtypes": "",
                "operating_condition": "",
                "predicted_target_signature": "",
                "audit_finding": "no_duplicate_selected_pyrolysis_signature",
                "interpretation": "No duplicate selected pyrolysis operating/target signatures were found.",
            }
        ]
    )


def build_ad_boundary_fairness_audit(planning_dir: Path, benchmark_dir: Path) -> pd.DataFrame:
    """Document AD as a boundary/reference diagnostic rather than a rejected technology."""

    ad_reference = _read_csv_if_exists(planning_dir / "ad_reference_diagnostics.csv")
    targeted_dir = _resolve_targeted_ablation_summary_dir_for_audit(benchmark_dir)
    ablations = _read_csv_if_exists(targeted_dir / "targeted_planning_ablations_summary.csv")
    scenarios = sorted(
        set(ad_reference.get("scenario_name", pd.Series(dtype="object")).dropna().astype(str))
        | set(ablations.get("scenario_name", pd.Series(dtype="object")).dropna().astype(str))
    )
    if not scenarios:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for scenario_name in scenarios:
        ad_ref_row = _first_matching_row(ad_reference, scenario_name=scenario_name)
        ad_reference_present = not ad_ref_row.empty
        primary_ad_share = (
            _optional_float(ad_ref_row.get("baseline_portfolio_share_pct"))
            if ad_reference_present
            else pd.NA
        )
        complement_10 = _ablation_share(
            ablations,
            scenario_name=scenario_name,
            ablation_family="ad_complementarity",
            ablation_key="ad_min_share_10pct",
            share_column="ad_allocated_share_pct",
        )
        complement_20 = _ablation_share(
            ablations,
            scenario_name=scenario_name,
            ablation_family="ad_complementarity",
            ablation_key="ad_min_share_20pct",
            share_column="ad_allocated_share_pct",
        )
        digestate_max = _max_ablation_share(
            ablations,
            scenario_name=scenario_name,
            ablation_family="coproduct_boundary",
            key_contains="digestate_rng_credit",
            share_column="ad_allocated_share_pct",
        )
        ad_floor_feasible = pd.notna(complement_10) and float(complement_10) >= 9.9
        evidence_status = _classify_ad_boundary_evidence_status(
            ad_reference_present=ad_reference_present,
            complement_10=complement_10,
            complement_20=complement_20,
            digestate_max=digestate_max,
        )
        role_conclusion = (
            "boundary_reference_not_technical_inferiority"
            if evidence_status == "evaluated" and ad_floor_feasible
            else "boundary_evidence_incomplete"
        )
        boundary_sentence = (
            "AD is excluded from the primary thermochemical optimizer because the evidence, cost, RNG, "
            "and digestate-revenue boundaries are not commensurate with HTC/pyrolysis rows; AD remains "
            "feasible as a biological-reference or policy-floor diagnostic and is not interpreted as "
            "technically inferior."
            if role_conclusion == "boundary_reference_not_technical_inferiority"
            else (
                "AD boundary evidence is incomplete because one or more AD floor/credit diagnostics are missing; "
                "do not use this export to support a non-inferiority boundary statement."
            )
        )
        rows.append(
            {
                "scenario_name": scenario_name,
                "primary_optimizer_ad_share_pct": round(float(primary_ad_share), 3)
                if pd.notna(primary_ad_share)
                else pd.NA,
                "ad_min_10pct_floor_share_pct": round(float(complement_10), 3)
                if pd.notna(complement_10)
                else pd.NA,
                "ad_min_20pct_floor_share_pct": round(float(complement_20), 3)
                if pd.notna(complement_20)
                else pd.NA,
                "digestate_rng_credit_max_ad_share_pct": round(float(digestate_max), 3)
                if pd.notna(digestate_max)
                else pd.NA,
                "ad_policy_floor_feasible": bool(ad_floor_feasible),
                "ad_boundary_evidence_status": evidence_status,
                "ad_role_conclusion": role_conclusion,
                "not_technical_inferiority_sentence": boundary_sentence,
            }
        )
    return pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)


def build_planning_transferability_risk_summary(
    planning_dir: Path,
    pathway_reliability: pd.DataFrame,
) -> pd.DataFrame:
    pathway_summary = _read_csv_if_exists(planning_dir / "pathway_summary.csv")
    if pathway_summary.empty or pathway_reliability.empty:
        return pd.DataFrame()

    reliability_columns = [
        column
        for column in [
            "pathway",
            "reliability_score",
            "reliability_tier",
            "reviewer_restriction_sentence",
        ]
        if column in pathway_reliability.columns
    ]
    reliability_lookup = pathway_reliability[reliability_columns].drop_duplicates(subset=["pathway"])
    rows: list[dict[str, object]] = []

    for scenario_name, frame in pathway_summary.groupby("scenario_name", dropna=False):
        working = frame.copy().merge(reliability_lookup, on="pathway", how="left")
        share = pd.to_numeric(working["portfolio_allocated_feed_share"], errors="coerce").fillna(0.0)
        selected = pd.to_numeric(working["portfolio_selected_count"], errors="coerce").fillna(0.0).gt(0.0)
        working["selected_share"] = np.where(selected, share, 0.0)
        total_selected_share = float(working["selected_share"].sum())
        if total_selected_share <= 0.0:
            total_selected_share = 1.0

        reliability_score = pd.to_numeric(working.get("reliability_score"), errors="coerce").fillna(0.0)
        missing_mask = working["reliability_tier"].isna()
        supportive_mask = working["reliability_tier"].astype(str).eq("conditional_support")
        limited_mask = working["reliability_tier"].astype(str).eq("limited_support")
        auxiliary_mask = working["reliability_tier"].astype(str).eq("auxiliary_only")

        supportive_share = float(working.loc[supportive_mask, "selected_share"].sum())
        limited_share = float(working.loc[limited_mask, "selected_share"].sum())
        auxiliary_share = float(working.loc[auxiliary_mask, "selected_share"].sum())
        missing_share = float(working.loc[missing_mask, "selected_share"].sum())
        weighted_score = float((working["selected_share"] * reliability_score).sum() / total_selected_share)
        evidence_ceiling = _classify_scenario_transferability_ceiling(
            weighted_score=weighted_score,
            auxiliary_share=auxiliary_share,
            limited_share=limited_share,
            missing_share=missing_share,
        )
        rows.append(
            {
                "scenario_name": scenario_name,
                "selected_pathway_count": int(selected.sum()),
                "supportive_transfer_share": supportive_share,
                "limited_transfer_share": limited_share,
                "auxiliary_transfer_share": auxiliary_share,
                "missing_transfer_share": missing_share,
                "weighted_transferability_score": weighted_score,
                "transferability_evidence_ceiling": evidence_ceiling,
                "transferability_note": _build_transferability_note(
                    evidence_ceiling=evidence_ceiling,
                    weighted_score=weighted_score,
                    auxiliary_share=auxiliary_share,
                    limited_share=limited_share,
                    missing_share=missing_share,
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)
    for column in [
        "supportive_transfer_share",
        "limited_transfer_share",
        "auxiliary_transfer_share",
        "missing_transfer_share",
        "weighted_transferability_score",
    ]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").round(3)
    return summary


def build_benchmark_claim_summary(benchmark_dir: Path) -> pd.DataFrame:
    summary = _read_csv_if_exists(benchmark_dir / "benchmark_summary.csv")
    shifts = _read_csv_if_exists(benchmark_dir / "benchmark_shift_summary.csv")
    statistical_summary = _read_csv_if_exists(benchmark_dir / "benchmark_statistical_summary.csv")
    if summary.empty or shifts.empty:
        return pd.DataFrame()

    baseline = summary[summary["benchmark_variant"] == "baseline_evidence_aware"].copy()
    if baseline.empty:
        return pd.DataFrame()
    baseline = baseline.rename(
        columns={
            "portfolio_score_mass": "baseline_portfolio_score_mass",
            "portfolio_carbon_load_kgco2e": "baseline_portfolio_carbon_load_kgco2e",
            "scenario_feed_coverage_ratio": "baseline_scenario_feed_coverage_ratio",
            "selected_pathways": "baseline_selected_pathways",
        }
    )
    merged = shifts.merge(
        baseline[
            [
                "scenario_name",
                "baseline_portfolio_score_mass",
                "baseline_portfolio_carbon_load_kgco2e",
                "baseline_scenario_feed_coverage_ratio",
                "baseline_selected_pathways",
            ]
        ],
        on="scenario_name",
        how="left",
        suffixes=("", "_baseline"),
    )
    if not statistical_summary.empty:
        statistical_columns = [
            column
            for column in [
                "scenario_name",
                "benchmark_variant",
                "bootstrap_replicate_count",
                "pathway_shift_count",
                "pathway_shift_rate",
                "pathway_shift_rate_ci_lower",
                "pathway_shift_rate_ci_upper",
                "case_shift_count",
                "case_shift_rate",
                "case_shift_rate_ci_lower",
                "case_shift_rate_ci_upper",
                "delta_portfolio_score_mass_median",
                "delta_portfolio_score_mass_ci_lower",
                "delta_portfolio_score_mass_ci_upper",
                "delta_portfolio_score_mass_ci_excludes_zero",
                "delta_portfolio_score_mass_sign_agreement_rate",
                "delta_portfolio_score_mass_empirical_p_value",
                "delta_portfolio_score_mass_direction",
                "delta_portfolio_carbon_load_kgco2e_median",
                "delta_portfolio_carbon_load_kgco2e_ci_lower",
                "delta_portfolio_carbon_load_kgco2e_ci_upper",
                "delta_portfolio_carbon_load_kgco2e_ci_excludes_zero",
                "delta_portfolio_carbon_load_kgco2e_sign_agreement_rate",
                "delta_portfolio_carbon_load_kgco2e_empirical_p_value",
                "delta_portfolio_carbon_load_kgco2e_direction",
                "delta_scenario_feed_coverage_ratio_median",
                "delta_scenario_feed_coverage_ratio_ci_lower",
                "delta_scenario_feed_coverage_ratio_ci_upper",
                "delta_scenario_feed_coverage_ratio_ci_excludes_zero",
                "delta_scenario_feed_coverage_ratio_sign_agreement_rate",
                "delta_scenario_feed_coverage_ratio_empirical_p_value",
                "delta_scenario_feed_coverage_ratio_direction",
                "effect_significance_tier",
            ]
            if column in statistical_summary.columns
        ]
        merged = merged.merge(
            statistical_summary[statistical_columns],
            on=["scenario_name", "benchmark_variant"],
            how="left",
        )
    merged["necessity_tier"] = merged.apply(_classify_benchmark_necessity_tier, axis=1)
    merged["necessity_note"] = merged.apply(_build_benchmark_necessity_note, axis=1)
    merged["manuscript_sentence"] = merged.apply(_build_benchmark_sentence, axis=1)
    return merged.sort_values(["scenario_name", "benchmark_variant"]).reset_index(drop=True)


def build_benchmark_manuscript_sentences(benchmark_claim_summary: pd.DataFrame) -> pd.DataFrame:
    if benchmark_claim_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for variant_key, frame in benchmark_claim_summary.groupby("benchmark_variant", dropna=False):
        changed_pathway_count = int(frame["portfolio_pathway_shift"].astype(str).eq("changed").sum())
        changed_case_count = int(frame["portfolio_case_shift"].astype(str).eq("changed").sum())
        strong_support_count = int(frame["necessity_tier"].astype(str).eq("supports_core_innovation").sum())
        secondary_support_count = int(
            frame["necessity_tier"].astype(str).eq("supports_secondary_innovation").sum()
        )
        rows.append(
            {
                "benchmark_variant": variant_key,
                "scenario_count": int(len(frame)),
                "changed_pathway_count": changed_pathway_count,
                "changed_case_count": changed_case_count,
                "supports_core_innovation_count": strong_support_count,
                "supports_secondary_innovation_count": secondary_support_count,
                "manuscript_sentence": _build_benchmark_aggregate_sentence(
                    variant_key=variant_key,
                    changed_pathway_count=changed_pathway_count,
                    changed_case_count=changed_case_count,
                    strong_support_count=strong_support_count,
                    secondary_support_count=secondary_support_count,
                    scenario_count=int(len(frame)),
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("benchmark_variant").reset_index(drop=True)


def build_planning_claim_flag_table(
    planning_dir: Path,
    scenario_dir: Path,
    pathway_reliability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    main_results = _read_csv_if_exists(planning_dir / "main_results_table.csv")
    recommendation_confidence = _load_or_build_planning_recommendation_confidence(planning_dir)
    pathway_summary = _read_csv_if_exists(planning_dir / "pathway_summary.csv")
    portfolio_allocations = _read_csv_if_exists(planning_dir / "portfolio_allocations.csv")
    scored_cases = _read_csv_if_exists(planning_dir / "scored_cases.csv")
    scenario_external_evidence = _read_csv_if_exists(planning_dir / "scenario_external_evidence.csv")
    stress_summary = _read_csv_if_exists(scenario_dir / "stress_test_summary.csv")
    if main_results.empty:
        return pd.DataFrame()

    planning_flags = main_results.copy()
    reliability_frame = pathway_reliability.copy() if pathway_reliability is not None else pd.DataFrame()
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
    if not recommendation_confidence.empty:
        confidence_columns = [
            "recommendation_confidence_score",
            "recommendation_confidence_tier",
            "recommendation_confidence_note",
        ]
        if any(column not in planning_flags.columns for column in confidence_columns):
            planning_flags = planning_flags.merge(
                recommendation_confidence[
                    [
                        "scenario_name",
                        "pathway",
                        "recommendation_confidence_score",
                        "recommendation_confidence_tier",
                        "recommendation_confidence_note",
                    ]
                ],
                on=["scenario_name", "pathway"],
                how="left",
            )
    if not reliability_frame.empty:
        reliability_columns = [
            column
            for column in [
                "pathway",
                "reliability_score",
                "reliability_tier",
                "reviewer_restriction_sentence",
            ]
            if column in reliability_frame.columns
        ]
        if reliability_columns:
            planning_flags = planning_flags.merge(
                reliability_frame[reliability_columns].drop_duplicates(subset=["pathway"]),
                on="pathway",
                how="left",
            )

    planning_flags["claim_status"] = planning_flags.apply(_classify_planning_claim_status, axis=1)
    planning_flags["claim_rule"] = planning_flags.apply(_describe_planning_claim_rule, axis=1)
    if "reliability_score" in planning_flags.columns:
        planning_flags["reliability_score"] = pd.to_numeric(
            planning_flags["reliability_score"],
            errors="coerce",
        )
    planning_flags["recommendation_evidence_ceiling"] = planning_flags.apply(
        _classify_recommendation_evidence_ceiling,
        axis=1,
    )
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
        "uq_stress_support",
        "max_uq_stress_selection_rate",
        "uncertainty_mode_sensitivity",
        "uncertainty_mode_case_switch_count",
        "uncertainty_mode_pathway_switch_count",
        "best_case_uq_ranking_note",
        "best_case_score_index",
        "claim_boundary",
        "recommendation_confidence_score",
        "recommendation_confidence_tier",
        "recommendation_confidence_note",
        "reliability_score",
        "reliability_tier",
        "recommendation_evidence_ceiling",
        "reviewer_restriction_sentence",
        "results_sentence",
    ]
    available = [column for column in selected_columns if column in planning_flags.columns]
    return planning_flags[available].sort_values(
        ["scenario_name", "selected_in_baseline_portfolio", "max_stress_selection_rate", "best_case_score_index"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def build_planning_artifact_consistency_summary(
    planning_dir: str | Path,
    *,
    figures_dir: str | Path | None = None,
    audit_dir: str | Path | None = None,
    planning_claim_flags: pd.DataFrame | None = None,
    tolerance_pct_point: float = 0.1,
) -> pd.DataFrame:
    """Compare manuscript-facing allocation shares against portfolio_allocations."""
    planning_root = Path(planning_dir)
    figures_root = Path(figures_dir) if figures_dir else FIGURES_TABLES_DIR
    audit_root = Path(audit_dir) if audit_dir else OUTPUTS_ROOT / "audit"
    expected = _portfolio_share_reference(planning_root / "portfolio_allocations.csv")
    if expected.empty:
        return pd.DataFrame(
            [
                {
                    "artifact_label": "portfolio_allocations.csv",
                    "scenario_name": pd.NA,
                    "pathway": pd.NA,
                    "expected_share_pct": pd.NA,
                    "observed_share_pct": pd.NA,
                    "absolute_difference_pct_point": pd.NA,
                    "consistency_status": "not_evaluated",
                    "consistency_note": "Source portfolio_allocations.csv is missing or lacks scenario/pathway allocation data.",
                }
            ]
        )

    artifact_sources: list[tuple[str, pd.DataFrame]] = [
        ("main_results_table.csv", _read_csv_if_exists(planning_root / "main_results_table.csv")),
        (
            "paper1_planning_results_table.csv",
            _read_csv_if_exists(figures_root / "paper1_planning_results_table.csv"),
        ),
        (
            "planning_claim_flag_table.csv",
            planning_claim_flags.copy()
            if planning_claim_flags is not None
            else _read_csv_if_exists(audit_root / "planning_claim_flag_table.csv"),
        ),
    ]
    rows: list[dict[str, object]] = []
    for artifact_label, artifact in artifact_sources:
        observed = _artifact_share_reference(artifact)
        if observed.empty:
            for _, ref_row in expected.iterrows():
                rows.append(
                    {
                        "artifact_label": artifact_label,
                        "scenario_name": ref_row["scenario_name"],
                        "pathway": ref_row["pathway"],
                        "expected_share_pct": ref_row["expected_share_pct"],
                        "observed_share_pct": pd.NA,
                        "absolute_difference_pct_point": pd.NA,
                        "consistency_status": "missing_artifact_or_columns",
                        "consistency_note": "Artifact is missing or lacks baseline_portfolio_share_pct.",
                    }
                )
            continue
        merged = expected.merge(observed, on=["scenario_name", "pathway"], how="left")
        for _, row in merged.iterrows():
            observed_share = (
                0.0
                if pd.isna(row.get("observed_share_pct"))
                else _optional_float(row.get("observed_share_pct"))
            )
            expected_share = _optional_float(row.get("expected_share_pct"))
            diff = (
                abs(float(observed_share) - float(expected_share))
                if pd.notna(observed_share) and pd.notna(expected_share)
                else pd.NA
            )
            status = "pass" if pd.notna(diff) and float(diff) <= tolerance_pct_point else "fail"
            rows.append(
                {
                    "artifact_label": artifact_label,
                    "scenario_name": row["scenario_name"],
                    "pathway": row["pathway"],
                    "expected_share_pct": round(float(expected_share), 3)
                    if pd.notna(expected_share)
                    else pd.NA,
                    "observed_share_pct": round(float(observed_share), 3)
                    if pd.notna(observed_share)
                    else pd.NA,
                    "absolute_difference_pct_point": round(float(diff), 3)
                    if pd.notna(diff)
                    else pd.NA,
                    "consistency_status": status,
                    "consistency_note": (
                        f"Allocation share differs by more than {tolerance_pct_point:.1f} percentage point."
                        if status == "fail"
                        else "Allocation share is aligned with portfolio_allocations.csv."
                    ),
                }
            )
    _append_profile_artifact_consistency_rows(
        rows,
        expected=expected,
        artifact_label="paper1_core_boundary_regime_table.csv",
        artifact=_read_csv_if_exists(figures_root / "paper1_core_boundary_regime_table.csv"),
        selector_column="boundary_regime",
        selector_value="Declared asymmetric credit + diversification rule",
        tolerance_pct_point=tolerance_pct_point,
    )
    _append_profile_artifact_consistency_rows(
        rows,
        expected=expected,
        artifact_label="paper1_driver_decomposition_table.csv",
        artifact=_read_csv_if_exists(figures_root / "paper1_driver_decomposition_table.csv"),
        selector_column="diagnostic",
        selector_value="Declared asymmetric baseline",
        tolerance_pct_point=tolerance_pct_point,
    )
    _append_policy_cost_consistency_rows(
        rows,
        portfolio_summary=_read_csv_if_exists(planning_root / "portfolio_summary.csv"),
        artifact=_read_csv_if_exists(figures_root / "paper1_policy_cost_decomposition_table.csv"),
    )
    return pd.DataFrame(rows).sort_values(["artifact_label", "scenario_name", "pathway"]).reset_index(drop=True)


def _append_profile_artifact_consistency_rows(
    rows: list[dict[str, object]],
    *,
    expected: pd.DataFrame,
    artifact_label: str,
    artifact: pd.DataFrame,
    selector_column: str,
    selector_value: str,
    tolerance_pct_point: float,
) -> None:
    """Compare compact manuscript profile cells such as ``P 89.3 / H 10.7``."""

    if artifact.empty:
        return
    scenario_columns = {
        "baseline_region_case": "baseline_region",
        "high_supply_case": "high_supply",
        "policy_support_case": "policy_support",
    }
    required_columns = {selector_column, *scenario_columns.values()}
    if not required_columns.issubset(artifact.columns):
        for _, ref_row in expected.iterrows():
            rows.append(
                {
                    "artifact_label": artifact_label,
                    "scenario_name": ref_row["scenario_name"],
                    "pathway": ref_row["pathway"],
                    "expected_share_pct": ref_row["expected_share_pct"],
                    "observed_share_pct": pd.NA,
                    "absolute_difference_pct_point": pd.NA,
                    "consistency_status": "missing_artifact_or_columns",
                    "consistency_note": f"Artifact lacks required compact-profile columns: {sorted(required_columns)}.",
                }
            )
        return
    matches = artifact[artifact[selector_column].astype(str).eq(selector_value)]
    if matches.empty:
        for _, ref_row in expected.iterrows():
            rows.append(
                {
                    "artifact_label": artifact_label,
                    "scenario_name": ref_row["scenario_name"],
                    "pathway": ref_row["pathway"],
                    "expected_share_pct": ref_row["expected_share_pct"],
                    "observed_share_pct": pd.NA,
                    "absolute_difference_pct_point": pd.NA,
                    "consistency_status": "missing_artifact_or_columns",
                    "consistency_note": f"Artifact lacks the synchronized row '{selector_value}'.",
                }
            )
        return
    profile_row = matches.iloc[0]
    for _, ref_row in expected.iterrows():
        scenario_name = str(ref_row["scenario_name"])
        pathway = str(ref_row["pathway"]).lower()
        if pathway not in {"pyrolysis", "htc", "ad"} or scenario_name not in scenario_columns:
            continue
        observed_share = _extract_profile_share_pct(profile_row.get(scenario_columns[scenario_name]), pathway)
        expected_share = _optional_float(ref_row.get("expected_share_pct"))
        diff = (
            abs(float(observed_share) - float(expected_share))
            if pd.notna(observed_share) and pd.notna(expected_share)
            else pd.NA
        )
        status = "pass" if pd.notna(diff) and float(diff) <= tolerance_pct_point else "fail"
        rows.append(
            {
                "artifact_label": artifact_label,
                "scenario_name": scenario_name,
                "pathway": pathway,
                "expected_share_pct": round(float(expected_share), 3)
                if pd.notna(expected_share)
                else pd.NA,
                "observed_share_pct": round(float(observed_share), 3)
                if pd.notna(observed_share)
                else pd.NA,
                "absolute_difference_pct_point": round(float(diff), 3)
                if pd.notna(diff)
                else pd.NA,
                "consistency_status": status,
                "consistency_note": (
                    f"Compact profile differs by more than {tolerance_pct_point:.1f} percentage point."
                    if status == "fail"
                    else "Compact profile is aligned with portfolio_allocations.csv."
                ),
            }
        )


def _append_policy_cost_consistency_rows(
    rows: list[dict[str, object]],
    *,
    portfolio_summary: pd.DataFrame,
    artifact: pd.DataFrame,
    tolerance_musd: float = 0.01,
) -> None:
    """Compare manuscript cost table against canonical portfolio_summary cost."""

    if portfolio_summary.empty or artifact.empty:
        return
    if not {"scenario_name", "portfolio_cost_objective"}.issubset(portfolio_summary.columns):
        return
    if not {"scenario", "final_net_cost_musd_per_year"}.issubset(artifact.columns):
        for scenario_name in portfolio_summary["scenario_name"].dropna().astype(str).unique():
            rows.append(
                {
                    "artifact_label": "paper1_policy_cost_decomposition_table.csv",
                    "scenario_name": scenario_name,
                    "pathway": "portfolio_cost_musd_per_year",
                    "expected_share_pct": pd.NA,
                    "observed_share_pct": pd.NA,
                    "absolute_difference_pct_point": pd.NA,
                    "consistency_status": "missing_artifact_or_columns",
                    "consistency_note": "Cost artifact lacks final_net_cost_musd_per_year.",
                }
            )
        return
    display_to_scenario = {
        "baseline-region": "baseline_region_case",
        "high-supply": "high_supply_case",
        "policy-support": "policy_support_case",
    }
    expected_cost = portfolio_summary.copy()
    expected_cost["expected_cost_musd"] = _numeric_column(expected_cost, "portfolio_cost_objective") / 1_000_000.0
    observed = artifact.copy()
    observed["scenario_name"] = observed["scenario"].astype(str).map(display_to_scenario).fillna(observed["scenario"].astype(str))
    observed["observed_cost_musd"] = _numeric_column(observed, "final_net_cost_musd_per_year")
    merged = expected_cost[["scenario_name", "expected_cost_musd"]].merge(
        observed[["scenario_name", "observed_cost_musd"]],
        on="scenario_name",
        how="left",
    )
    for _, row in merged.iterrows():
        expected_value = _optional_float(row.get("expected_cost_musd"))
        observed_value = _optional_float(row.get("observed_cost_musd"))
        diff = (
            abs(float(observed_value) - float(expected_value))
            if pd.notna(observed_value) and pd.notna(expected_value)
            else pd.NA
        )
        status = "pass" if pd.notna(diff) and float(diff) <= tolerance_musd else "fail"
        rows.append(
            {
                "artifact_label": "paper1_policy_cost_decomposition_table.csv",
                "scenario_name": row["scenario_name"],
                "pathway": "portfolio_cost_musd_per_year",
                "expected_share_pct": round(float(expected_value), 3)
                if pd.notna(expected_value)
                else pd.NA,
                "observed_share_pct": round(float(observed_value), 3)
                if pd.notna(observed_value)
                else pd.NA,
                "absolute_difference_pct_point": round(float(diff), 3)
                if pd.notna(diff)
                else pd.NA,
                "consistency_status": status,
                "consistency_note": (
                    f"Final cost differs by more than {tolerance_musd:.2f} MUSD y^-1."
                    if status == "fail"
                    else "Final cost is aligned with portfolio_summary.csv."
                ),
            }
        )


def _extract_profile_share_pct(value: object, pathway: str) -> float:
    text = str(value or "").strip()
    if not text or text.lower() == "nan" or text == "--":
        return 0.0
    lowered = text.lower()
    aliases = {
        "pyrolysis": ("p", "pyrolysis"),
        "htc": ("h", "htc"),
        "ad": ("ad",),
    }
    for canonical, pathway_aliases in aliases.items():
        if any(f"{alias}-only" in lowered for alias in pathway_aliases):
            return 100.0 if pathway == canonical else 0.0
    patterns = {
        "pyrolysis": r"(?:^|[^a-z0-9])(?:p|pyrolysis)\s*([0-9]+(?:\.[0-9]+)?)",
        "htc": r"(?:^|[^a-z0-9])(?:h|htc)\s*([0-9]+(?:\.[0-9]+)?)",
        "ad": r"(?:^|[^a-z0-9])(?:ad)\s*([0-9]+(?:\.[0-9]+)?)",
    }
    match = re.search(patterns.get(pathway, ""), lowered)
    return float(match.group(1)) if match else 0.0


def build_artifact_inventory(
    summary_paths: dict[str, Path],
    operation_dir: Path,
    planning_dir: Path,
    scenario_dir: Path,
    benchmark_dir: Path,
) -> pd.DataFrame:
    operation_freshness = _build_operation_artifact_freshness(
        operation_dir=operation_dir,
        planning_dir=planning_dir,
        scenario_dir=scenario_dir,
    )
    records: list[dict[str, object]] = []
    for label, path in summary_paths.items():
        records.append(
            {
                "artifact_group": "ml_summary",
                "artifact_label": label,
                "path": str(path),
                "exists": path.exists(),
                "freshness_status": "not_checked",
                "freshness_note": "",
            }
        )
    for file_name in [
        "main_results_table.csv",
        "main_results_table_manifest.json",
        "recommendation_confidence_summary.csv",
        "pathway_summary.csv",
        "portfolio_allocations.csv",
        "scenario_summary.csv",
        "optimization_diagnostics.csv",
        "run_config.json",
    ]:
        path = planning_dir / file_name
        records.append(
            {
                "artifact_group": "planning",
                "artifact_label": file_name,
                "path": str(path),
                "exists": path.exists(),
                "freshness_status": "source_truth",
                "freshness_note": "",
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
                "freshness_status": "source_truth",
                "freshness_note": "",
            }
        )
    for file_name in [
        "benchmark_summary.csv",
        "benchmark_allocations.csv",
        "benchmark_scenario_summary.csv",
        "benchmark_shift_summary.csv",
        "benchmark_diagnostics.csv",
        "benchmark_bootstrap_shift_samples.csv",
        "benchmark_statistical_summary.csv",
        "run_config.json",
    ]:
        path = benchmark_dir / file_name
        records.append(
            {
                "artifact_group": "benchmark",
                "artifact_label": file_name,
                "path": str(path),
                "exists": path.exists(),
                "freshness_status": "source_truth",
                "freshness_note": "",
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
                "freshness_status": operation_freshness["freshness_status"],
                "freshness_note": operation_freshness["freshness_note"],
            }
        )
    return pd.DataFrame(records)


def _build_operation_artifact_freshness(
    *,
    operation_dir: Path,
    planning_dir: Path,
    scenario_dir: Path,
) -> dict[str, str]:
    operation_run_config = _read_json_if_exists(operation_dir / "run_config.json")
    planning_run_config = _read_json_if_exists(planning_dir / "run_config.json")
    scenario_run_config = _read_json_if_exists(scenario_dir / "run_config.json")

    operation_timestamp = parse_manifest_timestamp(operation_run_config)
    planning_timestamp = parse_manifest_timestamp(planning_run_config)
    scenario_timestamp = parse_manifest_timestamp(scenario_run_config)
    required_timestamp = _max_timestamp(planning_timestamp, scenario_timestamp)

    source_planning_timestamp = _parse_timestamp_string(operation_run_config.get("source_planning_generated_at_utc"))
    source_scenario_timestamp = _parse_timestamp_string(operation_run_config.get("source_scenario_generated_at_utc"))

    if not operation_run_config:
        return {
            "freshness_status": "missing_run_config",
            "freshness_note": "Operation comparison run_config.json is missing, so artifact freshness cannot be verified.",
        }

    if planning_timestamp is None or scenario_timestamp is None:
        return {
            "freshness_status": "unknown",
            "freshness_note": "Planning/scenario manifest timestamps are unavailable, so operation freshness cannot be verified.",
        }

    if source_planning_timestamp and source_planning_timestamp + timedelta(seconds=1) < planning_timestamp:
        return {
            "freshness_status": "stale",
            "freshness_note": "Operation comparison artifacts record an older planning source manifest than the current planning outputs.",
        }

    if source_scenario_timestamp and source_scenario_timestamp + timedelta(seconds=1) < scenario_timestamp:
        return {
            "freshness_status": "stale",
            "freshness_note": "Operation comparison artifacts record an older scenario source manifest than the current scenario outputs.",
        }

    if operation_timestamp is None or required_timestamp is None:
        return {
            "freshness_status": "unknown",
            "freshness_note": "Operation comparison timestamp is unavailable, so freshness cannot be confirmed.",
        }

    if operation_timestamp + timedelta(seconds=1) < required_timestamp:
        return {
            "freshness_status": "stale",
            "freshness_note": "Operation comparison artifacts are older than the current planning/scenario source outputs and should be regenerated.",
        }

    return {
        "freshness_status": "current",
        "freshness_note": "Operation comparison artifacts are aligned with the current planning/scenario manifests.",
    }


def _parse_timestamp_string(value: object):
    if not value:
        return None
    try:
        return pd.Timestamp(str(value)).to_pydatetime()
    except Exception:
        return None


def _max_timestamp(*values):
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return max(valid_values)


def build_audit_manifest(
    summary_paths: dict[str, Path],
    operation_dir: Path,
    planning_dir: Path,
    scenario_dir: Path,
    benchmark_dir: Path,
) -> dict[str, object]:
    return {
        "expected_models": list(MODEL_KEYS),
        "expected_datasets": list(DATASET_KEYS),
        "expected_targets": list(TARGET_COLUMNS),
        "ml_summary_paths": {key: str(value) for key, value in summary_paths.items()},
        "planning_dir": str(planning_dir),
        "scenario_dir": str(scenario_dir),
        "operation_comparison_dir": str(operation_dir),
        "benchmark_dir": str(benchmark_dir),
        "confirmation_rules": [
            "strict_group is the main-table benchmark evidence tier",
            "leave_study_out is the stronger cross-study stress test",
            "planning_claim_flag_table is the manuscript-facing planning claim inventory",
            "recommendation_confidence_summary grades how strong a pathway recommendation remains after evidence and stress screening",
            "benchmark_claim_summary converts ablation outputs into scenario-level necessity evidence for the innovation claims",
            "benchmark_manuscript_sentences provides manuscript-safe wording for how benchmark variants alter recommendations",
            "benchmark_statistical_summary adds bootstrap-backed uncertainty intervals for benchmark deltas when repeated runs are enabled",
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


def _classify_benchmark_necessity_tier(row: pd.Series) -> str:
    pathway_shift = str(row.get("portfolio_pathway_shift", "") or "")
    case_shift = str(row.get("portfolio_case_shift", "") or "")
    score_delta = _abs_optional_float(row.get("delta_portfolio_score_mass"))
    carbon_delta = _abs_optional_float(row.get("delta_portfolio_carbon_load_kgco2e"))
    coverage_delta = _abs_optional_float(row.get("delta_scenario_feed_coverage_ratio"))
    has_material_metric_shift = score_delta >= 1.0 or carbon_delta >= 1.0 or coverage_delta >= 0.01

    if pathway_shift == "changed":
        return "supports_core_innovation"
    if case_shift == "changed" and has_material_metric_shift:
        return "supports_secondary_innovation"
    if has_material_metric_shift:
        return "supports_secondary_innovation"
    return "limited_effect"


def _build_benchmark_necessity_note(row: pd.Series) -> str:
    action_phrase, innovation_label = _benchmark_variant_labels(row.get("benchmark_variant"))
    tier = str(row.get("necessity_tier", "") or "")
    baseline_pathways = str(row.get("baseline_selected_pathways", "") or "baseline portfolio")
    variant_pathways = str(row.get("variant_selected_pathways", "") or "variant portfolio")
    significance_clause = _benchmark_significance_clause(row, prefix=" ")

    if tier == "supports_core_innovation":
        return (
            f"{action_phrase} changed the selected pathway set from {baseline_pathways} to {variant_pathways}, "
            f"so the {innovation_label} is materially necessary for the exported recommendation.{significance_clause}"
        )
    if tier == "supports_secondary_innovation":
        return (
            f"{action_phrase} did not force a full pathway replacement in every case, but it still altered "
            f"the preferred portfolio case or portfolio metrics, so the {innovation_label} remains decision-relevant.{significance_clause}"
        )
    return (
        f"{action_phrase} produced only limited movement in the exported portfolio, so the {innovation_label} "
        f"should be described as a bounded refinement rather than the sole driver of the result.{significance_clause}"
    )


def _build_benchmark_sentence(row: pd.Series) -> str:
    action_phrase, innovation_label = _benchmark_variant_labels(row.get("benchmark_variant"))
    scenario_name = str(row.get("scenario_name", "the audited scenario"))
    tier = str(row.get("necessity_tier", "") or "")
    baseline_pathways = str(row.get("baseline_selected_pathways", "") or "the baseline pathway set")
    variant_pathways = str(row.get("variant_selected_pathways", "") or "the variant pathway set")
    score_delta = _format_signed_delta(row.get("delta_portfolio_score_mass"), precision=2)
    carbon_delta = _format_signed_delta(row.get("delta_portfolio_carbon_load_kgco2e"), precision=2)
    coverage_delta = _format_signed_delta(row.get("delta_scenario_feed_coverage_ratio"), precision=3)
    significance_clause = _benchmark_significance_clause(row, prefix=" ")

    if tier == "supports_core_innovation":
        return (
            f"In {scenario_name}, {action_phrase.lower()} changed the selected pathways from "
            f"{baseline_pathways} to {variant_pathways}, showing that the {innovation_label} materially shapes "
            f"the recommendation (score delta {score_delta}, carbon delta {carbon_delta}, coverage delta {coverage_delta}).{significance_clause}"
        )
    if tier == "supports_secondary_innovation":
        return (
            f"In {scenario_name}, {action_phrase.lower()} did not always replace the full pathway set, but it "
            f"still changed the preferred portfolio case or portfolio metrics (score delta {score_delta}, carbon "
            f"delta {carbon_delta}, coverage delta {coverage_delta}), so the {innovation_label} is not cosmetic.{significance_clause}"
        )
    return (
        f"In {scenario_name}, {action_phrase.lower()} produced only limited movement relative to the exported "
        f"baseline portfolio, so the {innovation_label} should be written as a bounded refinement.{significance_clause}"
    )


def _build_benchmark_aggregate_sentence(
    *,
    variant_key: object,
    changed_pathway_count: int,
    changed_case_count: int,
    strong_support_count: int,
    secondary_support_count: int,
    scenario_count: int,
) -> str:
    action_phrase, innovation_label = _benchmark_variant_labels(variant_key)
    if scenario_count <= 0:
        return (
            f"{action_phrase} did not yield any auditable scenarios, so the necessity of the {innovation_label} "
            "cannot be summarized yet."
        )
    if strong_support_count > 0:
        return (
            f"{action_phrase} changed selected pathways in {changed_pathway_count}/{scenario_count} scenarios and "
            f"changed the top portfolio case in {changed_case_count}/{scenario_count} scenarios, supporting the "
            f"necessity of the {innovation_label}."
        )
    if secondary_support_count > 0:
        return (
            f"{action_phrase} did not broadly replace pathways, but it still produced material portfolio movement "
            f"in {secondary_support_count}/{scenario_count} scenarios and changed the top portfolio case in "
            f"{changed_case_count}/{scenario_count} scenarios, indicating a secondary yet non-cosmetic role for "
            f"the {innovation_label}."
        )
    return (
        f"{action_phrase} left both pathways and top portfolio cases largely unchanged across the audited "
        f"scenarios, so the {innovation_label} should be written as a bounded refinement rather than a dominant driver."
    )


def _benchmark_significance_clause(row: pd.Series, *, prefix: str = " ") -> str:
    significance_tier = str(row.get("effect_significance_tier", "") or "").strip()
    replicate_count = _optional_float(row.get("bootstrap_replicate_count"))
    pathway_shift_rate = _optional_float(row.get("pathway_shift_rate"))
    pathway_shift_ci_lower = _optional_float(row.get("pathway_shift_rate_ci_lower"))
    pathway_shift_ci_upper = _optional_float(row.get("pathway_shift_rate_ci_upper"))
    score_p_value = _optional_float(row.get("delta_portfolio_score_mass_empirical_p_value"))
    if not significance_tier:
        return ""
    if significance_tier == "highly_consistent":
        detail = ""
        if pd.notna(pathway_shift_rate):
            detail = f" pathway-shift rate {float(pathway_shift_rate) * 100:.0f}%"
            if pd.notna(pathway_shift_ci_lower) and pd.notna(pathway_shift_ci_upper):
                detail += (
                    f" (95% interval {float(pathway_shift_ci_lower) * 100:.0f}--"
                    f"{float(pathway_shift_ci_upper) * 100:.0f}%)"
                )
        return f"{prefix}Bootstrap repeats indicate a highly consistent effect{detail}."
    if significance_tier == "directionally_consistent":
        if pd.notna(score_p_value):
            return (
                f"{prefix}Bootstrap repeats indicate a directionally consistent effect "
                f"(empirical p={float(score_p_value):.3f} for the score shift)."
            )
        return f"{prefix}Bootstrap repeats indicate a directionally consistent effect."
    if significance_tier == "suggestive":
        if pd.notna(replicate_count):
            detail = ""
            if pd.notna(score_p_value):
                detail = f"; empirical p={float(score_p_value):.3f}"
            return (
                f"{prefix}Bootstrap repeats remain suggestive rather than fully stable "
                f"({int(float(replicate_count))} replicates{detail})."
            )
        return f"{prefix}Bootstrap repeats remain suggestive rather than fully stable."
    return f"{prefix}Bootstrap repeats suggest the effect is unstable and should be written conservatively."


def _benchmark_variant_labels(variant_key: object) -> tuple[str, str]:
    mapping = {
        "no_evidence_penalty": ("Removing evidence penalties", "evidence-aware design"),
        "no_robustness_penalty": ("Removing robustness penalties", "robustness-aware design"),
        "no_carbon_constraint": ("Relaxing the carbon constraint", "carbon guardrail"),
        "classic_multiobjective_optimizer": (
            "Replacing the method with a classic multi-objective optimizer",
            "evidence-aware and robustness-aware design",
        ),
        "greedy_weighted_score_heuristic": (
            "Replacing the optimizer with a greedy weighted-score heuristic",
            "portfolio optimization layer",
        ),
        "ranking_only_unconstrained": (
            "Removing portfolio caps and diversity constraints",
            "portfolio-constraint design",
        ),
        "baseline_evidence_aware": ("Using the evidence-aware baseline", "evidence-aware design"),
    }
    return mapping.get(str(variant_key), (f"Applying benchmark variant '{variant_key}'", "design choice"))


def _abs_optional_float(value: object) -> float:
    numeric = _optional_float(value)
    if pd.isna(numeric):
        return 0.0
    return abs(float(numeric))


def _format_signed_delta(value: object, *, precision: int) -> str:
    numeric = _optional_float(value)
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):+.{precision}f}"


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


def _read_ml_summary_frame(path: Path, label: str) -> pd.DataFrame:
    primary = _read_csv_if_exists(path)
    if label != "leave_study_out":
        return primary
    nested_primary = _read_csv_if_exists(
        path.parent / "leave_study_out" / "traditional_ml_suite_summary_leave_study_out.csv"
    )
    supplement = _read_csv_if_exists(BENCHMARK_OUTPUTS_DIR / "htc_model_compare_lso" / "traditional_ml_suite_summary_leave_study_out.csv")
    frames = [frame for frame in (supplement, primary, nested_primary) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    dedupe_columns = [column for column in ["dataset_key", "target_column", "model_key"] if column in combined.columns]
    if dedupe_columns:
        combined = combined.drop_duplicates(subset=dedupe_columns, keep="first")
    return combined.reset_index(drop=True)


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
        # Manifest files may contain both the model-selection benchmark metric
        # and a later refit/holdout metric. Manuscript validation tables report
        # benchmark_* metrics when present, so audit flags must use the same
        # evidence source instead of mixing benchmark and refit values.
        for target, candidates in {
            "selected_test_r2": ("benchmark_test_r2", "reporting_test_r2", "selected_test_r2", "test_r2", "refit_test_r2"),
            "selected_test_rmse": (
                "benchmark_test_rmse",
                "reporting_test_rmse",
                "selected_test_rmse",
                "test_rmse",
                "refit_test_rmse",
            ),
            "selected_test_mae": (
                "benchmark_test_mae",
                "reporting_test_mae",
                "selected_test_mae",
                "test_mae",
                "refit_test_mae",
            ),
        }.items():
            manifest[target] = _coalesce_numeric_columns(manifest, candidates)
        manifest = manifest.reset_index(drop=True)
        if frame.empty or not {"dataset_key", "target_column"}.issubset(frame.columns):
            return manifest
        manifest_pairs = set(
            zip(
                manifest.get("dataset_key", pd.Series(dtype="object")).astype(str),
                manifest.get("target_column", pd.Series(dtype="object")).astype(str),
            )
        )
        frame_pairs = list(
            zip(
                frame["dataset_key"].astype(str),
                frame["target_column"].astype(str),
            )
        )
        missing_frame = frame[[pair not in manifest_pairs for pair in frame_pairs]].copy()
        summary_selected = _selected_from_summary_frame(missing_frame)
        if summary_selected is None or summary_selected.empty:
            return manifest
        return pd.concat([manifest, summary_selected], ignore_index=True, sort=False)

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


def _coalesce_numeric_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        result = result.where(result.notna(), values)
    return result


def _selected_from_summary_frame(frame: pd.DataFrame) -> pd.DataFrame | None:
    if frame.empty or "model_key" not in frame.columns:
        return None

    selected = pd.DataFrame()
    if "is_selected_model" in frame.columns:
        selected_mask = frame["is_selected_model"].map(_coerce_bool_flag).fillna(False)
        selected = frame[selected_mask].copy()

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
            dataset_key = str(working["dataset_key"].iloc[0])
            working["_validation_r2_sort"] = pd.to_numeric(
                working.get("validation_r2", pd.Series([pd.NA] * len(working), index=working.index)),
                errors="coerce",
            ).fillna(float("-inf"))
            working["_model_priority_sort"] = working["model_key"].map(
                lambda value: _audit_model_priority_rank(dataset_key=dataset_key, model_key=str(value))
            )
            working["_validation_rmse_sort"] = pd.to_numeric(
                working.get("validation_rmse", pd.Series([pd.NA] * len(working), index=working.index)),
                errors="coerce",
            ).fillna(float("inf"))
            working["_validation_mae_sort"] = pd.to_numeric(
                working.get("validation_mae", pd.Series([pd.NA] * len(working), index=working.index)),
                errors="coerce",
            ).fillna(float("inf"))
            working = working.sort_values(
                ["_model_priority_sort", "_validation_r2_sort", "_validation_rmse_sort", "_validation_mae_sort", "model_key"],
                ascending=[True, False, True, True, True],
            ).head(1)
            selected_groups.append(
                working.drop(
                    columns=["_model_priority_sort", "_validation_r2_sort", "_validation_rmse_sort", "_validation_mae_sort"],
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


def _derive_feedstock_hhv_from_ultimate_analysis(row: pd.Series) -> float | object:
    required = {
        "feedstock_carbon_pct": "carbon",
        "feedstock_hydrogen_pct": "hydrogen",
        "feedstock_nitrogen_pct": "nitrogen",
        "feedstock_oxygen_pct": "oxygen",
        "feedstock_ash_pct": "ash",
    }
    values: dict[str, float] = {}
    for column, key in required.items():
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.isna(value):
            return pd.NA
        values[key] = float(value)
    hhv = (
        0.3491 * values["carbon"]
        + 1.1783 * values["hydrogen"]
        - 0.1034 * values["oxygen"]
        - 0.0151 * values["nitrogen"]
        - 0.0211 * values["ash"]
    )
    return float(hhv) if np.isfinite(hhv) and hhv > 0.0 else pd.NA


def _resolve_targeted_ablation_dir_for_audit(benchmark_dir: Path) -> Path:
    candidates = [
        benchmark_dir / "targeted_planning_ablations",
        benchmark_dir.parent / "targeted_planning_ablations",
        BENCHMARK_OUTPUTS_DIR / "targeted_planning_ablations",
    ]
    for candidate in candidates:
        if (candidate / "portfolio_cap_diagnostics.csv").exists():
            return candidate
    return candidates[0]


def _resolve_targeted_ablation_summary_dir_for_audit(benchmark_dir: Path) -> Path:
    candidates = [
        benchmark_dir / "targeted_planning_ablations",
        benchmark_dir.parent / "targeted_planning_ablations",
        BENCHMARK_OUTPUTS_DIR / "targeted_planning_ablations",
    ]
    for candidate in candidates:
        if (candidate / "targeted_planning_ablations_summary.csv").exists():
            return candidate
    return candidates[0]


def _planning_config_from_run_config(run_config: dict[str, object] | None) -> PlanningConfig:
    if not run_config:
        return PlanningConfig(enable_pareto_export=False)
    raw_config = run_config.get("planning_config", {})
    if not isinstance(raw_config, dict):
        return PlanningConfig(enable_pareto_export=False)
    weight_payload = raw_config.get("objective_weight_system", {})
    weights = weight_payload.get("weights", {}) if isinstance(weight_payload, dict) else {}
    preset = str(raw_config.get("objective_weight_preset", "balanced_cleaner_production"))
    objective_weight_system = get_objective_weight_system(
        preset_name=preset,
        energy=weights.get("energy") if isinstance(weights, dict) else None,
        environment=weights.get("environment") if isinstance(weights, dict) else None,
        cost=weights.get("cost") if isinstance(weights, dict) else None,
    )
    return PlanningConfig(
        objective_weight_preset=preset,
        objective_weight_system=objective_weight_system,
        top_k_per_scenario=int(raw_config.get("top_k_per_scenario", 5) or 5),
        max_portfolio_candidates=int(raw_config.get("max_portfolio_candidates", 3) or 3),
        max_candidate_share=float(raw_config.get("max_candidate_share", 0.45) or 0.45),
        max_subtype_share=float(raw_config.get("max_subtype_share", 0.60) or 0.60),
        min_distinct_subtypes=int(raw_config.get("min_distinct_subtypes", 2) or 2),
        deployable_capacity_fraction=float(raw_config.get("deployable_capacity_fraction", 0.85) or 0.85),
        robustness_factor=float(raw_config.get("robustness_factor", 0.35) or 0.35),
        carbon_budget_factor=float(raw_config.get("carbon_budget_factor", 1.0) or 1.0),
        constraint_relaxation_ratio=float(raw_config.get("constraint_relaxation_ratio", 1.0) or 1.0),
        subtype_relaxation_ratio=float(raw_config.get("subtype_relaxation_ratio", 1.0) or 1.0),
        enforce_candidate_cap=_coerce_bool_flag(raw_config.get("enforce_candidate_cap", True)),
        enforce_subtype_cap=_coerce_bool_flag(raw_config.get("enforce_subtype_cap", True)),
        enforce_max_selected=_coerce_bool_flag(raw_config.get("enforce_max_selected", True)),
        enforce_min_distinct_subtypes=_coerce_bool_flag(raw_config.get("enforce_min_distinct_subtypes", True)),
        scenario_metric_variance_scale=float(raw_config.get("scenario_metric_variance_scale", 1.0) or 1.0),
        primary_optimization_pathways=tuple(raw_config.get("primary_optimization_pathways", ("pyrolysis", "htc"))),
        scenario_external_evidence_table_path=raw_config.get("scenario_external_evidence_table_path") or None,
        optimization_method=str(raw_config.get("optimization_method", "auto") or "auto"),
        pyomo_solver_preference=str(raw_config.get("pyomo_solver_preference", "auto") or "auto"),
        pareto_point_count=0,
        enable_pareto_export=False,
        uncertainty_penalty_mode=str(raw_config.get("uncertainty_penalty_mode", "prefer_interval_mean")),
        evidence_utility_factor=float(raw_config.get("evidence_utility_factor", 0.15) or 0.15),
        allow_surrogate_fallback=_coerce_bool_flag(raw_config.get("allow_surrogate_fallback", True)),
        htc_model_priority=tuple(raw_config.get("htc_model_priority", PlanningConfig().htc_model_priority)),
        partial_surrogate_weight=float(raw_config.get("partial_surrogate_weight", PlanningConfig().partial_surrogate_weight)),
        static_fallback_weight=float(raw_config.get("static_fallback_weight", PlanningConfig().static_fallback_weight)),
        unsupported_pathway_weight=float(raw_config.get("unsupported_pathway_weight", PlanningConfig().unsupported_pathway_weight)),
        partial_surrogate_uncertainty_multiplier=float(
            raw_config.get(
                "partial_surrogate_uncertainty_multiplier",
                PlanningConfig().partial_surrogate_uncertainty_multiplier,
            )
        ),
        static_fallback_uncertainty_multiplier=float(
            raw_config.get(
                "static_fallback_uncertainty_multiplier",
                PlanningConfig().static_fallback_uncertainty_multiplier,
            )
        ),
        unsupported_pathway_uncertainty_multiplier=float(
            raw_config.get(
                "unsupported_pathway_uncertainty_multiplier",
                PlanningConfig().unsupported_pathway_uncertainty_multiplier,
            )
        ),
        partial_surrogate_information_premium_usd_per_ton=float(
            raw_config.get(
                "partial_surrogate_information_premium_usd_per_ton",
                PlanningConfig().partial_surrogate_information_premium_usd_per_ton,
            )
        ),
        static_fallback_information_premium_usd_per_ton=float(
            raw_config.get(
                "static_fallback_information_premium_usd_per_ton",
                PlanningConfig().static_fallback_information_premium_usd_per_ton,
            )
        ),
        unsupported_pathway_information_premium_usd_per_ton=float(
            raw_config.get(
                "unsupported_pathway_information_premium_usd_per_ton",
                PlanningConfig().unsupported_pathway_information_premium_usd_per_ton,
            )
        ),
        minimum_surrogate_artifact_test_r2=raw_config.get("minimum_surrogate_artifact_test_r2"),
    )


def _classify_hhv_dominance(
    *,
    max_abs_change: object,
    pathway_changed: bool,
    case_changed: bool,
) -> str:
    if pd.isna(max_abs_change):
        return "not_evaluated"
    if pathway_changed or float(max_abs_change) > 5.0:
        return "potentially_pathway_dominant"
    if float(max_abs_change) <= 1.0:
        return "not_pathway_dominant_but_case_sensitive" if case_changed else "not_pathway_dominant"
    return "limited_share_sensitivity_not_pathway_dominant"


def _build_hhv_dominance_sentence(
    *,
    scenario_name: str,
    affected_share: float,
    max_abs_change: object,
    conclusion: str,
) -> str:
    if pd.isna(max_abs_change):
        return f"{scenario_name}: HHV-imputation dominance was not evaluated."
    if conclusion == "potentially_pathway_dominant":
        return (
            f"{scenario_name}: HHV perturbation changes pathway shares by up to {float(max_abs_change):.1f} pp "
            "or changes selected pathways, so pathway-level conclusions require explicit HHV-boundary qualification."
        )
    case_note = "although selected case IDs can switch" if "case_sensitive" in conclusion else "with stable selected cases"
    return (
        f"{scenario_name}: {affected_share:.1f}% of selected allocation uses composition-derived feedstock HHV, "
        f"but ±10% replanning changes pathway shares by at most {float(max_abs_change):.1f} pp, {case_note}; "
        "HHV imputation is therefore not a pathway-dominant driver under the tested stresses."
    )


_SURROGATE_RANGE_FEATURES = (
    "feedstock_carbon_pct",
    "feedstock_hydrogen_pct",
    "feedstock_nitrogen_pct",
    "feedstock_oxygen_pct",
    "feedstock_moisture_pct",
    "feedstock_volatile_matter_pct",
    "feedstock_fixed_carbon_pct",
    "feedstock_ash_pct",
    "feedstock_hhv_mj_per_kg",
    "process_temperature_c",
    "residence_time_min",
    "heating_rate_c_per_min",
    "blend_manure_ratio",
    "blend_wet_waste_ratio",
)


def _build_training_range_lookup(training: pd.DataFrame) -> dict[str, dict[str, tuple[float, float]]]:
    if training.empty or "pathway" not in training.columns:
        return {}
    lookup: dict[str, dict[str, tuple[float, float]]] = {}
    for pathway, subset in training.groupby(training["pathway"].astype(str).str.lower(), dropna=False):
        ranges: dict[str, tuple[float, float]] = {}
        for column in _SURROGATE_RANGE_FEATURES:
            if column not in subset.columns:
                continue
            values = pd.to_numeric(subset[column], errors="coerce").dropna()
            if values.empty:
                continue
            ranges[column] = (float(values.min()), float(values.max()))
        lookup[str(pathway)] = ranges
    return lookup


def _leave_study_out_rows_for_pathway(flags: pd.DataFrame, pathway: str) -> pd.DataFrame:
    if flags.empty or not {"summary_label", "dataset_key"}.issubset(flags.columns):
        return pd.DataFrame()
    dataset_prefix = "pyrolysis" if str(pathway).lower().startswith("pyro") else str(pathway).lower()
    return flags[
        flags["summary_label"].astype(str).eq("leave_study_out")
        & flags["dataset_key"].astype(str).str.lower().str.contains(dataset_prefix, na=False)
    ].copy()


def _weakest_claim_status(statuses: pd.Series) -> str:
    order = {"unsupported": 0, "weak": 1, "supportive": 2}
    values = [str(value).lower() for value in statuses.dropna().tolist()]
    if not values:
        return "not_evaluated"
    return min(values, key=lambda value: order.get(value, 99))


def _selected_range_summary(
    selected: pd.DataFrame,
    training_ranges: dict[str, tuple[float, float]],
) -> dict[str, object]:
    if not training_ranges:
        return {
            "feature_range_status": "training_range_unavailable",
            "within_training_range_all_features": False,
            "out_of_range_feature_count": pd.NA,
            "out_of_range_features": "",
        }
    out_of_range: list[str] = []
    evaluated = 0
    for column, (minimum, maximum) in training_ranges.items():
        if column not in selected.columns:
            continue
        values = pd.to_numeric(selected[column], errors="coerce").dropna()
        if values.empty:
            continue
        evaluated += 1
        tolerance = max(abs(minimum), abs(maximum), 1.0) * 1e-9
        if float(values.min()) < minimum - tolerance or float(values.max()) > maximum + tolerance:
            out_of_range.append(
                f"{column}[{float(values.min()):.3g},{float(values.max()):.3g}] "
                f"vs train[{minimum:.3g},{maximum:.3g}]"
            )
    if evaluated == 0:
        return {
            "feature_range_status": "selected_features_unavailable",
            "within_training_range_all_features": False,
            "out_of_range_feature_count": pd.NA,
            "out_of_range_features": "",
        }
    return {
        "feature_range_status": "evaluated",
        "within_training_range_all_features": len(out_of_range) == 0,
        "out_of_range_feature_count": int(len(out_of_range)),
        "out_of_range_features": "; ".join(out_of_range),
    }


def _classify_surrogate_extrapolation_ceiling(
    *,
    weakest_status: str,
    all_in_range: bool,
    missing_range: bool,
) -> str:
    if missing_range or weakest_status == "not_evaluated":
        return "screening_only_validation_incomplete"
    if weakest_status == "unsupported":
        return "screening_only_external_validity_not_established"
    if weakest_status == "weak":
        return "evidence_gated_screening_only"
    if not all_in_range:
        return "interpolation_range_flag_screening_only"
    return "conditional_interpolation_support_screening_only"


def _build_surrogate_extrapolation_sentence(
    *,
    pathway: str,
    weakest_status: str,
    min_r2: object,
    ceiling: str,
    out_of_range_features: str,
) -> str:
    r2_text = "--" if pd.isna(min_r2) else f"{float(min_r2):.2f}"
    range_text = f"; selected features outside training ranges: {out_of_range_features}" if out_of_range_features else ""
    return (
        f"{pathway}: leave-study-out weakest target is {weakest_status} (min test R2={r2_text}), "
        f"so claims are capped at {ceiling}{range_text}. The surrogate is used as an evidence-gated "
        "screening diagnostic, not as facility-level external validation."
    )


def _allocated_pathway_share_pct(allocations: pd.DataFrame, scenario_name: str, pathway: str) -> float:
    if allocations.empty or not {"scenario_name", "pathway"}.issubset(allocations.columns):
        return 0.0
    subset = allocations[allocations["scenario_name"].astype(str).eq(str(scenario_name))].copy()
    if subset.empty:
        return 0.0
    allocated = _numeric_column(subset, "allocated_feed_ton_per_year")
    total = float(allocated.sum())
    if total <= 0.0:
        return 0.0
    pathway_total = float(
        allocated.loc[subset["pathway"].astype(str).str.lower().eq(str(pathway).lower())].sum()
    )
    return pathway_total / total * 100.0


def _selected_sample_ids_for_pathway(allocations: pd.DataFrame, *, scenario_name: str, pathway: str) -> str:
    if allocations.empty or not {"scenario_name", "pathway", "sample_id"}.issubset(allocations.columns):
        return ""
    subset = allocations[
        allocations["scenario_name"].astype(str).eq(str(scenario_name))
        & allocations["pathway"].astype(str).str.lower().eq(str(pathway).lower())
        & (_numeric_column(allocations, "allocated_feed_ton_per_year") > 0.0)
    ]
    return _join_pipe_values(subset.get("sample_id", pd.Series(dtype="object")))


def _cap_diagnostic_row(cap_diagnostics: pd.DataFrame, *, scenario_name: str, ablation_key: str) -> pd.Series:
    if cap_diagnostics.empty or not {"scenario_name", "ablation_key"}.issubset(cap_diagnostics.columns):
        return pd.Series(dtype="object")
    match = cap_diagnostics[
        cap_diagnostics["scenario_name"].astype(str).eq(str(scenario_name))
        & cap_diagnostics["ablation_key"].astype(str).eq(str(ablation_key))
    ]
    return match.iloc[0] if not match.empty else pd.Series(dtype="object")


def _first_matching_row(frame: pd.DataFrame, *, scenario_name: str) -> pd.Series:
    if frame.empty or "scenario_name" not in frame.columns:
        return pd.Series(dtype="object")
    match = frame[frame["scenario_name"].astype(str).eq(str(scenario_name))]
    return match.iloc[0] if not match.empty else pd.Series(dtype="object")


def _ablation_share(
    frame: pd.DataFrame,
    *,
    scenario_name: str,
    ablation_family: str,
    ablation_key: str,
    share_column: str,
) -> object:
    if frame.empty or not {"scenario_name", "ablation_family", "ablation_key", share_column}.issubset(frame.columns):
        return pd.NA
    match = frame[
        frame["scenario_name"].astype(str).eq(str(scenario_name))
        & frame["ablation_family"].astype(str).eq(str(ablation_family))
        & frame["ablation_key"].astype(str).eq(str(ablation_key))
    ]
    if match.empty:
        return pd.NA
    return _optional_float(match.iloc[0].get(share_column))


def _max_ablation_share(
    frame: pd.DataFrame,
    *,
    scenario_name: str,
    ablation_family: str,
    key_contains: str,
    share_column: str,
) -> object:
    if frame.empty or not {"scenario_name", "ablation_family", "ablation_key", share_column}.issubset(frame.columns):
        return pd.NA
    match = frame[
        frame["scenario_name"].astype(str).eq(str(scenario_name))
        & frame["ablation_family"].astype(str).eq(str(ablation_family))
        & frame["ablation_key"].astype(str).str.contains(str(key_contains), case=False, na=False)
    ]
    if match.empty:
        return pd.NA
    values = pd.to_numeric(match[share_column], errors="coerce").dropna()
    return float(values.max()) if not values.empty else pd.NA


def _classify_ad_boundary_evidence_status(
    *,
    ad_reference_present: bool,
    complement_10: object,
    complement_20: object,
    digestate_max: object,
) -> str:
    missing: list[str] = []
    if not ad_reference_present:
        missing.append("missing_ad_reference")
    if pd.isna(complement_10):
        missing.append("missing_ad_floor_10pct")
    if pd.isna(complement_20):
        missing.append("missing_ad_floor_20pct")
    if pd.isna(digestate_max):
        missing.append("missing_digestate_credit")
    return "evaluated" if not missing else "|".join(missing)


def _binding_constraint_interpretation(row: pd.Series, relaxed_pyro: object, base_pyro: float) -> str:
    binding = []
    if _coerce_bool_flag(row.get("candidate_cap_binding")):
        binding.append("candidate cap")
    if _coerce_bool_flag(row.get("subtype_cap_binding")):
        binding.append("subtype cap")
    if _coerce_bool_flag(row.get("carbon_budget_binding")):
        binding.append("residual carbon budget")
    if _coerce_bool_flag(row.get("min_distinct_subtypes_binding")):
        binding.append("minimum subtype diversity")
    if _coerce_bool_flag(row.get("max_selected_binding")):
        binding.append("maximum selected-row count")
    if pd.notna(relaxed_pyro):
        delta = float(relaxed_pyro) - float(base_pyro)
        shift = f"cap-relaxed pyrolysis share changes by {delta:+.1f} percentage points"
    else:
        shift = "cap-relaxation comparator unavailable"
    binding_text = ", ".join(binding) if binding else "no audited constraint"
    return f"Binding diagnostics: {binding_text}; {shift}."


def _format_duplicate_operating_condition(row: pd.Series) -> str:
    temp = _optional_float(row.get("process_temperature_c"))
    residence = _optional_float(row.get("residence_time_min"))
    heating = _optional_float(row.get("heating_rate_c_per_min"))
    heating_text = "--" if pd.isna(heating) else f"{float(heating):.1f}"
    return (
        f"T={float(temp):.1f} C, residence={float(residence):.1f} min, "
        f"heating={heating_text} C min^-1"
        if pd.notna(temp) and pd.notna(residence)
        else "--"
    )


def _format_duplicate_target_signature(row: pd.Series) -> str:
    fields = [
        ("yield", "predicted_product_char_yield_pct", "product_char_yield_pct"),
        ("HHV", "predicted_product_char_hhv_mj_per_kg", "product_char_hhv_mj_per_kg"),
        ("energy", "predicted_energy_recovery_pct", "energy_recovery_pct"),
        ("carbon", "predicted_carbon_retention_pct", "carbon_retention_pct"),
    ]
    parts: list[str] = []
    for label, predicted, fallback in fields:
        value = _optional_float(row.get(predicted, row.get(fallback)))
        if pd.notna(value):
            parts.append(f"{label}={float(value):.2f}")
    return ", ".join(parts) if parts else "--"


def _coerce_bool_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "selected"}


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


def _numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _portfolio_share_reference(path: Path) -> pd.DataFrame:
    allocations = _read_csv_if_exists(path)
    if allocations.empty or not {"scenario_name", "pathway"}.issubset(allocations.columns):
        return pd.DataFrame()
    working = allocations.copy()
    working["scenario_name"] = working["scenario_name"].astype(str)
    working["pathway"] = working["pathway"].astype(str).str.lower()
    if "allocated_feed_share" in working.columns:
        working["_share_pct"] = _numeric_column(working, "allocated_feed_share") * 100.0
    elif "allocated_feed_ton_per_year" in working.columns:
        allocated = _numeric_column(working, "allocated_feed_ton_per_year")
        scenario_total = allocated.groupby(working["scenario_name"]).transform("sum")
        working["_share_pct"] = (allocated / scenario_total.where(scenario_total > 0)).fillna(0.0) * 100.0
    else:
        return pd.DataFrame()
    reference = (
        working.groupby(["scenario_name", "pathway"], dropna=False)["_share_pct"]
        .sum()
        .rename("expected_share_pct")
        .reset_index()
    )
    scenario_names = reference["scenario_name"].drop_duplicates().tolist()
    standard_rows = pd.DataFrame(
        [
            {"scenario_name": scenario_name, "pathway": pathway, "expected_share_pct": 0.0}
            for scenario_name in scenario_names
            for pathway in ("pyrolysis", "htc")
        ]
    )
    return (
        pd.concat([reference, standard_rows], ignore_index=True)
        .groupby(["scenario_name", "pathway"], dropna=False)["expected_share_pct"]
        .sum()
        .reset_index()
    )


def _artifact_share_reference(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not {"scenario_name", "pathway", "baseline_portfolio_share_pct"}.issubset(frame.columns):
        return pd.DataFrame()
    working = frame.copy()
    working["scenario_name"] = working["scenario_name"].astype(str)
    working["pathway"] = working["pathway"].astype(str).str.lower()
    working["observed_share_pct"] = _numeric_column(working, "baseline_portfolio_share_pct")
    return (
        working.groupby(["scenario_name", "pathway"], dropna=False)["observed_share_pct"]
        .sum()
        .reset_index()
    )


def _load_or_build_planning_recommendation_confidence(planning_dir: Path) -> pd.DataFrame:
    existing = _read_csv_if_exists(planning_dir / "recommendation_confidence_summary.csv")
    if not existing.empty:
        return existing
    main_results = _read_csv_if_exists(planning_dir / "main_results_table.csv")
    if main_results.empty:
        return pd.DataFrame()
    return build_recommendation_confidence_summary(main_results)


def _read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _classify_recommendation_evidence_ceiling(row: pd.Series) -> str:
    return classify_recommendation_evidence_ceiling(
        claim_boundary=row.get("claim_boundary", ""),
        reliability_tier=row.get("reliability_tier", ""),
        selected=bool(row.get("selected_in_baseline_portfolio", False)),
        confidence_tier=row.get("recommendation_confidence_tier", ""),
    )


def _optional_float(value: object) -> float | object:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return pd.NA
    return float(numeric)


def _audit_model_priority_rank(*, dataset_key: str, model_key: str) -> int:
    if dataset_key in HTC_PRIORITY_DATASETS:
        try:
            return HTC_MODEL_PRIORITY.index(model_key)
        except ValueError:
            return len(HTC_MODEL_PRIORITY)
    return 0


def _classify_scenario_transferability_ceiling(
    *,
    weighted_score: float,
    auxiliary_share: float,
    limited_share: float,
    missing_share: float,
) -> str:
    return classify_scenario_transferability_ceiling(
        weighted_score=weighted_score,
        auxiliary_share=auxiliary_share,
        limited_share=limited_share,
        missing_share=missing_share,
    )


def _build_transferability_note(
    *,
    evidence_ceiling: str,
    weighted_score: float,
    auxiliary_share: float,
    limited_share: float,
    missing_share: float,
) -> str:
    return build_transferability_note(evidence_ceiling=evidence_ceiling)


def _mode_or_default(series: pd.Series, default: str) -> str:
    values = series.astype(str).replace("nan", "").replace("", pd.NA).dropna()
    if values.empty:
        return default
    mode = values.mode(dropna=True)
    if mode.empty:
        return default
    return str(mode.iloc[0])


def _join_pipe_values(series: pd.Series) -> str:
    values: list[str] = []
    for raw_value in series.fillna("").astype(str):
        for part in raw_value.split("|"):
            cleaned = part.strip()
            if cleaned and cleaned.lower() != "nan" and cleaned not in values:
                values.append(cleaned)
    return "|".join(values)


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
                f"surrogate-supported allocated share is {share_pct}, below the {SURROGATE_LED_SHARE_THRESHOLD * 100.0:.1f}% threshold."
            ),
            InconsistencyWarning,
        )


if __name__ == "__main__":
    raise SystemExit(main())
