# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .common import parse_manifest_timestamp
from .config import BENCHMARK_OUTPUTS_DIR, OUTPUTS_ROOT, resolve_surrogate_outputs_dir
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
        "planning_data_quality_summary": planning_data_quality,
        "benchmark_claim_summary": benchmark_claim_summary,
        "benchmark_manuscript_sentences": benchmark_manuscript_sentences,
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
    supplement = _read_csv_if_exists(BENCHMARK_OUTPUTS_DIR / "htc_model_compare_lso" / "traditional_ml_suite_summary_leave_study_out.csv")
    if primary.empty:
        return supplement
    if supplement.empty:
        return primary
    combined = pd.concat([supplement, primary], ignore_index=True)
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
