# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import OUTPUTS_ROOT, resolve_surrogate_outputs_dir
from .data import DATASET_KEYS, TARGET_COLUMNS
from .models import MODEL_KEYS


def _default_split_summary_paths() -> dict[str, Path]:
    surrogate_root = resolve_surrogate_outputs_dir()
    return {
        "recommended": surrogate_root / "traditional_ml_suite_summary.csv",
        "strict_group": surrogate_root / "traditional_ml_suite_summary_strict_group.csv",
        "leave_study_out": surrogate_root / "traditional_ml_suite_summary_leave_study_out.csv",
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
    ml_paths = {
        key: (active_outputs_root / path.relative_to(OUTPUTS_ROOT))
        for key, path in default_paths.items()
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
    ml_best = build_ml_best_result_summary(ml_paths)
    ml_flags = build_ml_claim_flag_table(ml_paths)
    planning_flags = build_planning_claim_flag_table(active_planning_dir, active_scenario_dir)
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
        "planning_claim_flag_table": planning_flags,
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


def build_ml_best_result_summary(summary_paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, path in summary_paths.items():
        frame = _read_csv_if_exists(path)
        if frame.empty or "test_r2" not in frame.columns:
            continue
        best = (
            frame.sort_values("test_r2", ascending=False)
            .groupby(["dataset_key", "target_column"], dropna=False)
            .first()
            .reset_index()
        )
        for _, row in best.iterrows():
            rows.append(
                {
                    "summary_label": label,
                    "dataset_key": row["dataset_key"],
                    "target_column": row["target_column"],
                    "best_model_key": row["model_key"],
                    "best_test_r2": row["test_r2"],
                    "best_test_rmse": row["test_rmse"],
                    "best_test_mae": row["test_mae"],
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["summary_label", "dataset_key", "target_column"]).reset_index(drop=True)


def build_ml_claim_flag_table(summary_paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strict_group = _read_csv_if_exists(summary_paths["strict_group"])
    leave_study_out = _read_csv_if_exists(summary_paths["leave_study_out"])

    rows.extend(
        _build_claim_rows_from_frame(
            strict_group,
            summary_label="strict_group",
            claim_rule="Paper 1 main-table benchmark evidence tier",
            positive_threshold=0.65,
        )
    )
    rows.extend(
        _build_claim_rows_from_frame(
            leave_study_out,
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
        if method_name == "sac" and abs(float(row["reward_improvement_vs_hold_plan_abs"])) < 1e-9:
            claim_status = "conservative_match"
            notes.append("matches_hold_plan_reward")
        if float(row["max_violation_mean"]) > 0.0:
            claim_status = "violation_prone"
            notes.append("nonzero_violation")
        if float(row["reward_std"]) > 1.0:
            claim_status = "unstable"
            notes.append("high_seed_variation")

        rows.append(
            {
                "scenario_name": scenario_name,
                "method_name": method_name,
                "method_type": row["method_type"],
                "claim_status": claim_status,
                "reward_mean": row["reward_mean"],
                "reward_std": row["reward_std"],
                "max_violation_mean": row["max_violation_mean"],
                "reward_improvement_vs_hold_plan_pct": row["reward_improvement_vs_hold_plan_pct"],
                "violation_aware_rank_within_scenario": row["violation_aware_rank_within_scenario"],
                "throughput_nonzero_rate_mean": behavior_row.get("throughput_nonzero_rate_mean", pd.NA),
                "severity_nonzero_rate_mean": behavior_row.get("severity_nonzero_rate_mean", pd.NA),
                "notes": ";".join(notes),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["scenario_name", "method_type", "method_name"]).reset_index(drop=True)


def build_planning_claim_flag_table(planning_dir: Path, scenario_dir: Path) -> pd.DataFrame:
    main_results = _read_csv_if_exists(planning_dir / "main_results_table.csv")
    pathway_summary = _read_csv_if_exists(planning_dir / "pathway_summary.csv")
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
    selected_columns = [
        "scenario_name",
        "pathway",
        "writing_label",
        "claim_status",
        "claim_rule",
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
    summary_label: str,
    claim_rule: str,
    positive_threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if frame.empty:
        return rows

    best = (
        frame.sort_values("test_r2", ascending=False)
        .groupby(["dataset_key", "target_column"], dropna=False)
        .first()
        .reset_index()
    )
    for _, row in best.iterrows():
        test_r2 = float(row["test_r2"])
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
                "best_model_key": row["model_key"],
                "best_test_r2": test_r2,
                "claim_status": claim_status,
            }
        )
    return rows


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _classify_planning_claim_status(row: pd.Series) -> str:
    selected = bool(row.get("selected_in_baseline_portfolio", False))
    stress_rate = float(pd.to_numeric(pd.Series([row.get("max_stress_selection_rate")]), errors="coerce").fillna(0.0).iloc[0])
    writing_label = str(row.get("writing_label", ""))
    if selected:
        return "supportive"
    if stress_rate > 0.0 and "environment-sensitive" in writing_label:
        return "conditional_support"
    if stress_rate > 0.0:
        return "stress_sensitive"
    if "comparison anchor" in writing_label:
        return "anchor_only"
    return "comparison_only"


def _describe_planning_claim_rule(row: pd.Series) -> str:
    selected = bool(row.get("selected_in_baseline_portfolio", False))
    stress_rate = float(pd.to_numeric(pd.Series([row.get("max_stress_selection_rate")]), errors="coerce").fillna(0.0).iloc[0])
    if selected:
        return "Selected in the baseline optimized portfolio under the current planning configuration."
    if stress_rate > 0.0:
        return "Not selected in the baseline portfolio but supported in at least one planning stress test."
    if str(row.get("writing_label", "")) == "comparison anchor":
        return "Retained as the manuscript comparison anchor rather than as a selected pathway."
    return "Available for pathway comparison but not currently supported as a selected planning recommendation."
