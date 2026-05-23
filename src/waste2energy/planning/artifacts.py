# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common import (
    build_reproducibility_manifest,
    build_run_manifest,
    write_json,
    write_reproducibility_manifest,
)
from ..config import PLANNING_OUTPUTS_DIR
from ..data.contracts import annotate_data_contract, summarize_data_contract, validate_planning_input_contract


def write_planning_outputs(
    *,
    scored: pd.DataFrame,
    scenario_recommendations: pd.DataFrame,
    pareto_candidates: pd.DataFrame,
    scenario_constraints: pd.DataFrame,
    portfolio_allocations: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    pathway_summary: pd.DataFrame,
    surrogate_predictions: pd.DataFrame,
    optimization_diagnostics: pd.DataFrame,
    planning_data_quality_summary: pd.DataFrame,
    planning_candidate_exclusions: pd.DataFrame,
    scenario_external_evidence: pd.DataFrame,
    output_dir: str | None,
    config,
    bundle,
    readiness: dict[str, str],
) -> dict[str, str]:
    target_dir = Path(output_dir) if output_dir else PLANNING_OUTPUTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "scored_cases": target_dir / "scored_cases.csv",
        "scenario_recommendations": target_dir / "scenario_recommendations.csv",
        "pareto_candidates": target_dir / "pareto_candidates.csv",
        "pareto_front": target_dir / "pareto_front.csv",
        "scenario_constraints": target_dir / "scenario_constraints.csv",
        "portfolio_allocations": target_dir / "portfolio_allocations.csv",
        "portfolio_summary": target_dir / "portfolio_summary.csv",
        "scenario_summary": target_dir / "scenario_summary.csv",
        "pathway_summary": target_dir / "pathway_summary.csv",
        "surrogate_predictions": target_dir / "surrogate_predictions.csv",
        "surrogate_transferability_summary": target_dir / "surrogate_transferability_summary.csv",
        "optimization_diagnostics": target_dir / "optimization_diagnostics.csv",
        "planning_data_quality_summary": target_dir / "planning_data_quality_summary.csv",
        "planning_candidate_exclusions": target_dir / "planning_candidate_exclusions.csv",
        "planning_data_contract_rows": target_dir / "planning_data_contract_rows.csv",
        "planning_data_contract_summary": target_dir / "planning_data_contract_summary.csv",
        "planning_data_contract_warnings": target_dir / "planning_data_contract_warnings.csv",
        "scenario_external_evidence": target_dir / "scenario_external_evidence.csv",
        "run_config": target_dir / "run_config.json",
        "reproducibility_manifest": target_dir / "reproducibility_manifest.json",
    }

    scored.to_csv(outputs["scored_cases"], index=False)
    scenario_recommendations.to_csv(outputs["scenario_recommendations"], index=False)
    pareto_candidates.to_csv(outputs["pareto_candidates"], index=False)
    pareto_candidates.to_csv(outputs["pareto_front"], index=False)
    scenario_constraints.to_csv(outputs["scenario_constraints"], index=False)
    portfolio_allocations.to_csv(outputs["portfolio_allocations"], index=False)
    portfolio_summary.to_csv(outputs["portfolio_summary"], index=False)
    scenario_summary.to_csv(outputs["scenario_summary"], index=False)
    pathway_summary.to_csv(outputs["pathway_summary"], index=False)
    surrogate_predictions.to_csv(outputs["surrogate_predictions"], index=False)
    surrogate_transferability_summary = _build_surrogate_transferability_summary(
        surrogate_predictions,
        portfolio_allocations,
    )
    surrogate_transferability_summary.to_csv(outputs["surrogate_transferability_summary"], index=False)
    optimization_diagnostics.to_csv(outputs["optimization_diagnostics"], index=False)
    planning_data_quality_summary.to_csv(outputs["planning_data_quality_summary"], index=False)
    planning_candidate_exclusions.to_csv(outputs["planning_candidate_exclusions"], index=False)
    data_contract_rows = _build_data_contract_rows(bundle.frame)
    data_contract_summary = _build_data_contract_summary_frame(bundle.frame)
    data_contract_warnings = _build_data_contract_warning_frame(bundle.frame)
    data_contract_rows.to_csv(outputs["planning_data_contract_rows"], index=False)
    data_contract_summary.to_csv(outputs["planning_data_contract_summary"], index=False)
    data_contract_warnings.to_csv(outputs["planning_data_contract_warnings"], index=False)
    scenario_external_evidence.to_csv(outputs["scenario_external_evidence"], index=False)

    run_config = build_run_manifest(
        dataset_path=str(bundle.dataset_path),
        scenario_names=list(bundle.scenario_names),
        pathways=list(bundle.pathways),
        real_cost_columns=list(bundle.real_cost_columns),
        unit_registry=bundle.unit_registry,
        planning_config=config,
        objective_weights=config.objective_weight_system,
        objective_readiness=readiness,
        scenario_external_evidence_table_path=config.scenario_external_evidence_table_path,
        row_count=int(len(scored)),
        data_contract_summary=data_contract_summary.to_dict(orient="records"),
        data_contract_warnings=data_contract_warnings["warning"].astype(str).tolist()
        if "warning" in data_contract_warnings.columns
        else [],
        surrogate_transferability_summary=surrogate_transferability_summary.to_dict(orient="records"),
        planner_variant="surrogate_driven_robust_multiobjective_optimizer",
        output_files={key: str(path) for key, path in outputs.items()},
    )
    write_json(outputs["run_config"], run_config)
    manifest_inputs = [bundle.dataset_path]
    if config.scenario_external_evidence_table_path:
        manifest_inputs.append(config.scenario_external_evidence_table_path)
    reproducibility_manifest = build_reproducibility_manifest(
        command="waste2energy-plan",
        inputs=manifest_inputs,
        outputs=[
            path
            for key, path in outputs.items()
            if key != "reproducibility_manifest"
        ],
        parameters={
            "objective_weight_preset": config.objective_weight_preset,
            "objective_weights": {
                "energy": config.energy_weight,
                "environment": config.environment_weight,
                "cost": config.cost_weight,
            },
            "primary_optimization_pathways": list(config.primary_optimization_pathways),
            "optimization_method": config.optimization_method,
            "pyomo_solver_preference": config.pyomo_solver_preference,
            "robustness_factor": config.robustness_factor,
            "carbon_budget_factor": config.carbon_budget_factor,
            "minimum_surrogate_artifact_test_r2": config.minimum_surrogate_artifact_test_r2,
        },
    )
    write_reproducibility_manifest(outputs["reproducibility_manifest"], reproducibility_manifest)
    return {key: str(path) for key, path in outputs.items()}


def _build_data_contract_rows(frame: pd.DataFrame) -> pd.DataFrame:
    annotated = annotate_data_contract(frame)
    columns = [
        "optimization_case_id",
        "sample_id",
        "scenario_name",
        "pathway",
        "source_dataset_kind",
        "evidence_provenance",
        "is_independent_observation",
        "is_scenario_expanded",
        "is_primary_evidence_for_surrogate",
    ]
    return annotated[[column for column in columns if column in annotated.columns]].copy()


def _build_data_contract_summary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_data_contract(frame)
    rows = [
        {
            "summary_scope": "all_planning_rows",
            "row_count": summary.row_count,
            "independent_observation_count": summary.independent_observation_count,
            "scenario_expanded_count": summary.scenario_expanded_count,
            "synthetic_candidate_count": summary.synthetic_candidate_count,
            "pathway": "",
            "pathway_row_count": "",
            "provenance": "",
            "provenance_row_count": "",
        }
    ]
    rows.extend(
        {
            "summary_scope": "pathway",
            "row_count": summary.row_count,
            "independent_observation_count": summary.independent_observation_count,
            "scenario_expanded_count": summary.scenario_expanded_count,
            "synthetic_candidate_count": summary.synthetic_candidate_count,
            "pathway": pathway,
            "pathway_row_count": count,
            "provenance": "",
            "provenance_row_count": "",
        }
        for pathway, count in sorted(summary.pathway_counts.items())
    )
    rows.extend(
        {
            "summary_scope": "provenance",
            "row_count": summary.row_count,
            "independent_observation_count": summary.independent_observation_count,
            "scenario_expanded_count": summary.scenario_expanded_count,
            "synthetic_candidate_count": summary.synthetic_candidate_count,
            "pathway": "",
            "pathway_row_count": "",
            "provenance": provenance,
            "provenance_row_count": count,
        }
        for provenance, count in sorted(summary.provenance_counts.items())
    )
    return pd.DataFrame(rows)


def _build_data_contract_warning_frame(frame: pd.DataFrame) -> pd.DataFrame:
    warnings = validate_planning_input_contract(frame)
    if not warnings:
        return pd.DataFrame([{"warning": "none", "severity": "pass"}])
    return pd.DataFrame(
        {
            "warning": warnings,
            "severity": [
                "boundary_note"
                if warning == "scenario_expanded_rows_must_not_be_counted_as_independent_evidence"
                else "review_required"
                for warning in warnings
            ],
        }
    )


def _build_surrogate_transferability_summary(
    surrogate_predictions: pd.DataFrame,
    portfolio_allocations: pd.DataFrame,
) -> pd.DataFrame:
    gate_columns = [
        column
        for column in surrogate_predictions.columns
        if column.endswith("_surrogate_evidence_gate")
    ]
    support_columns = [
        column
        for column in surrogate_predictions.columns
        if column.endswith("_surrogate_can_support_optimization")
    ]
    required = {"scenario_name", "pathway", "optimization_case_id"}
    if surrogate_predictions.empty or not gate_columns or not required.issubset(surrogate_predictions.columns):
        return pd.DataFrame(
            columns=[
                "scenario_name",
                "pathway",
                "candidate_count",
                "selected_allocated_feed_share",
                "worst_surrogate_evidence_gate",
                "optimization_supported_prediction_fraction",
                "transferability_risk_label",
            ]
        )

    selected_share = _selected_share_by_scenario_pathway(portfolio_allocations)
    severity = {
        "conditional_transfer": 3,
        "screening_only": 2,
        "static_fallback": 1,
        "unknown": 1,
        "unsupported": 0,
    }
    rows: list[dict[str, object]] = []
    for (scenario_name, pathway), frame in surrogate_predictions.groupby(["scenario_name", "pathway"], dropna=False):
        gates = [
            str(value)
            for column in gate_columns
            for value in frame[column].dropna().astype(str).tolist()
            if str(value).strip()
        ]
        worst_gate = min(gates, key=lambda gate: severity.get(gate, -1)) if gates else "unknown"
        if support_columns:
            support_values = pd.concat(
                [frame[column].astype(bool) for column in support_columns],
                ignore_index=True,
            )
            support_fraction = float(support_values.mean()) if len(support_values) else 0.0
        else:
            support_fraction = 0.0
        allocated_share = selected_share.get((str(scenario_name), str(pathway)), 0.0)
        rows.append(
            {
                "scenario_name": scenario_name,
                "pathway": pathway,
                "candidate_count": int(len(frame)),
                "selected_allocated_feed_share": allocated_share,
                "worst_surrogate_evidence_gate": worst_gate,
                "optimization_supported_prediction_fraction": support_fraction,
                "transferability_risk_label": _transferability_risk_label(
                    worst_gate=worst_gate,
                    support_fraction=support_fraction,
                    selected_share=allocated_share,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario_name", "pathway"]).reset_index(drop=True)


def _selected_share_by_scenario_pathway(portfolio_allocations: pd.DataFrame) -> dict[tuple[str, str], float]:
    required = {"scenario_name", "pathway", "allocated_feed_share"}
    if portfolio_allocations.empty or not required.issubset(portfolio_allocations.columns):
        return {}
    grouped = portfolio_allocations.copy()
    grouped["allocated_feed_share"] = pd.to_numeric(grouped["allocated_feed_share"], errors="coerce").fillna(0.0)
    return {
        (str(scenario_name), str(pathway)): float(frame["allocated_feed_share"].sum())
        for (scenario_name, pathway), frame in grouped.groupby(["scenario_name", "pathway"], dropna=False)
    }


def _transferability_risk_label(
    *,
    worst_gate: str,
    support_fraction: float,
    selected_share: float,
) -> str:
    if worst_gate in {"unsupported", "unknown", "static_fallback"} and selected_share > 0.0:
        return "selected_share_relies_on_non_transferable_or_fallback_surrogate"
    if support_fraction < 0.5 and selected_share > 0.0:
        return "selected_share_has_partial_surrogate_transferability"
    if worst_gate == "screening_only" and selected_share > 0.0:
        return "selected_share_screening_only_transferability"
    if selected_share > 0.0:
        return "selected_share_conditionally_transferable"
    return "not_selected_or_reference_only"
