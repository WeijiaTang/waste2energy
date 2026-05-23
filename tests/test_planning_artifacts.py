# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from waste2energy.planning.artifacts import write_planning_outputs
from waste2energy.planning.solve import PlanningConfig


def test_write_planning_outputs_exports_data_contract_and_reproducibility_manifest(tmp_path):
    dataset_path = tmp_path / "planning_input.csv"
    input_frame = pd.DataFrame(
        [
            {
                "optimization_case_id": "Waste2Energy::planning::htc::0001::baseline_region_case",
                "sample_id": "Waste2Energy::planning::htc::0001",
                "scenario_name": "baseline_region_case",
                "pathway": "htc",
                "source_dataset_kind": "synthetic_mixed_feed_htc_candidate",
            }
        ]
    )
    input_frame.to_csv(dataset_path, index=False)
    bundle = SimpleNamespace(
        frame=input_frame,
        dataset_path=dataset_path,
        scenario_names=("baseline_region_case",),
        pathways=("htc",),
        real_cost_columns=(),
        unit_registry={"planning_mass_unit_basis": "metric_ton"},
    )
    one_row = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "optimization_case_id": "case-1",
                "pathway": "htc",
                "allocated_feed_share": 1.0,
                "product_char_yield_pct_surrogate_evidence_gate": "conditional_transfer",
                "product_char_yield_pct_surrogate_can_support_optimization": True,
            }
        ]
    )

    outputs = write_planning_outputs(
        scored=one_row,
        scenario_recommendations=one_row,
        pareto_candidates=one_row,
        scenario_constraints=one_row,
        portfolio_allocations=one_row,
        portfolio_summary=one_row,
        scenario_summary=one_row,
        pathway_summary=one_row,
        surrogate_predictions=one_row,
        optimization_diagnostics=one_row,
        planning_data_quality_summary=one_row,
        planning_candidate_exclusions=one_row,
        scenario_external_evidence=pd.DataFrame(
            [{"scenario_name": "baseline_region_case", "evidence_source": "fixture"}]
        ),
        output_dir=str(tmp_path / "outputs"),
        config=PlanningConfig(),
        bundle=bundle,
        readiness={"energy": "ready"},
    )

    contract_rows = pd.read_csv(outputs["planning_data_contract_rows"])
    contract_summary = pd.read_csv(outputs["planning_data_contract_summary"])
    transferability_summary = pd.read_csv(outputs["surrogate_transferability_summary"])
    run_config = json.loads((tmp_path / "outputs" / "run_config.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (tmp_path / "outputs" / "reproducibility_manifest.json").read_text(encoding="utf-8")
    )

    assert contract_rows.loc[0, "evidence_provenance"] == "scenario_expanded_candidate"
    assert contract_summary.loc[0, "scenario_expanded_count"] == 1
    assert transferability_summary.loc[0, "selected_allocated_feed_share"] == 1.0
    assert transferability_summary.loc[0, "transferability_risk_label"] == (
        "selected_share_conditionally_transferable"
    )
    assert "data_contract_summary" in run_config
    assert "surrogate_transferability_summary" in run_config
    assert manifest["inputs"][0]["sha256"]
    assert any(output["path"].endswith("run_config.json") for output in manifest["outputs"])
