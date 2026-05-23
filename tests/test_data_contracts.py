# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from pathlib import Path

import pandas as pd

from waste2energy.data.contracts import (
    INDEPENDENT_OBSERVATION,
    SCENARIO_EXPANDED_CANDIDATE,
    annotate_data_contract,
    infer_row_provenance,
    read_and_summarize_planning_contract,
    summarize_data_contract,
    validate_planning_input_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_infer_row_provenance_prioritizes_scenario_expansion_over_candidate_kind():
    provenance = infer_row_provenance(
        {
            "source_dataset_kind": "synthetic_mixed_feed_htc_candidate",
            "scenario_name": "baseline_region_case",
            "optimization_case_id": "Waste2Energy::planning::htc::0001::baseline_region_case",
        }
    )

    assert provenance == SCENARIO_EXPANDED_CANDIDATE


def test_infer_row_provenance_identifies_independent_observation_rows():
    provenance = infer_row_provenance({"source_dataset_kind": "observed_literature_row"})

    assert provenance == INDEPENDENT_OBSERVATION


def test_planning_input_contract_does_not_count_scenarios_as_independent_evidence():
    path = ROOT / "data" / "processed" / "model_ready" / "optimization_input_dataset.csv"
    summary, warnings = read_and_summarize_planning_contract(path)

    assert summary.row_count == 630
    assert summary.pathway_counts == {"ad": 45, "baseline": 45, "htc": 360, "pyrolysis": 180}
    assert summary.independent_observation_count == 0
    assert summary.scenario_expanded_count == 630
    assert "no_independent_observations_planning_rows_are_candidates" in warnings
    assert "scenario_expanded_rows_must_not_be_counted_as_independent_evidence" in warnings


def test_validate_planning_input_contract_flags_duplicate_case_ids():
    frame = pd.DataFrame(
        [
            {"optimization_case_id": "case-1", "pathway": "htc", "source_dataset_kind": "observed"},
            {"optimization_case_id": "case-1", "pathway": "htc", "source_dataset_kind": "observed"},
        ]
    )

    warnings = validate_planning_input_contract(frame)

    assert "duplicate_optimization_case_id" in warnings


def test_annotate_data_contract_adds_primary_evidence_flag():
    frame = pd.DataFrame([{"source_dataset_kind": "observed_literature_row", "pathway": "htc"}])

    annotated = annotate_data_contract(frame)
    summary = summarize_data_contract(annotated)

    assert bool(annotated.loc[0, "is_primary_evidence_for_surrogate"])
    assert summary.independent_observation_count == 1

