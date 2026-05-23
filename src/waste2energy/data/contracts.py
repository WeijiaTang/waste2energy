# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

"""Data-contract helpers for reviewer-facing provenance checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


INDEPENDENT_OBSERVATION = "independent_observation"
SCENARIO_EXPANDED_CANDIDATE = "scenario_expanded_candidate"
SYNTHETIC_CANDIDATE = "synthetic_candidate"
LITERATURE_BOUNDED_REFERENCE = "literature_bounded_reference"
BASELINE_ANCHOR = "baseline_anchor"
UNKNOWN_PROVENANCE = "unknown"


@dataclass(frozen=True)
class DataContractSummary:
    row_count: int
    independent_observation_count: int
    scenario_expanded_count: int
    synthetic_candidate_count: int
    pathway_counts: dict[str, int]
    provenance_counts: dict[str, int]


def infer_row_provenance(row: pd.Series | dict[str, object]) -> str:
    """Infer a conservative evidence provenance label for one row."""

    source_kind = str(_get(row, "source_dataset_kind", "") or "").strip().lower()
    scenario_name = str(_get(row, "scenario_name", "") or "").strip()
    optimization_case_id = str(_get(row, "optimization_case_id", "") or "").strip().lower()
    sample_id = str(_get(row, "sample_id", "") or "").strip().lower()

    if scenario_name or optimization_case_id.count("::") >= 3:
        return SCENARIO_EXPANDED_CANDIDATE
    if "observed" in source_kind or source_kind in {"literature_observation", "primary_observation"}:
        return INDEPENDENT_OBSERVATION
    if "literature_bounded" in source_kind:
        return LITERATURE_BOUNDED_REFERENCE
    if "baseline" in source_kind:
        return BASELINE_ANCHOR
    if "synthetic" in source_kind or "candidate" in source_kind or "planning::" in sample_id:
        return SYNTHETIC_CANDIDATE
    return UNKNOWN_PROVENANCE


def annotate_data_contract(
    frame: pd.DataFrame,
    *,
    provenance_column: str = "evidence_provenance",
) -> pd.DataFrame:
    """Return a copy with row-level provenance and primary-evidence flags."""

    annotated = frame.copy()
    annotated[provenance_column] = [infer_row_provenance(row) for _, row in annotated.iterrows()]
    annotated["is_independent_observation"] = annotated[provenance_column].eq(INDEPENDENT_OBSERVATION)
    annotated["is_scenario_expanded"] = annotated[provenance_column].eq(SCENARIO_EXPANDED_CANDIDATE)
    annotated["is_primary_evidence_for_surrogate"] = annotated["is_independent_observation"]
    return annotated


def summarize_data_contract(
    frame: pd.DataFrame,
    *,
    pathway_column: str = "pathway",
    provenance_column: str = "evidence_provenance",
) -> DataContractSummary:
    """Summarize row provenance without inflating scenario-expanded evidence."""

    annotated = frame if provenance_column in frame.columns else annotate_data_contract(frame)
    pathway_counts = (
        annotated[pathway_column].astype(str).value_counts().sort_index().astype(int).to_dict()
        if pathway_column in annotated.columns
        else {}
    )
    provenance_counts = (
        annotated[provenance_column].astype(str).value_counts().sort_index().astype(int).to_dict()
        if provenance_column in annotated.columns
        else {}
    )
    return DataContractSummary(
        row_count=int(len(annotated)),
        independent_observation_count=int(annotated.get("is_independent_observation", pd.Series(dtype=bool)).sum()),
        scenario_expanded_count=int(annotated.get("is_scenario_expanded", pd.Series(dtype=bool)).sum()),
        synthetic_candidate_count=int(
            annotated[provenance_column].isin([SYNTHETIC_CANDIDATE, SCENARIO_EXPANDED_CANDIDATE]).sum()
        ),
        pathway_counts=pathway_counts,
        provenance_counts=provenance_counts,
    )


def validate_planning_input_contract(frame: pd.DataFrame) -> list[str]:
    """Return actionable contract warnings instead of raising on manuscript data."""

    warnings: list[str] = []
    required = {"optimization_case_id", "pathway", "source_dataset_kind"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        warnings.append(f"missing_required_columns:{'|'.join(missing)}")
        return warnings

    annotated = annotate_data_contract(frame)
    if annotated["optimization_case_id"].duplicated().any():
        warnings.append("duplicate_optimization_case_id")
    if annotated["is_independent_observation"].sum() == 0:
        warnings.append("no_independent_observations_planning_rows_are_candidates")
    if annotated["is_scenario_expanded"].sum() > 0:
        warnings.append("scenario_expanded_rows_must_not_be_counted_as_independent_evidence")
    return warnings


def read_and_summarize_planning_contract(path: str | Path) -> tuple[DataContractSummary, list[str]]:
    frame = pd.read_csv(path)
    return summarize_data_contract(frame), validate_planning_input_contract(frame)


def _get(row: pd.Series | dict[str, object], key: str, default: object = None) -> object:
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return row.get(key, default)

