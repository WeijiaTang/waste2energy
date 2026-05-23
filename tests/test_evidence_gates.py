# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import pandas as pd

from waste2energy.planning.evidence_gates import (
    SCREENING_ONLY,
    SUPPORTED,
    UNSUPPORTED,
    annotate_surrogate_gate_columns,
    classify_surrogate_transferability,
    summarize_surrogate_transferability,
)


def test_negative_leave_study_out_r2_is_not_optimization_grade():
    result = classify_surrogate_transferability(-0.04, "leave_study_out")

    assert result.status == UNSUPPORTED
    assert not result.can_support_optimization
    assert not result.can_support_screening
    assert result.reason == "negative_cross_study_r2"


def test_low_positive_cross_study_r2_is_screening_only():
    result = classify_surrogate_transferability(0.12, "strict_group")

    assert result.status == SCREENING_ONLY
    assert not result.can_support_optimization
    assert result.can_support_screening


def test_transferable_r2_can_support_conditional_optimization():
    result = classify_surrogate_transferability(0.45, "strict_group")

    assert result.status == SUPPORTED
    assert result.can_support_optimization
    assert result.can_support_screening


def test_annotate_surrogate_gate_columns_uses_available_test_metric():
    frame = pd.DataFrame(
        [
            {"target_column": "energy_recovery_pct", "split_strategy": "strict_group", "selected_test_r2": -0.1},
            {"target_column": "carbon_retention_pct", "split_strategy": "strict_group", "selected_test_r2": 0.1},
            {"target_column": "product_char_yield_pct", "split_strategy": "strict_group", "selected_test_r2": 0.7},
        ]
    )

    annotated = annotate_surrogate_gate_columns(frame)

    assert annotated["surrogate_evidence_gate"].tolist() == [
        UNSUPPORTED,
        SCREENING_ONLY,
        SUPPORTED,
    ]
    assert annotated["surrogate_can_support_optimization"].tolist() == [False, False, True]


def test_summarize_surrogate_transferability_reports_weakest_gate():
    frame = pd.DataFrame(
        [
            {
                "pathway": "pyrolysis",
                "target_column": "energy_recovery_pct",
                "split_strategy": "strict_group",
                "selected_test_r2": -0.1,
            },
            {
                "pathway": "pyrolysis",
                "target_column": "energy_recovery_pct",
                "split_strategy": "strict_group",
                "selected_test_r2": 0.5,
            },
        ]
    )

    summary = summarize_surrogate_transferability(frame)

    assert len(summary) == 1
    assert summary.loc[0, "artifact_count"] == 2
    assert summary.loc[0, "unsupported_count"] == 1
    assert summary.loc[0, "weakest_evidence_gate"] == UNSUPPORTED

