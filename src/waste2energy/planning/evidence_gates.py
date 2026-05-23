# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

"""Conservative evidence gates for surrogate-informed planning.

These helpers keep leave-study-out and other cross-study diagnostics separate
from ordinary in-sample model ranking.  They are intentionally small and
deterministic so manuscript/audit code can reuse the same conservative labels
instead of re-implementing ad hoc R² thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import pandas as pd


SUPPORTED = "conditional_transfer"
SCREENING_ONLY = "screening_only"
UNSUPPORTED = "unsupported"
UNKNOWN = "unknown"

LEAVE_STUDY_OUT_ALIASES = frozenset(
    {
        "leave_study_out",
        "leave-study-out",
        "leave_source_repo_out",
        "leave-source-repo-out",
        "strict_group",
        "strict-group",
    }
)


@dataclass(frozen=True)
class SurrogateGateResult:
    """Evidence qualification for one surrogate diagnostic value."""

    status: str
    can_support_optimization: bool
    can_support_screening: bool
    test_r2: float | None
    split_strategy: str
    reason: str


def classify_surrogate_transferability(
    test_r2: float | int | str | None,
    split_strategy: str | None,
    *,
    min_r2_for_optimization: float = 0.30,
    min_r2_for_screening: float = 0.00,
) -> SurrogateGateResult:
    """Classify whether a surrogate is defensible for planning inference.

    Negative or missing cross-study R² is not treated as optimization-grade
    evidence.  Values between the screening and optimization thresholds may be
    used only for conservative ranking/screening language.
    """

    strategy = str(split_strategy or "unknown").strip().lower() or "unknown"
    value = _to_finite_float(test_r2)
    if value is None:
        return SurrogateGateResult(
            status=UNKNOWN,
            can_support_optimization=False,
            can_support_screening=False,
            test_r2=None,
            split_strategy=strategy,
            reason="missing_transferability_metric",
        )

    is_cross_study = strategy in LEAVE_STUDY_OUT_ALIASES
    if value < min_r2_for_screening:
        reason = "negative_cross_study_r2" if is_cross_study else "negative_test_r2"
        return SurrogateGateResult(
            status=UNSUPPORTED,
            can_support_optimization=False,
            can_support_screening=False,
            test_r2=value,
            split_strategy=strategy,
            reason=reason,
        )

    if value < min_r2_for_optimization:
        reason = "low_cross_study_r2_screening_only" if is_cross_study else "low_test_r2_screening_only"
        return SurrogateGateResult(
            status=SCREENING_ONLY,
            can_support_optimization=False,
            can_support_screening=True,
            test_r2=value,
            split_strategy=strategy,
            reason=reason,
        )

    return SurrogateGateResult(
        status=SUPPORTED,
        can_support_optimization=True,
        can_support_screening=True,
        test_r2=value,
        split_strategy=strategy,
        reason="meets_transferability_threshold",
    )


def annotate_surrogate_gate_columns(
    frame: pd.DataFrame,
    *,
    metric_columns: tuple[str, ...] = ("selected_test_r2", "test_r2", "reporting_test_r2"),
    split_strategy_column: str = "split_strategy",
    min_r2_for_optimization: float = 0.30,
    min_r2_for_screening: float = 0.00,
) -> pd.DataFrame:
    """Return a copy with reviewer-facing surrogate evidence labels."""

    if frame.empty:
        return frame.copy()

    metric_column = next((column for column in metric_columns if column in frame.columns), None)
    annotated = frame.copy()
    results: list[SurrogateGateResult] = []
    for _, row in annotated.iterrows():
        result = classify_surrogate_transferability(
            row.get(metric_column) if metric_column else None,
            row.get(split_strategy_column, "unknown"),
            min_r2_for_optimization=min_r2_for_optimization,
            min_r2_for_screening=min_r2_for_screening,
        )
        results.append(result)

    annotated["surrogate_evidence_gate"] = [result.status for result in results]
    annotated["surrogate_evidence_reason"] = [result.reason for result in results]
    annotated["surrogate_can_support_optimization"] = [
        result.can_support_optimization for result in results
    ]
    annotated["surrogate_can_support_screening"] = [result.can_support_screening for result in results]
    return annotated


def summarize_surrogate_transferability(
    frame: pd.DataFrame,
    *,
    pathway_column: str = "pathway",
    target_column: str = "target_column",
) -> pd.DataFrame:
    """Summarize evidence gates by pathway/target for audit tables."""

    annotated = annotate_surrogate_gate_columns(frame)
    if annotated.empty:
        return pd.DataFrame(
            columns=[
                pathway_column,
                target_column,
                "artifact_count",
                "optimization_supported_count",
                "screening_only_count",
                "unsupported_count",
                "weakest_evidence_gate",
            ]
        )

    group_columns = [column for column in (pathway_column, target_column) if column in annotated.columns]
    if not group_columns:
        group_columns = ["surrogate_evidence_gate"]

    order = {SUPPORTED: 3, SCREENING_ONLY: 2, UNKNOWN: 1, UNSUPPORTED: 0}
    rows: list[Mapping[str, object]] = []
    for keys, group in annotated.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=False))
        gates = group["surrogate_evidence_gate"].astype(str)
        weakest = min(gates, key=lambda gate: order.get(gate, -1))
        row.update(
            {
                "artifact_count": int(len(group)),
                "optimization_supported_count": int(group["surrogate_can_support_optimization"].sum()),
                "screening_only_count": int(gates.eq(SCREENING_ONLY).sum()),
                "unsupported_count": int(gates.eq(UNSUPPORTED).sum()),
                "weakest_evidence_gate": weakest,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _to_finite_float(value: float | int | str | None) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if isfinite(converted) else None

