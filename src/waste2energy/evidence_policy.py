from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


EVIDENCE_POLICY_VERSION = "2026-05-q1-evidence-policy-v2"
SURROGATE_LED_SHARE_THRESHOLD = 0.80
RELIABILITY_CONDITIONAL_SUPPORT_THRESHOLD = 0.60
RELIABILITY_AUXILIARY_THRESHOLD = 0.34
WEAK_OR_UNSUPPORTED_AUXILIARY_THRESHOLD = 0.85

SUPPORT_SCORE_MAP = {
    "surrogate_supported": 1.00,
    "trained_surrogate_with_documented_fallback": 0.82,
    "documented_static_fallback": 0.48,
    "unsupported_pathway": 0.22,
    "unknown": 0.35,
}

CONFIDENCE_COMPONENT_WEIGHTS = {
    "support": 0.40,
    "stress": 0.35,
    "role": 0.25,
}


@dataclass(frozen=True)
class ReliabilityDecision:
    tier: str
    reviewer_sentence: str


@dataclass(frozen=True)
class PlanningEvidencePenaltyPolicy:
    evidence_utility_factor: float = 0.15
    partial_surrogate_weight: float = 0.70
    static_fallback_weight: float = 0.35
    unsupported_pathway_weight: float = 0.15
    partial_surrogate_uncertainty_multiplier: float = 1.35
    static_fallback_uncertainty_multiplier: float = 2.10
    unsupported_pathway_uncertainty_multiplier: float = 3.25
    partial_surrogate_information_premium_usd_per_ton: float = 8.0
    static_fallback_information_premium_usd_per_ton: float = 22.0
    unsupported_pathway_information_premium_usd_per_ton: float = 45.0


DEFAULT_PLANNING_EVIDENCE_POLICY = PlanningEvidencePenaltyPolicy()


def support_score_for_level(level: object) -> float:
    normalized = str(level or "").strip() or "unknown"
    return float(SUPPORT_SCORE_MAP.get(normalized, SUPPORT_SCORE_MAP["unknown"]))


def classify_confidence_tier(score: object) -> str:
    value = float(pd.to_numeric(pd.Series([score]), errors="coerce").fillna(0.0).iloc[0])
    if value >= 0.75:
        return "high"
    if value >= 0.60:
        return "moderate"
    if value >= 0.40:
        return "guarded"
    return "low"


def classify_pathway_reliability(
    *,
    pathway: object,
    reliability_score: float,
    weak_or_unsupported_ratio: float,
) -> ReliabilityDecision:
    _ = pathway
    if (
        reliability_score < RELIABILITY_AUXILIARY_THRESHOLD
        or weak_or_unsupported_ratio >= WEAK_OR_UNSUPPORTED_AUXILIARY_THRESHOLD
    ):
        return ReliabilityDecision(
            tier="auxiliary_only",
            reviewer_sentence="Cross-study evidence is currently auxiliary and does not support strong generalization.",
        )
    if (
        reliability_score >= RELIABILITY_CONDITIONAL_SUPPORT_THRESHOLD
        and weak_or_unsupported_ratio <= 0.75
    ):
        return ReliabilityDecision(
            tier="conditional_support",
            reviewer_sentence="Cross-study evidence remains pathway-specific and should be written with claim discipline.",
        )
    return ReliabilityDecision(
        tier="limited_support",
        reviewer_sentence="Cross-study evidence is limited and should not be generalized without qualification.",
    )


def classify_recommendation_evidence_ceiling(
    *,
    claim_boundary: object,
    reliability_tier: object,
    selected: bool,
    confidence_tier: object,
) -> str:
    claim_boundary_text = str(claim_boundary or "")
    if "anchor" in claim_boundary_text:
        return "anchor_only"
    if "comparison only" in claim_boundary_text:
        return "comparison_only"

    reliability = str(reliability_tier or "").strip()
    confidence = str(confidence_tier or "").strip()

    if reliability == "auxiliary_only":
        return "auxiliary_only"
    if reliability == "limited_support":
        return "guarded_transfer"
    if reliability == "conditional_support":
        if selected and confidence in {"high", "moderate"}:
            return "conditional_transfer_supported"
        return "conditional_transfer_caution"
    if not reliability:
        return "not_evaluated"
    return "not_evaluated"


def classify_scenario_transferability_ceiling(
    *,
    weighted_score: float,
    auxiliary_share: float,
    limited_share: float,
    missing_share: float,
) -> str:
    if auxiliary_share > 0.25 or missing_share > 0.10:
        return "auxiliary_or_missing_bounded"
    if weighted_score >= 0.60 and limited_share <= 0.40:
        return "conditional_transfer_supported"
    if weighted_score >= 0.40:
        return "guarded_transfer"
    return "limited_transfer"


def build_transferability_note(
    *,
    evidence_ceiling: str,
) -> str:
    if evidence_ceiling == "conditional_transfer_supported":
        return (
            "Selected portfolio mass remains mostly aligned with pathways that retain conditional cross-study support."
        )
    if evidence_ceiling == "guarded_transfer":
        return (
            "Selected portfolio mass depends partly on pathways with limited transferability, so claims should remain guarded."
        )
    if evidence_ceiling == "auxiliary_or_missing_bounded":
        return (
            "Selected portfolio mass includes auxiliary-only or unresolved transferability support and should not be generalized strongly."
        )
    return (
        "Cross-study transferability remains limited for the selected portfolio mass, so recommendation language should stay conservative."
    )
