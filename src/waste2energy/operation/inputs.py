# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from ..common import parse_manifest_timestamp
from ..config import PLANNING_OUTPUTS_DIR, SCENARIO_OUTPUTS_DIR, get_objective_weight_system


@dataclass(frozen=True)
class OperationInputBundle:
    planning_portfolio_allocations: pd.DataFrame
    planning_constraints: pd.DataFrame
    scenario_decision_stability: pd.DataFrame
    scenario_uncertainty_summary: pd.DataFrame
    scenario_cross_stability: pd.DataFrame
    planning_run_config: dict[str, object]
    scenario_run_config: dict[str, object]


def load_operation_input_bundle(
    planning_dir: str | Path | None = None,
    scenario_dir: str | Path | None = None,
) -> OperationInputBundle:
    planning_root = Path(planning_dir) if planning_dir else PLANNING_OUTPUTS_DIR / "baseline"
    scenario_root = Path(scenario_dir) if scenario_dir else SCENARIO_OUTPUTS_DIR / "baseline"

    files = {
        "planning_portfolio_allocations": planning_root / "portfolio_allocations.csv",
        "planning_constraints": planning_root / "scenario_constraints.csv",
        "planning_run_config": planning_root / "run_config.json",
        "scenario_decision_stability": scenario_root / "decision_stability.csv",
        "scenario_uncertainty_summary": scenario_root / "uncertainty_summary.csv",
        "scenario_cross_stability": scenario_root / "cross_scenario_stability.csv",
        "scenario_run_config": scenario_root / "run_config.json",
    }
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Required operation-layer input '{name}' not found: {path}")

    return OperationInputBundle(
        planning_portfolio_allocations=pd.read_csv(files["planning_portfolio_allocations"]),
        planning_constraints=pd.read_csv(files["planning_constraints"]),
        scenario_decision_stability=pd.read_csv(files["scenario_decision_stability"]),
        scenario_uncertainty_summary=pd.read_csv(files["scenario_uncertainty_summary"]),
        scenario_cross_stability=pd.read_csv(files["scenario_cross_stability"]),
        planning_run_config=json.loads(files["planning_run_config"].read_text(encoding="utf-8")),
        scenario_run_config=json.loads(files["scenario_run_config"].read_text(encoding="utf-8")),
    )


def build_operation_environment_specs(
    planning_dir: str | Path | None = None,
    scenario_dir: str | Path | None = None,
) -> pd.DataFrame:
    bundle = load_operation_input_bundle(planning_dir=planning_dir, scenario_dir=scenario_dir)
    _validate_operation_input_freshness(bundle)

    uncertainty = bundle.scenario_uncertainty_summary.copy()
    decision = bundle.scenario_decision_stability.copy()
    allocations = bundle.planning_portfolio_allocations.copy()
    constraints = bundle.planning_constraints.copy()
    cross_stability = bundle.scenario_cross_stability.copy()
    weight_system = _extract_weight_system(bundle.planning_run_config)

    decision_map = decision.set_index(["scenario_name", "sample_id"]).to_dict("index")
    constraint_map = constraints.set_index("scenario_name").to_dict("index")
    cross_map = cross_stability.set_index("sample_id").to_dict("index")
    global_refs = _build_global_physical_references(allocations)

    rows: list[dict[str, object]] = []
    for record in _select_operation_anchor_rows(
        uncertainty=uncertainty,
        allocations=allocations,
        decision_map=decision_map,
        constraint_map=constraint_map,
        cross_map=cross_map,
    ):
        summary = record["summary"]
        allocation = record["allocation"]
        scenario_name = str(summary["scenario_name"])
        dominant_sample_id = str(allocation.get("sample_id", ""))
        decision_row = record["decision_row"]
        constraint_row = record["constraint_row"]
        cross_row = record["cross_row"]

        effective_budget = _value(allocation, "effective_processing_budget_ton_per_year")
        candidate_capacity_cap = _value(allocation, "candidate_capacity_cap_ton_per_year")
        baseline_share = _safe_ratio(_value(allocation, "allocated_feed_ton_per_year"), effective_budget)
        avg_share = _value_from_mapping(decision_row, "avg_allocated_feed_share", baseline_share)
        max_share = _value_from_mapping(decision_row, "max_allocated_feed_share", baseline_share)
        coverage_range_ratio = _value(summary, "coverage_range_ratio")
        candidate_share_cap = _safe_ratio(candidate_capacity_cap, effective_budget)
        avg_share_source = (
            "decision_stability" if "avg_allocated_feed_share" in decision_row else "baseline_allocation_share_fallback"
        )
        max_share_source = (
            "decision_stability" if "max_allocated_feed_share" in decision_row else "baseline_allocation_share_fallback"
        )
        cross_scenario_selection_rate, cross_scenario_selection_rate_source = _value_from_mapping_with_source(
            cross_row,
            "cross_scenario_selection_rate",
            0.0,
            fallback_source="cross_scenario_default_zero",
        )
        selected_in_all_scenarios, selected_in_all_scenarios_source = _value_from_mapping_with_source(
            cross_row,
            "selected_in_all_scenarios",
            False,
            fallback_source="cross_scenario_default_false",
        )
        energy_range_ratio = _value(summary, "energy_range_ratio")
        environment_range_ratio = _value(summary, "environment_range_ratio")
        cost_range_ratio = _value(summary, "cost_range_ratio")
        energy_disturbance_amplitude = _clip(energy_range_ratio, 0.05, 0.35)
        environment_disturbance_amplitude = _clip(environment_range_ratio, 0.05, 0.35)
        cost_disturbance_amplitude = _clip(cost_range_ratio, 0.05, 0.35)
        coverage_disturbance_amplitude = _clip(coverage_range_ratio, 0.05, 0.35)

        lower_share = max(0.05, avg_share * max(0.50, 1.0 - coverage_range_ratio))
        upper_share = min(
            max(0.05, candidate_share_cap if candidate_share_cap > 0.0 else 1.0),
            max(max_share, baseline_share, avg_share + coverage_range_ratio * max(avg_share, 0.10)),
        )
        lower_share = min(lower_share, upper_share)
        target_share = min(max(avg_share, lower_share), upper_share)

        rows.append(
            {
                "scenario_name": scenario_name,
                "dominant_sample_id": dominant_sample_id,
                "dominant_case_id": allocation["optimization_case_id"],
                "manure_subtype": allocation.get("manure_subtype", ""),
                "pathway": allocation.get("pathway", ""),
                "planned_temperature_c": _value(allocation, "process_temperature_c"),
                "planned_residence_time_min": _value(allocation, "residence_time_min"),
                "planned_allocated_feed_ton_per_year": _value(allocation, "allocated_feed_ton_per_year"),
                "scenario_feed_budget_ton_per_year": _value(allocation, "scenario_feed_budget_ton_per_year"),
                "effective_processing_budget_ton_per_year": effective_budget,
                "candidate_capacity_cap_ton_per_year": candidate_capacity_cap,
                "avg_share_source": avg_share_source,
                "max_share_source": max_share_source,
                "scenario_candidate_share_lower_bound": lower_share,
                "scenario_candidate_share_target": target_share,
                "scenario_candidate_share_upper_bound": upper_share,
                "candidate_feed_lower_bound_ton_per_year": lower_share * effective_budget,
                "candidate_feed_target_ton_per_year": target_share * effective_budget,
                "candidate_feed_upper_bound_ton_per_year": upper_share * effective_budget,
                "nominal_energy_intensity_mj_per_ton": _value(
                    allocation, "planning_energy_intensity_mj_per_ton"
                ),
                "nominal_environment_intensity_kgco2e_per_ton": _value(
                    allocation, "planning_environment_intensity_kgco2e_per_ton"
                ),
                "nominal_carbon_load_kgco2e_per_ton": _value(
                    allocation, "planning_carbon_load_kgco2e_per_ton"
                ),
                "nominal_cost_intensity_proxy_or_real_per_ton": _value(
                    allocation, "planning_cost_intensity_proxy_or_real_per_ton"
                ),
                "global_energy_reference_mj_per_year": global_refs["energy"],
                "global_net_environment_reference_kgco2e_per_year": global_refs["net_environment"],
                "global_cost_reference_proxy_or_real_per_year": global_refs["cost"],
                "global_carbon_load_reference_kgco2e_per_year": global_refs["carbon_load"],
                "energy_disturbance_amplitude": energy_disturbance_amplitude,
                "environment_disturbance_amplitude": environment_disturbance_amplitude,
                "cost_disturbance_amplitude": cost_disturbance_amplitude,
                "coverage_disturbance_amplitude": coverage_disturbance_amplitude,
                "energy_disturbance_source": _clip_source(energy_range_ratio, 0.05, 0.35),
                "environment_disturbance_source": _clip_source(environment_range_ratio, 0.05, 0.35),
                "cost_disturbance_source": _clip_source(cost_range_ratio, 0.05, 0.35),
                "coverage_disturbance_source": _clip_source(coverage_range_ratio, 0.05, 0.35),
                "market_noise_amplitude": _clip(max(cost_range_ratio, energy_range_ratio) * 0.08, 0.02, 0.15),
                "feed_quality_noise_amplitude": _clip(coverage_range_ratio * 0.30, 0.02, 0.18),
                "carbon_noise_amplitude": _clip(environment_range_ratio * 0.08, 0.02, 0.15),
                "noise_seed_base": _stable_seed(f"{scenario_name}|{dominant_sample_id}"),
                "max_unmet_feed_ton_per_year": _value(summary, "unmet_feed_max"),
                "dominant_selection_rate": _value(summary, "dominant_selection_rate"),
                "stable_candidate_count": int(_value(summary, "stable_candidate_count")),
                "cross_scenario_selection_rate": cross_scenario_selection_rate,
                "cross_scenario_selection_rate_source": cross_scenario_selection_rate_source,
                "selected_in_all_scenarios": bool(selected_in_all_scenarios),
                "selected_in_all_scenarios_source": selected_in_all_scenarios_source,
                "capacity_binding_reason": str(constraint_row.get("capacity_binding_reason", "")),
                "anchor_selection_score": float(record["anchor_score"]),
                "objective_weight_preset": weight_system.preset_name,
                "reward_energy_weight": weight_system.energy,
                "reward_environment_weight": weight_system.environment,
                "reward_cost_weight": weight_system.cost,
            }
        )

    return pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)


def _extract_weight_system(run_config: dict[str, object]):
    payload = run_config.get("objective_weights") or {}
    weights = payload.get("weights") or {}
    return get_objective_weight_system(
        preset_name=str(payload.get("preset_name", "balanced_cleaner_production")),
        energy=_coerce_float(weights.get("energy"), default=None),
        environment=_coerce_float(weights.get("environment"), default=None),
        cost=_coerce_float(weights.get("cost"), default=None),
    )


def _select_operation_anchor_rows(
    *,
    uncertainty: pd.DataFrame,
    allocations: pd.DataFrame,
    decision_map: dict[tuple[str, str], dict[str, object]],
    constraint_map: dict[str, dict[str, object]],
    cross_map: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    used_sample_ids: set[str] = set()
    selected_anchor_rows: list[pd.Series] = []
    for _, summary in uncertainty.sort_values("scenario_name").iterrows():
        scenario_name = str(summary["scenario_name"])
        scenario_allocations = allocations[allocations["scenario_name"] == scenario_name].copy()
        if scenario_allocations.empty:
            continue
        candidates: list[dict[str, object]] = []
        for _, allocation in scenario_allocations.iterrows():
            sample_id = str(allocation.get("sample_id", ""))
            decision_row = decision_map.get((scenario_name, sample_id), {})
            cross_row = cross_map.get(sample_id, {})
            diversity_score = _anchor_diversity_score(allocation, selected_anchor_rows)
            baseline_share = _safe_ratio(
                _value(allocation, "allocated_feed_ton_per_year"),
                _value(allocation, "effective_processing_budget_ton_per_year"),
            )
            decision_share = _value_from_mapping(decision_row, "avg_allocated_feed_share", baseline_share)
            anchor_score = (
                0.60 * baseline_share
                + 0.20 * decision_share
                + 0.15 * (1.0 - _value_from_mapping(cross_row, "cross_scenario_selection_rate", 0.0))
                + 0.05 * diversity_score
            )
            candidates.append(
                {
                    "summary": summary,
                    "allocation": allocation,
                    "decision_row": decision_row,
                    "constraint_row": constraint_map.get(scenario_name, {}),
                    "cross_row": cross_row,
                    "sample_id": sample_id,
                    "anchor_score": float(anchor_score),
                }
            )
        unique_candidates = [candidate for candidate in candidates if candidate["sample_id"] not in used_sample_ids]
        ranked_candidates = unique_candidates or candidates
        ranked_candidates.sort(
            key=lambda item: (
                item["anchor_score"],
                _value(item["allocation"], "allocated_feed_ton_per_year"),
                _value(item["allocation"], "planning_environment_intensity_kgco2e_per_ton"),
            ),
            reverse=True,
        )
        if not ranked_candidates:
            continue
        chosen = ranked_candidates[0]
        used_sample_ids.add(chosen["sample_id"])
        selected_anchor_rows.append(chosen["allocation"])
        selected.append(chosen)
    return selected


def _anchor_diversity_score(candidate: pd.Series, selected_anchor_rows: list[pd.Series]) -> float:
    if not selected_anchor_rows:
        return 0.0
    distances: list[float] = []
    candidate_energy = _value(candidate, "planning_energy_intensity_mj_per_ton")
    candidate_environment = _value(candidate, "planning_environment_intensity_kgco2e_per_ton")
    candidate_cost = _value(candidate, "planning_cost_intensity_proxy_or_real_per_ton")
    for row in selected_anchor_rows:
        distances.append(
            abs(candidate_energy - _value(row, "planning_energy_intensity_mj_per_ton"))
            + abs(candidate_environment - _value(row, "planning_environment_intensity_kgco2e_per_ton"))
            + abs(candidate_cost - _value(row, "planning_cost_intensity_proxy_or_real_per_ton"))
        )
    return float(sum(distances) / max(len(distances), 1))


def _build_global_physical_references(allocations: pd.DataFrame) -> dict[str, float]:
    if allocations.empty:
        return {
            "energy": 1.0,
            "net_environment": 1.0,
            "cost": 1.0,
            "carbon_load": 1.0,
        }
    _require_numeric_columns(
        allocations,
        columns=(
            "allocated_feed_ton_per_year",
            "planning_energy_intensity_mj_per_ton",
            "planning_environment_intensity_kgco2e_per_ton",
            "planning_carbon_load_kgco2e_per_ton",
            "planning_cost_intensity_proxy_or_real_per_ton",
        ),
        context="operation global physical reference construction",
    )
    allocated_feed = pd.to_numeric(allocations["allocated_feed_ton_per_year"], errors="coerce")
    energy = (
        allocated_feed
        * pd.to_numeric(allocations["planning_energy_intensity_mj_per_ton"], errors="coerce")
    ).abs()
    environment = (
        allocated_feed
        * pd.to_numeric(allocations["planning_environment_intensity_kgco2e_per_ton"], errors="coerce")
    )
    carbon_load = (
        allocated_feed
        * pd.to_numeric(allocations["planning_carbon_load_kgco2e_per_ton"], errors="coerce")
    )
    net_environment = (environment - carbon_load).abs()
    cost = (
        allocated_feed
        * pd.to_numeric(allocations["planning_cost_intensity_proxy_or_real_per_ton"], errors="coerce")
    ).abs()
    return {
        "energy": max(float(energy.max()), 1.0),
        "net_environment": max(float(net_environment.max()), 1.0),
        "cost": max(float(cost.max()), 1.0),
        "carbon_load": max(float(carbon_load.abs().max()), 1.0),
    }


def _stable_seed(value: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(value))


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _clip_source(value: float, minimum: float, maximum: float) -> str:
    if value < minimum:
        return "lower_clip"
    if value > maximum:
        return "upper_clip"
    return "direct"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _value(row: pd.Series, column: str) -> float:
    if column not in row.index:
        raise ValueError(f"Operation input is missing required column '{column}'.")
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    if pd.isna(value):
        scenario_name = str(row.get("scenario_name", "unknown_scenario"))
        case_id = str(row.get("optimization_case_id", row.get("dominant_case_id", "unknown_case")))
        raise ValueError(
            f"Operation input for scenario '{scenario_name}' case '{case_id}' contains missing/non-numeric value in '{column}'."
        )
    return float(value)


def _value_from_mapping(mapping: dict[str, object], key: str, default: float | bool) -> float | bool:
    value = mapping.get(key, default)
    if isinstance(default, bool):
        return bool(value)
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"Operation mapping contains missing/non-numeric value in '{key}'.")
    return float(numeric)


def _value_from_mapping_with_source(
    mapping: dict[str, object],
    key: str,
    default: float | bool,
    *,
    fallback_source: str,
) -> tuple[float | bool, str]:
    if key in mapping:
        return _value_from_mapping(mapping, key, default), "mapped_value"
    return default, fallback_source


def _coerce_float(value: object, default: float | None = 0.0) -> float | None:
    if value is None and default is None:
        return None
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _validate_operation_input_freshness(bundle: OperationInputBundle) -> None:
    planning_timestamp = parse_manifest_timestamp(bundle.planning_run_config)
    scenario_timestamp = parse_manifest_timestamp(bundle.scenario_run_config)
    if planning_timestamp is None or scenario_timestamp is None:
        return
    if scenario_timestamp + timedelta(seconds=1) < planning_timestamp:
        raise ValueError(
            "Scenario outputs are older than the planning outputs used by the operation layer. "
            "Re-run 'waste2energy-scenario' after regenerating planning outputs."
        )


def _require_numeric_columns(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    context: str,
) -> None:
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"{context} requires column '{column}', but it is missing.")
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            invalid_rows = frame.loc[values.isna(), "optimization_case_id"] if "optimization_case_id" in frame.columns else frame.index[values.isna()]
            preview = ", ".join(str(value) for value in list(invalid_rows[:5]))
            raise ValueError(
                f"{context} encountered missing/non-numeric values in '{column}' for row(s): {preview}."
            )
