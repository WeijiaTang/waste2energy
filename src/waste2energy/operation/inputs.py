from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import PLANNING_OUTPUTS_DIR, SCENARIO_OUTPUTS_DIR


@dataclass(frozen=True)
class OperationInputBundle:
    planning_portfolio_allocations: pd.DataFrame
    planning_constraints: pd.DataFrame
    scenario_decision_stability: pd.DataFrame
    scenario_uncertainty_summary: pd.DataFrame
    scenario_cross_stability: pd.DataFrame


def load_operation_input_bundle(
    planning_dir: str | Path | None = None,
    scenario_dir: str | Path | None = None,
) -> OperationInputBundle:
    planning_root = Path(planning_dir) if planning_dir else PLANNING_OUTPUTS_DIR / "baseline"
    scenario_root = Path(scenario_dir) if scenario_dir else SCENARIO_OUTPUTS_DIR / "baseline"

    files = {
        "planning_portfolio_allocations": planning_root / "portfolio_allocations.csv",
        "planning_constraints": planning_root / "scenario_constraints.csv",
        "scenario_decision_stability": scenario_root / "decision_stability.csv",
        "scenario_uncertainty_summary": scenario_root / "uncertainty_summary.csv",
        "scenario_cross_stability": scenario_root / "cross_scenario_stability.csv",
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
    )


def build_operation_environment_specs(
    planning_dir: str | Path | None = None,
    scenario_dir: str | Path | None = None,
) -> pd.DataFrame:
    bundle = load_operation_input_bundle(planning_dir=planning_dir, scenario_dir=scenario_dir)

    uncertainty = bundle.scenario_uncertainty_summary.copy()
    decision = bundle.scenario_decision_stability.copy()
    allocations = bundle.planning_portfolio_allocations.copy()
    constraints = bundle.planning_constraints.copy()
    cross_stability = bundle.scenario_cross_stability.copy()

    decision_map = decision.set_index(["scenario_name", "sample_id"]).to_dict("index")
    constraint_map = constraints.set_index("scenario_name").to_dict("index")
    cross_map = cross_stability.set_index("sample_id").to_dict("index")

    rows: list[dict[str, object]] = []
    for _, summary in uncertainty.iterrows():
        scenario_name = str(summary["scenario_name"])
        dominant_sample_id = str(summary["dominant_sample_id"])
        allocation_rows = allocations[
            (allocations["scenario_name"] == scenario_name)
            & (allocations["sample_id"] == dominant_sample_id)
        ].copy()
        if allocation_rows.empty:
            continue

        allocation = allocation_rows.iloc[0]
        decision_row = decision_map.get((scenario_name, dominant_sample_id), {})
        constraint_row = constraint_map.get(scenario_name, {})
        cross_row = cross_map.get(dominant_sample_id, {})

        effective_budget = _value(allocation, "effective_processing_budget_ton_per_year")
        baseline_share = _safe_ratio(_value(allocation, "allocated_feed_ton_per_year"), effective_budget)
        avg_share = _value_from_mapping(decision_row, "avg_allocated_feed_share", baseline_share)
        max_share = _value_from_mapping(decision_row, "max_allocated_feed_share", baseline_share)
        coverage_range_ratio = _value(summary, "coverage_range_ratio")

        lower_share = max(0.05, avg_share * max(0.50, 1.0 - coverage_range_ratio))
        upper_share = min(
            1.0,
            max(max_share, baseline_share, avg_share + coverage_range_ratio * max(avg_share, 0.10)),
        )
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
                "candidate_capacity_cap_ton_per_year": _value(
                    allocation, "candidate_capacity_cap_ton_per_year"
                ),
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
                "nominal_cost_intensity_proxy_or_real_per_ton": _value(
                    allocation, "planning_cost_intensity_proxy_or_real_per_ton"
                ),
                "energy_disturbance_amplitude": _clip(_value(summary, "energy_range_ratio"), 0.05, 0.35),
                "environment_disturbance_amplitude": _clip(
                    _value(summary, "environment_range_ratio"), 0.05, 0.35
                ),
                "cost_disturbance_amplitude": _clip(_value(summary, "cost_range_ratio"), 0.05, 0.35),
                "coverage_disturbance_amplitude": _clip(coverage_range_ratio, 0.05, 0.35),
                "max_unmet_feed_ton_per_year": _value(summary, "unmet_feed_max"),
                "dominant_selection_rate": _value(summary, "dominant_selection_rate"),
                "stable_candidate_count": int(_value(summary, "stable_candidate_count")),
                "cross_scenario_selection_rate": _value_from_mapping(
                    cross_row, "cross_scenario_selection_rate", 0.0
                ),
                "selected_in_all_scenarios": bool(
                    _value_from_mapping(cross_row, "selected_in_all_scenarios", False)
                ),
                "capacity_binding_reason": str(
                    constraint_row.get("capacity_binding_reason", "")
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _value(row: pd.Series, column: str) -> float:
    if column not in row.index:
        return 0.0
    return float(pd.to_numeric(pd.Series([row[column]]), errors="coerce").fillna(0.0).iloc[0])


def _value_from_mapping(mapping: dict[str, object], key: str, default: float | bool) -> float | bool:
    value = mapping.get(key, default)
    if isinstance(default, bool):
        return bool(value)
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(default).iloc[0])
