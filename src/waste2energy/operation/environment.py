# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, pi, sin
from typing import Any

from ..config import get_objective_weight_system


@dataclass(frozen=True)
class OperationEnvironmentSpec:
    scenario_name: str
    dominant_sample_id: str
    dominant_case_id: str
    manure_subtype: str
    pathway: str
    planned_temperature_c: float
    planned_residence_time_min: float
    planned_allocated_feed_ton_per_year: float
    scenario_feed_budget_ton_per_year: float
    effective_processing_budget_ton_per_year: float
    candidate_capacity_cap_ton_per_year: float
    scenario_candidate_share_lower_bound: float
    scenario_candidate_share_target: float
    scenario_candidate_share_upper_bound: float
    candidate_feed_lower_bound_ton_per_year: float
    candidate_feed_target_ton_per_year: float
    candidate_feed_upper_bound_ton_per_year: float
    nominal_energy_intensity_mj_per_ton: float
    nominal_environment_intensity_kgco2e_per_ton: float
    nominal_cost_intensity_proxy_or_real_per_ton: float
    energy_disturbance_amplitude: float
    environment_disturbance_amplitude: float
    cost_disturbance_amplitude: float
    coverage_disturbance_amplitude: float
    max_unmet_feed_ton_per_year: float
    dominant_selection_rate: float
    stable_candidate_count: int
    cross_scenario_selection_rate: float
    selected_in_all_scenarios: bool
    capacity_binding_reason: str
    objective_weight_preset: str = "balanced_cleaner_production"
    reward_energy_weight: float = 0.40
    reward_environment_weight: float = 0.35
    reward_cost_weight: float = 0.25


class OperationEnvironment:
    def __init__(
        self,
        spec: OperationEnvironmentSpec,
        *,
        horizon_steps: int = 12,
        throughput_step_fraction: float = 0.05,
        severity_step: int = 1,
        max_abs_severity_offset: int = 2,
    ) -> None:
        self.spec = spec
        self.horizon_steps = max(1, int(horizon_steps))
        self.throughput_step_ton = spec.candidate_capacity_cap_ton_per_year * throughput_step_fraction
        self.severity_step = max(1, int(severity_step))
        self.max_abs_severity_offset = max(1, int(max_abs_severity_offset))
        self._state: dict[str, float] = {}
        self._time_index = 0
        self._weights = get_objective_weight_system(
            preset_name=spec.objective_weight_preset,
            energy=spec.reward_energy_weight,
            environment=spec.reward_environment_weight,
            cost=spec.reward_cost_weight,
        )

    def reset(self) -> dict[str, float]:
        self._time_index = 0
        self._state = {
            "time_index": 0.0,
            "throughput_ton_per_year": self.spec.candidate_feed_target_ton_per_year,
            "severity_offset": 0.0,
        }
        self._state.update(self._build_state_view())
        return self._state.copy()

    def step(self, action: tuple[int, int]) -> tuple[dict[str, float], float, bool, dict[str, Any]]:
        if not self._state:
            self.reset()

        throughput_signal = _clip_int(action[0])
        severity_signal = _clip_int(action[1])

        next_throughput = self._state["throughput_ton_per_year"] + throughput_signal * self.throughput_step_ton
        next_throughput = _clip(
            next_throughput,
            0.0,
            self.spec.candidate_capacity_cap_ton_per_year,
        )
        next_severity = self._state["severity_offset"] + severity_signal * self.severity_step
        next_severity = float(
            _clip(
                next_severity,
                -self.max_abs_severity_offset,
                self.max_abs_severity_offset,
            )
        )

        self._time_index += 1
        self._state = {
            "time_index": float(self._time_index),
            "throughput_ton_per_year": next_throughput,
            "severity_offset": next_severity,
        }
        state_view = self._build_state_view()
        self._state.update(state_view)

        reward_components = self._compute_reward_components(
            throughput_ton_per_year=next_throughput,
            candidate_share=state_view["candidate_share_of_effective_budget"],
            severity_offset=next_severity,
            energy_multiplier=state_view["energy_disturbance_multiplier"],
            environment_multiplier=state_view["environment_disturbance_multiplier"],
            cost_multiplier=state_view["cost_disturbance_multiplier"],
            throughput_signal=throughput_signal,
            severity_signal=severity_signal,
        )
        reward = reward_components["reward"]
        done = self._time_index >= self.horizon_steps
        info = {"reward_components": reward_components, "spec": asdict(self.spec)}
        return self._state.copy(), reward, done, info

    def _build_state_view(self) -> dict[str, float]:
        phase = 2.0 * pi * min(self._time_index, self.horizon_steps) / self.horizon_steps
        throughput_ton = self._state["throughput_ton_per_year"]
        candidate_share = _safe_ratio(throughput_ton, self.spec.effective_processing_budget_ton_per_year)
        target_gap = candidate_share - self.spec.scenario_candidate_share_target

        energy_multiplier = 1.0 + self.spec.energy_disturbance_amplitude * sin(phase)
        environment_multiplier = 1.0 + self.spec.environment_disturbance_amplitude * cos(phase)
        cost_multiplier = 1.0 + self.spec.cost_disturbance_amplitude * sin(phase + 0.7)
        load_multiplier = 1.0 + self.spec.coverage_disturbance_amplitude * cos(phase + 0.3)

        return {
            "time_fraction": _safe_ratio(self._time_index, self.horizon_steps),
            "throughput_utilization_of_candidate_cap": _safe_ratio(
                throughput_ton, self.spec.candidate_capacity_cap_ton_per_year
            ),
            "candidate_share_of_effective_budget": candidate_share,
            "candidate_share_target_gap": target_gap,
            "severity_offset": self._state["severity_offset"],
            "energy_disturbance_multiplier": energy_multiplier,
            "environment_disturbance_multiplier": environment_multiplier,
            "cost_disturbance_multiplier": cost_multiplier,
            "load_disturbance_multiplier": load_multiplier,
            "capacity_pressure": max(0.0, candidate_share - self.spec.scenario_candidate_share_upper_bound),
            "coverage_pressure": max(0.0, self.spec.scenario_candidate_share_lower_bound - candidate_share),
        }

    def _compute_reward_components(
        self,
        *,
        throughput_ton_per_year: float,
        candidate_share: float,
        severity_offset: float,
        energy_multiplier: float,
        environment_multiplier: float,
        cost_multiplier: float,
        throughput_signal: int,
        severity_signal: int,
    ) -> dict[str, float]:
        energy_effect = max(0.80, 1.0 + 0.04 * severity_offset)
        environment_effect = max(0.75, 1.0 - 0.03 * severity_offset)
        cost_effect = 1.0 + 0.05 * abs(severity_offset) + 0.02 * max(severity_offset, 0.0)

        realized_energy = (
            throughput_ton_per_year
            * self.spec.nominal_energy_intensity_mj_per_ton
            * energy_multiplier
            * energy_effect
        )
        realized_environment = (
            throughput_ton_per_year
            * self.spec.nominal_environment_intensity_kgco2e_per_ton
            * environment_multiplier
            * environment_effect
        )
        realized_cost = (
            throughput_ton_per_year
            * self.spec.nominal_cost_intensity_proxy_or_real_per_ton
            * cost_multiplier
            * cost_effect
        )

        nominal_energy = (
            self.spec.candidate_feed_target_ton_per_year * self.spec.nominal_energy_intensity_mj_per_ton
        )
        nominal_environment = (
            self.spec.candidate_feed_target_ton_per_year
            * self.spec.nominal_environment_intensity_kgco2e_per_ton
        )
        nominal_cost = (
            self.spec.candidate_feed_target_ton_per_year
            * self.spec.nominal_cost_intensity_proxy_or_real_per_ton
        )

        energy_term = _safe_ratio(realized_energy, nominal_energy)
        environment_term = _safe_ratio(realized_environment, nominal_environment)
        cost_term = _safe_ratio(realized_cost, nominal_cost)

        lower_violation = max(0.0, self.spec.scenario_candidate_share_lower_bound - candidate_share)
        upper_violation = max(0.0, candidate_share - self.spec.scenario_candidate_share_upper_bound)
        violation_penalty = 6.0 * (lower_violation + upper_violation)
        switching_penalty = 0.03 * (abs(throughput_signal) + abs(severity_signal))

        reward = (
            self._weights.energy * energy_term
            + self._weights.environment * environment_term
            - self._weights.cost * cost_term
            - violation_penalty
            - switching_penalty
        )
        return {
            "realized_energy": realized_energy,
            "realized_environment": realized_environment,
            "realized_cost": realized_cost,
            "energy_term": energy_term,
            "environment_term": environment_term,
            "cost_term": cost_term,
            "reward_energy_weight": self._weights.energy,
            "reward_environment_weight": self._weights.environment,
            "reward_cost_weight": self._weights.cost,
            "violation_penalty": violation_penalty,
            "switching_penalty": switching_penalty,
            "reward": reward,
        }


def spec_from_row(row: dict[str, Any]) -> OperationEnvironmentSpec:
    return OperationEnvironmentSpec(**row)


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _clip_int(value: Any) -> int:
    try:
        return int(_clip(int(value), -1, 1))
    except Exception:
        return 0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator

