# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi
from typing import Any

import numpy as np

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
    avg_share_source: str
    max_share_source: str
    scenario_candidate_share_lower_bound: float
    scenario_candidate_share_target: float
    scenario_candidate_share_upper_bound: float
    candidate_feed_lower_bound_ton_per_year: float
    candidate_feed_target_ton_per_year: float
    candidate_feed_upper_bound_ton_per_year: float
    nominal_energy_intensity_mj_per_ton: float
    nominal_environment_intensity_kgco2e_per_ton: float
    nominal_carbon_load_kgco2e_per_ton: float
    nominal_cost_intensity_proxy_or_real_per_ton: float
    global_energy_reference_mj_per_year: float
    global_net_environment_reference_kgco2e_per_year: float
    global_cost_reference_proxy_or_real_per_year: float
    global_carbon_load_reference_kgco2e_per_year: float
    energy_disturbance_amplitude: float
    environment_disturbance_amplitude: float
    cost_disturbance_amplitude: float
    coverage_disturbance_amplitude: float
    energy_disturbance_source: str
    environment_disturbance_source: str
    cost_disturbance_source: str
    coverage_disturbance_source: str
    market_noise_amplitude: float
    feed_quality_noise_amplitude: float
    carbon_noise_amplitude: float
    noise_seed_base: int
    max_unmet_feed_ton_per_year: float
    dominant_selection_rate: float
    stable_candidate_count: int
    cross_scenario_selection_rate: float
    cross_scenario_selection_rate_source: str
    selected_in_all_scenarios: bool
    selected_in_all_scenarios_source: str
    capacity_binding_reason: str
    anchor_selection_score: float
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
        self._disturbance_cache = self._build_disturbance_cache()
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
            market_noise_multiplier=state_view["market_noise_multiplier"],
            feed_quality_multiplier=state_view["feed_quality_multiplier"],
            carbon_noise_multiplier=state_view["carbon_noise_multiplier"],
            throughput_signal=throughput_signal,
            severity_signal=severity_signal,
        )
        reward = reward_components["reward"]
        done = self._time_index >= self.horizon_steps
        info = {"reward_components": reward_components, "spec": asdict(self.spec)}
        return self._state.copy(), reward, done, info

    def _build_state_view(self) -> dict[str, float]:
        throughput_ton = self._state["throughput_ton_per_year"]
        candidate_share = _safe_ratio(throughput_ton, self.spec.effective_processing_budget_ton_per_year)
        target_gap = candidate_share - self.spec.scenario_candidate_share_target
        disturbance = self._disturbance_cache[min(self._time_index, self.horizon_steps - 1)]

        return {
            "time_fraction": _safe_ratio(self._time_index, self.horizon_steps),
            "throughput_utilization_of_candidate_cap": _safe_ratio(
                throughput_ton, self.spec.candidate_capacity_cap_ton_per_year
            ),
            "candidate_share_of_effective_budget": candidate_share,
            "candidate_share_target_gap": target_gap,
            "severity_offset": self._state["severity_offset"],
            "energy_disturbance_multiplier": disturbance["energy_disturbance_multiplier"],
            "environment_disturbance_multiplier": disturbance["environment_disturbance_multiplier"],
            "cost_disturbance_multiplier": disturbance["cost_disturbance_multiplier"],
            "load_disturbance_multiplier": disturbance["load_disturbance_multiplier"],
            "market_noise_multiplier": disturbance["market_noise_multiplier"],
            "feed_quality_multiplier": disturbance["feed_quality_multiplier"],
            "carbon_noise_multiplier": disturbance["carbon_noise_multiplier"],
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
        market_noise_multiplier: float,
        feed_quality_multiplier: float,
        carbon_noise_multiplier: float,
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
            * feed_quality_multiplier
        )
        realized_environment = (
            throughput_ton_per_year
            * self.spec.nominal_environment_intensity_kgco2e_per_ton
            * environment_multiplier
            * environment_effect
            * carbon_noise_multiplier
        )
        realized_carbon_load = (
            throughput_ton_per_year
            * self.spec.nominal_carbon_load_kgco2e_per_ton
            * max(0.70, carbon_noise_multiplier)
        )
        realized_cost = (
            throughput_ton_per_year
            * self.spec.nominal_cost_intensity_proxy_or_real_per_ton
            * cost_multiplier
            * cost_effect
            * market_noise_multiplier
        )
        net_environment_benefit = max(0.0, realized_environment - realized_carbon_load)
        energy_term = realized_energy / max(self.spec.global_energy_reference_mj_per_year, 1.0)
        environment_term = net_environment_benefit / max(
            self.spec.global_net_environment_reference_kgco2e_per_year,
            1.0,
        )
        cost_term = realized_cost / max(self.spec.global_cost_reference_proxy_or_real_per_year, 1.0)
        carbon_load_term = realized_carbon_load / max(
            self.spec.global_carbon_load_reference_kgco2e_per_year,
            1.0,
        )

        lower_violation = max(0.0, self.spec.scenario_candidate_share_lower_bound - candidate_share)
        upper_violation = max(0.0, candidate_share - self.spec.scenario_candidate_share_upper_bound)
        violation_penalty = 6.0 * (lower_violation + upper_violation)
        switching_penalty = 0.03 * (abs(throughput_signal) + abs(severity_signal))
        violation_indicator = 1.0 if violation_penalty > 0.0 else 0.0
        resilience_index = max(0.0, 1.0 - violation_indicator - 0.10 * abs(severity_offset))

        reward = (
            self._weights.energy * energy_term
            + self._weights.environment * environment_term
            - self._weights.cost * cost_term
            - 0.30 * carbon_load_term
            - violation_penalty
            - switching_penalty
        )
        return {
            "realized_energy": realized_energy,
            "realized_environment": realized_environment,
            "realized_carbon_load": realized_carbon_load,
            "realized_net_environment_benefit": net_environment_benefit,
            "realized_cost": realized_cost,
            "energy_term": energy_term,
            "environment_term": environment_term,
            "cost_term": cost_term,
            "carbon_load_term": carbon_load_term,
            "reward_energy_weight": self._weights.energy,
            "reward_environment_weight": self._weights.environment,
            "reward_cost_weight": self._weights.cost,
            "violation_penalty": violation_penalty,
            "violation_indicator": violation_indicator,
            "resilience_index": resilience_index,
            "switching_penalty": switching_penalty,
            "reward": reward,
        }

    def _build_disturbance_cache(self) -> list[dict[str, float]]:
        rng = np.random.default_rng(int(self.spec.noise_seed_base))
        indices = np.arange(self.horizon_steps, dtype=float)
        seasonal = np.sin(2.0 * pi * indices / max(self.horizon_steps, 1))
        weekly = np.sin(2.0 * pi * indices / 168.0)
        market_shock = rng.weibull(2.0, self.horizon_steps)
        feed_quality = rng.weibull(2.6, self.horizon_steps)
        carbon_signal = rng.weibull(2.3, self.horizon_steps)

        market_noise = 1.0 + self.spec.market_noise_amplitude * ((market_shock / np.maximum(market_shock.mean(), 1e-9)) - 1.0)
        feed_noise = 1.0 + self.spec.feed_quality_noise_amplitude * ((feed_quality / np.maximum(feed_quality.mean(), 1e-9)) - 1.0)
        carbon_noise = 1.0 + self.spec.carbon_noise_amplitude * ((carbon_signal / np.maximum(carbon_signal.mean(), 1e-9)) - 1.0)

        energy = 1.0 + self.spec.energy_disturbance_amplitude * (0.6 * seasonal + 0.4 * weekly)
        environment = 1.0 + self.spec.environment_disturbance_amplitude * (0.5 * np.cos(2.0 * pi * indices / max(self.horizon_steps, 1)) + 0.5 * weekly)
        cost = 1.0 + self.spec.cost_disturbance_amplitude * (0.5 * seasonal + 0.5 * (market_noise - 1.0))
        load = 1.0 + self.spec.coverage_disturbance_amplitude * (0.7 * weekly + 0.3 * (feed_noise - 1.0))

        return [
            {
                "energy_disturbance_multiplier": float(np.clip(energy[i], 0.70, 1.35)),
                "environment_disturbance_multiplier": float(np.clip(environment[i], 0.70, 1.35)),
                "cost_disturbance_multiplier": float(np.clip(cost[i], 0.75, 1.45)),
                "load_disturbance_multiplier": float(np.clip(load[i], 0.75, 1.40)),
                "market_noise_multiplier": float(np.clip(market_noise[i], 0.80, 1.35)),
                "feed_quality_multiplier": float(np.clip(feed_noise[i], 0.80, 1.30)),
                "carbon_noise_multiplier": float(np.clip(carbon_noise[i], 0.80, 1.30)),
            }
            for i in range(self.horizon_steps)
        ]


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
