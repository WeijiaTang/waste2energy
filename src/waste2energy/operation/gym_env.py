from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .environment import OperationEnvironment, OperationEnvironmentSpec

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover - import guard for optional dependency
    gym = None
    spaces = None
    _GYM_IMPORT_ERROR = exc
else:
    _GYM_IMPORT_ERROR = None


class Waste2EnergyGymEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(
        self,
        spec: OperationEnvironmentSpec,
        *,
        horizon_steps: int = 12,
    ) -> None:
        _require_gymnasium()
        self.spec = spec
        self.horizon_steps = horizon_steps
        self.core = OperationEnvironment(spec, horizon_steps=horizon_steps)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -2.0, 0.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 2.0, 1.5, 1.5, 1.5, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._last_state: dict[str, float] | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._last_state = self.core.reset()
        return self._encode_observation(self._last_state), {"spec": asdict(self.spec), "state": self._last_state.copy()}

    def step(self, action):
        discrete_action = self._decode_action(np.asarray(action, dtype=np.float32))
        next_state, reward, done, info = self.core.step(discrete_action)
        self._last_state = next_state
        observation = self._encode_observation(next_state)
        terminated = False
        truncated = bool(done)
        info = dict(info)
        info["discrete_action"] = discrete_action
        info["state"] = next_state.copy()
        return observation, float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None

    def _encode_observation(self, state: dict[str, float]) -> np.ndarray:
        return np.array(
            [
                state["time_fraction"],
                state["throughput_utilization_of_candidate_cap"],
                state["candidate_share_target_gap"],
                state["severity_offset"] / max(1.0, float(self.core.max_abs_severity_offset)),
                state["energy_disturbance_multiplier"],
                state["environment_disturbance_multiplier"],
                state["cost_disturbance_multiplier"],
                state["capacity_pressure"],
                state["coverage_pressure"],
            ],
            dtype=np.float32,
        )

    def _decode_action(self, action: np.ndarray) -> tuple[int, int]:
        clipped = np.clip(action, -1.0, 1.0)
        throughput_signal = _to_discrete_signal(float(clipped[0]))
        severity_signal = _to_discrete_signal(float(clipped[1]))
        return throughput_signal, severity_signal


def build_gym_env(spec: OperationEnvironmentSpec, *, horizon_steps: int = 12) -> Waste2EnergyGymEnv:
    return Waste2EnergyGymEnv(spec, horizon_steps=horizon_steps)


def _to_discrete_signal(value: float) -> int:
    if value > 0.33:
        return 1
    if value < -0.33:
        return -1
    return 0


def _require_gymnasium() -> None:
    if gym is None or spaces is None:
        raise ImportError(
            "gymnasium is required for the RL appendix environment. "
            "Install the project dependencies to enable this layer."
        ) from _GYM_IMPORT_ERROR
