# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
MODEL_READY_DIR = ROOT / "data" / "processed" / "model_ready"
FIGURES_TABLES_DIR = ROOT / "data" / "processed" / "figures_tables"
OUTPUTS_ROOT = ROOT / "outputs"
LEGACY_SURROGATE_OUTPUTS_DIR = OUTPUTS_ROOT / "xgboost"
SURROGATE_OUTPUTS_DIR = OUTPUTS_ROOT / "surrogates"
PLANNING_OUTPUTS_DIR = OUTPUTS_ROOT / "planning"
SCENARIO_OUTPUTS_DIR = OUTPUTS_ROOT / "scenarios"
OPERATION_OUTPUTS_DIR = OUTPUTS_ROOT / "operation"
OUTPUTS_DIR = SURROGATE_OUTPUTS_DIR
RANDOM_STATE = 42

DEFAULT_OBJECTIVE_WEIGHT_PRESET = "balanced_cleaner_production"


@dataclass(frozen=True)
class ObjectiveWeights:
    energy: float
    environment: float
    cost: float

    def normalized(self) -> "ObjectiveWeights":
        total = self.energy + self.environment + self.cost
        if total <= 0.0:
            raise ValueError("Objective weights must sum to a positive value.")
        return ObjectiveWeights(
            energy=self.energy / total,
            environment=self.environment / total,
            cost=self.cost / total,
        )

    def as_dict(self) -> dict[str, float]:
        normalized = self.normalized()
        return {
            "energy": normalized.energy,
            "environment": normalized.environment,
            "cost": normalized.cost,
        }


OBJECTIVE_WEIGHT_PRESETS: dict[str, ObjectiveWeights] = {
    "balanced_cleaner_production": ObjectiveWeights(0.40, 0.35, 0.25),
    "energy_priority": ObjectiveWeights(0.55, 0.25, 0.20),
    "environment_priority": ObjectiveWeights(0.30, 0.50, 0.20),
    "cost_guardrail": ObjectiveWeights(0.30, 0.25, 0.45),
}


@dataclass(frozen=True)
class ObjectiveWeightSystem:
    preset_name: str = DEFAULT_OBJECTIVE_WEIGHT_PRESET
    weights: ObjectiveWeights = OBJECTIVE_WEIGHT_PRESETS[DEFAULT_OBJECTIVE_WEIGHT_PRESET]

    def normalized(self) -> ObjectiveWeights:
        return self.weights.normalized()

    @property
    def energy(self) -> float:
        return self.normalized().energy

    @property
    def environment(self) -> float:
        return self.normalized().environment

    @property
    def cost(self) -> float:
        return self.normalized().cost

    def as_dict(self) -> dict[str, object]:
        return {
            "preset_name": self.preset_name,
            "weights": self.normalized().as_dict(),
        }


def resolve_surrogate_outputs_dir() -> Path:
    if SURROGATE_OUTPUTS_DIR.exists():
        return SURROGATE_OUTPUTS_DIR
    if LEGACY_SURROGATE_OUTPUTS_DIR.exists():
        return LEGACY_SURROGATE_OUTPUTS_DIR
    return SURROGATE_OUTPUTS_DIR


def get_objective_weight_system(
    *,
    preset_name: str = DEFAULT_OBJECTIVE_WEIGHT_PRESET,
    energy: float | None = None,
    environment: float | None = None,
    cost: float | None = None,
) -> ObjectiveWeightSystem:
    if preset_name not in OBJECTIVE_WEIGHT_PRESETS:
        allowed = ", ".join(sorted(OBJECTIVE_WEIGHT_PRESETS))
        raise ValueError(f"Unsupported objective-weight preset '{preset_name}'. Choose from: {allowed}")

    base = OBJECTIVE_WEIGHT_PRESETS[preset_name]
    weights = ObjectiveWeights(
        energy=base.energy if energy is None else energy,
        environment=base.environment if environment is None else environment,
        cost=base.cost if cost is None else cost,
    ).normalized()
    return ObjectiveWeightSystem(preset_name=preset_name, weights=weights)


def perturb_objective_weights(
    base_system: ObjectiveWeightSystem,
    *,
    delta: float = 0.05,
) -> list[ObjectiveWeightSystem]:
    if delta <= 0.0:
        raise ValueError("delta must be positive for weight perturbation.")

    base = base_system.normalized()
    variants: list[ObjectiveWeightSystem] = [base_system]
    for focus in ("energy", "environment", "cost"):
        for direction in (-1.0, 1.0):
            changed = {
                "energy": base.energy,
                "environment": base.environment,
                "cost": base.cost,
            }
            changed[focus] = max(0.01, changed[focus] + direction * delta)
            variants.append(
                ObjectiveWeightSystem(
                    preset_name=f"{base_system.preset_name}:{focus}:{'down' if direction < 0 else 'up'}",
                    weights=ObjectiveWeights(**changed).normalized(),
                )
            )
    return _deduplicate_weight_systems(variants)


def _deduplicate_weight_systems(
    variants: Iterable[ObjectiveWeightSystem],
) -> list[ObjectiveWeightSystem]:
    deduplicated: list[ObjectiveWeightSystem] = []
    seen: set[tuple[float, float, float]] = set()
    for variant in variants:
        weights = variant.normalized()
        key = (round(weights.energy, 8), round(weights.environment, 8), round(weights.cost, 8))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(
            ObjectiveWeightSystem(
                preset_name=variant.preset_name,
                weights=weights,
            )
        )
    return deduplicated

