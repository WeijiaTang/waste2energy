# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

SHORT_TON_TO_METRIC_TON = 0.90718474
METRIC_TON_TO_SHORT_TON = 1.0 / SHORT_TON_TO_METRIC_TON

SUPPORTED_EMISSION_FACTOR_UNITS = frozenset(
    {
        "kgco2e_per_metric_ton",
        "kgco2e_per_short_ton",
    }
)


def emission_factor_to_metric_ton(value: float, source_unit: str) -> float:
    normalized_unit = normalize_emission_factor_unit(source_unit)
    if normalized_unit == "kgco2e_per_metric_ton":
        return float(value)
    if normalized_unit == "kgco2e_per_short_ton":
        return float(value) * METRIC_TON_TO_SHORT_TON
    raise ValueError(f"Unsupported emission-factor unit '{source_unit}'.")


def normalize_emission_factor_unit(raw_unit: str | None) -> str:
    value = str(raw_unit or "").strip().lower()
    if value in SUPPORTED_EMISSION_FACTOR_UNITS:
        return value
    raise ValueError(
        "Unsupported emission-factor unit "
        f"'{raw_unit}'. Supported units: {', '.join(sorted(SUPPORTED_EMISSION_FACTOR_UNITS))}."
    )
