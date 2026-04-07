# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import pandas as pd
import pytest

from waste2energy.common import METRIC_TON_TO_SHORT_TON
from waste2energy.planning.constraints import build_scenario_constraints
from waste2energy.planning.inputs import load_planning_input_bundle
from waste2energy.planning.optimization import _carbon_budget
from waste2energy.planning.solve import PlanningConfig


def test_planning_input_bundle_normalizes_baseline_emission_factor_units():
    bundle = load_planning_input_bundle()
    frame = bundle.frame

    assert "scenario_baseline_waste_treatment_emission_factor_kgco2e_per_metric_ton" in frame.columns
    assert "baseline_emission_factor_source_unit" in frame.columns
    baseline_rows = frame[frame["baseline_emission_factor_source_unit"] == "kgco2e_per_short_ton"].head(5)
    assert not baseline_rows.empty

    expected = (
        baseline_rows["scenario_baseline_waste_treatment_emission_factor_kgco2e_per_short_ton"]
        * METRIC_TON_TO_SHORT_TON
    )
    pd.testing.assert_series_equal(
        baseline_rows["scenario_baseline_waste_treatment_emission_factor_kgco2e_per_metric_ton"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_scenario_constraints_export_metric_ton_carbon_budget_basis():
    frame = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "scenario_wet_waste_feed_allocation_ton_per_year_proxy": 100.0,
                "facility_total_available_capacity_ton_per_year_reference": 100.0,
                "facility_total_permitted_capacity_ton_per_year_reference": 120.0,
                "organic_waste_recycling_capacity_needed_ton_per_year_reference": 100.0,
                "scenario_baseline_waste_treatment_emission_factor_kgco2e_per_metric_ton": 110.23113109243878,
                "scenario_baseline_waste_treatment_emission_factor_kgco2e_per_short_ton": 100.0,
                "baseline_emission_factor_source_unit": "kgco2e_per_short_ton",
                "baseline_emission_factor_internal_unit": "kgco2e_per_metric_ton",
                "planning_mass_unit_basis": "metric_ton",
                "short_ton_to_metric_ton_factor": 0.90718474,
            }
        ]
    )

    constraints = build_scenario_constraints(frame, PlanningConfig())
    row = constraints.iloc[0]

    assert row["planning_mass_unit_basis"] == "metric_ton"
    assert row["baseline_emission_factor_source_unit"] == "kgco2e_per_short_ton"
    assert row["baseline_emission_factor_internal_unit"] == "kgco2e_per_metric_ton"
    assert row["baseline_emission_factor_kgco2e_per_metric_ton"] == pytest.approx(110.23113109243878)
    assert row["carbon_budget_kgco2e"] == pytest.approx(9369.646142857296)


def test_carbon_budget_uses_metric_ton_baseline_emission_factor():
    scored = pd.DataFrame(
        [
            {
                "scenario_baseline_waste_treatment_emission_factor_kgco2e_per_metric_ton": 110.23113109243878,
            }
        ]
    )
    scenario_constraint = {
        "effective_processing_budget_ton_per_year": 100.0,
        "baseline_emission_factor_kgco2e_per_metric_ton": 110.23113109243878,
    }

    budget = _carbon_budget(scored, scenario_constraint, PlanningConfig())
    assert budget == pytest.approx(11023.113109243878)
