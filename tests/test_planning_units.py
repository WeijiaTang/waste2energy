# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import pandas as pd
import pytest

from waste2energy.common import METRIC_TON_TO_SHORT_TON
from waste2energy.planning.constraints import build_scenario_constraints
from waste2energy.planning.inputs import (
    DEFAULT_SCENARIO_EXTERNAL_EVIDENCE_TABLE,
    DEFAULT_SCENARIO_METRIC_ADJUSTMENT_TABLE,
    REQUIRED_PLANNING_COLUMNS,
    load_planning_input_bundle,
    load_scenario_external_evidence_table,
    load_scenario_metric_adjustment_table,
    normalize_planning_units,
    validate_planning_frame,
)
from waste2energy.planning.objectives import _build_carbon_load, assemble_objective_frame
from waste2energy.planning.optimization import _carbon_budget, build_candidate_score_frame, solve_scenario_optimization
from waste2energy.planning.solve import PlanningConfig, build_pathway_summary


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


def test_scenario_constraints_fail_fast_on_missing_required_numeric_input():
    frame = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "scenario_wet_waste_feed_allocation_ton_per_year_proxy": pd.NA,
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

    with pytest.raises(ValueError, match="scenario_wet_waste_feed_allocation_ton_per_year_proxy"):
        build_scenario_constraints(frame, PlanningConfig())


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


def test_candidate_score_frame_fails_fast_on_missing_solver_input():
    frame = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "optimization_case_id": "case-1",
                "planning_energy_intensity_mj_per_ton": 100.0,
                "planning_environment_intensity_kgco2e_per_ton": 50.0,
                "planning_cost_intensity_proxy_or_real_per_ton": pd.NA,
                "planning_carbon_load_kgco2e_per_ton": 20.0,
                "combined_uncertainty_ratio": 0.1,
            }
        ]
    )

    with pytest.raises(ValueError, match="planning_cost_intensity_proxy_or_real_per_ton"):
        build_candidate_score_frame(frame, PlanningConfig())


def test_missing_pathway_emission_factor_preserves_nan_for_fallback_logic():
    frame = pd.DataFrame(
        [
            {
                "baseline_waste_treatment_factor_unit_reference": "kgco2e_per_short_ton",
                "baseline_waste_treatment_emission_factor_kgco2e_per_short_ton_reference": 100.0,
                "scenario_baseline_waste_treatment_emission_factor_kgco2e_per_short_ton": 100.0,
                "pathway_emission_factor_kgco2e_per_short_ton_scenario_proxy": pd.NA,
            }
        ]
    )

    normalized = normalize_planning_units(frame)

    assert pd.isna(normalized.loc[0, "pathway_emission_factor_kgco2e_per_metric_ton_scenario_proxy"])


def test_carbon_load_falls_back_when_pathway_emission_factor_is_missing():
    frame = pd.DataFrame(
        {
            "pathway_emission_factor_kgco2e_per_metric_ton_scenario_proxy": [pd.NA],
            "planning_environment_intensity_kgco2e_per_ton": [150.0],
        }
    )
    baseline_emission = pd.Series([200.0], dtype=float)

    carbon_load = _build_carbon_load(frame, baseline_emission)

    assert carbon_load.iloc[0] == pytest.approx(50.0)


def test_carbon_load_clips_negative_explicit_emission_factor_to_zero():
    frame = pd.DataFrame(
        {
            "pathway_emission_factor_kgco2e_per_metric_ton_scenario_proxy": [-44.092452],
            "planning_environment_intensity_kgco2e_per_ton": [340.0],
        }
    )
    baseline_emission = pd.Series([363.742566], dtype=float)

    carbon_load = _build_carbon_load(frame, baseline_emission)

    assert carbon_load.iloc[0] == pytest.approx(0.0)


def test_planning_input_validation_rejects_missing_required_baseline_emission_factor():
    row = {column: 1.0 for column in REQUIRED_PLANNING_COLUMNS}
    row.update(
        {
            "optimization_case_id": "case-1",
            "sample_id": "sample-1",
            "scenario_name": "baseline_region_case",
            "pathway": "pyrolysis",
            "baseline_waste_treatment_factor_unit_reference": "kgco2e_per_short_ton",
            "cost_model_basis": "reference_cost_model",
            "cost_model_source_trace": "traceable_source",
            "baseline_waste_treatment_emission_factor_kgco2e_per_short_ton_reference": pd.NA,
        }
    )
    frame = pd.DataFrame([row])

    with pytest.raises(ValueError, match="baseline_waste_treatment_emission_factor_kgco2e_per_short_ton_reference"):
        validate_planning_frame(frame, load_planning_input_bundle().dataset_path)


def test_scenario_external_evidence_table_is_traceable_and_complete():
    frame, path = load_scenario_external_evidence_table()

    assert path == DEFAULT_SCENARIO_EXTERNAL_EVIDENCE_TABLE
    assert path == DEFAULT_SCENARIO_METRIC_ADJUSTMENT_TABLE
    assert len(frame) == 3
    assert frame[["scenario_name"]].drop_duplicates().shape[0] == len(frame)
    assert frame["feedstock_scale_factor"].gt(0.0).all()
    assert frame["feedstock_cost_elasticity"].ge(0.0).all()
    assert frame["carbon_tax_usd_per_ton_co2e"].ge(0.0).all()
    assert frame["evidence_source"].astype(str).str.len().gt(0).all()
    assert frame["evidence_reference"].astype(str).str.len().gt(0).all()
    assert frame["evidence_rationale"].astype(str).str.len().gt(0).all()


def test_assemble_objective_frame_excludes_candidate_with_missing_critical_cost_term():
    base_frame = pd.DataFrame(
        [
            {
                "optimization_case_id": "case-valid",
                "pathway": "pyrolysis",
                "scenario_name": "baseline_region_case",
                "scenario_wet_waste_feed_allocation_ton_per_year_proxy": 100.0,
                "scenario_total_mixed_feed_ton_per_year_proxy": 120.0,
                "scenario_baseline_waste_treatment_emission_factor_kgco2e_per_metric_ton": 200.0,
                "feedstock_carbon_pct": 45.0,
                "pathway_energy_intensity_mj_per_ton": 1000.0,
                "pathway_environment_benefit_kgco2e_per_ton": 80.0,
                "unit_net_system_cost_usd_per_ton": 55.0,
                "policy_multiplier": 1.0,
            },
            {
                "optimization_case_id": "case-missing-cost",
                "pathway": "pyrolysis",
                "scenario_name": "baseline_region_case",
                "scenario_wet_waste_feed_allocation_ton_per_year_proxy": 100.0,
                "scenario_total_mixed_feed_ton_per_year_proxy": 120.0,
                "scenario_baseline_waste_treatment_emission_factor_kgco2e_per_metric_ton": 200.0,
                "feedstock_carbon_pct": 45.0,
                "pathway_energy_intensity_mj_per_ton": 1000.0,
                "pathway_environment_benefit_kgco2e_per_ton": 80.0,
                "unit_net_system_cost_usd_per_ton": pd.NA,
                "policy_multiplier": 1.0,
            },
        ]
    )
    surrogate_predictions = base_frame[["optimization_case_id", "pathway"]].copy()

    retained, readiness, summary, exclusions = assemble_objective_frame(
        base_frame=base_frame,
        surrogate_predictions=surrogate_predictions,
        robustness_factor=0.35,
        real_cost_columns=("unit_net_system_cost_usd_per_ton",),
    )

    assert readiness["data_quality_status"].startswith("critical-missing candidates excluded")
    assert retained["optimization_case_id"].tolist() == ["case-valid"]
    assert exclusions["optimization_case_id"].tolist() == ["case-missing-cost"]
    assert "critical_missing:planning_cost_intensity_proxy_or_real_per_ton" in exclusions.iloc[0][
        "planning_exclusion_reason"
    ]
    assert summary.loc[0, "excluded_candidate_count"] == 1


def test_assemble_objective_frame_flags_noncritical_imputations():
    base_frame = pd.DataFrame(
        [
            {
                "optimization_case_id": "case-imputed",
                "pathway": "pyrolysis",
                "scenario_name": "baseline_region_case",
                "scenario_wet_waste_feed_allocation_ton_per_year_proxy": 100.0,
                "scenario_total_mixed_feed_ton_per_year_proxy": pd.NA,
                "scenario_baseline_waste_treatment_emission_factor_kgco2e_per_metric_ton": 220.0,
                "feedstock_carbon_pct": 45.0,
                "pathway_energy_intensity_mj_per_ton": 1100.0,
                "pathway_environment_benefit_kgco2e_per_ton": 95.0,
                "unit_net_system_cost_usd_per_ton": 60.0,
                "policy_multiplier": pd.NA,
            }
        ]
    )
    surrogate_predictions = base_frame[["optimization_case_id", "pathway"]].copy()

    retained, _, _, exclusions = assemble_objective_frame(
        base_frame=base_frame,
        surrogate_predictions=surrogate_predictions,
        robustness_factor=0.35,
        real_cost_columns=("unit_net_system_cost_usd_per_ton",),
    )

    assert exclusions.empty
    assert retained.loc[0, "planning_imputation_flag"]
    assert "scenario_total_mixed_feed_ton_per_year_proxy:fallback_to_scenario_wet_waste_feed_allocation" in retained.loc[
        0, "planning_imputation_notes"
    ]
    assert "policy_multiplier:default_policy_multiplier_one" in retained.loc[0, "planning_imputation_notes"]
    assert "combined_uncertainty_ratio:default_uncertainty_zero" in retained.loc[0, "planning_imputation_notes"]
    assert retained.loc[0, "scenario_total_mixed_feed_source"] == "fallback_to_wet_waste_feed_allocation"
    assert retained.loc[0, "policy_multiplier_source"] == "default_policy_multiplier_one"
    assert retained.loc[0, "combined_uncertainty_ratio_source"] == "default_uncertainty_zero"


def test_assemble_objective_frame_prefers_interval_mean_for_uncertainty_penalty():
    base_frame = pd.DataFrame(
        [
            {
                "optimization_case_id": "case-interval",
                "pathway": "pyrolysis",
                "scenario_name": "baseline_region_case",
                "scenario_wet_waste_feed_allocation_ton_per_year_proxy": 100.0,
                "scenario_total_mixed_feed_ton_per_year_proxy": 120.0,
                "scenario_baseline_waste_treatment_emission_factor_kgco2e_per_metric_ton": 220.0,
                "feedstock_carbon_pct": 45.0,
                "pathway_energy_intensity_mj_per_ton": 1100.0,
                "pathway_environment_benefit_kgco2e_per_ton": 95.0,
                "unit_net_system_cost_usd_per_ton": 60.0,
                "policy_multiplier": 1.0,
            }
        ]
    )
    surrogate_predictions = pd.DataFrame(
        [
            {
                "optimization_case_id": "case-interval",
                "pathway": "pyrolysis",
                "combined_uncertainty_ratio": 0.90,
                "product_char_yield_pct_uncertainty_ratio": 0.20,
                "product_char_hhv_mj_per_kg_uncertainty_ratio": 0.40,
                "energy_recovery_pct_uncertainty_ratio": 0.60,
            }
        ]
    )

    retained, _, _, _ = assemble_objective_frame(
        base_frame=base_frame,
        surrogate_predictions=surrogate_predictions,
        robustness_factor=0.35,
        real_cost_columns=("unit_net_system_cost_usd_per_ton",),
        config=PlanningConfig(uncertainty_penalty_mode="prefer_interval_mean"),
    )

    assert retained.loc[0, "combined_uncertainty_ratio"] == pytest.approx((0.20 + 0.40 + 0.60) / 3.0)
    assert retained.loc[0, "combined_uncertainty_ratio_source"] == "interval_mean_ratio"
    assert retained.loc[0, "uncertainty_penalty_mode"] == "prefer_interval_mean"
    assert retained.loc[0, "interval_mean_uncertainty_ratio"] == pytest.approx((0.20 + 0.40 + 0.60) / 3.0)
    assert retained.loc[0, "interval_max_uncertainty_ratio"] == pytest.approx(0.60)
    assert retained.loc[0, "uncertainty_ratio_combined_only"] == pytest.approx(0.90)


def test_assemble_objective_frame_supports_max_interval_ratio_mode():
    base_frame = pd.DataFrame(
        [
            {
                "optimization_case_id": "case-interval",
                "pathway": "pyrolysis",
                "scenario_name": "baseline_region_case",
                "scenario_wet_waste_feed_allocation_ton_per_year_proxy": 100.0,
                "scenario_total_mixed_feed_ton_per_year_proxy": 120.0,
                "scenario_baseline_waste_treatment_emission_factor_kgco2e_per_metric_ton": 220.0,
                "feedstock_carbon_pct": 45.0,
                "pathway_energy_intensity_mj_per_ton": 1100.0,
                "pathway_environment_benefit_kgco2e_per_ton": 95.0,
                "unit_net_system_cost_usd_per_ton": 60.0,
                "policy_multiplier": 1.0,
            }
        ]
    )
    surrogate_predictions = pd.DataFrame(
        [
            {
                "optimization_case_id": "case-interval",
                "pathway": "pyrolysis",
                "combined_uncertainty_ratio": 0.30,
                "product_char_yield_pct_uncertainty_ratio": 0.20,
                "product_char_hhv_mj_per_kg_uncertainty_ratio": 0.55,
                "energy_recovery_pct_uncertainty_ratio": 0.40,
            }
        ]
    )

    retained, _, _, _ = assemble_objective_frame(
        base_frame=base_frame,
        surrogate_predictions=surrogate_predictions,
        robustness_factor=0.35,
        real_cost_columns=("unit_net_system_cost_usd_per_ton",),
        config=PlanningConfig(uncertainty_penalty_mode="max_interval_ratio"),
    )

    assert retained.loc[0, "combined_uncertainty_ratio"] == pytest.approx(0.55)
    assert retained.loc[0, "combined_uncertainty_ratio_source"] == "interval_max_ratio"
    assert retained.loc[0, "uncertainty_penalty_mode"] == "max_interval_ratio"
    assert retained.loc[0, "uncertainty_ratio_interval_mean"] == pytest.approx((0.20 + 0.55 + 0.40) / 3.0)
    assert retained.loc[0, "uncertainty_ratio_max_interval"] == pytest.approx(0.55)


def test_assemble_objective_frame_can_force_combined_only_uncertainty_mode():
    base_frame = pd.DataFrame(
        [
            {
                "optimization_case_id": "case-combined",
                "pathway": "pyrolysis",
                "scenario_name": "baseline_region_case",
                "scenario_wet_waste_feed_allocation_ton_per_year_proxy": 100.0,
                "scenario_total_mixed_feed_ton_per_year_proxy": 120.0,
                "scenario_baseline_waste_treatment_emission_factor_kgco2e_per_metric_ton": 220.0,
                "feedstock_carbon_pct": 45.0,
                "pathway_energy_intensity_mj_per_ton": 1100.0,
                "pathway_environment_benefit_kgco2e_per_ton": 95.0,
                "unit_net_system_cost_usd_per_ton": 60.0,
                "policy_multiplier": 1.0,
            }
        ]
    )
    surrogate_predictions = pd.DataFrame(
        [
            {
                "optimization_case_id": "case-combined",
                "pathway": "pyrolysis",
                "combined_uncertainty_ratio": 0.33,
                "product_char_yield_pct_uncertainty_ratio": 0.20,
                "product_char_hhv_mj_per_kg_uncertainty_ratio": 0.55,
            }
        ]
    )

    retained, _, _, _ = assemble_objective_frame(
        base_frame=base_frame,
        surrogate_predictions=surrogate_predictions,
        robustness_factor=0.35,
        real_cost_columns=("unit_net_system_cost_usd_per_ton",),
        config=PlanningConfig(uncertainty_penalty_mode="combined_only"),
    )

    assert retained.loc[0, "combined_uncertainty_ratio"] == pytest.approx(0.33)
    assert retained.loc[0, "combined_uncertainty_ratio_source"] == "explicit_combined_uncertainty_ratio"
    assert retained.loc[0, "uncertainty_penalty_mode"] == "combined_only"
    assert retained.loc[0, "uncertainty_ratio_interval_mean"] == pytest.approx((0.20 + 0.55) / 2.0)
    assert retained.loc[0, "uncertainty_ratio_max_interval"] == pytest.approx(0.55)


def test_build_candidate_score_frame_exposes_uncertainty_mode_rank_swaps():
    scenario_frame = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "optimization_case_id": "case-a",
                "pathway": "pyrolysis",
                "planning_energy_intensity_mj_per_ton": 1000.0,
                "planning_environment_intensity_kgco2e_per_ton": 100.0,
                "planning_cost_intensity_proxy_or_real_per_ton": 50.0,
                "planning_carbon_load_kgco2e_per_ton": 10.0,
                "combined_uncertainty_ratio": 0.20,
                "effective_uncertainty_ratio": 0.20,
                "evidence_based_weight": 1.0,
                "interval_mean_uncertainty_ratio": 0.20,
                "interval_max_uncertainty_ratio": 0.80,
                "uncertainty_ratio_interval_mean": 0.20,
                "uncertainty_ratio_max_interval": 0.80,
                "uncertainty_ratio_combined_only": 0.20,
                "effective_uncertainty_ratio_interval_mean": 0.20,
                "effective_uncertainty_ratio_max_interval": 0.80,
                "effective_uncertainty_ratio_combined_only": 0.20,
            },
            {
                "scenario_name": "baseline_region_case",
                "optimization_case_id": "case-b",
                "pathway": "pyrolysis",
                "planning_energy_intensity_mj_per_ton": 1000.0,
                "planning_environment_intensity_kgco2e_per_ton": 100.0,
                "planning_cost_intensity_proxy_or_real_per_ton": 50.0,
                "planning_carbon_load_kgco2e_per_ton": 10.0,
                "combined_uncertainty_ratio": 0.40,
                "effective_uncertainty_ratio": 0.40,
                "evidence_based_weight": 1.0,
                "interval_mean_uncertainty_ratio": 0.40,
                "interval_max_uncertainty_ratio": 0.40,
                "uncertainty_ratio_interval_mean": 0.40,
                "uncertainty_ratio_max_interval": 0.40,
                "uncertainty_ratio_combined_only": 0.40,
                "effective_uncertainty_ratio_interval_mean": 0.40,
                "effective_uncertainty_ratio_max_interval": 0.40,
                "effective_uncertainty_ratio_combined_only": 0.40,
            },
        ]
    )

    scored = build_candidate_score_frame(
        scenario_frame,
        PlanningConfig(uncertainty_penalty_mode="prefer_interval_mean"),
    )
    by_id = scored.set_index("optimization_case_id")

    assert by_id.loc["case-a", "planning_rank_interval_mean"] == pytest.approx(1.0)
    assert by_id.loc["case-b", "planning_rank_interval_mean"] == pytest.approx(2.0)
    assert by_id.loc["case-a", "planning_rank_max_interval"] == pytest.approx(2.0)
    assert by_id.loc["case-b", "planning_rank_max_interval"] == pytest.approx(1.0)
    assert by_id.loc["case-a", "uncertainty_rank_span"] == pytest.approx(1.0)
    assert by_id.loc["case-a", "uncertainty_best_mode"] == "interval_mean"
    assert by_id.loc["case-a", "uncertainty_worst_mode"] == "max_interval"


def test_build_candidate_score_frame_uses_configurable_evidence_utility_factor():
    scenario_frame = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "optimization_case_id": "case-low",
                "pathway": "htc",
                "planning_energy_intensity_mj_per_ton": 1000.0,
                "planning_environment_intensity_kgco2e_per_ton": 100.0,
                "planning_cost_intensity_proxy_or_real_per_ton": 50.0,
                "planning_carbon_load_kgco2e_per_ton": 10.0,
                "combined_uncertainty_ratio": 0.20,
                "effective_uncertainty_ratio": 0.20,
                "evidence_based_weight": 0.35,
            },
            {
                "scenario_name": "baseline_region_case",
                "optimization_case_id": "case-high",
                "pathway": "pyrolysis",
                "planning_energy_intensity_mj_per_ton": 1000.0,
                "planning_environment_intensity_kgco2e_per_ton": 100.0,
                "planning_cost_intensity_proxy_or_real_per_ton": 50.0,
                "planning_carbon_load_kgco2e_per_ton": 10.0,
                "combined_uncertainty_ratio": 0.20,
                "effective_uncertainty_ratio": 0.20,
                "evidence_based_weight": 1.00,
            },
        ]
    )

    scored_zero = build_candidate_score_frame(
        scenario_frame,
        PlanningConfig(evidence_utility_factor=0.0),
    ).set_index("optimization_case_id")
    scored_high = build_candidate_score_frame(
        scenario_frame,
        PlanningConfig(evidence_utility_factor=1.0),
    ).set_index("optimization_case_id")

    assert scored_zero.loc["case-low", "weighted_score_per_ton"] == pytest.approx(
        scored_zero.loc["case-high", "weighted_score_per_ton"]
    )
    assert scored_high.loc["case-high", "weighted_score_per_ton"] > scored_high.loc["case-low", "weighted_score_per_ton"]


def test_pathway_summary_preserves_missing_best_case_uncertainty_ratio():
    scored = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "optimization_case_id": "case-1",
                "planning_score": 1.0,
                "planning_energy_objective": 100.0,
                "planning_environment_objective": 50.0,
                "planning_cost_objective": 10.0,
                "combined_uncertainty_ratio": pd.NA,
                "manure_subtype": "beef",
                "blend_manure_ratio": 0.3,
                "blend_wet_waste_ratio": 0.7,
            }
        ]
    )

    summary = build_pathway_summary(scored=scored, portfolio_allocations=pd.DataFrame())

    assert pd.isna(summary.loc[0, "best_case_uncertainty_ratio"])


def test_pathway_summary_fails_fast_on_missing_allocated_feed_value():
    scored = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "optimization_case_id": "case-1",
                "planning_score": 1.0,
                "planning_energy_objective": 100.0,
                "planning_environment_objective": 50.0,
                "planning_cost_objective": 10.0,
                "combined_uncertainty_ratio": 0.1,
                "manure_subtype": "beef",
                "blend_manure_ratio": 0.3,
                "blend_wet_waste_ratio": 0.7,
            }
        ]
    )
    allocations = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "optimization_case_id": "case-1",
                "allocated_feed_ton_per_year": pd.NA,
                "allocated_feed_share": 0.5,
                "allocated_energy_objective": 50.0,
                "allocated_environment_objective": 25.0,
                "allocated_cost_objective": 5.0,
                "allocated_carbon_load_kgco2e": 3.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="allocated_feed_ton_per_year"):
        build_pathway_summary(scored=scored, portfolio_allocations=allocations)


def test_scenario_optimization_requires_positive_throughput_for_selected_subtypes():
    scenario_frame = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_region_case",
                "pathway": "pyrolysis",
                "optimization_case_id": "case-a",
                "manure_subtype": "beef",
                "planning_energy_intensity_mj_per_ton": 120.0,
                "planning_environment_intensity_kgco2e_per_ton": 90.0,
                "planning_cost_intensity_proxy_or_real_per_ton": 10.0,
                "planning_carbon_load_kgco2e_per_ton": 1.0,
                "combined_uncertainty_ratio": 0.05,
                "effective_uncertainty_ratio": 0.05,
                "evidence_based_weight": 1.0,
            },
            {
                "scenario_name": "baseline_region_case",
                "pathway": "htc",
                "optimization_case_id": "case-b",
                "manure_subtype": "swine",
                "planning_energy_intensity_mj_per_ton": 5.0,
                "planning_environment_intensity_kgco2e_per_ton": 5.0,
                "planning_cost_intensity_proxy_or_real_per_ton": 200.0,
                "planning_carbon_load_kgco2e_per_ton": 1.0,
                "combined_uncertainty_ratio": 1.0,
                "effective_uncertainty_ratio": 1.0,
                "evidence_based_weight": 1.0,
            },
        ]
    )
    scenario_constraint = {
        "scenario_name": "baseline_region_case",
        "effective_processing_budget_ton_per_year": 100.0,
        "candidate_share_cap_ton_per_year": 100.0,
        "subtype_share_cap_ton_per_year": 100.0,
        "baseline_emission_factor_kgco2e_per_metric_ton": 1000.0,
        "scenario_feed_budget_ton_per_year": 100.0,
        "min_distinct_subtypes": 2,
    }
    config = PlanningConfig(
        optimization_method="scipy_milp",
        enforce_candidate_cap=True,
        enforce_subtype_cap=True,
        enforce_max_selected=True,
        enforce_min_distinct_subtypes=True,
        max_portfolio_candidates=2,
        min_distinct_subtypes=2,
    )

    result = solve_scenario_optimization(scenario_frame, scenario_constraint, config)

    assert set(result.allocations["manure_subtype"]) == {"beef", "swine"}
    assert result.allocations["allocated_feed_ton_per_year"].gt(0.0).all()
