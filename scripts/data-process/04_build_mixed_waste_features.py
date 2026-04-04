from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
UNIFIED_DIR = ROOT / "data" / "processed" / "unified_features"
SCENARIO_DIR = ROOT / "data" / "processed" / "scenario_inputs"
MODEL_READY_DIR = ROOT / "data" / "processed" / "model_ready"
CALIFORNIA_MODEL_INPUT = MODEL_READY_DIR / "california_food_waste_model_input.csv"
REGION_SCENARIO_INPUT = SCENARIO_DIR / "paper1_region_scenario_placeholder.csv"


BLEND_CASES = [
    ("manure_dominant", 0.7, 0.3),
    ("balanced_mixed_feed", 0.5, 0.5),
    ("wet_waste_enhanced", 0.3, 0.7),
]

FEEDSTOCK_COLUMNS = [
    "feedstock_carbon_pct",
    "feedstock_hydrogen_pct",
    "feedstock_nitrogen_pct",
    "feedstock_oxygen_pct",
    "feedstock_moisture_pct",
    "feedstock_volatile_matter_pct",
    "feedstock_fixed_carbon_pct",
    "feedstock_ash_pct",
    "feedstock_hhv_mj_per_kg",
]

TARGET_COLUMNS = [
    "product_char_yield_pct",
    "product_char_hhv_mj_per_kg",
    "energy_recovery_pct",
    "carbon_retention_pct",
]


def load_htc_reference_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.read_csv(UNIFIED_DIR / "wet_waste_biomass_opt_combined_standardized.csv")
    htc = combined[combined["pathway"] == "htc"].copy()
    htc = htc[htc["feedstock_group"].isin(["manure", "food_waste"])].copy()
    manure = htc[htc["feedstock_group"] == "manure"].copy()
    food = htc[htc["feedstock_group"] == "food_waste"].copy()
    return manure, food


def load_manure_subtype_reference() -> pd.DataFrame:
    energy_balance = pd.read_csv(UNIFIED_DIR / "manure_pyrolysis_energy_balance_long.csv")
    keys = [
        "Initial Moisture Content",
        "dry manure calorific value",
        "biochar yield",
        "biochar calorific value",
    ]
    subset = energy_balance[energy_balance["constant_name"].isin(keys)].copy()
    pivot = subset.pivot_table(
        index="livestock_type",
        columns="constant_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(
        columns={
            "Initial Moisture Content": "subtype_moisture_ratio",
            "dry manure calorific value": "subtype_feedstock_hhv_mj_per_kg",
            "biochar yield": "subtype_char_yield_ratio",
            "biochar calorific value": "subtype_char_hhv_mj_per_kg",
        }
    )
    return pivot


def load_california_food_waste_reference() -> dict[str, object]:
    frame = pd.read_csv(CALIFORNIA_MODEL_INPUT)
    if len(frame) != 1:
        raise RuntimeError(
            f"Expected exactly one California food-waste model-input row in {CALIFORNIA_MODEL_INPUT}"
        )
    row = frame.iloc[0].to_dict()
    if row["region_id"] != "us_ca":
        raise RuntimeError(
            f"Expected California model input region_id us_ca, found {row['region_id']}"
        )
    return row


def load_region_scenarios() -> pd.DataFrame:
    frame = pd.read_csv(REGION_SCENARIO_INPUT)
    california_frame = frame[frame["region_id"] == "us_ca"].copy()
    if california_frame.empty:
        raise RuntimeError(
            "No us_ca scenarios found in paper1_region_scenario_placeholder.csv. "
            "Run 05_build_region_placeholder_inputs.py first."
        )

    columns = [
        "region_id",
        "scenario_name",
        "manure_supply_multiplier",
        "wet_waste_supply_multiplier",
        "energy_price_multiplier",
        "emission_factor_multiplier",
        "policy_multiplier",
        "scenario_wet_waste_generation_ton_per_year",
        "scenario_wet_waste_collectable_ton_per_year_proxy",
        "electricity_price_usd_per_kwh_reference",
        "natural_gas_price_usd_per_mcf_reference",
        "scenario_electricity_price_usd_per_kwh",
        "scenario_natural_gas_price_usd_per_mcf",
        "energy_reference_file",
        "baseline_waste_treatment_pathway_reference",
        "baseline_waste_treatment_factor_unit_reference",
        "baseline_waste_treatment_emission_factor_kgco2e_per_short_ton_reference",
        "scenario_baseline_waste_treatment_emission_factor_kgco2e_per_short_ton",
        "baseline_waste_treatment_source_method_reference",
        "emission_reference_file",
        "policy_effective_year_reference",
        "policy_organic_waste_disposal_reduction_target_pct_reference",
        "policy_edible_food_recovery_target_pct_reference",
        "policy_procurement_target_ton_per_capita_reference",
        "policy_reference_file",
        "facility_capacity_data_status",
        "facility_capacity_reporting_cycle_reference",
        "facility_count_reference",
        "facility_total_permitted_capacity_ton_per_year_reference",
        "facility_total_available_capacity_ton_per_year_reference",
        "organic_waste_recycling_capacity_needed_ton_per_year_reference",
        "edible_food_recovery_capacity_available_ton_per_year_reference",
        "edible_food_recovery_capacity_needed_ton_per_year_reference",
        "facility_capacity_reference_file",
        "organics_facility_inventory_status",
        "organics_facility_count_reference",
        "composting_facility_count_reference",
        "anaerobic_digestion_facility_count_reference",
        "facility_inventory_reference_file",
        "scenario_grid_electricity_emission_factor_kgco2e_per_kwh",
        "collection_rate_basis",
        "notes",
    ]
    california_frame = california_frame[columns].rename(columns={"notes": "region_scenario_notes"})
    return california_frame.reset_index(drop=True)


def representative_conditions(food_df: pd.DataFrame, max_conditions: int = 4) -> pd.DataFrame:
    counts = (
        food_df.groupby(["process_temperature_c", "residence_time_min"], dropna=True)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "process_temperature_c", "residence_time_min"], ascending=[False, True, True])
    )
    selected = counts.head(max_conditions).copy()
    if selected.empty:
        selected = pd.DataFrame(
            [
                {"process_temperature_c": 210.0, "residence_time_min": 30.0, "count": 0},
                {"process_temperature_c": 240.0, "residence_time_min": 30.0, "count": 0},
                {"process_temperature_c": 270.0, "residence_time_min": 60.0, "count": 0},
            ]
        )
    return selected


def weighted_value(manure_value: float, wet_value: float, manure_ratio: float, wet_ratio: float) -> float:
    return manure_value * manure_ratio + wet_value * wet_ratio


def build_mixed_rows() -> pd.DataFrame:
    manure_df, food_df = load_htc_reference_frames()
    subtype_df = load_manure_subtype_reference()

    manure_medians = manure_df[FEEDSTOCK_COLUMNS + TARGET_COLUMNS].median(numeric_only=True)
    food_medians = food_df[FEEDSTOCK_COLUMNS + TARGET_COLUMNS].median(numeric_only=True)
    conditions = representative_conditions(food_df)

    rows: list[dict[str, object]] = []
    row_id = 1
    for _, subtype in subtype_df.iterrows():
        subtype_name = str(subtype["livestock_type"])
        subtype_feedstock = manure_medians.copy()
        subtype_targets = manure_medians.copy()

        subtype_feedstock["feedstock_moisture_pct"] = float(subtype["subtype_moisture_ratio"]) * 100.0
        subtype_feedstock["feedstock_hhv_mj_per_kg"] = float(subtype["subtype_feedstock_hhv_mj_per_kg"])
        subtype_targets["product_char_yield_pct"] = float(subtype["subtype_char_yield_ratio"]) * 100.0
        subtype_targets["product_char_hhv_mj_per_kg"] = float(subtype["subtype_char_hhv_mj_per_kg"])

        if pd.notna(subtype["subtype_feedstock_hhv_mj_per_kg"]) and pd.notna(
            subtype["subtype_char_hhv_mj_per_kg"]
        ):
            subtype_targets["energy_recovery_pct"] = (
                float(subtype["subtype_char_yield_ratio"])
                * float(subtype["subtype_char_hhv_mj_per_kg"])
                / float(subtype["subtype_feedstock_hhv_mj_per_kg"])
                * 100.0
            )

        for blend_name, manure_ratio, wet_ratio in BLEND_CASES:
            for _, condition in conditions.iterrows():
                row: dict[str, object] = {
                    "sample_id": f"Waste2Energy::mixed_htc::{row_id:04d}",
                    "source_repo": "Waste2EnergyDerived",
                    "source_file": "mixed_feature_generation",
                    "source_dataset_kind": "synthetic_blended_reference",
                    "feedstock_name": f"{subtype_name}_manure_plus_food_waste",
                    "blending_case": blend_name,
                    "reference_label": "weighted_blend_from_repository_references",
                    "pathway": "htc",
                    "feedstock_group": "mixed_manure_wet_waste",
                    "manure_subtype": subtype_name,
                    "wet_waste_reference_group": "food_waste",
                    "process_temperature_c": float(condition["process_temperature_c"]),
                    "residence_time_min": float(condition["residence_time_min"]),
                    "heating_rate_c_per_min": pd.NA,
                    "blend_manure_ratio": manure_ratio,
                    "blend_wet_waste_ratio": wet_ratio,
                    "mixture_generation_method": "weighted_median_blend_with_subtype_overrides",
                }

                for column in FEEDSTOCK_COLUMNS:
                    row[column] = weighted_value(
                        float(subtype_feedstock[column]),
                        float(food_medians[column]),
                        manure_ratio,
                        wet_ratio,
                    )

                for column in TARGET_COLUMNS:
                    row[column] = weighted_value(
                        float(subtype_targets[column]),
                        float(food_medians[column]),
                        manure_ratio,
                        wet_ratio,
                    )

                row_id += 1
                rows.append(row)

    return pd.DataFrame(rows)


def build_optimization_rows(
    mixed_df: pd.DataFrame,
    california_reference: dict[str, object],
    region_scenarios: pd.DataFrame,
) -> pd.DataFrame:
    reference_generation = float(california_reference["waste_generation_ton_per_year"])
    collection_rate = float(california_reference["collection_rate_pct_reference"])
    collectable_proxy = reference_generation * collection_rate / 100.0

    enriched = mixed_df.copy()
    enriched["region_id"] = california_reference["region_id"]
    enriched["region_name"] = california_reference["region_name"]
    enriched["country"] = california_reference["country"]
    enriched["analysis_reference_year"] = int(california_reference["analysis_reference_year"])
    enriched["wet_waste_reference_stream_type"] = california_reference["waste_stream_type"]
    enriched["wet_waste_generation_ton_per_year_reference"] = reference_generation
    enriched["wet_waste_collection_rate_pct_reference"] = collection_rate
    enriched["wet_waste_collectable_ton_per_year_proxy_reference"] = collectable_proxy
    enriched["wet_waste_source_bundle"] = california_reference["source_bundle"]
    enriched["wet_waste_pathway_relevance"] = california_reference["pathway_relevance"]
    enriched["wet_waste_reference_notes"] = california_reference["notes"]

    optimization_df = (
        enriched.assign(_merge_key=1)
        .merge(region_scenarios.assign(_merge_key=1), on=["_merge_key", "region_id"], how="inner")
        .drop(columns="_merge_key")
    )
    optimization_df["optimization_case_id"] = (
        optimization_df["sample_id"] + "::" + optimization_df["scenario_name"]
    )
    optimization_df["scenario_wet_waste_feed_allocation_ton_per_year_proxy"] = (
        optimization_df["scenario_wet_waste_collectable_ton_per_year_proxy"]
        * optimization_df["blend_wet_waste_ratio"]
    )
    optimization_df["optimization_region_source"] = "paper1_region_scenario_placeholder.csv"
    optimization_df["optimization_wet_waste_source"] = "california_food_waste_model_input.csv"
    optimization_df["optimization_energy_source"] = "paper1_region_scenario_placeholder.csv"
    optimization_df["optimization_emission_source"] = "paper1_region_scenario_placeholder.csv"
    return optimization_df


def build_assumptions_payload(
    mixed_df: pd.DataFrame,
    optimization_df: pd.DataFrame,
    california_reference: dict[str, object],
    region_scenarios: pd.DataFrame,
) -> dict[str, object]:
    return {
        "dataset_name": "optimization_input_dataset",
        "prototype_dataset_name": "paper1_mixed_waste_feature_prototypes",
        "generation_method": "weighted median blending using repository-derived reference samples",
        "base_pathway": "htc",
        "wet_waste_reference_group": "food_waste",
        "region_id": california_reference["region_id"],
        "region_name": california_reference["region_name"],
        "wet_waste_reference_source": "california_food_waste_model_input.csv",
        "region_scenario_source": "paper1_region_scenario_placeholder.csv",
        "manure_reference_source": "manure_pyrolysis_energy_balance_long.csv",
        "energy_reference_source": "paper1_region_scenario_placeholder.csv",
        "emission_reference_source": "paper1_region_scenario_placeholder.csv",
        "capacity_reference_source": "paper1_region_scenario_placeholder.csv",
        "facility_inventory_reference_source": "paper1_region_scenario_placeholder.csv",
        "policy_reference_source": "paper1_region_scenario_placeholder.csv",
        "assumptions": [
            "Mixed-feed rows are synthetic prototypes, not direct experimental observations.",
            "Feedstock chemistry is blended using weighted averages between repository-derived food-waste and manure references.",
            "Manure subtype moisture and heating-value fields are overridden using ManurePyrolysisIAM energy-balance references.",
            "Char yield and char HHV are partly adjusted using manure subtype biochar-yield and calorific-value references.",
            "Carbon-retention values fall back to manure and food-waste repository medians because subtype-specific carbon-retention measurements were not available in the copied workbook.",
            "Optimization rows are created by crossing synthetic mixed-feed prototypes with California us_ca regional scenarios.",
            "Scenario wet-waste availability fields are proxies derived from California food-waste generation and the current collection-rate reference.",
            "Explicit California electricity and natural-gas price values are carried into optimization rows from the regional scenario placeholder dataset.",
            "EPA WARM food-waste baseline waste-treatment emission factors are carried into optimization rows as regional environmental parameters.",
            "California SWIS facility-inventory metadata is carried into optimization rows as regional infrastructure context.",
            "SB 1383 policy context is carried into optimization rows as regional policy-reference metadata rather than as a direct process-model coefficient.",
        ],
        "feature_prototype_row_count": int(len(mixed_df)),
        "optimization_row_count": int(len(optimization_df)),
        "blend_cases": [case[0] for case in BLEND_CASES],
        "scenario_names": region_scenarios["scenario_name"].dropna().tolist(),
        "manure_subtypes": sorted(mixed_df["manure_subtype"].dropna().unique().tolist()),
        "baseline_energy_prices": {
            "electricity_price_usd_per_kwh_reference": float(
                region_scenarios["electricity_price_usd_per_kwh_reference"].iloc[0]
            ),
            "natural_gas_price_usd_per_mcf_reference": float(
                region_scenarios["natural_gas_price_usd_per_mcf_reference"].iloc[0]
            ),
        },
        "baseline_waste_treatment_emission_factor": {
            "management_pathway": str(
                region_scenarios["baseline_waste_treatment_pathway_reference"].iloc[0]
            ),
            "factor_unit": str(
                region_scenarios["baseline_waste_treatment_factor_unit_reference"].iloc[0]
            ),
            "factor_value": float(
                region_scenarios[
                    "baseline_waste_treatment_emission_factor_kgco2e_per_short_ton_reference"
                ].iloc[0]
            ),
        },
        "policy_reference_metrics": {
            "policy_effective_year_reference": float(
                region_scenarios["policy_effective_year_reference"].iloc[0]
            ),
            "policy_organic_waste_disposal_reduction_target_pct_reference": float(
                region_scenarios["policy_organic_waste_disposal_reduction_target_pct_reference"].iloc[0]
            ),
            "policy_edible_food_recovery_target_pct_reference": float(
                region_scenarios["policy_edible_food_recovery_target_pct_reference"].iloc[0]
            ),
            "policy_procurement_target_ton_per_capita_reference": float(
                region_scenarios["policy_procurement_target_ton_per_capita_reference"].iloc[0]
            ),
        },
        "facility_capacity_status": str(region_scenarios["facility_capacity_data_status"].iloc[0]),
        "facility_capacity_reporting_cycle_reference": str(
            region_scenarios["facility_capacity_reporting_cycle_reference"].iloc[0]
        ),
        "organics_facility_inventory_status": str(
            region_scenarios["organics_facility_inventory_status"].iloc[0]
        ),
    }


def main() -> None:
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)

    california_reference = load_california_food_waste_reference()
    region_scenarios = load_region_scenarios()
    mixed_df = build_mixed_rows()
    optimization_df = build_optimization_rows(mixed_df, california_reference, region_scenarios)

    mixed_out = UNIFIED_DIR / "paper1_mixed_waste_feature_prototypes.csv"
    optimization_out = MODEL_READY_DIR / "optimization_input_dataset.csv"
    assumptions_out = MODEL_READY_DIR / "mixed_waste_feature_assumptions.json"

    mixed_df.to_csv(mixed_out, index=False)
    optimization_df.to_csv(optimization_out, index=False)

    assumptions_payload = build_assumptions_payload(
        mixed_df,
        optimization_df,
        california_reference,
        region_scenarios,
    )
    assumptions_out.write_text(json.dumps(assumptions_payload, indent=2), encoding="utf-8")

    print(f"Wrote {mixed_out}")
    print(f"Wrote {optimization_out}")
    print(f"Wrote {assumptions_out}")


if __name__ == "__main__":
    main()
