from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_CA_DIR = ROOT / "data" / "raw" / "external-region-data" / "california"
WET_WASTE_DIR = RAW_CA_DIR / "wet_waste_supply"
UNIFIED_DIR = ROOT / "data" / "processed" / "unified_features"
SCENARIO_DIR = ROOT / "data" / "processed" / "scenario_inputs"
MODEL_READY_DIR = ROOT / "data" / "processed" / "model_ready"


def load_json_frame(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))


def load_food_waste_composition_reference() -> dict[str, float]:
    combined = pd.read_csv(UNIFIED_DIR / "wet_waste_biomass_opt_combined_standardized.csv")
    food = combined[combined["feedstock_group"] == "food_waste"].copy()
    if food.empty:
        raise RuntimeError("No food_waste rows found in wet_waste_biomass_opt_combined_standardized.csv")

    moisture = float(food["feedstock_moisture_pct"].median())
    carbon = float(food["feedstock_carbon_pct"].median())
    nitrogen = float(food["feedstock_nitrogen_pct"].median())
    ash = float(food["feedstock_ash_pct"].median())

    return {
        "moisture_pct_reference": moisture,
        "carbon_pct_reference": carbon,
        "nitrogen_pct_reference": nitrogen,
        "organic_fraction_pct_reference": 100.0 - ash,
    }


def build_county_summary() -> pd.DataFrame:
    county_reference = load_json_frame(WET_WASTE_DIR / "calrecycle_county_reference.json")
    residential_streams = load_json_frame(WET_WASTE_DIR / "calrecycle_residential_streams_countywide.json")
    business_materials = load_json_frame(WET_WASTE_DIR / "calrecycle_business_food_streams_countywide.json")
    census_population = load_json_frame(RAW_CA_DIR / "region_context" / "us_census_acs5_2023_california_county_population.json")

    residential_population = (
        residential_streams[
            [
                "county_id",
                "county_name",
                "PopulationHousehold",
                "OccupiedMultiFamilyUnits",
                "LocalGovernments",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "PopulationHousehold": "household_population_wcs",
                "OccupiedMultiFamilyUnits": "multi_family_units_wcs",
                "LocalGovernments": "calrecycle_region_name",
            }
        )
    )

    residential_food = residential_streams[residential_streams["MaterialTypeName"] == "Food"].copy()
    residential_food = residential_food[
        [
            "county_id",
            "county_name",
            "TonsTotal",
            "PercentTotal",
            "TonsSingleFamily",
            "TonsMultiFamily",
        ]
    ].rename(
        columns={
            "TonsTotal": "residential_food_tons_total",
            "PercentTotal": "residential_food_fraction_total",
            "TonsSingleFamily": "residential_food_tons_single_family",
            "TonsMultiFamily": "residential_food_tons_multi_family",
        }
    )

    residential_other_organic = residential_streams[
        residential_streams["MaterialTypeName"].isin(["Leaves and Grass", "Prunings and Trimmings", "Branches and Stumps"])
    ].copy()
    residential_other_organic = (
        residential_other_organic.groupby(["county_id", "county_name"], as_index=False)["TonsTotal"]
        .sum()
        .rename(columns={"TonsTotal": "residential_other_organic_tons_total"})
    )

    business_food = business_materials[business_materials["material_focus"] == "food"].copy()
    business_food = (
        business_food.groupby(["county_id", "county_name"], as_index=False)[
            ["TonsTotalGeneration", "TonsDisposed", "TonsCurbsideOrganics", "TonsOtherDiversion"]
        ]
        .sum()
        .rename(
            columns={
                "TonsTotalGeneration": "commercial_food_tons_total_generation",
                "TonsDisposed": "commercial_food_tons_disposed",
                "TonsCurbsideOrganics": "commercial_food_tons_curbside_organics",
                "TonsOtherDiversion": "commercial_food_tons_other_diversion",
            }
        )
    )

    census_population = pd.DataFrame(
        census_population.iloc[1:].values, columns=census_population.iloc[0].tolist()
    ).rename(columns={"NAME": "county_census_name", "B01003_001E": "acs5_population_2023", "county": "county_fips"})
    census_population["county_name"] = census_population["county_census_name"].str.replace(", California", "", regex=False)
    census_population["acs5_population_2023"] = census_population["acs5_population_2023"].astype(int)
    residential_population["household_population_wcs"] = residential_population["household_population_wcs"].astype(int)
    residential_population["multi_family_units_wcs"] = residential_population["multi_family_units_wcs"].astype(float)

    summary = county_reference.merge(
        residential_population,
        on=["county_id", "county_name"],
        how="left",
    )
    summary = summary.merge(residential_food, on=["county_id", "county_name"], how="left")
    summary = summary.merge(residential_other_organic, on=["county_id", "county_name"], how="left")
    summary = summary.merge(business_food, on=["county_id", "county_name"], how="left")
    summary = summary.merge(census_population[["county_name", "acs5_population_2023", "county_fips"]], on="county_name", how="left")

    numeric_fill_zero = [
        "residential_food_tons_total",
        "residential_food_fraction_total",
        "residential_food_tons_single_family",
        "residential_food_tons_multi_family",
        "residential_other_organic_tons_total",
        "commercial_food_tons_total_generation",
        "commercial_food_tons_disposed",
        "commercial_food_tons_curbside_organics",
        "commercial_food_tons_other_diversion",
    ]
    summary[numeric_fill_zero] = summary[numeric_fill_zero].fillna(0.0)
    summary["single_family_units_wcs"] = (
        summary["acs5_population_2023"] - summary["multi_family_units_wcs"]
    ).clip(lower=0)

    summary["total_food_waste_tons_reference"] = (
        summary["residential_food_tons_total"] + summary["commercial_food_tons_total_generation"]
    )
    summary["total_other_organic_tons_reference"] = summary["residential_other_organic_tons_total"]
    summary["total_wet_waste_proxy_tons_reference"] = (
        summary["total_food_waste_tons_reference"] + summary["total_other_organic_tons_reference"]
    )
    return summary.sort_values("county_id").reset_index(drop=True)


def build_statewide_reference(county_summary: pd.DataFrame) -> pd.DataFrame:
    totals = county_summary[
        [
            "residential_food_tons_total",
            "commercial_food_tons_total_generation",
            "commercial_food_tons_disposed",
            "commercial_food_tons_curbside_organics",
            "commercial_food_tons_other_diversion",
            "total_food_waste_tons_reference",
            "total_other_organic_tons_reference",
            "total_wet_waste_proxy_tons_reference",
        ]
    ].sum()
    composition = load_food_waste_composition_reference()

    food_collection_rate_pct = 0.0
    if totals["commercial_food_tons_total_generation"] > 0:
        food_collection_rate_pct = (
            (totals["commercial_food_tons_curbside_organics"] + totals["commercial_food_tons_other_diversion"])
            / totals["commercial_food_tons_total_generation"]
            * 100.0
        )

    row = {
        "region_id": "us_ca",
        "region_name": "California",
        "country": "United States",
        "analysis_reference_year": 2023,
        "waste_stream_type": "food_waste",
        "waste_generation_ton_per_year": float(totals["total_food_waste_tons_reference"]),
        "collection_rate_pct_reference": float(food_collection_rate_pct),
        "moisture_pct_reference": composition["moisture_pct_reference"],
        "organic_fraction_pct_reference": composition["organic_fraction_pct_reference"],
        "carbon_pct_reference": composition["carbon_pct_reference"],
        "nitrogen_pct_reference": composition["nitrogen_pct_reference"],
        "residential_food_tons_reference": float(totals["residential_food_tons_total"]),
        "commercial_food_tons_reference": float(totals["commercial_food_tons_total_generation"]),
        "commercial_food_disposed_tons_reference": float(totals["commercial_food_tons_disposed"]),
        "commercial_food_curbside_organics_tons_reference": float(totals["commercial_food_tons_curbside_organics"]),
        "commercial_food_other_diversion_tons_reference": float(totals["commercial_food_tons_other_diversion"]),
        "notes": "California food-waste generation reference built from CalRecycle countywide residential and business stream estimates, with composition medians from Wet-Waste-Biomass-Opt food-waste samples.",
    }
    return pd.DataFrame([row])


def build_model_input(statewide_reference: pd.DataFrame) -> pd.DataFrame:
    frame = statewide_reference.copy()
    frame["source_bundle"] = "california_official_wet_waste_reference"
    frame["pathway_relevance"] = "food_waste_primary"
    return frame


def main() -> None:
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)

    county_summary = build_county_summary()
    statewide_reference = build_statewide_reference(county_summary)
    model_input = build_model_input(statewide_reference)

    county_out = SCENARIO_DIR / "california_county_wet_waste_reference.csv"
    statewide_out = SCENARIO_DIR / "california_food_waste_reference.csv"
    model_out = MODEL_READY_DIR / "california_food_waste_model_input.csv"
    metadata_out = MODEL_READY_DIR / "california_food_waste_model_input_metadata.json"

    county_summary.to_csv(county_out, index=False)
    statewide_reference.to_csv(statewide_out, index=False)
    model_input.to_csv(model_out, index=False)

    metadata = {
        "dataset_name": "california_food_waste_model_input",
        "raw_inputs": [
            "data/raw/external-region-data/california/wet_waste_supply/calrecycle_residential_streams_countywide.json",
            "data/raw/external-region-data/california/wet_waste_supply/calrecycle_business_food_streams_countywide.json",
            "data/raw/external-region-data/california/region_context/us_census_acs5_2023_california_county_population.json",
            "data/processed/unified_features/wet_waste_biomass_opt_combined_standardized.csv",
        ],
        "notes": [
            "Residential and business countywide stream estimates are based on public CalRecycle WasteCharacterization endpoints.",
            "California food-waste generation is estimated by summing county-level residential food and commercial food generation references.",
            "Composition fields are anchored to repository-derived food-waste samples from Wet-Waste-Biomass-Opt because California official county stream endpoints provide tonnage and fractions but not chemical composition.",
        ],
        "row_count": int(len(model_input)),
    }
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {county_out}")
    print(f"Wrote {statewide_out}")
    print(f"Wrote {model_out}")
    print(f"Wrote {metadata_out}")


if __name__ == "__main__":
    main()
