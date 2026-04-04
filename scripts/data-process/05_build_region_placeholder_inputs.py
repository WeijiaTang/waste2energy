from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_EXTERNAL_DIR = ROOT / "data" / "raw" / "external-region-data"
SCENARIO_DIR = ROOT / "data" / "processed" / "scenario_inputs"
MODEL_READY_DIR = ROOT / "data" / "processed" / "model_ready"
UNIFIED_DIR = ROOT / "data" / "processed" / "unified_features"
CALIFORNIA_RAW_DIR = RAW_EXTERNAL_DIR / "california"
CALIFORNIA_SOURCE_MANIFEST = CALIFORNIA_RAW_DIR / "source_manifest.csv"
CALIFORNIA_WET_WASTE_MANIFEST = (
    CALIFORNIA_RAW_DIR / "wet_waste_supply" / "calrecycle_countywide_manifest.csv"
)
CALIFORNIA_WET_WASTE_REFERENCE = SCENARIO_DIR / "california_food_waste_reference.csv"
CALIFORNIA_WET_WASTE_MODEL_INPUT = MODEL_READY_DIR / "california_food_waste_model_input.csv"
CALIFORNIA_EGRID_WORKBOOK = (
    CALIFORNIA_RAW_DIR / "emission_factors" / "epa_egrid_2023_data.xlsx"
)
CALIFORNIA_WARM_REFERENCE = (
    CALIFORNIA_RAW_DIR
    / "emission_factors"
    / "california_waste_treatment_emission_factor_reference.csv"
)
CALIFORNIA_TREATMENT_MIX_REFERENCE = (
    SCENARIO_DIR / "california_food_waste_treatment_mix_reference.csv"
)
CALIFORNIA_ENERGY_REFERENCE = (
    CALIFORNIA_RAW_DIR / "energy_prices" / "california_energy_price_reference.csv"
)
CALIFORNIA_LIVESTOCK_OVERVIEW = (
    CALIFORNIA_RAW_DIR / "livestock_supply" / "usda_nass_california_livestock_overview_structured.csv"
)
CALIFORNIA_LIVESTOCK_RELEASE = (
    CALIFORNIA_RAW_DIR / "livestock_supply" / "usda_nass_california_livestock_release_summary.csv"
)
CALIFORNIA_POLICY_REFERENCE = (
    CALIFORNIA_RAW_DIR / "policy_reference" / "california_sb1383_policy_reference.csv"
)
CALIFORNIA_CAPACITY_SUMMARY = SCENARIO_DIR / "california_capacity_planning_summary.csv"
CALIFORNIA_CAPACITY_STATUS = MODEL_READY_DIR / "california_capacity_planning_status.json"
CALIFORNIA_FACILITY_INVENTORY = SCENARIO_DIR / "california_organics_facility_inventory.csv"
CALIFORNIA_FACILITY_SUMMARY = SCENARIO_DIR / "california_organics_facility_summary.csv"
CALIFORNIA_FACILITY_STATUS = MODEL_READY_DIR / "california_organics_facility_inventory_status.json"
TARGET_REGION_ID = "us_ca"
ANALYSIS_YEAR = 2024
EGRID_SOURCE_ID = "california_epa_egrid_xlsx"
WARM_SOURCE_ID = "california_epa_warm_organic_materials_pdf"

NRCS_MANURE_GENERATION_FACTORS = {
    "dairy": 0.0392,
    "swine": 1.91,
}
MANURE_COLLECTION_RATES = {
    "dairy": 85.0,
    "swine": 95.0,
}
LIVESTOCK_SOURCE_URLS = {
    "dairy": "https://www.nass.usda.gov/Quick_Stats/Ag_Overview/stateOverview.php?state=CALIFORNIA",
    "swine": "https://www.nass.usda.gov/Statistics_by_State/California/Publications/Livestock_Releases/Hogs/202412HOGS.pdf",
}


def load_single_row_csv(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise RuntimeError(f"Expected exactly one row in {path}, found {len(frame)}")
    return frame.iloc[0].to_dict()


def load_source_lookup() -> dict[str, dict[str, object]]:
    manifest = pd.read_csv(CALIFORNIA_SOURCE_MANIFEST)
    categories = [
        "region_context",
        "livestock_supply",
        "energy_prices",
        "emission_factors",
        "wet_waste_supply",
    ]
    lookup: dict[str, dict[str, object]] = {}
    for category in categories:
        subset = manifest[manifest["data_category"] == category].copy()
        if subset.empty:
            continue
        downloaded = subset[subset["status"] == "downloaded"].copy()
        if not downloaded.empty:
            subset = downloaded
        lookup[category] = subset.iloc[0].to_dict()
    return lookup


def load_source_record(source_id: str) -> dict[str, object]:
    manifest = pd.read_csv(CALIFORNIA_SOURCE_MANIFEST)
    match = manifest[manifest["source_id"] == source_id].copy()
    if match.empty:
        raise RuntimeError(f"Could not find source_id={source_id} in {CALIFORNIA_SOURCE_MANIFEST}")
    return match.iloc[0].to_dict()


def load_wet_waste_endpoint_bundle() -> dict[str, str]:
    manifest = pd.read_csv(CALIFORNIA_WET_WASTE_MANIFEST)
    urls = sorted(manifest["source_url_template"].dropna().unique().tolist())
    outputs = sorted(Path(value).name for value in manifest["output_file"].dropna().tolist())
    return {
        "source_organization": "; ".join(sorted(manifest["source_organization"].dropna().unique().tolist())),
        "source_url": " | ".join(urls),
        "download_or_publication_date": str(manifest["downloaded_at_utc"].dropna().max()),
        "raw_file_name": "; ".join(outputs),
    }


def load_california_grid_emission_factor() -> float:
    state_frame = pd.read_excel(CALIFORNIA_EGRID_WORKBOOK, sheet_name="ST23")
    california_row = state_frame[state_frame["State abbreviation"] == "CA"]
    if california_row.empty:
        raise RuntimeError("Could not find California row in EPA eGRID ST23 sheet")

    lb_per_mwh = float(
        california_row.iloc[0]["State annual CO2 equivalent total output emission rate (lb/MWh)"]
    )
    return lb_per_mwh * 0.45359237 / 1000.0


def load_baseline_waste_treatment_factor() -> dict[str, object]:
    if CALIFORNIA_TREATMENT_MIX_REFERENCE.exists():
        frame = pd.read_csv(CALIFORNIA_TREATMENT_MIX_REFERENCE)
        california = frame[
            (frame["region_id"] == TARGET_REGION_ID)
            & (frame["baseline_relevance"] == "baseline_default")
        ].copy()
        if not california.empty:
            row = california.iloc[0].to_dict()
            row["baseline_source_method"] = "california_weighted_mix"
            return row

    frame = pd.read_csv(CALIFORNIA_WARM_REFERENCE)
    california = frame[
        (frame["region_id"] == TARGET_REGION_ID)
        & (frame["baseline_relevance"] == "baseline_default")
    ].copy()
    if california.empty:
        raise RuntimeError(
            "Could not find baseline_default row in "
            "california_waste_treatment_emission_factor_reference.csv"
        )
    row = california.iloc[0].to_dict()
    row["baseline_source_method"] = "epa_warm_proxy"
    return row


def load_energy_reference() -> pd.DataFrame:
    frame = pd.read_csv(CALIFORNIA_ENERGY_REFERENCE)
    california = frame[frame["region_id"] == TARGET_REGION_ID].copy()
    if california.empty:
        raise RuntimeError("No us_ca rows found in california_energy_price_reference.csv")
    return california


def load_manure_moisture_reference() -> dict[str, float]:
    frame = pd.read_csv(UNIFIED_DIR / "manure_pyrolysis_energy_balance_long.csv")
    subset = frame[frame["constant_name"] == "Initial Moisture Content"].copy()
    subset["moisture_pct"] = subset["value"] * 100.0
    return dict(zip(subset["livestock_type"], subset["moisture_pct"]))


def load_livestock_counts() -> pd.DataFrame:
    overview = pd.read_csv(CALIFORNIA_LIVESTOCK_OVERVIEW)
    release = pd.read_csv(CALIFORNIA_LIVESTOCK_RELEASE)

    dairy_rows = overview[
        overview["metric_name"].astype(str).str.startswith("Cattle, Cows, Milk - Inventory")
    ].copy()
    swine_rows = release[release["metric_name"] == "hogs_inventory"].copy()

    dairy_count = dairy_rows.sort_values("reference_year", ascending=False)["value"]
    swine_count = swine_rows.sort_values("reference_year", ascending=False)["value"]

    if dairy_count.empty or pd.isna(dairy_count.iloc[0]):
        raise RuntimeError("Missing California dairy inventory in livestock overview structured file")
    if swine_count.empty or pd.isna(swine_count.iloc[0]):
        raise RuntimeError("Missing California swine inventory in livestock release summary file")

    return pd.DataFrame(
        [
            {
                "livestock_type": "dairy",
                "animal_count": float(dairy_count.iloc[0]),
                "source_url": LIVESTOCK_SOURCE_URLS["dairy"],
                "raw_file_name": "usda_nass_california_livestock_overview_structured.csv",
                "source_basis": "USDA NASS California state overview",
            },
            {
                "livestock_type": "swine",
                "animal_count": float(swine_count.iloc[0]),
                "source_url": LIVESTOCK_SOURCE_URLS["swine"],
                "raw_file_name": "usda_nass_california_livestock_release_summary.csv",
                "source_basis": "USDA NASS Pacific Region hogs release",
            },
        ]
    )


def load_capacity_bundle() -> dict[str, object]:
    status_payload: dict[str, object] = {}
    if CALIFORNIA_CAPACITY_STATUS.exists():
        status_payload = json.loads(CALIFORNIA_CAPACITY_STATUS.read_text(encoding="utf-8"))

    if CALIFORNIA_CAPACITY_SUMMARY.exists():
        summary_frame = pd.read_csv(CALIFORNIA_CAPACITY_SUMMARY)
        if "reporting_cycle" in summary_frame.columns:
            preferred = summary_frame[
                summary_frame["reporting_cycle"].astype(str) == "2025-2034"
            ].copy()
            if preferred.empty:
                preferred = summary_frame.sort_values("reporting_cycle").tail(1)
            summary = preferred.iloc[0].to_dict()
        else:
            if len(summary_frame) != 1:
                raise RuntimeError(
                    f"Expected one summary row in {CALIFORNIA_CAPACITY_SUMMARY}, found {len(summary_frame)}"
                )
            summary = summary_frame.iloc[0].to_dict()
        return {
            "status": "processed_summary_connected",
            "summary": summary,
            "reference_file": "data/processed/scenario_inputs/california_capacity_planning_summary.csv",
            "status_file": "data/processed/model_ready/california_capacity_planning_status.json",
        }

    return {
        "status": str(status_payload.get("status", "awaiting_manual_export")),
        "summary": None,
        "reference_file": "",
        "status_file": "data/processed/model_ready/california_capacity_planning_status.json"
        if CALIFORNIA_CAPACITY_STATUS.exists()
        else "",
    }


def load_facility_inventory_bundle() -> dict[str, object]:
    status_payload: dict[str, object] = {}
    if CALIFORNIA_FACILITY_STATUS.exists():
        status_payload = json.loads(CALIFORNIA_FACILITY_STATUS.read_text(encoding="utf-8"))

    if CALIFORNIA_FACILITY_SUMMARY.exists() and CALIFORNIA_FACILITY_INVENTORY.exists():
        summary_frame = pd.read_csv(CALIFORNIA_FACILITY_SUMMARY)
        inventory_frame = pd.read_csv(CALIFORNIA_FACILITY_INVENTORY)
        return {
            "status": "official_reference_connected",
            "summary": summary_frame.copy(),
            "inventory": inventory_frame.copy(),
            "reference_file": "data/processed/scenario_inputs/california_organics_facility_summary.csv",
            "inventory_file": "data/processed/scenario_inputs/california_organics_facility_inventory.csv",
            "status_file": "data/processed/model_ready/california_organics_facility_inventory_status.json",
        }

    return {
        "status": str(status_payload.get("status", "pending_swis_facility_inventory")),
        "summary": None,
        "inventory": None,
        "reference_file": "",
        "inventory_file": "",
        "status_file": "data/processed/model_ready/california_organics_facility_inventory_status.json"
        if CALIFORNIA_FACILITY_STATUS.exists()
        else "",
    }


def load_policy_bundle() -> dict[str, object]:
    if not CALIFORNIA_POLICY_REFERENCE.exists():
        return {
            "status": "pending_policy_dataset_curation",
            "frame": None,
            "reference_file": "",
        }

    frame = pd.read_csv(CALIFORNIA_POLICY_REFERENCE)
    california = frame[frame["region_id"] == TARGET_REGION_ID].copy()
    if california.empty:
        raise RuntimeError("No us_ca rows found in california_sb1383_policy_reference.csv")
    return {
        "status": "official_reference_connected",
        "frame": california.reset_index(drop=True),
        "reference_file": "data/raw/external-region-data/california/policy_reference/california_sb1383_policy_reference.csv",
    }


def extract_policy_metric(policy_frame: pd.DataFrame, policy_name: str) -> float | str:
    match = policy_frame[policy_frame["policy_name"] == policy_name].copy()
    if match.empty:
        return ""
    value = match.iloc[0]["support_level_value"]
    if pd.isna(value):
        return ""
    return float(value)


def build_region_metadata_template(
    california_reference: dict[str, object], source_lookup: dict[str, dict[str, object]]
) -> pd.DataFrame:
    region_context = source_lookup["region_context"]
    return pd.DataFrame(
        [
            {
                "region_id": TARGET_REGION_ID,
                "region_name": california_reference["region_name"],
                "country": california_reference["country"],
                "admin_level": "state",
                "analysis_year": ANALYSIS_YEAR,
                "time_resolution": "annual",
                "notes": (
                    "Waste2Energy Paper 1 region is locked to California and linked to official Census, "
                    "CalRecycle, EIA, EPA, and USDA reference files."
                ),
                "source_organization": region_context["source_organization"],
                "source_url": region_context["source_url"],
                "download_or_publication_date": region_context["downloaded_at_utc"],
            }
        ]
    )


def build_livestock_template() -> pd.DataFrame:
    livestock_counts = load_livestock_counts()
    moisture_lookup = load_manure_moisture_reference()
    timestamp = pd.Timestamp.utcnow().isoformat()
    rows = []
    for _, livestock in livestock_counts.iterrows():
        livestock_type = str(livestock["livestock_type"])
        animal_count = float(livestock["animal_count"])
        manure_generation = animal_count * NRCS_MANURE_GENERATION_FACTORS[livestock_type]
        rows.append(
            {
                "region_id": TARGET_REGION_ID,
                "year": ANALYSIS_YEAR,
                "livestock_type": livestock_type,
                "animal_count": animal_count,
                "manure_generation_ton_per_year": manure_generation,
                "manure_collection_rate_pct": MANURE_COLLECTION_RATES[livestock_type],
                "manure_moisture_pct": moisture_lookup[livestock_type],
                "data_unit_notes": (
                    "Animal counts are official California USDA values. Manure generation uses NRCS AWMFH "
                    "planning factors converted to wet tons per head-year."
                ),
                "source_organization": "USDA National Agricultural Statistics Service; USDA NRCS",
                "source_url": livestock["source_url"],
                "download_or_publication_date": timestamp,
                "raw_file_name": livestock["raw_file_name"],
                "notes": (
                    f"{livestock['source_basis']}; manure generation factor={NRCS_MANURE_GENERATION_FACTORS[livestock_type]} "
                    "wet tons/head-year; manure moisture from repository manure energy-balance reference."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_wet_waste_template(california_reference: dict[str, object]) -> pd.DataFrame:
    wet_waste_bundle = load_wet_waste_endpoint_bundle()
    rows = [
        {
            "region_id": TARGET_REGION_ID,
            "year": int(california_reference["analysis_reference_year"]),
            "waste_stream_type": "food_waste",
            "waste_generation_ton_per_year": float(california_reference["waste_generation_ton_per_year"]),
            "collection_rate_pct": float(california_reference["collection_rate_pct_reference"]),
            "moisture_pct": float(california_reference["moisture_pct_reference"]),
            "organic_fraction_pct": float(california_reference["organic_fraction_pct_reference"]),
            "carbon_pct": float(california_reference["carbon_pct_reference"]),
            "nitrogen_pct": float(california_reference["nitrogen_pct_reference"]),
            "data_unit_notes": (
                "Statewide annual tons with composition fields anchored to repository food-waste medians."
            ),
            "source_organization": wet_waste_bundle["source_organization"],
            "source_url": wet_waste_bundle["source_url"],
            "download_or_publication_date": wet_waste_bundle["download_or_publication_date"],
            "raw_file_name": wet_waste_bundle["raw_file_name"],
            "notes": (
                "Directly connected to california_food_waste_reference.csv and "
                "california_food_waste_model_input.csv for the us_ca Paper 1 region."
            ),
        },
        {
            "region_id": TARGET_REGION_ID,
            "year": int(california_reference["analysis_reference_year"]),
            "waste_stream_type": "municipal_wet_waste",
            "waste_generation_ton_per_year": "",
            "collection_rate_pct": "",
            "moisture_pct": "",
            "organic_fraction_pct": "",
            "carbon_pct": "",
            "nitrogen_pct": "",
            "data_unit_notes": "Reserved for a later broader California organics aggregation beyond food waste.",
            "source_organization": wet_waste_bundle["source_organization"],
            "source_url": wet_waste_bundle["source_url"],
            "download_or_publication_date": wet_waste_bundle["download_or_publication_date"],
            "raw_file_name": wet_waste_bundle["raw_file_name"],
            "notes": "Food-waste pathway is connected; broader municipal wet-waste proxy remains optional.",
        },
    ]
    return pd.DataFrame(rows)


def build_energy_price_template() -> pd.DataFrame:
    energy_reference = load_energy_reference().copy()
    energy_reference["year"] = energy_reference["analysis_year"].astype(int)
    energy_reference["source_url"] = energy_reference["source_url"].astype(str)
    energy_reference["download_or_publication_date"] = pd.Timestamp.utcnow().isoformat()

    converted_rows = []
    for _, row in energy_reference.iterrows():
        price_value = float(row["price_value"])
        price_unit = str(row["price_unit"])
        if row["energy_carrier"] == "electricity" and price_unit == "cents_per_kWh":
            price_value = price_value / 100.0
            price_unit = "USD_per_kWh"
        elif row["energy_carrier"] == "natural_gas" and price_unit == "USD_per_thousand_cubic_feet":
            price_unit = "USD_per_Mcf"

        converted_rows.append(
            {
                "region_id": TARGET_REGION_ID,
                "year": int(row["analysis_year"]),
                "energy_carrier": row["energy_carrier"],
                "price_value": price_value,
                "price_unit": price_unit,
                "price_case": row["price_case"],
                "source_organization": row["source_organization"],
                "source_url": row["source_url"],
                "download_or_publication_date": pd.Timestamp.utcnow().isoformat(),
                "raw_file_name": row["raw_file_name"],
                "notes": row["notes"],
            }
        )
    return pd.DataFrame(converted_rows)


def build_emission_factor_template(
    grid_emission_factor: float,
    baseline_waste_treatment_factor: dict[str, object],
) -> pd.DataFrame:
    grid_source = load_source_record(EGRID_SOURCE_ID)
    grid_raw_file_name = Path(str(grid_source["target_file"])).name
    rows = [
        {
            "region_id": TARGET_REGION_ID,
            "year": ANALYSIS_YEAR,
            "factor_name": "grid_electricity_emission_factor",
            "factor_value": grid_emission_factor,
            "factor_unit": "kgCO2e_per_kWh",
            "source_organization": grid_source["source_organization"],
            "source_url": grid_source["source_url"],
            "download_or_publication_date": grid_source["downloaded_at_utc"],
            "raw_file_name": grid_raw_file_name,
            "notes": (
                "Converted from EPA eGRID California state CO2-equivalent total output emission rate (lb/MWh)."
            ),
        },
        {
            "region_id": TARGET_REGION_ID,
            "year": ANALYSIS_YEAR,
            "factor_name": "baseline_waste_treatment_emission_factor",
            "factor_value": float(baseline_waste_treatment_factor["factor_value"]),
            "factor_unit": str(baseline_waste_treatment_factor["factor_unit"]),
            "source_organization": baseline_waste_treatment_factor["source_organization"],
            "source_url": baseline_waste_treatment_factor["source_url"],
            "download_or_publication_date": baseline_waste_treatment_factor[
                "download_or_publication_date"
            ],
            "raw_file_name": baseline_waste_treatment_factor["raw_file_name"],
            "notes": (
                (
                    "California-weighted food-waste baseline built from California disposal/treatment mix plus EPA WARM pathway factors. "
                    if baseline_waste_treatment_factor.get("baseline_source_method")
                    == "california_weighted_mix"
                    else "Default California Paper 1 baseline proxy from EPA WARM "
                    f"{baseline_waste_treatment_factor.get('source_exhibit', '')} "
                    f"({baseline_waste_treatment_factor['management_pathway']}). "
                )
                + f"{baseline_waste_treatment_factor['notes']}"
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_policy_reference_template() -> pd.DataFrame:
    policy_bundle = load_policy_bundle()
    if policy_bundle["frame"] is None:
        return pd.DataFrame(
            [
                {
                    "region_id": TARGET_REGION_ID,
                    "policy_name": "california_organics_diversion_policy_placeholder",
                    "policy_type": "regulatory_target_or_credit",
                    "policy_start_year": "",
                    "policy_end_year": "",
                    "support_level_value": "",
                    "support_level_unit": "",
                    "target_pathway": "general",
                    "source_organization": "",
                    "source_url": "",
                    "download_or_publication_date": "",
                    "raw_file_name": "",
                    "notes": (
                        "Region fixed to California; curated policy normalization can later encode SB 1383 and related "
                        "support mechanisms here."
                    ),
                }
            ]
        )
    return policy_bundle["frame"].copy()


def build_facility_capacity_template() -> pd.DataFrame:
    capacity_bundle = load_capacity_bundle()
    inventory_bundle = load_facility_inventory_bundle()
    inventory_summary = inventory_bundle["summary"]
    inventory_frame = inventory_bundle["inventory"]

    inventory_metrics: dict[str, object] = {
        "facility_inventory_data_status": inventory_bundle["status"],
        "organics_facility_count": "",
        "food_waste_facility_count": "",
        "composting_facility_count": "",
        "anaerobic_digestion_facility_count": "",
        "transfer_processing_facility_count": "",
        "facility_inventory_reference_file": inventory_bundle["reference_file"]
        or inventory_bundle["status_file"],
    }
    if inventory_summary is not None and inventory_frame is not None:
        inventory_metrics = {
            "facility_inventory_data_status": "official_reference_connected",
            "organics_facility_count": int(inventory_frame["SWIS Number"].dropna().nunique()),
            "food_waste_facility_count": int(
                inventory_frame.loc[
                    inventory_frame["accepts_food_wastes"].astype(bool), "SWIS Number"
                ]
                .dropna()
                .nunique()
            ),
            "composting_facility_count": int(
                inventory_frame.loc[
                    inventory_frame["technology_group"] == "composting", "SWIS Number"
                ]
                .dropna()
                .nunique()
            ),
            "anaerobic_digestion_facility_count": int(
                inventory_frame.loc[
                    inventory_frame["technology_group"] == "anaerobic_digestion", "SWIS Number"
                ]
                .dropna()
                .nunique()
            ),
            "transfer_processing_facility_count": int(
                inventory_frame.loc[
                    inventory_frame["technology_group"] == "transfer_processing", "SWIS Number"
                ]
                .dropna()
                .nunique()
            ),
            "facility_inventory_reference_file": inventory_bundle["reference_file"],
        }

    if capacity_bundle["summary"] is None:
        return pd.DataFrame(
            [
                {
                    "region_id": TARGET_REGION_ID,
                    "year": ANALYSIS_YEAR,
                    "capacity_data_status": capacity_bundle["status"],
                    "facility_row_count": "",
                    "unique_facility_count": "",
                    "unique_jurisdiction_count": "",
                    "total_permitted_capacity_ton_per_year": "",
                    "total_available_capacity_ton_per_year": "",
                    "total_used_capacity_ton_per_year": "",
                    "source_organization": "California Department of Resources Recycling and Recovery",
                    "source_url": "https://www2.calrecycle.ca.gov/LGCentral/datatools/reports/capacityplanning",
                    "download_or_publication_date": "",
                    "raw_file_name": "",
                    **inventory_metrics,
                    "notes": (
                        "CalRecycle capacity-planning intake is configured but still waiting for an official manual "
                        "export to be registered and processed."
                    ),
                }
            ]
        )

    summary = capacity_bundle["summary"]
    return pd.DataFrame(
        [
            {
                "region_id": TARGET_REGION_ID,
                "year": ANALYSIS_YEAR,
                "capacity_data_status": summary["capacity_data_status"],
                "reporting_cycle": summary.get("reporting_cycle", ""),
                "facility_row_count": int(summary["facility_row_count"]),
                "unique_facility_count": int(summary["unique_facility_count"]),
                "unique_jurisdiction_count": int(summary["unique_jurisdiction_count"]),
                "total_permitted_capacity_ton_per_year": summary[
                    "total_permitted_capacity_ton_per_year"
                ],
                "total_available_capacity_ton_per_year": summary[
                    "total_available_capacity_ton_per_year"
                ],
                "total_used_capacity_ton_per_year": summary["total_used_capacity_ton_per_year"],
                "total_organic_waste_recycling_capacity_needed_ton_per_year": summary.get(
                    "total_organic_waste_recycling_capacity_needed_ton_per_year", ""
                ),
                "total_edible_food_recovery_capacity_available_ton_per_year": summary.get(
                    "total_edible_food_recovery_capacity_available_ton_per_year", ""
                ),
                "total_edible_food_recovery_capacity_needed_ton_per_year": summary.get(
                    "total_edible_food_recovery_capacity_needed_ton_per_year", ""
                ),
                "source_organization": summary["source_organization"],
                "source_url": summary["source_url"],
                "download_or_publication_date": summary["registered_at_utc"],
                "raw_file_name": summary["registered_export_file"],
                **inventory_metrics,
                "notes": (
                    f"{summary['notes']} Facility-inventory counts are connected separately from official SWIS "
                    "site/activity/waste exports."
                ),
            }
        ]
    )


def build_interface_manifest() -> pd.DataFrame:
    rows = [
        {
            "template_file": "region_metadata_template.csv",
            "data_role": "region_context",
            "required_for_paper1": "yes",
            "joins_on": "region_id",
            "target_processed_output": "region_input_status.csv",
        },
        {
            "template_file": "livestock_manure_statistics_template.csv",
            "data_role": "feedstock_supply",
            "required_for_paper1": "yes",
            "joins_on": "region_id|year",
            "target_processed_output": "region_input_status.csv",
        },
        {
            "template_file": "wet_waste_generation_template.csv",
            "data_role": "feedstock_supply",
            "required_for_paper1": "yes",
            "joins_on": "region_id|year",
            "target_processed_output": "region_input_status.csv",
        },
        {
            "template_file": "regional_energy_prices_template.csv",
            "data_role": "scenario_parameter",
            "required_for_paper1": "yes",
            "joins_on": "region_id|year",
            "target_processed_output": "paper1_region_scenario_placeholder.csv",
        },
        {
            "template_file": "regional_facility_capacity_template.csv",
            "data_role": "infrastructure_capacity",
            "required_for_paper1": "optional_but_recommended",
            "joins_on": "region_id|year",
            "target_processed_output": "paper1_region_scenario_placeholder.csv",
        },
        {
            "template_file": "regional_emission_factors_template.csv",
            "data_role": "environment_parameter",
            "required_for_paper1": "yes",
            "joins_on": "region_id|year",
            "target_processed_output": "paper1_region_scenario_placeholder.csv",
        },
        {
            "template_file": "regional_policy_reference_template.csv",
            "data_role": "policy_parameter",
            "required_for_paper1": "optional_but_recommended",
            "joins_on": "region_id",
            "target_processed_output": "paper1_region_scenario_placeholder.csv",
        },
    ]
    return pd.DataFrame(rows)


def build_region_input_status(
    grid_emission_factor: float, baseline_waste_treatment_factor: dict[str, object]
) -> pd.DataFrame:
    capacity_bundle = load_capacity_bundle()
    facility_inventory_bundle = load_facility_inventory_bundle()
    policy_bundle = load_policy_bundle()
    rows = [
        {
            "region_id": TARGET_REGION_ID,
            "input_group": "region_context",
            "template_file": "region_metadata_template.csv",
            "status": "official_reference_connected",
            "paper1_required": "yes",
            "reference_file": "data/processed/scenario_inputs/california_food_waste_reference.csv",
            "next_action": "Use California as the default Paper 1 region unless the study is narrowed to a sub-state case.",
        },
        {
            "region_id": TARGET_REGION_ID,
            "input_group": "livestock_supply",
            "template_file": "livestock_manure_statistics_template.csv",
            "status": "official_reference_connected",
            "paper1_required": "yes",
            "reference_file": "data/raw/external-region-data/california/livestock_supply/usda_nass_california_livestock_release_summary.csv",
            "next_action": "Refine manure-generation factors later if a California-specific manure coefficient source is added.",
        },
        {
            "region_id": TARGET_REGION_ID,
            "input_group": "wet_waste_supply",
            "template_file": "wet_waste_generation_template.csv",
            "status": "official_reference_connected",
            "paper1_required": "yes",
            "reference_file": "data/processed/model_ready/california_food_waste_model_input.csv",
            "next_action": (
                "California food-waste supply is ready for optimization linkage; optionally extend to broader wet "
                "organics later."
            ),
        },
        {
            "region_id": TARGET_REGION_ID,
            "input_group": "energy_prices",
            "template_file": "regional_energy_prices_template.csv",
            "status": "official_reference_connected",
            "paper1_required": "yes",
            "reference_file": "data/raw/external-region-data/california/energy_prices/california_energy_price_reference.csv",
            "next_action": "Use the 2024 California electricity and natural gas prices as baseline Paper 1 market inputs.",
        },
        {
            "region_id": TARGET_REGION_ID,
            "input_group": "facility_capacity",
            "template_file": "regional_facility_capacity_template.csv",
            "status": (
                "official_reference_connected"
                if capacity_bundle["summary"] is not None
                and facility_inventory_bundle["summary"] is not None
                else (
                    "official_manual_export_connected"
                    if capacity_bundle["summary"] is not None
                    else str(capacity_bundle["status"])
                )
            ),
            "paper1_required": "optional_but_recommended",
            "reference_file": " | ".join(
                value
                for value in [
                    capacity_bundle["reference_file"] or capacity_bundle["status_file"],
                    facility_inventory_bundle["reference_file"]
                    or facility_inventory_bundle["status_file"],
                ]
                if value
            ),
            "next_action": (
                "County-cycle capacity summary and facility-level SWIS organics inventory are both connected."
                if capacity_bundle["summary"] is not None
                and facility_inventory_bundle["summary"] is not None
                else (
                    "Capacity summary is connected from an official CalRecycle manual export."
                    if capacity_bundle["summary"] is not None
                    else "Register an official CalRecycle capacity-planning export, run 07_build_california_capacity_inputs.py, then rerun this region builder."
                )
            ),
        },
        {
            "region_id": TARGET_REGION_ID,
            "input_group": "emission_factors",
            "template_file": "regional_emission_factors_template.csv",
            "status": "official_reference_connected",
            "paper1_required": "yes",
            "reference_file": (
                "data/raw/external-region-data/california/emission_factors/epa_egrid_2023_data.xlsx | "
                + (
                    "data/processed/scenario_inputs/california_food_waste_treatment_mix_reference.csv"
                    if baseline_waste_treatment_factor.get("baseline_source_method")
                    == "california_weighted_mix"
                    else "data/raw/external-region-data/california/emission_factors/"
                    "california_waste_treatment_emission_factor_reference.csv"
                )
            ),
            "next_action": (
                f"California grid electricity factor is populated ({grid_emission_factor:.6f} kgCO2e_per_kWh) "
                "and baseline waste-treatment factor is populated from "
                + (
                    "a California-weighted disposal/treatment mix "
                    if baseline_waste_treatment_factor.get("baseline_source_method")
                    == "california_weighted_mix"
                    else "EPA WARM "
                )
                + (
                f"({baseline_waste_treatment_factor['management_pathway']}="
                f"{float(baseline_waste_treatment_factor['factor_value']):.1f} "
                f"{baseline_waste_treatment_factor['factor_unit']})."
                )
            ),
        },
        {
            "region_id": TARGET_REGION_ID,
            "input_group": "policy_reference",
            "template_file": "regional_policy_reference_template.csv",
            "status": policy_bundle["status"],
            "paper1_required": "optional_but_recommended",
            "reference_file": policy_bundle["reference_file"],
            "next_action": (
                "Use structured SB 1383 regulatory, procurement, and enforcement parameters as California policy baseline."
                if policy_bundle["frame"] is not None
                else "Curate California organics-policy parameters if Paper 1 needs explicit policy sensitivity."
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_region_scenario_placeholder(
    california_model_input: dict[str, object],
    grid_emission_factor: float,
    baseline_waste_treatment_factor: dict[str, object],
) -> pd.DataFrame:
    reference_generation = float(california_model_input["waste_generation_ton_per_year"])
    collection_rate = float(california_model_input["collection_rate_pct_reference"])
    collectable_proxy = reference_generation * collection_rate / 100.0
    energy_reference = build_energy_price_template()
    electricity_price = float(
        energy_reference.loc[energy_reference["energy_carrier"] == "electricity", "price_value"].iloc[0]
    )
    natural_gas_price = float(
        energy_reference.loc[energy_reference["energy_carrier"] == "natural_gas", "price_value"].iloc[0]
    )
    manure_reference = build_livestock_template()
    manure_generation_total = float(manure_reference["manure_generation_ton_per_year"].sum())
    manure_collectable_proxy = float(
        (
            manure_reference["manure_generation_ton_per_year"]
            * manure_reference["manure_collection_rate_pct"]
            / 100.0
        ).sum()
    )
    capacity_bundle = load_capacity_bundle()
    capacity_summary = capacity_bundle["summary"]
    facility_inventory_bundle = load_facility_inventory_bundle()
    facility_inventory_summary = facility_inventory_bundle["summary"]
    facility_inventory = facility_inventory_bundle["inventory"]
    policy_bundle = load_policy_bundle()
    policy_frame = policy_bundle["frame"]
    organic_reduction_target = (
        extract_policy_metric(policy_frame, "sb1383_organic_waste_disposal_reduction_target_2025")
        if policy_frame is not None
        else ""
    )
    edible_food_target = (
        extract_policy_metric(policy_frame, "sb1383_edible_food_recovery_target_2025")
        if policy_frame is not None
        else ""
    )
    procurement_target = (
        extract_policy_metric(policy_frame, "sb1383_procurement_target_per_capita_2022_2026")
        if policy_frame is not None
        else ""
    )
    policy_effective_year = (
        extract_policy_metric(policy_frame, "sb1383_regulations_effective_2022")
        if policy_frame is not None
        else ""
    )
    baseline_waste_treatment_value = float(baseline_waste_treatment_factor["factor_value"])
    organics_facility_count = ""
    composting_facility_count = ""
    anaerobic_digestion_facility_count = ""
    if facility_inventory_summary is not None and facility_inventory is not None:
        organics_facility_count = int(facility_inventory["SWIS Number"].dropna().nunique())
        composting_facility_count = int(
            facility_inventory.loc[
                facility_inventory["technology_group"] == "composting", "SWIS Number"
            ]
            .dropna()
            .nunique()
        )
        anaerobic_digestion_facility_count = int(
            facility_inventory.loc[
                facility_inventory["technology_group"] == "anaerobic_digestion",
                "SWIS Number",
            ]
            .dropna()
            .nunique()
        )

    scenario_specs = [
        (
            "baseline_region_case",
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            "California baseline case using official statewide food-waste, energy-price, and livestock references.",
        ),
        (
            "high_supply_case",
            1.1,
            1.1,
            1.0,
            1.0,
            1.0,
            "California supply-expansion sensitivity around official food-waste and manure baselines.",
        ),
        (
            "policy_support_case",
            1.0,
            1.0,
            1.0,
            1.0,
            1.2,
            "California policy-support sensitivity anchored to official waste, energy, and livestock baselines.",
        ),
    ]

    rows = []
    for (
        scenario_name,
        manure_multiplier,
        wet_waste_multiplier,
        energy_multiplier,
        emission_multiplier,
        policy_multiplier,
        notes,
    ) in scenario_specs:
        rows.append(
            {
                "region_id": TARGET_REGION_ID,
                "region_name": california_model_input["region_name"],
                "country": california_model_input["country"],
                "analysis_reference_year": ANALYSIS_YEAR,
                "scenario_name": scenario_name,
                "manure_supply_multiplier": manure_multiplier,
                "wet_waste_supply_multiplier": wet_waste_multiplier,
                "energy_price_multiplier": energy_multiplier,
                "emission_factor_multiplier": emission_multiplier,
                "policy_multiplier": policy_multiplier,
                "wet_waste_reference_stream_type": california_model_input["waste_stream_type"],
                "wet_waste_generation_ton_per_year_reference": reference_generation,
                "wet_waste_collection_rate_pct_reference": collection_rate,
                "wet_waste_collectable_ton_per_year_proxy_reference": collectable_proxy,
                "scenario_wet_waste_generation_ton_per_year": reference_generation * wet_waste_multiplier,
                "scenario_wet_waste_collectable_ton_per_year_proxy": collectable_proxy * wet_waste_multiplier,
                "manure_generation_ton_per_year_reference": manure_generation_total,
                "manure_collectable_ton_per_year_proxy_reference": manure_collectable_proxy,
                "scenario_manure_generation_ton_per_year": manure_generation_total * manure_multiplier,
                "scenario_manure_collectable_ton_per_year_proxy": manure_collectable_proxy * manure_multiplier,
                "electricity_price_usd_per_kwh_reference": electricity_price,
                "natural_gas_price_usd_per_mcf_reference": natural_gas_price,
                "scenario_electricity_price_usd_per_kwh": electricity_price * energy_multiplier,
                "scenario_natural_gas_price_usd_per_mcf": natural_gas_price * energy_multiplier,
                "policy_effective_year_reference": policy_effective_year,
                "policy_organic_waste_disposal_reduction_target_pct_reference": organic_reduction_target,
                "policy_edible_food_recovery_target_pct_reference": edible_food_target,
                "policy_procurement_target_ton_per_capita_reference": procurement_target,
                "facility_capacity_data_status": (
                    capacity_summary["capacity_data_status"]
                    if capacity_summary is not None
                    else str(capacity_bundle["status"])
                ),
                "facility_capacity_reporting_cycle_reference": (
                    capacity_summary.get("reporting_cycle", "") if capacity_summary is not None else ""
                ),
                "facility_row_count_reference": (
                    int(capacity_summary["facility_row_count"]) if capacity_summary is not None else ""
                ),
                "facility_count_reference": (
                    int(capacity_summary["unique_facility_count"]) if capacity_summary is not None else ""
                ),
                "facility_total_permitted_capacity_ton_per_year_reference": (
                    capacity_summary["total_permitted_capacity_ton_per_year"]
                    if capacity_summary is not None
                    else ""
                ),
                "facility_total_available_capacity_ton_per_year_reference": (
                    capacity_summary["total_available_capacity_ton_per_year"]
                    if capacity_summary is not None
                    else ""
                ),
                "organic_waste_recycling_capacity_needed_ton_per_year_reference": (
                    capacity_summary.get(
                        "total_organic_waste_recycling_capacity_needed_ton_per_year", ""
                    )
                    if capacity_summary is not None
                    else ""
                ),
                "edible_food_recovery_capacity_available_ton_per_year_reference": (
                    capacity_summary.get(
                        "total_edible_food_recovery_capacity_available_ton_per_year", ""
                    )
                    if capacity_summary is not None
                    else ""
                ),
                "edible_food_recovery_capacity_needed_ton_per_year_reference": (
                    capacity_summary.get(
                        "total_edible_food_recovery_capacity_needed_ton_per_year", ""
                    )
                    if capacity_summary is not None
                    else ""
                ),
                "organics_facility_inventory_status": facility_inventory_bundle["status"],
                "organics_facility_count_reference": organics_facility_count,
                "composting_facility_count_reference": composting_facility_count,
                "anaerobic_digestion_facility_count_reference": anaerobic_digestion_facility_count,
                "moisture_pct_reference": float(california_model_input["moisture_pct_reference"]),
                "organic_fraction_pct_reference": float(
                    california_model_input["organic_fraction_pct_reference"]
                ),
                "carbon_pct_reference": float(california_model_input["carbon_pct_reference"]),
                "nitrogen_pct_reference": float(california_model_input["nitrogen_pct_reference"]),
                "grid_electricity_emission_factor_kgco2e_per_kwh_reference": grid_emission_factor,
                "scenario_grid_electricity_emission_factor_kgco2e_per_kwh": (
                    grid_emission_factor * emission_multiplier
                ),
                "baseline_waste_treatment_pathway_reference": baseline_waste_treatment_factor[
                    "management_pathway"
                ],
                "baseline_waste_treatment_factor_unit_reference": baseline_waste_treatment_factor[
                    "factor_unit"
                ],
                "baseline_waste_treatment_emission_factor_kgco2e_per_short_ton_reference": (
                    baseline_waste_treatment_value
                ),
                "scenario_baseline_waste_treatment_emission_factor_kgco2e_per_short_ton": (
                    baseline_waste_treatment_value * emission_multiplier
                ),
                "baseline_waste_treatment_source_method_reference": baseline_waste_treatment_factor.get(
                    "baseline_source_method", ""
                ),
                "collection_rate_basis": "commercial_food_diversion_proxy_from_calrecycle",
                "wet_waste_source_bundle": california_model_input["source_bundle"],
                "wet_waste_reference_file": "data/processed/model_ready/california_food_waste_model_input.csv",
                "manure_reference_file": "data/raw/external-region-data/livestock_manure_statistics_template.csv",
                "energy_reference_file": "data/raw/external-region-data/regional_energy_prices_template.csv",
                "emission_reference_file": "data/raw/external-region-data/regional_emission_factors_template.csv",
                "facility_capacity_reference_file": capacity_bundle["reference_file"]
                or capacity_bundle["status_file"],
                "facility_inventory_reference_file": facility_inventory_bundle["reference_file"]
                or facility_inventory_bundle["status_file"],
                "policy_reference_file": policy_bundle["reference_file"],
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def write_text_outputs() -> list[Path]:
    RAW_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)

    california_reference = load_single_row_csv(CALIFORNIA_WET_WASTE_REFERENCE)
    california_model_input = load_single_row_csv(CALIFORNIA_WET_WASTE_MODEL_INPUT)
    source_lookup = load_source_lookup()
    grid_emission_factor = load_california_grid_emission_factor()
    baseline_waste_treatment_factor = load_baseline_waste_treatment_factor()

    outputs: list[Path] = []

    raw_templates = {
        "region_metadata_template.csv": build_region_metadata_template(
            california_reference, source_lookup
        ),
        "livestock_manure_statistics_template.csv": build_livestock_template(),
        "wet_waste_generation_template.csv": build_wet_waste_template(california_reference),
        "regional_energy_prices_template.csv": build_energy_price_template(),
        "regional_facility_capacity_template.csv": build_facility_capacity_template(),
        "regional_emission_factors_template.csv": build_emission_factor_template(
            grid_emission_factor, baseline_waste_treatment_factor
        ),
        "regional_policy_reference_template.csv": build_policy_reference_template(),
    }

    for file_name, frame in raw_templates.items():
        output_path = RAW_EXTERNAL_DIR / file_name
        frame.to_csv(output_path, index=False)
        outputs.append(output_path)

    interface_manifest_path = SCENARIO_DIR / "region_input_interface.csv"
    build_interface_manifest().to_csv(interface_manifest_path, index=False)
    outputs.append(interface_manifest_path)

    region_status_path = MODEL_READY_DIR / "region_input_status.csv"
    build_region_input_status(
        grid_emission_factor, baseline_waste_treatment_factor
    ).to_csv(region_status_path, index=False)
    outputs.append(region_status_path)

    region_scenario_placeholder_path = SCENARIO_DIR / "paper1_region_scenario_placeholder.csv"
    build_region_scenario_placeholder(
        california_model_input,
        grid_emission_factor,
        baseline_waste_treatment_factor,
    ).to_csv(region_scenario_placeholder_path, index=False)
    outputs.append(region_scenario_placeholder_path)

    manifest_path = MODEL_READY_DIR / "region_placeholder_interface_manifest.json"
    manifest_payload = {
        "dataset_name": "region_placeholder_interface",
        "purpose": "Fix the external regional data interface for Waste2Energy Paper 1.",
        "default_region_id": TARGET_REGION_ID,
        "default_region_name": california_reference["region_name"],
        "raw_template_files": [path.name for path in outputs if path.parent == RAW_EXTERNAL_DIR],
        "processed_interface_files": [
            "region_input_interface.csv",
            "region_input_status.csv",
            "paper1_region_scenario_placeholder.csv",
        ],
        "notes": [
            "California us_ca is now the default Paper 1 regional interface target.",
            "Wet-waste supply is connected to official CalRecycle-derived California food-waste references.",
            "Energy prices are connected to official California EIA electricity and natural gas references.",
            "Baseline waste-treatment emission factors are connected to an EPA WARM structured food-waste reference.",
            "Facility-capacity intake is configured through an official CalRecycle manual-export connector.",
            "Livestock/manure supply is connected to official California USDA counts plus repository moisture and planning coefficients.",
            "Policy reference is connected to official SB 1383 statute and CalRecycle implementation guidance.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    outputs.append(manifest_path)

    return outputs


def main() -> None:
    outputs = write_text_outputs()
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
