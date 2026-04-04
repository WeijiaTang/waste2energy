from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_CA_DIR = ROOT / "data" / "raw" / "external-region-data" / "california"
WET_WASTE_DIR = RAW_CA_DIR / "wet_waste_supply"
SCENARIO_DIR = ROOT / "data" / "processed" / "scenario_inputs"
MODEL_READY_DIR = ROOT / "data" / "processed" / "model_ready"

SITE_EXPORT = WET_WASTE_DIR / "calrecycle_swis_sites_export.xlsx"
ACTIVITY_EXPORT = WET_WASTE_DIR / "calrecycle_swis_site_activities_export.xlsx"
WASTE_EXPORT = WET_WASTE_DIR / "calrecycle_swis_site_wastes_export.xlsx"

INVENTORY_OUTPUT = SCENARIO_DIR / "california_organics_facility_inventory.csv"
SUMMARY_OUTPUT = SCENARIO_DIR / "california_organics_facility_summary.csv"
STATUS_OUTPUT = MODEL_READY_DIR / "california_organics_facility_inventory_status.json"

ORGANIC_WASTE_TYPES = [
    "Food Wastes",
    "Green Materials",
    "Manure",
    "Sludge (BioSolids)",
    "Digestate",
    "Wood Waste",
    "Treated Wood Waste",
]
FINAL_TREATMENT_GROUPS = {"composting", "anaerobic_digestion"}


def utcnow_text() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def ensure_inputs() -> None:
    missing = [path for path in [SITE_EXPORT, ACTIVITY_EXPORT, WASTE_EXPORT] if not path.exists()]
    if missing:
        joined = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        raise FileNotFoundError(
            "Missing SWIS raw exports. Run "
            "`python scripts/data-crawl/download_official_sources.py --source-id "
            "california_swis_site_export_xlsx --source-id california_swis_site_activity_export_xlsx "
            f"--source-id california_swis_site_waste_export_xlsx` first. Missing: {joined}"
        )


def read_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def annualize_ton_unit(value: object, unit: object) -> float | pd.NA:
    if pd.isna(value) or pd.isna(unit):
        return pd.NA
    unit_text = str(unit).strip().lower()
    numeric_value = float(value)
    factors = {
        "tons per day": 365.0,
        "tons per week": 52.0,
        "tons per month": 12.0,
        "tons per year": 1.0,
    }
    factor = factors.get(unit_text)
    if factor is None:
        return pd.NA
    return numeric_value * factor


def classify_technology_group(activity_category: object, activity_name: object) -> str:
    category = str(activity_category).strip().lower()
    activity = str(activity_name).strip().lower()
    if category == "composting":
        return "composting"
    if category == "in-vessel digestion":
        return "anaerobic_digestion"
    if category == "transfer/processing":
        return "transfer_processing"
    if category == "disposal":
        return "disposal"
    if "digestion" in activity:
        return "anaerobic_digestion"
    if "compost" in activity:
        return "composting"
    return "other"


def build_waste_flags() -> pd.DataFrame:
    waste = read_excel(WASTE_EXPORT).copy()
    waste = waste[waste["Waste Type"].isin(ORGANIC_WASTE_TYPES)].copy()
    if waste.empty:
        raise RuntimeError("No organics rows found in SWIS site waste export")

    waste["accepts_food_wastes"] = waste["Waste Type"].eq("Food Wastes")
    waste["accepts_green_materials"] = waste["Waste Type"].eq("Green Materials")
    waste["accepts_manure"] = waste["Waste Type"].eq("Manure")
    waste["accepts_biosolids"] = waste["Waste Type"].eq("Sludge (BioSolids)")
    waste["accepts_digestate"] = waste["Waste Type"].eq("Digestate")
    waste["accepts_wood_waste"] = waste["Waste Type"].isin(["Wood Waste", "Treated Wood Waste"])

    grouped = (
        waste.groupby(["Site ID", "SWIS Number", "Site Name", "Activity"], as_index=False)
        .agg(
            accepted_waste_types=("Waste Type", lambda values: "; ".join(sorted(set(values)))),
            accepts_food_wastes=("accepts_food_wastes", "max"),
            accepts_green_materials=("accepts_green_materials", "max"),
            accepts_manure=("accepts_manure", "max"),
            accepts_biosolids=("accepts_biosolids", "max"),
            accepts_digestate=("accepts_digestate", "max"),
            accepts_wood_waste=("accepts_wood_waste", "max"),
        )
    )
    return grouped


def build_inventory() -> pd.DataFrame:
    site = read_excel(SITE_EXPORT).copy()
    activity = read_excel(ACTIVITY_EXPORT).copy()
    waste_flags = build_waste_flags()

    activity = activity.rename(columns={"SWIS Number ": "SWIS Number"})
    site = site.rename(columns={"SiteID": "Site ID", "SWIS Number ": "SWIS Number"})

    inventory = activity.merge(
        waste_flags,
        on=["Site ID", "SWIS Number", "Site Name", "Activity"],
        how="left",
    )
    inventory = inventory.merge(
        site[["Site ID", "SWIS Number", "Site Name", "Latitude", "Longitude", "County"]],
        on=["Site ID", "SWIS Number", "Site Name"],
        how="left",
        suffixes=("", "_site"),
    )

    inventory["accepted_waste_types"] = inventory["accepted_waste_types"].fillna("")
    for column in [
        "accepts_food_wastes",
        "accepts_green_materials",
        "accepts_manure",
        "accepts_biosolids",
        "accepts_digestate",
        "accepts_wood_waste",
    ]:
        inventory[column] = inventory[column].astype("boolean").fillna(False).astype(bool)

    inventory["technology_group"] = inventory.apply(
        lambda row: classify_technology_group(row.get("Category"), row.get("Activity")),
        axis=1,
    )
    inventory["is_active_activity"] = inventory["OperationalStatus"].astype(str).str.contains(
        "Active", case=False, na=False
    )
    inventory["is_organics_relevant"] = (
        inventory["accepted_waste_types"].ne("")
        | inventory["technology_group"].isin(
            ["composting", "anaerobic_digestion", "transfer_processing", "disposal"]
        )
    )
    inventory = inventory[inventory["is_active_activity"] & inventory["is_organics_relevant"]].copy()

    inventory["annualized_throughput_ton_per_year"] = inventory.apply(
        lambda row: annualize_ton_unit(row.get("Throughput"), row.get("ThroughputUnits")),
        axis=1,
    )
    inventory["annualized_capacity_ton_per_year"] = inventory.apply(
        lambda row: annualize_ton_unit(row.get("Capacity"), row.get("CapacityUnits")),
        axis=1,
    )
    inventory["annualized_ton_basis_available"] = inventory[
        ["annualized_throughput_ton_per_year", "annualized_capacity_ton_per_year"]
    ].notna().any(axis=1)
    inventory["final_treatment_relevance"] = inventory["technology_group"].isin(FINAL_TREATMENT_GROUPS)

    inventory = inventory.rename(
        columns={
            "Category": "activity_category",
            "Activity Classification": "activity_classification",
            "OperationalStatus": "activity_operational_status",
            "RegulatoryStatus": "activity_regulatory_status",
            "Site Operational Status": "site_operational_status",
            "Site Regulatory Status": "site_regulatory_status",
            "Throughput": "reported_throughput_value",
            "ThroughputUnits": "reported_throughput_unit",
            "Capacity": "reported_capacity_value",
            "CapacityUnits": "reported_capacity_unit",
        }
    )
    inventory["region_id"] = "us_ca"
    inventory["region_name"] = "California"
    inventory["country"] = "United States"
    inventory["source_organization"] = "California Department of Resources Recycling and Recovery"
    inventory["source_url"] = "https://www2.calrecycle.ca.gov/SolidWaste/Site/DataExport"
    inventory["raw_site_file_name"] = SITE_EXPORT.name
    inventory["raw_activity_file_name"] = ACTIVITY_EXPORT.name
    inventory["raw_waste_file_name"] = WASTE_EXPORT.name

    ordered_columns = [
        "region_id",
        "region_name",
        "country",
        "Site ID",
        "SWIS Number",
        "Site Name",
        "County",
        "Latitude",
        "Longitude",
        "Activity",
        "activity_category",
        "technology_group",
        "activity_classification",
        "activity_operational_status",
        "activity_regulatory_status",
        "site_operational_status",
        "site_regulatory_status",
        "accepted_waste_types",
        "accepts_food_wastes",
        "accepts_green_materials",
        "accepts_manure",
        "accepts_biosolids",
        "accepts_digestate",
        "accepts_wood_waste",
        "reported_throughput_value",
        "reported_throughput_unit",
        "annualized_throughput_ton_per_year",
        "reported_capacity_value",
        "reported_capacity_unit",
        "annualized_capacity_ton_per_year",
        "annualized_ton_basis_available",
        "final_treatment_relevance",
        "source_organization",
        "source_url",
        "raw_site_file_name",
        "raw_activity_file_name",
        "raw_waste_file_name",
    ]
    return inventory[ordered_columns].sort_values(
        ["technology_group", "County", "Site Name", "Activity"]
    ).reset_index(drop=True)


def build_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for technology_group, subset in inventory.groupby("technology_group", dropna=False):
        rows.append(
            {
                "region_id": "us_ca",
                "technology_group": technology_group,
                "activity_row_count": int(len(subset)),
                "unique_site_count": int(subset["SWIS Number"].dropna().nunique()),
                "food_waste_site_count": int(
                    subset.loc[subset["accepts_food_wastes"], "SWIS Number"].dropna().nunique()
                ),
                "rows_with_annualized_throughput_ton_per_year": int(
                    subset["annualized_throughput_ton_per_year"].notna().sum()
                ),
                "rows_with_annualized_capacity_ton_per_year": int(
                    subset["annualized_capacity_ton_per_year"].notna().sum()
                ),
                "total_annualized_throughput_ton_per_year": float(
                    subset["annualized_throughput_ton_per_year"].sum(min_count=1)
                )
                if subset["annualized_throughput_ton_per_year"].notna().any()
                else pd.NA,
                "total_annualized_capacity_ton_per_year": float(
                    subset["annualized_capacity_ton_per_year"].sum(min_count=1)
                )
                if subset["annualized_capacity_ton_per_year"].notna().any()
                else pd.NA,
                "source_url": "https://www2.calrecycle.ca.gov/SolidWaste/Site/DataExport",
                "notes": (
                    "Annualized ton-based fields convert only reported ton/day, ton/week, ton/month, and "
                    "ton/year units. Cubic-yard-only rows remain in inventory but are excluded from ton-based sums."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("technology_group").reset_index(drop=True)


def main() -> None:
    ensure_inputs()
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory()
    summary = build_summary(inventory)

    inventory.to_csv(INVENTORY_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    STATUS_OUTPUT.write_text(
        json.dumps(
            {
                "dataset_name": "california_organics_facility_inventory",
                "status": "official_reference_connected",
                "checked_at_utc": utcnow_text(),
                "raw_files": [
                    str(SITE_EXPORT.relative_to(ROOT)),
                    str(ACTIVITY_EXPORT.relative_to(ROOT)),
                    str(WASTE_EXPORT.relative_to(ROOT)),
                ],
                "inventory_output_file": str(INVENTORY_OUTPUT.relative_to(ROOT)),
                "summary_output_file": str(SUMMARY_OUTPUT.relative_to(ROOT)),
                "notes": [
                    "Inventory is built from official CalRecycle SWIS site, activity, and waste exports.",
                    "Technology-group aggregation is based on SWIS activity categories and accepted waste types.",
                    "Ton-based annualization excludes cubic-yard-only rows from throughput and capacity totals.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {INVENTORY_OUTPUT}")
    print(f"Wrote {SUMMARY_OUTPUT}")
    print(f"Wrote {STATUS_OUTPUT}")


if __name__ == "__main__":
    main()
