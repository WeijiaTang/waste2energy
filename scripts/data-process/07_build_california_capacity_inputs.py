from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_CA_DIR = ROOT / "data" / "raw" / "external-region-data" / "california"
WET_WASTE_DIR = RAW_CA_DIR / "wet_waste_supply"
SCENARIO_DIR = ROOT / "data" / "processed" / "scenario_inputs"
MODEL_READY_DIR = ROOT / "data" / "processed" / "model_ready"

CANONICAL_EXPORT_STEM = "calrecycle_capacity_planning_manual_export"
REGISTRATION_JSON = WET_WASTE_DIR / "calrecycle_capacity_planning_manual_export_registration.json"
STRUCTURED_OUTPUT = SCENARIO_DIR / "california_capacity_planning_structured.csv"
SUMMARY_OUTPUT = SCENARIO_DIR / "california_capacity_planning_summary.csv"
STATUS_OUTPUT = MODEL_READY_DIR / "california_capacity_planning_status.json"

COLUMN_CANDIDATES = {
    "jurisdiction_name": [
        "jurisdiction",
        "jurisdictionname",
        "county",
        "countyname",
        "city",
        "localjurisdiction",
        "planningentity",
        "entity",
    ],
    "facility_name": [
        "facility",
        "facilityname",
        "site",
        "sitename",
        "operation",
        "operationname",
        "facilityoperator",
    ],
    "activity_type": [
        "activity",
        "activitytype",
        "facilitytype",
        "processtype",
        "operationtype",
        "technology",
        "category",
    ],
    "material_scope": [
        "material",
        "materialtype",
        "materialcategory",
        "feedstock",
        "stream",
        "acceptedmaterial",
        "materialaccepted",
    ],
    "permitted_capacity_ton_per_year": [
        "permittedcapacity",
        "permittedcapacitytonsperyear",
        "permittedcapacitytpy",
        "capacitytonsperyear",
        "capacitytpy",
        "annualcapacity",
        "annualcapacitytons",
    ],
    "available_capacity_ton_per_year": [
        "availablecapacity",
        "availablecapacitytonsperyear",
        "availablecapacitytpy",
        "remainingcapacity",
        "remainingcapacitytonsperyear",
        "remainingcapacitytpy",
        "surpluscapacity",
        "surpluscapacitytpy",
    ],
    "used_capacity_ton_per_year": [
        "usedcapacity",
        "usedcapacitytonsperyear",
        "usedcapacitytpy",
        "utilizedcapacity",
        "currentthroughput",
        "currentthroughputtonsperyear",
    ],
    "operational_status": [
        "status",
        "operationalstatus",
        "facilitystatus",
    ],
}

COUNTY_CYCLE_SCHEMA = {
    "county_name": "County",
    "reporting_cycle": "Reporting cycle",
    "estimated_organic_waste_for_landfill_disposal_ton_per_year": "Estimated organic waste for landfill disposal (tons)",
    "organic_waste_recycling_capacity_available_ton_per_year": "Organic waste recycling capacity verifiably available (tons)",
    "organic_waste_recycling_capacity_needed_ton_per_year": "Needed organic waste recycling capacity (tons)",
    "estimated_edible_food_for_landfill_disposal_ton_per_year": "Estimated edible food for landfill disposal (tons)",
    "edible_food_recovery_capacity_available_ton_per_year": "Edible food recovery capacity verifiably available (tons)",
    "edible_food_recovery_capacity_needed_ton_per_year": "Needed edible food recovery capacity (tons)",
}


def utcnow_text() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def find_registered_export() -> Path | None:
    for suffix in [".xlsx", ".xls", ".csv"]:
        candidate = WET_WASTE_DIR / f"{CANONICAL_EXPORT_STEM}{suffix}"
        if candidate.exists():
            return candidate
    return None


def read_export(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        workbook = pd.ExcelFile(path)
        best_sheet = max(
            workbook.sheet_names,
            key=lambda sheet_name: pd.read_excel(path, sheet_name=sheet_name).shape[0],
        )
        frame = pd.read_excel(path, sheet_name=best_sheet)
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed:")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    return frame.reset_index(drop=True)


def match_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {normalize_name(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for candidate in candidates:
        for normalized_name, raw_name in normalized.items():
            if candidate in normalized_name or normalized_name in candidate:
                return raw_name
    return None


def to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace(r"[^0-9.\-]", "", regex=True)
    cleaned = cleaned.replace("", pd.NA)
    return pd.to_numeric(cleaned, errors="coerce")


def build_structured_capacity(frame: pd.DataFrame, export_file: Path, registered_at: str) -> pd.DataFrame:
    if all(column in frame.columns for column in COUNTY_CYCLE_SCHEMA.values()):
        structured = pd.DataFrame(
            {
                "region_id": "us_ca",
                "region_name": "California",
                "country": "United States",
                "facility_row_id": [f"california_capacity::{index + 1:04d}" for index in range(len(frame))],
                "jurisdiction_name": frame[COUNTY_CYCLE_SCHEMA["county_name"]].astype(str).str.strip(),
                "facility_name": (
                    frame[COUNTY_CYCLE_SCHEMA["county_name"]].astype(str).str.strip()
                    + "::"
                    + frame[COUNTY_CYCLE_SCHEMA["reporting_cycle"]].astype(str).str.strip()
                ),
                "activity_type": pd.Series(["county_capacity_planning_summary"] * len(frame)),
                "material_scope": pd.Series(
                    ["organic_waste_recycling_and_edible_food_recovery"] * len(frame)
                ),
                "operational_status": pd.Series(["reported_county_cycle"] * len(frame)),
                "reporting_cycle": frame[COUNTY_CYCLE_SCHEMA["reporting_cycle"]].astype(str).str.strip(),
                "estimated_organic_waste_for_landfill_disposal_ton_per_year": to_numeric(
                    frame[COUNTY_CYCLE_SCHEMA["estimated_organic_waste_for_landfill_disposal_ton_per_year"]]
                ),
                "organic_waste_recycling_capacity_available_ton_per_year": to_numeric(
                    frame[COUNTY_CYCLE_SCHEMA["organic_waste_recycling_capacity_available_ton_per_year"]]
                ),
                "organic_waste_recycling_capacity_needed_ton_per_year": to_numeric(
                    frame[COUNTY_CYCLE_SCHEMA["organic_waste_recycling_capacity_needed_ton_per_year"]]
                ),
                "estimated_edible_food_for_landfill_disposal_ton_per_year": to_numeric(
                    frame[COUNTY_CYCLE_SCHEMA["estimated_edible_food_for_landfill_disposal_ton_per_year"]]
                ),
                "edible_food_recovery_capacity_available_ton_per_year": to_numeric(
                    frame[COUNTY_CYCLE_SCHEMA["edible_food_recovery_capacity_available_ton_per_year"]]
                ),
                "edible_food_recovery_capacity_needed_ton_per_year": to_numeric(
                    frame[COUNTY_CYCLE_SCHEMA["edible_food_recovery_capacity_needed_ton_per_year"]]
                ),
                "source_organization": "California Department of Resources Recycling and Recovery",
                "source_url": "https://www2.calrecycle.ca.gov/LGCentral/datatools/reports/capacityplanning",
                "registered_export_file": export_file.name,
                "registered_at_utc": registered_at,
            }
        )
        structured["permitted_capacity_ton_per_year"] = (
            structured["organic_waste_recycling_capacity_available_ton_per_year"].fillna(0.0)
            + structured["organic_waste_recycling_capacity_needed_ton_per_year"].fillna(0.0)
        )
        structured["available_capacity_ton_per_year"] = structured[
            "organic_waste_recycling_capacity_available_ton_per_year"
        ]
        structured["used_capacity_ton_per_year"] = structured[
            "estimated_organic_waste_for_landfill_disposal_ton_per_year"
        ]
        structured["has_any_capacity_value"] = structured[
            [
                "estimated_organic_waste_for_landfill_disposal_ton_per_year",
                "organic_waste_recycling_capacity_available_ton_per_year",
                "organic_waste_recycling_capacity_needed_ton_per_year",
                "edible_food_recovery_capacity_available_ton_per_year",
                "edible_food_recovery_capacity_needed_ton_per_year",
            ]
        ].notna().any(axis=1)
        return structured

    mapping = {
        canonical: match_column(frame.columns.tolist(), candidates)
        for canonical, candidates in COLUMN_CANDIDATES.items()
    }
    if not any(
        mapping.get(column_name)
        for column_name in [
            "permitted_capacity_ton_per_year",
            "available_capacity_ton_per_year",
            "used_capacity_ton_per_year",
        ]
    ):
        raise RuntimeError(
            "Could not identify any capacity columns in the CalRecycle manual export. "
            f"Available columns: {list(frame.columns)}"
        )

    structured = pd.DataFrame(
        {
            "region_id": "us_ca",
            "region_name": "California",
            "country": "United States",
            "facility_row_id": [f"california_capacity::{index + 1:04d}" for index in range(len(frame))],
            "jurisdiction_name": (
                frame[mapping["jurisdiction_name"]].astype(str).str.strip()
                if mapping["jurisdiction_name"]
                else pd.Series([""] * len(frame))
            ),
            "facility_name": (
                frame[mapping["facility_name"]].astype(str).str.strip()
                if mapping["facility_name"]
                else pd.Series([f"facility_row_{index + 1:04d}" for index in range(len(frame))])
            ),
            "activity_type": (
                frame[mapping["activity_type"]].astype(str).str.strip()
                if mapping["activity_type"]
                else pd.Series([""] * len(frame))
            ),
            "material_scope": (
                frame[mapping["material_scope"]].astype(str).str.strip()
                if mapping["material_scope"]
                else pd.Series([""] * len(frame))
            ),
            "operational_status": (
                frame[mapping["operational_status"]].astype(str).str.strip()
                if mapping["operational_status"]
                else pd.Series([""] * len(frame))
            ),
            "permitted_capacity_ton_per_year": (
                to_numeric(frame[mapping["permitted_capacity_ton_per_year"]])
                if mapping["permitted_capacity_ton_per_year"]
                else pd.Series([pd.NA] * len(frame), dtype="float64")
            ),
            "available_capacity_ton_per_year": (
                to_numeric(frame[mapping["available_capacity_ton_per_year"]])
                if mapping["available_capacity_ton_per_year"]
                else pd.Series([pd.NA] * len(frame), dtype="float64")
            ),
            "used_capacity_ton_per_year": (
                to_numeric(frame[mapping["used_capacity_ton_per_year"]])
                if mapping["used_capacity_ton_per_year"]
                else pd.Series([pd.NA] * len(frame), dtype="float64")
            ),
            "source_organization": "California Department of Resources Recycling and Recovery",
            "source_url": "https://www2.calrecycle.ca.gov/LGCentral/datatools/reports/capacityplanning",
            "registered_export_file": export_file.name,
            "registered_at_utc": registered_at,
        }
    )
    structured["has_any_capacity_value"] = structured[
        [
            "permitted_capacity_ton_per_year",
            "available_capacity_ton_per_year",
            "used_capacity_ton_per_year",
        ]
    ].notna().any(axis=1)
    return structured


def build_summary(structured: pd.DataFrame, export_file: Path, registered_at: str) -> pd.DataFrame:
    if "reporting_cycle" in structured.columns:
        rows: list[dict[str, object]] = []
        for reporting_cycle, subset in structured.groupby("reporting_cycle", dropna=False):
            usable = subset[subset["has_any_capacity_value"]].copy()
            rows.append(
                {
                    "region_id": "us_ca",
                    "region_name": "California",
                    "country": "United States",
                    "analysis_year": datetime.now(UTC).year,
                    "capacity_data_status": "manual_export_processed",
                    "reporting_cycle": reporting_cycle,
                    "facility_row_count": int(len(subset)),
                    "facility_rows_with_capacity": int(len(usable)),
                    "unique_facility_count": int(
                        subset["facility_name"].replace("", pd.NA).dropna().nunique()
                    ),
                    "unique_jurisdiction_count": int(
                        subset["jurisdiction_name"].replace("", pd.NA).dropna().nunique()
                    ),
                    "total_permitted_capacity_ton_per_year": float(
                        usable["permitted_capacity_ton_per_year"].sum(min_count=1)
                    )
                    if usable["permitted_capacity_ton_per_year"].notna().any()
                    else pd.NA,
                    "total_available_capacity_ton_per_year": float(
                        usable["available_capacity_ton_per_year"].sum(min_count=1)
                    )
                    if usable["available_capacity_ton_per_year"].notna().any()
                    else pd.NA,
                    "total_used_capacity_ton_per_year": float(
                        usable["used_capacity_ton_per_year"].sum(min_count=1)
                    )
                    if usable["used_capacity_ton_per_year"].notna().any()
                    else pd.NA,
                    "total_organic_waste_recycling_capacity_needed_ton_per_year": float(
                        usable["organic_waste_recycling_capacity_needed_ton_per_year"].sum(min_count=1)
                    )
                    if usable["organic_waste_recycling_capacity_needed_ton_per_year"].notna().any()
                    else pd.NA,
                    "total_edible_food_recovery_capacity_available_ton_per_year": float(
                        usable["edible_food_recovery_capacity_available_ton_per_year"].sum(min_count=1)
                    )
                    if usable["edible_food_recovery_capacity_available_ton_per_year"].notna().any()
                    else pd.NA,
                    "total_edible_food_recovery_capacity_needed_ton_per_year": float(
                        usable["edible_food_recovery_capacity_needed_ton_per_year"].sum(min_count=1)
                    )
                    if usable["edible_food_recovery_capacity_needed_ton_per_year"].notna().any()
                    else pd.NA,
                    "activity_types_observed": "county_capacity_planning_summary",
                    "source_organization": "California Department of Resources Recycling and Recovery",
                    "source_url": "https://www2.calrecycle.ca.gov/LGCentral/datatools/reports/capacityplanning",
                    "registered_export_file": export_file.name,
                    "registered_at_utc": registered_at,
                    "notes": (
                        "Structured from a manually exported official CalRecycle county-level capacity-planning file. "
                        "Each row summarizes one county for one reporting cycle."
                    ),
                }
            )
        return pd.DataFrame(rows)

    usable = structured[structured["has_any_capacity_value"]].copy()
    unique_facilities = structured["facility_name"].replace("", pd.NA).dropna().nunique()
    unique_jurisdictions = structured["jurisdiction_name"].replace("", pd.NA).dropna().nunique()
    activity_types = sorted(
        {
            value
            for value in structured["activity_type"].dropna().astype(str).str.strip()
            if value
        }
    )
    summary = {
        "region_id": "us_ca",
        "region_name": "California",
        "country": "United States",
        "analysis_year": datetime.now(UTC).year,
        "capacity_data_status": "manual_export_processed",
        "reporting_cycle": "not_available",
        "facility_row_count": int(len(structured)),
        "facility_rows_with_capacity": int(len(usable)),
        "unique_facility_count": int(unique_facilities),
        "unique_jurisdiction_count": int(unique_jurisdictions),
        "total_permitted_capacity_ton_per_year": float(
            usable["permitted_capacity_ton_per_year"].sum(min_count=1)
        )
        if usable["permitted_capacity_ton_per_year"].notna().any()
        else pd.NA,
        "total_available_capacity_ton_per_year": float(
            usable["available_capacity_ton_per_year"].sum(min_count=1)
        )
        if usable["available_capacity_ton_per_year"].notna().any()
        else pd.NA,
        "total_used_capacity_ton_per_year": float(
            usable["used_capacity_ton_per_year"].sum(min_count=1)
        )
        if usable["used_capacity_ton_per_year"].notna().any()
        else pd.NA,
        "activity_types_observed": "; ".join(activity_types),
        "source_organization": "California Department of Resources Recycling and Recovery",
        "source_url": "https://www2.calrecycle.ca.gov/LGCentral/datatools/reports/capacityplanning",
        "registered_export_file": export_file.name,
        "registered_at_utc": registered_at,
        "notes": (
            "Structured from a manually exported official CalRecycle capacity-planning file. "
            "Review row-level fields before using facility-level results in manuscript tables."
        ),
    }
    return pd.DataFrame([summary])


def load_registration_timestamp() -> str:
    if REGISTRATION_JSON.exists():
        payload = json.loads(REGISTRATION_JSON.read_text(encoding="utf-8"))
        if payload:
            return str(payload[0].get("registered_at_utc", "")) or utcnow_text()
    return utcnow_text()


def write_status(payload: dict[str, object]) -> None:
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)

    export_file = find_registered_export()
    if export_file is None:
        payload = {
            "dataset_name": "california_capacity_planning_status",
            "status": "awaiting_manual_export",
            "source_organization": "California Department of Resources Recycling and Recovery",
            "source_url": "https://www2.calrecycle.ca.gov/LGCentral/datatools/reports/capacityplanning",
            "expected_raw_files": [
                f"data/raw/external-region-data/california/wet_waste_supply/{CANONICAL_EXPORT_STEM}.xlsx",
                f"data/raw/external-region-data/california/wet_waste_supply/{CANONICAL_EXPORT_STEM}.xls",
                f"data/raw/external-region-data/california/wet_waste_supply/{CANONICAL_EXPORT_STEM}.csv",
            ],
            "checked_at_utc": utcnow_text(),
            "notes": "No registered manual export was found yet. Use register_calrecycle_capacity_export.py first.",
        }
        write_status(payload)
        print("No registered CalRecycle capacity export found. Wrote pending status artifact.")
        print(f"Wrote {STATUS_OUTPUT}")
        return

    registered_at = load_registration_timestamp()
    raw_frame = read_export(export_file)
    structured = build_structured_capacity(raw_frame, export_file, registered_at)
    summary = build_summary(structured, export_file, registered_at)

    structured.to_csv(STRUCTURED_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    write_status(
        {
            "dataset_name": "california_capacity_planning_status",
            "status": "processed_summary_connected",
            "source_organization": "California Department of Resources Recycling and Recovery",
            "source_url": "https://www2.calrecycle.ca.gov/LGCentral/datatools/reports/capacityplanning",
            "raw_export_file": str(export_file.relative_to(ROOT)),
            "structured_output_file": str(STRUCTURED_OUTPUT.relative_to(ROOT)),
            "summary_output_file": str(SUMMARY_OUTPUT.relative_to(ROOT)),
            "checked_at_utc": utcnow_text(),
        }
    )

    print(f"Wrote {STRUCTURED_OUTPUT}")
    print(f"Wrote {SUMMARY_OUTPUT}")
    print(f"Wrote {STATUS_OUTPUT}")


if __name__ == "__main__":
    main()
