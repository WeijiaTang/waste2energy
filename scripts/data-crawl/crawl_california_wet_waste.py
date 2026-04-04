from __future__ import annotations

import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
RAW_CA_DIR = ROOT / "data" / "raw" / "external-region-data" / "california"
WET_WASTE_DIR = RAW_CA_DIR / "wet_waste_supply"
MANIFEST_JSON = WET_WASTE_DIR / "calrecycle_countywide_manifest.json"
MANIFEST_CSV = WET_WASTE_DIR / "calrecycle_countywide_manifest.csv"

BASE_URL = "https://www2.calrecycle.ca.gov"
HEADERS = {"User-Agent": "Waste2EnergyResearchBot/0.1 (+https://github.com/openai)"}

RESIDENTIAL_STUDY_ID = 103
BUSINESS_STUDY_ID = 104
TARGET_BUSINESS_MATERIALS = {
    40: "food",
}


@dataclass(frozen=True)
class CountyRecord:
    county_id: int
    county_name: str
    countywide_local_government_id: int
    countywide_local_government_name: str


def fetch_text(url: str, *, params: dict[str, object] | None = None) -> str:
    response = requests.get(url, params=params, headers=HEADERS, timeout=120)
    response.raise_for_status()
    return response.text


def fetch_json(url: str, *, params: dict[str, object] | None = None) -> object:
    response = requests.get(url, params=params, headers=HEADERS, timeout=120)
    response.raise_for_status()
    return response.json()


def parse_counties() -> list[tuple[int, str]]:
    page = fetch_text(f"{BASE_URL}/WasteCharacterization/ResidentialStreams")
    pattern = re.compile(r'<option value="(\d+)">([^<]+)</option>')
    counties: list[tuple[int, str]] = []
    for value, name in pattern.findall(page):
        county_id = int(value)
        county_name = name.strip()
        if county_id > 0:
            counties.append((county_id, county_name))
    return counties


def lookup_countywide_local_government(county_id: int, county_name: str) -> CountyRecord:
    payload = fetch_json(
        f"{BASE_URL}/WasteCharacterization/_LocalGovernmentsByCounty",
        params={"countyID": county_id},
    )
    for item in payload:
        name = str(item["Name"])
        if "(Countywide)" in name:
            return CountyRecord(
                county_id=county_id,
                county_name=county_name,
                countywide_local_government_id=int(item["ID"]),
                countywide_local_government_name=name,
            )
    raise RuntimeError(f"Could not find countywide local government for county_id={county_id} ({county_name})")


def county_query_params(record: CountyRecord) -> dict[str, object]:
    return {
        "countyID": record.county_id,
        "localGovernmentIDList": record.countywide_local_government_id,
        "localGovernmentIDListString": str(record.countywide_local_government_id),
    }


def fetch_residential_population(record: CountyRecord) -> dict[str, object]:
    params = {"studyID": RESIDENTIAL_STUDY_ID, **county_query_params(record)}
    payload = fetch_json(f"{BASE_URL}/WasteCharacterization/_ResidentialPopulationsByStudy", params=params)
    return {
        "county_id": record.county_id,
        "county_name": record.county_name,
        "countywide_local_government_id": record.countywide_local_government_id,
        "countywide_local_government_name": record.countywide_local_government_name,
        **payload,
    }


def fetch_residential_streams(record: CountyRecord) -> list[dict[str, object]]:
    params = {"studyID": RESIDENTIAL_STUDY_ID, **county_query_params(record)}
    payload = fetch_json(f"{BASE_URL}/WasteCharacterization/_ResidentialStreamsGridData", params=params)
    rows = []
    for item in payload["Data"]:
        rows.append(
            {
                "county_id": record.county_id,
                "county_name": record.county_name,
                "countywide_local_government_id": record.countywide_local_government_id,
                "countywide_local_government_name": record.countywide_local_government_name,
                **item,
            }
        )
    return rows


def fetch_business_material_streams(record: CountyRecord) -> list[dict[str, object]]:
    all_rows: list[dict[str, object]] = []
    for material_type_id, material_focus in TARGET_BUSINESS_MATERIALS.items():
        params = {
            "studyID": BUSINESS_STUDY_ID,
            "materialTypeID": material_type_id,
            **county_query_params(record),
        }
        payload = fetch_json(f"{BASE_URL}/WasteCharacterization/_BusinessGroupStreamsGridData", params=params)
        for item in payload["Data"]:
            all_rows.append(
                {
                    "county_id": record.county_id,
                    "county_name": record.county_name,
                    "countywide_local_government_id": record.countywide_local_government_id,
                    "countywide_local_government_name": record.countywide_local_government_name,
                    "material_focus": material_focus,
                    "material_type_id_requested": material_type_id,
                    **item,
                }
            )
    return all_rows


def collect_county_payload(record: CountyRecord) -> dict[str, object]:
    return {
        "residential_streams": fetch_residential_streams(record),
        "business_material_streams": fetch_business_material_streams(record),
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_manifest(records: list[dict[str, str]]) -> None:
    write_json(MANIFEST_JSON, records)
    fieldnames = [
        "dataset_id",
        "source_organization",
        "source_url_template",
        "study_id",
        "output_file",
        "downloaded_at_utc",
        "status",
        "record_count",
        "notes",
    ]
    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    WET_WASTE_DIR.mkdir(parents=True, exist_ok=True)

    counties = [lookup_countywide_local_government(county_id, county_name) for county_id, county_name in parse_counties()]

    residential_streams: list[dict[str, object]] = []
    business_material_streams: list[dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        payloads = list(executor.map(collect_county_payload, counties))

    for payload in payloads:
        residential_streams.extend(payload["residential_streams"])
        business_material_streams.extend(payload["business_material_streams"])

    county_reference = [record.__dict__ for record in counties]

    outputs = {
        "calrecycle_county_reference.json": county_reference,
        "calrecycle_residential_streams_countywide.json": residential_streams,
        "calrecycle_business_food_streams_countywide.json": business_material_streams,
    }

    records: list[dict[str, str]] = []
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    for filename, payload in outputs.items():
        path = WET_WASTE_DIR / filename
        write_json(path, payload)
        dataset_id = filename.removesuffix(".json")
        if "residential_streams" in filename:
            source_url_template = f"{BASE_URL}/WasteCharacterization/_ResidentialStreamsGridData"
            study_id = str(RESIDENTIAL_STUDY_ID)
            notes = "Countywide residential material streams using the countywide local-government selection."
        elif "business_food_streams" in filename:
            source_url_template = f"{BASE_URL}/WasteCharacterization/_BusinessGroupStreamsGridData"
            study_id = str(BUSINESS_STUDY_ID)
            notes = "Countywide business-group food streams using materialTypeID=40."
        else:
            source_url_template = f"{BASE_URL}/WasteCharacterization/_LocalGovernmentsByCounty"
            study_id = ""
            notes = "County and countywide jurisdiction lookup table derived from the public WasteCharacterization site."

        records.append(
            {
                "dataset_id": dataset_id,
                "source_organization": "California Department of Resources Recycling and Recovery",
                "source_url_template": source_url_template,
                "study_id": study_id,
                "output_file": str(path.relative_to(ROOT)),
                "downloaded_at_utc": timestamp,
                "status": "downloaded",
                "record_count": str(len(payload)),
                "notes": notes,
            }
        )

    write_manifest(records)

    print(f"Wrote {MANIFEST_JSON}")
    print(f"Wrote {MANIFEST_CSV}")
    for record in records:
        print(f"{record['dataset_id']}: {record['record_count']} rows -> {record['output_file']}")


if __name__ == "__main__":
    main()
