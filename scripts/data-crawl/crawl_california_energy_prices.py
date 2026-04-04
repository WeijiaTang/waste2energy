from __future__ import annotations

import csv
import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
RAW_CA_DIR = ROOT / "data" / "raw" / "external-region-data" / "california"
ENERGY_DIR = RAW_CA_DIR / "energy_prices"

ELECTRICITY_HTML = ENERGY_DIR / "eia_california_electricity_profile_2024.html"
ELECTRICITY_SUMMARY_CSV = ENERGY_DIR / "eia_california_electricity_profile_summary_2024.csv"
NATURAL_GAS_HTML = ENERGY_DIR / "eia_california_natural_gas_industrial_price_history.html"
NATURAL_GAS_XLS = ENERGY_DIR / "eia_california_natural_gas_industrial_price_history.xls"
NATURAL_GAS_HISTORY_CSV = ENERGY_DIR / "eia_california_natural_gas_industrial_price_annual.csv"
REFERENCE_JSON = ENERGY_DIR / "california_energy_price_reference.json"
REFERENCE_CSV = ENERGY_DIR / "california_energy_price_reference.csv"
MANIFEST_JSON = ENERGY_DIR / "california_energy_price_manifest.json"
MANIFEST_CSV = ENERGY_DIR / "california_energy_price_manifest.csv"

USER_AGENT = "Waste2EnergyResearchBot/0.1 (+https://github.com/openai)"
HEADERS = {"User-Agent": USER_AGENT}

ELECTRICITY_PROFILE_URL = "https://www.eia.gov/electricity/state/california/"
NATURAL_GAS_HISTORY_URL = "https://www.eia.gov/dnav/ng/hist/n3035ca3A.htm"
NATURAL_GAS_XLS_URL = "https://www.eia.gov/dnav/ng/hist_xls/N3035CA3a.xls"


def fetch_text(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return response.text


def fetch_bytes(url: str) -> bytes:
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    return response.content


def parse_electricity_profile(html: str) -> tuple[int, pd.DataFrame, dict[str, object]]:
    summary = pd.read_html(io.StringIO(html))[0].copy()
    year_match = re.search(r"California Electricity Profile\s+(\d{4})", html, re.IGNORECASE)
    analysis_year = int(year_match.group(1)) if year_match else datetime.now().year
    summary["analysis_year"] = analysis_year

    target = summary[summary["Item"].astype(str).str.contains("Average retail price", regex=False)].copy()
    if target.empty:
        raise RuntimeError("Could not locate Average retail price row on the California electricity profile page")

    reference_row = {
        "region_id": "us_ca",
        "region_name": "California",
        "country": "United States",
        "energy_carrier": "electricity",
        "market_segment": "all_retail",
        "price_case": "baseline",
        "analysis_year": analysis_year,
        "price_value": float(target.iloc[0]["Value"]),
        "price_unit": "cents_per_kWh",
        "source_organization": "U.S. Energy Information Administration",
        "source_url": ELECTRICITY_PROFILE_URL,
        "landing_page_url": ELECTRICITY_PROFILE_URL,
        "raw_file_name": ELECTRICITY_HTML.name,
        "notes": "Parsed from the official California electricity profile summary page average retail price row.",
    }
    return analysis_year, summary, reference_row


def expand_natural_gas_history(html: str) -> pd.DataFrame:
    history_table = pd.read_html(io.StringIO(html))[4].copy()
    records: list[dict[str, object]] = []
    for _, row in history_table.iterrows():
        decade_label = str(row["Decade"]).strip()
        match = re.search(r"(\d{4})", decade_label)
        if not match:
            continue
        decade_start = int(match.group(1))
        for offset in range(10):
            column_name = f"Year-{offset}"
            if column_name not in row.index:
                continue
            raw_value = row[column_name]
            if pd.isna(raw_value) or str(raw_value).strip() == "":
                continue
            records.append(
                {
                    "region_id": "us_ca",
                    "region_name": "California",
                    "country": "United States",
                    "energy_carrier": "natural_gas",
                    "market_segment": "industrial",
                    "analysis_year": decade_start + offset,
                    "price_value": float(raw_value),
                    "price_unit": "USD_per_thousand_cubic_feet",
                    "source_organization": "U.S. Energy Information Administration",
                    "source_url": NATURAL_GAS_HISTORY_URL,
                    "landing_page_url": NATURAL_GAS_HISTORY_URL,
                    "raw_file_name": NATURAL_GAS_XLS.name,
                    "notes": "Parsed from the official EIA annual California industrial natural gas price history table.",
                }
            )
    frame = pd.DataFrame(records).sort_values("analysis_year").reset_index(drop=True)
    return frame


def write_manifest(records: list[dict[str, str]]) -> None:
    MANIFEST_JSON.write_text(json.dumps(records, indent=2), encoding="utf-8")
    fieldnames = [
        "dataset_id",
        "source_url",
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
    ENERGY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()

    electricity_html = fetch_text(ELECTRICITY_PROFILE_URL)
    ELECTRICITY_HTML.write_text(electricity_html, encoding="utf-8")
    electricity_year, electricity_summary, electricity_reference = parse_electricity_profile(electricity_html)
    electricity_summary.to_csv(ELECTRICITY_SUMMARY_CSV, index=False)

    natural_gas_html = fetch_text(NATURAL_GAS_HISTORY_URL)
    NATURAL_GAS_HTML.write_text(natural_gas_html, encoding="utf-8")
    NATURAL_GAS_XLS.write_bytes(fetch_bytes(NATURAL_GAS_XLS_URL))
    natural_gas_history = expand_natural_gas_history(natural_gas_html)
    natural_gas_history.to_csv(NATURAL_GAS_HISTORY_CSV, index=False)

    natural_gas_reference_frame = natural_gas_history[natural_gas_history["analysis_year"] == electricity_year].copy()
    if natural_gas_reference_frame.empty:
        raise RuntimeError(
            f"No California natural gas annual value found for {electricity_year} in EIA history table"
        )

    natural_gas_reference = natural_gas_reference_frame.iloc[0].to_dict()
    natural_gas_reference["price_case"] = "baseline"

    reference_frame = pd.DataFrame([electricity_reference, natural_gas_reference])
    REFERENCE_JSON.write_text(
        json.dumps(reference_frame.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    reference_frame.to_csv(REFERENCE_CSV, index=False)

    manifest_records = [
        {
            "dataset_id": "eia_california_electricity_profile_snapshot",
            "source_url": ELECTRICITY_PROFILE_URL,
            "output_file": str(ELECTRICITY_HTML.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "downloaded",
            "record_count": str(len(electricity_summary)),
            "notes": "Official California electricity profile HTML snapshot.",
        },
        {
            "dataset_id": "eia_california_electricity_profile_summary",
            "source_url": ELECTRICITY_PROFILE_URL,
            "output_file": str(ELECTRICITY_SUMMARY_CSV.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "generated",
            "record_count": str(len(electricity_summary)),
            "notes": "Structured summary table extracted from the California electricity profile page.",
        },
        {
            "dataset_id": "eia_california_natural_gas_history_snapshot",
            "source_url": NATURAL_GAS_HISTORY_URL,
            "output_file": str(NATURAL_GAS_HTML.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "downloaded",
            "record_count": str(len(natural_gas_history)),
            "notes": "Official California natural gas industrial price history HTML snapshot.",
        },
        {
            "dataset_id": "eia_california_natural_gas_history_xls",
            "source_url": NATURAL_GAS_XLS_URL,
            "output_file": str(NATURAL_GAS_XLS.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "downloaded",
            "record_count": str(len(natural_gas_history)),
            "notes": "Official California natural gas industrial price history XLS file.",
        },
        {
            "dataset_id": "eia_california_natural_gas_history_structured",
            "source_url": NATURAL_GAS_HISTORY_URL,
            "output_file": str(NATURAL_GAS_HISTORY_CSV.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "generated",
            "record_count": str(len(natural_gas_history)),
            "notes": "Structured annual California industrial natural gas price series extracted from EIA history page.",
        },
        {
            "dataset_id": "eia_california_energy_price_reference",
            "source_url": ELECTRICITY_PROFILE_URL,
            "output_file": str(REFERENCE_CSV.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "generated",
            "record_count": str(len(reference_frame)),
            "notes": f"Combined California electricity and natural gas baseline price references aligned to {electricity_year}.",
        },
    ]
    write_manifest(manifest_records)

    print(f"Wrote {ELECTRICITY_HTML}")
    print(f"Wrote {ELECTRICITY_SUMMARY_CSV}")
    print(f"Wrote {NATURAL_GAS_HTML}")
    print(f"Wrote {NATURAL_GAS_XLS}")
    print(f"Wrote {NATURAL_GAS_HISTORY_CSV}")
    print(f"Wrote {REFERENCE_JSON}")
    print(f"Wrote {REFERENCE_CSV}")
    print(f"Wrote {MANIFEST_JSON}")
    print(f"Wrote {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
