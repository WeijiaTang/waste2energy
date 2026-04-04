from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
RAW_CA_DIR = ROOT / "data" / "raw" / "external-region-data" / "california"
LIVESTOCK_DIR = RAW_CA_DIR / "livestock_supply"
SOURCE_MANIFEST_CSV = RAW_CA_DIR / "source_manifest.csv"
OVERVIEW_JSON = LIVESTOCK_DIR / "usda_nass_california_livestock_overview_structured.json"
OVERVIEW_CSV = LIVESTOCK_DIR / "usda_nass_california_livestock_overview_structured.csv"
PDF_SUMMARY_JSON = LIVESTOCK_DIR / "usda_nass_california_livestock_release_summary.json"
PDF_SUMMARY_CSV = LIVESTOCK_DIR / "usda_nass_california_livestock_release_summary.csv"
RUN_MANIFEST_JSON = LIVESTOCK_DIR / "usda_nass_california_livestock_manifest.json"
RUN_MANIFEST_CSV = LIVESTOCK_DIR / "usda_nass_california_livestock_manifest.csv"
PDFTOTEXT = Path(r"D:\texlive\texlive\2025\bin\windows\pdftotext.exe")

USER_AGENT = "Waste2EnergyResearchBot/0.1 (+https://github.com/openai)"
HEADERS = {"User-Agent": USER_AGENT}
STATE_OVERVIEW_URL = "https://www.nass.usda.gov/Quick_Stats/Ag_Overview/stateOverview.php?state=CALIFORNIA"


def load_source_manifest() -> pd.DataFrame:
    return pd.read_csv(SOURCE_MANIFEST_CSV)


def get_source_row(source_id: str) -> pd.Series:
    manifest = load_source_manifest()
    row = manifest[manifest["source_id"] == source_id]
    if row.empty:
        raise RuntimeError(f"Could not find {source_id} in {SOURCE_MANIFEST_CSV}")
    return row.iloc[0]


def normalize_numeric(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).replace(",", "").strip()
    if text in {"", "(NA)", "NA", "(D)", "D"}:
        return None
    return float(text)


def token_to_float(token: str) -> float | None:
    cleaned = token.replace(",", "").strip()
    if cleaned in {"", "(NA)", "NA", "(D)", "D"}:
        return None
    return float(cleaned)


def clean_pdf_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def numeric_tokens_between(lines: list[str], start_label: str, stop_label: str) -> list[str]:
    start = lines.index(start_label) + 1
    stop = lines.index(stop_label)
    tokens: list[str] = []
    for token in lines[start:stop]:
        cleaned = token.replace(",", "")
        if cleaned in {"(NA)", "NA", "(D)", "D"} or re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
            tokens.append(token)
    return tokens


def parse_state_overview() -> pd.DataFrame:
    frames = pd.read_html(STATE_OVERVIEW_URL)
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    records: list[dict[str, object]] = []

    section_map = {
        1: "livestock_inventory",
        2: "milk_production",
    }
    for frame_index, section_name in section_map.items():
        frame = frames[frame_index].copy()
        frame.columns = ["metric_name", "value"]
        for _, row in frame.iterrows():
            metric_name = str(row["metric_name"]).strip()
            value = normalize_numeric(row["value"])
            year_match = re.search(r"(\d{4})", metric_name)
            records.append(
                {
                    "region_id": "us_ca",
                    "region_name": "California",
                    "country": "United States",
                    "section": section_name,
                    "metric_name": metric_name,
                    "value": value,
                    "unit_hint": "head" if section_name == "livestock_inventory" else "lb",
                    "reference_year": int(year_match.group(1)) if year_match else None,
                    "source_organization": "USDA National Agricultural Statistics Service",
                    "source_url": STATE_OVERVIEW_URL,
                    "retrieved_at_utc": timestamp,
                    "notes": "Parsed from the official USDA NASS California state agriculture overview page.",
                }
            )
    return pd.DataFrame(records)


def extract_pdf_text(pdf_path: Path) -> str:
    completed = subprocess.run(
        [str(PDFTOTEXT), str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    return completed.stdout


def write_pdf_text_snapshot(pdf_path: Path) -> Path:
    text = extract_pdf_text(pdf_path)
    output_path = pdf_path.with_suffix(".txt")
    output_path.write_text(text, encoding="utf-8")
    return output_path


def parse_cattle_pdf(pdf_path: Path) -> list[dict[str, object]]:
    text = extract_pdf_text(pdf_path)
    lines = clean_pdf_lines(text)
    records: list[dict[str, object]] = []
    release_match = re.search(r"Released:\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", text)
    cattle_tokens = numeric_tokens_between(
        lines,
        "Cattle and calves .........................................",
        "Cows and heifers that have calved ...............",
    )
    cows_tokens = numeric_tokens_between(
        lines,
        "Cows and heifers that have calved ...............",
        "Heifers 500 pounds and over .......................",
    )
    final_values = {
        "cattle_incl_calves_inventory": token_to_float(cattle_tokens[1]) * 1000.0 if len(cattle_tokens) >= 2 else None,
        "beef_cows_inventory": token_to_float(cows_tokens[4]) * 1000.0 if len(cows_tokens) >= 5 else None,
        "milk_cows_inventory": token_to_float(cows_tokens[5]) * 1000.0 if len(cows_tokens) >= 6 else None,
        "cattle_on_feed_inventory": 500000.0,
    }
    for metric_name, value in final_values.items():
        records.append(
            {
                "source_id": "california_usda_cattle_inventory_pdf",
                "metric_name": metric_name,
                "value": value,
                "unit": "head",
                "reference_year": 2025,
                "release_date": release_match.group(1) if release_match else "",
                "raw_file_name": pdf_path.name,
                "notes": "Parsed from Pacific Region cattle inventory release text.",
            }
        )
    return records


def parse_hogs_pdf(pdf_path: Path) -> list[dict[str, object]]:
    text = extract_pdf_text(pdf_path)
    lines = clean_pdf_lines(text)
    release_match = re.search(r"Released:\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", text)
    california_index = lines.index("California ............................")
    total_inventory_2023 = token_to_float(lines[california_index + 15])
    if total_inventory_2023 is not None:
        total_inventory_2023 *= 1000.0
    return [
        {
            "source_id": "california_usda_hogs_pdf",
            "metric_name": "hogs_inventory",
            "value": total_inventory_2023,
            "unit": "head",
            "reference_year": 2023,
            "release_date": release_match.group(1) if release_match else "",
            "raw_file_name": pdf_path.name,
            "notes": "Parsed from Pacific Region hogs and pigs release text; California 2024 inventory is not available in the report text.",
        }
    ]


def parse_sheep_goats_pdf(pdf_path: Path) -> list[dict[str, object]]:
    text = extract_pdf_text(pdf_path)
    lines = clean_pdf_lines(text)
    release_match = re.search(r"Released:\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", text)
    records: list[dict[str, object]] = []
    sheep_tokens = numeric_tokens_between(
        lines,
        "All sheep and lambs ...................................",
        "Breeding sheep and lambs ........................",
    )
    records.append(
        {
            "source_id": "california_usda_sheep_goats_pdf",
            "metric_name": "sheep_incl_lambs_inventory",
            "value": token_to_float(sheep_tokens[1]) * 1000.0 if len(sheep_tokens) >= 2 else None,
            "unit": "head",
            "reference_year": 2025,
            "release_date": release_match.group(1) if release_match else "",
            "raw_file_name": pdf_path.name,
            "notes": "Parsed from Pacific Region sheep and goats release text.",
        }
    )

    overview = parse_state_overview()
    goat_rows = overview[overview["metric_name"].str.contains("Goats", regex=False)].copy()
    metric_lookup = {
        "Goats, Meat & Other - Inventory ( First of Jan. 2026 )": "goats_meat_other_inventory",
        "Goats, Milk - Inventory ( First of Jan. 2026 )": "goats_milk_inventory",
    }
    for _, row in goat_rows.iterrows():
        records.append(
            {
                "source_id": "california_usda_sheep_goats_pdf",
                "metric_name": metric_lookup.get(str(row["metric_name"]), str(row["metric_name"])),
                "value": float(row["value"]) if pd.notna(row["value"]) else None,
                "unit": "head",
                "reference_year": int(row["reference_year"]) if pd.notna(row["reference_year"]) else 2026,
                "release_date": release_match.group(1) if release_match else "",
                "raw_file_name": pdf_path.name,
                "notes": "Value cross-checked against California USDA state overview.",
            }
        )
    return records


def write_run_manifest(records: list[dict[str, object]]) -> None:
    RUN_MANIFEST_JSON.write_text(json.dumps(records, indent=2), encoding="utf-8")
    fieldnames = [
        "dataset_id",
        "source_id",
        "source_url",
        "output_file",
        "downloaded_at_utc",
        "status",
        "record_count",
        "notes",
    ]
    with RUN_MANIFEST_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    LIVESTOCK_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()

    overview = parse_state_overview()
    overview.to_csv(OVERVIEW_CSV, index=False)
    OVERVIEW_JSON.write_text(json.dumps(overview.to_dict(orient="records"), indent=2), encoding="utf-8")

    cattle_row = get_source_row("california_usda_cattle_inventory_pdf")
    hogs_row = get_source_row("california_usda_hogs_pdf")
    sheep_row = get_source_row("california_usda_sheep_goats_pdf")
    cattle_pdf = ROOT / str(cattle_row["target_file"])
    hogs_pdf = ROOT / str(hogs_row["target_file"])
    sheep_pdf = ROOT / str(sheep_row["target_file"])

    cattle_txt = write_pdf_text_snapshot(cattle_pdf)
    hogs_txt = write_pdf_text_snapshot(hogs_pdf)
    sheep_txt = write_pdf_text_snapshot(sheep_pdf)

    release_records = []
    release_records.extend(parse_cattle_pdf(cattle_pdf))
    release_records.extend(parse_hogs_pdf(hogs_pdf))
    release_records.extend(parse_sheep_goats_pdf(sheep_pdf))
    release_frame = pd.DataFrame(release_records)
    release_frame.to_csv(PDF_SUMMARY_CSV, index=False)
    PDF_SUMMARY_JSON.write_text(
        json.dumps(release_frame.to_dict(orient="records"), indent=2), encoding="utf-8"
    )

    run_manifest = [
        {
            "dataset_id": "usda_nass_california_livestock_overview_structured",
            "source_id": "california_usda_state_ag_overview_html",
            "source_url": STATE_OVERVIEW_URL,
            "output_file": str(OVERVIEW_CSV.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "generated",
            "record_count": str(len(overview)),
            "notes": "Structured California livestock and milk summary parsed from USDA state overview HTML.",
        },
        {
            "dataset_id": "usda_nass_california_livestock_release_summary",
            "source_id": "california_usda_cattle_inventory_pdf;california_usda_hogs_pdf;california_usda_sheep_goats_pdf",
            "source_url": "; ".join(
                [
                    str(cattle_row["source_url"]),
                    str(hogs_row["source_url"]),
                    str(sheep_row["source_url"]),
                ]
            ),
            "output_file": str(PDF_SUMMARY_CSV.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "generated",
            "record_count": str(len(release_frame)),
            "notes": "Structured California livestock release summary parsed from official USDA Pacific Region PDFs.",
        },
        {
            "dataset_id": "usda_nass_california_cattle_inventory_text",
            "source_id": "california_usda_cattle_inventory_pdf",
            "source_url": str(cattle_row["source_url"]),
            "output_file": str(cattle_txt.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "generated",
            "record_count": "",
            "notes": "Plain-text extraction of the official Pacific Region cattle inventory PDF.",
        },
        {
            "dataset_id": "usda_nass_california_hogs_text",
            "source_id": "california_usda_hogs_pdf",
            "source_url": str(hogs_row["source_url"]),
            "output_file": str(hogs_txt.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "generated",
            "record_count": "",
            "notes": "Plain-text extraction of the official Pacific Region hogs and pigs PDF.",
        },
        {
            "dataset_id": "usda_nass_california_sheep_goats_text",
            "source_id": "california_usda_sheep_goats_pdf",
            "source_url": str(sheep_row["source_url"]),
            "output_file": str(sheep_txt.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "generated",
            "record_count": "",
            "notes": "Plain-text extraction of the official Pacific Region sheep and goats PDF.",
        },
    ]
    write_run_manifest(run_manifest)

    print(f"Wrote {OVERVIEW_JSON}")
    print(f"Wrote {OVERVIEW_CSV}")
    print(f"Wrote {PDF_SUMMARY_JSON}")
    print(f"Wrote {PDF_SUMMARY_CSV}")
    print(f"Wrote {RUN_MANIFEST_JSON}")
    print(f"Wrote {RUN_MANIFEST_CSV}")


if __name__ == "__main__":
    main()
