from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_CA_DIR = ROOT / "data" / "raw" / "external-region-data" / "california"
POLICY_DIR = RAW_CA_DIR / "policy_reference"

SB1383_BILL_TEXT_URL = "https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=201520160SB1383"
FOOD_RECOVERY_FAQ_URL = "https://calrecycle.ca.gov/organics/slcp/faq/foodrecovery/"
PROCUREMENT_URL = "https://calrecycle.ca.gov/organics/slcp/procurement/recoveredorganicwasteproducts/"
ENFORCEMENT_URL = "https://calrecycle.ca.gov/organics/slcp/enforcement/"

SB1383_BILL_TEXT_HTML = POLICY_DIR / "leginfo_sb1383_bill_text.html"
FOOD_RECOVERY_FAQ_HTML = POLICY_DIR / "calrecycle_sb1383_food_recovery_faq.html"
PROCUREMENT_HTML = POLICY_DIR / "calrecycle_sb1383_procurement_recovered_organic_waste_products.html"
ENFORCEMENT_HTML = POLICY_DIR / "calrecycle_sb1383_enforcement.html"
REFERENCE_JSON = POLICY_DIR / "california_sb1383_policy_reference.json"
REFERENCE_CSV = POLICY_DIR / "california_sb1383_policy_reference.csv"
MANIFEST_JSON = POLICY_DIR / "california_sb1383_policy_manifest.json"
MANIFEST_CSV = POLICY_DIR / "california_sb1383_policy_manifest.csv"

USER_AGENT = "Waste2EnergyResearchBot/0.1 (+https://github.com/openai)"


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8", errors="replace")


def utcnow_text() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def write_manifest(rows: list[dict[str, object]]) -> None:
    MANIFEST_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = utcnow_text()

    bill_text = fetch_text(SB1383_BILL_TEXT_URL)
    food_recovery_faq = fetch_text(FOOD_RECOVERY_FAQ_URL)
    procurement_page = fetch_text(PROCUREMENT_URL)
    enforcement_page = fetch_text(ENFORCEMENT_URL)

    SB1383_BILL_TEXT_HTML.write_text(bill_text, encoding="utf-8")
    FOOD_RECOVERY_FAQ_HTML.write_text(food_recovery_faq, encoding="utf-8")
    PROCUREMENT_HTML.write_text(procurement_page, encoding="utf-8")
    ENFORCEMENT_HTML.write_text(enforcement_page, encoding="utf-8")

    reference_rows = [
        {
            "region_id": "us_ca",
            "region_name": "California",
            "country": "United States",
            "policy_name": "sb1383_organic_waste_disposal_reduction_target_2025",
            "policy_type": "regulatory_target",
            "policy_start_year": 2016,
            "policy_end_year": 2025,
            "support_level_value": 75.0,
            "support_level_unit": "pct_reduction_from_2014_organic_waste_disposal",
            "target_pathway": "organic_waste_diversion",
            "source_organization": "California Legislature",
            "source_url": SB1383_BILL_TEXT_URL,
            "download_or_publication_date": timestamp,
            "raw_file_name": SB1383_BILL_TEXT_HTML.name,
            "notes": (
                "SB 1383 establishes a statewide target to reduce disposal of organic waste "
                "75 percent below the 2014 level by 2025."
            ),
        },
        {
            "region_id": "us_ca",
            "region_name": "California",
            "country": "United States",
            "policy_name": "sb1383_edible_food_recovery_target_2025",
            "policy_type": "regulatory_target",
            "policy_start_year": 2016,
            "policy_end_year": 2025,
            "support_level_value": 20.0,
            "support_level_unit": "pct_of_currently_disposed_edible_food_recovered_for_human_consumption",
            "target_pathway": "edible_food_recovery",
            "source_organization": "California Department of Resources Recycling and Recovery",
            "source_url": FOOD_RECOVERY_FAQ_URL,
            "download_or_publication_date": timestamp,
            "raw_file_name": FOOD_RECOVERY_FAQ_HTML.name,
            "notes": (
                "CalRecycle clarifies that the 20 percent edible-food-recovery target by 2025 is a "
                "statewide goal, not an individual jurisdiction quota."
            ),
        },
        {
            "region_id": "us_ca",
            "region_name": "California",
            "country": "United States",
            "policy_name": "sb1383_regulations_effective_2022",
            "policy_type": "implementation_requirement",
            "policy_start_year": 2022,
            "policy_end_year": "",
            "support_level_value": 2022,
            "support_level_unit": "effective_year",
            "target_pathway": "general",
            "source_organization": "California Legislature",
            "source_url": SB1383_BILL_TEXT_URL,
            "download_or_publication_date": timestamp,
            "raw_file_name": SB1383_BILL_TEXT_HTML.name,
            "notes": (
                "Regulations adopted under PRC 42652.5 take effect on or after January 1, 2022."
            ),
        },
        {
            "region_id": "us_ca",
            "region_name": "California",
            "country": "United States",
            "policy_name": "sb1383_procurement_target_per_capita_2022_2026",
            "policy_type": "procurement_requirement",
            "policy_start_year": 2022,
            "policy_end_year": 2026,
            "support_level_value": 0.08,
            "support_level_unit": "tons_organic_waste_per_resident_per_year",
            "target_pathway": "recovered_organic_waste_products",
            "source_organization": "California Department of Resources Recycling and Recovery",
            "source_url": PROCUREMENT_URL,
            "download_or_publication_date": timestamp,
            "raw_file_name": PROCUREMENT_HTML.name,
            "notes": (
                "Each jurisdiction's procurement target is calculated as population multiplied by "
                "0.08 tons of organic waste per resident per year; the 2022 target remains in effect "
                "through December 31, 2026."
            ),
        },
        {
            "region_id": "us_ca",
            "region_name": "California",
            "country": "United States",
            "policy_name": "sb1383_jurisdiction_enforcement_program_2022",
            "policy_type": "compliance_requirement",
            "policy_start_year": 2022,
            "policy_end_year": "",
            "support_level_value": 1.0,
            "support_level_unit": "required_program",
            "target_pathway": "compliance_enforcement",
            "source_organization": "California Department of Resources Recycling and Recovery",
            "source_url": ENFORCEMENT_URL,
            "download_or_publication_date": timestamp,
            "raw_file_name": ENFORCEMENT_HTML.name,
            "notes": (
                "Jurisdictions must implement inspection and enforcement programs to ensure "
                "organic waste generators comply with SB 1383 requirements."
            ),
        },
    ]

    reference_frame = pd.DataFrame(reference_rows)
    REFERENCE_JSON.write_text(
        json.dumps(reference_frame.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    reference_frame.to_csv(REFERENCE_CSV, index=False)

    manifest_rows = [
        {
            "dataset_id": "california_sb1383_bill_text_snapshot",
            "source_url": SB1383_BILL_TEXT_URL,
            "output_file": str(SB1383_BILL_TEXT_HTML.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "downloaded",
            "record_count": "",
            "notes": "Official California Legislature SB 1383 bill text HTML snapshot.",
        },
        {
            "dataset_id": "california_sb1383_food_recovery_faq_snapshot",
            "source_url": FOOD_RECOVERY_FAQ_URL,
            "output_file": str(FOOD_RECOVERY_FAQ_HTML.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "downloaded",
            "record_count": "",
            "notes": "Official CalRecycle food recovery FAQ HTML snapshot.",
        },
        {
            "dataset_id": "california_sb1383_procurement_snapshot",
            "source_url": PROCUREMENT_URL,
            "output_file": str(PROCUREMENT_HTML.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "downloaded",
            "record_count": "",
            "notes": "Official CalRecycle procurement requirements HTML snapshot.",
        },
        {
            "dataset_id": "california_sb1383_enforcement_snapshot",
            "source_url": ENFORCEMENT_URL,
            "output_file": str(ENFORCEMENT_HTML.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "downloaded",
            "record_count": "",
            "notes": "Official CalRecycle enforcement overview HTML snapshot.",
        },
        {
            "dataset_id": "california_sb1383_policy_reference",
            "source_url": FOOD_RECOVERY_FAQ_URL,
            "output_file": str(REFERENCE_CSV.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "generated",
            "record_count": len(reference_rows),
            "notes": "Structured California SB 1383 policy reference table for Waste2Energy Paper 1.",
        },
    ]
    write_manifest(manifest_rows)

    print(f"Wrote {SB1383_BILL_TEXT_HTML}")
    print(f"Wrote {FOOD_RECOVERY_FAQ_HTML}")
    print(f"Wrote {PROCUREMENT_HTML}")
    print(f"Wrote {ENFORCEMENT_HTML}")
    print(f"Wrote {REFERENCE_CSV}")
    print(f"Wrote {REFERENCE_JSON}")
    print(f"Wrote {MANIFEST_CSV}")
    print(f"Wrote {MANIFEST_JSON}")


if __name__ == "__main__":
    main()
