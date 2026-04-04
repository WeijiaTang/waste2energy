from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import fitz
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_CA_DIR = ROOT / "data" / "raw" / "external-region-data" / "california"
EMISSION_DIR = RAW_CA_DIR / "emission_factors"

WARM_PDF_URL = "https://www.epa.gov/system/files/documents/2023-12/warm_organic_materials_v16_dec.pdf"
WARM_LANDING_PAGE_URL = "https://www.epa.gov/warm/versions-waste-reduction-model"

WARM_PDF = EMISSION_DIR / "epa_warm_organic_materials_v16_dec_2023.pdf"
WARM_TEXT = EMISSION_DIR / "epa_warm_organic_materials_v16_dec_2023.txt"
REFERENCE_CSV = EMISSION_DIR / "california_waste_treatment_emission_factor_reference.csv"
REFERENCE_JSON = EMISSION_DIR / "california_waste_treatment_emission_factor_reference.json"
MANIFEST_CSV = EMISSION_DIR / "california_waste_treatment_emission_factor_manifest.csv"
MANIFEST_JSON = EMISSION_DIR / "california_waste_treatment_emission_factor_manifest.json"


def utcnow_text() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def write_manifest(rows: list[dict[str, object]]) -> None:
    MANIFEST_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_warm_text() -> str:
    if not WARM_PDF.exists():
        raise FileNotFoundError(
            "Missing EPA WARM PDF. Run "
            "`python scripts/data-crawl/download_official_sources.py --source-id "
            "california_epa_warm_organic_materials_pdf` first."
        )

    pdftotext_path = shutil.which("pdftotext")
    if pdftotext_path:
        subprocess.run(
            [pdftotext_path, "-layout", str(WARM_PDF), str(WARM_TEXT)],
            check=True,
        )
        return WARM_TEXT.read_text(encoding="utf-8", errors="replace")

    document = fitz.open(WARM_PDF)
    text = "\n".join(page.get_text("text") for page in document)
    WARM_TEXT.write_text(text, encoding="utf-8")
    return text


def parse_warm_number(token: str) -> float:
    token = token.strip()
    if token.startswith("(") and token.endswith(")"):
        return -float(token[1:-1])
    return float(token)


def require_match(pattern: str, text: str, label: str) -> re.Match[str]:
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        raise RuntimeError(f"Could not extract {label} from EPA WARM food-waste text")
    return match


def build_reference_rows(text: str, timestamp: str) -> list[dict[str, object]]:
    summary_match = require_match(
        (
            r"Exhibit 1-10: Net Emissions for Food Waste and Mixed Organics under Each Materials "
            r"Management Option.*?Food Waste\s+\((?P<source_reduction>[0-9.]+)\)\s+NA\s+"
            r"\((?P<composting>[0-9.]+)\)\s+\((?P<combustion>[0-9.]+)\)\s+"
            r"(?P<landfilling>[0-9.]+)\s+\((?P<anaerobic_digestion>[0-9.]+)\)"
        ),
        text,
        "Exhibit 1-10 summary row",
    )
    landfill_match = require_match(
        (
            r"Exhibit 1-49: Components of the Landfill Emission Factor.*?Food Waste\s+"
            r"(?P<ch4_no_recovery>[0-9.]+)\s+(?P<ch4_flaring>[0-9.]+)\s+"
            r"(?P<ch4_electricity>[0-9.]+)\s+\((?P<carbon_storage>[0-9.]+)\)\s+"
            r"(?P<transport>[0-9.]+)\s+(?P<no_recovery>[0-9.]+)\s+"
            r"(?P<flaring>[0-9.]+)\s+(?P<electricity_generation>[0-9.]+)"
        ),
        text,
        "Exhibit 1-49 landfill row",
    )
    dry_curing_match = require_match(
        (
            r"Exhibit 1-50: Dry Anaerobic Digestion Emission Factors for Food Waste with Digestate "
            r"Curing.*?Food Waste\s+(?P<process>[0-9.]+)\s+\((?P<utility>[0-9.]+)\)\s+"
            r"\((?P<fertilizer>[0-9.]+)\)\s+\((?P<soil>[0-9.]+)\)\s+(?P<non_energy>[0-9.]+)\s+"
            r"(?P<transport>[0-9.]+)\s+\((?P<net>[0-9.]+)\)"
        ),
        text,
        "Exhibit 1-50 dry digestion curing row",
    )
    dry_direct_match = require_match(
        (
            r"Exhibit 1-51: Dry Anaerobic Digestion Emission Factors for Food Waste with Direct Land "
            r"Application.*?Food Waste\s+(?P<process>[0-9.]+)\s+\((?P<utility>[0-9.]+)\)\s+"
            r"\((?P<fertilizer>[0-9.]+)\)\s+\((?P<soil>[0-9.]+)\)\s+(?P<non_energy>[0-9.]+)\s+"
            r"(?P<transport>[0-9.]+)\s+\((?P<net>[0-9.]+)\)"
        ),
        text,
        "Exhibit 1-51 dry digestion direct-application row",
    )
    wet_curing_match = require_match(
        (
            r"Exhibit 1-52: Wet Anaerobic Digestion Emission Factors for Food Waste with Digestate "
            r"Curing.*?Food Waste\s+(?P<process>[0-9.]+)\s+\((?P<utility>[0-9.]+)\)\s+"
            r"\((?P<fertilizer>[0-9.]+)\)\s+\((?P<soil>[0-9.]+)\)\s+(?P<non_energy>[0-9.]+)\s+"
            r"(?P<transport>[0-9.]+)\s+\((?P<net>[0-9.]+)\)"
        ),
        text,
        "Exhibit 1-52 wet digestion curing row",
    )
    wet_direct_match = require_match(
        (
            r"Exhibit 1-53: Wet Anaerobic Digestion Emission Factors for Food Waste with Direct Land "
            r"Application.*?Food Waste\s+(?P<process>[0-9.]+)\s+\((?P<utility>[0-9.]+)\)\s+"
            r"\((?P<fertilizer>[0-9.]+)\)\s+\((?P<soil>[0-9.]+)\)\s+(?P<non_energy>[0-9.]+)\s+"
            r"(?P<transport>[0-9.]+)\s+\((?P<net>[0-9.]+)\)"
        ),
        text,
        "Exhibit 1-53 wet digestion direct-application row",
    )

    base_fields = {
        "region_id": "us_ca",
        "region_name": "California",
        "country": "United States",
        "waste_stream_type": "food_waste",
        "factor_unit": "kgCO2e_per_short_ton",
        "source_organization": "U.S. Environmental Protection Agency",
        "source_url": WARM_PDF_URL,
        "landing_page_url": WARM_LANDING_PAGE_URL,
        "download_or_publication_date": timestamp,
        "raw_file_name": WARM_PDF.name,
        "extracted_text_file": WARM_TEXT.name,
    }

    pathway_rows = [
        {
            "management_pathway": "landfilling_summary_all_landfills",
            "baseline_relevance": "summary_reference",
            "source_exhibit": "Exhibit 1-10",
            "source_metric_value_mtco2e_per_short_ton": parse_warm_number(
                summary_match.group("landfilling")
            ),
            "notes": (
                "EPA WARM chapter-level food-waste landfilling summary value. Included as a broad reference "
                "but not used as the default California baseline because Exhibit 1-49 provides explicit landfill "
                "gas management variants."
            ),
        },
        {
            "management_pathway": "landfill_without_lfg_recovery",
            "baseline_relevance": "upper_bound_landfill",
            "source_exhibit": "Exhibit 1-49",
            "source_metric_value_mtco2e_per_short_ton": parse_warm_number(
                landfill_match.group("no_recovery")
            ),
            "notes": (
                "EPA WARM food-waste landfill case without landfill-gas recovery. Retained as a higher-emission "
                "comparison case."
            ),
        },
        {
            "management_pathway": "landfill_with_lfg_recovery_and_flaring",
            "baseline_relevance": "alternative_landfill_case",
            "source_exhibit": "Exhibit 1-49",
            "source_metric_value_mtco2e_per_short_ton": parse_warm_number(
                landfill_match.group("flaring")
            ),
            "notes": (
                "EPA WARM food-waste landfill case with landfill-gas recovery and flaring."
            ),
        },
        {
            "management_pathway": "landfill_with_lfg_recovery_and_electricity_generation",
            "baseline_relevance": "baseline_default",
            "source_exhibit": "Exhibit 1-49",
            "source_metric_value_mtco2e_per_short_ton": parse_warm_number(
                landfill_match.group("electricity_generation")
            ),
            "notes": (
                "Selected as the default California Paper 1 baseline proxy. This is an inference from EPA WARM "
                "Exhibit 1-49 rather than a California facility-specific statewide average. It represents a "
                "controlled landfill pathway with landfill-gas recovery and electricity generation."
            ),
        },
        {
            "management_pathway": "combustion_with_energy_recovery",
            "baseline_relevance": "alternative_treatment_pathway",
            "source_exhibit": "Exhibit 1-10",
            "source_metric_value_mtco2e_per_short_ton": parse_warm_number(
                f"({summary_match.group('combustion')})"
            ),
            "notes": "EPA WARM food-waste combustion net emissions summary value.",
        },
        {
            "management_pathway": "composting",
            "baseline_relevance": "alternative_treatment_pathway",
            "source_exhibit": "Exhibit 1-10",
            "source_metric_value_mtco2e_per_short_ton": parse_warm_number(
                f"({summary_match.group('composting')})"
            ),
            "notes": "EPA WARM food-waste composting net emissions summary value.",
        },
        {
            "management_pathway": "anaerobic_digestion_summary",
            "baseline_relevance": "alternative_treatment_pathway",
            "source_exhibit": "Exhibit 1-10",
            "source_metric_value_mtco2e_per_short_ton": parse_warm_number(
                f"({summary_match.group('anaerobic_digestion')})"
            ),
            "notes": (
                "EPA WARM food-waste anaerobic digestion net-emissions summary value. Used when a "
                "California facility inventory can identify digestion share but not wet-vs-dry digestion subtype."
            ),
        },
        {
            "management_pathway": "dry_anaerobic_digestion_with_digestate_curing",
            "baseline_relevance": "alternative_treatment_pathway",
            "source_exhibit": "Exhibit 1-50",
            "source_metric_value_mtco2e_per_short_ton": parse_warm_number(
                f"({dry_curing_match.group('net')})"
            ),
            "notes": "EPA WARM food-waste dry anaerobic digestion pathway with digestate curing.",
        },
        {
            "management_pathway": "dry_anaerobic_digestion_with_direct_land_application",
            "baseline_relevance": "alternative_treatment_pathway",
            "source_exhibit": "Exhibit 1-51",
            "source_metric_value_mtco2e_per_short_ton": parse_warm_number(
                f"({dry_direct_match.group('net')})"
            ),
            "notes": "EPA WARM food-waste dry anaerobic digestion pathway with direct land application.",
        },
        {
            "management_pathway": "wet_anaerobic_digestion_with_digestate_curing",
            "baseline_relevance": "alternative_treatment_pathway",
            "source_exhibit": "Exhibit 1-52",
            "source_metric_value_mtco2e_per_short_ton": parse_warm_number(
                f"({wet_curing_match.group('net')})"
            ),
            "notes": "EPA WARM food-waste wet anaerobic digestion pathway with digestate curing.",
        },
        {
            "management_pathway": "wet_anaerobic_digestion_with_direct_land_application",
            "baseline_relevance": "alternative_treatment_pathway",
            "source_exhibit": "Exhibit 1-53",
            "source_metric_value_mtco2e_per_short_ton": parse_warm_number(
                f"({wet_direct_match.group('net')})"
            ),
            "notes": "EPA WARM food-waste wet anaerobic digestion pathway with direct land application.",
        },
    ]

    rows: list[dict[str, object]] = []
    for row in pathway_rows:
        mtco2e_per_short_ton = float(row["source_metric_value_mtco2e_per_short_ton"])
        rows.append(
            {
                **base_fields,
                **row,
                "factor_value": mtco2e_per_short_ton * 1000.0,
            }
        )
    return rows


def main() -> None:
    EMISSION_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = utcnow_text()
    text = ensure_warm_text()

    reference_rows = build_reference_rows(text, timestamp)
    reference_frame = pd.DataFrame(reference_rows)
    reference_frame.to_csv(REFERENCE_CSV, index=False)
    REFERENCE_JSON.write_text(
        json.dumps(reference_frame.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    manifest_rows = [
        {
            "dataset_id": "california_epa_warm_food_waste_text_extract",
            "source_url": WARM_PDF_URL,
            "landing_page_url": WARM_LANDING_PAGE_URL,
            "output_file": str(WARM_TEXT.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "generated",
            "record_count": "",
            "notes": "Plain-text extraction of the EPA WARM organic materials PDF for traceable factor parsing.",
        },
        {
            "dataset_id": "california_waste_treatment_emission_factor_reference",
            "source_url": WARM_PDF_URL,
            "landing_page_url": WARM_LANDING_PAGE_URL,
            "output_file": str(REFERENCE_CSV.relative_to(ROOT)),
            "downloaded_at_utc": timestamp,
            "status": "generated",
            "record_count": len(reference_rows),
            "notes": (
                "Structured EPA WARM food-waste treatment emission-factor reference for California Paper 1. "
                "The default baseline row is an explicitly documented analytical proxy."
            ),
        },
    ]
    write_manifest(manifest_rows)

    print(f"Wrote {WARM_TEXT}")
    print(f"Wrote {REFERENCE_CSV}")
    print(f"Wrote {REFERENCE_JSON}")
    print(f"Wrote {MANIFEST_CSV}")
    print(f"Wrote {MANIFEST_JSON}")


if __name__ == "__main__":
    main()
