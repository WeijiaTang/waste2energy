from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
RAW_BASE_DIR = ROOT / "data" / "raw" / "external-region-data" / "california"
MANIFEST_PATH = Path(__file__).with_name("california_sources.json")
OUTPUT_JSON_PATH = RAW_BASE_DIR / "source_manifest.json"
OUTPUT_CSV_PATH = RAW_BASE_DIR / "source_manifest.csv"

USER_AGENT = "Waste2EnergyResearchBot/0.1 (+https://github.com/openai)"

CATEGORY_DIRS = {
    "region_context": RAW_BASE_DIR / "region_context",
    "livestock_supply": RAW_BASE_DIR / "livestock_supply",
    "wet_waste_supply": RAW_BASE_DIR / "wet_waste_supply",
    "energy_prices": RAW_BASE_DIR / "energy_prices",
    "emission_factors": RAW_BASE_DIR / "emission_factors",
    "policy_reference": RAW_BASE_DIR / "policy_reference",
}


@dataclass(frozen=True)
class SourceEntry:
    source_id: str
    dataset_name: str
    source_organization: str
    source_url: str
    landing_page_url: str
    region_id: str
    region_name: str
    country: str
    data_category: str
    target_filename: str
    expected_format: str
    retrieval_method: str
    license_notes: str
    notes: str

    @property
    def output_path(self) -> Path:
        return CATEGORY_DIRS[self.data_category] / self.target_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download California official data sources into data/raw.")
    parser.add_argument(
        "--source-id",
        action="append",
        dest="source_ids",
        default=[],
        help="Limit downloads to one or more source IDs from california_sources.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download supported sources even if the target file already exists.",
    )
    return parser.parse_args()


def load_sources() -> list[SourceEntry]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [SourceEntry(**item) for item in payload]


def iter_selected_sources(source_ids: list[str]) -> Iterable[SourceEntry]:
    sources = load_sources()
    if not source_ids:
        return sources

    wanted = set(source_ids)
    selected = [source for source in sources if source.source_id in wanted]
    missing = sorted(wanted - {source.source_id for source in selected})
    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(f"Unknown source_id values: {missing_text}")
    return selected


def load_existing_records() -> dict[str, dict[str, str]]:
    if not OUTPUT_JSON_PATH.exists():
        return {}

    payload = json.loads(OUTPUT_JSON_PATH.read_text(encoding="utf-8"))
    return {item["source_id"]: item for item in payload}


def ensure_output_dirs() -> None:
    RAW_BASE_DIR.mkdir(parents=True, exist_ok=True)
    for path in CATEGORY_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response:
        return response.read()


def sha256_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_source(source: SourceEntry, force: bool) -> dict[str, str]:
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    output_path = source.output_path

    base_record = {
        "source_id": source.source_id,
        "dataset_name": source.dataset_name,
        "source_organization": source.source_organization,
        "source_url": source.source_url,
        "landing_page_url": source.landing_page_url,
        "region_id": source.region_id,
        "region_name": source.region_name,
        "country": source.country,
        "data_category": source.data_category,
        "target_file": str(output_path.relative_to(ROOT)),
        "expected_format": source.expected_format,
        "retrieval_method": source.retrieval_method,
        "license_notes": source.license_notes,
        "notes": source.notes,
        "downloaded_at_utc": timestamp,
        "status": "",
        "sha256": "",
        "file_size_bytes": "",
        "error_message": "",
    }

    if source.retrieval_method == "manual_export_required":
        base_record["status"] = "manual_export_required"
        return base_record

    if output_path.exists() and not force:
        existing = output_path.read_bytes()
        base_record["status"] = "already_present"
        base_record["sha256"] = sha256_digest(existing)
        base_record["file_size_bytes"] = str(len(existing))
        return base_record

    try:
        content = fetch_bytes(source.source_url)
        output_path.write_bytes(content)
        base_record["status"] = "downloaded"
        base_record["sha256"] = sha256_digest(content)
        base_record["file_size_bytes"] = str(len(content))
        return base_record
    except HTTPError as exc:
        base_record["status"] = "download_failed"
        base_record["error_message"] = f"HTTP {exc.code}: {exc.reason}"
        return base_record
    except URLError as exc:
        base_record["status"] = "download_failed"
        base_record["error_message"] = f"URL error: {exc.reason}"
        return base_record
    except TimeoutError as exc:
        base_record["status"] = "download_failed"
        base_record["error_message"] = f"Timeout: {exc}"
        return base_record


def write_output_manifests(records: list[dict[str, str]]) -> None:
    OUTPUT_JSON_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")

    fieldnames = [
        "source_id",
        "dataset_name",
        "source_organization",
        "source_url",
        "landing_page_url",
        "region_id",
        "region_name",
        "country",
        "data_category",
        "target_file",
        "expected_format",
        "retrieval_method",
        "license_notes",
        "notes",
        "downloaded_at_utc",
        "status",
        "sha256",
        "file_size_bytes",
        "error_message",
    ]
    with OUTPUT_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    all_sources = load_sources()
    selected_ids = set(args.source_ids)
    existing_records = load_existing_records()

    records: list[dict[str, str]] = []
    for source in all_sources:
        should_run = not selected_ids or source.source_id in selected_ids
        if should_run:
            record = download_source(source, force=args.force)
        else:
            record = existing_records.get(source.source_id)
            if record is None:
                record = {
                    "source_id": source.source_id,
                    "dataset_name": source.dataset_name,
                    "source_organization": source.source_organization,
                    "source_url": source.source_url,
                    "landing_page_url": source.landing_page_url,
                    "region_id": source.region_id,
                    "region_name": source.region_name,
                    "country": source.country,
                    "data_category": source.data_category,
                    "target_file": str(source.output_path.relative_to(ROOT)),
                    "expected_format": source.expected_format,
                    "retrieval_method": source.retrieval_method,
                    "license_notes": source.license_notes,
                    "notes": source.notes,
                    "downloaded_at_utc": "",
                    "status": "not_requested_in_current_run",
                    "sha256": "",
                    "file_size_bytes": "",
                    "error_message": "",
                }
        records.append(record)

    write_output_manifests(records)

    downloaded = sum(1 for record in records if record["status"] == "downloaded")
    already_present = sum(1 for record in records if record["status"] == "already_present")
    manual = sum(1 for record in records if record["status"] == "manual_export_required")
    failed = sum(1 for record in records if record["status"] == "download_failed")

    print(f"Manifest written to {OUTPUT_JSON_PATH}")
    print(f"Manifest written to {OUTPUT_CSV_PATH}")
    print(
        "Summary: "
        f"downloaded={downloaded}, already_present={already_present}, "
        f"manual_export_required={manual}, failed={failed}"
    )
    for record in records:
        print(f"{record['source_id']}: {record['status']} -> {record['target_file']}")
        if record["error_message"]:
            print(f"  error: {record['error_message']}")


if __name__ == "__main__":
    main()
