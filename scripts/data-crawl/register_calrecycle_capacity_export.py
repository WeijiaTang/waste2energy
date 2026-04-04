from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_CA_DIR = ROOT / "data" / "raw" / "external-region-data" / "california"
WET_WASTE_DIR = RAW_CA_DIR / "wet_waste_supply"
SOURCE_MANIFEST_JSON = RAW_CA_DIR / "source_manifest.json"
SOURCE_MANIFEST_CSV = RAW_CA_DIR / "source_manifest.csv"

CAPACITY_SOURCE_ID = "california_calrecycle_capacity_planning_portal"
CANONICAL_BASENAME = "calrecycle_capacity_planning_manual_export"
REGISTRATION_JSON = WET_WASTE_DIR / "calrecycle_capacity_planning_manual_export_registration.json"
REGISTRATION_CSV = WET_WASTE_DIR / "calrecycle_capacity_planning_manual_export_registration.csv"
ACCEPTED_SUFFIXES = {".xlsx", ".xls", ".csv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register a manual CalRecycle capacity-planning export into the canonical California raw path."
    )
    parser.add_argument(
        "--source-path",
        required=True,
        help="Path to the manually exported official CalRecycle capacity-planning file (.xlsx, .xls, or .csv).",
    )
    parser.add_argument(
        "--exported-at",
        default="",
        help="Optional export timestamp or date supplied by the operator.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional operator notes about the manual export session.",
    )
    return parser.parse_args()


def sha256_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_source_manifest(
    copied_path: Path,
    timestamp: str,
    file_hash: str,
    file_size_bytes: int,
    original_source_path: Path,
    exported_at: str,
    notes: str,
) -> None:
    if not SOURCE_MANIFEST_JSON.exists():
        return

    records = json.loads(SOURCE_MANIFEST_JSON.read_text(encoding="utf-8"))
    for record in records:
        if record.get("source_id") != CAPACITY_SOURCE_ID:
            continue
        record["target_file"] = str(copied_path.relative_to(ROOT))
        record["expected_format"] = copied_path.suffix.lstrip(".")
        record["retrieval_method"] = "manual_export_registered"
        record["downloaded_at_utc"] = timestamp
        record["status"] = "manual_export_registered"
        record["sha256"] = file_hash
        record["file_size_bytes"] = str(file_size_bytes)
        record["error_message"] = ""
        suffix = f" Original export path: {original_source_path}."
        if exported_at:
            suffix += f" Operator-supplied export timestamp: {exported_at}."
        if notes:
            suffix += f" Operator notes: {notes}"
        record["notes"] = f"{record.get('notes', '').strip()}{suffix}".strip()

    SOURCE_MANIFEST_JSON.write_text(json.dumps(records, indent=2), encoding="utf-8")
    write_csv(SOURCE_MANIFEST_CSV, records)


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_path).expanduser().resolve()
    if not source_path.exists():
        raise SystemExit(f"Source export does not exist: {source_path}")
    if source_path.suffix.lower() not in ACCEPTED_SUFFIXES:
        accepted = ", ".join(sorted(ACCEPTED_SUFFIXES))
        raise SystemExit(f"Unsupported export format {source_path.suffix}. Accepted: {accepted}")

    WET_WASTE_DIR.mkdir(parents=True, exist_ok=True)
    destination = WET_WASTE_DIR / f"{CANONICAL_BASENAME}{source_path.suffix.lower()}"
    shutil.copy2(source_path, destination)

    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    file_hash = sha256_digest(destination)
    file_size_bytes = destination.stat().st_size

    registration_record = {
        "source_id": CAPACITY_SOURCE_ID,
        "source_organization": "California Department of Resources Recycling and Recovery",
        "source_url": "https://www2.calrecycle.ca.gov/LGCentral/datatools/reports/capacityplanning",
        "original_source_path": str(source_path),
        "registered_file": str(destination.relative_to(ROOT)),
        "registered_at_utc": timestamp,
        "operator_exported_at": args.exported_at,
        "sha256": file_hash,
        "file_size_bytes": file_size_bytes,
        "notes": args.notes,
    }

    REGISTRATION_JSON.write_text(json.dumps([registration_record], indent=2), encoding="utf-8")
    write_csv(REGISTRATION_CSV, [registration_record])
    update_source_manifest(
        copied_path=destination,
        timestamp=timestamp,
        file_hash=file_hash,
        file_size_bytes=file_size_bytes,
        original_source_path=source_path,
        exported_at=args.exported_at,
        notes=args.notes,
    )

    print(f"Registered manual export: {destination}")
    print(f"Wrote {REGISTRATION_JSON}")
    print(f"Wrote {REGISTRATION_CSV}")


if __name__ == "__main__":
    main()
