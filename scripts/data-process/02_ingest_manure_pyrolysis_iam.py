from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "ManurePyrolysisIAM"
OUT_UNIFIED_DIR = ROOT / "data" / "processed" / "unified_features"
OUT_SCENARIO_DIR = ROOT / "data" / "processed" / "scenario_inputs"


HIGH_PRIORITY_TABLES = {
    "biochar_c_avoidance.csv": "environmental_reference",
    "biochar_c_sequestration.csv": "environmental_reference",
    "biochar_ghg_avoided_decomposition.csv": "environmental_reference",
    "biochar_manure_supply_GWP_kg_FU.csv": "environmental_reference",
    "biochar_price.csv": "economic_reference",
    "biochar_soil_N2O_flux.csv": "environmental_reference",
    "feedstock_cost_pyrolysis.csv": "economic_reference",
    "total_cost_pyrolysis.csv": "economic_reference",
    "unit_cost_pyrolysis.csv": "economic_reference",
}


def classify_table(file_name: str) -> tuple[str, str]:
    if file_name in HIGH_PRIORITY_TABLES:
        return HIGH_PRIORITY_TABLES[file_name], "high"
    lowered = file_name.lower()
    if "cost" in lowered or "price" in lowered:
        return "economic_reference", "medium"
    if "ghg" in lowered or "n2o" in lowered or "luc" in lowered or "carbon" in lowered:
        return "environmental_reference", "medium"
    if "land" in lowered:
        return "land_use_reference", "low"
    if "food" in lowered or "pcals" in lowered or "feed" in lowered or "herd" in lowered:
        return "market_reference", "low"
    return "general_reference", "low"


def load_energy_balance_long() -> pd.DataFrame:
    workbook = RAW_DIR / "pyrolysis_energy_balance.xlsx"
    excel = pd.ExcelFile(workbook)
    frames: list[pd.DataFrame] = []

    for sheet_name in excel.sheet_names:
        frame = excel.parse(sheet_name)
        frame.columns = ["constant_name", "value", "units", "notes"]
        frame["livestock_type"] = sheet_name.lower()
        frame["source_repo"] = "ManurePyrolysisIAM"
        frame["source_file"] = "pyrolysis_energy_balance.xlsx"
        frame["source_dataset_kind"] = "energy_balance_reference"
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    ordered_columns = [
        "source_repo",
        "source_file",
        "source_dataset_kind",
        "livestock_type",
        "constant_name",
        "value",
        "units",
        "notes",
    ]
    return combined[ordered_columns]


def build_baseline_inventory() -> pd.DataFrame:
    baseline_dir = RAW_DIR / "baseline_supplementary_tables"
    rows: list[dict[str, object]] = []

    for csv_file in sorted(baseline_dir.glob("*.csv")):
        frame = pd.read_csv(csv_file)
        table_role, paper1_priority = classify_table(csv_file.name)
        rows.append(
            {
                "source_repo": "ManurePyrolysisIAM",
                "source_file": csv_file.name,
                "table_role": table_role,
                "paper1_priority": paper1_priority,
                "row_count": len(frame),
                "column_count": len(frame.columns),
                "columns": "|".join(str(column) for column in frame.columns),
            }
        )

    return pd.DataFrame(rows)


def build_reference_manifest(inventory: pd.DataFrame) -> pd.DataFrame:
    manifest = inventory.copy()
    manifest["paper1_use"] = manifest["table_role"].map(
        {
            "economic_reference": "parameter_reference",
            "environmental_reference": "parameter_reference",
            "market_reference": "scenario_reference",
            "land_use_reference": "background_reference",
            "general_reference": "background_reference",
        }
    )
    return manifest.sort_values(["paper1_priority", "source_file"]).reset_index(drop=True)


def write_metadata(outputs: list[Path]) -> None:
    payload = {
        "source_repo": "ManurePyrolysisIAM",
        "source_repository_url": "https://github.com/PEESEgroup/ManurePyrolysisIAM",
        "processed_files": [path.name for path in outputs],
        "note": "Paper 1 uses the workbook and curated baseline supplementary tables as manure-side reference data.",
    }
    output_path = OUT_UNIFIED_DIR / "manure_pyrolysis_iam_metadata.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    OUT_UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

    energy_balance = load_energy_balance_long()
    inventory = build_baseline_inventory()
    manifest = build_reference_manifest(inventory)

    energy_balance_out = OUT_UNIFIED_DIR / "manure_pyrolysis_energy_balance_long.csv"
    inventory_out = OUT_UNIFIED_DIR / "manure_pyrolysis_baseline_table_inventory.csv"
    manifest_out = OUT_SCENARIO_DIR / "manure_pyrolysis_reference_manifest.csv"

    energy_balance.to_csv(energy_balance_out, index=False)
    inventory.to_csv(inventory_out, index=False)
    manifest.to_csv(manifest_out, index=False)

    write_metadata([energy_balance_out, inventory_out, manifest_out])

    print(f"Wrote {energy_balance_out}")
    print(f"Wrote {inventory_out}")
    print(f"Wrote {manifest_out}")


if __name__ == "__main__":
    main()
