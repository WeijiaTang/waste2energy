from __future__ import annotations

import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "Wet-Waste-Biomass-Opt"
OUT_DIR = ROOT / "data" / "processed" / "unified_features"
SUPPORT_DOCX = RAW_DIR / "SI_Dataset(1).docx"


HTC_RENAME_MAP = {
    "Biomass type": "feedstock_name",
    "C (%)": "feedstock_carbon_pct",
    "H (%)": "feedstock_hydrogen_pct",
    "N (%)": "feedstock_nitrogen_pct",
    "O (%)": "feedstock_oxygen_pct",
    "WC (%)": "feedstock_moisture_pct",
    "VM (%)": "feedstock_volatile_matter_pct",
    "FC (%)": "feedstock_fixed_carbon_pct",
    "Ash (%)": "feedstock_ash_pct",
    "T (°C)": "process_temperature_c",
    "RT (min)": "residence_time_min",
    "Char_yield (%)": "product_char_yield_pct",
    "Char_C (%)": "product_char_carbon_pct",
    "Char_H (%)": "product_char_hydrogen_pct",
    "Char_N (%)": "product_char_nitrogen_pct",
    "Char_O (%)": "product_char_oxygen_pct",
    "HHV (MJ/kg)": "product_char_hhv_mj_per_kg",
    "ER (%)": "energy_recovery_pct",
    "CR (%)": "carbon_retention_pct",
    "Reference": "reference_label",
}


PYROLYSIS_RENAME_MAP = {
    "C (wt%)": "feedstock_carbon_pct",
    "H (wt%)": "feedstock_hydrogen_pct",
    "N (wt%)": "feedstock_nitrogen_pct",
    "O (wt%)": "feedstock_oxygen_pct",
    "Bio_HHV": "feedstock_hhv_mj_per_kg",
    "FC (%)": "feedstock_fixed_carbon_pct",
    "VM (%)": "feedstock_volatile_matter_pct",
    "Ash (%)": "feedstock_ash_pct",
    "T (°C)": "process_temperature_c",
    "RT (min)": "residence_time_min",
    "HT (°C/min)": "heating_rate_c_per_min",
    "Yield_char (%)": "product_char_yield_pct",
    "C_char (%)": "product_char_carbon_pct",
    "H_char (%)": "product_char_hydrogen_pct",
    "O_char (%)": "product_char_oxygen_pct",
    "N_char (%)": "product_char_nitrogen_pct",
    "Char_HHV": "product_char_hhv_mj_per_kg",
    "ER": "energy_recovery_pct",
    "CR": "carbon_retention_pct",
    "REF": "reference_label",
    "https://doi.org/10.1016/j.enconman.2020.113258": "source_extra_value_1",
    "Unnamed: 21": "source_extra_value_2",
    "Unnamed: 22": "source_extra_value_3",
}


TEXT_COLUMNS = {
    "feedstock_name",
    "feedstock_group",
    "pathway",
    "source_repo",
    "source_file",
    "sample_id",
    "reference_label",
    "source_extra_value_1",
    "source_extra_value_2",
    "source_extra_value_3",
    "source_dataset_kind",
    "blending_case",
}

DOCX_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def categorize_feedstock(name: str) -> str:
    if not name:
        return "unknown"

    lowered = name.lower()
    keyword_map = {
        "manure": "manure",
        "sludge": "sludge",
        "food": "food_waste",
        "kitchen": "food_waste",
        "sewage": "sludge",
        "digestate": "digestate",
        "algae": "algae",
        "straw": "lignocellulosic",
        "husk": "lignocellulosic",
        "wood": "lignocellulosic",
        "grass": "lignocellulosic",
        "paper": "paper_waste",
        "municipal": "municipal_waste",
    }
    for keyword, group in keyword_map.items():
        if keyword in lowered:
            return group
    return "other_biomass"


def standardize_frame(
    frame: pd.DataFrame,
    rename_map: dict[str, str],
    pathway: str,
    source_file: str,
) -> pd.DataFrame:
    data = frame.rename(columns=rename_map).copy()
    data = data.loc[:, ~data.columns.astype(str).str.startswith("Unnamed")]
    if "feedstock_name" not in data.columns:
        data["feedstock_name"] = pd.NA

    data["feedstock_group"] = data["feedstock_name"].fillna("").map(categorize_feedstock)
    data["pathway"] = pathway
    data["source_repo"] = "Wet-Waste-Biomass-Opt"
    data["source_file"] = source_file
    data["source_dataset_kind"] = "literature_experiment"
    data["blending_case"] = "single_source_reference"
    data["blend_manure_ratio"] = pd.NA
    data["blend_wet_waste_ratio"] = pd.NA
    data["sample_id"] = [
        f"Wet-Waste-Biomass-Opt::{pathway}::{idx:04d}" for idx in range(1, len(data) + 1)
    ]

    for column in data.columns:
        if column not in TEXT_COLUMNS:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    ordered_columns = [
        "sample_id",
        "source_repo",
        "source_file",
        "source_dataset_kind",
        "pathway",
        "feedstock_name",
        "feedstock_group",
        "blending_case",
        "blend_manure_ratio",
        "blend_wet_waste_ratio",
        "feedstock_carbon_pct",
        "feedstock_hydrogen_pct",
        "feedstock_nitrogen_pct",
        "feedstock_oxygen_pct",
        "feedstock_moisture_pct",
        "feedstock_volatile_matter_pct",
        "feedstock_fixed_carbon_pct",
        "feedstock_ash_pct",
        "feedstock_hhv_mj_per_kg",
        "process_temperature_c",
        "residence_time_min",
        "heating_rate_c_per_min",
        "product_char_yield_pct",
        "product_char_carbon_pct",
        "product_char_hydrogen_pct",
        "product_char_nitrogen_pct",
        "product_char_oxygen_pct",
        "product_char_hhv_mj_per_kg",
        "energy_recovery_pct",
        "carbon_retention_pct",
        "reference_label",
        "source_extra_value_1",
        "source_extra_value_2",
        "source_extra_value_3",
    ]
    for column in ordered_columns:
        if column not in data.columns:
            data[column] = pd.NA

    return data[ordered_columns]


def load_reference_support_tables() -> dict[str, pd.DataFrame]:
    if not SUPPORT_DOCX.exists():
        return {}

    with zipfile.ZipFile(SUPPORT_DOCX) as archive:
        document_xml = archive.read("word/document.xml")

    root = ET.fromstring(document_xml)
    tables = root.findall(".//w:tbl", DOCX_NAMESPACE)
    if len(tables) < 2:
        return {}

    return {
        "pyrolysis": _prepare_support_table(
            _word_table_to_frame(tables[0]),
            PYROLYSIS_RENAME_MAP,
            pathway="pyrolysis",
        ),
        "htc": _prepare_support_table(
            _word_table_to_frame(tables[1]).rename(columns={"Char_ yield (%)": "Char_yield (%)"}),
            HTC_RENAME_MAP,
            pathway="htc",
        ),
    }


def _word_table_to_frame(table_element) -> pd.DataFrame:
    rows: list[list[str]] = []
    for row in table_element.findall("./w:tr", DOCX_NAMESPACE):
        cells = [_word_cell_text(cell) for cell in row.findall("./w:tc", DOCX_NAMESPACE)]
        rows.append(cells)

    return pd.DataFrame(rows[1:], columns=rows[0])


def _word_cell_text(cell_element) -> str:
    texts = [node.text for node in cell_element.findall(".//w:t", DOCX_NAMESPACE) if node.text]
    return "".join(texts).strip()


def _prepare_support_table(
    frame: pd.DataFrame,
    rename_map: dict[str, str],
    pathway: str,
) -> pd.DataFrame:
    support = frame.rename(columns=rename_map).copy()
    if "reference_label" not in support.columns and "Reference" in support.columns:
        support = support.rename(columns={"Reference": "reference_label"})
    support = support.loc[:, ~support.columns.astype(str).str.startswith("Unnamed")]
    support["reference_label"] = support["reference_label"].replace("", pd.NA).ffill()
    support["pathway"] = pathway

    ordered_columns = [
        "pathway",
        "feedstock_carbon_pct",
        "feedstock_hydrogen_pct",
        "feedstock_nitrogen_pct",
        "feedstock_oxygen_pct",
        "feedstock_moisture_pct",
        "feedstock_volatile_matter_pct",
        "feedstock_fixed_carbon_pct",
        "feedstock_ash_pct",
        "feedstock_hhv_mj_per_kg",
        "process_temperature_c",
        "residence_time_min",
        "heating_rate_c_per_min",
        "product_char_yield_pct",
        "product_char_carbon_pct",
        "product_char_hydrogen_pct",
        "product_char_nitrogen_pct",
        "product_char_oxygen_pct",
        "product_char_hhv_mj_per_kg",
        "energy_recovery_pct",
        "carbon_retention_pct",
        "reference_label",
    ]
    for column in ordered_columns:
        if column not in support.columns:
            support[column] = pd.NA

    return _normalize_alignment_frame(support[ordered_columns].copy())


def _normalize_alignment_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].map(_normalize_alignment_value)
    return normalized


def _normalize_alignment_value(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text or text.lower() in {"nan", "na"}:
        return ""

    try:
        numeric = float(text)
    except ValueError:
        return text

    if numeric.is_integer():
        return str(int(numeric))
    return format(numeric, ".15g")


def fill_reference_labels_with_support(
    standardized: pd.DataFrame,
    support: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if support is None or support.empty:
        return standardized, {"matched_rows": 0, "unmatched_rows": int(len(standardized))}

    enriched = standardized.copy()
    enriched["reference_label"] = enriched["reference_label"].astype("object")
    comparable_columns = [
        column
        for column in support.columns
        if column != "reference_label"
        and column in enriched.columns
        and support[column].ne("").any()
    ]

    normalized_base = _normalize_alignment_frame(enriched[comparable_columns + ["reference_label"]])
    normalized_support = _normalize_alignment_frame(support[comparable_columns + ["reference_label"]])

    base_index = 0
    support_index = 0
    matched_rows = 0

    while base_index < len(normalized_base) and support_index < len(normalized_support):
        base_key = tuple(normalized_base.loc[base_index, comparable_columns])
        support_key = tuple(normalized_support.loc[support_index, comparable_columns])

        if base_key == support_key:
            support_reference = normalized_support.loc[support_index, "reference_label"]
            if support_reference:
                enriched.loc[base_index, "reference_label"] = support_reference
            matched_rows += 1
            base_index += 1
            support_index += 1
            continue

        next_base_matches = base_index + 1 < len(normalized_base) and tuple(
            normalized_base.loc[base_index + 1, comparable_columns]
        ) == support_key
        next_support_matches = support_index + 1 < len(normalized_support) and tuple(
            normalized_support.loc[support_index + 1, comparable_columns]
        ) == base_key

        if next_base_matches:
            base_index += 1
            continue
        if next_support_matches:
            support_index += 1
            continue

        base_index += 1
        support_index += 1

    return enriched, {
        "matched_rows": matched_rows,
        "unmatched_rows": int(len(enriched) - matched_rows),
    }


def write_metadata(files: list[Path]) -> None:
    payload = {
        "source_repo": "Wet-Waste-Biomass-Opt",
        "source_repository_url": "https://github.com/PEESEgroup/Wet-Waste-Biomass-Opt",
        "paper_doi": "https://doi.org/10.1016/j.jclepro.2023.138606",
        "processed_files": [file.name for file in files],
    }
    output_path = OUT_DIR / "wet_waste_biomass_opt_metadata.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    htc = pd.read_csv(RAW_DIR / "HTC_data.csv")
    pyrolysis = pd.read_csv(RAW_DIR / "Pyrolysis_data.csv")
    reference_support = load_reference_support_tables()

    standardized_htc = standardize_frame(htc, HTC_RENAME_MAP, "htc", "HTC_data.csv")
    standardized_pyrolysis = standardize_frame(
        pyrolysis, PYROLYSIS_RENAME_MAP, "pyrolysis", "Pyrolysis_data.csv"
    )
    standardized_htc, htc_support_stats = fill_reference_labels_with_support(
        standardized_htc,
        reference_support.get("htc"),
    )
    standardized_pyrolysis, pyrolysis_support_stats = fill_reference_labels_with_support(
        standardized_pyrolysis,
        reference_support.get("pyrolysis"),
    )
    combined_records = standardized_htc.to_dict("records") + standardized_pyrolysis.to_dict("records")
    combined = pd.DataFrame.from_records(combined_records, columns=standardized_htc.columns)

    htc_out = OUT_DIR / "wet_waste_biomass_opt_htc_standardized.csv"
    pyrolysis_out = OUT_DIR / "wet_waste_biomass_opt_pyrolysis_standardized.csv"
    combined_out = OUT_DIR / "wet_waste_biomass_opt_combined_standardized.csv"

    standardized_htc.to_csv(htc_out, index=False)
    standardized_pyrolysis.to_csv(pyrolysis_out, index=False)
    combined.to_csv(combined_out, index=False)

    write_metadata([htc_out, pyrolysis_out, combined_out])

    print(f"Wrote {htc_out}")
    print(f"Wrote {pyrolysis_out}")
    print(f"Wrote {combined_out}")
    print(f"HTC citation-support match stats: {htc_support_stats}")
    print(f"Pyrolysis citation-support match stats: {pyrolysis_support_stats}")


if __name__ == "__main__":
    main()
